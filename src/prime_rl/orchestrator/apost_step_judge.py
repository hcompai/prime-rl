"""A-posteriori per-step LLM judge credit assignment.

Background
----------
GRPO assigns a single scalar advantage per rollout, broadcast to every
completion token. For multi-turn agentic tasks this mis-attributes credit:
in a failed rollout, *all* turns (including good ones) get penalised; in a
successful one, *all* turns (including mistakes that were later recovered)
get reinforced.

This module is the "external judge" sibling of ``self_judge.py``. Rather
than asking the policy to label its own turns inline (which couples the
labels to the chat template and tokenisation), an external Claude Sonnet
4.6 judge reads the full trace post-hoc and emits a continuous score
``score[t] ∈ [0, 1]`` per assistant turn — the fraction of N generic
criteria the judge marked satisfied for that turn. The env writes the
score list into ``state["_apost_step_scores"]`` and a completion-bonus
boolean into ``state["_apost_judge_completion_bonus"]`` (see the
``hai_desktop_env`` package); the orchestrator picks them up via the
``REQUIRED_STATE_COLUMNS`` plumbing in ``envs.py``.

The formula (mirrors apost-step-rewards-plan.html §4 in hai-gui-env):

    raw_w[t]   = base + (1 - base) * score[t]                  # base ∈ [0, 1]
    raw_w[T]  *= (1 + completion_bonus)                        # iff final tool was
                                                               # `answer` AND task
                                                               # was scored as success
    raw_w[t]   = max(raw_w[t], 1.0)        if A < 0  AND clamp_fail_dampening
    m[t]       = raw_w[t] * total_len / sum_s (raw_w[s] * len[s])    # mass preserve
    per_tok[i] = A * m[turn(i)]

Anti-hack properties carried over from ``self_judge.py``:

- **Token-mass preservation.** Mean per-token advantage equals the scalar
  broadcast baseline. Uniform ``score[t] ≡ const`` collapses to that
  baseline. Only differential scores create per-turn signal.
- **Sign-aware clamp.** Without the clamp, high-score steps in failed
  rollouts would be blamed more than low-score ones — the wrong direction.
  Clamping to ``>= 1.0`` in fail rollouts keeps the scalar baseline as a
  floor on blame.

Compared to ``self_judge.py``:

- No tokenizer-aware label detection (the policy emits nothing special;
  the score lives entirely on the orchestrator side).
- No ``mask_label_tokens`` (no label tokens exist to mask).
- No ``flip_false_achieved`` (continuous scores, not discrete labels).
- Lighter weight: a few dozen lines of pure-Python arithmetic per rollout.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.transport.types import TrainingSample


@dataclass(frozen=True)
class ApostStepJudgeSpec:
    """Resolved orchestrator-side state for the a-posteriori step judge.

    ``base`` is the floor weight in success rollouts: a step with score 0
    still gets ``base`` × the scalar advantage, so judge false-negatives
    don't zero out useful tokens. ``base = 1.0`` is a no-op; ``base = 0.0``
    multiplies the worst-scoring steps by zero before mass-renorm.

    ``completion_bonus`` multiplies the answer-closing turn's raw weight by
    ``(1 + completion_bonus)`` when the task was scored as success. Adds a
    "close the deal" pressure on top of the M/N score.

    ``clamp_fail_dampening`` toggles the sign-aware clamp in failed rollouts.
    See the module docstring for the motivation; matches the same flag on
    ``SelfJudgeSpec``.
    """

    base: float
    completion_bonus: float
    clamp_fail_dampening: bool = True


def resolve_apost_step_judge_spec(
    base: float,
    completion_bonus: float,
    clamp_fail_dampening: bool = True,
) -> ApostStepJudgeSpec:
    """Validate config knobs and produce a frozen spec.

    Args:
        base: Floor weight in success rollouts; must be in ``[0, 1]``.
        completion_bonus: Multiplicative bonus on the answer-closing turn
            when the task succeeded; must be ``>= 0``.
        clamp_fail_dampening: Sign-aware clamp in failed rollouts.

    Returns:
        Frozen ``ApostStepJudgeSpec``.

    Raises:
        ValueError: If ``base`` is outside ``[0, 1]`` or
            ``completion_bonus`` is negative.
    """
    if not 0.0 <= base <= 1.0:
        raise ValueError(f"base must be in [0, 1], got {base}")
    if completion_bonus < 0.0:
        raise ValueError(f"completion_bonus must be >= 0, got {completion_bonus}")
    return ApostStepJudgeSpec(
        base=base,
        completion_bonus=completion_bonus,
        clamp_fail_dampening=clamp_fail_dampening,
    )


def attach_apost_step_advantages_to_rollout(
    samples: list["TrainingSample"],
    step_scores: list[float],
    scalar_advantage: float,
    completion_bonus_applies: bool,
    spec: ApostStepJudgeSpec,
) -> dict[str, int | float] | None:
    """Apply per-step credit assignment across *all* samples of a rollout.

    Mutates ``sample.completion_advantages`` in place on each input sample.
    Mirrors ``self_judge.attach_self_judge_advantages_to_rollout`` for
    extension-broken rollouts: pools spans across all samples so mass
    preservation is computed on the full rollout, not per-sample.

    Args:
        samples: All ``TrainingSample`` objects belonging to one rollout.
            Each carries its own ``completion_mask`` (contiguous trainable
            spans = one assistant turn each).
        step_scores: Per-assistant-turn scores in ``[0, 1]``, in chronological
            order across the whole rollout. Length is matched against the
            total turn-span count derived from ``completion_mask``; on
            mismatch we fall back to no-op rather than corrupting the gradient.
        scalar_advantage: Rollout scalar advantage (post-GRPO baseline).
        completion_bonus_applies: True iff the rollout's final tool call was
            ``answer`` AND the task scored as success. Decided env-side
            (see ``hai_desktop_env.environment._completion_bonus_applies``)
            so the orchestrator stays env-agnostic.
        spec: Resolved knobs.

    Returns:
        Aggregated stats dict for wandb, or ``None`` if the rollout had no
        trainable tokens / no spans / a turn-count mismatch.
    """
    if not samples or scalar_advantage is None:
        return None

    spans_per_sample: list[list[tuple[int, int]]] = [_find_turn_token_spans(s.completion_mask) for s in samples]
    flat_spans: list[tuple[int, int, int]] = [
        (si, s, e) for si, spans in enumerate(spans_per_sample) for s, e in spans
    ]
    if not flat_spans:
        return None
    if len(step_scores) != len(flat_spans):
        # Length mismatch is a hard signal something is wrong upstream
        # (env / orchestrator turn-counting disagree). Fail-open rather
        # than guess a per-step alignment that may be wrong.
        return {
            "n_turns": len(flat_spans),
            "n_samples": len(samples),
            "n_score_mismatch": 1,
        }

    span_lens: list[int] = [
        sum(1 for p in range(s, e) if samples[si].completion_mask[p]) for si, s, e in flat_spans
    ]
    total_len: int = sum(span_lens)
    if total_len == 0:
        return {
            "n_turns": len(flat_spans),
            "n_samples": len(samples),
            "n_no_trainable_tokens": 1,
        }

    raw_weights: list[float] = [spec.base + (1.0 - spec.base) * float(s) for s in step_scores]
    if completion_bonus_applies:
        raw_weights[-1] *= 1.0 + spec.completion_bonus
    if spec.clamp_fail_dampening and scalar_advantage < 0:
        raw_weights = [max(w, 1.0) for w in raw_weights]

    weighted_sum: float = sum(w * L for w, L in zip(raw_weights, span_lens))
    if weighted_sum <= 0:
        return {
            "n_turns": len(flat_spans),
            "n_samples": len(samples),
            "n_zero_weight_sum": 1,
        }
    scale: float = total_len / weighted_sum
    multipliers: list[float] = [w * scale for w in raw_weights]

    for sample in samples:
        sample.completion_advantages = [scalar_advantage] * len(sample.completion_ids)
    for (si, s, e), m in zip(flat_spans, multipliers):
        adv_t: float = scalar_advantage * m
        per_tok = samples[si].completion_advantages
        for pos in range(s, e):
            per_tok[pos] = adv_t

    return {
        "n_turns": len(flat_spans),
        "n_samples": len(samples),
        "n_completion_bonus_fired": 1 if completion_bonus_applies else 0,
        "mean_multiplier": sum(multipliers) / len(multipliers),
        "max_multiplier": max(multipliers),
        "min_multiplier": min(multipliers),
        "mean_score": sum(step_scores) / len(step_scores),
    }


def _find_turn_token_spans(completion_mask: list[bool]) -> list[tuple[int, int]]:
    """Identify (start, end_exclusive) spans of contiguous trainable tokens.

    Each span corresponds to one assistant turn's generated tokens. The
    inter-turn gaps (env-response prompt re-injection) are ``mask=False`` and
    naturally separate the spans.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, m in enumerate(completion_mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(completion_mask)))
    return spans
