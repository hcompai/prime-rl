"""LLM self-judging per-turn credit assignment.

Background
----------
GRPO assigns a single scalar advantage per rollout, broadcast to every
completion token. For multi-turn agentic tasks this mis-attributes credit:
in a failed rollout, *all* turns (including good ones) get penalized; in a
successful one, *all* turns (including mistakes that were later recovered)
get reinforced. Rollout forensics on the cal VLC overfit run showed 76%
of failed rollouts clicked the correct `Input/Codecs` tab and received
zero reward for that.

Mechanism
---------
We ask the policy itself to label each turn with a discrete progress tag
on the first line of its assistant message::

    PROGRESS_LAST_TURN={REGRESS|NEUTRAL|PROGRESS|ACHIEVED}

The env parses these labels per turn and propagates them via
``state["_progress_labels"]``. This module turns the labels into per-token
advantage multipliers with two anti-hacking properties:

1. **Sign-aware weighting.** ``w[t] = 1 + α · label_value[t] · sign(adv)``
   means: in succ rollouts, PROGRESS turns get *more* positive advantage
   and REGRESS turns *less*; in fail rollouts, REGRESS turns get *more*
   negative advantage (clear blame) and PROGRESS turns *less* (soft
   landing for the model believing it was on the right track).

2. **Token-mass preservation.** Per-token multipliers are rescaled so that
   ``sum_t (m[t] · len[t]) == total_completion_len``. This means a
   rollout where the model labels *every* turn identically gets exactly
   the same gradient signal as the scalar-advantage baseline — uniform
   labelling cannot hack the reward. Only *differential* labelling
   across turns within a rollout creates per-step signal.

Optionally we also mask the ``PROGRESS_LAST_TURN=<label>`` prefix from
the loss so no gradient flows through the label tokens themselves; the
model relies on its prior (pre-RL) calibration for labelling, and the
gradient shapes only the reasoning / tool-call tokens.

Tested in v3/v4 of ``scripts/test_self_judge_v3.py`` against the cal
run's vLLM: discrete labels held format 95-100% of the time at T=1.0,
and calibration on 5 distinct VLC visual states was correct on 4/5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from prime_rl.transport.types import TrainingSample

# Label → relative progress value. ACHIEVED is intentionally 2× PROGRESS to
# reflect both its rarity (one turn per successful rollout) and its semantic
# weight (task done, not just moved closer). REGRESS is symmetric to PROGRESS;
# we don't push it to -2 because spotting truly regressive actions is harder
# for the judge than spotting progress (see v3 test: REGRESS recall ≈ 30%).
LABEL_VALUE: dict[str, float] = {
    "REGRESS": -1.0,
    "NEUTRAL": 0.0,
    "PROGRESS": 1.0,
    "ACHIEVED": 2.0,
}

# Fallback when the model fails to emit / emits an out-of-vocabulary label.
# NEUTRAL = zero contribution to the linear weight formula — equivalent to
# "this turn gets the rollout-mean advantage", which is the GRPO default.
DEFAULT_LABEL = "NEUTRAL"


@dataclass(frozen=True)
class SelfJudgeSpec:
    """Resolved tokenizer-side state for the self-judge masker.

    ``alpha`` controls the per-turn weight magnitude; with α=0.3 a PROGRESS
    turn gets w=1.3 (×1.3 advantage) and a REGRESS one w=0.7 (×0.7) in a
    succ rollout, with the signs flipped for fail rollouts.

    ``label_prefix_token_ids`` maps each canonical label to the token ids
    that prefix the assistant turn (``PROGRESS_LAST_TURN=<label>\\n``).
    Used to detect and mask those tokens from the loss; the leading run
    of newlines/whitespace tokens the model often emits before the label
    is handled by tolerating a small ``mask_search_offset`` window.

    ``mask_label_tokens`` toggles masking; off is fine because mass
    preservation already neutralises uniform-label hacking.
    """

    alpha: float
    label_prefix_token_ids: dict[str, list[int]]
    mask_label_tokens: bool = True
    mask_search_offset: int = 12


def _encode_label_prefix(tokenizer: "PreTrainedTokenizerBase", label: str) -> list[int]:
    """Tokenise the literal ``PROGRESS_LAST_TURN=<label>\\n`` prefix.

    Special tokens are disabled so we get the same token sequence the model
    would emit mid-stream. Newlines at the start are stripped because the
    Qwen3 chat template ends the ``<|im_start|>assistant\\n`` opener with a
    newline, so the next token is the start of the user-content stream.
    """
    text = f"PROGRESS_LAST_TURN={label}\n"
    return tokenizer.encode(text, add_special_tokens=False)


def resolve_self_judge_spec(
    tokenizer: "PreTrainedTokenizerBase",
    alpha: float,
    mask_label_tokens: bool = True,
) -> SelfJudgeSpec:
    """Pre-compute label-prefix token sequences once at orchestrator start."""
    prefix_ids = {label: _encode_label_prefix(tokenizer, label) for label in LABEL_VALUE}
    return SelfJudgeSpec(
        alpha=alpha,
        label_prefix_token_ids=prefix_ids,
        mask_label_tokens=mask_label_tokens,
    )


def _find_turn_token_spans(
    completion_mask: list[bool],
) -> list[tuple[int, int]]:
    """Identify (start, end_exclusive) spans of contiguous trainable tokens.

    Each span corresponds to one assistant turn's generated tokens. The
    inter-turn gaps (env-response prompt re-injection) are mask=False and
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


def _detect_label_prefix(
    completion_ids: list[int],
    span_start: int,
    span_end: int,
    spec: SelfJudgeSpec,
) -> tuple[str, int] | None:
    """Within a turn span, find the ``PROGRESS_LAST_TURN=<label>\\n`` prefix.

    Returns ``(label, prefix_end_in_sample)`` or ``None`` if no prefix is
    detected at the start of the turn. We tolerate up to
    ``mask_search_offset`` leading tokens (e.g. ``<think>\\n`` openers,
    leading newlines) before requiring the prefix to begin.
    """
    max_offset = min(spec.mask_search_offset, span_end - span_start)
    for offset in range(max_offset):
        pos = span_start + offset
        for label, prefix in spec.label_prefix_token_ids.items():
            if pos + len(prefix) > span_end:
                continue
            if completion_ids[pos : pos + len(prefix)] == prefix:
                return label, pos + len(prefix)
    return None


def attach_self_judge_advantages(
    sample: "TrainingSample",
    progress_labels: list[str],
    scalar_advantage: float,
    spec: SelfJudgeSpec,
) -> dict[str, int | float] | None:
    """Replace per-token advantages with self-judged per-turn weights.

    Mutates ``sample.completion_advantages`` and (when enabled)
    ``sample.completion_mask``. Returns a small metrics dict for wandb or
    None if no turns were processed.

    Algorithm
    ---------
    1. Walk ``completion_mask`` to find one (start, end) span per assistant
       turn (mask=True bursts separated by mask=False env-response prompts).
    2. For each span, detect the leading ``PROGRESS_LAST_TURN=<label>\\n``
       prefix (or fall back to ``progress_labels[i]`` if positional). Mask
       those tokens from the loss when ``spec.mask_label_tokens``.
    3. Compute per-turn linear weights
           w[t] = 1 + α · label_value[t] · sign(advantage)
       and rescale to ``m[t] = w[t] · total_len / Σ_s w[s] · len[s]``,
       enforcing the token-mass-preservation invariant.
    4. Fill ``sample.completion_advantages`` with ``adv · m[turn(token)]``
       per completion-token position; non-turn tokens (inter-turn prompts)
       get the bare scalar advantage (they are mask=False anyway).
    """
    completion_ids = sample.completion_ids
    completion_mask = sample.completion_mask
    if not completion_ids or scalar_advantage is None:
        return None

    turn_spans = _find_turn_token_spans(completion_mask)
    if not turn_spans:
        return None

    # Align labels with turn spans. Some rollouts have more trajectory steps
    # than this sample contains (extension property broke mid-rollout); only
    # use the labels covering this sample's spans.
    labels = [(progress_labels[i] if i < len(progress_labels) else DEFAULT_LABEL) for i in range(len(turn_spans))]
    labels = [lbl if lbl in LABEL_VALUE else DEFAULT_LABEL for lbl in labels]

    # Mask the label-prefix tokens (so no gradient flows through the literal
    # `PROGRESS_LAST_TURN=PROGRESS` characters) and refresh the turn spans
    # to exclude those masked tokens from the weight-rescaling denominator.
    n_masked = 0
    if spec.mask_label_tokens:
        new_mask = list(completion_mask)
        for span_start, span_end in turn_spans:
            hit = _detect_label_prefix(completion_ids, span_start, span_end, spec)
            if hit is None:
                continue
            _, prefix_end = hit
            for pos in range(span_start, prefix_end):
                if new_mask[pos]:
                    new_mask[pos] = False
                    n_masked += 1
        sample.completion_mask = new_mask
        completion_mask = new_mask
        turn_spans = _find_turn_token_spans(completion_mask)
        # Realign labels with the (possibly shorter) span list.
        labels = [(progress_labels[i] if i < len(progress_labels) else DEFAULT_LABEL) for i in range(len(turn_spans))]
        labels = [lbl if lbl in LABEL_VALUE else DEFAULT_LABEL for lbl in labels]

    if not turn_spans:
        return {"n_turns": 0, "n_masked_tokens": n_masked}

    # Sign of the rollout advantage. For exact-zero (group baseline tie),
    # treat as +1 to avoid wiping the per-turn signal entirely; tied groups
    # are rare and the gradient magnitude is zero anyway.
    adv_sign = 1.0 if scalar_advantage >= 0 else -1.0
    span_lens = [end - start for start, end in turn_spans]
    total_len = sum(span_lens)
    if total_len == 0:
        return {"n_turns": 0, "n_masked_tokens": n_masked}

    raw_weights = [1.0 + spec.alpha * LABEL_VALUE[lbl] * adv_sign for lbl in labels]
    # Clamp to a non-negative floor so a strong adverse label can't flip the
    # advantage sign on its own turn (which would conflict with the rollout
    # outcome supervision). With α≤0.5 and label_value∈[-1, 2] this is a
    # no-op; we keep the clamp for forward compatibility on tuning sweeps.
    raw_weights = [max(0.05, w) for w in raw_weights]

    weighted_sum = sum(w * l for w, l in zip(raw_weights, span_lens))
    if weighted_sum <= 0:
        return {"n_turns": len(turn_spans), "n_masked_tokens": n_masked}
    scale = total_len / weighted_sum
    multipliers = [w * scale for w in raw_weights]

    per_token_adv = [scalar_advantage] * len(completion_ids)
    for (start, end), m in zip(turn_spans, multipliers):
        adv_t = scalar_advantage * m
        for pos in range(start, end):
            per_token_adv[pos] = adv_t
    sample.completion_advantages = per_token_adv

    # Returned for orchestrator-side aggregation into wandb.
    label_counts = {lbl: 0 for lbl in LABEL_VALUE}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    return {
        "n_turns": len(turn_spans),
        "n_masked_tokens": n_masked,
        "label_count_REGRESS": label_counts["REGRESS"],
        "label_count_NEUTRAL": label_counts["NEUTRAL"],
        "label_count_PROGRESS": label_counts["PROGRESS"],
        "label_count_ACHIEVED": label_counts["ACHIEVED"],
        "mean_multiplier": sum(multipliers) / len(multipliers),
        "max_multiplier": max(multipliers),
        "min_multiplier": min(multipliers),
    }
