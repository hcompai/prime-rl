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
    of ``PROGRESS_LAST_TURN=<label>``. Used to detect and mask those tokens
    from the loss. The label sits between ``</think>`` and ``<tool_call>``
    in the new turn format, so we scan the entire turn span (not just a
    short prefix window).

    ``mask_label_tokens`` toggles masking; off is fine because mass
    preservation already neutralises uniform-label hacking.
    """

    alpha: float
    # Each label maps to a list of *equivalent* token-id sequences that
    # match the literal ``PROGRESS_LAST_TURN=<label>`` in different
    # preceding-context tokenisations (BPE merges depend on the boundary).
    label_prefix_token_ids: dict[str, list[list[int]]]
    mask_label_tokens: bool = True
    # When True, re-interpret ACHIEVED → REGRESS in failed rollouts. The
    # rollout-level eval is ground truth — when the model claimed ACHIEVED
    # and the eval rejected, that turn was a high-confidence mistake, not
    # a near-success. Without this flip the sign-aware weighting *protects*
    # false-ACHIEVED turns from negative gradient (weight ~0), reinforcing
    # over-confidence. See rollout forensics in iter-10 commit message:
    # 27% of ACHIEVED-emitting rollouts actually failed in iter-8.
    flip_false_achieved: bool = True
    # When True, clamp per-turn weights to ≥ 1.0 in failed rollouts. The
    # original sign-aware formula assumed honest labels: a PROGRESS-labelled
    # turn in a fail rollout is "the model believes this action was OK but
    # the rollout failed overall" → soft-landing weight 0.5. But iter-8
    # rollout forensics showed labels are *systematically optimistic*: the
    # dominant failure mode (`write_desktop` typing "Desktop" instead of
    # "/home/hai/Desktop") was reliably labelled PROGRESS by the model
    # because the post-action screenshot looked right. Dampening blame on
    # that turn (0.5 + mass-renorm → ~0.6×) let the broken strategy
    # persist across 43 steps with no learning. Clamping to ≥ 1.0 keeps
    # the *amplification* of REGRESS turns (the one signal we can trust:
    # "model thinks it screwed up AND it did") but removes the dampening,
    # so positive labels in fail rollouts get full scalar blame instead
    # of reduced blame. Combined with `flip_false_achieved`, ACHIEVED in
    # a fail rollout becomes REGRESS-equivalent (amplified blame).
    clamp_fail_dampening: bool = True


def _encode_label_prefix_variants(tokenizer: "PreTrainedTokenizerBase", label: str) -> list[list[int]]:
    """Tokenise ``PROGRESS_LAST_TURN=<label>`` in a few preceding-context
    variants and return only the suffix that corresponds to the literal
    itself.

    BPE tokenisation is context-dependent: the tokens for
    ``PROGRESS_LAST_TURN=ACHIEVED`` differ depending on whether the
    preceding character is ``>`` (from ``</think>``), ``\n``, or a space.
    We pre-compute all plausible variants once and try each at detection
    time. Empirically with Qwen3-VL there are usually 1–2 distinct
    encodings; we deduplicate.
    """
    text = f"PROGRESS_LAST_TURN={label}"
    base_ids = tokenizer.encode(text, add_special_tokens=False)
    variants: list[list[int]] = [base_ids]
    # Encode in context with various leading characters; strip the leading
    # tokens to get just the literal's suffix tokens. We diff against a
    # baseline encoding of just the lead string to isolate the suffix.
    for lead in ("\n", " ", "\n\n", ">\n", "</think>\n"):
        lead_ids = tokenizer.encode(lead, add_special_tokens=False)
        full_ids = tokenizer.encode(lead + text, add_special_tokens=False)
        if len(full_ids) >= len(lead_ids) and full_ids[: len(lead_ids)] == lead_ids:
            suffix = full_ids[len(lead_ids):]
        else:
            # BPE merged across the boundary; keep the tail that ends with
            # the same final token as base_ids (heuristic, length-bounded).
            cut = max(0, len(full_ids) - len(base_ids) - 2)
            suffix = full_ids[cut:]
        if suffix and suffix not in variants:
            variants.append(suffix)
    return variants


def resolve_self_judge_spec(
    tokenizer: "PreTrainedTokenizerBase",
    alpha: float,
    mask_label_tokens: bool = True,
    flip_false_achieved: bool = True,
    clamp_fail_dampening: bool = True,
) -> SelfJudgeSpec:
    """Pre-compute label-prefix token sequences once at orchestrator start."""
    prefix_ids = {label: _encode_label_prefix_variants(tokenizer, label) for label in LABEL_VALUE}
    return SelfJudgeSpec(
        alpha=alpha,
        label_prefix_token_ids=prefix_ids,
        mask_label_tokens=mask_label_tokens,
        flip_false_achieved=flip_false_achieved,
        clamp_fail_dampening=clamp_fail_dampening,
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
) -> tuple[str, int, int] | None:
    """Find the ``PROGRESS_LAST_TURN=<label>`` token sequence anywhere in span.

    The label sits between ``</think>`` and the tool call in the new turn
    format, so we scan the full span (typically a few hundred tokens; the
    inner ``for pos`` loop is bounded by span length and a tiny set of
    label prefixes — cheap in absolute terms).

    Returns ``(label, prefix_start, prefix_end_exclusive)`` or ``None``.
    """
    # First label-variant found wins (turns have one label by construction).
    for pos in range(span_start, span_end):
        for label, variants in spec.label_prefix_token_ids.items():
            for prefix in variants:
                n = len(prefix)
                if pos + n > span_end:
                    continue
                if completion_ids[pos : pos + n] == prefix:
                    return label, pos, pos + n
    return None


def attach_self_judge_advantages_to_rollout(
    samples: list["TrainingSample"],
    progress_labels: list[str],
    scalar_advantage: float,
    spec: SelfJudgeSpec,
) -> dict[str, int | float] | None:
    """Apply per-turn credit assignment across *all* samples of a rollout.

    When ``interleave_rollout`` splits a rollout into multiple TrainingSamples
    (extension property breaking, e.g. due to context compaction), naive
    per-sample mass preservation degenerates: a single-turn sample has only
    one span, so its multiplier is trivially 1.0 and no per-turn signal is
    delivered. This function instead pools spans across all samples of the
    rollout so the linear weights compete globally, and only renormalises
    once on the union — preserving the invariant ``Σ_s m[s] · len[s] = Σ
    len[s]`` over the full rollout.

    Mutates each sample's ``completion_advantages`` and (when enabled)
    ``completion_mask`` in place. Returns aggregated stats for wandb.
    """
    if not samples or scalar_advantage is None:
        return None

    # Per-sample turn spans + detected labels + prefix spans.
    sample_spans: list[list[tuple[int, int]]] = []
    sample_labels: list[list[str]] = []
    sample_prefixes: list[list[tuple[int, int] | None]] = []
    for sample in samples:
        cm = sample.completion_mask
        cids = sample.completion_ids
        if not cids:
            sample_spans.append([])
            sample_labels.append([])
            sample_prefixes.append([])
            continue
        spans = _find_turn_token_spans(cm)
        labels: list[str] = []
        prefixes: list[tuple[int, int] | None] = []
        for s, e in spans:
            hit = _detect_label_prefix(cids, s, e, spec)
            if hit is None:
                labels.append(DEFAULT_LABEL)
                prefixes.append(None)
            else:
                lbl, ps, pe = hit
                labels.append(lbl if lbl in LABEL_VALUE else DEFAULT_LABEL)
                prefixes.append((ps, pe))
        sample_spans.append(spans)
        sample_labels.append(labels)
        sample_prefixes.append(prefixes)

    # Flatten into a global turn list. Track which sample / span index each
    # global turn belongs to so we can map multipliers back.
    flat: list[tuple[int, int, str, tuple[int, int] | None]] = []  # (sample_idx, span_idx, label, prefix)
    for si, (spans, labels, prefixes) in enumerate(zip(sample_spans, sample_labels, sample_prefixes)):
        for span_idx, (lbl, pref) in enumerate(zip(labels, prefixes)):
            flat.append((si, span_idx, lbl, pref))

    if not flat:
        return None

    # Best-effort positional fallback: if detection failed for some turns
    # but the env-side parser caught labels in order, fill in the gaps
    # using positional indexing into the rollout-wide labels list. This is
    # naive (sample turn-order vs trajectory turn-order can disagree when
    # extension breaks then resumes) and only helps when the rollout is
    # mostly linear; on disagreement the rare token-level detection
    # already produced the correct label per span.
    if progress_labels:
        # Number turns in flat order across all samples.
        for i, (si, span_idx, lbl, pref) in enumerate(flat):
            if pref is None and lbl == DEFAULT_LABEL and i < len(progress_labels):
                cand = progress_labels[i]
                if cand in LABEL_VALUE:
                    flat[i] = (si, span_idx, cand, pref)

    # Apply label masking before computing span_lens.
    n_masked = 0
    if spec.mask_label_tokens:
        for sample, prefixes in zip(samples, sample_prefixes):
            if not prefixes:
                continue
            new_mask = list(sample.completion_mask)
            for pref in prefixes:
                if pref is None:
                    continue
                ps, pe = pref
                for pos in range(ps, pe):
                    if new_mask[pos]:
                        new_mask[pos] = False
                        n_masked += 1
            sample.completion_mask = new_mask

    # Trainable-token length per global turn (after masking).
    span_lens: list[int] = []
    for si, span_idx, _lbl, _pref in flat:
        s, e = sample_spans[si][span_idx]
        cm = samples[si].completion_mask
        span_lens.append(sum(1 for p in range(s, e) if cm[p]))
    total_len = sum(span_lens)
    if total_len == 0:
        return {
            "n_turns": len(flat),
            "n_samples": len(samples),
            "n_masked_tokens": n_masked,
        }

    adv_sign = 1.0 if scalar_advantage >= 0 else -1.0
    # Demote ACHIEVED → REGRESS in failed rollouts: a model that claimed
    # task completion which the eval rejected made a high-confidence
    # mistake on that turn, not a near-success. Without this, the
    # sign-aware formula gives ACHIEVED-in-fail near-zero weight (least
    # blame), reinforcing over-confidence — see rollout forensics in
    # iter-10. Track the count for wandb diagnostics.
    n_flipped_achieved = 0
    effective_labels: list[str] = []
    for _si, _idx, lbl, _pref in flat:
        if spec.flip_false_achieved and adv_sign < 0 and lbl == "ACHIEVED":
            effective_labels.append("REGRESS")
            n_flipped_achieved += 1
        else:
            effective_labels.append(lbl)
    raw_weights = [1.0 + spec.alpha * LABEL_VALUE[lbl] * adv_sign for lbl in effective_labels]
    raw_weights = [max(0.05, w) for w in raw_weights]
    # Clamp dampening in fail rollouts: positive labels (PROGRESS) in
    # failed rollouts are *systematically optimistic* (the model thinks
    # the action helped because the screenshot looks like progress, but
    # the rollout-level eval said no). The original soft-landing weight
    # 0.5× *protected* those turns from blame — the exact opposite of
    # what we want when the labels are unreliable. After clamping, only
    # REGRESS turns get amplified (1.5×) and everything else gets the
    # scalar baseline (1.0×) before mass-preservation.
    if spec.clamp_fail_dampening and adv_sign < 0:
        raw_weights = [max(1.0, w) for w in raw_weights]

    weighted_sum = sum(w * L for w, L in zip(raw_weights, span_lens))
    if weighted_sum <= 0:
        return {
            "n_turns": len(flat),
            "n_samples": len(samples),
            "n_masked_tokens": n_masked,
        }
    scale = total_len / weighted_sum
    multipliers = [w * scale for w in raw_weights]

    # Initialise per_token_advantages on every sample, then overlay the
    # per-turn multipliers.
    for sample in samples:
        sample.completion_advantages = [scalar_advantage] * len(sample.completion_ids)
    for (si, span_idx, _lbl, _pref), m in zip(flat, multipliers):
        s, e = sample_spans[si][span_idx]
        adv_t = scalar_advantage * m
        per_tok = samples[si].completion_advantages
        for pos in range(s, e):
            per_tok[pos] = adv_t

    label_counts = {lbl: 0 for lbl in LABEL_VALUE}
    for _si, _idx, lbl, _pref in flat:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    n_detected = sum(1 for _si, _idx, _lbl, pref in flat if pref is not None)
    return {
        "n_turns": len(flat),
        "n_samples": len(samples),
        "n_masked_tokens": n_masked,
        "n_token_detected_labels": n_detected,
        "n_flipped_achieved": n_flipped_achieved,
        "label_count_REGRESS": label_counts["REGRESS"],
        "label_count_NEUTRAL": label_counts["NEUTRAL"],
        "label_count_PROGRESS": label_counts["PROGRESS"],
        "label_count_ACHIEVED": label_counts["ACHIEVED"],
        "mean_multiplier": sum(multipliers) / len(multipliers),
        "max_multiplier": max(multipliers),
        "min_multiplier": min(multipliers),
    }


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

    # Per-turn label extracted DIRECTLY from completion_ids. This is the
    # ground-truth label the model emitted at that exact token position, so
    # it correctly aligns with the turn even when ``interleave_rollout``
    # splits a long rollout into multiple TrainingSamples (extension
    # property breaking, mid-rollout context compaction). The
    # ``progress_labels`` argument is kept only as a positional fallback for
    # turns where the prefix tokenisation didn't match (rare, e.g. when the
    # model emits an alt phrasing); the fallback indexes into the full
    # rollout list at the sample-local turn index and is best-effort.
    detected_labels: list[str] = []
    detected_prefix_spans: list[tuple[int, int] | None] = []
    for span_start, span_end in turn_spans:
        hit = _detect_label_prefix(completion_ids, span_start, span_end, spec)
        if hit is None:
            detected_labels.append(DEFAULT_LABEL)
            detected_prefix_spans.append(None)
        else:
            lbl, ps, pe = hit
            detected_labels.append(lbl if lbl in LABEL_VALUE else DEFAULT_LABEL)
            detected_prefix_spans.append((ps, pe))

    # Best-effort fallback for turns where token-level detection failed:
    # if the env-side parser caught the label for this turn index, prefer
    # that over the NEUTRAL default. This only helps when the sample
    # contains the first N turns of the rollout in order; for
    # later-in-rollout samples it'll typically be wrong but those turns
    # already have a token-level detection (the prefix is identical
    # regardless of turn index), so this branch is rare.
    labels: list[str] = []
    for i, lbl in enumerate(detected_labels):
        if lbl != DEFAULT_LABEL or detected_prefix_spans[i] is not None:
            labels.append(lbl)
        elif i < len(progress_labels) and progress_labels[i] in LABEL_VALUE:
            labels.append(progress_labels[i])
        else:
            labels.append(DEFAULT_LABEL)

    # Mask only the literal ``PROGRESS_LAST_TURN=<label>`` token span so no
    # gradient flows through the label characters (anti-hacking belt to
    # mass-preservation's suspenders). Turn spans are intentionally NOT
    # recomputed here: the masking creates tiny mask=False holes inside each
    # turn but we still want to count those positions toward the turn's
    # length so the surrounding reasoning + tool-call tokens carry the
    # per-turn multiplier consistently.
    n_masked = 0
    if spec.mask_label_tokens:
        new_mask = list(completion_mask)
        for pref in detected_prefix_spans:
            if pref is None:
                continue
            prefix_start, prefix_end = pref
            for pos in range(prefix_start, prefix_end):
                if new_mask[pos]:
                    new_mask[pos] = False
                    n_masked += 1
        sample.completion_mask = new_mask
        completion_mask = new_mask

    if not turn_spans:
        return {"n_turns": 0, "n_masked_tokens": n_masked}

    # Sign of the rollout advantage. For exact-zero (group baseline tie),
    # treat as +1 to avoid wiping the per-turn signal entirely; tied groups
    # are rare and the gradient magnitude is zero anyway.
    adv_sign = 1.0 if scalar_advantage >= 0 else -1.0
    # Use the trainable-token count per turn for the mass-preservation
    # denominator. After label masking some token positions in each turn are
    # mask=False; they shouldn't count toward the turn's "weight" because
    # the loss ignores them anyway.
    span_lens = [sum(1 for p in range(s, e) if completion_mask[p]) for s, e in turn_spans]
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
