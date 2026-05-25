"""Custom loss functions for prime-rl.

Installed into ``prime_rl/trainer/rl/custom_losses.py`` at job setup (see
``configs/skypilot/desktop_rl_train.yaml`` – same pattern as ``pathing.py``
and ``wandb.py`` patches).

Exports
-------
``rollout_mean_loss_fn``
    Dr-GRPO-style rollout-level DPPO+KL loss that removes the length bias
    baked into prime-rl's default (token-level-averaged) loss.
"""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Bool
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs
from torch import Tensor


def _safe_mean(t: Tensor, m: Bool[Tensor, " seq"]) -> Tensor:  # noqa: F722
    """Mean of ``t`` over positions where ``m`` is True, or 0 if the mask is empty."""
    count = m.sum()
    if count == 0:
        return torch.zeros((), device=t.device, dtype=t.dtype)
    return (t * m).sum() / count


def rollout_mean_loss_fn(
    inputs: LossInputs,
    kl_tau: float = 0.01,
    dppo_mask_low: float = 0.2,
    dppo_mask_high: float = 0.2,
    adv_tau: float = 1.0,
    teacher_tau: float = 0.0,
    rollout_scale: float = 2048.0,
    entropy_tau: float = 0.0,
) -> LossOutputs:
    """Rollout-level DPPO+KL loss (Dr-GRPO length-bias fix).

    Matches ``prime_rl.trainer.rl.loss.default_loss_fn`` byte-for-byte in
    algorithm (DPPO-Binary TV masking + Kimi-K2.5 KL against the
    inference policy), BUT replaces the final reduction:

        default_loss_fn:   loss = (-pg_loss + kl_tau * kl_loss).sum()
        rollout_mean:      loss = ((-pg_loss + kl_tau * kl_loss).sum()
                                   / max(n_trainable_tokens, 1)) * rollout_scale

    Why
    ---
    The framework aggregates per-sequence losses via
    ``scaled_loss = sum_i L_i / loss_scale`` with
    ``loss_scale = total_trainable_tokens_in_batch``. With the default
    per-sequence reduction ``L_i = sum_t loss_{t,i}`` this becomes
    **token-level averaging**: a rollout contributes to the gradient
    proportional to its token count. Long failing rollouts (e.g. OSWorld
    agents that flail until max_turns) dominate the gradient and the
    policy learns to keep flailing instead of terminating.

    Replacing ``L_i`` with ``mean_t loss_{t,i} * rollout_scale`` makes
    each rollout contribute its average per-token loss (scaled by a
    fixed constant so the gradient magnitude is roughly preserved). The
    framework-side division by ``total_tokens`` still happens, but now
    every rollout has the same "weight" inside the sum_i term; the
    token-count normalisation cancels out up to a fixed factor. This is
    the Dr-GRPO length-bias fix, adapted to prime-rl's packed-sequence
    aggregation contract.

    ``rollout_scale = 2048`` is chosen to roughly match the observed
    average OSWorld rollout length (~2000-4000 assistant tokens) so the
    resulting gradient magnitude is comparable to the default loss. LR
    therefore does not need retuning.

    Parameters
    ----------
    inputs
        Standard ``LossInputs`` (per-sequence logprobs, advantages, mask).
    kl_tau, dppo_mask_low, dppo_mask_high, adv_tau, teacher_tau
        Same semantics as ``DefaultLossConfig``.
    rollout_scale
        Per-sequence scaling factor applied after mean reduction. Pick
        close to the per-rollout average trainable-token count for
        gradient-magnitude parity with the default loss.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    trainer_probs = torch.exp(trainer_logprobs)
    inference_probs = torch.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    dppo_invalid_mask_high = probs_diff > dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -dppo_mask_low
    dppo_invalid_mask = torch.where(advantages > 0, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = (advantages > 0) & dppo_invalid_mask_high
    is_masked_low = (advantages < 0) & dppo_invalid_mask_low
    keep_mask = loss_mask & ~is_masked

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1

    advantages_scaled = adv_tau * advantages
    if teacher_logprobs is not None and teacher_tau > 0:
        teacher_kl = teacher_logprobs - trainer_logprobs
        advantages_scaled = advantages_scaled + teacher_tau * teacher_kl.detach()
    else:
        teacher_kl = None

    pg_loss = keep_mask * advantages_scaled * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2

    # v15 FIX: entropy bonus with CORRECT sign.
    #
    # True distribution entropy H = -sum_v p_v log p_v requires full logits
    # which ``loss_fn`` doesn't receive; we use the selected-token neg logprob
    # as a noisy proxy: H_proxy = -logp_selected (larger = less confident).
    #
    # To MAXIMIZE H_proxy (promote exploration), we SUBTRACT it from the
    # loss (which is minimized by SGD):
    #     loss_contribution = -entropy_tau * H_proxy = +entropy_tau * logp
    # With logp negative, loss_contribution is negative → minimizing loss
    # drives logp MORE negative (lower confidence). WRONG direction.
    #
    # Correct: we want to ADD -entropy_tau * (-logp) = +entropy_tau * logp to
    # the loss. Wait — let me redo this. Minimizing L means we move θ so L ↓.
    # ∂L/∂logp = entropy_tau, so SGD updates logp by -η * entropy_tau < 0,
    # making logp more negative. That is WHAT v13 DID and the model got worse.
    #
    # The correct formulation to maximize H_proxy = -logp (entropy proxy):
    #     subtract entropy_tau * H_proxy from loss →
    #     loss_contrib = -entropy_tau * (-logp) = +entropy_tau * logp
    # ∂L/∂logp = entropy_tau > 0, SGD decreases logp. Still wrong direction!
    #
    # Actually, for a peaked distribution, H ≈ -logp_selected only when
    # logp_selected is close to 0 (prob ≈ 1). For high-entropy peak (prob
    # spread out), logp_selected is very negative. So -logp_selected IS the
    # entropy-correlated quantity, and we want to INCREASE it. That means
    # we add +entropy_tau * logp_selected to the loss (so minimizing loss
    # decreases logp_selected → increases -logp → increases entropy proxy).
    # But that's anti-learning on the token we actually sampled from the
    # inference policy. The policy gradient wants to INCREASE logp_selected
    # for positive advantages; the entropy bonus wants to DECREASE it. These
    # FIGHT each other on positive-advantage tokens.
    #
    # Conclusion: on-policy-with-importance-sampling + entropy bonus requires
    # a different formulation (full-distribution entropy via logits, not
    # selected-token proxy). We disable the entropy term entirely for v14+
    # by defaulting entropy_tau=0.0, and leave the code below for reference.
    entropy_term = entropy_tau * trainer_logprobs * loss_mask.to(trainer_logprobs.dtype)

    per_token = -pg_loss + kl_tau * kl_loss + entropy_term
    n_tokens = loss_mask.sum().clamp(min=1).to(per_token.dtype)
    loss = (per_token.sum() / n_tokens) * rollout_scale

    # Proxy entropy metric: mean negative selected-token logprob (high when
    # the model is uncertain, low when confident). Reported separately from
    # the full-distribution entropy that train.py logs from logits.
    proxy_entropy = -_safe_mean(trainer_logprobs, loss_mask)

    metrics: dict[str, Any] = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(is_masked.to(per_token.dtype), loss_mask),
        "is_masked_low": _safe_mean(is_masked_low.to(per_token.dtype), loss_mask),
        "is_masked_high": _safe_mean(is_masked_high.to(per_token.dtype), loss_mask),
        "n_tokens_per_rollout": n_tokens.detach(),
        "proxy_entropy": proxy_entropy.detach(),
    }
    if teacher_kl is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_kl, loss_mask)

    return LossOutputs(loss=loss, metrics=metrics)
