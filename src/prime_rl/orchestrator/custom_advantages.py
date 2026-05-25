"""Custom advantage functions for prime-rl.

Installed into ``prime_rl/orchestrator/custom_advantages.py`` at job setup
(see ``configs/skypilot/desktop_rl_train.yaml``).

Exports
-------
``positive_only_advantage_fn``
    Clamps negative advantages to zero. Removes the "push away from everything
    the model did" signal from failed rollouts while keeping the full positive
    reinforcement on successful rollouts.

Why
---
In a group with mostly-successful rollouts (e.g. 6 successes + 2 failures),
raw GRPO advantages are asymmetric::

    baseline = 6/8 = 0.75
    success  adv = +0.25
    failure  adv = -0.75

Multiplied by token counts, failures dominate the magnitude (their
|advantage| is 3x AND they typically have far more tokens — they flail to
max_turns). The PG gradient is then dominated by "push-down" signal on the
common GUI-action tokens shared between success and failure rollouts.

Positive-only advantage cuts the failure-driven push-down contribution
entirely, so we get pure positive reinforcement on winners. This is the
RFT / ReSTEM / REINFORCE-on-positives recipe: gives up contrast signal but
trades bias for stability.

Not currently wired into ``configs/rl_desktop.toml`` (we use prime-rl's
native ``length_shaping`` instead), but kept here because some code paths
still reference the module.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from prime_rl.orchestrator.advantage import AdvantageInputs, AdvantageOutputs


def positive_only_advantage_fn(
    inputs: AdvantageInputs,
    clamp_min: float = 0.0,
    length_shaping: bool = False,
) -> AdvantageOutputs:
    """GRPO advantage clamped to be non-negative.

    Computes standard GRPO advantage (reward - group_baseline), then clamps
    any negative value to ``clamp_min`` (default 0). When ``length_shaping``
    is true, applies the correctness-gated brevity bonus from prime-rl's
    ``_efficiency_length_shaping`` BEFORE clamping (so short-correct rollouts
    still get amplified advantages).

    Args:
        inputs: Contains ``rewards`` of shape ``(num_problems, rollouts_per_example)``
            and ``completion_lengths`` of the same shape.
        clamp_min: Lower bound for advantages. Set to a small negative number
            (e.g. -0.1) to allow a mild push-down signal while preventing
            catastrophic failure dominance.
        length_shaping: Whether to apply prime-rl's correctness-gated brevity
            bonus before clamping.

    Returns:
        AdvantageOutputs with clamped advantage tensor of the same shape as inputs.
    """
    rewards: Float[Tensor, "P R"] = inputs.rewards  # noqa: F722

    if length_shaping:
        from prime_rl.orchestrator.advantage import _efficiency_length_shaping

        completion_lengths = inputs.completion_lengths.to(dtype=rewards.dtype)
        advantages = _efficiency_length_shaping(rewards, completion_lengths)
    else:
        baseline = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - baseline

    advantages = torch.clamp(advantages, min=clamp_min)
    return AdvantageOutputs(advantages=advantages)
