"""Unit tests for prime_rl.orchestrator.apost_step_judge."""

import pytest

from prime_rl.orchestrator.apost_step_judge import (
    ApostStepJudgeSpec,
    _find_turn_token_spans,
    attach_apost_step_advantages_to_rollout,
    resolve_apost_step_judge_spec,
)
from prime_rl.transport.types import TrainingSample


def _make_sample(turn_lens: list[int], gap: int = 1) -> TrainingSample:
    """Build a TrainingSample with `len(turn_lens)` turns of given lengths.

    Each turn's tokens are loss-mask True; turns are separated by `gap`
    mask=False tokens (the env-response slot between turns).
    """
    completion_ids: list[int] = []
    completion_mask: list[bool] = []
    tok_id = 0
    for i, length in enumerate(turn_lens):
        for _ in range(length):
            completion_ids.append(tok_id)
            completion_mask.append(True)
            tok_id += 1
        if i < len(turn_lens) - 1:
            for _ in range(gap):
                completion_ids.append(tok_id)
                completion_mask.append(False)
                tok_id += 1
    return TrainingSample(
        prompt_ids=[0],
        prompt_mask=[False],
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=[0.0] * len(completion_ids),
        completion_temperatures=[1.0] * len(completion_ids),
    )


def test_resolve_validates_base_range():
    with pytest.raises(ValueError, match="base must be in"):
        resolve_apost_step_judge_spec(base=-0.1, completion_bonus=0.5)
    with pytest.raises(ValueError, match="base must be in"):
        resolve_apost_step_judge_spec(base=1.1, completion_bonus=0.5)


def test_resolve_validates_completion_bonus_sign():
    with pytest.raises(ValueError, match="completion_bonus must be"):
        resolve_apost_step_judge_spec(base=0.5, completion_bonus=-0.1)


def test_find_turn_token_spans_separated_by_mask_gaps():
    mask = [True, True, False, True, True, True, False, True]
    assert _find_turn_token_spans(mask) == [(0, 2), (3, 6), (7, 8)]


def test_find_turn_token_spans_ends_on_true_tail():
    mask = [False, True, True]
    assert _find_turn_token_spans(mask) == [(1, 3)]


def test_uniform_scores_collapse_to_scalar_baseline():
    """Mass-preservation invariant: uniform scores → uniform multipliers ≈ 1.0."""
    sample = _make_sample(turn_lens=[5, 5, 5])
    spec = resolve_apost_step_judge_spec(base=0.5, completion_bonus=0.0)

    stats = attach_apost_step_advantages_to_rollout(
        [sample],
        step_scores=[0.6, 0.6, 0.6],
        scalar_advantage=2.0,
        completion_bonus_applies=False,
        spec=spec,
    )

    assert stats is not None
    # All trainable tokens get the same per-token advantage.
    trainable_advs = [
        sample.completion_advantages[i] for i, m in enumerate(sample.completion_mask) if m
    ]
    assert all(a == pytest.approx(2.0, rel=1e-6) for a in trainable_advs)
    assert stats["min_multiplier"] == pytest.approx(1.0, rel=1e-6)
    assert stats["max_multiplier"] == pytest.approx(1.0, rel=1e-6)


def test_differential_scores_produce_signed_per_turn_multipliers():
    sample = _make_sample(turn_lens=[4, 4, 4])
    spec = resolve_apost_step_judge_spec(base=0.5, completion_bonus=0.0)

    attach_apost_step_advantages_to_rollout(
        [sample],
        step_scores=[0.0, 0.5, 1.0],
        scalar_advantage=1.0,
        completion_bonus_applies=False,
        spec=spec,
    )

    # Raw weights: [0.5, 0.75, 1.0] → sum*len = (0.5+0.75+1.0)*4 = 9.0; total_len = 12;
    # scale = 12/9 = 4/3. Multipliers: [2/3, 1.0, 4/3].
    starts = [0, 5, 10]  # gap=1 between turns
    expected = [2 / 3, 1.0, 4 / 3]
    for start, exp in zip(starts, expected):
        adv = sample.completion_advantages[start]
        assert adv == pytest.approx(exp, rel=1e-6)


def test_completion_bonus_only_affects_last_turn():
    sample = _make_sample(turn_lens=[3, 3])
    spec = resolve_apost_step_judge_spec(base=0.5, completion_bonus=1.0)  # 2× on last

    attach_apost_step_advantages_to_rollout(
        [sample],
        step_scores=[1.0, 1.0],
        scalar_advantage=1.0,
        completion_bonus_applies=True,
        spec=spec,
    )

    # raw_w = [1.0, 1.0 * (1+1)] = [1.0, 2.0]; weighted_sum = 1*3 + 2*3 = 9; total = 6;
    # multipliers = [1*(6/9), 2*(6/9)] = [2/3, 4/3].
    first_turn_start = 0
    last_turn_start = 4  # 3 + 1 gap
    assert sample.completion_advantages[first_turn_start] == pytest.approx(2 / 3, rel=1e-6)
    assert sample.completion_advantages[last_turn_start] == pytest.approx(4 / 3, rel=1e-6)


def test_clamp_in_fail_rollouts_prevents_blame_inversion():
    """In a failed rollout, high-score steps must NOT receive more blame than low-score ones."""
    sample = _make_sample(turn_lens=[3, 3])
    spec = resolve_apost_step_judge_spec(base=0.5, completion_bonus=0.0, clamp_fail_dampening=True)

    attach_apost_step_advantages_to_rollout(
        [sample],
        step_scores=[0.0, 1.0],
        scalar_advantage=-1.0,
        completion_bonus_applies=False,
        spec=spec,
    )

    # With clamp: raw_w = [max(0.5, 1.0), max(1.0, 1.0)] = [1.0, 1.0] → uniform.
    # Both turns get the scalar baseline, no blame inversion.
    first = sample.completion_advantages[0]
    second = sample.completion_advantages[4]
    assert first == pytest.approx(second, rel=1e-6)
    assert first == pytest.approx(-1.0, rel=1e-6)


def test_clamp_disabled_inverts_blame_in_fail_rollout():
    """Sanity check: with clamp off, the §4 footgun manifests as expected."""
    sample = _make_sample(turn_lens=[3, 3])
    spec = resolve_apost_step_judge_spec(base=0.5, completion_bonus=0.0, clamp_fail_dampening=False)

    attach_apost_step_advantages_to_rollout(
        [sample],
        step_scores=[0.0, 1.0],
        scalar_advantage=-1.0,
        completion_bonus_applies=False,
        spec=spec,
    )

    # raw_w = [0.5, 1.0]; the high-score step has weight 1.0 vs 0.5.
    # With A = -1, the high-score step gets MORE blame — confirming the
    # clamp is doing real work in the test above.
    first = sample.completion_advantages[0]
    second = sample.completion_advantages[4]
    assert abs(second) > abs(first)


def test_score_count_mismatch_short_circuits():
    """Length mismatch between scores and detected turn spans returns a marker stat."""
    sample = _make_sample(turn_lens=[2, 2, 2])
    spec = resolve_apost_step_judge_spec(base=0.5, completion_bonus=0.0)

    stats = attach_apost_step_advantages_to_rollout(
        [sample],
        step_scores=[0.5],  # only 1, but trace has 3 turns
        scalar_advantage=1.0,
        completion_bonus_applies=False,
        spec=spec,
    )
    assert stats is not None
    assert stats.get("n_score_mismatch") == 1
    # No per-token advantages should be set on the sample.
    assert sample.completion_advantages is None


def test_pools_spans_across_multi_sample_rollout():
    """When extension breaks split a rollout into 2 samples, mass-preservation pools globally."""
    sample_a = _make_sample(turn_lens=[2, 2])
    sample_b = _make_sample(turn_lens=[2])
    spec = resolve_apost_step_judge_spec(base=0.5, completion_bonus=0.0)

    stats = attach_apost_step_advantages_to_rollout(
        [sample_a, sample_b],
        step_scores=[0.0, 0.5, 1.0],
        scalar_advantage=1.0,
        completion_bonus_applies=False,
        spec=spec,
    )

    assert stats is not None
    assert stats["n_turns"] == 3
    assert stats["n_samples"] == 2
    # mass preservation: mean of per-turn (multiplier * len) / total_len ≈ 1
    # (verified implicitly by checking the trainable advantages average to A).
    total_trainable_adv = 0.0
    total_trainable = 0
    for s in (sample_a, sample_b):
        for i, m in enumerate(s.completion_mask):
            if m:
                total_trainable_adv += s.completion_advantages[i]
                total_trainable += 1
    assert total_trainable_adv / total_trainable == pytest.approx(1.0, rel=1e-6)


def test_spec_is_immutable():
    spec = ApostStepJudgeSpec(base=0.5, completion_bonus=0.5)
    with pytest.raises(Exception):
        spec.base = 0.9  # frozen dataclass
