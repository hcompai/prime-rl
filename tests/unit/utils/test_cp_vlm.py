"""Unit tests for the VLM-CP image-embed slice publishing in ``utils/cp.py``.

CPU-only: exercises ``setup_vlm_cp_params`` and the ``_VLM_CP_PARAMS``
lifecycle in isolation from the distributed CP machinery. The slice
math is what makes ring/ulysses CP correct for VLMs whose images span
CP shard boundaries — without it, every rank's ``masked_scatter`` would
use ``image_embeds[:local_count]`` instead of the right global subset.
"""

import torch

from prime_rl.utils.cp import (
    clear_vlm_cp_image_slice,
    get_vlm_cp_image_slice,
    set_vlm_cp_image_slice,
    setup_vlm_cp_params,
)


def test_vlm_cp_image_slice_lifecycle():
    clear_vlm_cp_image_slice()
    assert get_vlm_cp_image_slice() is None
    set_vlm_cp_image_slice(4, 12)
    assert get_vlm_cp_image_slice() == (4, 12)
    clear_vlm_cp_image_slice()
    assert get_vlm_cp_image_slice() is None


def test_setup_vlm_cp_params_no_split_at_boundary():
    """Single image fully inside one CP rank — only that rank gets a non-zero slice."""
    clear_vlm_cp_image_slice()
    # 16-token sequence, cp=4 → shards of 4. Image at positions 8..11 (full rank 2).
    mm_types = torch.zeros(1, 16, dtype=torch.long)
    mm_types[0, 8:12] = 1

    for rank in range(4):
        sharded = setup_vlm_cp_params(mm_types, cp_rank=rank, cp_world_size=4)
        start, count = get_vlm_cp_image_slice()
        assert sharded.shape == (1, 4)
        if rank == 2:
            assert (start, count) == (0, 4)
        else:
            # No image in this rank's slice; start = images-before = 0 for ranks 0,1 and 4 for rank 3.
            assert count == 0
            assert start == (4 if rank == 3 else 0)


def test_setup_vlm_cp_params_image_spanning_boundary():
    """Image straddling rank-1/rank-2 boundary — slice must split image_embeds correctly.

    This is the bug Phase 2b fixes: without the slice, every rank would index
    image_embeds[:local_count] and overlap incorrectly.
    """
    clear_vlm_cp_image_slice()
    # 16-token sequence, cp=4 → shards of 4. Image at positions 6..9 (2 tokens
    # on rank 1, 2 tokens on rank 2).
    mm_types = torch.zeros(1, 16, dtype=torch.long)
    mm_types[0, 6:10] = 1

    setup_vlm_cp_params(mm_types, cp_rank=0, cp_world_size=4)
    assert get_vlm_cp_image_slice() == (0, 0)

    setup_vlm_cp_params(mm_types, cp_rank=1, cp_world_size=4)
    # Rank 1 gets the first 2 image_embeds (positions 6-7 in global).
    assert get_vlm_cp_image_slice() == (0, 2)

    setup_vlm_cp_params(mm_types, cp_rank=2, cp_world_size=4)
    # Rank 2 gets the next 2 image_embeds (positions 8-9). Critical: start=2,
    # not 0 — without this, rank 2 would scatter image_embeds[0:2] (rank 1's
    # slice) instead of image_embeds[2:4].
    assert get_vlm_cp_image_slice() == (2, 2)

    setup_vlm_cp_params(mm_types, cp_rank=3, cp_world_size=4)
    assert get_vlm_cp_image_slice() == (4, 0)


def test_setup_vlm_cp_params_multiple_images():
    """Multiple images across CP ranks — counts must reflect per-rank image-token totals."""
    clear_vlm_cp_image_slice()
    # 16-token sequence, cp=2 → shards of 8. Two images: [2:5] on rank 0, [9:13] on rank 1.
    mm_types = torch.zeros(1, 16, dtype=torch.long)
    mm_types[0, 2:5] = 1  # 3 tokens (rank 0)
    mm_types[0, 9:13] = 1  # 4 tokens (rank 1)

    setup_vlm_cp_params(mm_types, cp_rank=0, cp_world_size=2)
    assert get_vlm_cp_image_slice() == (0, 3)

    setup_vlm_cp_params(mm_types, cp_rank=1, cp_world_size=2)
    assert get_vlm_cp_image_slice() == (3, 4)


def test_setup_vlm_cp_params_returns_sharded_token_type_ids():
    """``setup_vlm_cp_params`` returns the sharded mm_token_type_ids alongside publishing the slice."""
    clear_vlm_cp_image_slice()
    mm_types = torch.arange(8, dtype=torch.long).unsqueeze(0)
    sharded_rank0 = setup_vlm_cp_params(mm_types, cp_rank=0, cp_world_size=2)
    sharded_rank1 = setup_vlm_cp_params(mm_types, cp_rank=1, cp_world_size=2)
    torch.testing.assert_close(sharded_rank0, torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(sharded_rank1, torch.tensor([[4, 5, 6, 7]]))
