"""Unit tests for the document-aware VLM packing path in ``CatDataset``.

CPU-only: exercises the packing algorithm directly without touching the
collator (which moves tensors to CUDA). The fake VLM samples emitted by
``FakeDataset(multimodal=...)`` aren't shape-compatible with any real
vision encoder — these tests verify packer correctness in isolation.
"""

import torch

from prime_rl.configs.sft import FakeMultimodalConfig
from prime_rl.trainer.sft.data import CatDataset, FakeDataset


def _take(dataset, n):
    out = []
    for sample in dataset:
        out.append(sample)
        if len(out) >= n:
            break
    return out


def test_fake_dataset_multimodal_emits_mm_fields():
    mm = FakeMultimodalConfig(image_token_id=42, images_per_sample=2, image_tokens_per_image=3, fake_feature_dim=4)
    ds = FakeDataset(vocab_size=10000, seq_len=64, multimodal=mm)
    sample = next(iter(ds))

    assert "mm_kwargs" in sample and "mm_token_type_ids" in sample
    assert sample["mm_kwargs"]["pixel_values"].shape == (2 * 3, 4)
    assert sample["mm_kwargs"]["image_grid_thw"].shape == (2, 3)
    # type_ids must agree with input_ids on placeholder positions
    image_positions = [i for i, t in enumerate(sample["mm_token_type_ids"]) if t == 1]
    assert all(sample["input_ids"][i] == 42 for i in image_positions)
    assert len(image_positions) == 2 * 3


def test_cat_dataset_vlm_packs_multiple_docs_without_split():
    """Multiple short docs fit into one pack; doc boundaries preserved via position_ids resets."""
    mm = FakeMultimodalConfig(images_per_sample=1, image_tokens_per_image=2, fake_feature_dim=4)
    # Fixed length 16 → ~6 docs fit in seq_len=128.
    ds = FakeDataset(vocab_size=1000, seq_len=16, length="fixed", multimodal=mm)
    packer = CatDataset(ds, seq_len=128, pad_token_id=0)

    pack = next(iter(packer))

    assert len(pack["input_ids"]) == 128
    assert len(pack["position_ids"]) == 128
    assert len(pack["mm_token_type_ids"]) == 128
    # position_ids must reset to 0 at each doc start — utils/sequence.py uses
    # this as the cu_seqlens boundary marker for document-aware attention.
    doc_starts = [i for i, p in enumerate(pack["position_ids"]) if p == 0]
    assert len(doc_starts) >= 2, "expected multiple docs packed into one pack"
    # Within each doc the position_ids must be strictly increasing.
    for start, end in zip(doc_starts, doc_starts[1:] + [128]):
        assert pack["position_ids"][start:end] == list(range(end - start))


def test_cat_dataset_vlm_mm_kwargs_concat_along_dim0():
    """pixel_values / image_grid_thw concatenated across docs along dim 0."""
    mm = FakeMultimodalConfig(images_per_sample=2, image_tokens_per_image=3, fake_feature_dim=4)
    ds = FakeDataset(vocab_size=1000, seq_len=16, length="fixed", multimodal=mm)
    packer = CatDataset(ds, seq_len=128, pad_token_id=0)

    pack = next(iter(packer))
    n_docs = sum(1 for p in pack["position_ids"] if p == 0)
    # Each doc contributes 2 images × 3 placeholder tokens to pixel_values
    # (concat along dim 0), and 2 rows to image_grid_thw.
    assert pack["mm_kwargs"]["pixel_values"].shape == (n_docs * 2 * 3, 4)
    assert pack["mm_kwargs"]["image_grid_thw"].shape == (n_docs * 2, 3)
    assert pack["mm_kwargs"]["pixel_values"].dtype == torch.float32
    assert pack["mm_kwargs"]["image_grid_thw"].dtype == torch.long


def test_cat_dataset_vlm_pads_underfilled_pack():
    """When the next doc would overflow, yield the current pack padded to seq_len."""
    mm = FakeMultimodalConfig(images_per_sample=1, image_tokens_per_image=2, fake_feature_dim=4)
    # Doc len 10, seq_len 25 → 2 docs (20 tokens) fit; the 3rd would overflow.
    # Expected pack: 20 real tokens + 5 pad tokens.
    ds = FakeDataset(vocab_size=1000, seq_len=10, length="fixed", multimodal=mm)
    packer = CatDataset(ds, seq_len=25, pad_token_id=0)

    pack = next(iter(packer))
    assert len(pack["input_ids"]) == 25
    # The last 5 positions are pad: loss_mask False, mm_token_type_ids 0.
    assert pack["loss_mask"][20:] == [False] * 5
    assert pack["mm_token_type_ids"][20:] == [0] * 5
    assert pack["input_ids"][20:] == [0] * 5  # pad_token_id
