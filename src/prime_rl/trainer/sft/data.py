import json
import uuid
from collections import defaultdict
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from jaxtyping import Bool, Int
from renderers.base import Renderer, build_training_sample
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import DataConfig, FakeMultimodalConfig, LossMaskConfig, SFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import (
    IncrementalTokenizationError,
    build_incremental_token_mask,
    deserialize_tool_calls,
    normalize_messages,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger

STACKING_DATASET_BUCKET_TIMEOUT = 10


class Sample(TypedDict, total=False):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]

    # Optional multimodal fields. Absent for text-only samples; populated by a
    # VLM-aware dataset (Phase 1). Mirrors the RL ``TensorMicroBatch`` contract
    # (see ``trainer/rl/data.py``): ``mm_kwargs`` is a flat dict matching the
    # model's HF forward signature (e.g. ``pixel_values``, ``image_grid_thw``
    # for Qwen3-VL), and ``mm_token_type_ids`` is renderer-supplied per-token
    # modality ids (0=text, 1=image, 2=video).
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: list[int] | None


class Batch(TypedDict, total=False):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]

    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[Tensor, "batch seq"] | None


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self):
        self.step, self.epoch = 0, 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.fast_forward = False
        self._setup_world_info()

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.fast_forward = True
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]

    def _setup_world_info(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        self.data_rank = get_world().rank * num_workers + worker_id
        self.data_world_size = get_world().world_size * num_workers


class FakeDataset(StatefulIterableDataset):
    """A dataset of fake tokens"""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        length: Literal["fixed", "variable"] = "fixed",
        input_ids: Literal["increasing", "random"] = "random",
        multimodal: FakeMultimodalConfig | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.input_ids = input_ids
        # ``multimodal`` is opt-in. When set, each yielded sample carries
        # toy ``mm_kwargs`` / ``mm_token_type_ids`` so the document-aware
        # VLM packer can be exercised without a real VLM dataset.
        self.multimodal = multimodal

    def __iter__(self):
        while True:
            self.step += 1

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            seq_len = int(torch.randint(1, self.seq_len, (1,)).item()) if self.length == "variable" else self.seq_len
            input_ids = (
                [self.step - 1] * (seq_len + 1)
                if self.input_ids == "increasing"
                else torch.randint(0, self.vocab_size, (self.seq_len + 1,)).long().tolist()
            )
            position_ids = list(range(seq_len))
            loss_mask = [True] * seq_len
            fake_sample = {
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:],
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if self.multimodal is not None:
                self._inject_multimodal(fake_sample, self.multimodal)
            self.num_samples["fake"] += 1
            self.num_tokens["fake"] += len(input_ids)
            yield fake_sample

    @staticmethod
    def _inject_multimodal(sample: dict, mm: FakeMultimodalConfig) -> None:
        """Overlay fake image-placeholder tokens and attach toy mm tensors.

        Mirrors the RL contract (``trainer/rl/data.py::TensorMicroBatch``):
        ``mm_kwargs`` is a flat dict matching the model forward signature
        (``pixel_values`` + ``image_grid_thw`` for Qwen3-VL families), with
        no batch dim — per-image concat along dim 0. ``mm_token_type_ids``
        is per-token (0=text, 1=image). Shapes here are toy and won't run
        through a real vision encoder; the goal is to exercise the packer.
        """
        seq = len(sample["input_ids"])
        n_imgs = mm.images_per_sample
        tokens_per_img = mm.image_tokens_per_image
        total_img_tokens = n_imgs * tokens_per_img
        if total_img_tokens >= seq:
            # Not enough room for placeholders + at least one text token —
            # leave this sample text-only rather than overflowing.
            return

        # Place each image's run of placeholders at evenly-spaced offsets so
        # multiple-image docs exercise interleaving. The exact offsets don't
        # matter for packer correctness, only that mm_token_type_ids agrees
        # with input_ids.
        ids = list(sample["input_ids"])
        type_ids = [0] * seq
        stride = max(1, (seq - total_img_tokens) // n_imgs)
        cursor = 0
        for _ in range(n_imgs):
            for k in range(tokens_per_img):
                pos = cursor + k
                ids[pos] = mm.image_token_id
                type_ids[pos] = 1
            cursor += tokens_per_img + stride

        sample["input_ids"] = ids
        # Re-derive target_ids so EOS/image-token placement stays consistent
        # with input_ids (target = next-token shift, like SFTDataset).
        sample["target_ids"] = ids[1:] + [ids[-1]]
        sample["mm_token_type_ids"] = type_ids
        # ``pixel_values``: per-image, leading dim = grid token count, second
        # dim = a small fake feature dim. Concatenate per-image along dim 0
        # so the packer's dim-0 concat across docs is exercised non-trivially.
        sample["mm_kwargs"] = {
            "pixel_values": torch.zeros(n_imgs * tokens_per_img, mm.fake_feature_dim, dtype=torch.float32),
            # ``image_grid_thw``: one row per image, [T, H, W]. Toy 1×1×K so
            # T*H*W == tokens_per_img — the convention every Qwen3-VL family
            # tile uses.
            "image_grid_thw": torch.tensor([[1, 1, tokens_per_img]] * n_imgs, dtype=torch.long),
        }


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt/completion or raw messages format."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer | None,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        renderer: Renderer | None = None,
    ):
        super().__init__()
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(self.dataset)
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.max_examples = max_examples
        self.max_epochs = max_epochs
        self.renderer = renderer
        self._warned_chat_template_kwargs = False

        if self.tokenizer is None:
            self.logger.warning("No tokenizer provided, will not process examples")

        # If specified, select a subset of the dataset
        if self.max_examples is not None:
            self.num_examples = min(self.num_examples, self.max_examples)
            self.dataset = self.dataset.take(self.max_examples)

        # Get the data rank and world size
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        assert get_world().world_size % non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = get_world().rank // non_dp_size * num_workers + worker_id
        self.data_world_size = get_world().world_size // non_dp_size * num_workers

    def _process(self, example: dict) -> dict | None:
        # Skip processing if no tokenizer was provided
        if self.tokenizer is None:
            return example

        def resolve_messages(example: dict) -> list[dict]:
            # `messages` takes precedence over explicit split fields and is interpreted
            # as a whole-chat training sample with an empty prompt.
            if "messages" in example:
                messages = normalize_messages(example["messages"], default_role="assistant")
            elif "prompt" in example and "completion" in example:
                messages = normalize_messages(example["prompt"], default_role="user") + normalize_messages(
                    example["completion"], default_role="assistant"
                )
            else:
                raise ValueError(
                    "All examples in the dataset must have either a 'messages' column "
                    "or both 'prompt' and 'completion' columns for SFT"
                )

            # Deserialize tool call arguments from message list, if present - assumes OAI format
            # Reference: https://platform.openai.com/docs/guides/function-calling#handling-function-calls
            messages = deserialize_tool_calls(messages)

            # Strip content from all messages so that incremental tokenization works
            # NOTE: This has the side effect that we do never train on leading or trailing whitespace
            return strip_message_content(messages)

        messages = resolve_messages(example)

        # Parse available tools, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#function-tool-example
        # Accepts either `tools` or `tool_defs` (the verifiers rollout format),
        # as either a JSON-encoded string of a list or a list of dicts. Tools
        # arriving in the verifiers shape are converted to OAI form so any
        # downstream chat template can consume them.
        raw_tools = example.get("tools", example.get("tool_defs"))
        if not raw_tools:
            tools = []
        else:
            if isinstance(raw_tools, str):
                raw_tools = json.loads(raw_tools)
            tools = [
                t
                if isinstance(t, dict) and t.get("type") == "function" and "function" in t
                else {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "parameters": t.get("parameters"),
                        **({} if t.get("strict") is None else {"strict": t["strict"]}),
                    },
                }
                for t in raw_tools
            ]

        def should_mask(message: dict) -> bool:
            assert "role" in message, "Message must have a role"
            match message["role"]:
                case "user":
                    return True if self.loss_mask_config.user else False
                case "assistant":
                    return True if self.loss_mask_config.assistant else False
                case "system":
                    return True if self.loss_mask_config.system else False
                case "tool":
                    return True if self.loss_mask_config.tool else False
                case _:
                    raise ValueError(f"Invalid message role: {message['role']}")

        if self.renderer is not None:
            if example.get("chat_template_kwargs") and not self._warned_chat_template_kwargs:
                self.logger.warning(
                    "Example carries chat_template_kwargs but use_renderer=True; "
                    "renderers don't forward chat_template_kwargs (model-specific "
                    "renderers bake their template behavior in). These kwargs will "
                    "be ignored. Further warnings suppressed for this dataset."
                )
                self._warned_chat_template_kwargs = True

            input_ids, loss_mask = build_training_sample(
                self.renderer,
                messages,
                role_to_mask=should_mask,
                tools=tools,
            )
        else:
            try:
                input_ids, loss_mask = build_incremental_token_mask(
                    self.tokenizer,
                    messages,
                    role_to_mask=should_mask,
                    tools=tools,
                    chat_template_kwargs=example.get("chat_template_kwargs", {}),
                    collapse_consecutive_tool_messages=True,
                )
            except IncrementalTokenizationError as e:
                self.logger.warning(f"Skipping example {example.get('__index', '')}: {e}")
                return None

        # If EOS token is not found, manually append it
        if not self.tokenizer.eos_token_id in input_ids:
            self.logger.warning(
                f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. Is something wrong with the chat template? Manually appending EOS token..."
            )
            input_ids.append(cast(int, self.tokenizer.eos_token_id))
            loss_mask.append(True)

        # Prepare inputs
        target_ids = input_ids.copy()[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

        # Create sample (with one fake target for the last token)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
        }

    def __iter__(self):
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
        while True:
            self.step += 1

            # Determine epoch from current step
            epoch = (self.step - 1) // self.num_examples

            # Break if max epochs is reached
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break

            # Update stored epoch if new epoch is reached, optionally shuffle
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            # Get example
            example = dataset[(self.step - 1) % self.num_examples]

            # Process example
            processed_example = self._process(cast(dict, example))

            # If processed example is None, skip it (e.g. if tokenized sample exceeds context window)
            if processed_example is None:
                continue

            # Yield the example
            example = cast(dict, example)
            subset_or_split = example.get("__subset") or example.get("__split")
            self.logger.debug(
                f"Yield example {example.get('__index', '')}"
                + (f" from {subset_or_split} " if subset_or_split else " ")
                + f"with {len(processed_example.get('input_ids', []))} tokens ({sum(processed_example.get('loss_mask', []))} trainable tokens)"
            )
            self.num_samples[subset_or_split] += 1
            self.num_tokens[subset_or_split] += len(processed_example.get("input_ids", []))
            yield processed_example


class CatDataset(StatefulIterableDataset):
    """A dataset that concatenates samples into a single sequence with a fixed length.

    Dispatches on the first sample: text-only samples use the original overrun
    + truncate-to-``seq_len`` policy; multimodal samples (carrying ``mm_kwargs``
    or ``mm_token_type_ids``) use a document-aware path that never splits a
    document mid-stream — splitting would separate an image's placeholder
    tokens from its ``pixel_values`` entry. The VLM path concatenates per-doc
    ``mm_kwargs`` tensors along dim 0 (mirroring the RL orchestrator
    convention; see ``orchestrator/trajectories.py::_pack_mm_kwargs_from_renderer``)
    and pads each pack to ``seq_len`` so all packs have a consistent shape.

    Document-aware attention masking falls out for free: the existing
    ``cu_seqlens`` derivation in ``utils/sequence.py`` keys on
    ``position_ids == 0`` markers, and the VLM path preserves each
    document's per-doc 0-indexed position_ids unchanged.
    """

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int, pad_token_id: int = 0):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len
        # Used only when padding underfilled VLM packs. Defaults to 0 because
        # not every tokenizer defines a pad token; loss_mask=False on pad
        # positions so the choice never affects training.
        self.pad_token_id = pad_token_id

    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def __iter__(self):
        iterator = iter(self.dataset)
        try:
            first = next(iterator)
        except StopIteration:
            return

        # Peek the first sample to choose the packing strategy. Mixed streams
        # (text + mm samples interleaved) are not supported: the text path's
        # ``extend(value)`` would fail on a dict ``mm_kwargs`` with a clear
        # assertion, and the VLM path assumes every doc carries the same
        # ``mm_kwargs`` schema.
        is_multimodal = first.get("mm_kwargs") is not None or first.get("mm_token_type_ids") is not None

        def chained():
            yield first
            yield from iterator

        if is_multimodal:
            yield from self._iter_vlm(chained())
        else:
            yield from self._iter_text(chained())

    def _iter_text(self, samples):
        packed_samples, seq_len = defaultdict(list), 0
        for sample in samples:
            # Add sample to packed samples
            for key, value in sample.items():
                assert isinstance(value, list), f"Value for key {key} must be a list"
                packed_samples[key].extend(value)

            # Update sequence length
            seq_len += len(sample["input_ids"])

            # If batch is full, truncate and yield it
            if seq_len >= self.seq_len:
                for key, value in packed_samples.items():
                    assert isinstance(value, list), f"Value for key {key} must be a list"
                    packed_samples[key] = value[: self.seq_len]
                yield packed_samples
                packed_samples, seq_len = defaultdict(list), 0

    def _iter_vlm(self, samples):
        packed_samples = defaultdict(list)
        pending_mm_kwargs: list[dict[str, Tensor]] = []
        seq_len_acc = 0

        for sample in samples:
            sample_len = len(sample["input_ids"])

            # Truncating mid-doc is unsafe (could split an image from its
            # placeholders). Drop oversize docs with a warning so the stream
            # keeps flowing; the upstream filter should size samples to fit.
            if sample_len > self.seq_len:
                self.logger.warning(
                    f"Dropping multimodal sample of length {sample_len} > seq_len ({self.seq_len}); "
                    "truncation would risk splitting an image from its placeholders."
                )
                continue

            # Flush before adding when this doc would overflow. Yields a
            # (possibly underfilled) pack rather than splitting the new doc.
            if seq_len_acc > 0 and seq_len_acc + sample_len > self.seq_len:
                self._finalize_vlm_pack(packed_samples, pending_mm_kwargs, seq_len_acc)
                yield packed_samples
                packed_samples = defaultdict(list)
                pending_mm_kwargs = []
                seq_len_acc = 0

            for key, value in sample.items():
                if key == "mm_kwargs":
                    if value is not None:
                        pending_mm_kwargs.append(value)
                elif isinstance(value, list):
                    packed_samples[key].extend(value)

            seq_len_acc += sample_len

            # Exact fit — yield without pad to keep packs maximally dense.
            if seq_len_acc == self.seq_len:
                self._finalize_vlm_pack(packed_samples, pending_mm_kwargs, seq_len_acc)
                yield packed_samples
                packed_samples = defaultdict(list)
                pending_mm_kwargs = []
                seq_len_acc = 0

    def _finalize_vlm_pack(
        self,
        packed_samples: dict,
        pending_mm_kwargs: list[dict[str, Tensor]],
        seq_len_acc: int,
    ) -> None:
        """In place: pad token-level lists to ``seq_len`` and attach concatenated ``mm_kwargs``."""
        pad = self.seq_len - seq_len_acc
        if pad > 0:
            packed_samples["input_ids"].extend([self.pad_token_id] * pad)
            packed_samples["target_ids"].extend([self.pad_token_id] * pad)
            # Continue position_ids from the last value so pad tokens stay
            # inside the trailing document's cu_seqlens window — opening a
            # fresh window on pad-only tokens would waste attention compute
            # on pad self-attention.
            last_pos = packed_samples["position_ids"][-1] if packed_samples["position_ids"] else -1
            packed_samples["position_ids"].extend(range(last_pos + 1, last_pos + 1 + pad))
            packed_samples["loss_mask"].extend([False] * pad)
            packed_samples["mm_token_type_ids"].extend([0] * pad)

        mm_kwargs: dict[str, Tensor] = {}
        if pending_mm_kwargs:
            keys = set(pending_mm_kwargs[0].keys())
            for doc in pending_mm_kwargs[1:]:
                if set(doc.keys()) != keys:
                    raise ValueError(
                        "Inconsistent mm_kwargs keys across packed docs "
                        f"({sorted(keys)} vs {sorted(doc.keys())}). Every doc in a "
                        "multimodal stream must share the same model's forward signature."
                    )
            for k in pending_mm_kwargs[0]:
                mm_kwargs[k] = torch.cat([doc[k] for doc in pending_mm_kwargs], dim=0)
        packed_samples["mm_kwargs"] = mm_kwargs


class StackDataset(StatefulIterableDataset):
    """A dataset that stacks samples into batch with a fixed area"""

    def __init__(self, dataset: StatefulIterableDataset, max_area: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.max_area = max_area
        assert self.max_area % 256 == 0
        self.bucket_sizes = []
        while max_area % 256 == 0:
            self.bucket_sizes.insert(0, max_area)
            max_area //= 2
        self.logger.debug(f"Initialized {len(self.bucket_sizes)} buckets (bucket_sizes={self.bucket_sizes})")
        # Checkpoint state
        self.step = 0
        self.buckets = [[] for _ in range(len(self.bucket_sizes))]
        self.bucket_timers: list[int | None] = [None] * len(self.buckets)

    def state_dict(self) -> dict:
        return {
            "dataset": self.dataset.state_dict(),
            "step": self.step,
            "buckets": self.buckets,
            "bucket_timers": self.bucket_timers,
        }

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.step = state_dict["step"]
        self.buckets = state_dict["buckets"]
        self.bucket_timers = state_dict["bucket_timers"]

    def __iter__(self):
        for sample in self.dataset:
            if sample.get("mm_kwargs") is not None or sample.get("mm_token_type_ids") is not None:
                raise NotImplementedError(
                    "Multimodal samples are not supported by StackDataset bucketing. "
                    "Use pack_function='cat' for VLM: CatDataset has a document-aware "
                    "path that concatenates per-doc mm_kwargs and never splits an image."
                )
            # Truncate sample if it's longer than max area
            len_sample = len(sample["input_ids"])
            if len_sample > self.max_area:
                for key, value in sample.items():
                    assert isinstance(value, list)
                    sample[key] = sample[key][: self.max_area]
                len_sample = self.max_area

            # Add sample to bucket
            def find_bucket_idx(len_sample: int) -> int:
                bucket_idx = 0
                while bucket_idx < len(self.bucket_sizes) - 1 and len_sample > self.bucket_sizes[bucket_idx]:
                    bucket_idx += 1
                return bucket_idx

            bucket_idx = find_bucket_idx(len_sample)
            self.buckets[bucket_idx].append(sample)

            # Check if bucket has timed out
            bucket_timer = self.bucket_timers[bucket_idx]
            if bucket_timer is not None:
                hit_timeout = bucket_timer + STACKING_DATASET_BUCKET_TIMEOUT < self.step
            else:
                hit_timeout = False

            # Check if bucket is full
            is_full = self.bucket_sizes[bucket_idx] * len(self.buckets[bucket_idx]) >= self.max_area

            if is_full or hit_timeout:
                if hit_timeout:
                    while bucket_idx < len(self.buckets) - 1:
                        if (
                            self.bucket_sizes[bucket_idx + 1]
                            * (len(self.buckets[bucket_idx]) + len(self.buckets[bucket_idx + 1]))
                            < self.max_area
                        ):
                            self.buckets[bucket_idx + 1].extend(self.buckets[bucket_idx])
                            self.buckets[bucket_idx] = []
                            self.bucket_timers[bucket_idx] = None
                            bucket_idx += 1
                        else:
                            break

                    while self.bucket_sizes[bucket_idx] * len(self.buckets[bucket_idx]) < self.max_area:
                        dummy_sample = {}
                        for key, value in sample.items():
                            dummy_sample[key] = [0]
                        self.buckets[bucket_idx].append(dummy_sample)

                packed_samples = defaultdict(list)
                num_samples, num_tokens, num_trainable_tokens, num_pad_tokens = 0, 0, 0, 0
                for bucket_item in self.buckets[bucket_idx]:
                    num_samples += 1
                    for key, value in bucket_item.items():
                        pad_tokens = [0] * (self.bucket_sizes[bucket_idx] - len(value))
                        if key == "loss_mask":
                            num_tokens += len(value)
                            num_trainable_tokens += sum(value)
                            num_pad_tokens += len(pad_tokens)
                        packed_samples[key].append(value + pad_tokens)
                reason = "bucket is full" if is_full else "because bucket timed out"
                reason += " and " if is_full and hit_timeout else ""
                reason += "bucket timed out" if hit_timeout else ""
                self.logger.debug(
                    f"Yield bucket {bucket_idx} because {reason} with {num_samples=}, {num_tokens=}, {num_trainable_tokens=}, {num_pad_tokens=}"
                )
                yield packed_samples
                self.step += 1
                self.buckets[bucket_idx] = []
                self.bucket_timers[bucket_idx] = None
            else:
                if self.bucket_timers[bucket_idx] is None:
                    self.bucket_timers[bucket_idx] = self.step


def stack_collate(samples: list[Sample]) -> Batch:
    batch: Batch = {
        "input_ids": torch.tensor(samples[0]["input_ids"], dtype=torch.long, device="cuda"),
        "position_ids": torch.tensor(samples[0]["position_ids"], dtype=torch.long, device="cuda"),
        "loss_mask": torch.tensor(samples[0]["loss_mask"], dtype=torch.bool, device="cuda"),
        "target_ids": torch.tensor(samples[0]["target_ids"], dtype=torch.long, device="cuda"),
    }
    # Pass multimodal tensors through if present. ``mm_kwargs`` follows the RL
    # convention: no batch dim, per-image concat along dim 0 (see
    # ``trainer/rl/data.py::_micro_batch_to_tensor``). The model's HF forward
    # is the schema.
    mm_kwargs = samples[0].get("mm_kwargs")
    if mm_kwargs is not None:
        batch["mm_kwargs"] = {k: v.to("cuda") for k, v in mm_kwargs.items()}
    mm_token_type_ids = samples[0].get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        batch["mm_token_type_ids"] = torch.tensor(mm_token_type_ids, dtype=torch.long, device="cuda")
    return batch


def cat_collate(samples: list[Sample]) -> Batch:
    batch: Batch = {
        "input_ids": torch.stack([torch.tensor(sample["input_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "position_ids": torch.stack([torch.tensor(sample["position_ids"]) for sample in samples], dim=0)
        .long()
        .to("cuda"),
        "loss_mask": torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples], dim=0).bool().to("cuda"),
        "target_ids": torch.stack([torch.tensor(sample["target_ids"]) for sample in samples], dim=0).long().to("cuda"),
    }
    # Defensive pass-through: CatDataset currently refuses to pack multimodal
    # samples (NotImplementedError), so we only ever see one sample here when
    # mm fields are set. Phase 1 will revisit this when document-aware VLM
    # packing lands.
    mm_kwargs = samples[0].get("mm_kwargs") if samples else None
    if mm_kwargs is not None:
        batch["mm_kwargs"] = {k: v.to("cuda") for k, v in mm_kwargs.items()}
    mm_token_type_ids = samples[0].get("mm_token_type_ids") if samples else None
    if mm_token_type_ids is not None:
        batch["mm_token_type_ids"] = torch.tensor(mm_token_type_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    return batch


def setup_and_interleave_datasets(
    dataset_name: str,
    subsets_and_splits: list[tuple[str | None, str]],
    probabilities: list[float] | None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"],
    seed: int = 0,
) -> Dataset:
    logger = get_logger()
    datasets = []
    for subset, split in subsets_and_splits:
        logger.debug(f"Loading dataset {dataset_name} with {subset=} and {split=}")
        dataset = cast(Dataset, load_dataset(dataset_name, subset, split=split))
        num_examples = len(dataset)
        dataset = dataset.add_column("__subset", [subset] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__split", [split] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__index", list(range(num_examples)), new_fingerprint=str(uuid.uuid4()))
        datasets.append(dataset)
    if len(datasets) > 1:
        logger.debug(f"Interleaving datasets with {probabilities=} and {stopping_strategy=}")
        dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
            seed=seed,
        )
    else:
        dataset = datasets[0]

    return dataset


def load_sft_dataset(config: SFTDataConfig) -> Dataset:
    """Load and interleave the raw HF dataset. This is the expensive I/O step."""
    logger = get_logger()
    if config.subsets is None and config.splits is None:
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, "train")],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is not None and config.splits is None:
        logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(subset, "train") for subset in config.subsets],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is None and config.splits is not None:
        logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, split) for split in config.splits],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    else:
        assert config.subsets is not None and config.splits is not None
        logger.debug(f"Loading datasets for subsets {config.subsets} with splits {config.splits}")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=list(zip(config.subsets, config.splits)),
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    non_dp_size: int = 1,
    *,
    max_epochs: int | None = None,
    raw_dataset: Dataset | None = None,
    renderer: Renderer | None = None,
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size,
            seq_len=config.seq_len,
            length=config.length,
            input_ids=config.input_ids,
            multimodal=config.multimodal,
        )
    elif config.type == "sft":
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config)
        return SFTDataset(
            raw_dataset,
            tokenizer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            renderer=renderer,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(
    dataset: StatefulIterableDataset,
    config: DataConfig,
    pad_token_id: int = 0,
) -> StatefulDataLoader:
    if config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        # ``pad_token_id`` is only consumed by the VLM packing path (used to
        # pad underfilled doc-aware packs); text-only packs never pad.
        packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size, pad_token_id=pad_token_id)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
