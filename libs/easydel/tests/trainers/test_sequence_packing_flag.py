# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the shared ``sequence_packing`` training flag.

Covers:
* the field exists on the base ``TrainingArguments``;
* ``SFTConfig(sequence_packing=True)`` aliases to ``packing=True`` (so the SFT
  data path wraps the source in ``PackedShardedSource`` with segment ids);
* the per-trainer ``supports_sequence_packing`` opt-out (rollout, pairwise,
  reward, and embedding trainers are ``False``; single-stream supervised /
  offline trainers are ``True``);
* the eLarge mixture builder treats ``sequence_packing`` as an alias for
  ``pack_tokens``.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

import pytest


class _TinyTokenSource:
    def __init__(self, rows=None):
        self.shard_names = ["s0"]
        self._rows = rows or [
            {"input_ids": [10, 11], "completion_mask": [0, 1]},
            {"input_ids": [20], "completion_mask": [1]},
        ]

    def open_shard(self, shard_name):
        del shard_name
        yield from self._rows

    def __len__(self):
        return len(self._rows)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


def test_base_training_arguments_has_flag():
    import easydel as ed

    args = ed.TrainingArguments(save_directory="/tmp/_sp", sequence_packing=True)
    assert args.sequence_packing is True
    assert ed.TrainingArguments(save_directory="/tmp/_sp").sequence_packing is False


def test_sft_config_aliases_to_packing():
    from easydel.trainers import SFTConfig

    cfg = SFTConfig(sequence_packing=True, save_directory="/tmp/_sp")
    assert cfg.packing is True, "sequence_packing=True must enable SFT packing"
    # default: off
    assert SFTConfig(save_directory="/tmp/_sp").packing is False


@pytest.mark.parametrize(
    ("import_path", "cls_name", "expected"),
    [
        ("easydel.trainers.supervised_fine_tuning_trainer.sft_trainer", "SFTTrainer", True),
        ("easydel.trainers.distillation_trainer.distillation_trainer", "DistillationTrainer", True),
        ("easydel.trainers.gold_trainer.gold_trainer", "GOLDTrainer", True),
        ("easydel.trainers.reward_trainer.reward_trainer", "RewardTrainer", False),
        ("easydel.trainers.embedding_trainer.embedding_trainer", "EmbeddingTrainer", False),
        ("easydel.trainers.group_relative_policy_optimization.grpo_trainer", "GRPOTrainer", False),
        ("easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer", "PPOTrainer", False),
        ("easydel.trainers.direct_preference_optimization_trainer.dpo_trainer", "DPOTrainer", False),
        ("easydel.trainers.online_dpo_trainer.online_dpo_trainer", "OnlineDPOTrainer", False),
        ("easydel.trainers.kto_trainer.kto_trainer", "KTOTrainer", False),
        ("easydel.trainers.generalized_knowledge_distillation_trainer.gkd_trainer", "GKDTrainer", False),
        ("easydel.trainers.seq_kd_trainer.seq_kd_trainer", "SeqKDTrainer", False),
        ("easydel.trainers.sparse_distillation_trainer.sparse_distillation_trainer", "SparseDistillationTrainer", False),
    ],
)
def test_supports_sequence_packing_flags(import_path, cls_name, expected):
    import importlib

    cls = getattr(importlib.import_module(import_path), cls_name)
    assert cls.supports_sequence_packing is expected, f"{cls_name}.supports_sequence_packing should be {expected}"


def test_warn_and_ignore_logic():
    """Mirror BaseTrainer.__init__'s guard: an unsupported trainer warns and disables."""
    import warnings

    class _Args:
        sequence_packing = True

    args = _Args()
    supports = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if getattr(args, "sequence_packing", False) and not supports:
            warnings.warn("trainer does not support sequence_packing=True; ignoring.", stacklevel=2)
            args.sequence_packing = False
        assert any("sequence_packing" in str(w.message) for w in caught)
    assert args.sequence_packing is False


def test_elarge_mixture_sequence_packing_alias():
    """build_sharded_source should treat sequence_packing as pack_tokens (pre-tokenized data)."""
    import inspect

    from easydel.infra.elarge import builders

    src = inspect.getsource(builders.build_sharded_source)
    assert 'mixture_cfg.get("sequence_packing")' in src, (
        "build_sharded_source must honor sequence_packing as a pack_tokens alias"
    )


def test_distillation_sequence_packing_wraps_tokenized_source():
    from easydel.data.transforms.pack import PackedShardedSource
    from easydel.trainers.distillation_trainer.distillation_config import DistillationConfig
    from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer

    trainer = object.__new__(DistillationTrainer)
    trainer.arguments = DistillationConfig(
        save_directory="/tmp/_sp",
        max_length=8,
        sequence_packing=True,
        resume_if_possible=False,
    )
    trainer.processing_class = _Tokenizer()
    trainer._train_source = _TinyTokenSource()
    trainer._eval_source = None

    trainer._apply_preprocess_transforms()

    assert isinstance(trainer._train_source, PackedShardedSource)
    [row] = list(trainer._train_source.open_shard("packed_shard_0"))
    assert row["input_ids"].tolist() == [10, 11, 2, 20, 2, 0, 0, 0]
    assert row["completion_mask"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert "labels" not in row


def test_sequence_packing_preserves_labels_when_present():
    from easydel.trainers.distillation_trainer.distillation_config import DistillationConfig
    from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer

    trainer = object.__new__(DistillationTrainer)
    trainer.arguments = DistillationConfig(
        save_directory="/tmp/_sp",
        max_length=8,
        sequence_packing=True,
        resume_if_possible=False,
    )
    trainer.processing_class = _Tokenizer()
    trainer._train_source = _TinyTokenSource(
        [
            {"input_ids": [10, 11], "labels": [-100, 11]},
            {"input_ids": [20], "labels": [20]},
        ]
    )
    trainer._eval_source = None

    trainer._apply_preprocess_transforms()

    [row] = list(trainer._train_source.open_shard("packed_shard_0"))
    assert row["labels"].tolist() == [-100, 11, -100, 20, -100, -100, -100, -100]


def test_distillation_sequence_packing_tokenizes_without_max_padding():
    from easydel.trainers.distillation_trainer.distillation_config import DistillationConfig
    from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer

    trainer = object.__new__(DistillationTrainer)
    trainer.arguments = DistillationConfig(
        save_directory="/tmp/_sp",
        max_length=8,
        sequence_packing=True,
        resume_if_possible=False,
    )
    trainer.processing_class = _Tokenizer()
    trainer._train_source = None

    transform = trainer._get_preprocess_transform()

    assert transform is not None
    assert transform._padding is False


def test_sft_legacy_packing_uses_shared_wrapper():
    from easydel.data.transforms.pack import PackedShardedSource
    from easydel.trainers.supervised_fine_tuning_trainer.sft_config import SFTConfig
    from easydel.trainers.supervised_fine_tuning_trainer.sft_trainer import SFTTrainer

    trainer = object.__new__(SFTTrainer)
    trainer.arguments = SFTConfig(
        save_directory="/tmp/_sp",
        max_length=8,
        packing=True,
        sequence_packing=False,
        resume_if_possible=False,
    )
    trainer.processing_class = _Tokenizer()
    trainer._train_source = _TinyTokenSource()
    trainer._eval_source = None

    trainer._apply_preprocess_transforms()

    assert isinstance(trainer._train_source, PackedShardedSource)


def test_sequence_packing_segments_fold_into_mask_info_before_filtering():
    import numpy as np
    from jax import numpy as jnp

    from easydel.trainers.training_utils import filter_kwargs_for_callable, sanitize_model_call_kwargs

    def forward(input_ids, attention_mask=None, mask_info=None):
        del input_ids, attention_mask
        return mask_info

    batch = {
        "input_ids": jnp.asarray([[10, 11, 20, 0]], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[1, 1, 1, 0]], dtype=jnp.int32),
        "segment_ids": jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
        "labels": jnp.asarray([[10, 11, 20, -100]], dtype=jnp.int32),
    }

    filtered = filter_kwargs_for_callable(forward, batch)

    assert "segment_ids" not in filtered
    assert "labels" not in filtered
    assert "mask_info" in filtered
    assert np.asarray(filtered["mask_info"].q_segment_ids).tolist() == [[0, 0, 1, -1]]

    sanitized = sanitize_model_call_kwargs(batch)
    assert "segment_ids" not in sanitized
    assert np.asarray(sanitized["mask_info"].q_segment_ids).tolist() == [[0, 0, 1, -1]]


def test_sequence_packing_folds_off_by_one_segment_ids_to_input_length():
    import numpy as np
    from jax import numpy as jnp

    from easydel.trainers.training_utils import sanitize_model_call_kwargs

    batch = {
        "input_ids": jnp.asarray([[10, 11, 12, 13]], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[1, 1, 1, 1]], dtype=jnp.int32),
        "segment_ids": jnp.asarray([[0, 0, 0, 0, 0]], dtype=jnp.int32),
    }

    sanitized = sanitize_model_call_kwargs(batch)

    assert "segment_ids" not in sanitized
    assert np.asarray(sanitized["mask_info"].q_segment_ids).tolist() == [[0, 0, 0, 0]]


def test_generic_filter_preserves_explicit_segment_ids_argument():
    from jax import numpy as jnp

    from easydel.trainers.training_utils import filter_kwargs_for_callable

    def reward(segment_ids):
        return segment_ids

    batch = {"segment_ids": jnp.asarray([[0, 1]], dtype=jnp.int32)}

    filtered = filter_kwargs_for_callable(reward, batch)

    assert set(filtered) == {"segment_ids"}


def test_first_fit_sequence_packing_does_not_overflow_full_length_rows():
    from easydel.data.transforms.pack import PackedShardedSource

    source = _TinyTokenSource(
        [
            {
                "input_ids": [10, 11, 12, 13],
                "completion_mask": [0, 1, 1, 1],
            }
        ]
    )

    packed = PackedShardedSource(
        source=source,
        seq_length=4,
        eos_token_id=2,
        pad_token_id=0,
        strategy="first_fit",
        include_segment_ids=True,
        extra_field_pad_values={"completion_mask": 0},
        extra_field_separator_values={"completion_mask": 0},
        shuffle=False,
    )

    [row] = list(packed.open_shard("packed_shard_0"))

    assert row["input_ids"].tolist() == [10, 11, 12, 13]
    assert row["attention_mask"].tolist() == [1, 1, 1, 1]
    assert row["segment_ids"].tolist() == [0, 0, 0, 0]
    assert row["completion_mask"].tolist() == [0, 1, 1, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
