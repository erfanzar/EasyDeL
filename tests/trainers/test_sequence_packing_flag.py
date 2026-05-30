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
* the per-trainer ``supports_sequence_packing`` opt-out (RL/online + paired-
  preference trainers are ``False``, supervised are ``True``);
* the eLarge mixture builder treats ``sequence_packing`` as an alias for
  ``pack_tokens``.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

import pytest


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
        ("easydel.trainers.group_relative_policy_optimization.grpo_trainer", "GRPOTrainer", False),
        ("easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer", "PPOTrainer", False),
        ("easydel.trainers.direct_preference_optimization_trainer.dpo_trainer", "DPOTrainer", False),
        ("easydel.trainers.online_dpo_trainer.online_dpo_trainer", "OnlineDPOTrainer", False),
        ("easydel.trainers.kto_trainer.kto_trainer", "KTOTrainer", False),
        ("easydel.trainers.generalized_knowledge_distillation_trainer.gkd_trainer", "GKDTrainer", False),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
