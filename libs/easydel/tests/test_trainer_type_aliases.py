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

from __future__ import annotations

import sys

import pytest

_REQUIRES_PY311 = sys.version_info >= (3, 11)


def _load_trainer_types_module():
    from easydel.infra.elarge.types import training

    return training


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_nash_md_alias_normalized():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "nash_md"})
    assert config["trainer_type"] == "nash-md"


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_ppo_normalized_and_defaults_present():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "ppo"})
    assert config["trainer_type"] == "ppo"
    assert config["trainer_prefix"] == "PPO"
    assert "cliprange" in config
    assert "vf_coef" in config
    assert config["entropy_coef"] is None
    assert config["logprob_vocab_chunk_size"] is None
    assert config["presence_penalty"] == 0.0
    assert config["frequency_penalty"] == 0.0
    assert config["repetition_penalty"] == 1.0


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_grpo_normalized_and_logprob_chunk_default_present():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "grpo"})
    assert config["trainer_type"] == "grpo"
    assert config["trainer_prefix"] == "GRPO"
    assert config["ref_logps_chunk_size"] is None
    assert config["completion_chunk_size"] is None
    assert config["max_loss_completion_tokens"] is None
    assert config["logprob_vocab_chunk_size"] is None


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_dpo_normalized_and_logprob_chunk_default_present():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "dpo"})
    assert config["trainer_type"] == "dpo"
    assert config["trainer_prefix"] == "DPO"
    assert config["logprob_vocab_chunk_size"] is None


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_nash_md_alias_resolves_classes():
    pytest.importorskip("eformer.paths")
    from easydel.infra.elarge.types import get_trainer_class, get_training_arguments_class

    assert get_trainer_class("nash_md") is get_trainer_class("nash-md")
    assert get_training_arguments_class("nash_md") is get_training_arguments_class("nash-md")


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_ppo_resolves_classes():
    pytest.importorskip("eformer.paths")
    from easydel.infra.elarge.types import get_trainer_class, get_training_arguments_class

    assert get_trainer_class("ppo").__name__ == "PPOTrainer"
    assert get_training_arguments_class("ppo").__name__ == "PPOConfig"


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_on_policy_distillation_normalized_and_defaults_present():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "on_policy_distillation"})
    assert config["trainer_type"] == "on_policy_distillation"
    assert config["trainer_prefix"] == "OnPolicyDistillation"
    assert config["alpha"] == 0.9
    assert config["temperature"] == 2.0
    assert config["logits_chunk_size"] is None
    assert config["max_prompt_length"] == 512
    assert config["max_completion_length"] == 256


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_rlvr_normalized_and_disable_defaults_present():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "rlvr"})
    assert config["trainer_type"] == "rlvr"
    assert config["trainer_prefix"] == "RLVR"
    assert config["length_penalty_target"] is None
    assert config["reward_clip_range"] is None


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_on_policy_distillation_resolves_classes():
    pytest.importorskip("eformer.paths")
    from easydel.infra.elarge.types import get_trainer_class, get_training_arguments_class

    assert get_trainer_class("on_policy_distillation").__name__ == "OnPolicyDistillationTrainer"
    assert get_training_arguments_class("on_policy_distillation").__name__ == "OnPolicyDistillationConfig"


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_seq_kd_normalized_and_defaults_present():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "seq_kd"})
    assert config["trainer_type"] == "seq_kd"
    assert config["trainer_prefix"] == "SeqKD"
    assert config["max_prompt_length"] == 512
    assert config["max_completion_length"] == 256


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_seq_kd_resolves_classes():
    pytest.importorskip("eformer.paths")
    from easydel.infra.elarge.types import get_trainer_class, get_training_arguments_class

    assert get_trainer_class("seq_kd").__name__ == "SeqKDTrainer"
    assert get_training_arguments_class("seq_kd").__name__ == "SeqKDConfig"


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_sparse_distillation_normalized_and_defaults_present():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "sparse_distillation"})
    assert config["trainer_type"] == "sparse_distillation"
    assert config["trainer_prefix"] == "SparseDistillation"
    assert config["alpha"] == 0.9
    assert config["temperature"] == 2.0
    assert config["top_k_teacher"] == 20
    assert config["max_prompt_length"] == 512
    assert config["max_completion_length"] == 256


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_sparse_distillation_resolves_classes():
    pytest.importorskip("eformer.paths")
    from easydel.infra.elarge.types import get_trainer_class, get_training_arguments_class

    assert get_trainer_class("sparse_distillation").__name__ == "SparseDistillationTrainer"
    assert get_training_arguments_class("sparse_distillation").__name__ == "SparseDistillationConfig"
