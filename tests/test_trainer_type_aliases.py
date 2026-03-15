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
    assert config["max_prompt_length"] == 512
    assert config["max_completion_length"] == 256


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
