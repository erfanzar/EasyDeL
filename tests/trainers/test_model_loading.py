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

from types import SimpleNamespace

import pytest

from easydel.infra.loss_utils import LossMetrics
from easydel.trainers import model_loading
from easydel.trainers.binary_classifier_optimization_trainer.bco_trainer import BCOTrainer
from easydel.trainers.contrastive_preference_optimization_trainer.cpo_trainer import CPOTrainer
from easydel.trainers.direct_preference_optimization_trainer.dpo_trainer import DPOTrainer
from easydel.trainers.distillation_trainer.distillation_config import DistillationConfig
from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer
from easydel.trainers.generalized_knowledge_distillation_trainer.gkd_trainer import GKDTrainer
from easydel.trainers.kto_trainer.kto_trainer import KTOTrainer
from easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_trainer import ORPOTrainer
from easydel.trainers.proximal_policy_optimization_trainer.ppo_config import PPOConfig
from easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer import PPOTrainer


def test_string_model_ids_are_rejected_at_trainer_boundaries():
    with pytest.raises(ValueError, match="do not load policy model from string ids"):
        model_loading.reject_string_model_id("repo/model", role="policy model")
    with pytest.raises(ValueError, match="Load 'repo/model'"):
        model_loading.reject_string_model_id("repo/model")


def test_bco_string_model_resolvers_reject_string_loading():
    with pytest.raises(ValueError, match="do not load policy model"):
        BCOTrainer._resolve_policy_model("repo/policy")
    with pytest.raises(ValueError, match="do not load reference model"):
        BCOTrainer._resolve_reference_model("repo/reference")


def test_orpo_string_model_resolver_rejects_string_loading():
    with pytest.raises(ValueError, match="do not load policy model"):
        ORPOTrainer._resolve_policy_model("repo/policy")


def test_cpo_string_model_resolver_rejects_string_loading():
    with pytest.raises(ValueError, match="do not load policy model"):
        CPOTrainer._resolve_policy_model("repo/policy")


def test_cpo_policy_model_is_required():
    with pytest.raises(ValueError, match="policy model"):
        CPOTrainer._resolve_policy_model(None)


def test_kto_string_model_resolvers_reject_string_loading():
    with pytest.raises(ValueError, match="do not load policy model"):
        KTOTrainer._resolve_policy_model("repo/policy")
    with pytest.raises(ValueError, match="do not load reference model"):
        KTOTrainer._resolve_reference_model("repo/reference")


def test_gkd_teacher_string_model_rejects_string_loading():
    with pytest.raises(ValueError, match="do not load teacher model"):
        GKDTrainer._resolve_teacher_model(teacher_model="repo/teacher")


def test_bco_source_has_reference_logps_detects_materialized_column():
    class FakeSource:
        def __init__(self, sample):
            self.sample = sample
            self.shard_names = ["s0"]

        def open_shard(self, shard_name):
            del shard_name
            yield self.sample

    assert BCOTrainer._source_has_reference_logps(FakeSource({"reference_logps": 0.1}))
    assert not BCOTrainer._source_has_reference_logps(FakeSource({"completion_logps": 0.1}))


def test_ppo_string_policy_and_reward_loading_are_rejected():
    with pytest.raises(ValueError, match="do not load policy model"):
        PPOTrainer._resolve_policy_model("repo/policy", PPOConfig())
    with pytest.raises(ValueError, match="model` must be provided"):
        PPOTrainer._resolve_policy_model(None, PPOConfig())
    with pytest.raises(ValueError, match="do not load reward model"):
        PPOTrainer._resolve_reward_func_model("repo/reward")


def test_ppo_num_ppo_epochs_reuses_one_rollout_batch(monkeypatch):
    state = SimpleNamespace(step=0)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.arguments = SimpleNamespace(num_ppo_epochs=3)
    trainer.pruning_module = None
    trainer._train_shared_fn_extra_args = ()
    trainer._train_shared_fn_static_args = ()
    trainer._runtime_trace = lambda *args, **kwargs: None
    trainer._runtime_batch_summary = lambda batch: batch
    trainer._is_memory_oom_exception = lambda exc: False

    preprocess_calls = []
    compiled_calls = []

    def preprocess(*, state, batch, is_train):
        preprocess_calls.append((state.step, batch, is_train))
        return {"input_ids": "rolled-out"}, {"rollout_score": 1.0}

    def compiled_step(state, batch):
        compiled_calls.append((state.step, batch))
        return SimpleNamespace(step=state.step + 1), LossMetrics(
            loss=float(state.step),
            accuracy=1.0,
            other_metrics={"ppo_epoch": state.step},
        )

    trainer._preprocess_batch_input = preprocess
    trainer.sharded_training_step_function = compiled_step
    monkeypatch.setattr(
        "easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer.jax.block_until_ready",
        lambda value: value,
    )
    monkeypatch.setattr(
        "easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer.jax.device_get",
        lambda value: value,
    )

    final_state, metrics, error = trainer._execute_train_step(state=state, batch={"prompt": "hi"})

    assert error is None
    assert final_state.step == 3
    assert preprocess_calls == [(0, {"prompt": "hi"}, True)]
    assert compiled_calls == [
        (0, {"input_ids": "rolled-out"}),
        (1, {"input_ids": "rolled-out"}),
        (2, {"input_ids": "rolled-out"}),
    ]
    assert metrics.other_metrics == {"rollout_score": 1.0, "ppo_epoch": 2}


def test_ppo_num_ppo_epochs_must_be_positive():
    with pytest.raises(ValueError, match="num_ppo_epochs"):
        PPOConfig(num_ppo_epochs=0)


def test_ppo_trl_aliases_are_wired_or_guarded():
    cfg = PPOConfig(response_length=53, stop_token_id=2)
    assert cfg.max_completion_length == 53
    assert cfg.max_length == cfg.max_prompt_length + 53
    assert cfg.generation_extra_kwargs["eos_token_id"] == 2
    assert cfg.batch_size == cfg.total_batch_size
    assert cfg.local_batch_size == cfg.total_batch_size
    assert cfg.mini_batch_size == cfg.total_batch_size

    with pytest.raises(ValueError, match="mutually exclusive"):
        PPOConfig(stop_token="eos", stop_token_id=2)
    with pytest.raises(ValueError, match="num_mini_batches"):
        PPOConfig(num_mini_batches=2)
    with pytest.raises(ValueError, match="local_rollout_forward_batch_size"):
        PPOConfig(local_rollout_forward_batch_size=64)
    with pytest.raises(ValueError, match="push_to_hub"):
        PPOConfig(push_to_hub=True)
    with pytest.raises(ValueError, match="ds3_gather_for_generation"):
        PPOConfig(ds3_gather_for_generation=False)
    with pytest.raises(ValueError, match="world_size"):
        PPOConfig(world_size=2)
    with pytest.raises(ValueError, match="local_batch_size"):
        PPOConfig(local_batch_size=999)


def test_dpo_reference_model_string_resolver_rejects_string_loading():
    with pytest.raises(ValueError, match="do not load reference model"):
        DPOTrainer._resolve_reference_model("repo/reference")


def test_distillation_string_model_resolvers_reject_string_loading():
    with pytest.raises(ValueError, match="do not load student model"):
        DistillationTrainer._resolve_student_model("repo/student")
    with pytest.raises(ValueError, match="do not load teacher model"):
        DistillationTrainer._resolve_teacher_model(
            teacher_model="repo/teacher",
            teacher_model_revision=None,
        )


def test_distillation_requires_initialized_teacher():
    with pytest.raises(ValueError, match="student_model"):
        DistillationTrainer._resolve_student_model(None)
    with pytest.raises(ValueError, match="teacher_model"):
        DistillationTrainer._resolve_teacher_model(
            teacher_model=None,
            teacher_model_revision=None,
        )
    with pytest.raises(ValueError, match="teacher_model_revision"):
        DistillationTrainer._resolve_teacher_model(
            teacher_model=None,
            teacher_model_revision="teacher",
        )


def test_distillation_trl_compat_fields_are_guarded():
    unsupported_cases = [
        ("lmbda", {"lmbda": 0.5}),
        ("max_prompt_length", {"max_prompt_length": 128}),
        ("max_completion_length", {"max_completion_length": 64}),
        ("num_generations", {"num_generations": 2}),
        ("generation_batch_size", {"generation_batch_size": 4}),
        ("On-policy sampling", {"top_p": 0.9}),
    ]

    for expected_message, kwargs in unsupported_cases:
        with pytest.raises(ValueError, match=expected_message):
            DistillationConfig(**kwargs)

    with pytest.raises(ValueError, match="teacher_model_revision"):
        DistillationConfig(teacher_model_revision="main")
    DistillationConfig(beta=0.1, loss_top_k=8, loss_add_tail=True, disable_dropout=True)
    DistillationConfig(wandb_project="proj", log_completions=True)
    with pytest.raises(ValueError, match="beta"):
        DistillationConfig(beta=1.1)
    with pytest.raises(ValueError, match="loss_top_k"):
        DistillationConfig(loss_top_k=-1)
    with pytest.raises(ValueError, match="loss_add_tail"):
        DistillationConfig(loss_add_tail=True)
    with pytest.raises(ValueError, match="log_completions_steps"):
        DistillationConfig(log_completions_steps=0)
    with pytest.raises(ValueError, match="num_completions_to_print"):
        DistillationConfig(num_completions_to_print=0)
