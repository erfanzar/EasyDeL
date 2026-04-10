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

import inspect
from dataclasses import fields

import pytest

from easydel.infra.elarge.model import eLargeModel
from easydel.infra.elarge.types import (
    BASE_TRAINER_DEFAULTS,
    TRAINER_SPECIFIC_DEFAULTS,
    BaseTrainerCfg,
    get_training_arguments_class,
    normalize_trainer_config,
)
from easydel.trainers.training_configurations import TrainingArguments


def test_base_trainer_cfg_covers_training_arguments_fields():
    training_argument_fields = {
        field_obj.name for field_obj in fields(TrainingArguments) if not field_obj.name.startswith("_")
    }
    config_keys = set(BaseTrainerCfg.__required_keys__) | set(BaseTrainerCfg.__optional_keys__)
    missing_keys = sorted(training_argument_fields - config_keys)
    assert missing_keys == [], f"BaseTrainerCfg is missing TrainingArguments fields: {missing_keys}"


def test_normalized_configs_construct_every_training_arguments_class():
    trainer_types = sorted(set(TRAINER_SPECIFIC_DEFAULTS) | {"base"})
    for trainer_type in trainer_types:
        config = normalize_trainer_config(
            {
                "trainer_type": trainer_type,
                "quantization_mode": "nf4",
                "quantization_group_size": 64,
                "quantization_bits": 4,
                "lmhead_chunksize": 128,
                "esurge_use_tqdm": True,
                "esurge_enable_prefix_caching": False,
                "esurge_data_parallelism_axis": "dp",
                "esurge_max_num_seq_buckets": [1, 2, 4, 8],
            }
        )
        args_cls = get_training_arguments_class(trainer_type)
        payload = {key: value for key, value in config.items() if key != "trainer_type"}

        signature = inspect.signature(args_cls.__init__)
        valid_parameters = set(signature.parameters) - {"self"}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            unexpected = sorted(set(payload) - valid_parameters)
            assert unexpected == [], f"{trainer_type} has unsupported eLarge config keys: {unexpected}"

        args = args_cls(**payload)
        assert args.quantization_mode == "nf4"
        assert args.quantization_group_size == 64
        assert args.quantization_bits == 4
        assert args.lmhead_chunksize == 128
        assert args.esurge_use_tqdm is True
        assert args.esurge_enable_prefix_caching is False
        assert args.esurge_data_parallelism_axis == "dp"
        assert args.esurge_max_num_seq_buckets == [1, 2, 4, 8]


def test_base_defaults_do_not_include_unknown_training_arguments_keys():
    args_signature = inspect.signature(TrainingArguments.__init__)
    valid_parameters = set(args_signature.parameters) - {"self", "max_sequence_length", "quantization_block"}
    unknown_defaults = sorted(set(BASE_TRAINER_DEFAULTS) - valid_parameters)
    assert unknown_defaults == [], f"BASE_TRAINER_DEFAULTS includes unknown keys: {unknown_defaults}"


def test_training_arguments_normalize_single_benchmark_config():
    args = TrainingArguments(
        model_name="dummy",
        total_batch_size=1,
        benchmarks={"name": "code", "tasks": "humaneval", "enable_thinking": True},
        benchmark_interval=8,
    )

    assert args.benchmark_interval == 8
    assert isinstance(args.benchmarks, list)
    assert args.benchmarks == [{"name": "code", "tasks": "humaneval", "enable_thinking": True}]


def test_training_arguments_reject_removed_benchmark_task_alias():
    with pytest.raises(TypeError, match="must use `tasks`"):
        TrainingArguments(
            model_name="dummy",
            total_batch_size=1,
            benchmarks={"name": "code", "task": "humaneval", "enable_thinking": True},
            benchmark_interval=8,
        )


def test_training_arguments_preserve_none_step_start_point_for_auto_resume():
    args = TrainingArguments(
        model_name="dummy",
        total_batch_size=1,
    )

    assert args.step_start_point is None
    assert args.force_step_start_point is False


def test_elarge_build_training_arguments_preserves_step_start_point():
    elm = eLargeModel({"model": {"name_or_path": "dummy-model"}})
    elm.set_trainer(
        "sft",
        total_batch_size=1,
        step_start_point=123,
        force_step_start_point=True,
        resume_if_possible=False,
    )

    args = elm.build_training_arguments()

    assert args.step_start_point == 123
    assert args.force_step_start_point is True
    assert args.resume_if_possible is False


def test_normalize_trainer_config_defaults_step_start_point_to_none():
    config = normalize_trainer_config(
        {
            "trainer_type": "sft",
            "total_batch_size": 1,
        }
    )

    assert "step_start_point" in config
    assert config["step_start_point"] is None
    assert "force_step_start_point" in config
    assert config["force_step_start_point"] is False


def test_normalize_trainer_config_defaults_lmhead_chunksize_to_none():
    config = normalize_trainer_config(
        {
            "trainer_type": "sft",
            "total_batch_size": 1,
        }
    )

    assert "lmhead_chunksize" in config
    assert config["lmhead_chunksize"] is None


def test_normalize_trainer_config_defaults_dpo_logprob_vocab_chunk_size():
    config = normalize_trainer_config(
        {
            "trainer_type": "dpo",
            "total_batch_size": 1,
        }
    )

    assert "logprob_vocab_chunk_size" in config
    assert config["logprob_vocab_chunk_size"] is None


@pytest.mark.parametrize(
    "trainer_type",
    ["cpo", "bco", "kto", "orpo", "grpo", "ppo", "sdpo", "xpo", "nash-md"],
)
def test_normalize_trainer_config_defaults_chunked_logprob_trainers(trainer_type: str):
    config = normalize_trainer_config(
        {
            "trainer_type": trainer_type,
            "total_batch_size": 1,
        }
    )

    assert "logprob_vocab_chunk_size" in config
    assert config["logprob_vocab_chunk_size"] is None


def test_normalize_trainer_config_defaults_grpo_chunking_knobs_to_none():
    config = normalize_trainer_config(
        {
            "trainer_type": "grpo",
            "total_batch_size": 1,
        }
    )

    assert config["ref_logps_chunk_size"] is None
    assert config["completion_chunk_size"] is None
    assert config["max_loss_completion_tokens"] is None


def test_normalize_trainer_config_defaults_distillation_chunk_size_to_none():
    config = normalize_trainer_config(
        {
            "trainer_type": "distillation",
            "total_batch_size": 1,
        }
    )

    assert config["logits_chunk_size"] is None


def test_normalize_trainer_config_defaults_ppo_entropy_coef_to_none():
    config = normalize_trainer_config(
        {
            "trainer_type": "ppo",
            "total_batch_size": 1,
        }
    )

    assert config["entropy_coef"] is None


def test_normalize_trainer_config_defaults_rlvr_disable_knobs_to_none():
    config = normalize_trainer_config(
        {
            "trainer_type": "rlvr",
            "total_batch_size": 1,
        }
    )

    assert config["length_penalty_target"] is None
    assert config["reward_clip_range"] is None


@pytest.mark.parametrize(
    ("trainer_type", "overrides", "expected_none_fields"),
    [
        (
            "grpo",
            {
                "ref_logps_chunk_size": 0,
                "completion_chunk_size": 0,
                "max_loss_completion_tokens": 0,
            },
            ("ref_logps_chunk_size", "completion_chunk_size", "max_loss_completion_tokens"),
        ),
        (
            "distillation",
            {
                "hidden_state_loss_weight": 0.0,
                "attention_loss_weight": 0.0,
                "logits_chunk_size": 0,
            },
            ("hidden_state_loss_weight", "attention_loss_weight", "logits_chunk_size"),
        ),
        (
            "ppo",
            {
                "entropy_coef": 0.0,
            },
            ("entropy_coef",),
        ),
        (
            "rlvr",
            {
                "length_penalty_target": 0,
                "reward_clip_range": 0.0,
            },
            ("length_penalty_target", "reward_clip_range"),
        ),
        (
            "gkd",
            {
                "lmbda": 0.0,
            },
            ("lmbda",),
        ),
    ],
)
def test_zero_disabled_trainer_knobs_normalize_to_none(
    trainer_type: str,
    overrides: dict[str, float | int],
    expected_none_fields: tuple[str, ...],
):
    args_cls = get_training_arguments_class(trainer_type)
    args = args_cls(
        model_name="dummy",
        total_batch_size=1,
        use_wandb=False,
        **overrides,
    )

    for field_name in expected_none_fields:
        assert getattr(args, field_name) is None


def test_training_arguments_default_tpu_preemption_checkpoint_settings():
    args = TrainingArguments(
        model_name="dummy",
        total_batch_size=1,
    )

    assert args.save_tpu_preemption_checkpoints is True


def test_training_arguments_default_training_generation_wandb_logging_enabled():
    args = TrainingArguments(
        model_name="dummy",
        total_batch_size=1,
    )

    assert args.log_training_generations_to_wandb is True


def test_ppo_penalties_inherit_into_generation_preview_fields():
    args_cls = get_training_arguments_class("ppo")
    args = args_cls(
        model_name="dummy",
        total_batch_size=1,
        max_length=256,
        max_prompt_length=128,
        presence_penalty=0.4,
        frequency_penalty=0.2,
        repetition_penalty=1.3,
        use_wandb=False,
    )

    assert args.generation_presence_penalty == pytest.approx(0.4)
    assert args.generation_frequency_penalty == pytest.approx(0.2)
    assert args.generation_repetition_penalty == pytest.approx(1.3)
