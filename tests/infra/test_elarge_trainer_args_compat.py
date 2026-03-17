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


def test_elarge_build_training_arguments_preserves_step_start_point():
    elm = eLargeModel({"model": {"name_or_path": "dummy-model"}})
    elm.set_trainer(
        "sft",
        total_batch_size=1,
        step_start_point=123,
        resume_if_possible=False,
    )

    args = elm.build_training_arguments()

    assert args.step_start_point == 123
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


def test_training_arguments_default_tpu_preemption_checkpoint_settings():
    args = TrainingArguments(
        model_name="dummy",
        total_batch_size=1,
    )

    assert args.save_tpu_preemption_checkpoints is True
