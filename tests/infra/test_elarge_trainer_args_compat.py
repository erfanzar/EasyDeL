from __future__ import annotations

import inspect
from dataclasses import fields

from easydel.infra.elarge_model.trainer_types import (
    BASE_TRAINER_DEFAULTS,
    TRAINER_SPECIFIC_DEFAULTS,
    BaseTrainerCfg,
    get_training_arguments_class,
    normalize_trainer_config,
)
from easydel.trainers.training_configurations import TrainingArguments


def test_base_trainer_cfg_covers_training_arguments_fields():
    training_argument_fields = {field_obj.name for field_obj in fields(TrainingArguments) if not field_obj.name.startswith("_")}
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


def test_base_defaults_do_not_include_unknown_training_arguments_keys():
    args_signature = inspect.signature(TrainingArguments.__init__)
    valid_parameters = set(args_signature.parameters) - {"self", "max_sequence_length", "quantization_block"}
    unknown_defaults = sorted(set(BASE_TRAINER_DEFAULTS) - valid_parameters)
    assert unknown_defaults == [], f"BASE_TRAINER_DEFAULTS includes unknown keys: {unknown_defaults}"
