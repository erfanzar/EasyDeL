import pytest

from easydel.trainers.training_configurations import TrainingArguments
from easydel.trainers.training_utils import resolve_straight_through_emulator


def _base_training_args() -> dict:
    return {
        "model_name": "dummy-model",
        "save_directory": "tmp-easydel-tests",
        "total_batch_size": 1,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
    }


def test_training_arguments_accepts_quantization_group_size():
    cfg = TrainingArguments(**_base_training_args(), quantization_group_size=64)
    assert cfg.quantization_group_size == 64


def test_training_arguments_legacy_quantization_block_maps_with_warning():
    with pytest.warns(FutureWarning, match="quantization_block"):
        cfg = TrainingArguments(**_base_training_args(), quantization_block=32)
    assert cfg.quantization_group_size == 32
    assert cfg.quantization_block == 32


def test_training_arguments_from_dict_maps_legacy_quantization_block():
    cfg = TrainingArguments.from_dict({**_base_training_args(), "quantization_block": 16})
    assert cfg.quantization_group_size == 16


def test_training_arguments_load_from_json_maps_legacy_quantization_block():
    cfg = TrainingArguments.load_from_json({**_base_training_args(), "quantization_block": 8})
    assert cfg.quantization_group_size == 8


def test_resolve_straight_through_emulator_supports_legacy_alias_with_warning():
    with pytest.warns(FutureWarning, match="quantization_block"):
        emulator = resolve_straight_through_emulator(
            quantization_mode="nf4",
            quantization_group_size=None,
            tensor_straight_through=None,
            straight_through_emulator=None,
            quantization_block=64,
        )
    assert emulator is not None


def test_resolve_straight_through_emulator_legacy_alias_without_group_size_arg():
    with pytest.warns(FutureWarning, match="quantization_block"):
        emulator = resolve_straight_through_emulator(
            quantization_mode="nf4",
            tensor_straight_through=None,
            straight_through_emulator=None,
            quantization_block=64,
        )
    assert emulator is not None
