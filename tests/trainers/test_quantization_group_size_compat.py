import pytest
from jax import numpy as jnp

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


def test_training_arguments_accepts_quantization_bits():
    cfg = TrainingArguments(**_base_training_args(), quantization_mode="affine", quantization_bits=4)
    assert cfg.quantization_bits == 4


def test_training_arguments_rejects_invalid_affine_quantization_bits():
    with pytest.raises(ValueError, match="quantization_bits"):
        TrainingArguments(**_base_training_args(), quantization_mode="affine", quantization_bits=9)


def test_training_arguments_rejects_invalid_fixed_mode_quantization_bits():
    with pytest.raises(ValueError, match="quantization_bits"):
        TrainingArguments(**_base_training_args(), quantization_mode="nf4", quantization_bits=8)


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
            quantization_bits=4,
            tensor_straight_through=None,
            straight_through_emulator=None,
            quantization_block=64,
        )
    assert emulator is not None


def test_resolve_straight_through_emulator_legacy_alias_without_group_size_arg():
    with pytest.warns(FutureWarning, match="quantization_block"):
        emulator = resolve_straight_through_emulator(
            quantization_mode="nf4",
            quantization_bits=4,
            tensor_straight_through=None,
            straight_through_emulator=None,
            quantization_block=64,
        )
    assert emulator is not None


def test_resolve_straight_through_emulator_handles_1d_tensors():
    emulator = resolve_straight_through_emulator(
        quantization_mode="nf4",
        quantization_group_size=64,
        quantization_bits=4,
        tensor_straight_through=None,
        straight_through_emulator=None,
    )
    assert emulator is not None
    graphstate = {"norm": jnp.linspace(-1.0, 1.0, 77, dtype=jnp.float32)}
    transformed = emulator(graphstate)
    assert transformed["norm"].shape == graphstate["norm"].shape
    assert transformed["norm"].dtype == graphstate["norm"].dtype
