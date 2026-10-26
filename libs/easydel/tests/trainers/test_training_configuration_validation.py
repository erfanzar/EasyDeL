"""Regression tests for trainer configuration behavior, with no array computation."""

from dataclasses import dataclass, field

import pytest
from easydel.trainers.training_configurations import TrainingArguments
from easydel.trainers.training_utils import make_default_tensor_straight_through

pytestmark = pytest.mark.cpu_ok


@pytest.fixture(autouse=True)
def no_distributed_initialization(monkeypatch):
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)


def arguments(**kwargs):
    return TrainingArguments(use_wandb=False, report_metrics=False, **kwargs)


@pytest.mark.parametrize(
    ("duration", "seconds"),
    [
        ("1d", 86400),
        ("2h30m", 9000),
        ("2h30min", 9000),
        ("1 day 2 hours 3 minutes 4 seconds", 93784),
        (" 90MIN ", 5400),
        ("0s", 0),
        (None, None),
    ],
)
def test_duration_seconds(duration, seconds):
    assert arguments(training_time_limit=duration).training_time_seconds == seconds


@pytest.mark.parametrize("duration", ["", "1", "1h junk", "2h30", "-2h", "1.5h", "1d!", 3])
def test_invalid_duration_is_rejected_during_construction(duration):
    with pytest.raises(ValueError, match="Invalid time format"):
        arguments(training_time_limit=duration)


@pytest.mark.parametrize(("mode", "bits"), [("nf4", 8), ("mxfp8", 4), ("affine", 1), ("affine", 0)])
def test_config_and_ste_reject_incompatible_bits_consistently(mode, bits):
    with pytest.raises(ValueError) as config_error:
        arguments(quantization_mode=mode, quantization_bits=bits)
    with pytest.raises(ValueError) as ste_error:
        make_default_tensor_straight_through(mode, quantization_bits=bits)
    assert str(config_error.value) == str(ste_error.value)


def test_missing_save_directory_disables_tensorboard(tmp_path):
    args = arguments(save_directory=None, model_name="model")
    assert args.get_tensorboard() is None
    assert str(args._get_save_directory_milestone(1)) == "/dev/null"
    with pytest.raises(ValueError, match="save_directory and model_name"):
        args.get_streaming_checkpointer()
    assert list(tmp_path.iterdir()) == []


def test_missing_model_name_disables_tensorboard(tmp_path):
    args = arguments(save_directory=str(tmp_path), model_name=None)
    assert args.get_tensorboard() is None
    assert list(tmp_path.iterdir()) == []


def test_non_none_subclass_max_length_default_honors_deprecated_alias():
    @dataclass
    class CustomArguments(TrainingArguments):
        max_length: int | None = field(default=512)

    with pytest.warns(FutureWarning, match="max_sequence_length"):
        args = CustomArguments(max_sequence_length=128, use_wandb=False, report_metrics=False)
    assert args.max_length == 128
