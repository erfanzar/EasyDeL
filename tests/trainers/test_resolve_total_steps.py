import pytest

from easydel.trainers.training_utils import resolve_total_steps


def test_forced_training_steps_are_optimizer_steps():
    assert (
        resolve_total_steps(
            forced_steps=100,
            total_data_len=None,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=4,
            is_train=True,
        )
        == 100
    )


def test_computed_training_steps_apply_gradient_accumulation():
    assert (
        resolve_total_steps(
            forced_steps=None,
            total_data_len=3200,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=4,
            is_train=True,
        )
        == 25
    )


def test_eval_steps_do_not_apply_gradient_accumulation():
    assert (
        resolve_total_steps(
            forced_steps=None,
            total_data_len=3200,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=4,
            is_train=False,
        )
        == 100
    )


def test_missing_total_data_len_raises():
    with pytest.raises(ValueError):
        resolve_total_steps(
            forced_steps=None,
            total_data_len=None,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=1,
            is_train=True,
        )

