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

"""Pure-Python checkpoint-resume failure contract (no model computation)."""

from types import SimpleNamespace

import pytest
from easydel.trainers.base_trainer import BaseTrainer
from easydel.trainers.trainer.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


class _FreshPathReached(Exception):
    """Stop the constructor before any model computation on the fresh path."""


def _inputs():
    model = SimpleNamespace(
        _model_type="test-model",
        dtype="float32",
        param_dtype="float32",
        precision=None,
        model_task="causal-lm",
        config=SimpleNamespace(
            partition_axis=None,
            sharding_axis_names=(),
            sharding_axis_dims=(),
            sharding_dcn_axis_dims=(),
        ),
    )
    model_state = SimpleNamespace(model=model)
    arguments = SimpleNamespace(
        model_name=None,
        resume_if_possible=True,
        _get_save_directory=lambda create=False: "/unused/weights",
        get_tx_template=lambda: None,
    )
    return arguments, model_state


@pytest.mark.parametrize(
    ("failure_stage", "error"),
    [
        ("load", "checkpoint data unavailable"),
        ("auxiliary", "auxiliary checkpoint invalid"),
    ],
)
def test_discovered_checkpoint_failure_aborts_initialization(monkeypatch, failure_stage, error):
    checkpoint = "/unused/weights/checkpoint-7"
    arguments, model_state = _inputs()
    calls = []
    monkeypatch.setattr("spectrax.serialization.find_latest_checkpoint", lambda directory: checkpoint)

    def load_state(**kwargs):
        calls.append(("load", kwargs["load_directory"]))
        if failure_stage == "load":
            raise OSError("checkpoint data unavailable")
        return SimpleNamespace(step=7)

    def load_auxiliary(self, path):
        calls.append(("auxiliary", path))
        if failure_stage == "auxiliary":
            raise ValueError("auxiliary checkpoint invalid")

    def remove_loaded(self, path):
        calls.append(("remove", path))
        if failure_stage == "remove":
            raise OSError("checkpoint cleanup failed")

    monkeypatch.setattr("easydel.trainers.base_trainer.EasyDeLState.load_state", load_state)
    monkeypatch.setattr("easydel.trainers.base_trainer.jax.device_get", lambda step: step)
    monkeypatch.setattr(Trainer, "_load_auxiliary_states", load_auxiliary)
    monkeypatch.setattr(Trainer, "_maybe_remove_loaded_checkpoint", remove_loaded)

    def unexpected_fresh_start(self, state):
        raise AssertionError("found checkpoint must not restart training")

    monkeypatch.setattr(Trainer, "_in_training_mode", unexpected_fresh_start)

    with pytest.raises(RuntimeError, match="Failed to resume from checkpoint") as caught:
        BaseTrainer.__init__(object.__new__(Trainer), arguments=arguments, model_state=model_state)

    assert checkpoint in str(caught.value)
    assert error in str(caught.value)
    assert "resume_if_possible=False" in str(caught.value)
    assert str(caught.value.__cause__) == error
    assert calls[0] == ("load", checkpoint)
    assert len(calls) == {"load": 1, "auxiliary": 2}[failure_stage]


def test_cleanup_failure_after_successful_restore_does_not_abort(monkeypatch):
    """Failing to delete the old checkpoint must not discard the state that was just restored."""
    checkpoint = "/unused/weights/checkpoint-7"
    arguments, model_state = _inputs()
    resumed = SimpleNamespace(step=7)
    monkeypatch.setattr("spectrax.serialization.find_latest_checkpoint", lambda directory: checkpoint)
    monkeypatch.setattr("easydel.trainers.base_trainer.EasyDeLState.load_state", lambda **kwargs: resumed)
    monkeypatch.setattr("easydel.trainers.base_trainer.jax.device_get", lambda step: step)
    monkeypatch.setattr(Trainer, "_load_auxiliary_states", lambda self, path: None)

    def remove_loaded(self, path):
        raise OSError("checkpoint cleanup failed")

    monkeypatch.setattr(Trainer, "_maybe_remove_loaded_checkpoint", remove_loaded)
    seen = []

    def stop_at_training_mode(self, state):
        seen.append((state, self._resumed_from_checkpoint))
        raise _FreshPathReached()

    monkeypatch.setattr(Trainer, "_in_training_mode", stop_at_training_mode)
    with pytest.raises(_FreshPathReached):
        BaseTrainer.__init__(object.__new__(Trainer), arguments=arguments, model_state=model_state)
    assert seen == [(resumed, True)]


def test_no_discovered_checkpoint_continues_fresh(monkeypatch):
    arguments, model_state = _inputs()
    monkeypatch.setattr("spectrax.serialization.find_latest_checkpoint", lambda directory: None)

    def unexpected_load(**kwargs):
        raise AssertionError("load must not be called")

    monkeypatch.setattr("easydel.trainers.base_trainer.EasyDeLState.load_state", unexpected_load)
    seen = []

    def stop_at_fresh_path(self, state):
        seen.append(state)
        raise _FreshPathReached()

    monkeypatch.setattr(Trainer, "_in_training_mode", stop_at_fresh_path)
    with pytest.raises(_FreshPathReached):
        BaseTrainer.__init__(object.__new__(Trainer), arguments=arguments, model_state=model_state)
    assert seen == [model_state]
