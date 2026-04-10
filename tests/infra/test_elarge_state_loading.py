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

from types import SimpleNamespace

from easydel.infra.elarge.builders import to_load_state_kwargs
from easydel.infra.elarge.model import eLargeModel


def test_to_load_state_kwargs_preserves_state_loader_defaults():
    cfg = {"model": {"name_or_path": "/tmp/checkpoint"}}

    kwargs = to_load_state_kwargs(cfg)

    assert kwargs["load_directory"] == "/tmp/checkpoint"
    assert kwargs["device"] == "cpu"
    assert kwargs["auto_shard_model"] is True
    assert kwargs["verbose"] is True


def test_build_state_loads_once_and_reuses_cached_model(monkeypatch):
    elm = eLargeModel({"model": {"name_or_path": "/tmp/checkpoint"}})

    fake_model = object()
    fake_state = SimpleNamespace(model=fake_model)
    calls: list[dict[str, object]] = []

    def _fake_load_state(cls, **kwargs):
        calls.append(kwargs)
        return fake_state

    monkeypatch.setattr(
        "easydel.infra.base_state.EasyDeLState.load_state",
        classmethod(_fake_load_state),
    )

    loaded = elm.build_state()
    loaded_again = elm.build_state()

    assert loaded is fake_state
    assert loaded_again is fake_state
    assert elm.build_model() is fake_model
    assert len(calls) == 1
    assert calls[0]["load_directory"] == "/tmp/checkpoint"


def test_build_state_force_rebuild_reloads_checkpoint(monkeypatch):
    elm = eLargeModel({"model": {"name_or_path": "/tmp/checkpoint"}})

    states = [SimpleNamespace(model=object()), SimpleNamespace(model=object())]
    calls = {"count": 0}

    def _fake_load_state(cls, **kwargs):
        del kwargs
        state = states[calls["count"]]
        calls["count"] += 1
        return state

    monkeypatch.setattr(
        "easydel.infra.base_state.EasyDeLState.load_state",
        classmethod(_fake_load_state),
    )

    first = elm.build_state()
    second = elm.build_state(force_rebuild=True)

    assert first is states[0]
    assert second is states[1]
    assert elm.build_model() is states[1].model
    assert calls["count"] == 2


def test_build_trainer_prefers_loaded_state(monkeypatch):
    captured = {}

    class FakeState:
        pass

    class _CaptureTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("easydel.infra.elarge.model.EasyDeLState", FakeState)

    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "trainer": {"trainer_type": "base"},
    }
    elm._model = object()
    elm._state = FakeState()
    elm._tokenizer = object()
    elm.build_training_arguments = lambda *args, **kwargs: "args"

    elm.build_trainer(trainer_class=_CaptureTrainer)

    assert captured["model_state"] is elm._state
    assert "model" not in captured
