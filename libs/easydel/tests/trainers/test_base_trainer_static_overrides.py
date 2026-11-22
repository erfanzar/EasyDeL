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

"""Issue #291: runtime config overrides are applied through the trainer instance without a TypeError."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from easydel.trainers.base_trainer import BaseTrainer

pytestmark = pytest.mark.cpu_ok


def _bare_trainer() -> BaseTrainer:
    """An uninitialised trainer instance, so the method is resolved through instance attribute lookup."""
    probe = type("_ProbeTrainer", (BaseTrainer,), {})
    probe.__abstractmethods__ = frozenset()
    return object.__new__(probe)


def test_override_applies_when_called_through_an_instance():
    trainer = _bare_trainer()
    config = SimpleNamespace(lmhead_chunksize=None)
    state = SimpleNamespace(model=SimpleNamespace(config=config))

    trainer._apply_runtime_model_config_overrides_to_state(state, SimpleNamespace(lmhead_chunksize=512))

    assert config.lmhead_chunksize == 512


def test_override_is_noop_without_state_or_value():
    trainer = _bare_trainer()
    config = SimpleNamespace(lmhead_chunksize=7)
    state = SimpleNamespace(model=SimpleNamespace(config=config))

    trainer._apply_runtime_model_config_overrides_to_state(None, SimpleNamespace(lmhead_chunksize=512))
    trainer._apply_runtime_model_config_overrides_to_state(state, SimpleNamespace(lmhead_chunksize=None))

    assert config.lmhead_chunksize == 7
