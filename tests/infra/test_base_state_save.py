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

import json

import jax.numpy as jnp

from easydel.infra.base_state import EasyDeLState


class _ModelStub:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def save_pretrained(self, **kwargs):
        self.calls.append(dict(kwargs))


class _StateStub:
    def __init__(self, model):
        self.model = model
        self.step = jnp.array(7, dtype=jnp.int32)

    def save_optimizer(self, **kwargs):
        raise AssertionError("save_optimizer should not be called in this test")


def test_save_state_forwards_gather_fns_and_writes_metadata(tmp_path):
    model = _ModelStub()
    state = _StateStub(model)
    gather_fns = {"params": lambda x: x}

    EasyDeLState.save_state(
        state,
        save_directory=tmp_path,
        save_optimizer=False,
        gather_fns=gather_fns,
    )

    assert len(model.calls) == 1
    assert str(model.calls[0]["save_directory"]) == str(tmp_path)
    assert model.calls[0]["gather_fns"] is gather_fns
    assert model.calls[0]["float_dtype"] is None
    assert model.calls[0]["step"] == 7
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["step"] == 7
    assert metadata["is_temporary"] is False
