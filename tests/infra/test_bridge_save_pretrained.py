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

from pathlib import Path

from easydel.infra.mixins.bridge import EasyBridgeMixin


class _ConfigStub:
    def __init__(self):
        self.architectures = None

    def save_pretrained(self, save_directory):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "config.json").write_text("{}")


class _StateStub:
    def to_pure_dict(self):
        return {"weight": "raw"}


class _BridgeModelStub:
    def __init__(self):
        self.config = _ConfigStub()
        self.generation_config = None
        self.mesh = "mesh"
        self._gather_fns = {"weight": lambda value: f"gathered:{value}"}

    def can_generate(self):
        return False


def test_save_model_files_uses_default_gather_fns_when_not_provided(tmp_path, monkeypatch):
    saved_trees: list[dict[str, object]] = []

    class _CheckpointerStub:
        def __init__(self, **kwargs):
            pass

        def save_pytree(self, *, tree, prefix, mesh, dtype):
            saved_trees.append(tree)
            return str(tmp_path / f"{prefix}.ckpt")

    monkeypatch.setattr("easydel.infra.mixins.bridge.nn.split", lambda *args, **kwargs: (None, _StateStub()))
    monkeypatch.setattr("easydel.infra.mixins.bridge.Checkpointer", _CheckpointerStub)

    EasyBridgeMixin._save_model_files(
        _BridgeModelStub(),
        save_directory=tmp_path,
        gather_fns=None,
        float_dtype=None,
        step=None,
    )

    assert saved_trees == [{"weight": "gathered:raw"}]
