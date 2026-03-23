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

import jax
import numpy as np
from jax import numpy as jnp

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


class _ArrayStateStub:
    def to_pure_dict(self):
        return {"weight": jnp.ones((2, 2), dtype=jnp.float32)}


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
    monkeypatch.setattr("easydel.infra.mixins.bridge.jax.process_count", lambda: 1)

    EasyBridgeMixin._save_model_files(
        _BridgeModelStub(),
        save_directory=tmp_path,
        gather_fns=None,
        float_dtype=None,
        step=None,
    )

    assert saved_trees == [{"weight": "gathered:raw"}]


def test_save_model_files_skips_default_gather_fns_in_multiprocess_mode(tmp_path, monkeypatch):
    saved_trees: list[dict[str, object]] = []

    class _CheckpointerStub:
        def __init__(self, **kwargs):
            pass

        def save_pytree(self, *, tree, prefix, mesh, dtype):
            saved_trees.append(tree)
            return str(tmp_path / f"{prefix}.ckpt")

    monkeypatch.setattr("easydel.infra.mixins.bridge.nn.split", lambda *args, **kwargs: (None, _StateStub()))
    monkeypatch.setattr("easydel.infra.mixins.bridge.Checkpointer", _CheckpointerStub)
    monkeypatch.setattr("easydel.infra.mixins.bridge.jax.process_count", lambda: 2)

    EasyBridgeMixin._save_model_files(
        _BridgeModelStub(),
        save_directory=tmp_path,
        gather_fns=None,
        float_dtype=None,
        step=None,
    )

    assert saved_trees == [{"weight": "raw"}]


def test_save_model_files_uses_compatibility_helper_in_multiprocess_mode(tmp_path, monkeypatch):
    saved_trees: list[dict[str, object]] = []

    class _CheckpointerStub:
        def __init__(self, **kwargs):
            pass

        def save_pytree(self, *, tree, prefix, mesh, dtype):
            saved_trees.append(tree)
            return str(tmp_path / f"{prefix}.ckpt")

    monkeypatch.setattr("easydel.infra.mixins.bridge.nn.split", lambda *args, **kwargs: (None, _ArrayStateStub()))
    monkeypatch.setattr("easydel.infra.mixins.bridge.Checkpointer", _CheckpointerStub)
    monkeypatch.setattr("easydel.infra.mixins.bridge.jax.process_count", lambda: 2)
    monkeypatch.setattr(
        "easydel.infra.mixins.bridge.ensure_multiprocess_checkpoint_compatible",
        lambda tree, *, mesh, context: {"weight": "compat"},
    )

    EasyBridgeMixin._save_model_files(
        _BridgeModelStub(),
        save_directory=tmp_path,
        gather_fns=None,
        float_dtype=None,
        step=None,
    )

    assert saved_trees == [{"weight": "compat"}]


def test_save_model_files_normalizes_numpy_arrays_before_checkpoint_write(tmp_path, monkeypatch):
    saved_trees: list[dict[str, object]] = []

    class _NumpyStateStub:
        def to_pure_dict(self):
            return {"weight": np.ones((2, 2), dtype=np.float32)}

    class _CheckpointerStub:
        def __init__(self, **kwargs):
            pass

        def save_pytree(self, *, tree, prefix, mesh, dtype):
            saved_trees.append(tree)
            return str(tmp_path / f"{prefix}.ckpt")

    monkeypatch.setattr("easydel.infra.mixins.bridge.nn.split", lambda *args, **kwargs: (None, _NumpyStateStub()))
    monkeypatch.setattr("easydel.infra.mixins.bridge.Checkpointer", _CheckpointerStub)

    EasyBridgeMixin._save_model_files(
        _BridgeModelStub(),
        save_directory=tmp_path,
        gather_fns={},
        float_dtype=None,
        step=None,
    )

    assert len(saved_trees) == 1
    assert isinstance(saved_trees[0]["weight"], jax.Array)


def test_save_model_files_offloads_numpy_arrays_to_cpu_before_checkpoint_write(tmp_path, monkeypatch):
    saved_trees: list[dict[str, object]] = []
    device_put_calls: list[tuple[object, object]] = []
    info_calls: list[tuple[object, ...]] = []

    class _NumpyStateStub:
        def to_pure_dict(self):
            return {"layer": {"weight": np.ones((2, 2), dtype=np.float32)}}

    class _CheckpointerStub:
        def __init__(self, **kwargs):
            pass

        def save_pytree(self, *, tree, prefix, mesh, dtype):
            saved_trees.append(tree)
            return str(tmp_path / f"{prefix}.ckpt")

    cpu_device = object()

    def _fake_device_put(value, device):
        device_put_calls.append((value, device))
        return jnp.asarray(value)

    monkeypatch.setattr("easydel.infra.mixins.bridge.nn.split", lambda *args, **kwargs: (None, _NumpyStateStub()))
    monkeypatch.setattr("easydel.infra.mixins.bridge.Checkpointer", _CheckpointerStub)
    monkeypatch.setattr(
        "easydel.infra.mixins.bridge.jax.devices",
        lambda platform=None: [cpu_device] if platform == "cpu" else [],
    )
    monkeypatch.setattr("easydel.infra.mixins.bridge.jax.device_put", _fake_device_put)
    monkeypatch.setattr("easydel.infra.mixins.bridge.logger.info", lambda *args, **kwargs: info_calls.append(args))

    EasyBridgeMixin._save_model_files(
        _BridgeModelStub(),
        save_directory=tmp_path,
        gather_fns={},
        float_dtype=None,
        step=None,
    )

    assert len(saved_trees) == 1
    assert isinstance(saved_trees[0]["layer"]["weight"], jax.Array)
    assert len(device_put_calls) == 1
    assert isinstance(device_put_calls[0][0], np.ndarray)
    assert device_put_calls[0][1] is cpu_device
    assert any("layer.weight" in str(call) for call in info_calls)
