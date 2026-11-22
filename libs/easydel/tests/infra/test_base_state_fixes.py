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

"""Regression tests for ``EasyDeLState`` save/load behavior."""

import json
import types

import pytest
from easydel.infra.base_state import EasyDeLState
from spectrax.serialization import Checkpointer


class _MeshCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _CheckpointerStub:
    def __init__(self):
        self.calls: list[dict] = []

    def save_pytree(self, **kwargs):
        self.calls.append(kwargs)


def _optimizer_state_stub():
    return type(
        "_OptimizerStateStub",
        (),
        {
            "opt_state": {"momentum": [1.0, 2.0]},
            "model": type("_Model", (), {"mesh": _MeshCtx()})(),
        },
    )()


@pytest.mark.cpu_ok
def test_save_optimizer_records_step_in_checkpoint_extras(tmp_path):
    checkpointer = _CheckpointerStub()
    EasyDeLState.save_optimizer(_optimizer_state_stub(), save_directory=tmp_path, checkpointer=checkpointer, step=5)

    assert len(checkpointer.calls) == 1
    assert checkpointer.calls[0]["extras"] == {"step": 5}
    # ``step=`` itself must not be forwarded: it would nest run-{step}/run-{step} under the checkpoint directory.
    assert "step" not in checkpointer.calls[0]


@pytest.mark.cpu_ok
def test_save_optimizer_without_step_records_no_extras(tmp_path):
    checkpointer = _CheckpointerStub()
    EasyDeLState.save_optimizer(_optimizer_state_stub(), save_directory=tmp_path, checkpointer=checkpointer)

    assert len(checkpointer.calls) == 1
    assert "extras" not in checkpointer.calls[0]
    assert "step" not in checkpointer.calls[0]


@pytest.mark.cpu_ok
@pytest.mark.parametrize("step", [None, 5])
def test_save_optimizer_persists_step_metadata_without_nested_run_directory(tmp_path, step):
    checkpointer = Checkpointer(base_path=str(tmp_path), save_interval=None, step_policies=[])
    EasyDeLState.save_optimizer(_optimizer_state_stub(), tmp_path, checkpointer=checkpointer, step=step)
    structure = json.loads((tmp_path / "tx_structure.json").read_text())
    assert structure["extras"] == ({} if step is None else {"step": step})
    assert not (tmp_path / f"run-{step}").exists()


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "error",
    [
        KeyError("Missing array for key 'tx.weight'"),
        ValueError("Array shape mismatch for key 'tx.weight': expected (4,), got (8,)"),
        ValueError("unrelated serialization error"),
    ],
)
def test_load_optimizer_propagates_restore_errors_without_discarding_state(tmp_path, error):
    original = {"momentum": [1.0, 2.0]}
    state = types.SimpleNamespace(
        tx=types.SimpleNamespace(init=lambda params: {}),
        opt_state=original,
        graphstate={},
        model=types.SimpleNamespace(mesh=None, resolve_shardings_regex=lambda: ()),
    )

    class FailingCheckpointer:
        def load_pytree(self, **kwargs):
            raise error

    with pytest.raises(type(error)) as caught:
        EasyDeLState.load_optimizer(state, tmp_path, checkpointer=FailingCheckpointer())
    assert caught.value is error
    assert state.opt_state is original


@pytest.mark.cpu_ok
def test_shard_optimizer_state_reports_uninitialized_state():
    state = types.SimpleNamespace(opt_state=None)
    with pytest.raises(ValueError, match="not initialized"):
        EasyDeLState.shard_optimizer_state(state)


@pytest.mark.cpu_ok
def test_documented_checkpoint_aliases_remain_available():
    from easydel.infra.base_state import AM, WEIGHTS_NAME
    from spectrax.serialization import AsyncCheckpointManager

    assert AM is AsyncCheckpointManager
    assert WEIGHTS_NAME == "easydel-model.parameters"


@pytest.mark.cpu_ok
@pytest.mark.parametrize("use_shape_structs", [False, True])
def test_shard_with_shape_supports_direct_shardings_and_eval_shape_metadata(use_shape_structs):
    import jax
    from easydel.infra.base_config import EasyDeLBaseConfig
    from jax import numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1)).mesh.jax_mesh
    target = NamedSharding(mesh, PartitionSpec())
    state = EasyDeLState(
        step=jax.ShapeDtypeStruct((), jnp.int32),
        graphdef=None,
        graphstate={"weight": jax.ShapeDtypeStruct((4,), jnp.float32)},
        graphother={},
        tx=None,
        opt_state=None,
    )
    plan = jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=target) if use_shape_structs else target,
        state,
    )
    placed = state.shard_with_shape(plan)
    assert placed.step.sharding == target
    assert placed.graphstate["weight"].sharding == target
    assert placed.graphstate["weight"].shape == (4,)
    assert placed.graphstate["weight"].dtype == jnp.float32


@pytest.mark.cpu_ok
@pytest.mark.parametrize("child_timestamp", ["2026-10-18T09:00:00", "2026-10-19T12:00:00+03:00", "invalid"])
def test_state_checkpoint_discovery_accepts_mixed_timestamp_formats(tmp_path, child_timestamp):
    from spectrax.serialization.checkpointer import find_latest_checkpoint

    resume = tmp_path / "_resume_model"
    resume.mkdir()
    (resume / "metadata.json").write_text(json.dumps({"step": 1, "timestamp": child_timestamp}))
    (tmp_path / "metadata.json").write_text(json.dumps({"step": 1, "timestamp": "2026-10-19T10:00:00+00:00"}))
    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path)
    # Checkpointer construction also performs discovery before an optimizer load.
    Checkpointer(base_path=str(tmp_path), save_interval=None, step_policies=[])


@pytest.mark.cpu_ok
@pytest.mark.parametrize("explicit_signature", [False, True])
def test_state_save_retains_standard_kwargs_for_legacy_savers(tmp_path, explicit_signature):
    calls = []

    def save_pretrained(**kwargs):
        calls.append(kwargs)

    if explicit_signature:

        def save_pretrained(*, save_directory, float_dtype, step):
            calls.append({"save_directory": save_directory, "float_dtype": float_dtype, "step": step})

    model = types.SimpleNamespace(save_pretrained=save_pretrained, lora_is_enabled=False)
    state = types.SimpleNamespace(model=model, step=7, opt_state=None)
    EasyDeLState.save_state(state, tmp_path, save_optimizer=False)
    assert len(calls) == 1
    assert set(calls[0]) == {"save_directory", "float_dtype", "step"}
    assert str(calls[0]["save_directory"]) == str(tmp_path)
    assert calls[0]["step"] == 7
    assert calls[0]["float_dtype"] is None
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["step"] == 7
    assert metadata["has_optimizer_state"] is False
