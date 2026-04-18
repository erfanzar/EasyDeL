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

import jax
import jax.numpy as jnp

from easydel.infra.base_state import RESUME_MODEL_SUBDIR, EasyDeLState


class _ModelStub:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.unwrap_calls: list[bool] = []
        self.lora_is_enabled = False

    def save_pretrained(self, **kwargs):
        self.calls.append(dict(kwargs))

    def unwrap_lora_to_layers(self, verbose: bool = False):
        self.unwrap_calls.append(bool(verbose))
        return self


class _StateStub:
    def __init__(self, model, *, opt_state: object | None = None):
        self.model = model
        self.step = jnp.array(7, dtype=jnp.int32)
        self.opt_state = opt_state

    def save_optimizer(self, **kwargs):
        raise AssertionError("save_optimizer should not be called in this test")


def test_save_state_forwards_standard_save_kwargs_and_writes_metadata(tmp_path):
    model = _ModelStub()
    state = _StateStub(model)

    EasyDeLState.save_state(
        state,
        save_directory=tmp_path,
        save_optimizer=False,
    )

    assert len(model.calls) == 1
    assert set(model.calls[0]) == {"save_directory", "float_dtype", "step"}
    assert str(model.calls[0]["save_directory"]) == str(tmp_path)
    assert model.calls[0]["float_dtype"] is None
    assert model.calls[0]["step"] == 7
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["step"] == 7
    assert metadata["is_temporary"] is False
    assert metadata["has_optimizer_state"] is False
    assert metadata["has_resume_model"] is False


def test_save_state_merge_lora_before_save_writes_resume_copy_and_keeps_optimizer(tmp_path, monkeypatch):
    original_model = _ModelStub()
    original_model.lora_is_enabled = True
    merged_copy = _ModelStub()
    merged_copy.lora_is_enabled = True

    optimizer_calls: list[dict[str, object]] = []
    state = _StateStub(original_model, opt_state={"momentum": jnp.ones((1,), dtype=jnp.float32)})
    state.save_optimizer = lambda **kwargs: optimizer_calls.append(dict(kwargs))

    monkeypatch.setattr("easydel.infra.base_state.deepcopy_model", lambda model: merged_copy)

    EasyDeLState.save_state(
        state,
        save_directory=tmp_path,
        save_optimizer=True,
        merge_lora_before_save=True,
    )

    assert len(optimizer_calls) == 1
    assert str(optimizer_calls[0]["save_directory"]) == str(tmp_path)
    assert optimizer_calls[0]["float_dtype"] is None
    assert optimizer_calls[0]["step"] == 7
    assert len(original_model.calls) == 1
    assert str(original_model.calls[0]["save_directory"]) == str(tmp_path / RESUME_MODEL_SUBDIR)
    assert original_model.unwrap_calls == []
    assert merged_copy.unwrap_calls == [False]
    assert len(merged_copy.calls) == 1
    assert str(merged_copy.calls[0]["save_directory"]) == str(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["has_optimizer_state"] is True
    assert metadata["has_resume_model"] is True


def test_save_state_merge_lora_before_save_keeps_resume_copy_without_optimizer_state(tmp_path, monkeypatch):
    original_model = _ModelStub()
    original_model.lora_is_enabled = True
    merged_copy = _ModelStub()
    merged_copy.lora_is_enabled = True

    optimizer_calls: list[dict[str, object]] = []
    state = _StateStub(original_model, opt_state=None)
    state.save_optimizer = lambda **kwargs: optimizer_calls.append(dict(kwargs))

    monkeypatch.setattr("easydel.infra.base_state.deepcopy_model", lambda model: merged_copy)

    EasyDeLState.save_state(
        state,
        save_directory=tmp_path,
        save_optimizer=True,
        merge_lora_before_save=True,
    )

    assert len(optimizer_calls) == 1
    assert len(original_model.calls) == 1
    assert str(original_model.calls[0]["save_directory"]) == str(tmp_path / RESUME_MODEL_SUBDIR)
    assert original_model.unwrap_calls == []
    assert merged_copy.unwrap_calls == [False]
    assert len(merged_copy.calls) == 1
    assert str(merged_copy.calls[0]["save_directory"]) == str(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["has_optimizer_state"] is False
    assert metadata["has_resume_model"] is True


def test_save_state_merge_lora_before_save_without_optimizer_writes_resume_copy_for_model_only_resume(
    tmp_path,
    monkeypatch,
):
    original_model = _ModelStub()
    original_model.lora_is_enabled = True
    merged_copy = _ModelStub()
    merged_copy.lora_is_enabled = True

    state = _StateStub(original_model)
    state.save_optimizer = lambda **kwargs: (_ for _ in ()).throw(AssertionError("optimizer save should be skipped"))

    monkeypatch.setattr("easydel.infra.base_state.deepcopy_model", lambda model: merged_copy)

    EasyDeLState.save_state(
        state,
        save_directory=tmp_path,
        save_optimizer=False,
        merge_lora_before_save=True,
    )

    assert len(original_model.calls) == 1
    assert str(original_model.calls[0]["save_directory"]) == str(tmp_path / RESUME_MODEL_SUBDIR)
    assert original_model.unwrap_calls == []
    assert merged_copy.unwrap_calls == [False]
    assert len(merged_copy.calls) == 1
    assert str(merged_copy.calls[0]["save_directory"]) == str(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["has_optimizer_state"] is False
    assert metadata["has_resume_model"] is True


def test_save_optimizer_preserves_tree_in_multiprocess_mode(tmp_path, monkeypatch):
    saved_trees: list[dict[str, object]] = []

    class _MeshCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _CheckpointerStub:
        def save_pytree(self, **kwargs):
            saved_trees.append(kwargs["tree"])

    state = type(
        "_OptimizerStateStub",
        (),
        {
            "opt_state": {"momentum": jnp.ones((2,), dtype=jnp.float32)},
            "model": type("_Model", (), {"mesh": _MeshCtx()})(),
        },
    )()

    monkeypatch.setattr("easydel.infra.base_state.jax.process_count", lambda: 2)

    EasyDeLState.save_optimizer(
        state,
        save_directory=tmp_path,
        checkpointer=_CheckpointerStub(),
    )

    assert len(saved_trees) == 1
    assert isinstance(saved_trees[0]["momentum"], jax.Array)


def test_save_optimizer_skips_rank_zero_only_directory_creation_on_nonzero_process_for_remote_paths(monkeypatch):
    saved_trees: list[dict[str, object]] = []
    mkdir_calls: list[dict[str, object]] = []

    class _MeshCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _CheckpointerStub:
        def save_pytree(self, **kwargs):
            saved_trees.append(kwargs["tree"])

    class _FakePath:
        def mkdir(self, *args, **kwargs):
            mkdir_calls.append(dict(kwargs))

        def __str__(self):
            return "gs://bucket/fake-checkpoint"

    state = type(
        "_OptimizerStateStub",
        (),
        {
            "opt_state": {"momentum": jnp.ones((2,), dtype=jnp.float32)},
            "model": type("_Model", (), {"mesh": _MeshCtx()})(),
        },
    )()

    monkeypatch.setattr("easydel.infra.base_state.ePath", lambda path: _FakePath())
    monkeypatch.setattr("easydel.infra.base_state.jax.process_index", lambda: 1)

    EasyDeLState.save_optimizer(
        state,
        save_directory="ignored",
        checkpointer=_CheckpointerStub(),
    )

    assert mkdir_calls == []
    assert len(saved_trees) == 1
