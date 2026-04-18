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
import numpy as np
import optax
import pytest
from flax import nnx as nn

import easydel as ed
from easydel.infra.base_state import RESUME_MODEL_SUBDIR, EasyDeLState, _is_optimizer_template_incompatibility


def _build_tiny_llama(rng_seed: int = 0):
    config = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.add_basic_configurations(
        sharding_axis_dims=(1, 1, -1, 1, 1),
        use_sharding_constraint=False,
    )
    with config.mesh:
        return ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=nn.Rngs(rng_seed),
        )


def _build_tiny_lora_llama(rng_seed: int = 0):
    model = _build_tiny_llama(rng_seed)
    with model.mesh:
        return model.apply_lora_to_layers(
            lora_rank=2,
            lora_pattern=r"lm_head",
            verbose=False,
            rngs=nn.Rngs(rng_seed + 1),
        )


def test_load_optimizer_falls_back_to_saved_structure_when_template_init_fails(tmp_path):
    model = _build_tiny_llama(0)
    tx = optax.adam(1e-3)
    state = EasyDeLState.create(model=model, tx=tx, init_opt_state=True)
    grads = jax.tree_util.tree_map(jnp.ones_like, state.graphstate)
    state = state.apply_gradients(grads=grads)
    state.save_optimizer(tmp_path)

    fresh_model = _build_tiny_llama(1)
    fresh_state = EasyDeLState.create(model=fresh_model, init_opt_state=False)

    class BrokenTx:
        def init(self, params):
            del params
            raise RuntimeError("template init should not be required")

    restored = fresh_state.load_optimizer(tmp_path, tx_template=BrokenTx())

    assert restored.opt_state is not None

    orig_leaves, orig_treedef = jax.tree_util.tree_flatten(state.opt_state)
    restored_leaves, restored_treedef = jax.tree_util.tree_flatten(restored.opt_state)

    assert restored_treedef == orig_treedef
    assert len(restored_leaves) == len(orig_leaves)

    for original, loaded in zip(orig_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(jax.device_get(loaded)), np.asarray(jax.device_get(original)))


def test_create_model_respects_lora_graphstate_type():
    model = _build_tiny_lora_llama(0)
    tx = optax.adam(1e-3)

    state_from_model = EasyDeLState.create(model=model, tx=tx, init_opt_state=True)
    reference_state = model.to_state().init_tx(tx)

    model_graph_leaves, model_graph_treedef = jax.tree_util.tree_flatten(state_from_model.graphstate)
    ref_graph_leaves, ref_graph_treedef = jax.tree_util.tree_flatten(reference_state.graphstate)
    model_opt_leaves, model_opt_treedef = jax.tree_util.tree_flatten(state_from_model.opt_state)
    ref_opt_leaves, ref_opt_treedef = jax.tree_util.tree_flatten(reference_state.opt_state)

    assert state_from_model.model.lora_is_enabled is True
    assert model_graph_treedef == ref_graph_treedef
    assert len(model_graph_leaves) == len(ref_graph_leaves)
    assert model_opt_treedef == ref_opt_treedef
    assert len(model_opt_leaves) == len(ref_opt_leaves)


def test_load_optimizer_rejects_incompatible_template_after_missing_array_key(tmp_path):
    model = _build_tiny_llama(0)
    tx = optax.adam(1e-3)
    state = EasyDeLState.create(model=model, tx=tx, init_opt_state=True)
    grads = jax.tree_util.tree_map(jnp.ones_like, state.graphstate)
    state = state.apply_gradients(grads=grads)

    fresh_model = _build_tiny_llama(1)
    fresh_state = EasyDeLState.create(model=fresh_model, tx=tx, init_opt_state=False)

    class StubCheckpointer:
        def __init__(self, opt_state):
            self.opt_state = opt_state
            self.templates = []

        def load_pytree(self, **kwargs):
            self.templates.append(kwargs["template"])
            if kwargs["template"] is not None:
                raise KeyError("Missing array for key 'tx.fake_leaf' in checkpoint.")
            return self.opt_state, {"step": 7}

    checkpointer = StubCheckpointer(state.opt_state)
    with pytest.raises(KeyError, match="Missing array for key"):
        fresh_state.load_optimizer(tmp_path, checkpointer=checkpointer, tx_template=tx)

    assert len(checkpointer.templates) == 1
    assert checkpointer.templates[0] is not None


def test_load_state_restores_merged_lora_checkpoint_with_optimizer(tmp_path):
    model = _build_tiny_lora_llama(0)
    tx = optax.adam(1e-3)
    state = model.to_state().init_tx(tx)
    grads = jax.tree_util.tree_map(jnp.ones_like, state.graphstate)
    state = state.apply_gradients(grads=grads)

    state.save_state(
        tmp_path,
        save_optimizer=True,
        merge_lora_before_save=True,
    )

    restored = EasyDeLState.load_state(
        load_directory=tmp_path,
        auto_shard_model=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1),
        tx_template=tx,
    )

    original_opt_leaves, original_opt_treedef = jax.tree_util.tree_flatten(state.opt_state)
    restored_opt_leaves, restored_opt_treedef = jax.tree_util.tree_flatten(restored.opt_state)
    original_graph_leaves, original_graph_treedef = jax.tree_util.tree_flatten(state.graphstate)
    restored_graph_leaves, restored_graph_treedef = jax.tree_util.tree_flatten(restored.graphstate)

    assert restored.model.lora_is_enabled is True
    assert int(jax.device_get(restored.step)) == 1
    assert restored.opt_state is not None
    assert restored_graph_treedef == original_graph_treedef
    assert len(restored_graph_leaves) == len(original_graph_leaves)
    assert restored_opt_treedef == original_opt_treedef
    assert len(restored_opt_leaves) == len(original_opt_leaves)


def test_load_state_ignores_incompatible_optimizer_template(tmp_path, monkeypatch):
    model = _build_tiny_llama(0)
    tx = optax.adam(1e-3)
    from easydel.infra import base_module as base_module_mod
    from easydel.modules.auto import auto_configuration
    (tmp_path / "metadata.json").write_text(json.dumps({"step": 11, "has_optimizer_state": True}))

    def _raise_template_mismatch(self, *args, **kwargs):
        del self, args, kwargs
        raise KeyError("Missing array for key 'tx.fake_leaf' in checkpoint.")

    def _fake_from_pretrained(cls, *args, **kwargs):
        del cls, args, kwargs
        return _build_tiny_llama(1)

    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: model.config),
    )
    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "bind_model_task",
        staticmethod(lambda model_task, architectures: model_task),
    )
    monkeypatch.setattr(
        base_module_mod.EasyDeLBaseModule,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(EasyDeLState, "load_optimizer", _raise_template_mismatch)

    restored = EasyDeLState.load_state(
        load_directory=tmp_path,
        auto_shard_model=False,
        tx_template=tx,
    )

    assert restored.opt_state is None
    assert int(jax.device_get(restored.step)) == 11


class TestIsOptimizerTemplateIncompatibility:
    def test_keyerror_missing_array(self):
        exc = KeyError("Missing array for key 'tx.some_leaf' in checkpoint.")
        assert _is_optimizer_template_incompatibility(exc) is True

    def test_valueerror_shape_mismatch(self):
        exc = ValueError("Array shape mismatch for key 'tx.weight': expected (4,), got (8,)")
        assert _is_optimizer_template_incompatibility(exc) is True

    def test_keyerror_unrelated_message(self):
        exc = KeyError("some_other_key")
        assert _is_optimizer_template_incompatibility(exc) is False

    def test_valueerror_unrelated_message(self):
        exc = ValueError("invalid literal for int()")
        assert _is_optimizer_template_incompatibility(exc) is False

    def test_runtime_error_not_matched(self):
        exc = RuntimeError("Missing array for key 'tx.leaf'")
        assert _is_optimizer_template_incompatibility(exc) is False

    def test_type_error_not_matched(self):
        exc = TypeError("Array shape mismatch for key 'tx.leaf'")
        assert _is_optimizer_template_incompatibility(exc) is False


def test_load_optimizer_rejects_incompatible_template_after_shape_mismatch(tmp_path):
    """ValueError with 'Array shape mismatch' must propagate, not be swallowed."""
    model = _build_tiny_llama(0)
    tx = optax.adam(1e-3)
    state = EasyDeLState.create(model=model, tx=tx, init_opt_state=True)
    grads = jax.tree_util.tree_map(jnp.ones_like, state.graphstate)
    state = state.apply_gradients(grads=grads)

    fresh_model = _build_tiny_llama(1)
    fresh_state = EasyDeLState.create(model=fresh_model, tx=tx, init_opt_state=False)

    class StubCheckpointer:
        def __init__(self, opt_state):
            self.opt_state = opt_state
            self.templates = []

        def load_pytree(self, **kwargs):
            self.templates.append(kwargs["template"])
            if kwargs["template"] is not None:
                raise ValueError("Array shape mismatch for key 'tx.weight': expected (4,), got (8,)")
            return self.opt_state, {"step": 3}

    checkpointer = StubCheckpointer(state.opt_state)
    with pytest.raises(ValueError, match="Array shape mismatch for key"):
        fresh_state.load_optimizer(tmp_path, checkpointer=checkpointer, tx_template=tx)

    assert len(checkpointer.templates) == 1
    assert checkpointer.templates[0] is not None


def test_load_state_ignores_shape_mismatch_from_optimizer(tmp_path, monkeypatch):
    """load_state should log optimizer shape mismatches and continue loading the model."""
    model = _build_tiny_llama(0)
    tx = optax.adam(1e-3)
    from easydel.infra import base_module as base_module_mod
    from easydel.modules.auto import auto_configuration
    (tmp_path / "metadata.json").write_text(json.dumps({"step": 19, "has_optimizer_state": True}))

    def _raise_shape_mismatch(self, *args, **kwargs):
        del self, args, kwargs
        raise ValueError("Array shape mismatch for key 'tx.weight': expected (4,), got (8,)")

    def _fake_from_pretrained(cls, *args, **kwargs):
        del cls, args, kwargs
        return _build_tiny_llama(1)

    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: model.config),
    )
    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "bind_model_task",
        staticmethod(lambda model_task, architectures: model_task),
    )
    monkeypatch.setattr(
        base_module_mod.EasyDeLBaseModule,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(EasyDeLState, "load_optimizer", _raise_shape_mismatch)

    restored = EasyDeLState.load_state(
        load_directory=tmp_path,
        auto_shard_model=False,
        tx_template=tx,
    )

    assert restored.opt_state is None
    assert int(jax.device_get(restored.step)) == 19


def test_load_state_prefers_resume_model_subdir_for_model_restore(tmp_path, monkeypatch):
    model = _build_tiny_llama(0)
    from easydel.infra import base_module as base_module_mod
    from easydel.modules.auto import auto_configuration

    resume_dir = tmp_path / RESUME_MODEL_SUBDIR
    resume_dir.mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps({"has_optimizer_state": True, "has_resume_model": True}))
    config_paths: list[object] = []
    model_paths: list[object] = []
    optimizer_paths: list[object] = []

    def _fake_config_from_pretrained(path, *args, **kwargs):
        del args, kwargs
        config_paths.append(path)
        return model.config

    def _fake_from_pretrained(cls, *args, **kwargs):
        del cls, args
        model_paths.append(kwargs["pretrained_model_name_or_path"])
        return _build_tiny_llama(1)

    def _fake_load_optimizer(self, *, load_directory, tx_template=None, **kwargs):
        del tx_template, kwargs
        optimizer_paths.append(load_directory)
        return self

    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "from_pretrained",
        staticmethod(_fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "bind_model_task",
        staticmethod(lambda model_task, architectures: model_task),
    )
    monkeypatch.setattr(
        base_module_mod.EasyDeLBaseModule,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(EasyDeLState, "load_optimizer", _fake_load_optimizer)

    EasyDeLState.load_state(
        load_directory=tmp_path,
        auto_shard_model=False,
    )

    assert [str(path) for path in config_paths] == [str(resume_dir)]
    assert [str(path) for path in model_paths] == [str(resume_dir)]
    assert [str(path) for path in optimizer_paths] == [str(tmp_path)]


def test_load_state_prefers_resume_model_subdir_for_model_only_resume(tmp_path, monkeypatch):
    model = _build_tiny_llama(0)
    from easydel.infra import base_module as base_module_mod
    from easydel.modules.auto import auto_configuration

    resume_dir = tmp_path / RESUME_MODEL_SUBDIR
    resume_dir.mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps({"has_optimizer_state": False, "has_resume_model": True}))
    config_paths: list[object] = []
    model_paths: list[object] = []
    optimizer_paths: list[object] = []

    def _fake_config_from_pretrained(path, *args, **kwargs):
        del args, kwargs
        config_paths.append(path)
        return model.config

    def _fake_from_pretrained(cls, *args, **kwargs):
        del cls, args
        model_paths.append(kwargs["pretrained_model_name_or_path"])
        return _build_tiny_llama(1)

    def _fake_load_optimizer(self, *, load_directory, tx_template=None, **kwargs):
        del tx_template, kwargs
        optimizer_paths.append(load_directory)
        return self

    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "from_pretrained",
        staticmethod(_fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "bind_model_task",
        staticmethod(lambda model_task, architectures: model_task),
    )
    monkeypatch.setattr(
        base_module_mod.EasyDeLBaseModule,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(EasyDeLState, "load_optimizer", _fake_load_optimizer)

    EasyDeLState.load_state(
        load_directory=tmp_path,
        auto_shard_model=False,
    )

    assert [str(path) for path in config_paths] == [str(resume_dir)]
    assert [str(path) for path in model_paths] == [str(resume_dir)]
    assert optimizer_paths == []


def test_load_state_metadata_can_disable_stale_resume_model(tmp_path, monkeypatch):
    model = _build_tiny_llama(0)
    from easydel.infra import base_module as base_module_mod
    from easydel.modules.auto import auto_configuration

    resume_dir = tmp_path / RESUME_MODEL_SUBDIR
    resume_dir.mkdir()
    (tmp_path / "tx").mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps({"has_optimizer_state": False, "has_resume_model": False}))
    config_paths: list[object] = []
    model_paths: list[object] = []
    optimizer_paths: list[object] = []

    def _fake_config_from_pretrained(path, *args, **kwargs):
        del args, kwargs
        config_paths.append(path)
        return model.config

    def _fake_from_pretrained(cls, *args, **kwargs):
        del cls, args
        model_paths.append(kwargs["pretrained_model_name_or_path"])
        return _build_tiny_llama(1)

    def _fake_load_optimizer(self, *, load_directory, tx_template=None, **kwargs):
        del tx_template, kwargs
        optimizer_paths.append(load_directory)
        return self

    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "from_pretrained",
        staticmethod(_fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        auto_configuration.AutoEasyDeLConfig,
        "bind_model_task",
        staticmethod(lambda model_task, architectures: model_task),
    )
    monkeypatch.setattr(
        base_module_mod.EasyDeLBaseModule,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(EasyDeLState, "load_optimizer", _fake_load_optimizer)

    EasyDeLState.load_state(
        load_directory=tmp_path,
        auto_shard_model=False,
    )

    assert [str(path) for path in config_paths] == [str(tmp_path)]
    assert [str(path) for path in model_paths] == [str(tmp_path)]
    assert optimizer_paths == []
