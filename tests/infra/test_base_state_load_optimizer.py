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

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx as nn

import easydel as ed
from easydel.infra.base_state import EasyDeLState, _is_optimizer_template_incompatibility


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


def test_load_state_ignores_incompatible_optimizer_template(tmp_path, monkeypatch):
    model = _build_tiny_llama(0)
    tx = optax.adam(1e-3)
    from easydel.infra import base_module as base_module_mod
    from easydel.modules.auto import auto_configuration

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
    assert int(jax.device_get(restored.step)) == 0


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
    assert int(jax.device_get(restored.step)) == 0
