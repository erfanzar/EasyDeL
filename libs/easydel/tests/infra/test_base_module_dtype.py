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

"""``half()`` / ``float()`` must retarget the runtime dtype of every submodule, not only the root."""

import easydel as ed
import jax
import spectrax as spx
from easydel.utils.traversals import iter_module_search
from jax import numpy as jnp


def _tiny_llama():
    config = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config.add_basic_configurations(sharding_axis_dims=(1, 1, -1, 1, 1, 1), use_sharding_constraint=False)
    with config.mesh:
        return ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


def _submodule_dtypes(model):
    return {module.dtype for _, module in iter_module_search(model) if hasattr(module, "dtype")}


def test_half_updates_submodule_runtime_dtype():
    model = _tiny_llama().half()
    assert model.dtype == jnp.float16
    assert {jnp.dtype(d) for d in _submodule_dtypes(model)} == {jnp.dtype(jnp.float16)}


def test_float_after_half_restores_submodule_runtime_dtype():
    model = _tiny_llama().half().float()
    assert model.dtype == jnp.float32
    assert {jnp.dtype(d) for d in _submodule_dtypes(model)} == {jnp.dtype(jnp.float32)}


def test_half_can_preserve_runtime_dtype_while_casting_parameters():
    model = _tiny_llama().half(change_runtime_dtype=False)
    assert {jnp.dtype(d) for d in _submodule_dtypes(model)} == {jnp.dtype(jnp.float32)}
    for _, module in iter_module_search(model):
        if hasattr(module, "param_dtype"):
            assert jnp.dtype(module.param_dtype) == jnp.dtype(jnp.float16)
    _, params, _ = model.split_module()
    floating_leaves = [leaf for leaf in jax.tree_util.tree_leaves(params) if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert floating_leaves
    assert all(leaf.dtype == jnp.float16 for leaf in floating_leaves)
