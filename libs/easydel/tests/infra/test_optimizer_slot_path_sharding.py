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

"""Optimizer slots of wrapped optimizers (Muon, masked / multi_transform) mirror their parameter's sharding.

Wrappers prefix a slot's path and substitute ``MaskedNode`` for parameters a branch does not own, so
``optax.tree_map_params`` cannot zip a sharding tree against the state and no subtree is shaped like
the parameter tree. Every parameter-owned slot must still land on its parameter's sharding instead of
being replicated (or crashing placement).
"""

import easydel as ed
import jax
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from eformer.optimizers import MuonConfig, MuonOptimizer
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


@pytest.fixture(scope="module")
def fsdp_llama():
    if jax.device_count() < 4:
        pytest.skip("needs >=4 devices")
    config = ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    config.add_basic_configurations(sharding_axis_dims=(1, 1, -1, 1, 1, 1), use_sharding_constraint=False)
    with config.mesh:
        model = ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        return model.shard_model()


def _muon():
    return MuonOptimizer(config=MuonConfig()).build(optax.constant_schedule(1e-3))


def _masked():
    labels = lambda tree: jax.tree_util.tree_map(lambda x: "mat" if x.ndim == 2 else "vec", tree)  # noqa: E731
    return optax.multi_transform({"mat": optax.adam(1e-3), "vec": optax.sgd(1e-3, momentum=0.9)}, labels)


def _sharded_params(state):
    """(keystr, NamedSharding) for every parameter that is actually split over a mesh axis."""
    out = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state.graphstate)[0]:
        sharding = getattr(leaf, "sharding", None)
        if isinstance(sharding, NamedSharding) and any(a is not None for a in tuple(sharding.spec)):
            out.append((jax.tree_util.keystr(path), leaf.shape, sharding))
    assert out, "fixture produced no fsdp-sharded parameter"
    return out


def _assert_slots_mirror_params(state):
    opt_leaves = [
        (jax.tree_util.keystr(path), leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
        if hasattr(leaf, "shape")
    ]
    checked = 0
    for param_key, shape, param_sharding in _sharded_params(state):
        slots = [leaf for key, leaf in opt_leaves if key.endswith(param_key) and leaf.shape == shape]
        assert slots, f"no optimizer slot found for {param_key}"
        for slot in slots:
            assert isinstance(slot.sharding, NamedSharding)
            assert slot.sharding.spec == param_sharding.spec, (
                f"slot for {param_key} placed {slot.sharding.spec}, parameter is {param_sharding.spec}"
            )
            checked += 1
    assert checked
    for key, leaf in opt_leaves:
        if leaf.shape == ():
            assert leaf.sharding.spec == PartitionSpec(), key


@pytest.mark.parametrize("make_tx", [_muon, _masked], ids=["muon", "multi_transform"])
def test_init_tx_and_shard_state_mirror_param_shardings(fsdp_llama, make_tx):
    state = EasyDeLState.create(model=fsdp_llama).init_tx(make_tx())
    _assert_slots_mirror_params(state)
    resharded = state.shard_state()
    _assert_slots_mirror_params(resharded)


@pytest.mark.parametrize("make_tx", [_muon, _masked], ids=["muon", "multi_transform"])
def test_tx_less_shard_state_places_loaded_slots(fsdp_llama, make_tx):
    """A state carrying opt_state but no tx (as after a checkpoint load) still gets parameter-sharded slots."""
    state = EasyDeLState.create(model=fsdp_llama)
    replicated = NamedSharding(state.mesh.jax_mesh, PartitionSpec())
    opt_state = jax.device_put(make_tx().init(state.graphstate), replicated)
    placed = state.replace(tx=None, opt_state=opt_state).shard_state()
    _assert_slots_mirror_params(placed)
