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

import re

import jax
import jax.numpy as jnp
import optax
import pytest
import spectrax as spx
from jax.sharding import NamedSharding, PartitionSpec

import easydel as ed
import easydel.infra.base_state as base_state_module
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import sharding_matches


@pytest.fixture(scope="module")
def tiny_sharded_llama():
    module_config, module_class = ed.get_modules_by_type("llama", ed.TaskType.CAUSAL_LM)
    config = module_config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    config.add_basic_configurations(
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        use_sharding_constraint=False,
    )
    with config.mesh:
        model = module_class.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        return model.shard_model()


def test_to_state_handles_meta_graphother_leaves(tiny_sharded_llama):
    state = tiny_sharded_llama.to_state()
    assert isinstance(state, EasyDeLState)


def test_state_gather_paths_handle_graphother_tree(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama)
    gathered_model_state = state.gather_model()
    assert isinstance(gathered_model_state, EasyDeLState)
    gathered_state = state.gather_state()
    assert isinstance(gathered_state, EasyDeLState)


def test_shard_state_places_rng_count_with_explicit_named_sharding(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama).shard_state()
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    rng_count_leaf = None

    def _path_to_str(path):
        return "/".join(str(getattr(k, "name", getattr(k, "idx", getattr(k, "key", k)))) for k in path)

    for path, leaf in flat:
        path_str = _path_to_str(path)
        if "graphother" in path_str and "rng" in path_str:
            rng_count_leaf = leaf
            break

    assert rng_count_leaf is not None, "Expected RNG leaf in graphother tree."
    sharding = getattr(rng_count_leaf, "sharding", None)
    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == PartitionSpec()


def test_optimizer_gather_works_without_mesh_context_and_create_validation(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama).init_tx(optax.adam(1e-3))
    gathered_opt_state = state.gather_optimizer_state()
    assert isinstance(gathered_opt_state, EasyDeLState)

    gdef, gstate = spx.export(tiny_sharded_llama)
    gstate = gstate.filter("parameters", copy=False)
    gstate.exclude("parameters")
    with pytest.raises(ValueError):
        EasyDeLState.create(graphdef=gdef, graphstate=gstate, graphother=None)


def test_init_tx_places_optimizer_value_slots_with_named_sharding(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama).init_tx(optax.adam(1e-3))

    def _has_sharded_axis(spec: PartitionSpec) -> bool:
        return any(axis_spec is not None for axis_spec in tuple(spec))

    slot_shardings = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(state.opt_state):
        path_names = {getattr(key, "name", None) for key in path}
        if not ({"mu", "nu"} & path_names) or not hasattr(leaf, "shape"):
            continue
        sharding = getattr(leaf, "sharding", None)
        assert isinstance(sharding, NamedSharding)
        slot_shardings.append(sharding)

    assert slot_shardings, "Expected Adam optimizer slot shardings."
    assert any(_has_sharded_axis(sharding.spec) for sharding in slot_shardings), (
        "Optimizer value slots unexpectedly collapsed to replicated-only shardings."
    )


def test_partition_rules_are_open_ended_for_state_suffixes(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama)
    rules = state.model.resolve_shardings_regex()
    target_pattern = next(pattern for pattern, _ in rules if "model/norm/kernel" in pattern)

    assert target_pattern.endswith("(?:/.*)?$")
    assert re.search(target_pattern, "model/norm/kernel")
    assert re.search(target_pattern, "0/mu/model/norm/kernel/value")
    assert re.search(target_pattern, "0/mu/model/norm/kernel/value/extra")


def test_init_tx_skips_redundant_device_put_for_already_sharded_slots(monkeypatch, tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama)
    original_fastpath = base_state_module.device_put_if_sharding_mismatch
    calls = {"matched": 0, "placed": 0}

    def counting_fastpath(leaf, sharding, *, donate=False):
        if sharding_matches(leaf, sharding):
            calls["matched"] += 1
        else:
            calls["placed"] += 1
        return original_fastpath(leaf, sharding, donate=donate)

    monkeypatch.setattr(base_state_module, "device_put_if_sharding_mismatch", counting_fastpath)
    updated = state.init_tx(optax.adam(1e-3))

    assert updated.tx is not None
    assert updated.opt_state is not None
    assert calls["matched"] > 0
    assert calls["placed"] > 0
