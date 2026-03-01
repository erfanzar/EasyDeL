import re

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx as nn
from jax.sharding import NamedSharding, PartitionSpec

import easydel as ed
import easydel.infra.base_state as base_state_module
from easydel.infra.base_state import EasyDeLState


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
        sharding_axis_dims=(1, 1, -1, 1, 1),
        use_sharding_constraint=False,
    )
    with config.mesh:
        model = module_class.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=nn.Rngs(0),
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
        if "graphother" in path_str and "rngs" in path_str and "count" in path_str and "value" in path_str:
            rng_count_leaf = leaf
            break

    assert rng_count_leaf is not None, "Expected RNG count leaf in graphother tree."
    sharding = getattr(rng_count_leaf, "sharding", None)
    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == PartitionSpec()


def test_optimizer_gather_works_without_mesh_context_and_create_validation(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama).init_tx(optax.adam(1e-3))
    gathered_opt_state = state.gather_optimizer_state()
    assert isinstance(gathered_opt_state, EasyDeLState)

    graphdef, graphstate, _ = nn.split(tiny_sharded_llama, nn.Param, ...)
    with pytest.raises(ValueError):
        EasyDeLState.create(graphdef=graphdef, graphstate=graphstate, graphother=None)


def test_partition_rules_match_optimizer_value_paths(tiny_sharded_llama):
    from eformer import escale as es

    state = EasyDeLState.create(model=tiny_sharded_llama)
    rules = state.model._get_partition_rules(None)
    eval_opt_state = jax.eval_shape(lambda: optax.adam(1e-3).init(state.graphstate))
    partition_specs = es.match_partition_rules(rules, eval_opt_state)

    def _has_sharded_axis(spec: jax.sharding.PartitionSpec) -> bool:
        return any(axis_spec is not None for axis_spec in tuple(spec))

    spec_leaves = [
        spec for spec in jax.tree_util.tree_leaves(partition_specs) if isinstance(spec, jax.sharding.PartitionSpec)
    ]
    assert spec_leaves, "Expected optimizer partition-spec leaves."
    assert any(
        _has_sharded_axis(spec) for spec in spec_leaves
    ), "Optimizer partition specs unexpectedly collapsed to replicated-only specs."


def test_partition_rules_are_open_ended_for_state_suffixes(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama)
    rules = state.model._get_partition_rules(None)
    target_pattern = next(pattern for pattern, _ in rules if "model/norm/kernel" in pattern)

    assert target_pattern.endswith("(?:/.*)?$")
    assert re.search(target_pattern, "model/norm/kernel")
    assert re.search(target_pattern, "0/mu/model/norm/kernel/value")
    assert re.search(target_pattern, "0/mu/model/norm/kernel/value/extra")


def test_init_tx_builds_corrected_explicit_output_shardings(monkeypatch, tiny_sharded_llama):
    import eformer.escale as es

    state = EasyDeLState.create(model=tiny_sharded_llama)
    compile_calls = {"count": 0}
    captured: dict[str, object] = {}

    def fake_ejit(fn, **kwargs):
        compile_calls["count"] += 1
        captured.update(kwargs)

        def wrapped(graphstate):
            del graphstate
            return {"ok": True}

        return wrapped

    def fake_match_partition_rules(_rules, tree, *args, **kwargs):
        del _rules, args, kwargs
        return jax.tree_util.tree_map(
            lambda leaf: (
                jax.sharding.PartitionSpec("does_not_exist")
                if hasattr(leaf, "shape") and len(getattr(leaf, "shape", ())) > 0
                else jax.sharding.PartitionSpec()
            ),
            tree,
        )

    monkeypatch.setattr(base_state_module, "ejit", fake_ejit)
    monkeypatch.setattr(es, "match_partition_rules", fake_match_partition_rules)
    updated = state.init_tx(optax.adam(1e-3))

    assert compile_calls["count"] == 1
    assert updated.tx is not None
    assert updated.opt_state == {"ok": True}

    out_shardings = captured.get("out_shardings")
    assert out_shardings is not None

    mesh_axis_names = set(state.mesh.axis_names)

    def _iter_axis_names(spec):
        for axis_spec in tuple(spec):
            if axis_spec is None:
                continue
            if isinstance(axis_spec, tuple):
                yield from axis_spec
            else:
                yield axis_spec

    for leaf in jax.tree_util.tree_leaves(out_shardings):
        if leaf is None:
            continue
        assert isinstance(leaf, jax.sharding.NamedSharding)
        axis_names = set(_iter_axis_names(leaf.spec))
        assert "does_not_exist" not in axis_names
        assert axis_names.issubset(mesh_axis_names)
