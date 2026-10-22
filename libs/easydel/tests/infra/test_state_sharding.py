"""Public state-placement APIs preserve explicit variable shardings (accelerator only)."""

import jax
import numpy as np
import optax
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from .test_base_module_dtype import _tiny_llama


def _state_with_sharded_buffer_and_parameter():
    model = _tiny_llama()
    length = jax.device_count() * 2
    metadata = {"sharding": spx.Sharding(mesh_axes=("fsdp",))}
    model.extra_buffer = spx.Buffer(jnp.arange(length, dtype=jnp.float32), metadata=metadata)
    model.extra_weight = spx.Parameter(jnp.arange(length, dtype=jnp.float32), metadata=metadata)
    return EasyDeLState.create(model=model)


def test_shard_model_preserves_nonparameter_variable_sharding_and_values():
    state = _state_with_sharded_buffer_and_parameter()
    expected = np.arange(jax.device_count() * 2, dtype=np.float32)
    placed = state.shard_model()
    buffer = placed.model.extra_buffer.value
    assert buffer.sharding.spec == PartitionSpec("fsdp")
    np.testing.assert_array_equal(np.asarray(buffer), expected)
    assert placed.opt_state is state.opt_state


def test_public_shard_optimizer_state_places_external_moments_from_parameter_metadata():
    state = _state_with_sharded_buffer_and_parameter()
    tx = optax.adam(1e-3)
    external = tx.init(state.graphstate)
    state = state.replace(tx=tx)
    placed = state.shard_optimizer_state(external)
    matched = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(placed.opt_state)[0]:
        if "extra_weight" in jax.tree_util.keystr(path):
            matched.append(leaf)
            assert leaf.sharding.spec == PartitionSpec("fsdp")
            np.testing.assert_array_equal(np.asarray(leaf), np.zeros(jax.device_count() * 2, dtype=np.float32))
    assert len(matched) == 2  # Adam's first and second moment.
    assert state.opt_state is None


def test_public_shard_with_shape_places_each_array_using_supplied_plan():
    state = _state_with_sharded_buffer_and_parameter()
    mesh = state.mesh.jax_mesh
    replicated = NamedSharding(mesh, PartitionSpec())
    plan = jax.tree_util.tree_map(lambda _: replicated, state)
    placed = state.shard_with_shape(plan)
    before = jax.tree_util.tree_leaves(state)
    after = jax.tree_util.tree_leaves(placed)
    assert len(before) == len(after)
    for original, actual in zip(before, after, strict=True):
        assert actual.sharding == replicated
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(original))


def test_public_shard_state_mesh_override_places_parameters_buffers_moments_and_step():
    state = _state_with_sharded_buffer_and_parameter()
    tx = optax.adam(1e-3)
    state = state.replace(tx=tx, opt_state=tx.init(state.graphstate))
    original_mesh = state.mesh
    devices = list(jax.devices())
    assert len(devices) >= 2, "This regression needs a multi-device accelerator mesh"
    selected = devices[: len(devices) // 2]
    override = spx.SpxMesh(
        jax_mesh=jax.sharding.Mesh(
            np.asarray(selected).reshape(1, 1, len(selected), 1, 1, 1),
            ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        ),
        mpmd_axis=None,
    )
    placed = state.shard_state(mesh=override)
    target_devices = set(selected)
    for tree in (placed.graphstate, placed.graphother, placed.opt_state, placed.step):
        for leaf in jax.tree_util.tree_leaves(tree):
            if isinstance(leaf, jax.Array):
                assert leaf.devices() == target_devices
    expected = np.arange(jax.device_count() * 2, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(placed.model.extra_weight.value), expected)
    np.testing.assert_array_equal(np.asarray(placed.model.extra_buffer.value), expected)
    assert placed.model.extra_weight.value.sharding.spec == PartitionSpec("fsdp")
    assert placed.model.extra_buffer.value.sharding.spec == PartitionSpec("fsdp")
    moments = [
        leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(placed.opt_state)[0]
        if "extra_weight" in jax.tree_util.keystr(path)
    ]
    assert len(moments) == 2
    for moment in moments:
        assert moment.sharding.spec == PartitionSpec("fsdp")
        np.testing.assert_array_equal(np.asarray(moment), np.zeros_like(expected))
    assert placed.model.mesh is original_mesh
