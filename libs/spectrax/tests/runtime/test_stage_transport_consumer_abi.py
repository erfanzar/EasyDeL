# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage dispatch must supply arrays matching the compiled consumer ABI."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.runtime.mpmd.runtime import _assemble_invars, _assemble_invars_from_plan, _prepare_invar_assembly_plan


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("source_kind", ["stage", "prev"])
@pytest.mark.parametrize("consumer_layout", ["replicated", "sharded", "legacy"])
def test_unannotated_edges_enter_compiled_consumer_with_exact_layout(prepared, source_kind, consumer_layout):
    devices = jax.devices()[:4]
    assert len(devices) == 4, "This regression requires the four-device pipeline mesh."
    source_mesh = jax.sharding.Mesh(np.asarray(devices[:2]), ("tp",))
    destination_mesh = jax.sharding.Mesh(np.asarray(devices[2:]), ("tp",))
    partition = jax.sharding.PartitionSpec
    source_sharding = jax.sharding.NamedSharding(source_mesh, partition("tp", None))
    fallback = jax.sharding.NamedSharding(destination_mesh, partition())
    target = (
        jax.sharding.NamedSharding(destination_mesh, partition(None, "tp")) if consumer_layout == "sharded" else fallback
    )
    host = np.arange(32, dtype=np.float32).reshape(8, 4)
    source = jax.device_put(host, source_sharding)
    source_map = [("stage", 0, 0, None, 0)] if source_kind == "stage" else [("prev", 0)]
    common = dict(
        flat_args=[],
        explicit_in_sh={},
        prev_outputs=(source,),
        all_cluster_outputs=[(source,)],
        ri=1,
        my_sh=fallback,
        rank_devices=set(devices[2:]),
        rank_submeshes=[source_mesh, destination_mesh],
        mpmd_mesh=None,
        expected_shardings=None if consumer_layout == "legacy" else (target,),
    )
    if prepared:
        plan = _prepare_invar_assembly_plan(source_map, {}, set(), 1)
        (value,) = _assemble_invars_from_plan(plan, **common)
    else:
        (value,) = _assemble_invars(source_map, placed={}, dynamic=set(), **common)
    # Checking both the positional layout and a real compiled consumer catches
    # accidental source-layout preservation even when element values are right.
    assert value.sharding.is_equivalent_to(target, ndim=2)
    np.testing.assert_array_equal(np.asarray(value), host)
    consumer = jax.jit(lambda x: x + jnp.float32(1), in_shardings=(target,), out_shardings=target)
    executable = consumer.lower(jax.ShapeDtypeStruct(host.shape, jnp.float32, sharding=target)).compile()
    result = executable(value)
    assert result.sharding.is_equivalent_to(target, ndim=2)
    np.testing.assert_array_equal(np.asarray(result), host + np.float32(1))
