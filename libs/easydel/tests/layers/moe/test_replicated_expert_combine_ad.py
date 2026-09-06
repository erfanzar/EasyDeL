"""Replicated expert fan-out transpose must SUM, not overwrite, replicas."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.moe import _communication_utils as communication
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


@pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() != 4, reason="four TPU devices required")
def test_replicated_expert_combine_primal_jvp_and_transpose():
    mesh = Mesh(np.array(jax.devices()), ("ep",))
    counts = jnp.array([1, 2, 0, 1], jnp.int32)

    @jax.shard_map(mesh=mesh, in_specs=P("ep", None), out_specs=P(), check_vma=False)
    def f(x):
        return communication.replicated_expert_combine(x, counts, axis_name="ep", output_rows=4)

    x = jnp.arange(48, dtype=jnp.float32).reshape(16, 3)
    f = jax.jit(f)
    def reference(a):
        return a[jnp.array([0, 4, 5, 12])]
    np.testing.assert_array_equal(f(x), reference(x))
    _, got = jax.jvp(f, (x,), (jnp.cos(x),))
    _, want = jax.jvp(reference, (x,), (jnp.cos(x),))
    np.testing.assert_array_equal(got, want)
    got = jax.jit(jax.grad(lambda a: jnp.sum(f(a))))(x)
    want = jax.grad(lambda a: jnp.sum(reference(a)))(x)
    np.testing.assert_array_equal(got, want)
