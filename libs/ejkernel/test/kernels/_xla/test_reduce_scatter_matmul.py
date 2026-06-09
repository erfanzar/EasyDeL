# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, shard_map
from jax.sharding import Mesh, PartitionSpec

from ejkernel.kernels._xla.reduce_scatter_matmul import reduce_scatter_matmul


def _candidate_tp_sizes() -> tuple[int, ...]:
    n = len(jax.devices())
    if n <= 1:
        return (1,)
    sizes = [2]
    if n not in sizes:
        sizes.append(n)
    return tuple(sorted(set(sizes)))


_TP_SIZES = _candidate_tp_sizes()


def _mesh_and_tp_axis(tp: int) -> tuple[Mesh, int]:
    devices = jax.devices()[:tp]
    mesh = Mesh(np.array(devices), axis_names=("tp",))
    return mesh, tp


@pytest.mark.parametrize("tp", _TP_SIZES)
@pytest.mark.parametrize(
    ("m_local", "k_shard", "n", "dtype", "bm", "bn", "bk"),
    [
        (128, 64, 128, jnp.bfloat16, 128, 128, 128),
        (96, 128, 192, jnp.float32, 64, 96, 64),
    ],
)
def test_forward_matches_collective_reference(
    tp: int,
    m_local: int,
    k_shard: int,
    n: int,
    dtype: jnp.dtype,
    bm: int,
    bn: int,
    bk: int,
):
    mesh, tp = _mesh_and_tp_axis(tp)
    m = m_local * tp

    key = jax.random.PRNGKey(300 + tp)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k_shard * tp), dtype=dtype)
    y = jax.random.normal(ky, (n, k_shard * tp), dtype=dtype)

    kernel_fn = shard_map(
        functools.partial(
            reduce_scatter_matmul,
            axis_name="tp",
            bm=bm,
            bn=bn,
            bk=bk,
        ),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )

    def reference_fn(x_shard, y_shard):
        partial_out = jnp.dot(x_shard, y_shard.T)
        return lax.psum_scatter(partial_out, axis_name="tp", scatter_dimension=0, tiled=True)

    ref = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )

    out = kernel_fn(x, y)
    expected = ref(x, y)

    assert out.shape == expected.shape
    if dtype == jnp.float32:
        assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)
    else:
        assert jnp.allclose(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tp", _TP_SIZES)
@pytest.mark.parametrize(
    ("m_local", "k_shard", "n"),
    [
        (64, 32, 64),
        (96, 64, 96),
    ],
)
def test_backward_matches_collective_reference(tp: int, m_local: int, k_shard: int, n: int):
    mesh, tp = _mesh_and_tp_axis(tp)
    m = m_local * tp

    key = jax.random.PRNGKey(400 + tp)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k_shard * tp), dtype=jnp.float32)
    y = jax.random.normal(ky, (n, k_shard * tp), dtype=jnp.float32)

    kernel_fn = shard_map(
        functools.partial(reduce_scatter_matmul, axis_name="tp"),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )

    def reference_fn(x_shard, y_shard):
        partial_out = jnp.dot(x_shard, y_shard.T)
        return lax.psum_scatter(partial_out, axis_name="tp", scatter_dimension=0, tiled=True)

    ref = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )

    def loss_kernel(x, y):
        out = kernel_fn(x, y)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    def loss_ref(x, y):
        out = ref(x, y)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    dx, dy = jax.grad(loss_kernel, argnums=(0, 1))(x, y)
    dx_ref, dy_ref = jax.grad(loss_ref, argnums=(0, 1))(x, y)

    assert dx.shape == x.shape
    assert dy.shape == y.shape
    assert jnp.isfinite(dx).all()
    assert jnp.isfinite(dy).all()
    assert jnp.allclose(dx, dx_ref, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(dy, dy_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ("x", "y", "tp_size", "err"),
    [
        (
            jnp.ones((8, 4), dtype=jnp.float32),
            jnp.ones((8, 4), dtype=jnp.bfloat16),
            2,
            "share dtype",
        ),
        (
            jnp.ones((8, 4), dtype=jnp.float32),
            jnp.ones((8, 4), dtype=jnp.float32),
            0,
            "tp_size must be >= 1",
        ),
    ],
)
def test_input_validation_raises_value_error(x: jax.Array, y: jax.Array, tp_size: int, err: str):
    with pytest.raises(ValueError, match=err):
        _ = reduce_scatter_matmul(
            x,
            y,
            axis_name="tp",
            tp_size=tp_size,
        )
