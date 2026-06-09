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

from ejkernel.kernels._xla.all_gather_matmul import all_gather_matmul


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
    ("m_local", "k", "n_local", "rhs_transpose", "dtype", "bn", "bk"),
    [
        (64, 128, 64, False, jnp.bfloat16, 128, 128),
        (96, 256, 96, False, jnp.float32, 96, 64),
        (64, 128, 96, True, jnp.bfloat16, 96, 64),
    ],
)
def test_forward_matches_collective_reference(
    tp: int,
    m_local: int,
    k: int,
    n_local: int,
    rhs_transpose: bool,
    dtype: jnp.dtype,
    bn: int,
    bk: int,
):
    mesh, tp = _mesh_and_tp_axis(tp)
    m = m_local * tp
    n = n_local * tp

    key = jax.random.PRNGKey(100 + tp + int(rhs_transpose))
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k), dtype=dtype)
    if rhs_transpose:
        y = jax.random.normal(ky, (n, k), dtype=dtype)
        y_spec = PartitionSpec("tp", None)
    else:
        y = jax.random.normal(ky, (k, n), dtype=dtype)
        y_spec = PartitionSpec(None, "tp")

    kernel_fn = shard_map(
        functools.partial(
            all_gather_matmul,
            axis_name="tp",
            rhs_transpose=rhs_transpose,
            bn=bn,
            bk=bk,
        ),
        mesh=mesh,
        in_specs=(PartitionSpec("tp", None), y_spec),
        out_specs=PartitionSpec(None, "tp"),
        check_vma=False,
    )

    def reference_fn(x_shard, y_shard):
        x_full = lax.all_gather(x_shard, axis_name="tp", axis=0, tiled=True)
        if rhs_transpose:
            return jnp.dot(x_full, y_shard.T)
        return jnp.dot(x_full, y_shard)

    ref = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(PartitionSpec("tp", None), y_spec),
        out_specs=PartitionSpec(None, "tp"),
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
    ("m_local", "k", "n_local", "rhs_transpose"),
    [
        (32, 64, 32, False),
        (32, 64, 32, True),
        (48, 128, 64, False),
    ],
)
def test_backward_matches_collective_reference(
    tp: int,
    m_local: int,
    k: int,
    n_local: int,
    rhs_transpose: bool,
):
    mesh, tp = _mesh_and_tp_axis(tp)
    m = m_local * tp
    n = n_local * tp

    key = jax.random.PRNGKey(200 + tp + int(rhs_transpose))
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k), dtype=jnp.float32)
    if rhs_transpose:
        y = jax.random.normal(ky, (n, k), dtype=jnp.float32)
        y_spec = PartitionSpec("tp", None)
    else:
        y = jax.random.normal(ky, (k, n), dtype=jnp.float32)
        y_spec = PartitionSpec(None, "tp")

    kernel_fn = shard_map(
        functools.partial(all_gather_matmul, axis_name="tp", rhs_transpose=rhs_transpose),
        mesh=mesh,
        in_specs=(PartitionSpec("tp", None), y_spec),
        out_specs=PartitionSpec(None, "tp"),
        check_vma=False,
    )

    def reference_fn(x_shard, y_shard):
        x_full = lax.all_gather(x_shard, axis_name="tp", axis=0, tiled=True)
        if rhs_transpose:
            return jnp.dot(x_full, y_shard.T)
        return jnp.dot(x_full, y_shard)

    ref = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(PartitionSpec("tp", None), y_spec),
        out_specs=PartitionSpec(None, "tp"),
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
    ("x", "y", "rhs_transpose", "tp_size", "err"),
    [
        (
            jnp.ones((8, 4), dtype=jnp.float32),
            jnp.ones((4, 8), dtype=jnp.bfloat16),
            False,
            2,
            "share dtype",
        ),
        (
            jnp.ones((8, 4), dtype=jnp.float32),
            jnp.ones((4, 8), dtype=jnp.float32),
            False,
            0,
            "tp_size must be >= 1",
        ),
    ],
)
def test_input_validation_raises_value_error(
    x: jax.Array,
    y: jax.Array,
    rhs_transpose: bool,
    tp_size: int,
    err: str,
):
    with pytest.raises(ValueError, match=err):
        _ = all_gather_matmul(
            x,
            y,
            axis_name="tp",
            rhs_transpose=rhs_transpose,
            tp_size=tp_size,
        )
