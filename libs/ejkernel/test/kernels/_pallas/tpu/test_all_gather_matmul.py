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
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec

from ejkernel.kernels._pallas.tpu.all_gather_matmul import all_gather_matmul as all_gather_matmul_pallas
from ejkernel.kernels._xla.all_gather_matmul import all_gather_matmul as all_gather_matmul_xla


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def _candidate_tp_sizes() -> tuple[int, ...]:
    n = len(jax.devices())
    if n <= 1:
        return (1,)
    sizes = [2]
    if n not in sizes:
        sizes.append(n)
    return tuple(sorted(set(sizes)))


_TP_SIZES = _candidate_tp_sizes()
_HEAVY_TP_SIZES = _TP_SIZES[:1]


def _mesh_and_tp(tp: int) -> tuple[Mesh, int]:
    devices = np.array(jax.devices("tpu")[:tp])
    mesh = Mesh(devices, axis_names=("tp",))
    return mesh, tp


@pytest.mark.parametrize("tp", _HEAVY_TP_SIZES)
@pytest.mark.parametrize(
    ("m_local", "k", "n_local", "rhs_transpose", "bn", "bk"),
    [
        (64, 128, 128, False, 128, 128),
        (64, 128, 128, True, 128, 128),
    ],
)
def test_forward_matches_xla(tp: int, m_local: int, k: int, n_local: int, rhs_transpose: bool, bn: int, bk: int):
    mesh, tp = _mesh_and_tp(tp)
    m = m_local * tp
    n = n_local * tp

    key = jax.random.PRNGKey(500 + tp + int(rhs_transpose))
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    if rhs_transpose:
        y = jax.random.normal(ky, (n, k), dtype=jnp.bfloat16)
        y_spec = PartitionSpec("tp", None)
    else:
        y = jax.random.normal(ky, (k, n), dtype=jnp.bfloat16)
        y_spec = PartitionSpec(None, "tp")

    pallas_fn = shard_map(
        functools.partial(
            all_gather_matmul_pallas,
            axis_name="tp",
            rhs_transpose=rhs_transpose,
            tp_size=tp,
            bn=bn,
            bk=bk,
        ),
        mesh=mesh,
        in_specs=(PartitionSpec("tp", None), y_spec),
        out_specs=PartitionSpec(None, "tp"),
        check_vma=False,
    )

    xla_fn = shard_map(
        functools.partial(all_gather_matmul_xla, axis_name="tp", rhs_transpose=rhs_transpose),
        mesh=mesh,
        in_specs=(PartitionSpec("tp", None), y_spec),
        out_specs=PartitionSpec(None, "tp"),
        check_vma=False,
    )

    out_pallas = pallas_fn(x, y)
    out_xla = xla_fn(x, y)

    assert out_pallas.shape == out_xla.shape
    assert jnp.allclose(out_pallas, out_xla, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tp", _HEAVY_TP_SIZES)
@pytest.mark.parametrize("rhs_transpose", [False, True])
def test_backward_shapes_and_finite(tp: int, rhs_transpose: bool):
    mesh, tp = _mesh_and_tp(tp)
    m_local = 64
    m = m_local * tp
    k = 128
    n_local = 128
    n = n_local * tp

    key = jax.random.PRNGKey(1)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k), dtype=jnp.float32)
    if rhs_transpose:
        y = jax.random.normal(ky, (n, k), dtype=jnp.float32)
        y_spec = PartitionSpec("tp", None)
    else:
        y = jax.random.normal(ky, (k, n), dtype=jnp.float32)
        y_spec = PartitionSpec(None, "tp")

    pallas_fn = shard_map(
        functools.partial(
            all_gather_matmul_pallas,
            axis_name="tp",
            rhs_transpose=rhs_transpose,
            tp_size=tp,
            bn=128,
            bk=128,
        ),
        mesh=mesh,
        in_specs=(PartitionSpec("tp", None), y_spec),
        out_specs=PartitionSpec(None, "tp"),
        check_vma=False,
    )

    def loss(x, y):
        out = pallas_fn(x, y)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    dx, dy = jax.grad(loss, argnums=(0, 1))(x, y)

    assert dx.shape == x.shape
    assert dy.shape == y.shape
    assert jnp.isfinite(dx).all()
    assert jnp.isfinite(dy).all()


@pytest.mark.parametrize(
    ("x", "y", "rhs_transpose", "bn", "bk", "err"),
    [
        (
            jnp.ones((31, 128), dtype=jnp.bfloat16),
            jnp.ones((128, 64), dtype=jnp.bfloat16),
            False,
            64,
            128,
            "divisible by 16",
        ),
        (
            jnp.ones((128, 256), dtype=jnp.bfloat16),
            jnp.ones((256, 128), dtype=jnp.bfloat16),
            False,
            0,
            128,
            "bn and bk must be positive",
        ),
        (
            jnp.ones((128, 256), dtype=jnp.bfloat16),
            jnp.ones((256, 128), dtype=jnp.bfloat16),
            False,
            96,
            128,
            "n_per_device",
        ),
        (
            jnp.ones((128, 256), dtype=jnp.bfloat16),
            jnp.ones((256, 128), dtype=jnp.bfloat16),
            False,
            64,
            192,
            "must be divisible by bk",
        ),
    ],
)
def test_invalid_block_or_shape_configs_raise(
    x: jax.Array,
    y: jax.Array,
    rhs_transpose: bool,
    bn: int,
    bk: int,
    err: str,
):
    with pytest.raises(ValueError, match=err):
        _ = all_gather_matmul_pallas(
            x,
            y,
            axis_name="tp",
            rhs_transpose=rhs_transpose,
            tp_size=2,
            bn=bn,
            bk=bk,
        )
