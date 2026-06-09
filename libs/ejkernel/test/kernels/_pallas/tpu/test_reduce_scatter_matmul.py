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

from ejkernel.kernels._pallas.tpu.reduce_scatter_matmul import reduce_scatter_matmul as reduce_scatter_matmul_pallas
from ejkernel.kernels._xla.reduce_scatter_matmul import reduce_scatter_matmul as reduce_scatter_matmul_xla


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
    ("m_local", "k_shard", "n", "dtype", "bm", "bn", "bk"),
    [
        (128, 128, 128, jnp.bfloat16, 64, 128, 64),
        (256, 128, 128, jnp.float32, 128, 128, 128),
    ],
)
def test_forward_matches_xla(
    tp: int,
    m_local: int,
    k_shard: int,
    n: int,
    dtype: jnp.dtype,
    bm: int,
    bn: int,
    bk: int,
):
    mesh, tp = _mesh_and_tp(tp)
    m = m_local * tp
    k = k_shard * tp

    key = jax.random.PRNGKey(700 + tp)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k), dtype=dtype)
    y = jax.random.normal(ky, (n, k), dtype=dtype)

    pallas_fn = shard_map(
        functools.partial(
            reduce_scatter_matmul_pallas,
            axis_name="tp",
            tp_size=tp,
            bm=bm,
            bn=bn,
            bk=bk,
        ),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )

    xla_fn = shard_map(
        functools.partial(reduce_scatter_matmul_xla, axis_name="tp"),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )

    out_pallas = pallas_fn(x, y)
    out_xla = xla_fn(x, y)

    assert out_pallas.shape == out_xla.shape
    # Bidirectional RSMM communicates/reduces bf16 partials in ring order, so
    # absolute drift vs one-shot XLA can be larger than matmul-only kernels.
    if dtype == jnp.float32:
        assert jnp.allclose(out_pallas, out_xla, rtol=1e-3, atol=1e-3)
    else:
        assert jnp.allclose(out_pallas, out_xla, rtol=1e-2, atol=1e-2)


def test_backward_shapes_and_finite():
    mesh, tp = _mesh_and_tp(_HEAVY_TP_SIZES[0])
    m = 256 * tp
    k_shard = 128
    k = k_shard * tp
    n = 128

    key = jax.random.PRNGKey(1)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m, k), dtype=jnp.float32)
    y = jax.random.normal(ky, (n, k), dtype=jnp.float32)

    pallas_fn = shard_map(
        functools.partial(
            reduce_scatter_matmul_pallas,
            axis_name="tp",
            tp_size=tp,
            bm=128,
            bn=128,
            bk=128,
        ),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
        out_specs=PartitionSpec("tp", None),
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
    ("x", "y", "bm", "bn", "bk", "err"),
    [
        (
            jnp.ones((255, 256), dtype=jnp.float32),
            jnp.ones((128, 256), dtype=jnp.float32),
            64,
            64,
            64,
            "must be divisible by num_devices",
        ),
        (
            jnp.ones((258, 256), dtype=jnp.float32),
            jnp.ones((128, 256), dtype=jnp.float32),
            64,
            64,
            64,
            "M_block",
        ),
        (
            jnp.ones((256, 256), dtype=jnp.float32),
            jnp.ones((128, 256), dtype=jnp.float32),
            96,
            64,
            64,
            "M_half_block",
        ),
        (
            jnp.ones((256, 256), dtype=jnp.float32),
            jnp.ones((130, 256), dtype=jnp.float32),
            64,
            64,
            64,
            "must be divisible by bn",
        ),
        (
            jnp.ones((256, 130), dtype=jnp.float32),
            jnp.ones((128, 130), dtype=jnp.float32),
            64,
            64,
            64,
            "K_shard",
        ),
    ],
)
def test_invalid_configs_raise(
    x: jax.Array,
    y: jax.Array,
    bm: int,
    bn: int,
    bk: int,
    err: str,
):
    with pytest.raises(ValueError, match=err):
        _ = reduce_scatter_matmul_pallas(
            x,
            y,
            axis_name="tp",
            tp_size=2,
            bm=bm,
            bn=bn,
            bk=bk,
        )
