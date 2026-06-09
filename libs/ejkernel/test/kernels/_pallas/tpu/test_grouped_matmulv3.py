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

"""Tests for Pallas TPU grouped_matmulv3 implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.grouped_matmulv3 import grouped_matmulv3
from ejkernel.kernels._xla.grouped_matmul import grouped_matmul as grouped_matmul_xla
from ejkernel.kernels._xla.grouped_matmulv3 import grouped_matmulv3 as grouped_matmulv3_xla


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_numerical_correctness_vs_xla(transpose_rhs):
    key = jax.random.PRNGKey(0)
    key, kl, kr = jax.random.split(key, 3)

    group_sizes = jnp.array([64, 64], dtype=jnp.int32)
    m = int(group_sizes.sum())
    k = 64
    n = 32
    lhs = jax.random.normal(kl, (m, k), dtype=jnp.float32)

    if transpose_rhs:
        rhs = jax.random.normal(kr, (len(group_sizes), n, k), dtype=jnp.float32)
    else:
        rhs = jax.random.normal(kr, (len(group_sizes), k, n), dtype=jnp.float32)

    out_tpu = grouped_matmulv3(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=(128, 128, 128))
    out_xla = grouped_matmul_xla(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=None)

    assert out_tpu.shape == out_xla.shape
    assert jnp.allclose(out_tpu, out_xla, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_numerical_correctness_vs_xla_with_rhs_scale_bias(transpose_rhs):
    key = jax.random.PRNGKey(11)
    key, kl, kr, ks, kb = jax.random.split(key, 5)

    group_sizes = jnp.array([64, 64], dtype=jnp.int32)
    m = int(group_sizes.sum())
    k = 64
    n = 32
    lhs = jax.random.normal(kl, (m, k), dtype=jnp.float32)
    rhs_scale = jax.random.normal(ks, (len(group_sizes), 2, 1, n), dtype=jnp.float32)
    rhs_bias = jax.random.normal(kb, (len(group_sizes), 1, n), dtype=jnp.float32)

    if transpose_rhs:
        rhs = jax.random.normal(kr, (len(group_sizes), n, k), dtype=jnp.float32)
    else:
        rhs = jax.random.normal(kr, (len(group_sizes), k, n), dtype=jnp.float32)

    out_tpu = grouped_matmulv3(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        transpose_rhs=transpose_rhs,
        tiling=(128, 128, 128),
    )
    out_xla = grouped_matmulv3_xla(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        transpose_rhs=transpose_rhs,
        tiling=None,
    )

    assert out_tpu.shape == out_xla.shape
    assert jnp.allclose(out_tpu, out_xla, rtol=5e-3, atol=1.1e-1)


def test_backward_shapes():
    key = jax.random.PRNGKey(1)
    key, kl, kr = jax.random.split(key, 3)

    group_sizes = jnp.array([64, 64], dtype=jnp.int32)
    m = int(group_sizes.sum())
    k = 64
    n = 16
    lhs = jax.random.normal(kl, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(kr, (len(group_sizes), k, n), dtype=jnp.float32)

    def loss_fn(lhs, rhs):
        out = grouped_matmulv3(lhs, rhs, group_sizes, tiling=(128, 128, 128))
        return jnp.sum(out**2)

    dlhs, drhs = jax.grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    assert dlhs.shape == lhs.shape
    assert drhs.shape == rhs.shape
    assert jnp.isfinite(dlhs).all()
    assert jnp.isfinite(drhs).all()


def test_backward_shapes_with_rhs_scale_bias():
    key = jax.random.PRNGKey(12)
    key, kl, kr, ks, kb = jax.random.split(key, 5)

    group_sizes = jnp.array([64, 64], dtype=jnp.int32)
    m = int(group_sizes.sum())
    k = 64
    n = 16
    lhs = jax.random.normal(kl, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(kr, (len(group_sizes), k, n), dtype=jnp.float32)
    rhs_scale = jax.random.normal(ks, (len(group_sizes), 2, 1, n), dtype=jnp.float32)
    rhs_bias = jax.random.normal(kb, (len(group_sizes), 1, n), dtype=jnp.float32)

    def loss_fn(lhs, rhs, rhs_scale, rhs_bias):
        out = grouped_matmulv3(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            tiling=(128, 128, 128),
        )
        return jnp.sum(out**2)

    dlhs, drhs, dscale, dbias = jax.grad(loss_fn, argnums=(0, 1, 2, 3))(lhs, rhs, rhs_scale, rhs_bias)
    assert dlhs.shape == lhs.shape
    assert drhs.shape == rhs.shape
    assert dscale.shape == rhs_scale.shape
    assert dbias.shape == rhs_bias.shape
    assert jnp.isfinite(dlhs).all()
    assert jnp.isfinite(drhs).all()
    assert jnp.isfinite(dscale).all()
    assert jnp.isfinite(dbias).all()
