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

"""Tests for Pallas TPU grouped_matmulv2 implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.grouped_matmulv2 import grouped_matmulv2
from ejkernel.kernels._xla.grouped_matmul import grouped_matmul as grouped_matmul_xla


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

    out_tpu = grouped_matmulv2(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=(128, 128, 128))
    out_xla = grouped_matmul_xla(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=None)

    assert out_tpu.shape == out_xla.shape
    assert jnp.allclose(out_tpu, out_xla, rtol=1e-3, atol=1e-3)


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
        out = grouped_matmulv2(lhs, rhs, group_sizes, tiling=(128, 128, 128))
        return jnp.sum(out**2)

    dlhs, drhs = jax.grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    assert dlhs.shape == lhs.shape
    assert drhs.shape == rhs.shape
    assert jnp.isfinite(dlhs).all()
    assert jnp.isfinite(drhs).all()
