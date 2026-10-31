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

"""Pallas grouped matmuls with a raised scoped-VMEM limit, and the local-FFN tile rule that relies on it.

* ``vmem_limit_bytes`` lets whole-contraction tiles compile; results match an independent float64 reference.
* ``vmem_limit_bytes=None`` is bit-identical to omitting it (same kernel).
* ``gmm_tiling`` / ``tgmm_tiling`` pick the large tiles only for divisible shapes that fit a raised limit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.grouped_matmul._pallas_impl import grouped_matmul, transposed_grouped_matmul
from ejkernel.kernels._pallas.tpu.moe_local_ffn import gmm_tiling, tgmm_tiling

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

BF16, F32 = jnp.bfloat16, jnp.float32
LIMIT = 80 * 2**20
M, K, N, G = 1024, 4096, 1024, 8


def _group_sizes(rng):
    sizes = rng.multinomial(M - 96, np.ones(G) / G)
    sizes[2] = 0  # an empty group
    return sizes.astype(np.int32)  # sum < M: the tail rows belong to no group


def _bf16(rng, shape, scale=1.0):
    return jnp.asarray(scale * rng.standard_normal(shape), F32).astype(BF16)


def _np(a):
    return np.asarray(jnp.asarray(a, F32), np.float64)


def _ref_gmm(lhs, rhs, sizes, transpose_rhs):
    lhs, rhs = _np(lhs), _np(rhs)
    out, start = [], 0
    for g, s in enumerate(sizes):
        w = rhs[g].T if transpose_rhs else rhs[g]
        out.append(lhs[start : start + s] @ w)
        start += s
    return np.concatenate(out, axis=0), start


@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_whole_contraction_tiles_with_raised_limit_match_float64(transpose_rhs):
    rng = np.random.default_rng(0)
    sizes = _group_sizes(rng)
    lhs = _bf16(rng, (M, K))
    rhs = _bf16(rng, (G, N, K) if transpose_rhs else (G, K, N), 0.02)
    gs = jnp.asarray(sizes)
    out = grouped_matmul(lhs, rhs, gs, BF16, (256, K, N // 2), transpose_rhs=transpose_rhs, vmem_limit_bytes=LIMIT)
    ref, rows = _ref_gmm(lhs, rhs, sizes, transpose_rhs)
    got = _np(out)[:rows]
    assert np.isfinite(got).all()
    assert np.linalg.norm(got - ref) <= 4e-3 * np.linalg.norm(ref)


def test_in_place_weight_gradient_with_raised_limit_matches_float64():
    rng = np.random.default_rng(1)
    sizes = _group_sizes(rng)
    lhs = _bf16(rng, (M, K))
    rhs = _bf16(rng, (M, N))
    acc = _bf16(rng, (G, K, N), 4.0)
    out = transposed_grouped_matmul(
        lhs.T, rhs, jnp.asarray(sizes), BF16, (256, 2048, 1024), existing_out=acc, vmem_limit_bytes=LIMIT
    )
    want, start = _np(acc).copy(), 0
    for g, s in enumerate(sizes):
        want[g] += _np(lhs)[start : start + s].T @ _np(rhs)[start : start + s]
        start += s
    got = _np(out)
    np.testing.assert_array_equal(got[2], _np(acc)[2])  # the empty group keeps its running sum exactly
    assert np.linalg.norm(got - want) <= 4e-3 * np.linalg.norm(want)


def test_unset_limit_is_bit_identical_to_the_previous_call():
    rng = np.random.default_rng(2)
    sizes = _group_sizes(rng)
    gs = jnp.asarray(sizes)
    rows = int(sizes.sum())  # rows past sum(group_sizes) are unwritten (unspecified, possibly NaN)
    lhs, rhs = _bf16(rng, (M, K)), _bf16(rng, (G, K, N), 0.02)
    tile = (512, 1024, 1024)
    a = grouped_matmul(lhs, rhs, gs, BF16, tile)
    b = grouped_matmul(lhs, rhs, gs, BF16, tile, vmem_limit_bytes=None)
    np.testing.assert_array_equal(np.asarray(a)[:rows], np.asarray(b)[:rows])
    ta = transposed_grouped_matmul(lhs.T, lhs, gs, F32, (512, 512, 1024))
    tb = transposed_grouped_matmul(lhs.T, lhs, gs, F32, (512, 512, 1024), vmem_limit_bytes=None)
    np.testing.assert_array_equal(np.asarray(ta), np.asarray(tb))


@pytest.mark.cpu_ok
def test_registered_entry_differentiates_with_and_without_a_limit():
    # The registered grouped_matmul is a custom_vjp over the impl; its static arguments must line up with the
    # impl's full signature (custom_vjp binds every default), or every grad call raises a TypeError.
    from ejkernel.kernels._pallas.tpu.grouped_matmul import grouped_matmul as registered

    rng = np.random.default_rng(3)
    sizes = np.full(G, 128, np.int32)
    lhs, rhs = _bf16(rng, (M, 512)), _bf16(rng, (G, 512, 256), 0.05)

    def loss(a, b):
        return jnp.sum(registered(a, b, jnp.asarray(sizes), BF16, (128, 128, 128)).astype(F32) ** 2)

    def ref_loss(a, b):
        a32, b32 = a.astype(F32), b.astype(F32).reshape(G, 1, 512, 256)
        out = jnp.einsum("gtk,gtkn->gtn", a32.reshape(G, 128, 512), jnp.broadcast_to(b32, (G, 128, 512, 256)))
        return jnp.sum(out**2)

    da, db = jax.jit(jax.grad(loss, argnums=(0, 1)))(lhs, rhs)
    ra, rb = jax.jit(jax.grad(ref_loss, argnums=(0, 1)))(lhs, rhs)
    for got, want in ((da, ra), (db, rb)):
        assert np.isfinite(_np(got)).all()
        assert np.linalg.norm(_np(got) - _np(want)) <= 2e-2 * np.linalg.norm(_np(want))


@pytest.mark.cpu_ok
def test_local_ffn_tile_rule():
    # Divisible window shapes (73728 rows, H=4096, M=2048) take the large tiles with a raised limit ...
    assert gmm_tiling(73728, 4096, 2048, 2, LIMIT) == ((256, 4096, 1024), LIMIT)
    assert gmm_tiling(73728, 2048, 4096, 2, LIMIT) == ((256, 2048, 2048), LIMIT)
    assert tgmm_tiling(73728, 4096, 2048, 2, 2, LIMIT) == ((256, 2048, 1024), LIMIT)
    assert tgmm_tiling(73728, 2048, 4096, 2, 2, LIMIT) == ((256, 2048, 1024), LIMIT)
    # ... and the fallback tiles at the default limit without one, or when the shape does not divide.
    assert gmm_tiling(73728, 4096, 2048, 2, None) == ((512, 1024, 1024), None)
    assert gmm_tiling(73728, 4096, 2000, 2, LIMIT) == ((512, 1024, 1024), None)
    assert tgmm_tiling(73728, 4096, 2048, 2, 2, None) == ((512, 512, 1024), None)
    # A contraction too large for the limit falls back as well.
    assert gmm_tiling(73728, 65536, 2048, 2, 16 * 2**20)[1] is None
