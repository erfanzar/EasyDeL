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

"""Row padding vs. row tiling of the Pallas grouped matmul at small row counts.

A 6-row decode step (one request at top-6) against ``block_m = 512`` used to fail at trace time
("512 must be divisible by x-dimension tile size (6)") because the operation wrapper padded the rows
to ``block_m`` but clamped the row tile with the PRE-padding row count. These tests pin the pad/tile
pairing and run the compiled TPU kernels (v1 / v2 / v3, forward, ``existing_out``, the custom VJP and
the shard_map path) at row counts below, equal to and above the row tile against a dense reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules.operations import grouped_matmul
from ejkernel.modules.operations.configs import GroupedMatmulConfig
from ejkernel.modules.operations.grouped_matmul import GroupedMatmul, pad_rows_and_tile
from jax.sharding import Mesh, PartitionSpec


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="compiled Pallas TPU grouped matmul requires a TPU")

CFG = GroupedMatmulConfig(platform="pallas", backend="tpu", block_m=512, block_n=512, block_k=1024)
NUM_GROUPS, K, N = 4, 256, 384


def _group_sizes(m: int) -> np.ndarray:
    """Uneven sizes with an empty group; they sum to ``m``."""
    sizes = np.asarray([m // 3, 0, m // 3, m - 2 * (m // 3)], np.int32)
    assert sizes.sum() == m
    return sizes


def _dyadic(rng, shape, dtype=jnp.bfloat16):
    """Values in {-1, 0, 1} / 8: every partial sum below is exact in float32, so parity is bitwise."""
    return jnp.asarray(rng.integers(-1, 2, size=shape).astype(np.float32) / 8, dtype)


def _reference(lhs, rhs, sizes, *, transpose_rhs=False):
    lhs = np.asarray(lhs, np.float32)
    rhs = np.asarray(rhs, np.float32)
    out, start = [], 0
    for g, size in enumerate(sizes):
        w = rhs[g].T if transpose_rhs else rhs[g]
        out.append(lhs[start : start + size] @ w)
        start += size
    return np.concatenate(out, 0)


@pytest.mark.parametrize("m", [6, 96, 256, 512, 520, 1000, 1024])
def test_row_tile_divides_the_padded_rows(m):
    lhs = jnp.zeros((m, K), jnp.bfloat16)
    existing = jnp.ones((m, N), jnp.float32)
    padded, padded_existing, padded_size, (tm, tk, tn) = pad_rows_and_tile(
        lhs, CFG, K, N, do_padding=True, existing_out=existing
    )
    assert padded.shape == (m + padded_size, K)
    assert padded_existing.shape == (m + padded_size, N)
    assert padded.shape[0] % CFG.block_m == 0
    assert padded.shape[0] % tm == 0, (m, padded.shape[0], tm)
    assert (tk, tn) == (K, N)
    np.testing.assert_array_equal(np.asarray(padded_existing[m:]), 0)


def test_tiling_sentinel_and_no_padding():
    lhs = jnp.zeros((6, K), jnp.bfloat16)
    sentinel = GroupedMatmulConfig(block_m=512, block_n=0, block_k=0)
    out_lhs, _, padded_size, tiling = pad_rows_and_tile(lhs, sentinel, K, N, do_padding=True)
    assert tiling is None and padded_size == 506 and out_lhs.shape == (512, K)
    out_lhs, _, padded_size, tiling = pad_rows_and_tile(lhs, CFG, K, N, do_padding=False)
    assert padded_size == 0 and out_lhs.shape == (6, K) and tiling == (6, K, N)


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("m", [6, 96, 1000])
def test_small_row_counts_match_dense_reference(m, version):
    rng = np.random.default_rng(m + version)
    sizes = _group_sizes(m)
    lhs = _dyadic(rng, (m, K))
    rhs = _dyadic(rng, (NUM_GROUPS, K, N))
    out = jax.jit(
        lambda a, b, g: grouped_matmul(a, b, g, platform="pallas", cfg=CFG, use_v2=version == 2, use_v3=version == 3)
    )(lhs, rhs, jnp.asarray(sizes))
    assert out.shape == (m, N) and out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(out), _reference(lhs, rhs, sizes))


@pytest.mark.parametrize("m", [6, 96, 1000])
def test_existing_out_and_transpose_rhs(m):
    """``existing_out`` is the aliased output buffer of the Pallas kernel (Megablox contract).

    Rows covered by a group are written with the product; rows past ``sum(group_sizes)`` keep the
    ``existing_out`` values. The uncovered tail checks that ``existing_out`` is padded in lockstep
    with ``lhs`` when the rows are padded up to ``block_m``.
    """
    rng = np.random.default_rng(7 * m)
    tail = max(1, m // 4)
    sizes = _group_sizes(m - tail)
    lhs = _dyadic(rng, (m, K))
    rhs_t = _dyadic(rng, (NUM_GROUPS, N, K))
    existing = _dyadic(rng, (m, N), jnp.float32)
    out = jax.jit(lambda a, b, g, e: grouped_matmul(a, b, g, None, e, transpose_rhs=True, platform="pallas", cfg=CFG))(
        lhs, rhs_t, jnp.asarray(sizes), existing
    )
    out = np.asarray(out)
    assert out.shape == (m, N)
    np.testing.assert_array_equal(out[: m - tail], _reference(lhs[: m - tail], rhs_t, sizes, transpose_rhs=True))
    np.testing.assert_array_equal(out[m - tail :], np.asarray(existing)[m - tail :])


@pytest.mark.parametrize("m", [6, 96, 1000])
def test_vjp_matches_dense_reference(m):
    rng = np.random.default_rng(11 * m)
    sizes = _group_sizes(m)
    lhs = _dyadic(rng, (m, K), jnp.float32)
    rhs = _dyadic(rng, (NUM_GROUPS, K, N), jnp.float32)
    cot = _dyadic(rng, (m, N), jnp.float32)
    gs = jnp.asarray(sizes)

    def loss_kernel(a, b):
        return jnp.sum(grouped_matmul(a, b, gs, platform="pallas", cfg=CFG) * cot)

    def loss_dense(a, b):
        ids = jnp.repeat(jnp.arange(NUM_GROUPS), gs, total_repeat_length=m)
        out = jnp.einsum("mk,mkn->mn", a, b[ids], precision=jax.lax.Precision.HIGHEST)
        return jnp.sum(out * cot)

    d_lhs, d_rhs = jax.jit(jax.grad(loss_kernel, argnums=(0, 1)))(lhs, rhs)
    r_lhs, r_rhs = jax.jit(jax.grad(loss_dense, argnums=(0, 1)))(lhs, rhs)
    assert d_lhs.shape == lhs.shape and d_rhs.shape == rhs.shape
    np.testing.assert_allclose(np.asarray(d_lhs), np.asarray(r_lhs), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(d_rhs), np.asarray(r_rhs), rtol=0, atol=1e-6)


@pytest.mark.parametrize("m", [6, 96, 1000])
def test_shard_map_path_pads_inside_the_shard(m):
    """Output columns sharded over 4 chips: each shard runs the padded small-row Pallas call."""
    assert "create_shard_map_wrapper" in GroupedMatmul.__dict__
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four devices")
    mesh = Mesh(np.asarray(devices[:4]), ("tp",))
    n = 512  # 128 output columns per shard
    rng = np.random.default_rng(13 * m)
    sizes = _group_sizes(m)
    lhs = _dyadic(rng, (m, K))
    rhs = _dyadic(rng, (NUM_GROUPS, K, n))
    out = grouped_matmul(
        lhs,
        rhs,
        jnp.asarray(sizes),
        platform="pallas",
        cfg=CFG,
        mesh=mesh,
        in_specs=(PartitionSpec(), PartitionSpec(None, None, "tp"), PartitionSpec()),
        out_specs=PartitionSpec(None, "tp"),
    )
    assert out.shape == (m, n)
    np.testing.assert_array_equal(np.asarray(out), _reference(lhs, rhs, sizes))
