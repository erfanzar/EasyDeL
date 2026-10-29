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

"""Real-TPU tests of the tile-safe streaming row permute (Mosaic compile, not interpret mode).

References are independent: a NumPy slice-copy loop over the run table and ``x[index]`` with an explicit index.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.row_permute import (
    TILE,
    invert_runs,
    permute_rows_runs,
    permute_rows_runs_pallas,
    row_permute,
    runs_to_index,
)
from ejkernel.kernels._registry import Backend, Platform, kernel_registry

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel needs a real TPU")


def segment_runs(rng, rows: int, n_src: int, n_exp: int, fill: float = 0.8, force: dict | None = None):
    """Source-major / expert-minor segments -> the expert-major run table and the valid prefix length.

    ``force`` pins ``{(source, expert): size}`` entries (e.g. one run longer than a chunk, one shorter than a tile,
    one empty); the remaining segments share the rest of the valid prefix with random, partly empty sizes.
    """
    force = force or {}
    weights = rng.integers(1, 6, size=(n_src, n_exp)).astype(np.float64)
    weights[rng.random(weights.shape) < 0.15] = 0  # empty segments
    free = np.ones((n_src, n_exp), bool)
    for key in force:
        free[key] = False
    total = int(rows * fill)
    budget = total - sum(force.values())
    assert budget > 0
    sizes = np.zeros((n_src, n_exp), np.int64)
    sizes[free] = np.floor(weights[free] / max(weights[free].sum(), 1) * budget).astype(np.int64)
    for key, size in force.items():
        sizes[key] = size
    first_free = tuple(np.argwhere(free)[0])
    sizes[first_free] += total - sizes.sum()
    recv_starts = np.cumsum(sizes.reshape(-1)) - sizes.reshape(-1)  # (s, e) order
    sorted_sizes = sizes.T.reshape(-1)  # (e, s) order
    sorted_starts = np.cumsum(sorted_sizes) - sorted_sizes
    runs = [
        (sorted_starts[e * n_src + s], recv_starts[s * n_exp + e], sorted_sizes[e * n_src + s])
        for e in range(n_exp)
        for s in range(n_src)
    ]
    return np.asarray(runs, np.int32), int(sizes.sum())


def numpy_permute(x, runs, out_rows, init=None):
    """Independent reference: slice copies (uncovered rows from ``init`` or zero)."""
    x = np.asarray(x)
    out = np.zeros((out_rows, *x.shape[1:]), x.dtype) if init is None else np.array(init, copy=True)
    covered = np.zeros(out_rows, bool)
    for d, s, n in np.asarray(runs):
        if n > 0:
            out[d : d + n] = x[s : s + n]
            covered[d : d + n] = True
    return out, covered


def bits(a):
    a = np.asarray(a)
    return a.view({2: np.uint16, 4: np.uint32}[a.dtype.itemsize])


def make_x(rng, shape, dtype):
    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.asarray(rng.integers(-(2**30), 2**30, size=shape), dtype)
    return jnp.asarray(rng.standard_normal(shape) * 100, dtype)


@pytest.mark.parametrize(
    ("rows", "width", "dtype", "chunk_rows"),
    [
        (1024, 256, jnp.bfloat16, 64),
        (784, 128, jnp.float32, 48),
        (512, 128, jnp.int32, 256),
        (2048, 640, jnp.bfloat16, 256),
        (1536, 384, jnp.uint32, 128),
    ],
)
def test_bit_exact_vs_gather_and_inverse(rows, width, dtype, chunk_rows):
    """Tile-unaligned starts on both ends, runs longer than a chunk and shorter than a tile, empty runs: bit-exact
    vs ``x[index]`` and the slice-copy reference; the inverse from the unsorted inverse table restores the prefix."""
    rng = np.random.default_rng(rows + width)
    runs, valid = segment_runs(rng, rows, n_src=4, n_exp=3, force={(1, 2): 5, (2, 0): 0, (3, 1): chunk_rows + 37})
    live = runs[runs[:, 2] > 0]
    assert np.any((live[:, 0] % TILE != 0) & (live[:, 1] % TILE != 0) & ((live[:, 0] - live[:, 1]) % TILE != 0))
    assert np.any(live[:, 2] > chunk_rows) and np.any(live[:, 2] < TILE) and np.any(runs[:, 2] == 0)
    x = make_x(rng, (rows, width), dtype)
    out = jax.jit(lambda x, r: permute_rows_runs_pallas(x, r, chunk_rows=chunk_rows))(x, jnp.asarray(runs))
    ref, covered = numpy_permute(x, runs, rows)
    assert covered.sum() == valid
    np.testing.assert_array_equal(bits(out), bits(ref))  # covered rows bit-exact, uncovered rows zero
    idx, cov = runs_to_index(jnp.asarray(runs), rows)
    np.testing.assert_array_equal(np.asarray(cov), covered)
    np.testing.assert_array_equal(bits(out)[covered], bits(x[idx])[covered])
    inv = invert_runs(jnp.asarray(runs))
    assert not np.all(np.diff(np.asarray(inv)[:, 0]) >= 0)  # the inverse table is not in destination order
    back = jax.jit(lambda y, r: permute_rows_runs_pallas(y, r, chunk_rows=chunk_rows))(out, inv)
    np.testing.assert_array_equal(bits(back)[:valid], bits(x)[:valid])


def test_full_coverage_leaves_no_row_unwritten():
    """With a tail run covering the rest, every output row is written (NaN init would show any hole)."""
    rng = np.random.default_rng(7)
    rows, width = 640, 128
    runs, valid = segment_runs(rng, rows, n_src=3, n_exp=3, fill=0.7)
    runs = np.concatenate([runs, [[valid, 0, rows - valid]]]).astype(np.int32)
    x = make_x(rng, (rows, width), jnp.bfloat16)
    init = jnp.full((rows, width), jnp.nan, jnp.bfloat16)
    out = jax.jit(lambda x, r, i: permute_rows_runs_pallas(x, r, init=i, chunk_rows=64))(x, jnp.asarray(runs), init)
    idx, covered = runs_to_index(jnp.asarray(runs), rows)
    assert bool(np.all(np.asarray(covered)))
    assert not np.any(np.isnan(np.asarray(out, np.float32)))
    np.testing.assert_array_equal(bits(out), bits(x[idx]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_init_kept_across_gaps_and_padding_runs_skipped(dtype):
    """Destination gaps (a run removed mid-range, no tail) keep ``init`` on both sides of every partial tile;
    zero-length padding rows in the table are skipped; a gap spanning several windows is handled."""
    rng = np.random.default_rng(3)
    rows, width = 1024, 256
    runs, valid = segment_runs(rng, rows, n_src=3, n_exp=2)
    order = np.argsort(runs[:, 0])
    drop = {int(order[2]), int(order[len(order) // 2])}
    keep = [i for i in range(len(runs)) if i not in drop]
    runs = np.concatenate([runs[keep], np.zeros((5, 3), np.int32)]).astype(np.int32)
    runs = runs[rng.permutation(len(runs))]  # any order
    x = make_x(rng, (rows, width), dtype)
    init = make_x(rng, (rows, width), dtype)
    out = jax.jit(lambda x, r, i: permute_rows_runs_pallas(x, r, init=i, chunk_rows=64))(x, jnp.asarray(runs), init)
    ref, covered = numpy_permute(x, runs, rows, init=init)
    assert 0 < covered.sum() < valid
    np.testing.assert_array_equal(bits(out), bits(ref))


def test_gaps_without_init_are_zero_and_match_xla_reference():
    rng = np.random.default_rng(5)
    rows, width = 768, 128
    runs, _ = segment_runs(rng, rows, n_src=4, n_exp=2, fill=0.6)
    runs = runs[[i for i in range(len(runs)) if i % 3 != 1]]
    x = make_x(rng, (rows, width), jnp.float32)
    out = jax.jit(lambda x, r: permute_rows_runs_pallas(x, r, chunk_rows=32))(x, jnp.asarray(runs))
    ref, _ = numpy_permute(x, runs, rows)
    np.testing.assert_array_equal(bits(out), bits(ref))
    xla = kernel_registry.get("row_permute", platform=Platform.XLA, backend=Backend.ANY)
    np.testing.assert_array_equal(bits(out), bits(jax.jit(xla)(x, jnp.asarray(runs))))


def test_out_rows_differs_from_rows_and_empty_table():
    rng = np.random.default_rng(9)
    x = make_x(rng, (256, 128), jnp.float32)
    runs = np.asarray([[3, 200, 40], [100, 17, 33], [400, 0, 112]], np.int32)
    out = jax.jit(lambda x, r: permute_rows_runs_pallas(x, r, out_rows=512, chunk_rows=64))(x, jnp.asarray(runs))
    ref, _ = numpy_permute(x, runs, 512)
    np.testing.assert_array_equal(bits(out), bits(ref))
    empty = jax.jit(lambda x, r: permute_rows_runs_pallas(x, r))(x, jnp.zeros((0, 3), jnp.int32))
    np.testing.assert_array_equal(np.asarray(empty), np.zeros((256, 128), np.float32))


def test_rejects_unsupported_shapes_and_dtypes():
    runs = jnp.zeros((4, 3), jnp.int32)
    with pytest.raises(ValueError, match="multiples of"):
        permute_rows_runs_pallas(jnp.zeros((520, 128), jnp.float32), runs)
    with pytest.raises(ValueError, match="multiples of"):
        permute_rows_runs_pallas(jnp.zeros((512, 200), jnp.float32), runs)
    with pytest.raises(ValueError, match="bfloat16"):
        permute_rows_runs_pallas(jnp.zeros((512, 128), jnp.int16), runs)
    with pytest.raises(ValueError, match="bfloat16"):
        permute_rows_runs_pallas(jnp.zeros((512, 128), jnp.float16), runs)
    with pytest.raises(ValueError, match="int32"):
        permute_rows_runs_pallas(jnp.zeros((512, 128), jnp.float32), jnp.zeros((4, 3), jnp.float32))
    with pytest.raises(ValueError, match="chunk_rows"):
        permute_rows_runs_pallas(jnp.zeros((512, 128), jnp.float32), runs, chunk_rows=40)


@pytest.mark.parametrize("dtype", [jnp.int16, jnp.float16])
def test_registered_entry_falls_back_to_gather_for_unsupported_dtype(dtype):
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.integers(-100, 100, size=(520, 128)), dtype)
    runs = np.asarray([[0, 300, 200], [200, 5, 300]], np.int32)
    out = jax.jit(row_permute)(x, jnp.asarray(runs))
    ref, _ = numpy_permute(x, runs, 520)
    np.testing.assert_array_equal(bits(out), bits(ref))


@pytest.mark.parametrize("with_init", [False, True])
def test_custom_vjp_equals_gather_gradient(with_init):
    rng = np.random.default_rng(11)
    rows, width = 640, 128
    runs, valid = segment_runs(rng, rows, n_src=4, n_exp=2, fill=0.85)  # partial permutation: some rows unread
    runs_j = jnp.asarray(runs)
    x = make_x(rng, (rows, width), jnp.float32)
    init = make_x(rng, (rows, width), jnp.float32)
    cot = make_x(rng, (rows, width), jnp.float32)
    idx, covered = runs_to_index(runs_j, rows)
    mask = covered[:, None]

    def loss_k(x, init):
        out = permute_rows_runs(x, runs_j, init=init if with_init else None, chunk_rows=64)
        return jnp.sum(out * cot)

    def loss_r(x, init):
        base = init if with_init else jnp.zeros_like(x)
        return jnp.sum(jnp.where(mask, x[idx], base) * cot)

    gk = jax.jit(jax.grad(loss_k, argnums=(0, 1)))(x, init)
    gr = jax.jit(jax.grad(loss_r, argnums=(0, 1)))(x, init)
    np.testing.assert_array_equal(np.asarray(gk[0]), np.asarray(gr[0]))
    if with_init:
        np.testing.assert_array_equal(np.asarray(gk[1]), np.asarray(gr[1]))
    assert valid < rows


def test_registry_entries_and_signatures():
    assert kernel_registry.validate_signatures("row_permute")
    pallas = kernel_registry.get("row_permute", platform=Platform.PALLAS, backend=Backend.TPU)
    xla = kernel_registry.get("row_permute", platform=Platform.XLA, backend=Backend.ANY)
    assert pallas is row_permute and xla is not pallas


def test_compiles_at_production_shape():
    """Mosaic compile (tile verifier included) at [262144, 4096] bf16 with 289 runs, without allocating buffers."""
    x = jax.ShapeDtypeStruct((262144, 4096), jnp.bfloat16)
    runs = jax.ShapeDtypeStruct((289, 3), jnp.int32)
    jax.jit(lambda x, r: permute_rows_runs_pallas(x, r)).lower(x, runs).compile()
    jax.jit(lambda x, r, i: permute_rows_runs_pallas(x, r, init=i)).lower(x, runs, x).compile()
