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

"""Pallas TPU MoE row gathers vs independent numpy references, plus fallback and registry contracts."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.moe_row_gather import (
    gather_sum_rows,
    moe_row_gather,
    moe_row_gather_sum,
    row_gather_sum_unsupported_reason,
    row_gather_unsupported_reason,
)
from ejkernel.kernels._pallas.tpu.moe_row_gather._pallas_impl import gather_sum_block_tokens
from ejkernel.kernels._registry import Backend, Platform, kernel_registry

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

BF16, F32 = jnp.bfloat16, jnp.float32
SRC, ROWS, H = 2048, 2048, 2048
TOKENS, K = 512, 4


def _bf16(rng, shape):
    return jnp.asarray(rng.standard_normal(shape), F32).astype(BF16)


def _np(a):
    return np.asarray(jnp.asarray(a, F32), np.float64)


def _custom_calls(fn, *args):
    return jax.jit(fn).lower(*args).compile().as_text().count("tpu_custom_call")


def test_gather_is_bit_exact_and_runs_the_kernel():
    rng = np.random.default_rng(0)
    x = _bf16(rng, (SRC, H))
    idx = jnp.asarray(rng.integers(0, SRC, ROWS), jnp.int32)  # duplicates included
    assert row_gather_unsupported_reason(x, idx) is None
    out, dot = jax.jit(lambda a, i: moe_row_gather(a, i))(x, idx)
    assert dot is None and out.dtype == BF16 and out.shape == (ROWS, H)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x)[np.asarray(idx)])
    assert _custom_calls(lambda a, i: moe_row_gather(a, i)[0], x, idx) >= 2  # pack + gather


def test_gather_scale_and_dot():
    rng = np.random.default_rng(1)
    x, y = _bf16(rng, (SRC, H)), _bf16(rng, (ROWS, H))
    idx = jnp.asarray(rng.permutation(SRC)[:ROWS], jnp.int32)
    scale = jnp.asarray(rng.uniform(0.1, 2.0, ROWS), F32)
    assert row_gather_unsupported_reason(x, idx, scale, y) is None
    out, dot = jax.jit(lambda a, i, s, d: moe_row_gather(a, i, s, d))(x, idx, scale, y)
    rows = np.asarray(x.astype(F32))[np.asarray(idx)]
    want = np.asarray(jnp.asarray(rows * np.asarray(scale)[:, None], F32).astype(BF16))
    np.testing.assert_array_equal(np.asarray(out), want)
    ref_dot = np.sum(rows.astype(np.float64) * _np(y), axis=-1)
    np.testing.assert_allclose(np.asarray(dot, np.float64), ref_dot, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("out_dtype", [None, F32])
def test_gather_sum_matches_float64(out_dtype):
    rng = np.random.default_rng(2)
    y = _bf16(rng, (SRC, H))
    idx = jnp.asarray(rng.permutation(SRC)[: TOKENS * K].reshape(TOKENS, K), jnp.int32)
    w = jnp.asarray(rng.uniform(0.0, 1.0, (TOKENS, K)), F32)
    assert row_gather_sum_unsupported_reason(y, idx, w, out_dtype) is None
    out = jax.jit(lambda a, i, ww: moe_row_gather_sum(a, i, ww, out_dtype))(y, idx, w)
    assert out.shape == (TOKENS, H) and out.dtype == jnp.dtype(out_dtype or BF16)
    ref = np.einsum("tkh,tk->th", _np(y)[np.asarray(idx)], _np(w))
    got = _np(out)
    # fp32 accumulation; the only rounding is the final cast (<= half a bf16 ulp, at most 2^-8 relative).
    tol = 2.0**-8 if out_dtype is None else 1e-6
    assert np.all(np.abs(got - ref) <= tol * np.abs(ref) + 1e-5)
    assert _custom_calls(lambda a, i, ww: moe_row_gather_sum(a, i, ww, out_dtype), y, idx, w) >= 2


def test_out_of_range_indices_are_clamped():
    rng = np.random.default_rng(3)
    x = _bf16(rng, (SRC, H))
    idx = np.asarray(rng.integers(0, SRC, ROWS), np.int32)
    idx[:3] = (-5, SRC, SRC + 7)
    out, _ = moe_row_gather(x, jnp.asarray(idx))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x)[np.clip(idx, 0, SRC - 1)])


@pytest.mark.parametrize(
    "shape,dtype,match",
    [((SRC, 384), BF16, "hidden size"), ((SRC, H), F32, "bfloat16"), ((SRC - 8, H), BF16, "source rows")],
)
def test_unsupported_shapes_report_a_reason_and_still_compute(shape, dtype, match):
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.standard_normal(shape), F32).astype(dtype)
    idx = jnp.asarray(rng.integers(0, shape[0], ROWS), jnp.int32)
    reason = row_gather_unsupported_reason(x, idx)
    assert reason is not None and match in reason, reason
    out, _ = moe_row_gather(x, idx)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x)[np.asarray(idx)])
    w = jnp.ones((ROWS // K, K), F32)
    assert row_gather_sum_unsupported_reason(x, idx.reshape(-1, K), w) is not None


def test_gather_sum_skip_negative_matches_float64_on_sparse_slots():
    # ~3/4 of the slots inactive, as for one tp shard's routes under experts-over-tp.
    rng = np.random.default_rng(5)
    y = _bf16(rng, (SRC, H))
    tokens = 1024
    idx = rng.integers(0, SRC, (tokens, K)).astype(np.int32)
    idx[rng.uniform(size=(tokens, K)) < 0.75] = -1
    w = jnp.asarray(rng.uniform(0.1, 1.0, (tokens, K)), F32)
    fn = jax.jit(lambda a, i, ww: gather_sum_rows(a, i, ww, F32, skip_negative=True))
    out = fn(y, jnp.asarray(idx), w)
    live = idx >= 0
    ref = np.einsum("tkh,tk->th", _np(y)[np.where(live, idx, 0)], np.where(live, _np(w), 0.0))
    assert np.all(np.abs(_np(out) - ref) <= 1e-6 * np.abs(ref) + 1e-5)
    assert (
        _custom_calls(lambda a, i, ww: gather_sum_rows(a, i, ww, F32, skip_negative=True), y, jnp.asarray(idx), w) >= 2
    )


def test_gather_sum_skip_negative_ignores_stale_and_clamped_rows():
    # Row 0 is NaN: a clamped negative index (the non-skip behaviour) would read it. Even token blocks also load the
    # NaN row into every buffer slot, so odd blocks that skip those slots would read it back if the stale buffer
    # rows were not masked.
    rng = np.random.default_rng(6)
    y = np.array(jnp.asarray(rng.standard_normal((SRC, H)), F32).astype(BF16).astype(F32))  # writable copy
    y[0] = np.nan
    y = jnp.asarray(y, F32).astype(BF16)
    block = gather_sum_block_tokens(K)
    tokens = 4 * block
    idx = rng.integers(1, SRC, (tokens, K)).astype(np.int32)
    for b in range(0, 4, 2):
        idx[b * block : (b + 1) * block] = 0
    for b in range(1, 4, 2):
        idx[b * block : (b + 1) * block, 1:] = -1
    w = jnp.asarray(rng.uniform(0.1, 1.0, (tokens, K)), F32)
    out = _np(jax.jit(lambda a, i, ww: gather_sum_rows(a, i, ww, F32, skip_negative=True))(y, jnp.asarray(idx), w))
    for b in range(1, 4, 2):
        rows = slice(b * block, (b + 1) * block)
        assert np.isfinite(out[rows]).all(), f"block {b} picked up a stale or clamped NaN row"
        ref = _np(y)[idx[rows, 0]] * _np(w)[rows, :1]
        assert np.all(np.abs(out[rows] - ref) <= 1e-6 * np.abs(ref) + 1e-5)


def test_gather_sum_skip_negative_equals_dense_slots_with_zero_weight():
    # The live-slot lists must land exactly the live rows in their buffer rows: pointing every dead slot at a finite
    # row with weight 0 makes the dense path sum the same fp32 terms in the same order, so the outputs are equal.
    # Includes an empty grid step (no copy at all) and a full one (every slot live).
    rng = np.random.default_rng(7)
    y = _bf16(rng, (SRC, H))
    block = gather_sum_block_tokens(K)
    tokens = 4 * block
    idx = rng.integers(0, SRC, (tokens, K)).astype(np.int32)
    dead = rng.uniform(size=(tokens, K)) < 0.75
    dead[:block] = True
    dead[block : 2 * block] = False
    w = rng.uniform(0.1, 1.0, (tokens, K)).astype(np.float32)
    sparse = jax.jit(lambda a, i, ww: gather_sum_rows(a, i, ww, F32, skip_negative=True))(
        y, jnp.asarray(np.where(dead, -1, idx)), jnp.asarray(w)
    )
    dense = jax.jit(lambda a, i, ww: gather_sum_rows(a, i, ww, F32))(
        y, jnp.asarray(idx), jnp.asarray(np.where(dead, 0.0, w).astype(np.float32))
    )
    np.testing.assert_array_equal(np.asarray(sparse), np.asarray(dense))
    assert np.all(np.asarray(sparse)[:block] == 0.0)


def test_registry_entries_share_one_signature():
    for name in ("moe_row_gather", "moe_row_gather_sum"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert kernel_registry.validate_signatures(name)
        assert kernel_registry.get(name, platform=Platform.PALLAS, backend=Backend.TPU) is not None
        assert kernel_registry.get(name, platform=Platform.XLA, backend=Backend.ANY) is not None
        assert kernel_registry.get(name, backend=Backend.TPU) is kernel_registry.get(
            name, platform=Platform.PALLAS, backend=Backend.TPU
        )
