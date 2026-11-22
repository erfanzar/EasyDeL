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

"""Pallas TPU quantized row gather (int8 codes + f32 row scales in one DMA per row) vs numpy takes."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.moe_row_gather import (
    gather_quantized_rows,
    moe_row_gather_quantized,
    quantized_row_gather_unsupported_reason,
)
from ejkernel.kernels._registry import Backend, Platform, kernel_registry

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")


def _inputs(rng, n, h):
    codes = jnp.asarray(rng.integers(-128, 128, (n, h)), jnp.int32).astype(jnp.int8)
    # Scales with special bit patterns: negative zero, the smallest normal, NaN and inf. Subnormal scales are flushed to
    # zero on TPU and the row quantization never produces one there (test_quantize_rows_scales_are_gathered_exactly).
    scale = np.asarray(rng.standard_normal(n) * 1e-2, np.float32)
    scale[:4] = (-0.0, np.finfo(np.float32).tiny, np.nan, np.inf)
    return codes, jnp.asarray(scale)


@pytest.mark.parametrize("n,h,rows", [(2048, 4096, 3072), (1024, 2048, 2048), (4096, 1536, 1024)])
def test_gather_quantized_is_bitwise(n, h, rows):
    rng = np.random.default_rng(n + h)
    codes, scale = _inputs(rng, n, h)
    idx = np.asarray(rng.integers(0, n, rows), np.int32)  # duplicates included
    idx[:4] = (0, 1, 2, 3)
    assert quantized_row_gather_unsupported_reason(codes, scale, jnp.asarray(idx)) is None
    got_c, got_s = jax.jit(gather_quantized_rows)(codes, scale, jnp.asarray(idx))
    assert got_c.dtype == jnp.int8 and got_c.shape == (rows, h)
    assert got_s.dtype == jnp.float32 and got_s.shape == (rows,)
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(codes)[idx])
    np.testing.assert_array_equal(np.asarray(got_s).view(np.int32), np.asarray(scale).view(np.int32)[idx])


def test_quantize_rows_scales_are_gathered_exactly():
    # The production contract: codes and scales from the A8 row quantization, including all-zero rows and rows whose
    # absmax is the smallest normal bfloat16 (its scale amax / 127 is subnormal on the host and flushed on TPU).
    from ejkernel.modules import grouped_matmul_channelwise_quantize_rows

    rng = np.random.default_rng(3)
    x = np.asarray(rng.standard_normal((2048, 4096)) * 3, np.float32)
    x[:4] = 0.0
    x[4:8] = float(jnp.finfo(jnp.bfloat16).tiny)
    x[8:12] *= 1e-30
    codes, scale = jax.jit(lambda v: grouped_matmul_channelwise_quantize_rows(v, 8))(jnp.asarray(x, jnp.bfloat16))
    scale = scale[:, 0]
    idx = np.asarray(rng.integers(0, 2048, 3072), np.int32)
    idx[:12] = np.arange(12)
    got_c, got_s = jax.jit(gather_quantized_rows)(codes, scale, jnp.asarray(idx))
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(codes)[idx])
    np.testing.assert_array_equal(np.asarray(got_s).view(np.int32), np.asarray(scale).view(np.int32)[idx])


def test_registry_entry_clamps_and_matches_xla():
    rng = np.random.default_rng(1)
    codes, scale = _inputs(rng, 2048, 4096)
    idx = np.asarray(rng.integers(0, 2048, 2048), np.int32)
    idx[:3] = (-5, 2048, 4000)
    pallas = jax.jit(moe_row_gather_quantized)(codes, scale, jnp.asarray(idx))
    xla = kernel_registry.get("moe_row_gather_quantized", platform=Platform.XLA, backend=Backend.ANY)
    ref = jax.jit(xla)(codes, scale, jnp.asarray(idx))
    for a, b in zip(pallas, ref, strict=True):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.int8 if a.dtype == jnp.int8 else np.int32),
            np.asarray(b).view(np.int8 if b.dtype == jnp.int8 else np.int32),
        )
    hlo = jax.jit(moe_row_gather_quantized).lower(codes, scale, jnp.asarray(idx)).compile().as_text()
    assert hlo.count("tpu_custom_call") >= 2  # pack + gather


@pytest.mark.parametrize(
    "n,h,dtype,match",
    [(2048, 640, jnp.int8, "hidden size"), (2048, 4096, jnp.int32, "int8"), (2040, 4096, jnp.int8, "source rows")],
)
def test_unsupported_shapes_report_a_reason_and_fall_back(n, h, dtype, match):
    rng = np.random.default_rng(2)
    codes = jnp.asarray(rng.integers(-128, 128, (n, h)), jnp.int32).astype(dtype)
    scale = jnp.asarray(rng.standard_normal(n), jnp.float32)
    idx = jnp.asarray(rng.integers(0, n, 1024), jnp.int32)
    reason = quantized_row_gather_unsupported_reason(codes, scale, idx)
    assert reason is not None and match in reason, reason
    got_c, got_s = moe_row_gather_quantized(codes, scale, idx)
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(codes)[np.asarray(idx)])
    np.testing.assert_array_equal(np.asarray(got_s), np.asarray(scale)[np.asarray(idx)])


def test_registry_entries_share_one_signature():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kernel_registry.validate_signatures("moe_row_gather_quantized")
    assert kernel_registry.get("moe_row_gather_quantized", backend=Backend.TPU) is kernel_registry.get(
        "moe_row_gather_quantized", platform=Platform.PALLAS, backend=Backend.TPU
    )
