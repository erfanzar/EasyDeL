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

"""Stress test for GDR Pallas kernels with realistic Qwen3Next dimensions.

Tests all code paths (chunked prefill, single-step decode, recurrent fallback)
with dimensions matching production models to catch VMEM out-of-range errors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.gated_delta_rule import gated_delta_rule as gdr_pallas
from ejkernel.kernels._xla.gated_delta_rule import gated_delta_rule as gdr_xla


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="TPU required")

QWEN3_NEXT_DIMS = dict(qk_dim=128, v_dim=128, heads=16)
QWEN3_NEXT_SMALL_DIMS = dict(qk_dim=128, v_dim=128, heads=4)


def _make(batch, seq_len, heads, qk_dim, v_dim, dtype=jnp.float32, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(keys[0], (batch, seq_len, heads, qk_dim), dtype=dtype)
    k = jax.random.normal(keys[1], (batch, seq_len, heads, qk_dim), dtype=dtype)
    v = jax.random.normal(keys[2], (batch, seq_len, heads, v_dim), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(keys[3], (batch, seq_len, heads), dtype=dtype))
    decay = jax.random.normal(keys[4], (batch, seq_len, heads), dtype=dtype) * -0.01
    return q, k, v, beta, decay


class TestChunkedPrefill:
    """Test _chunk_gdr_fwd (Phase 1 + Phase 2) with production dimensions."""

    @pytest.mark.parametrize("chunk_size", [64, 128, 256])
    def test_chunked_fwd_realistic_dims(self, chunk_size):
        q, k, v, beta, decay = _make(1, 512, **QWEN3_NEXT_SMALL_DIMS, seed=10)
        out, state = gdr_pallas(q, k, v, beta, decay, chunk_size=chunk_size)
        assert out.shape == v.shape
        assert state.shape == (
            1,
            QWEN3_NEXT_SMALL_DIMS["heads"],
            QWEN3_NEXT_SMALL_DIMS["qk_dim"],
            QWEN3_NEXT_SMALL_DIMS["v_dim"],
        )
        assert jnp.all(jnp.isfinite(out)), f"NaN/Inf in output, chunk_size={chunk_size}"
        assert jnp.all(jnp.isfinite(state)), f"NaN/Inf in state, chunk_size={chunk_size}"

    @pytest.mark.parametrize("chunk_size", [64, 128, 256])
    def test_chunked_matches_xla_recurrent(self, chunk_size):
        q, k, v, beta, decay = _make(1, 256, **QWEN3_NEXT_SMALL_DIMS, dtype=jnp.float32, seed=11)
        out_p, st_p = gdr_pallas(q, k, v, beta, decay, chunk_size=chunk_size)
        out_x, st_x = gdr_xla(q, k, v, beta, decay, chunk_size=chunk_size, use_chunked=False)
        assert jnp.allclose(out_p, out_x, atol=5e-2, rtol=0), f"Output diff: {jnp.max(jnp.abs(out_p - out_x))}"
        assert jnp.allclose(st_p, st_x, atol=5e-2, rtol=0), f"State diff: {jnp.max(jnp.abs(st_p - st_x))}"

    def test_chunked_full_heads(self):
        q, k, v, beta, decay = _make(1, 256, **QWEN3_NEXT_DIMS, seed=12)
        out, state = gdr_pallas(q, k, v, beta, decay, chunk_size=64)
        assert out.shape == v.shape
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    def test_chunked_bf16(self):
        q, k, v, beta, decay = _make(1, 256, **QWEN3_NEXT_SMALL_DIMS, dtype=jnp.bfloat16, seed=13)
        out, state = gdr_pallas(q, k, v, beta, decay, chunk_size=64)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    def test_chunked_with_initial_state(self):
        q, k, v, beta, decay = _make(1, 128, **QWEN3_NEXT_SMALL_DIMS, seed=14)
        init = (
            jax.random.normal(
                jax.random.PRNGKey(99),
                (1, QWEN3_NEXT_SMALL_DIMS["heads"], QWEN3_NEXT_SMALL_DIMS["qk_dim"], QWEN3_NEXT_SMALL_DIMS["v_dim"]),
                dtype=jnp.float32,
            )
            * 0.01
        )
        out, state = gdr_pallas(q, k, v, beta, decay, chunk_size=64, initial_state=init)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    def test_chunked_no_decay(self):
        q, k, v, beta, _ = _make(1, 128, **QWEN3_NEXT_SMALL_DIMS, seed=15)
        out, state = gdr_pallas(q, k, v, beta, None, chunk_size=64)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    def test_chunked_unaligned_seqlen(self):
        q, k, v, beta, decay = _make(1, 137, **QWEN3_NEXT_SMALL_DIMS, seed=16)
        out, _state = gdr_pallas(q, k, v, beta, decay, chunk_size=64)
        assert out.shape == v.shape
        assert jnp.all(jnp.isfinite(out))


class TestSingleStepDecode:
    """Test _single_step_gdr_fwd with production dimensions."""

    def test_single_step_basic(self):
        dims = QWEN3_NEXT_SMALL_DIMS
        q, k, v, beta, decay = _make(1, 1, **dims, seed=20)
        init = (
            jax.random.normal(
                jax.random.PRNGKey(21),
                (1, dims["heads"], dims["qk_dim"], dims["v_dim"]),
                dtype=jnp.float32,
            )
            * 0.01
        )
        out, state = gdr_pallas(q, k, v, beta, decay, initial_state=init)
        assert out.shape == (1, 1, dims["heads"], dims["v_dim"])
        assert state.shape == init.shape
        assert jnp.all(jnp.isfinite(out)), "Single-step output NaN/Inf"
        assert jnp.all(jnp.isfinite(state)), "Single-step state NaN/Inf"

    def test_single_step_full_heads(self):
        dims = QWEN3_NEXT_DIMS
        q, k, v, beta, decay = _make(1, 1, **dims, seed=22)
        init = jnp.zeros((1, dims["heads"], dims["qk_dim"], dims["v_dim"]), dtype=jnp.float32)
        out, state = gdr_pallas(q, k, v, beta, decay, initial_state=init)
        assert out.shape == (1, 1, dims["heads"], dims["v_dim"])
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    def test_single_step_bf16(self):
        dims = QWEN3_NEXT_SMALL_DIMS
        q, k, v, beta, decay = _make(1, 1, **dims, dtype=jnp.bfloat16, seed=23)
        init = jnp.zeros((1, dims["heads"], dims["qk_dim"], dims["v_dim"]), dtype=jnp.bfloat16)
        out, _state = gdr_pallas(q, k, v, beta, decay, initial_state=init)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_single_step_matches_xla(self):
        dims = QWEN3_NEXT_SMALL_DIMS
        q, k, v, beta, decay = _make(1, 1, **dims, dtype=jnp.float32, seed=24)
        init = (
            jax.random.normal(
                jax.random.PRNGKey(25),
                (1, dims["heads"], dims["qk_dim"], dims["v_dim"]),
                dtype=jnp.float32,
            )
            * 0.01
        )
        out_p, st_p = gdr_pallas(q, k, v, beta, decay, initial_state=init)
        out_x, st_x = gdr_xla(q, k, v, beta, decay, initial_state=init, use_chunked=False)
        assert jnp.allclose(out_p, out_x, atol=1e-4, rtol=0), f"Output diff: {jnp.max(jnp.abs(out_p - out_x))}"
        assert jnp.allclose(st_p, st_x, atol=1e-4, rtol=0), f"State diff: {jnp.max(jnp.abs(st_p - st_x))}"

    def test_single_step_no_decay(self):
        dims = QWEN3_NEXT_SMALL_DIMS
        q, k, v, beta, _ = _make(1, 1, **dims, seed=26)
        init = jnp.zeros((1, dims["heads"], dims["qk_dim"], dims["v_dim"]), dtype=jnp.float32)
        out, state = gdr_pallas(q, k, v, beta, None, initial_state=init)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    @pytest.mark.parametrize("batch", [1, 2, 4])
    def test_single_step_batched(self, batch):
        dims = QWEN3_NEXT_SMALL_DIMS
        q, k, v, beta, decay = _make(batch, 1, **dims, seed=27)
        init = jnp.zeros((batch, dims["heads"], dims["qk_dim"], dims["v_dim"]), dtype=jnp.float32)
        out, _state = gdr_pallas(q, k, v, beta, decay, initial_state=init)
        assert out.shape == (batch, 1, dims["heads"], dims["v_dim"])
        assert jnp.all(jnp.isfinite(out))


class TestPrefillThenDecode:
    """Test the full prefill→decode flow that mirrors inference."""

    def test_prefill_then_multi_step_decode(self):
        dims = QWEN3_NEXT_SMALL_DIMS
        prefill_len = 128
        num_decode_steps = 5

        q, k, v, beta, decay = _make(1, prefill_len + num_decode_steps, **dims, seed=30)

        out_pf, state = gdr_pallas(
            q[:, :prefill_len],
            k[:, :prefill_len],
            v[:, :prefill_len],
            beta[:, :prefill_len],
            decay[:, :prefill_len],
            chunk_size=64,
        )
        assert jnp.all(jnp.isfinite(out_pf)), "Prefill output NaN/Inf"
        assert jnp.all(jnp.isfinite(state)), "Prefill state NaN/Inf"

        for step in range(num_decode_steps):
            t = prefill_len + step
            out_dec, state = gdr_pallas(
                q[:, t : t + 1],
                k[:, t : t + 1],
                v[:, t : t + 1],
                beta[:, t : t + 1],
                decay[:, t : t + 1],
                initial_state=state,
            )
            assert jnp.all(jnp.isfinite(out_dec)), f"Decode step {step} output NaN/Inf"
            assert jnp.all(jnp.isfinite(state)), f"Decode step {step} state NaN/Inf"

    def test_prefill_decode_matches_full_recurrent(self):
        dims = QWEN3_NEXT_SMALL_DIMS
        prefill_len = 64
        total_len = prefill_len + 3

        q, k, v, beta, decay = _make(1, total_len, **dims, dtype=jnp.float32, seed=31)

        out_full, _ = gdr_xla(q, k, v, beta, decay, use_chunked=False)

        _, state = gdr_pallas(
            q[:, :prefill_len],
            k[:, :prefill_len],
            v[:, :prefill_len],
            beta[:, :prefill_len],
            decay[:, :prefill_len],
            chunk_size=64,
        )
        decode_outs = []
        for t in range(prefill_len, total_len):
            out_t, state = gdr_pallas(
                q[:, t : t + 1],
                k[:, t : t + 1],
                v[:, t : t + 1],
                beta[:, t : t + 1],
                decay[:, t : t + 1],
                initial_state=state,
            )
            decode_outs.append(out_t)

        decode_cat = jnp.concatenate(decode_outs, axis=1)
        expected = out_full[:, prefill_len:]
        assert jnp.allclose(decode_cat, expected, atol=5e-2, rtol=0), (
            f"Prefill+decode vs full recurrent max diff: {jnp.max(jnp.abs(decode_cat - expected))}"
        )


class TestRecurrentFallback:
    """Test _recurrent_gdr_fwd (XLA fallback via Pallas interface)."""

    def test_recurrent_fallback_basic(self):
        q, k, v, beta, decay = _make(1, 64, **QWEN3_NEXT_SMALL_DIMS, seed=40)
        out, state = gdr_pallas(q, k, v, beta, decay, use_chunked=False)
        assert out.shape == v.shape
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    def test_recurrent_fallback_full_heads(self):
        q, k, v, beta, decay = _make(1, 32, **QWEN3_NEXT_DIMS, seed=41)
        out, state = gdr_pallas(q, k, v, beta, decay, use_chunked=False)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))
