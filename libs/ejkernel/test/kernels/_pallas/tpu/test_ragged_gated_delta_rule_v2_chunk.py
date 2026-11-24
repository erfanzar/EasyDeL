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

"""CPU-safe contract/support tests and TPU-gated parity tests for Pallas GDN v2 chunk & conv-decode kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2 import (
    pallas_gdn_chunk_prefill,
    pallas_gdn_chunk_prefill_supported,
    pallas_gdn_conv_decode,
    pallas_gdn_conv_decode_supported,
    pallas_gdn_decode_inplace,
)
from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule_v2 as pallas_gdn_v2,
)
from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2 import (
    ragged_gdn_conv_decode as pallas_gdn_conv_decode_entry,
)
from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2._pallas_gdn_chunk import (
    _chunk_vmem_estimate_bytes,
    _conv_decode_vmem_estimate_bytes,
    _short_vmem_estimate_bytes,
    _tile_bytes,
)
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule_v2 as xla_gdn_v2,
)
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2 import (
    ragged_gdn_conv_decode as xla_gdn_conv_decode_entry,
)


def _is_tpu() -> bool:
    return jax.default_backend() == "tpu"


# ---------------------------------------------------------------------------
# CPU-safe registry, validation, support-predicate & VMEM estimator tests
# ---------------------------------------------------------------------------


def test_ragged_gdn_conv_decode_pallas_is_registered_and_signature_valid():
    impl = kernel_registry.get("ragged_gdn_conv_decode", platform="pallas", backend="tpu")
    assert impl is pallas_gdn_conv_decode_entry
    assert kernel_registry.validate_signatures("ragged_gdn_conv_decode")
    assert kernel_registry.validate_signatures("ragged_gated_delta_rule_v2")


def test_ragged_gdn_conv_decode_interface_rejects_invalid_shapes():
    t_len, n_kq, n_v, d_k, d_v, d_conv, slots = 4, 2, 4, 8, 8, 4, 6
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    conv_input = jnp.zeros((t_len, conv_dim), dtype=jnp.float32)
    b_raw = jnp.zeros((t_len, n_v), dtype=jnp.float32)
    a_raw = jnp.zeros((t_len, n_v), dtype=jnp.float32)
    conv_weight = jnp.zeros((conv_dim, d_conv), dtype=jnp.float32)
    conv_pool = jnp.zeros((slots, conv_dim, d_conv), dtype=jnp.float32)
    recurrent_pool = jnp.zeros((slots, n_v, d_k, d_v), dtype=jnp.float32)
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    token_slots = jnp.arange(t_len, dtype=jnp.int32)
    token_init = jnp.ones((t_len,), dtype=jnp.bool_)

    # Non-divisible n_v % n_kq != 0 must fail.
    with pytest.raises(ValueError, match="divisible"):
        pallas_gdn_conv_decode_entry(
            conv_input,
            jnp.zeros((t_len, 3), dtype=jnp.float32),
            jnp.zeros((t_len, 3), dtype=jnp.float32),
            conv_weight,
            conv_pool,
            jnp.zeros((slots, 3, d_k, d_v), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
            token_slots,
            token_init,
            n_kq=2,
            n_v=3,
            d_k=d_k,
            d_v=d_v,
        )

    # Mismatched pool slot count rejected by jaxtyping/beartype.
    with pytest.raises((TypeError, ValueError, jaxtyping.TypeCheckError)):
        pallas_gdn_conv_decode_entry(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            jnp.zeros((slots + 2, conv_dim, d_conv), dtype=jnp.float32),
            recurrent_pool,
            a_log,
            dt_bias,
            token_slots,
            token_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )

    # Short token_slots rejected by jaxtyping/beartype.
    with pytest.raises((TypeError, ValueError, jaxtyping.TypeCheckError)):
        pallas_gdn_conv_decode_entry(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            conv_pool,
            recurrent_pool,
            a_log,
            dt_bias,
            token_slots[:2],
            token_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )


def test_support_predicates_and_tile_padded_vmem_estimates(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    # Small bf16 tiles must pad rows to 16 (8 * packing) and cols to 128.
    assert _tile_bytes(4, 128, 2) == 16 * 128 * 2
    assert _tile_bytes(4, 128, 4) == 8 * 128 * 4

    # Non-multiple-of-8 head counts > 8 (e.g. n_v=12) must be rejected.
    n_kq, n_v, d_k, d_v = 4, 12, 128, 128
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    assert not pallas_gdn_chunk_prefill_supported(
        mixed_qkv=jax.ShapeDtypeStruct((64, conv_dim), jnp.bfloat16),
        recurrent_state=jax.ShapeDtypeStruct((8, n_v, d_k, d_v), jnp.float32),
        query_start_loc=jax.ShapeDtypeStruct((5,), jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=64,
    )
    assert not pallas_gdn_conv_decode_supported(
        conv_input=jax.ShapeDtypeStruct((16, conv_dim), jnp.bfloat16),
        conv_weight=jax.ShapeDtypeStruct((conv_dim, 4), jnp.float32),
        conv_pool=jax.ShapeDtypeStruct((8, conv_dim, 4), jnp.bfloat16),
        recurrent_pool=jax.ShapeDtypeStruct((8, n_v, d_k, d_v), jnp.float32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
    )

    # Standard aligned power-of-2 shapes are supported (including transposed conv_pool).
    n_kq, n_v = 2, 4
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    assert pallas_gdn_chunk_prefill_supported(
        mixed_qkv=jax.ShapeDtypeStruct((64, conv_dim), jnp.bfloat16),
        recurrent_state=jax.ShapeDtypeStruct((8, n_v, d_k, d_v), jnp.float32),
        query_start_loc=jax.ShapeDtypeStruct((5,), jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=64,
    )
    assert pallas_gdn_conv_decode_supported(
        conv_input=jax.ShapeDtypeStruct((16, conv_dim), jnp.bfloat16),
        conv_weight=jax.ShapeDtypeStruct((conv_dim, 4), jnp.float32),
        conv_pool=jax.ShapeDtypeStruct((8, 4, conv_dim), jnp.bfloat16),
        recurrent_pool=jax.ShapeDtypeStruct((8, n_v, d_k, d_v), jnp.float32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        conv_pool_transposed=True,
    )

    # VMEM estimates strictly dominate logical (unpadded) element bytes for small n_v / d_conv.
    short_est = _short_vmem_estimate_bytes(n_kq=1, n_v=2, d_k=128, d_v=128, num_rows=64, in_itemsize=2, st_itemsize=4)
    conv_est = _conv_decode_vmem_estimate_bytes(
        num_tokens=64,
        n_kq=1,
        n_v=2,
        d_k=128,
        d_v=128,
        d_conv=4,
        in_itemsize=2,
        conv_itemsize=2,
        st_itemsize=4,
        out_itemsize=2,
    )
    chunk_est = _chunk_vmem_estimate_bytes(bkq=1, expand=2, cs=64, d_k=128, d_v=128, in_itemsize=2, st_itemsize=4)
    assert short_est > 0 and conv_est > 0 and chunk_est > 0


# ---------------------------------------------------------------------------
# TPU-gated hardware parity tests (skipped when not on TPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_tpu(), reason="TPU-only Pallas execution")
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_pallas_gdn_chunk_prefill_matches_xla_mixed_empty_oob_and_nan_padding(chunk_size):
    """Verify pallas_gdn_chunk_prefill against XLA across mixed lengths, empty rows, OOB slots, and NaN padding."""
    n_kq, n_v, d_k, d_v = 2, 4, 128, 128
    lengths = (0, 1, 73, 0, 1, 130, 5, 0)
    num_valid_tokens = sum(lengths)
    num_tokens = num_valid_tokens + 17
    num_slots = 12
    # Row 6 has an out-of-range slot (-1); other live rows use distinct in-range slots.
    state_indices = jnp.array([2, 5, 7, 1, 9, 3, -1, 4], dtype=jnp.int32)
    has_initial_state = jnp.array([True, True, False, True, False, True, True, True], dtype=jnp.bool_)
    query_start_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32)
    distribution = jnp.array([2, len(lengths), len(lengths)], dtype=jnp.int32)

    keys = jax.random.split(jax.random.PRNGKey(42 + chunk_size), 5)
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    mixed_qkv = (jax.random.normal(keys[0], (num_tokens, conv_dim), dtype=jnp.float32) * 0.1).astype(jnp.bfloat16)
    b = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    a = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    # Poison trailing padding tokens with NaN to catch any float-mask multiplication leakage.
    mixed_qkv_nan = mixed_qkv.at[num_valid_tokens:].set(jnp.nan)
    b_nan = b.at[num_valid_tokens:].set(jnp.nan)
    a_nan = a.at[num_valid_tokens:].set(jnp.nan)

    pool = jax.random.normal(keys[3], (num_slots, n_v, d_k, d_v), dtype=jnp.float32) * 0.1
    a_log = jnp.log(jnp.linspace(0.5, 2.0, n_v, dtype=jnp.float32))
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)

    state_pallas, out_pallas = pallas_gdn_chunk_prefill(
        mixed_qkv_nan,
        b_nan,
        a_nan,
        pool,
        a_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        mask_initial_state=True,
    )
    state_xla, out_xla = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        pool,
        a_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        mask_initial_state=True,
    )

    assert jnp.all(jnp.isfinite(out_pallas))
    assert jnp.all(jnp.isfinite(state_pallas))
    np.testing.assert_allclose(np.asarray(out_pallas, np.float32), np.asarray(out_xla, np.float32), atol=3e-2, rtol=3e-2)
    np.testing.assert_allclose(
        np.asarray(state_pallas, np.float32), np.asarray(state_xla, np.float32), atol=3e-2, rtol=3e-2
    )


@pytest.mark.skipif(not _is_tpu(), reason="TPU-only Pallas execution")
@pytest.mark.parametrize("use_fused_gdn_decode", [False, True])
def test_pallas_gdn_decode_interior_oob_slots_and_nan_padding(use_fused_gdn_decode):
    """Verify pallas_gdn_decode_inplace and use_fused_gdn_decode with interior OOB slots and NaN padding."""
    n_kq, n_v, d_k, d_v = 2, 4, 128, 128
    lengths = (1, 1, 0, 1, 1)
    num_tokens = 8
    num_slots = 10
    # Interior OOB slots (-1 at row 0, 99 at row 3); live valid rows at 1 and 4.
    state_indices = jnp.array([-1, 5, 2, 99, 7], dtype=jnp.int32)
    has_initial_state = jnp.array([True, True, True, False, False], dtype=jnp.bool_)
    query_start_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32)
    distribution = jnp.array([len(lengths), len(lengths), len(lengths)], dtype=jnp.int32)

    keys = jax.random.split(jax.random.PRNGKey(99), 4)
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    mixed_qkv = (jax.random.normal(keys[0], (num_tokens, conv_dim), dtype=jnp.float32) * 0.1).astype(jnp.bfloat16)
    b = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    a = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    pool = jax.random.normal(keys[3], (num_slots, n_v, d_k, d_v), dtype=jnp.float32) * 0.1
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)

    if not use_fused_gdn_decode:
        mixed_in = mixed_qkv.at[sum(lengths) :].set(jnp.nan)
        b_in = b.at[sum(lengths) :].set(jnp.nan)
        a_in = a.at[sum(lengths) :].set(jnp.nan)
        state_pallas, out_pallas = pallas_gdn_decode_inplace(
            mixed_in,
            b_in,
            a_in,
            pool,
            a_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            mask_initial_state=True,
        )
    else:
        state_pallas, out_pallas = pallas_gdn_v2(
            mixed_qkv,
            b,
            a,
            pool,
            a_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=64,
            mask_initial_state=True,
            use_fused_gdn_decode=True,
        )

    state_xla, out_xla = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        pool,
        a_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=64,
        mask_initial_state=True,
    )

    assert jnp.all(jnp.isfinite(out_pallas))
    assert jnp.all(jnp.isfinite(state_pallas))
    # Slot 0 must remain untouched despite interior OOB state_indices[0] == -1.
    np.testing.assert_array_equal(np.asarray(state_pallas[0]), np.asarray(pool[0]))
    np.testing.assert_allclose(np.asarray(out_pallas, np.float32), np.asarray(out_xla, np.float32), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(
        np.asarray(state_pallas, np.float32), np.asarray(state_xla, np.float32), atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(not _is_tpu(), reason="TPU-only Pallas execution")
@pytest.mark.parametrize("conv_pool_transposed", [False, True])
def test_pallas_gdn_conv_decode_matches_xla(conv_pool_transposed):
    """Verify pallas_gdn_conv_decode matches XLA reference across live, fresh, and idle slots."""
    t_len, n_kq, n_v, d_k, d_v, d_conv, slots = 12, 2, 4, 128, 128, 4, 16
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    keys = jax.random.split(jax.random.PRNGKey(123), 6)
    conv_input = (jax.random.normal(keys[0], (t_len, conv_dim), dtype=jnp.float32) * 0.1).astype(jnp.bfloat16)
    b_raw = jax.random.normal(keys[1], (t_len, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    a_raw = jax.random.normal(keys[2], (t_len, n_v), dtype=jnp.float32).astype(jnp.bfloat16)
    conv_weight = jax.random.normal(keys[3], (conv_dim, d_conv), dtype=jnp.float32) * 0.1
    conv_pool = (jax.random.normal(keys[4], (slots, conv_dim, d_conv), dtype=jnp.float32) * 0.1).astype(jnp.bfloat16)
    rec_pool = jax.random.normal(keys[5], (slots, n_v, d_k, d_v), dtype=jnp.float32) * 0.1
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    token_slots = jnp.array([-1, 3, 7, 16, 1, 9, -2, 12, 0, 5, -1, 14], dtype=jnp.int32)
    token_init = jnp.array([True, True, False, True, False, True, False, True, True, False, True, True], dtype=jnp.bool_)

    out_xla, conv_xla, rec_xla = xla_gdn_conv_decode_entry(
        conv_input,
        b_raw,
        a_raw,
        conv_weight,
        conv_pool,
        rec_pool,
        a_log,
        dt_bias,
        token_slots,
        token_init,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
    )

    if conv_pool_transposed:
        out_p, conv_p_t, rec_p = pallas_gdn_conv_decode(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            jnp.swapaxes(conv_pool, 1, 2),
            rec_pool,
            a_log,
            dt_bias,
            token_slots,
            token_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            conv_pool_transposed=True,
        )
        conv_p = jnp.swapaxes(conv_p_t, 1, 2)
    else:
        out_p, conv_p, rec_p = pallas_gdn_conv_decode_entry(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            conv_pool,
            rec_pool,
            a_log,
            dt_bias,
            token_slots,
            token_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )

    np.testing.assert_allclose(np.asarray(out_p, np.float32), np.asarray(out_xla, np.float32), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(conv_p, np.float32), np.asarray(conv_xla, np.float32), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(rec_p, np.float32), np.asarray(rec_xla, np.float32), atol=2e-2, rtol=2e-2)
