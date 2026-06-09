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

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.modules.operations import quantized_matmul
from ejkernel.quantization import (
    QuantizedArray,
    QuantRuntimeConfig,
    dense_quantized_matmul,
    dequantize,
    prepack_quantized_array,
    prepack_quantized_weights,
    quantize,
    quantize_array,
)
from ejkernel.quantization._utils.bitpack import _pack_bits, _unpack_bits
from ejkernel.quantization.runtime import resolve_runtime_config

qmm_op = importlib.import_module("ejkernel.modules.operations.quantized_matmul")


def _prng_key_or_skip(seed: int) -> jax.Array:
    """Build a PRNG key or skip when TPU runtime is unavailable in this process."""
    try:
        return jax.random.PRNGKey(seed)
    except RuntimeError as err:
        msg = str(err)
        if "Unable to initialize backend 'tpu'" in msg or "libtpu.so" in msg:
            pytest.skip("Skipping test: JAX TPU backend is unavailable in this process.")
        raise


@pytest.mark.parametrize(
    "mode,group_size,bits",
    [
        ("nf4", 64, 4),
        ("mxfp4", 32, 4),
        ("mxfp8", 32, 8),
        ("nvfp4", 16, 4),
        ("nvfp8", 16, 8),
    ],
)
def test_runtime_codebook_modes_match_reference_dequant(mode: str, group_size: int, bits: int):
    key = jax.random.PRNGKey(42)
    w = jax.random.normal(key, (64, 128), dtype=jnp.float32)

    cfg_new = QuantRuntimeConfig(enable_threshold_codebook=True, enable_parity_fallback=False)
    cfg_ref = QuantRuntimeConfig(enable_threshold_codebook=False, enable_parity_fallback=True)

    wq_new, s_new = quantize(w, mode=mode, group_size=group_size, bits=bits, axis="row", runtime_config=cfg_new)
    wq_ref, s_ref = quantize(w, mode=mode, group_size=group_size, bits=bits, axis="row", runtime_config=cfg_ref)

    dq_new = dequantize(wq_new, s_new, None, mode=mode, group_size=group_size, bits=bits, axis="row")
    dq_ref = dequantize(wq_ref, s_ref, None, mode=mode, group_size=group_size, bits=bits, axis="row")

    np.testing.assert_allclose(np.asarray(dq_new), np.asarray(dq_ref), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "mode,group_size,bits",
    [
        ("affine", 64, 1),
        ("affine", 64, 2),
        ("affine", 64, 4),
        ("nf4", 64, 4),
        ("mxfp4", 32, 4),
        ("mxfp8", 32, 8),
        ("nvfp4", 16, 4),
        ("nvfp8", 16, 8),
    ],
)
def test_zero_tensor_quant_dequant_is_finite(mode: str, group_size: int, bits: int):
    w = jnp.zeros((64, 128), dtype=jnp.float32)
    out = quantize(w, mode=mode, group_size=group_size, bits=bits, axis="row")

    if mode == "affine":
        w_q, scales, zeros = out
        w_dq = dequantize(w_q, scales, zeros, mode=mode, group_size=group_size, bits=bits, axis="row")
    else:
        w_q, scales = out
        w_dq = dequantize(w_q, scales, None, mode=mode, group_size=group_size, bits=bits, axis="row")

    assert bool(jnp.all(jnp.isfinite(w_dq)))


@pytest.mark.parametrize(
    "meta_mode,expected_dtype",
    [
        ("fp16", jnp.float16),
        ("bf16", jnp.bfloat16),
        ("fp32", jnp.float32),
    ],
)
def test_affine_metadata_dtype_control(meta_mode: str, expected_dtype: jnp.dtype):
    key = jax.random.PRNGKey(123)
    w = jax.random.normal(key, (64, 128), dtype=jnp.float32)
    cfg = QuantRuntimeConfig(affine_metadata_dtype=meta_mode)
    w_q, scales, zeros = quantize(
        w,
        mode="affine",
        bits=4,
        group_size=64,
        axis="row",
        runtime_config=cfg,
    )
    assert scales.dtype == expected_dtype
    assert zeros.dtype == expected_dtype

    out = dequantize(
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=64,
        axis="row",
        runtime_config=cfg,
    )
    assert out.dtype == jnp.float32


@pytest.mark.parametrize(
    "out_mode,expected_dtype",
    [
        ("fp16", jnp.float16),
        ("bf16", jnp.bfloat16),
        ("fp32", jnp.float32),
        ("compute", jnp.float16),
    ],
)
def test_dequant_output_dtype_control(out_mode: str, expected_dtype: jnp.dtype):
    key = jax.random.PRNGKey(331)
    w = jax.random.normal(key, (64, 128), dtype=jnp.float32)
    q_cfg = QuantRuntimeConfig(affine_metadata_dtype="fp16")
    w_q, scales, zeros = quantize(
        w,
        mode="affine",
        bits=4,
        group_size=64,
        axis="row",
        runtime_config=q_cfg,
    )
    dq_cfg = QuantRuntimeConfig(
        prefer_compute_dtype="fp16",
        dequant_output_dtype=out_mode,
    )
    out = dequantize(
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=64,
        axis="row",
        runtime_config=dq_cfg,
    )
    assert out.dtype == expected_dtype


def test_dequant_unpack_policy_parity():
    key = jax.random.PRNGKey(777)
    w = jax.random.normal(key, (64, 128), dtype=jnp.float32)
    w_q, scales = quantize(
        w,
        mode="nf4",
        bits=4,
        group_size=64,
        axis="row",
        runtime_config=QuantRuntimeConfig(enable_threshold_codebook=True),
    )
    out_fast = dequantize(
        w_q,
        scales,
        None,
        mode="nf4",
        bits=4,
        group_size=64,
        axis="row",
        runtime_config=QuantRuntimeConfig(dequant_unpack_policy="fast"),
    )
    out_generic = dequantize(
        w_q,
        scales,
        None,
        mode="nf4",
        bits=4,
        group_size=64,
        axis="row",
        runtime_config=QuantRuntimeConfig(dequant_unpack_policy="generic"),
    )
    np.testing.assert_allclose(np.asarray(out_fast), np.asarray(out_generic), rtol=1e-6, atol=1e-6)


def test_runtime_config_none_uses_fastest_profile():
    assert resolve_runtime_config(None) == QuantRuntimeConfig.fastest_for_backend()


def test_dense_quantized_matmul_alignment_padding_equivalence():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(909), 2)
    x = jax.random.normal(key_x, (7, 192), dtype=jnp.float32)
    # N=130 intentionally misaligned to test output-dimension padding/slicing.
    w = jax.random.normal(key_w, (130, 192), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=64,
        axis="col",
    )
    y_no_pad = dense_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=64,
        axis="col",
        align_multiple=0,
    )
    y_pad = dense_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=64,
        axis="col",
        align_multiple=128,
    )
    np.testing.assert_allclose(np.asarray(y_pad), np.asarray(y_no_pad), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "bits,values_per_word,mask",
    [
        (1, 32, 0x1),
        (2, 16, 0x3),
        (4, 8, 0xF),
        (8, 4, 0xFF),
    ],
)
def test_bitpack_alignment_and_padding_behavior(bits: int, values_per_word: int, mask: int):
    values = (jnp.arange(20, dtype=jnp.uint32).reshape(2, 10) & jnp.uint32(mask)).astype(jnp.uint32)

    packed = _pack_bits(values, bits, prefer_fast_u4_u8=True, strict_shape_alignment=False)
    unpacked = _unpack_bits(packed, values.shape[-1], bits, prefer_fast_u4_u8=True)
    np.testing.assert_array_equal(np.asarray(values), np.asarray(unpacked))

    with pytest.raises(ValueError, match=f"multiple of {values_per_word}"):
        _pack_bits(values, bits, prefer_fast_u4_u8=True, strict_shape_alignment=True)


def test_quantized_array_is_jittable_pytree():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(2026), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float16)
    w = jax.random.normal(key_w, (128, 64), dtype=jnp.float16)

    wq = prepack_quantized_array(w, mode="affine", bits=4, group_size=64, transpose=True)

    fn = jax.jit(lambda xi, wi: wi.matmul(xi, fuse=False))
    y = fn(x, wq)
    assert y.shape == (8, 128)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_quantized_array_roundtrip_and_constructor_validation():
    key = jax.random.PRNGKey(7)
    w = jax.random.normal(key, (64, 128), dtype=jnp.float32)

    q_aff = quantize_array(w, mode="affine", bits=4, group_size=32, axis="row")
    assert isinstance(q_aff, QuantizedArray)
    assert q_aff.zeros is not None
    assert len(q_aff.as_tuple()) == 3
    dq = q_aff.dequantize()
    assert dq.shape == (w.shape[1], w.shape[0])
    assert bool(jnp.all(jnp.isfinite(dq)))

    q_nf4 = quantize_array(w, mode="nf4", axis="row")
    assert q_nf4.zeros is None
    assert len(q_nf4.as_tuple()) == 2

    q_aff2 = quantize_array(w, mode="affine", bits=2, group_size=64, axis="row")
    assert q_aff2.bits == 2
    assert q_aff2.actual_bits_per_value > 2.0

    with pytest.raises(ValueError, match="requires `zeros`"):
        QuantizedArray.from_quantized(
            q_aff.data,
            q_aff.scales,
            None,
            mode="affine",
            bits=q_aff.bits,
            group_size=q_aff.group_size,
            axis=q_aff.axis,
        )

    with pytest.raises(ValueError, match="zeros must be None"):
        QuantizedArray.from_quantized(
            q_nf4.data,
            q_nf4.scales,
            jnp.zeros_like(q_nf4.scales),
            mode="nf4",
            bits=q_nf4.bits,
            group_size=q_nf4.group_size,
            axis=q_nf4.axis,
        )


def test_quantized_matmul_fuse_false_matches_dense_reference():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(11), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=32, axis="row")

    y_fallback = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
        platform="xla",
        fuse=False,
    )
    y_dense = dense_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
    )
    np.testing.assert_allclose(np.asarray(y_fallback), np.asarray(y_dense), rtol=1e-6, atol=1e-6)


def test_quantized_matmul_fuse_true_runtime_behavior():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(12), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=32, axis="row")

    y_fallback = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
        platform="xla",
        fuse=False,
    )

    if jax.default_backend() == "mps":
        with pytest.warns(RuntimeWarning, match="falls back to reference"):
            y_fused = quantized_matmul(
                x,
                w_q,
                scales,
                zeros,
                mode="affine",
                bits=4,
                group_size=32,
                axis="row",
                platform="xla",
                fuse=True,
            )
        np.testing.assert_allclose(np.asarray(y_fused), np.asarray(y_fallback), rtol=1e-6, atol=1e-6)
    else:
        y_fused = quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            platform="xla",
            fuse=True,
        )
        assert y_fused.shape == y_fallback.shape
        assert bool(jnp.all(jnp.isfinite(y_fused)))
        # Fused kernels dequantize and accumulate in reduced precision.
        # Keep this as a behavioral guard against large drift, not exact parity.
        np.testing.assert_allclose(np.asarray(y_fused), np.asarray(y_fallback), rtol=2e-2, atol=7e-2)


def test_quantized_matmul_strict_fuse_conflicts_with_allow_dense_fallback():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(121), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=32, axis="row")

    with pytest.raises(ValueError, match="allow_dense_fallback=True is incompatible"):
        quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            platform="xla",
            fuse=True,
            strict_fuse=True,
            allow_dense_fallback=True,
        )


def test_quantized_matmul_explicit_tpu_controls_no_env(monkeypatch):
    key_x, key_w = jax.random.split(jax.random.PRNGKey(122), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=32, axis="row")

    captured: dict[str, object] = {}
    original_impl = qmm_op._quantized_matmul_impl

    def _fake_impl(xi, wi, si, zi, /, **kwargs):
        captured["allow_dense_fallback"] = kwargs["allow_dense_fallback"]
        captured["cfg"] = kwargs.get("cfg")
        n = int(si.shape[-1]) * int(kwargs["group_size"])
        return jnp.zeros((int(xi.shape[0]), n), dtype=jnp.float32)

    monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", _fake_impl)
    try:
        y = qmm_op.quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            platform="pallas",
            fuse=True,
            strict_fuse=False,
            allow_dense_fallback=False,
            tpu_path="packed",
        )
    finally:
        monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", original_impl)

    assert y.shape == (8, 32)
    assert captured["allow_dense_fallback"] is False
    cfg = captured["cfg"]
    assert cfg is not None
    assert cfg.tpu_path == "packed"


def test_quantized_matmul_use_best_config_populates_missing_controls(monkeypatch):
    key_x, key_w = jax.random.split(_prng_key_or_skip(1001), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=32, axis="row")

    captured: dict[str, object] = {}
    original_impl = qmm_op._quantized_matmul_impl

    def _fake_policy(**_kwargs):
        return {
            "fuse": True,
            "strict_fuse": False,
            "allow_dense_fallback": False,
            "platform": "pallas",
            "tpu_path": "packed",
        }

    def _fake_impl(xi, wi, si, zi, /, **kwargs):
        captured["allow_dense_fallback"] = kwargs["allow_dense_fallback"]
        captured["platform"] = kwargs["platform"]
        captured["cfg"] = kwargs.get("cfg")
        n = int(si.shape[-1]) * int(kwargs["group_size"])
        return jnp.zeros((int(xi.shape[0]), n), dtype=jnp.float32)

    monkeypatch.setattr(qmm_op, "_lookup_best_qmm_policy", _fake_policy)
    monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", _fake_impl)
    try:
        y = qmm_op.quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            platform="auto",
            fuse=True,
            use_best_config=True,
        )
    finally:
        monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", original_impl)

    assert y.shape == (8, 32)
    assert captured["allow_dense_fallback"] is False
    assert captured["platform"] == "pallas"
    cfg = captured["cfg"]
    assert cfg is not None
    assert cfg.tpu_path == "packed"


def test_quantized_matmul_use_best_config_respects_explicit_controls(monkeypatch):
    key_x, key_w = jax.random.split(_prng_key_or_skip(1002), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=32, axis="row")

    captured: dict[str, object] = {}
    original_impl = qmm_op._quantized_matmul_impl

    def _fake_policy(**_kwargs):
        return {
            "fuse": True,
            "strict_fuse": False,
            "allow_dense_fallback": True,
            "platform": "pallas",
            "tpu_path": "packed",
        }

    def _fake_impl(xi, wi, si, zi, /, **kwargs):
        captured["allow_dense_fallback"] = kwargs["allow_dense_fallback"]
        captured["platform"] = kwargs["platform"]
        captured["cfg"] = kwargs.get("cfg")
        n = int(si.shape[-1]) * int(kwargs["group_size"])
        return jnp.zeros((int(xi.shape[0]), n), dtype=jnp.float32)

    monkeypatch.setattr(qmm_op, "_lookup_best_qmm_policy", _fake_policy)
    monkeypatch.setattr(qmm_op, "_should_try_tpu_predecode_once_default", lambda **_kwargs: False)
    monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", _fake_impl)
    try:
        y = qmm_op.quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            platform="xla",
            fuse=True,
            strict_fuse=True,
            allow_dense_fallback=False,
            tpu_path="predecode",
            use_best_config=True,
        )
    finally:
        monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", original_impl)

    assert y.shape == (8, 32)
    assert captured["allow_dense_fallback"] is False
    assert captured["platform"] == "xla"
    assert captured["cfg"] is None


def test_quantized_matmul_use_best_config_can_disable_fuse(monkeypatch):
    key_x, key_w = jax.random.split(_prng_key_or_skip(1003), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=32, axis="row")

    original_impl = qmm_op._quantized_matmul_impl

    def _fake_policy(**_kwargs):
        return {"fuse": False, "platform": "xla"}

    def _fail_impl(*_args, **_kwargs):
        raise AssertionError("_quantized_matmul_impl should not be called when policy disables fuse.")

    monkeypatch.setattr(qmm_op, "_lookup_best_qmm_policy", _fake_policy)
    monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", _fail_impl)
    try:
        y_auto = qmm_op.quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            platform="xla",
            fuse=True,
            use_best_config=True,
        )
    finally:
        monkeypatch.setattr(qmm_op, "_quantized_matmul_impl", original_impl)

    y_ref = qmm_op.quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
        platform="xla",
        fuse=False,
        use_best_config=False,
    )
    np.testing.assert_allclose(np.asarray(y_auto), np.asarray(y_ref), rtol=1e-6, atol=1e-6)


def test_lookup_best_qmm_policy_non_affine_concrete_prefers_packed_pallas():
    policy = qmm_op._lookup_best_qmm_policy(
        backend_name="tpu",
        mode="mxfp8",
        m_tokens=16,
        runtime_axis="row",
        runtime_transpose=False,
        weights_concrete=True,
    )
    assert policy["fuse"] is True
    assert policy["platform"] == "pallas"
    assert policy["tpu_path"] == "packed"


def test_lookup_best_qmm_policy_non_affine_traced_prefers_packed_pallas():
    """Non-affine traced weights should use fused Pallas packed kernel.

    The Pallas packed kernel is trace-safe: legality checks use only
    .shape (concrete during tracing), and in-kernel dequant is pure
    arithmetic (jnp.where chains, bit ops) with no table lookups.
    """
    policy = qmm_op._lookup_best_qmm_policy(
        backend_name="tpu",
        mode="mxfp8",
        m_tokens=16,
        runtime_axis="row",
        runtime_transpose=False,
        weights_concrete=False,
    )
    assert policy["fuse"] is True
    assert policy["platform"] == "pallas"
    assert policy["tpu_path"] == "packed"


def test_lookup_best_qmm_policy_nf4_concrete_small_prefers_packed_pallas():
    policy = qmm_op._lookup_best_qmm_policy(
        backend_name="tpu",
        mode="nf4",
        m_tokens=16,
        runtime_axis="row",
        runtime_transpose=False,
        weights_concrete=True,
    )
    assert policy["fuse"] is True
    assert policy["platform"] == "pallas"
    assert policy["tpu_path"] == "packed"


def test_lookup_best_qmm_policy_nf4_concrete_large_prefers_packed_pallas():
    policy = qmm_op._lookup_best_qmm_policy(
        backend_name="tpu",
        mode="nf4",
        m_tokens=2048,
        runtime_axis="row",
        runtime_transpose=False,
        weights_concrete=True,
    )
    assert policy["fuse"] is True
    assert policy["platform"] == "pallas"
    assert policy["tpu_path"] == "packed"


def test_quantized_array_matmul_fuse_switch():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(13), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    q = prepack_quantized_array(w, mode="affine", bits=4, group_size=32, axis="row")

    y_ref = q.matmul(x, axis="row", fuse=False)
    if jax.default_backend() == "mps":
        with pytest.warns(RuntimeWarning, match="falls back to reference"):
            y_fused = q.matmul(x, axis="row", fuse=True, platform="xla")
        np.testing.assert_allclose(np.asarray(y_fused), np.asarray(y_ref), rtol=1e-6, atol=1e-6)
    else:
        y_fused = q.matmul(x, axis="row", fuse=True, platform="xla")
        assert y_fused.shape == y_ref.shape
        assert bool(jnp.all(jnp.isfinite(y_fused)))


def test_quantized_array_matmul_forwards_tpu_controls(monkeypatch):
    key_x, key_w = jax.random.split(jax.random.PRNGKey(123), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    q = prepack_quantized_array(w, mode="affine", bits=4, group_size=32, axis="row")

    captured: dict[str, object] = {}
    import ejkernel.modules.operations as ops

    original_fused = ops.quantized_matmul

    def _fake_fused(xi, wi, si, zi, /, **kwargs):
        captured.update(kwargs)
        n = int(si.shape[-1]) * int(kwargs["group_size"])
        return jnp.zeros((int(xi.shape[0]), n), dtype=jnp.float32)

    monkeypatch.setattr(ops, "quantized_matmul", _fake_fused)
    try:
        y = q.matmul(
            x,
            axis="row",
            fuse=True,
            platform="pallas",
            strict_fuse=False,
            allow_dense_fallback=False,
            tpu_path="packed",
        )
    finally:
        monkeypatch.setattr(ops, "quantized_matmul", original_fused)

    assert y.shape == (8, 32)
    assert captured["allow_dense_fallback"] is False
    assert captured["tpu_path"] == "packed"


@pytest.mark.parametrize("bits", [1, 2, 3, 5, 6, 7])
def test_quantized_matmul_affine_non_power_two_bits_match_reference(bits: int):
    key_x, key_w = jax.random.split(jax.random.PRNGKey(27 + bits), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 64), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=bits, group_size=32, axis="row")

    y_auto = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=bits,
        group_size=32,
        axis="row",
        platform="xla",
        fuse=True,
    )

    y_ref = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=bits,
        group_size=32,
        axis="row",
        platform="xla",
        fuse=False,
    )
    np.testing.assert_allclose(np.asarray(y_auto), np.asarray(y_ref), rtol=2e-2, atol=2e-2)
