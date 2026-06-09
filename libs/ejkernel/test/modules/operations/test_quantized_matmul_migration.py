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

import contextlib
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

qmm_module = importlib.import_module("ejkernel.modules.operations.quantized_matmul")
from ejkernel.modules.operations import quantized_matmul  # noqa

from ejkernel.quantization import dequantize, prepack_quantized_weights, quantize  # noqa

from ejkernel.quantization._utils import select_qmm_kernel_family  # noqa


def test_explicit_non_affine_mode_ignores_bits_argument():
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (64, 64), dtype=jnp.float32)

    w_q_default, scales_default = quantize(w, mode="mxfp8")
    w_q_with_bits, scales_with_bits = quantize(w, mode="mxfp8", bits=4)

    np.testing.assert_array_equal(np.asarray(w_q_default), np.asarray(w_q_with_bits))
    np.testing.assert_array_equal(np.asarray(scales_default), np.asarray(scales_with_bits))


def test_nf4_bits_argument_is_ignored():
    key = jax.random.PRNGKey(5)
    w = jax.random.normal(key, (64, 64), dtype=jnp.float32)

    w_q_default, scales_default = quantize(w, mode="nf4")
    w_q_with_bits, scales_with_bits = quantize(w, mode="nf4", bits=8)

    np.testing.assert_array_equal(np.asarray(w_q_default), np.asarray(w_q_with_bits))
    np.testing.assert_array_equal(np.asarray(scales_default), np.asarray(scales_with_bits))


@pytest.mark.parametrize("mode", ["mxfp", "nvfp"])
def test_family_mode_names_are_rejected(mode: str):
    key = jax.random.PRNGKey(6)
    w = jax.random.normal(key, (16, 32), dtype=jnp.float32)
    with pytest.raises(ValueError, match="explicit mode names"):
        quantize(w, mode=mode, bits=4)


def test_affine_dequantize_uses_zeros_only():
    key = jax.random.PRNGKey(1)
    w = jax.random.normal(key, (32, 64), dtype=jnp.float32)

    w_q, scales, zeros = quantize(w, mode="affine", bits=4, group_size=32, axis="row")
    w_zero = dequantize(w_q, scales, zeros, mode="affine", bits=4, group_size=32, axis="row")

    with pytest.raises(ValueError, match="requires `zeros`"):
        dequantize(w_q, scales, None, mode="affine", bits=4, group_size=32, axis="row")
    with pytest.raises(TypeError):
        dequantize(  # type: ignore[call-arg]
            w_q,
            scales,
            None,
            biases=jnp.zeros_like(scales),
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
        )

    assert w_zero.shape == (w.shape[1], w.shape[0])


def test_axis_row_and_col_paths_match_reference():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(2), 2)
    x = jax.random.normal(key_x, (8, 64), dtype=jnp.float32)
    w = jax.random.normal(key_w, (64, 64), dtype=jnp.float32)

    w_q_row, scales_row, zeros_row = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
    )
    w_q_col, scales_col, zeros_col = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=32,
        axis="col",
    )

    y_row = quantized_matmul(
        x,
        w_q_row,
        scales_row,
        zeros_row,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
        platform="xla",
    )
    y_col = quantized_matmul(
        x,
        w_q_col,
        scales_col,
        zeros_col,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
        axis="col",
        platform="xla",
    )
    ref = x @ w.T

    np.testing.assert_allclose(np.asarray(y_row), np.asarray(ref), rtol=0.35, atol=2.5)
    np.testing.assert_allclose(np.asarray(y_col), np.asarray(ref), rtol=0.35, atol=2.5)


def test_quantized_matmul_affine_uses_zeros_only():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(12), 2)
    x = jax.random.normal(key_x, (4, 32), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 32), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
    )
    y_zero = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
        platform="xla",
    )
    with pytest.raises(ValueError, match="requires `zeros`"):
        quantized_matmul(
            x,
            w_q,
            scales,
            None,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            platform="xla",
        )
    with pytest.raises(TypeError):
        quantized_matmul(  # type: ignore[call-arg]
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            biases=jnp.zeros_like(scales),
            platform="xla",
        )

    assert y_zero.shape == (x.shape[0], w.shape[0])


def test_axis_col_cuda_request_does_not_fallback(monkeypatch: pytest.MonkeyPatch):
    def _fake_detect_platform(_algorithm, platform="auto", **_kwargs):
        if platform in ("auto", None):
            return qmm_module.Platform.CUDA
        return qmm_module.Platform(platform) if isinstance(platform, str) else platform

    monkeypatch.setattr(qmm_module, "detect_platform", _fake_detect_platform)
    captured: dict[str, object] = {}

    class _FakeExecutor:
        chooser = object()

        def __call__(self, _kernel, **kwargs):
            captured.update(kwargs)
            x = kwargs["x"]
            w = kwargs["w"]
            transpose = bool(kwargs["transpose"])
            m = int(x.shape[0])
            n = int(w.shape[0]) if transpose else int(kwargs["scales"].shape[1]) * int(kwargs["group_size"])
            return jnp.zeros((m, n), dtype=x.dtype)

    monkeypatch.setattr(qmm_module, "_quantized_matmul_executor", _FakeExecutor())
    monkeypatch.setattr(qmm_module, "policy_override", lambda *_a, **_k: contextlib.nullcontext())

    key_x, key_w = jax.random.split(jax.random.PRNGKey(3), 2)
    x = jax.random.normal(key_x, (4, 32), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 32), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=32,
        axis="col",
    )

    y = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
        axis="col",
        platform="auto",
    )

    assert y.shape == (x.shape[0], w.shape[0])
    assert captured.get("_resolved_platform") == "cuda"
    assert captured.get("platform") == "cuda"


def test_kernel_family_auto_policy_matches_spec():
    family, parts = select_qmm_kernel_family(
        m=128,
        mode="affine",
        bits=8,
        gemv_mode="auto",
        revsplit_k="auto",
        revsplit_k_parts=None,
    )
    assert family == "gemm"
    assert parts is None

    family, parts = select_qmm_kernel_family(
        m=8,
        mode="affine",
        bits=4,
        gemv_mode="auto",
        revsplit_k="auto",
        revsplit_k_parts=None,
    )
    assert family == "gemm_splitk"
    assert parts is None

    family, parts = select_qmm_kernel_family(
        m=1,
        mode="nf4",
        bits=4,
        gemv_mode="auto",
        revsplit_k="auto",
        revsplit_k_parts=None,
    )
    assert family == "gemv_revsplitk"
    assert parts == 2

    family, parts = select_qmm_kernel_family(
        m=1,
        mode="nvfp8",
        bits=8,
        gemv_mode="auto",
        revsplit_k="auto",
        revsplit_k_parts=None,
    )
    assert family == "gemv_splitk"
    assert parts is None

    family, parts = select_qmm_kernel_family(
        m=1,
        mode="mxfp4",
        bits=4,
        gemv_mode="auto",
        revsplit_k="auto",
        revsplit_k_parts=None,
    )
    assert family == "gemm_splitk"
    assert parts is None


def test_kernel_family_override_validation():
    with pytest.raises(ValueError, match="gemv_mode='on' requires M == 1"):
        select_qmm_kernel_family(
            m=8,
            mode="affine",
            bits=4,
            gemv_mode="on",
            revsplit_k="auto",
            revsplit_k_parts=None,
        )

    with pytest.raises(ValueError, match="effective 4-bit mode"):
        select_qmm_kernel_family(
            m=1,
            mode="nvfp8",
            bits=8,
            gemv_mode="auto",
            revsplit_k="on",
            revsplit_k_parts=2,
        )

    with pytest.raises(ValueError, match="revsplit_k_parts must be one of"):
        select_qmm_kernel_family(
            m=1,
            mode="nf4",
            bits=4,
            gemv_mode="auto",
            revsplit_k="auto",
            revsplit_k_parts=3,
        )

    with pytest.raises(ValueError, match="revsplit_k='on' requires revsplit_k_parts >= 2"):
        select_qmm_kernel_family(
            m=1,
            mode="affine",
            bits=4,
            gemv_mode="auto",
            revsplit_k="on",
            revsplit_k_parts=1,
        )


def test_kernel_family_auto_clamps_revsplit_parts_to_two():
    family, parts = select_qmm_kernel_family(
        m=1,
        mode="nf4",
        bits=4,
        gemv_mode="auto",
        revsplit_k="auto",
        revsplit_k_parts=1,
    )
    assert family == "gemv_revsplitk"
    assert parts == 2


def test_quantized_matmul_accepts_gemv_and_revsplit_kwargs():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(20), 2)
    x = jax.random.normal(key_x, (4, 32), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 32), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
    )

    y = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
        gemv_mode="off",
        revsplit_k="off",
        revsplit_k_parts=2,
        platform="xla",
    )
    assert y.shape == (x.shape[0], w.shape[0])


def test_quantized_matmul_validates_gemv_override():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(21), 2)
    x = jax.random.normal(key_x, (8, 32), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 32), dtype=jnp.float32)
    w_q, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=32,
        axis="row",
    )

    with pytest.raises(ValueError, match="gemv_mode='on' requires M == 1"):
        quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=32,
            axis="row",
            gemv_mode="on",
            platform="xla",
        )
