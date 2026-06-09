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

# ruff: noqa: E402

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_has_cutlass = importlib.util.find_spec("cutlass") is not None
if not _has_cutlass:
    pytest.skip("CUTLASS CuTe DSL not installed", allow_module_level=True)
if jax.devices()[0].platform != "gpu":
    pytest.skip("CUTE tests require GPU backend", allow_module_level=True)

from ejkernel.callib._cute_ffi import has_cute_ffi_support
from ejkernel.kernels._cute.quantized_matmul import quantized_matmul as cute_quantized_matmul
from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

cute_qmm_bwd = importlib.import_module("ejkernel.kernels._cute.quantized_matmul._cute_impl_bwd")

if not has_cute_ffi_support():
    pytest.skip("CuTe primitive support is required for CuTe kernel tests", allow_module_level=True)


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 1),
        ("affine", 2),
        ("affine", 3),
        ("affine", 4),
        ("affine", 5),
        ("affine", 6),
        ("affine", 7),
        ("affine", 8),
        ("nf4", 4),
        ("mxfp4", 4),
        ("mxfp8", 8),
        ("nvfp4", 4),
        ("nvfp8", 8),
    ],
)
def test_quantized_matmul_cute_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(0 if mode == "affine" else 1)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    out_cute = cute_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)

    rtol = 1.5e-1 if mode == "affine" else 6e-2
    atol = 1.5e-1 if mode == "affine" else 6e-2
    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


def test_quantized_matmul_cute_transpose_matches_xla():
    key = jax.random.PRNGKey(42)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 8, 32, 32

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, transpose=False, group_size=32)

    dev = jax.devices("gpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    zeros = jax.device_put(zeros, dev)

    out_cute = cute_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


def test_quantized_matmul_cute_transpose_cache_key_regression():
    key = jax.random.PRNGKey(1234)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 8, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    w_q_nt, scales_nt, biases_nt = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        transpose=True,
        group_size=32,
    )
    w_q_t, scales_t, biases_t = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        transpose=False,
        group_size=32,
    )

    dev = jax.devices("gpu")[0]
    x = jax.device_put(x, dev)
    w_q_nt = jax.device_put(w_q_nt, dev)
    scales_nt = jax.device_put(scales_nt, dev)
    biases_nt = jax.device_put(biases_nt, dev)
    w_q_t = jax.device_put(w_q_t, dev)
    scales_t = jax.device_put(scales_t, dev)
    biases_t = jax.device_put(biases_t, dev)

    out_cute_nt = cute_quantized_matmul(
        x,
        w_q_nt,
        scales_nt,
        biases_nt,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=32,
    )
    out_cute_t = cute_quantized_matmul(
        x,
        w_q_t,
        scales_t,
        biases_t,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
    )
    out_xla_nt = xla_quantized_matmul(
        x,
        w_q_nt,
        scales_nt,
        biases_nt,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=32,
    )
    out_xla_t = xla_quantized_matmul(
        x,
        w_q_t,
        scales_t,
        biases_t,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
    )

    out_cute_nt = jax.block_until_ready(out_cute_nt)
    out_cute_t = jax.block_until_ready(out_cute_t)
    out_xla_nt = jax.block_until_ready(out_xla_nt)
    out_xla_t = jax.block_until_ready(out_xla_t)

    np.testing.assert_allclose(
        np.asarray(out_cute_nt, dtype=np.float32),
        np.asarray(out_xla_nt, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )
    np.testing.assert_allclose(
        np.asarray(out_cute_t, dtype=np.float32),
        np.asarray(out_xla_t, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 1),
        ("affine", 2),
        ("affine", 3),
        ("affine", 4),
        ("affine", 5),
        ("affine", 6),
        ("affine", 7),
        ("affine", 8),
        ("nf4", 4),
        ("mxfp4", 4),
        ("mxfp8", 8),
        ("nvfp4", 4),
        ("nvfp8", 8),
    ],
)
def test_quantized_matmul_cute_grad_input_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(202 if mode == "affine" else 303)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    def _loss_cute(x_in):
        y = cute_quantized_matmul(
            x_in,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    def _loss_xla(x_in):
        y = xla_quantized_matmul(
            x_in,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    g_cute = jax.block_until_ready(jax.grad(_loss_cute)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))

    np.testing.assert_allclose(
        np.asarray(g_cute, dtype=np.float32),
        np.asarray(g_xla, dtype=np.float32),
        rtol=7e-2,
        atol=7e-2,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 4),
        ("affine", 8),
        ("nf4", 4),
        ("mxfp4", 4),
        ("mxfp8", 8),
        ("nvfp4", 4),
        ("nvfp8", 8),
    ],
)
def test_quantized_matmul_cute_grad_input_same_kernel_path(monkeypatch: pytest.MonkeyPatch, mode: str, bits: int):
    key = jax.random.PRNGKey(404 if mode == "affine" else 505)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    def _forbidden_dequant(*args, **kwargs):
        raise AssertionError(f"Unexpected dequant fallback in CuTe grad path for mode={mode}.")

    monkeypatch.setattr(cute_qmm_bwd, "dequantize", _forbidden_dequant)

    def _loss(x_in):
        y = cute_quantized_matmul(
            x_in,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    gx = jax.block_until_ready(jax.grad(_loss)(x))
    assert gx.shape == (m, k)
