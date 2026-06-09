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

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._triton.quantized_matmul import quantized_matmul as triton_quantized_matmul
from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

triton_qmm_bwd = importlib.import_module("ejkernel.kernels._triton.quantized_matmul._triton_impl_bwd")

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def _device_put_all(dev, *arrays):
    return [jax.device_put(arr, dev) for arr in arrays]


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
def test_quantized_matmul_triton_matches_xla(mode: str, bits: int):
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
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    out_triton = triton_quantized_matmul(
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

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    rtol = 1.5e-1 if mode == "affine" else 6e-2
    atol = 1.5e-1 if mode == "affine" else 6e-2
    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=rtol,
        atol=atol,
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
def test_quantized_matmul_triton_grad_input_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(11 if mode == "affine" else 13)
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
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    def _loss_triton(x_in):
        y = triton_quantized_matmul(
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

    g_triton = jax.block_until_ready(jax.grad(_loss_triton)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))

    rtol = 1.5e-1 if mode == "affine" else 7e-2
    atol = 1.5e-1 if mode == "affine" else 7e-2
    np.testing.assert_allclose(
        np.asarray(g_triton, dtype=np.float32),
        np.asarray(g_xla, dtype=np.float32),
        rtol=rtol,
        atol=atol,
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
def test_quantized_matmul_triton_grad_input_same_kernel_path(monkeypatch: pytest.MonkeyPatch, mode: str, bits: int):
    key = jax.random.PRNGKey(17 if mode == "affine" else 19)
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
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    def _forbidden_dequant(*args, **kwargs):
        raise AssertionError(f"Unexpected dequant fallback in Triton grad path for mode={mode}.")

    monkeypatch.setattr(triton_qmm_bwd, "quantized_matmul_dequant_triton", _forbidden_dequant)

    def _loss(x_in):
        y = triton_quantized_matmul(
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
        ("nvfp8", 8),
    ],
)
def test_quantized_matmul_triton_m1_gemv_paths_match_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(101 if mode == "affine" else 103)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 1, 256, 512

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)
    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    out_triton = triton_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        gemv_mode="auto",
        revsplit_k="auto",
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        gemv_mode="auto",
        revsplit_k="auto",
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    rtol = 1.5e-1 if mode == "affine" else 7e-2
    atol = 1.5e-1 if mode == "affine" else 7e-2
    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 4),
        ("nf4", 4),
        ("mxfp4", 4),
    ],
)
def test_quantized_matmul_triton_vmap_over_x_matches_xla(mode: str, bits: int):
    """vmap over x with shared weights — uses the reshape fast-path."""
    key = jax.random.PRNGKey(31)
    kx, kw = jax.random.split(key, 2)
    b, m, k, n = 3, 8, 64, 64

    x = jax.random.normal(kx, (b, m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    def _single_triton(xi):
        return triton_quantized_matmul(
            xi,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    def _single_xla(xi):
        return xla_quantized_matmul(
            xi,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    out_triton = jax.block_until_ready(jax.vmap(_single_triton)(x))
    out_xla = jax.block_until_ready(jax.vmap(_single_xla)(x))

    assert out_triton.shape == (b, m, n)
    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("mxfp4", 4),
        ("nf4", 4),
    ],
)
def test_quantized_matmul_triton_jit_vmap_over_all_inputs(mode: str, bits: int):
    """vmap over x, w, and scales — uses the lax.map fallback."""
    key = jax.random.PRNGKey(37)
    kx, kpack = jax.random.split(key, 2)
    b, m, k, n = 2, 8, 64, 64

    x = jax.random.normal(kx, (b, m, k), dtype=jnp.float16)
    pack_keys = jax.random.split(kpack, b)

    w_q_parts, scales_parts = [], []
    for pk in pack_keys:
        w = jax.random.normal(pk, (n, k), dtype=jnp.float16)
        w_q_i, scales_i = prepack_quantized_weights(w, mode=mode, bits=bits)
        w_q_parts.append(w_q_i)
        scales_parts.append(scales_i)

    w_q_batched = jnp.stack(w_q_parts, axis=0)
    scales_batched = jnp.stack(scales_parts, axis=0)

    dev = jax.devices("gpu")[0]
    x, w_q_batched, scales_batched = _device_put_all(dev, x, w_q_batched, scales_batched)

    def _single_triton(xi, wi, si):
        return triton_quantized_matmul(
            xi,
            wi,
            si,
            None,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    def _single_xla(xi, wi, si):
        return xla_quantized_matmul(
            xi,
            wi,
            si,
            None,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    out_triton = jax.block_until_ready(jax.jit(jax.vmap(_single_triton))(x, w_q_batched, scales_batched))
    out_xla = jax.block_until_ready(jax.jit(jax.vmap(_single_xla))(x, w_q_batched, scales_batched))

    assert out_triton.shape == (b, m, n)
    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=9e-2,
        atol=9e-2,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 4),
        ("nf4", 4),
        ("mxfp4", 4),
    ],
)
def test_quantized_matmul_triton_nested_vmap_over_x_matches_xla(mode: str, bits: int):
    """Nested vmap over x with shared weights exercises rank>3 reshape fast-path."""
    key = jax.random.PRNGKey(41)
    kx, kw = jax.random.split(key, 2)
    b1, b2, m, k, n = 2, 3, 8, 64, 64

    x = jax.random.normal(kx, (b1, b2, m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    def _single_triton(xi):
        return triton_quantized_matmul(
            xi,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    def _single_xla(xi):
        return xla_quantized_matmul(
            xi,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    out_triton = jax.block_until_ready(jax.vmap(jax.vmap(_single_triton))(x))
    out_xla = jax.block_until_ready(jax.vmap(jax.vmap(_single_xla))(x))

    assert out_triton.shape == (b1, b2, m, n)
    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=7e-2,
        atol=7e-2,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 4),
        ("nf4", 4),
    ],
)
def test_quantized_matmul_triton_nested_vmap_grad_over_x_matches_xla(mode: str, bits: int):
    """Nested vmap gradient checks backward rank>3 reshape fast-path."""
    key = jax.random.PRNGKey(43)
    kx, kw = jax.random.split(key, 2)
    b1, b2, m, k, n = 2, 3, 8, 64, 64

    x = jax.random.normal(kx, (b1, b2, m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    def _single_triton(xi):
        return triton_quantized_matmul(
            xi,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    def _single_xla(xi):
        return xla_quantized_matmul(
            xi,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
        )

    def _loss_triton(x_in):
        y = jax.vmap(jax.vmap(_single_triton))(x_in)
        return jnp.mean(y.astype(jnp.float32))

    def _loss_xla(x_in):
        y = jax.vmap(jax.vmap(_single_xla))(x_in)
        return jnp.mean(y.astype(jnp.float32))

    g_triton = jax.block_until_ready(jax.grad(_loss_triton)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))

    assert g_triton.shape == (b1, b2, m, k)
    np.testing.assert_allclose(
        np.asarray(g_triton, dtype=np.float32),
        np.asarray(g_xla, dtype=np.float32),
        rtol=9e-2,
        atol=9e-2,
    )
