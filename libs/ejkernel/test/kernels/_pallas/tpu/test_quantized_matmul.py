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

from ejkernel.kernels._pallas.tpu.quantized_matmul import quantized_matmul as pallas_quantized_matmul
from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

pallas_qmm_iface = importlib.import_module("ejkernel.kernels._pallas.tpu.quantized_matmul._interface")
pallas_qmm_bwd = importlib.import_module("ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_bwd")
pallas_qmm_fwd = importlib.import_module("ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_fwd")


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


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
def test_quantized_matmul_pallas_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(0 if mode == "affine" else 1)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    out_pallas = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
        bits=bits,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
        bits=bits,
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_pallas, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


def test_quantized_matmul_pallas_transpose_matches_xla():
    key = jax.random.PRNGKey(42)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)

    # transpose=True expects packed weights in (N, K)-logical layout.
    w_q, scales, biases = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        transpose=False,
        group_size=32,
    )

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    biases = jax.device_put(biases, dev)

    out_pallas = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=True,
        mode="affine",
        bits=4,
        group_size=32,
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_pallas, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


def test_quantized_matmul_pallas_ignores_use_bf16_flag():
    key = jax.random.PRNGKey(99)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, biases = prepack_quantized_weights(w, mode="affine", bits=4)

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    biases = jax.device_put(biases, dev)

    out_true = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        use_bf16=True,
    )
    out_false = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        use_bf16=False,
    )

    out_true = jax.block_until_ready(out_true)
    out_false = jax.block_until_ready(out_false)
    np.testing.assert_allclose(
        np.asarray(out_true, dtype=np.float32),
        np.asarray(out_false, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_quantized_matmul_pallas_large_n_packed_matches_xla(monkeypatch: pytest.MonkeyPatch):
    key = jax.random.PRNGKey(1337)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, biases = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    biases = jax.device_put(biases, dev)
    monkeypatch.setenv("EJKERNEL_QMM_TPU_PATH", "packed")

    out_pallas = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        block_n=128,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        block_n=128,
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)
    np.testing.assert_allclose(
        np.asarray(out_pallas, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


def test_quantized_matmul_pallas_packed_illegal_falls_back_to_xla(monkeypatch: pytest.MonkeyPatch):
    key = jax.random.PRNGKey(1441)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, biases = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    biases = jax.device_put(biases, dev)
    monkeypatch.setenv("EJKERNEL_QMM_TPU_PATH", "packed")

    def _force_illegal(*args, **kwargs):
        return False

    monkeypatch.setattr(pallas_qmm_iface, "_is_packed_tpu_legal", _force_illegal)
    out_pallas = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        block_n=128,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        block_n=128,
    )
    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)
    np.testing.assert_allclose(
        np.asarray(out_pallas, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


def test_quantized_matmul_pallas_large_n_grad_input_matches_xla(monkeypatch: pytest.MonkeyPatch):
    key = jax.random.PRNGKey(1553)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, biases = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    biases = jax.device_put(biases, dev)
    monkeypatch.setenv("EJKERNEL_QMM_TPU_PATH", "packed")

    def _loss_pallas(x_in):
        y = pallas_quantized_matmul(
            x_in,
            w_q,
            scales,
            biases,
            transpose=False,
            mode="affine",
            bits=4,
            group_size=64,
            block_n=128,
        )
        return jnp.mean(y)

    def _loss_xla(x_in):
        y = xla_quantized_matmul(
            x_in,
            w_q,
            scales,
            biases,
            transpose=False,
            mode="affine",
            bits=4,
            group_size=64,
            block_n=128,
        )
        return jnp.mean(y)

    g_pallas = jax.block_until_ready(jax.grad(_loss_pallas)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))
    np.testing.assert_allclose(
        np.asarray(g_pallas, dtype=np.float32),
        np.asarray(g_xla, dtype=np.float32),
        rtol=7e-2,
        atol=7e-2,
    )


@pytest.mark.parametrize("legacy_path", ["hybrid", "predecode"])
def test_quantized_matmul_pallas_legacy_paths_alias_to_packed(monkeypatch: pytest.MonkeyPatch, legacy_path: str):
    key = jax.random.PRNGKey(1777 if legacy_path == "hybrid" else 1888)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 8, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, biases = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    biases = jax.device_put(biases, dev)

    assert pallas_qmm_iface._normalize_tpu_path(legacy_path) == "packed"
    assert pallas_qmm_iface._normalize_tpu_path("packed") == "packed"
    out_legacy = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        tpu_path=legacy_path,
    )
    out_packed = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        tpu_path="packed",
    )
    out_legacy = jax.block_until_ready(out_legacy)
    out_packed = jax.block_until_ready(out_packed)
    np.testing.assert_allclose(
        np.asarray(out_legacy, dtype=np.float32),
        np.asarray(out_packed, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
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
def test_quantized_matmul_pallas_fused_common_path(monkeypatch: pytest.MonkeyPatch, mode: str, bits: int):
    key = jax.random.PRNGKey(123 if mode == "affine" else 321)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    def _forbidden_xla(*args, **kwargs):
        raise AssertionError(f"Unexpected XLA fallback in fused Pallas common path for mode={mode}.")

    monkeypatch.setattr(pallas_qmm_iface, "_xla_quantized_matmul", _forbidden_xla)
    out = pallas_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
        bits=bits,
    )
    out = jax.block_until_ready(out)
    assert out.shape == (m, n)


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
def test_quantized_matmul_pallas_grad_input_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(17 if mode == "affine" else 19)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    def _loss_pallas(x_in):
        y = pallas_quantized_matmul(
            x_in,
            w_q,
            scales,
            biases,
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
            biases,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    g_pallas = jax.block_until_ready(jax.grad(_loss_pallas)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))
    np.testing.assert_allclose(
        np.asarray(g_pallas, dtype=np.float32),
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
def test_quantized_matmul_pallas_grad_input_fused_common_path(monkeypatch: pytest.MonkeyPatch, mode: str, bits: int):
    key = jax.random.PRNGKey(29 if mode == "affine" else 31)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("tpu")[0]
    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    def _forbidden_xla(*args, **kwargs):
        raise AssertionError(f"Unexpected XLA fallback in fused Pallas grad path for mode={mode}.")

    monkeypatch.setattr(pallas_qmm_iface, "_xla_quantized_matmul", _forbidden_xla)
    monkeypatch.setattr(pallas_qmm_bwd, "_xla_quantized_matmul", _forbidden_xla)

    def _loss(x_in):
        y = pallas_quantized_matmul(
            x_in,
            w_q,
            scales,
            biases,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    gx = jax.block_until_ready(jax.grad(_loss)(x))
    assert gx.shape == (m, k)
