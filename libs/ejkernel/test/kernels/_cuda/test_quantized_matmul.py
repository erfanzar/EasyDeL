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

import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._cuda.quantized_matmul import quantized_matmul as cuda_quantized_matmul
from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "gpu" or shutil.which("nvcc") is None,
    reason="CUDA quantized_matmul tests require GPU backend and nvcc",
)

try:
    from ejkernel.kernels._cuda.quantized_matmul._build import build_cuda_lib

    build_cuda_lib()
except RuntimeError as exc:
    pytest.skip(f"CUDA quantized_matmul build failed: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _force_cuda_compute_type():
    prev = os.environ.get("EJKERNEL_QMM_CUDA_COMPUTE")
    # Use fp16 tensor-core GEMM to match XLA's preferred_element_type=float32
    # with fp16 inputs (which XLA lowers to FAST_16F tensor cores on GPU).
    os.environ["EJKERNEL_QMM_CUDA_COMPUTE"] = "f"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("EJKERNEL_QMM_CUDA_COMPUTE", None)
        else:
            os.environ["EJKERNEL_QMM_CUDA_COMPUTE"] = prev


def _device_put_all(dev, *arrays):
    return [jax.device_put(arr, dev) for arr in arrays]


def _group_sizes_for_mode(mode: str) -> list[int]:
    mode = mode.lower()
    if mode in ("affine", "nf4"):
        return [32, 64, 128]
    if mode in ("mxfp4", "mxfp8"):
        return [32]
    if mode in ("nvfp4", "nvfp8"):
        return [16]
    raise ValueError(f"Unsupported quantization mode: {mode}")


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("transpose", [False, True])
def test_quantized_matmul_cuda_affine_all_bits_matches_xla(bits: int, transpose: bool):
    key = jax.random.PRNGKey(503 + bits)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64
    group_size = 32

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)
    w_q, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=bits,
        group_size=group_size,
        transpose=not transpose,
    )

    dev = jax.devices("gpu")[0]
    x, w_q, scales, zeros = _device_put_all(dev, x, w_q, scales, zeros)

    out_cuda = cuda_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=transpose,
        mode="affine",
        bits=bits,
        group_size=group_size,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=transpose,
        mode="affine",
        bits=bits,
        group_size=group_size,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=1.5e-1,
        atol=1.5e-1,
    )


@pytest.mark.parametrize(
    "mode,group_size",
    [(mode, gs) for mode in ["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"] for gs in _group_sizes_for_mode(mode)],
)
@pytest.mark.parametrize("x_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("m", [8192])
@pytest.mark.parametrize("k", [4096, 8192])
@pytest.mark.parametrize("n", [8192])
def test_quantized_matmul_cuda_matches_xla(mode: str, group_size: int, x_dtype: jnp.dtype, m, k, n):
    key = jax.random.PRNGKey(0 if mode == "affine" else 1)
    kx, kw = jax.random.split(key, 2)

    x = jax.random.normal(kx, (m, k), dtype=x_dtype)
    w = jax.random.normal(kw, (n, k), dtype=x_dtype)

    packed = prepack_quantized_weights(w, mode=mode, group_size=group_size)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    transpose = False
    out_cuda = cuda_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=transpose,
        mode=mode,
        group_size=group_size,
    )
    use_bf16 = x_dtype == jnp.bfloat16
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=transpose,
        mode=mode,
        group_size=group_size,
        use_bf16=use_bf16,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla.astype(x.dtype))

    assert out_cuda.dtype == x.dtype
    rtol = 6e-2
    atol = 6e-2
    if x_dtype == jnp.bfloat16:
        rtol = max(rtol, 1.5e-1)
        atol = max(atol, 1.5e-1)
    if mode != "affine":
        rtol = 1.5e-1
        atol = 1.5e-1
    else:
        # Affine mode: CUDA uses a monolithic cuBLAS GEMM while XLA uses a
        # blocked-tiled algorithm (K/64 tiles accumulated in fp32).  Different
        # accumulation order means near-zero outputs (catastrophic cancellation)
        # can differ by O(eps_compute * k), independently of magnitude.  Large-
        # magnitude outputs differ by 1-2 ULPs of the compute dtype (covered by
        # rtol).  Empirical worst-case (k=8192): fp16→0.26, bf16→0.57, fp32→0.28.
        if x_dtype == jnp.float16:
            rtol = 3e-3  # ~3 fp16 ULPs relative for large-magnitude elements
            atol = 0.30  # near-zero cancellation ceiling for k ≤ 8192
        elif x_dtype == jnp.bfloat16:
            rtol = 2e-2  # ~3 bf16 ULPs relative for large-magnitude elements
            atol = 0.60  # near-zero cancellation ceiling for k ≤ 8192
        else:  # float32 (both CUDA & XLA use fp16/TF32 precision internally)
            rtol = 5e-3
            atol = 0.32  # near-zero cancellation ceiling for k ≤ 8192
    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [("affine", 4), ("nf4", 4), ("mxfp8", 8), ("nvfp4", 4)],
)
def test_quantized_matmul_cuda_grad_input_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(77 if mode == "affine" else 79)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 256, 256

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

    def _loss_cuda(x_in):
        y = cuda_quantized_matmul(
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

    g_cuda = jax.block_until_ready(jax.grad(_loss_cuda)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))

    np.testing.assert_allclose(
        np.asarray(g_cuda, dtype=np.float32),
        np.asarray(g_xla, dtype=np.float32),
        rtol=8e-2,
        atol=8e-2,
    )
