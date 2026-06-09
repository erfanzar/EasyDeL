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

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

from ._utils import assert_allclose, device_platform, has_triton

pytestmark = [
    pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only Triton validation"),
    pytest.mark.skipif(not has_triton(), reason="Triton not installed"),
]


@pytest.mark.parametrize("mode", ["affine", "nf4", "mxfp4"])
def test_quantized_matmul_triton_matches_xla(mode: str):
    key = jax.random.PRNGKey(0 if mode == "affine" else 1)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    out_triton = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        platform="triton",
    )
    out_xla = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        platform="xla",
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    assert_allclose(out_triton, out_xla, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize(
    "mode,bits",
    [("affine", 4), ("nf4", 4), ("mxfp4", 4), ("nvfp8", 8)],
)
def test_quantized_matmul_triton_grad_input_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(23 if mode == "affine" else 29)
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

    def _loss_triton(x_in):
        y = quantized_matmul(
            x_in,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
            platform="triton",
        )
        return jnp.mean(y)

    def _loss_xla(x_in):
        y = quantized_matmul(
            x_in,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
            platform="xla",
        )
        return jnp.mean(y)

    g_triton = jax.block_until_ready(jax.grad(_loss_triton)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))
    assert_allclose(g_triton, g_xla, atol=7e-2, rtol=7e-2)
