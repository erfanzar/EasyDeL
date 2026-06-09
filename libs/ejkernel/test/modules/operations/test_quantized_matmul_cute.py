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

from ._utils import assert_allclose, device_platform, has_cutlass

try:
    from ejkernel.callib._cute_ffi import has_cute_ffi_support
    from ejkernel.kernels._cute.quantized_matmul._cute_impl import _wrap_vmap_compatible_jax_call
except ModuleNotFoundError as exc:
    if exc.name == "cuda" or (exc.name is not None and exc.name.startswith("cuda.")):
        pytest.skip("CUDA Python bindings required for CuTe tests", allow_module_level=True)
    raise

pytestmark = [
    pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only CUTE validation"),
    pytest.mark.skipif(not has_cutlass(), reason="CUTLASS CuTe DSL not installed"),
    pytest.mark.skipif(not has_cute_ffi_support(), reason="CuTe primitive support unavailable"),
]


@pytest.mark.parametrize("mode,bits", [("affine", 4), ("nf4", 4), ("mxfp4", 4)])
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

    out_cute = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="cute",
    )
    out_xla = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="xla",
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)

    assert_allclose(out_cute, out_xla, atol=6e-2, rtol=6e-2)


def test_quantized_matmul_cute_jit_matches_xla():
    key = jax.random.PRNGKey(123)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)
    w_q, scales = prepack_quantized_weights(w, mode="mxfp4", bits=4)

    @jax.jit
    def run(xi, wi, si):
        return quantized_matmul(
            xi,
            wi,
            si,
            None,
            transpose=False,
            mode="mxfp4",
            bits=4,
            platform="cute",
        )

    out_jit = run(x, w_q, scales)
    out_ref = quantized_matmul(
        x,
        w_q,
        scales,
        None,
        transpose=False,
        mode="mxfp4",
        bits=4,
        platform="xla",
    )

    out_jit = jax.block_until_ready(out_jit)
    out_ref = jax.block_until_ready(out_ref)

    assert_allclose(out_jit, out_ref, atol=6e-2, rtol=6e-2)


def test_quantized_matmul_cute_vmap_over_x_matches_xla():
    key = jax.random.PRNGKey(7)
    kx, kw = jax.random.split(key, 2)
    b, m, k, n = 3, 8, 64, 64

    x = jax.random.normal(kx, (b, m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)
    w_q, scales = prepack_quantized_weights(w, mode="mxfp4", bits=4)

    def run(platform: str):
        def _single(xi):
            return quantized_matmul(
                xi,
                w_q,
                scales,
                None,
                transpose=False,
                mode="mxfp4",
                bits=4,
                platform=platform,
            )

        return jax.vmap(_single)(x)

    out_cute = run("cute")
    out_xla = run("xla")

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_cute, out_xla, atol=6e-2, rtol=6e-2)


def test_quantized_matmul_cute_jit_vmap_over_all_inputs_matches_xla():
    key = jax.random.PRNGKey(11)
    kx, kpack = jax.random.split(key, 2)
    b, m, k, n = 2, 8, 64, 64

    x = jax.random.normal(kx, (b, m, k), dtype=jnp.float16)
    pack_keys = jax.random.split(kpack, b)

    w_q_parts = []
    scales_parts = []
    for pk in pack_keys:
        w = jax.random.normal(pk, (n, k), dtype=jnp.float16)
        w_q_i, scales_i = prepack_quantized_weights(w, mode="mxfp4", bits=4)
        w_q_parts.append(w_q_i)
        scales_parts.append(scales_i)

    w_q_batched = jnp.stack(w_q_parts, axis=0)
    scales_batched = jnp.stack(scales_parts, axis=0)

    def run(platform: str):
        def _single(xi, wi, si):
            return quantized_matmul(
                xi,
                wi,
                si,
                None,
                transpose=False,
                mode="mxfp4",
                bits=4,
                platform=platform,
            )

        return jax.jit(jax.vmap(_single))(x, w_q_batched, scales_batched)

    out_cute = run("cute")
    out_xla = run("xla")

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_cute, out_xla, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize(
    "mode,bits",
    [("affine", 4), ("nf4", 4), ("mxfp4", 4), ("nvfp8", 8)],
)
def test_quantized_matmul_cute_grad_input_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(41 if mode == "affine" else 43)
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

    def _loss_cute(x_in):
        y = quantized_matmul(
            x_in,
            w_q,
            scales,
            zeros,
            transpose=False,
            mode=mode,
            bits=bits,
            platform="cute",
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

    g_cute = jax.block_until_ready(jax.grad(_loss_cute)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))
    assert_allclose(g_cute, g_xla, atol=7e-2, rtol=7e-2)


def test_cute_qmm_vmap_wrapper_supports_mixed_batched_inputs():
    def base_call(a, b):
        return a + 2.0 * b

    wrapped = _wrap_vmap_compatible_jax_call(base_call)
    a = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    b = jnp.arange(4, dtype=jnp.float32)

    out = jax.jit(jax.vmap(lambda ai: wrapped(ai, b)))(a)
    ref = jax.vmap(lambda ai: base_call(ai, b))(a)
    assert_allclose(out, ref, atol=0.0)


def test_cute_qmm_vmap_wrapper_supports_all_batched_inputs():
    def base_call(a, b):
        return a * b

    wrapped = _wrap_vmap_compatible_jax_call(base_call)
    a = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    b = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

    out = jax.jit(jax.vmap(wrapped))(a, b)
    ref = jax.vmap(base_call)(a, b)
    assert_allclose(out, ref, atol=0.0)
