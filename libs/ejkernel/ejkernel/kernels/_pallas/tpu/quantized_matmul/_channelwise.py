"""Opt-in aligned BF16-to-integer dense matmul with fused row quantization.

Only the primal is a Pallas kernel. The derivative is the existing XLA
represented-weight surrogate; this does not promise backward memory savings.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _primal(x, weights, scales, bits):
    if jax.default_backend() != "tpu":
        raise ValueError("Pallas channelwise dense matmul requires TPU hardware")
    dtype = jnp.int4 if bits == 4 else jnp.int8
    if x.dtype != jnp.bfloat16 or weights.dtype != dtype:
        raise ValueError("Pallas requires BF16 input and matching INT4/INT8 weights")
    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("Pallas requires rank-two activation and weight arrays")
    m, k = x.shape
    if weights.shape[0] != k:
        raise ValueError("Pallas requires matching contracting dimensions")
    n = weights.shape[1]
    if min(m, k, n) <= 0:
        raise ValueError("Pallas requires positive M/K/N dimensions")
    if scales.ndim != 2 or scales.shape[1] != n or scales.shape[0] != 1:
        raise ValueError("Pallas requires scales with shape [1, n]")
    if not jnp.issubdtype(scales.dtype, jnp.floating):
        raise TypeError("Pallas requires floating-point scales")
    bm = 128 if bits == 8 and m % 128 == 0 else 64
    if m % bm or k % 128 or n % 128:
        raise ValueError("Pallas requires M divisible by 64 and K/N divisible by 128")
    if k > 4096 or n > 4096:
        raise ValueError("Pallas full-K dense tiles currently require K/N <= 4096")
    bound = (1 << (bits - 1)) - 1

    def kernel(xr, wr, sr, yr):
        a = xr[...].astype(jnp.float32)
        peak = jnp.max(jnp.abs(a), axis=1, keepdims=True)
        row_scale = jnp.where(peak == 0, 1.0, peak / bound)
        normalized = a / row_scale
        lower = jnp.floor(jnp.abs(normalized))
        half = jnp.abs(a) * (2 * bound) == (2 * lower + 1) * peak
        even = lower + (lower.astype(jnp.int32) & 1).astype(jnp.float32)
        rounded = jnp.where(half, jnp.sign(a) * even, jnp.round(normalized))
        codes = jnp.clip(rounded, -bound, bound).astype(dtype)
        acc = jax.lax.dot_general(codes, wr[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
        yr[...] = (acc.astype(jnp.float32) * sr[...] * row_scale).astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(m // bm,),
        in_specs=(
            pl.BlockSpec((bm, k), lambda i: (i, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
            pl.BlockSpec((1, n), lambda i: (0, 0)),
        ),
        out_specs=pl.BlockSpec((bm, n), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=40 * 1024 * 1024),
        name=f"channelwise_dense_i{bits}_m{bm}",
    )(x, weights, scales)


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def channelwise_quantized_matmul_pallas(x, weights, scales, bits):
    return _primal(x, weights, scales, bits)


@channelwise_quantized_matmul_pallas.defjvp
def _jvp(bits, primals, tangents):
    from ...._xla.quantized_matmul._channelwise import _quantized_activation_matmul

    _, tangent = jax.jvp(lambda x, w, s: _quantized_activation_matmul(x, w, s, bits), primals, tangents)
    return channelwise_quantized_matmul_pallas(*primals, bits), tangent
