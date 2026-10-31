"""Shared rowwise symmetric integer quantization for XLA kernels."""

import jax.numpy as jnp


def quantize_rows(x, bits):
    """Return codes and scales with stable nearest-even BF16 half ties.

    TPU reciprocal lowering may perturb ``x / (amax / qmax)`` to the wrong
    side of an exact half. A cross-product check restores true half ties;
    its products are exact in FP32 for normal BF16 inputs and 4/8-bit bounds.
    FP32 input arithmetic still obeys ordinary FP32 rounding.
    """
    x32 = x.astype(jnp.float32)
    bound = (1 << (bits - 1)) - 1
    amax = jnp.max(jnp.abs(x32), axis=1, keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / bound)
    normalized = x32 / scale
    rounded = jnp.round(normalized)
    lower = jnp.floor(jnp.abs(normalized))
    is_half = (jnp.abs(x32) * (2 * bound)) == ((2 * lower + 1) * amax)
    # Finite normalized magnitudes are bounded by qmax, so integer parity
    # avoids lowering a full floating-point remainder for every activation.
    parity = (lower.astype(jnp.int32) & 1).astype(jnp.float32)
    nearest_even = lower + parity
    rounded = jnp.where(is_half, jnp.sign(x32) * nearest_even, rounded)
    dtype = jnp.int4 if bits == 4 else jnp.int8
    return jnp.clip(rounded, -bound, bound).astype(dtype), scale
