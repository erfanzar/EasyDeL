"""Utility functions for quantized linear layer operations.

This module provides low-level utility functions for working with quantized
weights in linear transformations, particularly for NF4 (4-bit NormalFloat)
quantization.

Functions:
    nf4xf32_to_f32: Convert NF4 quantized values to float32.
    i4tou4: Convert signed int4 to unsigned int4.
    nf4_qmm_jax: Perform quantized matrix multiplication with NF4 weights.

Note:
    These functions are primarily intended for internal use within the
    quantized linear layer implementations. For most use cases, prefer
    using the higher-level ParallelLinearQuantized class.
"""

import jax
from ejkernel.callib import cdiv, ejit  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp


def nf4xf32_to_f32(x):
    """Convert NF4 quantized values to float32 using polynomial approximation.

    This function implements a polynomial approximation to dequantize 4-bit
    NormalFloat (NF4) values back to float32. NF4 is a quantization format
    optimized for neural network weights, providing better representation
    of normally distributed values than uniform quantization.

    The polynomial coefficients are derived to approximate the NF4 lookup
    table values, providing a smooth and differentiable dequantization.

    Args:
        x: Input array containing NF4 quantized values (integers 0-15).
            Will be cast to float32 internally.

    Returns:
        Array of float32 values corresponding to the dequantized NF4 values.
        The output range is approximately [-1, 1].

    Example:
        >>> import jax.numpy as jnp
        >>> nf4_values = jnp.array([0, 7, 8, 15])
        >>> float_values = nf4xf32_to_f32(nf4_values)
    """
    x = x.astype(jnp.float32)
    return (
        x
        * (
            x * (x * (x * (1.82943132356953e-5 * x - 0.00068587779130373) + 0.0100420261313669) - 0.0722703570217226)
            + 0.346075459755188
        )
        - 0.994166218659335
    )


def i4tou4(x):
    """Convert signed int4 values to unsigned int4 values.

    Transforms signed 4-bit integers (range -8 to 7) to unsigned 4-bit
    integers (range 0 to 15) by adding 16 to negative values.

    Args:
        x: Input array containing signed int4 values in range [-8, 7].

    Returns:
        Array of unsigned int4 values in range [0, 15].

    Example:
        >>> import jax.numpy as jnp
        >>> signed = jnp.array([-8, -1, 0, 7])
        >>> unsigned = i4tou4(signed)
        >>> # unsigned = [8, 15, 0, 7]
    """
    return jnp.where(x < 0, 16 + x, x)


@ejit(static_argnames=["BK", "BM", "BNQL", "compute_dtype"])  # pyright: ignore[reportUntypedFunctionDecorator]
def nf4_qmm_jax(
    x: jax.Array,
    wq: jax.Array,
    wscale: jax.Array,
    BK: int = 2048,
    BM: int = 2048,
    BNQL: int = 2048,
    compute_dtype: jnp.dtype = jnp.bfloat16,
):
    """Perform quantized matrix multiplication with NF4 weights.

    Implements a blocked matrix multiplication where weights are stored in
    NF4 (4-bit NormalFloat) format. This function handles the dequantization
    of weights on-the-fly during the matrix multiplication, enabling memory
    efficient inference with quantized models.

    The computation is performed in blocks to optimize memory access patterns
    and enable efficient hardware utilization. Each weight block is dequantized,
    scaled, and then multiplied with the corresponding input block.

    Args:
        x: Input activation tensor of shape (M, K) where M is the batch
            dimension and K is the input feature dimension.
        wq: Quantized weight tensor in packed NF4 format. Shape is
            (K, qnumblocks, qfeatures) where qnumblocks is the number of
            quantization blocks and qfeatures is the packed feature dimension.
        wscale: Per-block scaling factors for dequantization. Shape is
            (K, qnumblocks). Each scale is applied to its corresponding
            block of dequantized weights.
        BK: Block size along the K (reduction) dimension. Defaults to 2048.
            Smaller values reduce memory usage but may impact performance.
        BM: Block size along the M (batch) dimension. Defaults to 2048.
        BNQL: Block size hint for the N (output) dimension in terms of
            quantization blocks. Defaults to 2048.
        compute_dtype: Data type for intermediate computations.
            Defaults to jnp.bfloat16 for efficiency on modern hardware.

    Returns:
        Output tensor of shape (M, UN) where UN is the unpacked output
        feature dimension (qblocksize * qnumblocks).

    Note:
        - The weights are stored with two 4-bit values packed per byte
        - Each weight is unpacked, converted from int4 to uint4, dequantized
          using polynomial approximation, and then scaled
        - This function is JIT-compiled with ejit for performance

    Example:
        >>> import jax.numpy as jnp
        >>> # Assuming quantized weights and scales are prepared
        >>> x = jnp.ones((32, 768), dtype=jnp.bfloat16)
        >>> output = nf4_qmm_jax(x, wq, wscale)
    """
    qfeatures = wq.shape[-1]
    qnumblocks = wq.shape[-2]
    qblocksize = qfeatures * 2
    UN = qblocksize * qnumblocks
    M = x.shape[0]
    K = x.shape[1]
    bnql = min(BNQL, qnumblocks)
    BN = min(cdiv(bnql, qnumblocks) * qnumblocks, UN)
    bm, bk = min(BM, M), min(BK, K)
    num_mblocks = cdiv(M, bm)
    num_kblocks = cdiv(K, bk)
    num_nblocks = cdiv(UN, BN)
    BQ = cdiv(num_nblocks, qnumblocks)
    output = jnp.zeros([M, UN])
    for midx in range(num_mblocks):
        for nidx in range(num_nblocks):
            acc = jnp.zeros([bm, BN], compute_dtype)
            for kidx in range(num_kblocks):
                a = jax.lax.dynamic_slice(x, (midx * bm, kidx * bk), (bm, bk))
                b = jax.lax.dynamic_slice(wq, (kidx * bk, nidx * BQ, 0), (bk, BQ, qfeatures))
                bs = jax.lax.dynamic_slice(wscale, (kidx * bk, nidx * BQ), (bk, BQ))

                b = jnp.stack([(b >> 4) & 0xF, b & 0xF], axis=-1)
                *batch_dims, num_blocks, _ = b.shape
                b = b.reshape(*batch_dims, num_blocks, -1)
                b = i4tou4(b)
                b = nf4xf32_to_f32(b).reshape(*b.shape[:-2], -1)
                w = (b * jnp.expand_dims(bs, -1)).reshape(bk, -1).astype(compute_dtype)
                acc += jnp.dot(a, w)
            output = output.at[
                midx * bm : (midx * bm) + bm,
                nidx * BN : (nidx * BN) + BN,
            ].set(acc)
    return output
