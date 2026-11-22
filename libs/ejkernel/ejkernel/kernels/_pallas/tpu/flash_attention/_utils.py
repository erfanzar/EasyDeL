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


"""Utility functions and data structures for Flash Attention TPU kernels.

This module provides common utilities, constants, and data structures used
by the Flash Attention forward and backward pass implementations on TPU.

Key components:
- SegmentIds: Named tuple for query/KV segment IDs in packed sequences
- BlockSizes: Dataclass for configuring attention tile sizes
- Reference MHA implementations for testing and cost estimation
- Helper functions for block verification and cost estimation

TPU Architecture Constants:
- NUM_LANES (128): Number of lanes in TPU vector unit
- NUM_SUBLANES (8): Number of sublanes for segment ID handling
- MIN_BLOCK_SIZE (128): Minimum block size for TPU efficiency
- DEFAULT_MASK_VALUE: Large negative value for masking (~-0.7 * float32_max)

Example:
    >>> from ejkernel.kernels._pallas.tpu.flash_attention._utils import (
    ...     BlockSizes, SegmentIds, mha_reference
    ... )
    >>> block_sizes = BlockSizes.get_default(batch=2, heads=8, q_len=1024, kv_len=1024, d=64)
    >>> segment_ids = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
"""Default value for masked attention positions, approximately -0.7 * float32_max."""

NUM_LANES = 128
"""Number of lanes in TPU vector processing unit."""

NUM_SUBLANES = 8
"""Number of sublanes for efficient segment ID broadcasting."""

MIN_BLOCK_SIZE = 128
"""Minimum efficient block size for TPU MXU operations."""

TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))
"""Einsum dimension numbers for Q @ K^T matrix multiplication."""


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers).
    Only the token with the same id can attend to each other.

    Attributes:
      q: segment ids along the Q sequence.
      kv: segment ids along the KV sequence.
    """

    q: jax.Array
    kv: jax.Array


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tile sizes parameterizing FlashAttention kernels.

    Those parameters have negligible effect on numerics, but affect performance
    greatly.
    """

    block_q: int
    block_k_major: int
    block_k: int
    block_b: int

    block_q_major_dkv: int | None = None
    block_k_major_dkv: int | None = None
    block_k_dkv: int | None = None
    block_q_dkv: int | None = None

    block_k_major_dq: int | None = None
    block_k_dq: int | None = None
    block_q_dq: int | None = None

    def __post_init__(self):
        """Validate major/minor block-size relationships on construction."""

        def verify_major_minor(prefix, suffix, major, minor):
            if minor > major:
                raise ValueError(f"{prefix}{suffix}={minor} should be smaller than {prefix}_major{suffix}={major}")
            if major % minor != 0:
                raise ValueError(f"{prefix}{suffix}={minor} should divide {prefix}_major{suffix}={major}")

        verify_major_minor("block_k", "", self.block_k_major, self.block_k)
        if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
            verify_major_minor("block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv)
        if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
            verify_major_minor("block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv)
        if self.block_k_major_dq is not None and self.block_k_dq is not None:
            verify_major_minor("block_k", "_dq", self.block_k_major_dq, self.block_k_dq)

    @property
    def has_backward_blocks(self) -> bool:
        """Return True if all backward-pass block fields are set (not None)."""
        backward_blocks = (
            self.block_q_major_dkv,
            self.block_k_major_dkv,
            self.block_q_dkv,
            self.block_k_dkv,
            self.block_k_major_dq,
            self.block_k_dq,
            self.block_q_dq,
        )
        return all(b is not None for b in backward_blocks)

    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
        """Return a default BlockSizes with all tiles set to 128.

        All arguments are accepted for API compatibility but are currently
        ignored — the returned sizes are unconditionally 128 for every field.

        Args:
            batch_size: Batch size (unused).
            num_heads: Number of attention heads (unused).
            q_seq_len: Query sequence length (unused).
            kv_len: Key/value sequence length (unused).
            d_model: Head dimension (unused).

        Returns:
            BlockSizes with block_q, block_k_major, block_k, block_b, and all
            backward fields set to 128 / 1 respectively.
        """
        del batch_size, num_heads, q_seq_len, kv_len, d_model
        return BlockSizes(
            block_q=128,
            block_k_major=128,
            block_k=128,
            block_b=1,
            block_q_major_dkv=128,
            block_k_major_dkv=128,
            block_k_dkv=128,
            block_q_dkv=128,
            block_k_major_dq=128,
            block_k_dq=128,
            block_q_dq=128,
        )


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    """Verify that a block size is valid for a given dimension.

    Ensures the block size doesn't exceed the dimension and optionally
    verifies that the dimension is evenly divisible by the block size.

    Args:
        block_name: Name of the block parameter for error messages.
        dim_name: Name of the dimension for error messages.
        block: The block size to verify.
        dim: The dimension size to check against.
        should_divide: If True, require dim to be divisible by block.

    Raises:
        ValueError: If block > dim or if should_divide and dim % block != 0.
    """
    if block > dim:
        raise ValueError(f"{block_name}={block} should be smaller or equal to {dim_name}={dim}")
    if should_divide and dim % block != 0:
        raise ValueError(f"{dim_name}={dim} should be divisible by {block_name}={block}")


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
    """Calculate the total memory size of an array in bytes.

    Args:
        x: A JAX array or ShapeDtypeStruct to measure.

    Returns:
        Total size in bytes (product of shape elements times dtype itemsize).
    """
    return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    softmax_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    """Estimate computational cost for the Flash Attention forward pass.

    Creates a cost estimate that helps Pallas schedule kernel execution
    and optimize memory transfers. The estimate includes FLOPs,
    transcendental operations (exp, log), and memory bandwidth.

    Args:
        q: Query tensor for shape/dtype information.
        k: Key tensor for shape/dtype information.
        v: Value tensor for shape/dtype information.
        ab: Optional attention bias tensor.
        segment_ids: Optional segment IDs for packed sequences.
        causal: Whether causal masking is applied.
        softmax_scale: Softmax scaling factor.
        kernel_inputs_specs: Input tensor specifications for byte counting.
        kernel_outputs_specs: Output tensor specifications for byte counting.

    Returns:
        CostEstimate with FLOPs, transcendentals, and bytes_accessed.
    """
    body_cost = pl.estimate_cost(mha_reference, q, k, v, ab, segment_ids, causal=causal, softmax_scale=softmax_scale)
    input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
    output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )


def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    softmax_scale: float = 1.0,
    save_residuals: bool = False,
):
    """Reference multi-head attention implementation without custom VJP.

    A straightforward implementation of scaled dot-product attention for
    testing and cost estimation. This version does not have custom gradients
    and uses standard JAX autodiff.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias [batch, num_heads, q_seq_len, kv_seq_len].
        segment_ids: Optional SegmentIds for segment masking.
        causal: Whether to apply causal masking.
        mask_value: Value for masked positions.
        softmax_scale: Scaling factor for attention scores.
        save_residuals: If True, return (output, l, m) tuple.

    Returns:
        If save_residuals: Tuple of (output, l, m) where l is softmax sum
        and m is softmax max for each position.
        Otherwise: Just the output tensor.
    """
    logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
    if ab is not None:
        logits += ab
    if softmax_scale != 1.0:
        logits *= softmax_scale

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, :, :]

    if causal:
        _, _, q_seq_len, _ = q.shape
        _, _, kv_seq_len, _ = k.shape
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, :, :]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    weights = unnormalized / l[..., None]
    out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
    if save_residuals:
        return out, l, m
    return out


@functools.partial(jax.jit, static_argnames=["causal", "mask_value", "softmax_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    softmax_scale=1.0,
):
    """JIT-compiled reference multi-head attention with custom VJP.

    This is the main reference implementation used for testing and cost
    estimation. It wraps _mha_reference with JIT compilation and bfloat16
    matmul precision for TPU compatibility.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias tensor.
        segment_ids: Optional SegmentIds for segment masking.
        causal: Whether to apply causal masking.
        mask_value: Value for masked positions.
        softmax_scale: Scaling factor for attention scores.

    Returns:
        Attention output tensor [batch, num_heads, q_seq_len, head_dim].
    """
    return _mha_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
        save_residuals=False,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    softmax_scale: float,
    save_residuals: bool,
):
    """Internal reference MHA with custom VJP for efficient gradient computation.

    Wraps ``mha_reference_no_custom_vjp`` with a custom VJP rule that avoids
    recomputing the full attention matrix during the backward pass.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias tensor.
        segment_ids: Optional SegmentIds for segment masking.
        causal: Whether to apply causal masking.
        mask_value: Value for masked positions.
        softmax_scale: Scaling factor for attention scores.
        save_residuals: If True, return residuals for gradient computation.

    Returns:
        Attention output, or (output, l, m) tuple if save_residuals is True.
    """
    return mha_reference_no_custom_vjp(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
        save_residuals=save_residuals,
    )


def _mha_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    softmax_scale: float,
    save_residuals: bool,
):
    """Forward pass for custom VJP reference MHA.

    Computes attention and saves residuals (output, softmax normalizer,
    softmax max) needed for the backward pass.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias tensor.
        segment_ids: Optional SegmentIds for segment masking.
        causal: Whether to apply causal masking.
        mask_value: Value for masked positions.
        softmax_scale: Scaling factor for attention scores.
        save_residuals: Must be False (higher-order AD not supported).

    Returns:
        Tuple of (output, residuals) where residuals contains all tensors
        needed for the backward pass.

    Raises:
        NotImplementedError: If save_residuals is True.
    """
    if save_residuals:
        raise NotImplementedError
    res = _mha_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
        save_residuals=True,
    )
    assert isinstance(res, tuple)
    out, l, m = res
    return out, (q, k, v, ab, segment_ids, out, l, m)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "mask_value",
        "softmax_scale",
    ],
)
def mha_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    softmax_scale: float = 1.0,
):
    """Compute gradients for reference MHA implementation.

    Recomputes attention probabilities from saved residuals and computes
    gradients for query, key, value, and optional bias tensors.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias tensor.
        segment_ids: Optional SegmentIds for segment masking.
        o: Forward pass output tensor.
        l: Softmax normalizer (sum of exponentials) from forward pass.
        m: Softmax max values from forward pass.
        do: Gradient of the output tensor.
        causal: Whether causal masking was applied.
        mask_value: Value used for masked positions.
        softmax_scale: Scaling factor used for attention scores.

    Returns:
        Tuple of (dq, dk, dv, dab) gradients.

    Raises:
        NotImplementedError: If softmax_scale != 1.0.
    """
    if softmax_scale != 1.0:
        raise NotImplementedError

    logits = jnp.einsum(
        "bhqc,bhkc->bhqk",
        q.astype(jnp.float32),
        k.astype(jnp.float32),
    )
    if ab is not None:
        logits += ab

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, :, :]

    if causal:
        _, _, q_seq_len, _ = q.shape
        _, _, kv_seq_len, _ = k.shape
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, :, :]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    unnormalized = jnp.exp(logits - m[..., None])
    p = unnormalized / l[..., None]
    dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

    dp = jnp.einsum("bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32))

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[..., None]

    ds = (dp - di) * p
    dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
    dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

    dab = ds if ab is not None else None
    return dq, dk, dv, dab


def _mha_reference_bwd(
    causal: bool,
    mask_value: float,
    softmax_scale: float,
    save_residuals: bool,
    residuals,
    do,
):
    """Custom VJP backward pass for reference MHA.

    Unpacks saved residuals and delegates to ``mha_reference_bwd`` for
    the actual gradient computation.

    Args:
        causal: Whether causal masking was applied (non-differentiable).
        mask_value: Value for masked positions (non-differentiable).
        softmax_scale: Scaling factor (non-differentiable).
        save_residuals: Whether residuals were saved (non-differentiable).
        residuals: Saved tensors from the forward pass.
        do: Gradient of the output tensor.

    Returns:
        Tuple of gradients (dq, dk, dv, dab, None) corresponding to
        (q, k, v, ab, segment_ids) inputs.
    """
    del save_residuals
    q, k, v, ab, segment_ids, o, l, m = residuals
    dq, dk, dv, dab = mha_reference_bwd(
        q,
        k,
        v,
        ab,
        segment_ids,
        o,
        l,
        m,
        do,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
    )
    return dq, dk, dv, dab, None


_mha_reference.defvjp(fwd=_mha_reference_fwd, bwd=_mha_reference_bwd)


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    """Check if a block position is below or on the causal diagonal.

    Used to determine if a query block can attend to a key-value block
    in causal attention. A query block can attend to a KV block if the
    last row of the query block is at or after the first column of the KV block.

    Args:
        r: Row block index (query block index).
        r_blk_size: Size of row blocks (query block size).
        c: Column block index (KV block index).
        c_blk_size: Size of column blocks (KV block size).

    Returns:
        Boolean indicating if the query block can attend to the KV block.
        True if the last element of query block r can see the first element
        of KV block c under causal masking.
    """
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)
