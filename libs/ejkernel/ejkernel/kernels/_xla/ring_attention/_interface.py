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

"""Ring attention implementation for distributed sequence processing.

This module provides an implementation of Ring Attention, a technique for
computing attention across extremely long sequences by distributing the
computation across multiple devices in a ring topology.

Ring Attention enables processing sequences that exceed the memory capacity
of a single device by:
1. Partitioning query, key, and value tensors across devices
2. Rotating key-value pairs around the ring in a pipeline fashion
3. Computing local attention blocks and accumulating results
4. Using online softmax for numerically stable accumulation

This is particularly useful for:
- Training on very long documents (100K+ tokens)
- Distributed inference with sequence parallelism
- Memory-efficient attention for large context windows

Key features:
1. Sequence parallelism: Distributes sequence dimension across devices
2. Memory efficiency: O(N/P) memory per device for P-way parallelism
3. Blockwise computation: Processes attention in query/key chunks
4. Online softmax: Numerically stable accumulation across blocks
5. Custom VJP: Efficient gradient computation for training

Algorithm overview:
- Each device holds a shard of Q, K, V along the sequence dimension
- Key-value pairs are rotated using collective ppermute operations
- Each device computes attention for its query shard against all K-V shards
- Results are accumulated using online softmax algorithm
- After P rotations, each device has complete attention output for its queries

Supported features:
- Causal and non-causal attention modes
- Segment IDs for packed sequence processing
- Sliding window attention for local patterns
- Attention sinks via softmax_aux parameter
- Logits soft capping for numerical stability
- Dropout with configurable probability
- Configurable chunk sizes for memory/compute trade-off

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.ring_attention import ring_attention
    >>>
    >>> # Assuming 4-way sequence parallelism
    >>> batch, local_seq_len, num_heads, head_dim = 2, 512, 8, 64
    >>> q = jnp.ones((batch, local_seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, local_seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, local_seq_len, num_heads, head_dim))
    >>>
    >>> # Run without axis_name for single-device testing
    >>> output = ring_attention(q, k, v, causal=True)
    >>>
    >>> # In distributed setting with pmap/shard_map:
    >>> # output = ring_attention(q, k, v, causal=True, axis_name="seq")

Reference:
    Ring Attention with Blockwise Transformers for Near-Infinite Context
    https://arxiv.org/abs/2310.01889
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from functools import partial

import chex
import jax
import jax.lax as lax
import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from ejkernel.callib import ejit
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _ring_attention_bwd
from ._xla_impl_fwd import _ring_attention_fwd

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


@partial(jax.custom_vjp, nondiff_argnums=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
def _ring_attention(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None = None,
    q_segment_ids: chex.Array | None = None,
    kv_segment_ids: chex.Array | None = None,
    q_position_ids: chex.Array | None = None,
    kv_position_ids: chex.Array | None = None,
    softmax_aux: chex.Array | None = None,
    axis_name: str | None = None,
    float32_logits: bool = True,
    softmax_scale: float | None = None,
    query_chunk_size: int = 512,
    key_chunk_size: int = 512,
    causal_block_size: int | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    pdrop: float = 0.0,
    dtype: DTypeLike = jnp.float32,
    policy=jax.checkpoint_policies.nothing_saveable,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    prevent_cse: bool = True,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    causal: bool = False,
):
    """Compute ring attention with blockwise transformers (internal, with custom VJP).

    This is the workhorse called by ``ring_attention`` after argument
    normalization. Non-differentiable arguments (indices 9–25) are treated
    as static by ``jax.custom_vjp``.

    Args:
        query: Query array of shape (batch, q_len, num_heads, dim_per_head).
        key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
        value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
        bias: Optional additive bias of shape (batch, num_heads, q_len, kv_len).
        q_segment_ids: Optional query segment IDs of shape (batch, q_len).
            If both q_segment_ids and kv_segment_ids are None, no segment masking
            is applied.  If only one is provided, it is broadcast to both sides.
        kv_segment_ids: Optional key/value segment IDs of shape (batch, kv_len).
        q_position_ids: Optional query position IDs of shape (batch, q_len).
            When provided together with kv_position_ids, causal and sliding-window
            masking use these explicit positions instead of chunk offsets.
        kv_position_ids: Optional key/value position IDs of shape (batch, kv_len).
        softmax_aux: Optional attention-sink logits of shape [num_sinks] (1-D) or
            [num_heads, num_sinks] (2-D).  These values participate in softmax
            normalization but do not contribute to the output, allowing the model
            to absorb surplus probability mass.
        axis_name: Name of the JAX collective axis for ring communication.
            If None, runs in single-device (blockwise) mode.
        float32_logits: If True, cast query and key to float32 before computing
            logits for improved numerical stability.
        softmax_scale: Multiplicative scale applied to QK^T logits.  Defaults
            to ``1/sqrt(head_dim)`` (standard attention scaling).  Note: the
            forward implementation stores this as a divisor internally
            (``1/softmax_scale``), so passing ``1/sqrt(head_dim)`` correctly
            produces ``logits * (1/sqrt(head_dim))``.
        query_chunk_size: Number of query tokens processed per chunk.
        key_chunk_size: Number of key tokens processed per chunk.
        causal_block_size: Block size used for efficient causal masking.  If None
            and causal=True, defaults to query_chunk_size.
        deterministic: If True, disables dropout.
        dropout_rng: PRNG key for dropout; required when deterministic=False and
            pdrop > 0.
        pdrop: Dropout probability.
        dtype: Output and accumulation dtype.
        policy: JAX checkpoint policy for remat (controls what is saved vs
            recomputed during backprop).
        precision: JAX matmul precision.
        prevent_cse: Passed to jax.checkpoint; prevents common-subexpression
            elimination across checkpointed blocks.
        sliding_window: Local attention window.  An int applies a symmetric window;
            a tuple (left, right) applies an asymmetric window.  None = full attention.
        logits_soft_cap: If set, applies ``cap * tanh(logits / cap)`` before
            softmax to prevent logit explosion.
        attention_sink_size: Number of leading KV positions that every query
            always attends to (StreamingLLM-style sink).  0 = disabled.
        causal: If True, applies causal masking using causal_block_size.

    Returns:
        Output array of shape (batch, q_len, num_heads, dim_per_head).
    """

    if q_segment_ids is None and kv_segment_ids is not None:
        q_segment_ids = kv_segment_ids
    elif kv_segment_ids is None and q_segment_ids is not None:
        kv_segment_ids = q_segment_ids

    if causal and causal_block_size is None:
        causal_block_size = query_chunk_size

    y, _ = _ring_attention_fwd(
        query,
        key,
        value,
        bias,
        q_segment_ids,
        kv_segment_ids,
        q_position_ids,
        kv_position_ids,
        softmax_aux,
        axis_name,
        float32_logits,
        softmax_scale,
        query_chunk_size,
        key_chunk_size,
        causal_block_size,
        deterministic,
        dropout_rng,
        pdrop,
        dtype,
        policy,
        precision,
        prevent_cse,
        sliding_window,
        logits_soft_cap,
        attention_sink_size,
        causal,
    )
    return y


_ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)
_ring_attention = ejit(
    _ring_attention, static_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
)


@kernel_registry.register("ring_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    q_position_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_position_ids: Int[Array, "batch seq_len_k"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = False,
    logits_soft_cap: float | None = None,
    softmax_scale: float | None = None,
    axis_name: str | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    fused_backward: bool = False,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Compute ring attention for distributed long-sequence processing.

    Ring attention distributes attention computation across devices by rotating
    key-value pairs in a ring topology. Each device computes attention for its
    local query shard against all key-value shards.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim].
            In distributed mode, this is the local shard of queries.
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim].
            In distributed mode, this is the local shard of keys.
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim].
            In distributed mode, this is the local shard of values.
        q_segment_ids: Optional query segment IDs [batch, seq_len_q].
            Used for packed sequence processing to prevent cross-sequence attention.
        kv_segment_ids: Optional key/value segment IDs [batch, seq_len_k].
            If only one of q/kv segment IDs is provided, it's used for both.
        q_position_ids: Optional query position IDs [batch, seq_len_q].
            Used for position-based masking in packed sequences.
        kv_position_ids: Optional key/value position IDs [batch, seq_len_k].
            Used for position-based masking in packed sequences.
        softmax_aux: Optional attention sink logits [num_sinks].
            Participate in softmax normalization but don't contribute to output.
        bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k].
            Additive bias applied to attention scores before softmax.
        mask_builder: Optional mask builder function (unused in XLA backend).
        sliding_window: Optional local attention window. Can be:
            - int: Symmetric window (same left and right)
            - tuple[int, int]: Asymmetric (left_window, right_window)
            - None: Full attention (default)
        chunk_size: Size for both query and key chunks. If None, uses
            min(512, seq_len) as default.
        causal: If True, applies causal masking. Default False.
        logits_soft_cap: Soft cap for attention logits via tanh.
            Applies: cap * tanh(logits / cap).
        softmax_scale: Scaling factor for QK^T. Defaults to 1/sqrt(head_dim).
        axis_name: Name of the JAX collective axis for ring communication.
            Required for distributed execution with pmap/shard_map.
            If None, runs in single-device mode.
        fwd_params: Forward pass parameters (q_blocksize, kv_blocksize).
        bwd_params: Backward pass parameters (unused in XLA backend).
        fused_backward: Whether to fuse backward pass (unused in XLA backend).

    Returns:
        Attention output [batch, seq_len_q, num_heads, head_dim].
        In distributed mode, this is the complete output for the local query shard.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._xla.ring_attention import ring_attention
        >>>
        >>> # Single-device mode (axis_name=None)
        >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>>
        >>> output = ring_attention(q, k, v, causal=True, chunk_size=256)
        >>> output.shape
        (2, 1024, 8, 64)

    Note:
        - For distributed execution, use within jax.pmap or jax.shard_map
        - The chunk_size should evenly divide the sequence length
        - Memory usage scales as O(chunk_size² + seq_len * head_dim) per device
    """
    del mask_builder, bwd_params, fused_backward

    if fwd_params is None:
        fwd_params = FwdParams()

    default_q = min(512, query.shape[1])
    default_k = min(512, key.shape[1])
    qcs_default = default_q if fwd_params.q_blocksize is None else int(fwd_params.q_blocksize)
    kcs_default = default_k if fwd_params.kv_blocksize is None else int(fwd_params.kv_blocksize)

    if chunk_size is not None:
        qcs_default = int(chunk_size)
        kcs_default = int(chunk_size)

    q_len = int(query.shape[1])
    kv_len = int(key.shape[1])

    def _divisor_or_self(length: int, candidate: int) -> int:
        """Find the largest divisor of length that is at most candidate.

        If candidate evenly divides length, returns candidate directly.
        Otherwise searches downward from candidate to find the largest
        divisor of length.

        Args:
            length: The sequence length to divide.
            candidate: The preferred chunk size (upper bound).

        Returns:
            The largest divisor of length that is <= candidate, or 1.
        """
        candidate = max(1, min(int(candidate), length))
        if length % candidate == 0:
            return candidate
        for d in range(candidate, 0, -1):
            if length % d == 0:
                return d
        return 1

    qcs = _divisor_or_self(q_len, qcs_default)
    kcs = _divisor_or_self(kv_len, kcs_default)

    return _ring_attention(
        query,
        key,
        value,
        bias,
        q_segment_ids,
        kv_segment_ids,
        q_position_ids,
        kv_position_ids,
        softmax_aux,
        axis_name,
        True,
        softmax_scale,
        qcs,
        kcs,
        None,
        True,
        None,
        0.0,
        jnp.float32,
        jax.checkpoint_policies.nothing_saveable,
        jax.lax.Precision.DEFAULT,
        True,
        sliding_window,
        logits_soft_cap,
        0,
        causal,
    )
