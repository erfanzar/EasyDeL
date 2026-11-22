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

"""Native Sparse Attention interface with block selection and compression.

This module provides the public API for Native Sparse Attention (NSA) that
combines compressed global attention with fine-grained sparse block selection.
Includes VJP for training support and top-k block selection algorithms.
"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _sparse_attention_bwd
from ._xla_impl_fwd import _sparse_attention_fwd


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _sparse_attention_with_vjp(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Block-sparse attention with custom VJP for gradient computation.

    Computes sparse attention where each query attends to a subset of key
    blocks specified by block_indices. Custom VJP enables efficient gradients.

    Args:
        query: Query tensor [batch, seq_len, num_q_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        block_indices: Per-token indices of blocks to attend to
            [batch, seq_len, num_kv_heads, num_selected_blocks]
        block_counts: Number of blocks per token [batch, seq_len, num_kv_heads]
        block_size: Size of each attention block
        softmax_scale: Attention scaling factor

    Returns:
        Sparse attention output [batch, seq_len, num_q_heads, head_dim]
    """
    return _sparse_attention_fwd(query, key, value, block_indices, block_counts, block_size, softmax_scale)


def _sparse_attention_fwd_vjp(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
):
    """Forward rule for sparse attention VJP.

    Computes sparse attention output and saves residuals for backward pass.

    Args:
        query, key, value: Input tensors for attention
        block_indices, block_counts: Sparsity pattern specification
        block_size: Attention block size
        softmax_scale: Attention scaling factor

    Returns:
        Tuple of (output, residuals) for gradient computation
    """
    output = _sparse_attention_fwd(query, key, value, block_indices, block_counts, block_size, softmax_scale)
    residuals = (query, key, value, block_indices, block_counts, block_size, softmax_scale)
    return output, residuals


def _sparse_attention_bwd_vjp(
    block_size: int,
    softmax_scale: float,
    residuals: tuple,
    do: Float[Array, "batch seq_len num_q_heads head_dim"],
):
    """Backward rule for sparse attention VJP.

    Computes gradients with respect to query, key, and value tensors.
    Block indices and counts don't contribute gradients.

    Args:
        block_size: Attention block size (nondiff)
        softmax_scale: Scaling factor (nondiff)
        residuals: Saved tensors from forward pass
        do: Gradient of loss with respect to output

    Returns:
        Tuple of gradients (dq, dk, dv, None, None)
    """
    query, key, value, block_indices, block_counts, block_size_, softmax_scale_ = residuals
    dq, dk, dv = _sparse_attention_bwd(query, key, value, block_indices, block_counts, block_size_, softmax_scale_, do)

    return (dq, dk, dv, None, None)


_sparse_attention_with_vjp.defvjp(_sparse_attention_fwd_vjp, _sparse_attention_bwd_vjp)


def _nsa_compression_xla(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    k_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    v_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    block_size: int,
    softmax_scale: float,
) -> tuple[Float[Array, "batch seq_len num_q_heads head_dim"], Float[Array, "batch seq_len num_q_heads"]]:
    """Compute compressed attention over mean-pooled key/value blocks with GQA support.

    This function implements the compressed attention component of Native
    Sparse Attention (NSA). It computes attention between full-resolution
    queries and block-compressed (mean-pooled) keys and values. The
    compressed representations provide a coarse global context summary.

    A causal block mask ensures each query token only attends to blocks
    that have been fully completed (i.e., all tokens in the block precede
    the query position).

    Args:
        query: Query tensor of shape [batch, seq_len, num_q_heads, head_dim].
        k_cmp: Compressed (mean-pooled) keys of shape
            [batch, num_blocks, num_kv_heads, head_dim].
        v_cmp: Compressed (mean-pooled) values of shape
            [batch, num_blocks, num_kv_heads, head_dim].
        block_size: Number of tokens per block used for compression.
        softmax_scale: Multiplicative scaling factor for attention scores.

    Returns:
        Tuple of (output, lse) where:
            - output: Compressed attention output of shape
              [batch, seq_len, num_q_heads, head_dim].
            - lse: Log-sum-exp of attention scores of shape
              [batch, seq_len, num_q_heads], used for subsequent
              top-k block selection.
    """
    _batch, seq_len, num_q_heads, _head_dim = query.shape
    num_kv_heads = k_cmp.shape[2]
    group_size = num_q_heads // num_kv_heads
    num_blocks = k_cmp.shape[1]

    k_cmp_expanded = jnp.repeat(k_cmp, group_size, axis=2)
    v_cmp_expanded = jnp.repeat(v_cmp, group_size, axis=2)

    scores = jnp.einsum("bsnd,bmnd->bsnm", query, k_cmp_expanded) * softmax_scale

    t_ids = jnp.arange(seq_len, dtype=jnp.int32)
    s_completed = (t_ids + 1) // block_size
    c_ids = jnp.arange(num_blocks, dtype=jnp.int32)

    block_mask = c_ids[None, :] < s_completed[:, None]
    block_mask = block_mask[None, :, None, :]

    scores_masked = jnp.where(block_mask, scores, -jnp.inf)

    lse_raw = logsumexp(scores_masked, axis=-1)
    lse = jnp.where(s_completed[None, :, None] > 0, lse_raw, 0.0)

    p = jnp.exp(scores - lse[..., None])
    p = jnp.where(block_mask, p, 0.0)

    p_sum = p.sum(axis=-1)
    o_num = jnp.einsum("bsnm,bmnd->bsnd", p, v_cmp_expanded)
    output = jnp.where(
        (p_sum > 0)[..., None],
        o_num / jnp.maximum(p_sum, 1e-9)[..., None],
        jnp.zeros_like(o_num),
    )

    return output, lse


def _nsa_topk_xla(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    k_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    lse: Float[Array, "batch seq_len num_q_heads"],
    block_counts: int,
    block_size: int,
    softmax_scale: float,
) -> Int[Array, "batch seq_len num_kv_heads num_selected_blocks"]:
    """Select top-k blocks for each query token based on attention scores.

    Implements the block selection algorithm for Native Sparse Attention:
    1. Compute attention probabilities using compressed keys and LSE
    2. Force-include the current block (probability = 1.0)
    3. Sum probabilities across query head groups for GQA
    4. Select top-k blocks based on aggregated scores

    This matches Triton semantics where ties are broken by block index.

    Args:
        query: Query tensor [batch, seq_len, num_q_heads, head_dim]
        k_cmp: Compressed (mean-pooled) keys [batch, num_blocks, num_kv_heads, head_dim]
        lse: Log-sum-exp from compressed attention [batch, seq_len, num_q_heads]
        block_counts: Number of blocks to select per query token
        block_size: Size of each attention block
        softmax_scale: Attention scaling factor

    Returns:
        Block indices for each token [batch, seq_len, num_kv_heads, num_selected_blocks]
        Invalid blocks (beyond current position) are marked with -1.
    """
    B, T, HQ, _D = query.shape
    C = k_cmp.shape[1]
    H = k_cmp.shape[2]
    G = HQ // H

    k_cmp_exp = jnp.repeat(k_cmp, G, axis=2)
    k_cmp_exp = jnp.swapaxes(k_cmp_exp, 1, 2)

    scores = jnp.einsum("bthd,bhcd->bthc", query, k_cmp_exp) * softmax_scale

    qb = jnp.arange(T, dtype=jnp.int32) // block_size
    c_ids = jnp.arange(C, dtype=jnp.int32)
    mask = c_ids[None, None, None, :] <= qb[None, :, None, None]
    scores = jnp.where(mask, scores, -jnp.inf)

    p = jnp.exp(scores - lse[..., None])

    one_hot_qb = jax.nn.one_hot(qb, C, dtype=p.dtype)
    p = jnp.where(one_hot_qb[None, :, None, :] > 0, 1.0, p)

    p_sum = p.reshape(B, T, H, G, C).sum(axis=3)
    future_mask = jnp.arange(C, dtype=jnp.int32)[None, None, None, :] > qb[None, :, None, None]
    p_sum = jnp.where(future_mask, -jnp.inf, p_sum)

    tie = jnp.arange(C, dtype=p_sum.dtype)[None, None, None, :] * jnp.array(1e-8, dtype=p_sum.dtype)
    p_sum_adj = p_sum + tie
    S = block_counts if isinstance(block_counts, int) else int(block_counts)
    S = min(int(S), int(C))
    p_flat = p_sum_adj.reshape(B * T * H, C)
    if isinstance(p_flat, jax.Array) and p_flat.device.platform in ("gpu", "cuda"):
        cpu = jax.devices("cpu")[0]
        p_cpu = jax.device_put(p_flat, cpu)
        _, idx_cpu = jax.lax.top_k(p_cpu, S)
        idx_flat = jax.device_put(idx_cpu, p_flat.device)
    else:
        _, idx_flat = jax.lax.top_k(p_flat, S)
    idx = idx_flat.reshape(B, T, H, S).astype(jnp.int32)
    idx = jnp.where(idx <= qb[None, :, None, None], idx, jnp.full_like(idx, -1))
    return idx


@kernel_registry.register("apply_native_sparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def apply_native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: (
        Int[Array, "batch seq_len num_kv_heads num_selected_blocks"]
        | Int[Array, "batch num_kv_heads num_blocks num_selected_blocks"]
    ),
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | Int[Array, "batch num_kv_heads num_blocks"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    token_indices: Int[Array, "total_tokens"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Apply block-sparse attention with a pre-computed sparsity pattern using JAX/XLA.

    Computes sparse attention where each query token attends only to the subset
    of K/V blocks prescribed by ``block_indices``.  This reduces the cost from
    O(seq_len²) to O(seq_len * num_selected_blocks * block_size).

    Internally, both K and V are padded to the next multiple of ``block_size``
    and reshaped into blocks.  Attention is then computed per-token by gathering
    the selected blocks and applying causal masking followed by softmax.

    The custom VJP (``_sparse_attention_with_vjp``) is used so that gradients
    flow only through the differentiable inputs (Q, K, V); ``block_indices`` and
    ``block_counts`` receive ``None`` gradients.

    Args:
        query: Query tensor ``[batch, seq_len, num_q_heads, head_dim]``.
        key: Key tensor ``[batch, seq_len, num_kv_heads, head_dim]``.  GQA is
            supported: ``num_q_heads`` must be a multiple of ``num_kv_heads``.
        value: Value tensor ``[batch, seq_len, num_kv_heads, head_dim]``.
        block_indices: Block selection indices.  Two layouts are accepted:

            - Per-token: ``[batch, seq_len, num_kv_heads, num_selected_blocks]``
            - Per-query-block: ``[batch, num_kv_heads, num_blocks, num_selected_blocks]``
              (automatically expanded to per-token with ``num_blocks =
              ceil(seq_len / block_size)``).

            Values of ``-1`` are treated as sentinel (invalid) entries and are
            masked out regardless of ``block_counts``.
        block_counts: Number of valid blocks per query position.  Can be:

            - ``int``: uniform count for all positions and heads.
            - ``[batch, seq_len, num_kv_heads]``: per-token counts.
            - ``[batch, num_kv_heads, num_blocks]``: per-query-block counts
              (expanded to per-token).
        block_size: Size of each block in tokens (static; determines tiling).
        softmax_scale: QK scaling factor.  Defaults to ``1 / sqrt(head_dim)``.
        cu_seqlens: Not supported on this XLA path; raises ``NotImplementedError``
            if provided.
        token_indices: Not supported on this XLA path; raises ``NotImplementedError``
            if provided.
        block_k: Accepted for API compatibility with Triton; ignored by XLA.
        block_v: Accepted for API compatibility with Triton; ignored by XLA.
        num_warps: Accepted for API compatibility with Triton; ignored by XLA.
        num_stages: Accepted for API compatibility with Triton; ignored by XLA.

    Returns:
        Attention output ``[batch, seq_len, num_q_heads, head_dim]``.

    Raises:
        NotImplementedError: If ``cu_seqlens`` is not None.
        NotImplementedError: If ``token_indices`` is not None.
        ValueError: If ``block_indices`` is not a 4-D array.
        ValueError: If the shape of ``block_indices`` does not match either
            accepted layout for the given ``seq_len``, ``num_kv_heads``, and
            ``block_size``.
        ValueError: If ``block_counts`` is not an int or a 3-D array matching
            one of the accepted layouts.

    Examples:
        >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        >>> block_size = 64
        >>> num_blocks = seq_len // block_size
        >>>
        >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>>
        >>>
        >>> block_counts = 4
        >>> block_indices = jnp.tile(
        ...     jnp.arange(4)[None, None, None, :],
        ...     (batch, num_heads, num_blocks, 1)
        ... )
        >>>
        >>> output = apply_native_sparse_attention(
        ...     query, key, value, block_indices, block_counts, block_size
        ... )
        >>> output.shape
        (2, 1024, 8, 64)

        >>>
        >>> def create_local_pattern(num_blocks, window=2):
        ...     indices = []
        ...     for i in range(num_blocks):
        ...         local = list(range(max(0, i-window), min(num_blocks, i+window+1)))
        ...
        ...         local = local + [0] * (window*2+1 - len(local))
        ...         indices.append(local)
        ...     return jnp.array(indices)
        >>>
        >>> local_indices = create_local_pattern(num_blocks, window=2)
        >>> local_indices = jnp.tile(local_indices[None, None, :, :], (batch, num_heads, 1, 1))
        >>> output = apply_native_sparse_attention(
        ...     query, key, value, local_indices, block_counts=5, block_size=block_size
        ... )
    """
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens is not supported in XLA apply_native_sparse_attention implementation")
    if token_indices is not None:
        raise NotImplementedError("token_indices is not supported in XLA apply_native_sparse_attention implementation")

    if softmax_scale is None:
        softmax_scale = float(1.0 / jnp.sqrt(query.shape[-1]))
    else:
        softmax_scale = float(softmax_scale)

    batch = query.shape[0]
    seq_len = query.shape[1]
    num_kv_heads = key.shape[2]
    num_blocks = (seq_len + block_size - 1) // block_size

    block_indices_arr = jnp.asarray(block_indices, dtype=jnp.int32)
    if block_indices_arr.ndim != 4:
        raise ValueError(f"block_indices must be a 4D int32 array, got shape {block_indices_arr.shape}.")

    if block_indices_arr.shape[1] == seq_len and block_indices_arr.shape[2] == num_kv_heads:
        block_indices_tok = block_indices_arr
    elif block_indices_arr.shape[1] == num_kv_heads and block_indices_arr.shape[2] == num_blocks:
        qb = (jnp.arange(seq_len, dtype=jnp.int32) // block_size).astype(jnp.int32)
        block_indices_blk = jnp.transpose(block_indices_arr, (0, 2, 1, 3))
        block_indices_tok = block_indices_blk[:, qb, :, :]
    else:
        raise ValueError(
            "block_indices must have shape (batch, seq_len, num_kv_heads, num_selected_blocks) "
            "or (batch, num_kv_heads, num_blocks, num_selected_blocks). "
            f"Got shape {block_indices_arr.shape} for seq_len={seq_len}, num_kv_heads={num_kv_heads}, "
            f"num_blocks={num_blocks}, block_size={block_size}."
        )

    if isinstance(block_counts, int):
        block_counts_tok = jnp.full((batch, seq_len, num_kv_heads), int(block_counts), dtype=jnp.int32)
    else:
        block_counts_arr = jnp.asarray(block_counts, dtype=jnp.int32)
        if block_counts_arr.ndim != 3:
            raise ValueError(f"block_counts must be an int or a 3D int32 array, got shape {block_counts_arr.shape}.")
        if block_counts_arr.shape == (batch, seq_len, num_kv_heads):
            block_counts_tok = block_counts_arr
        elif block_counts_arr.shape == (batch, num_kv_heads, num_blocks):
            qb = (jnp.arange(seq_len, dtype=jnp.int32) // block_size).astype(jnp.int32)
            block_counts_blk = jnp.transpose(block_counts_arr, (0, 2, 1))
            block_counts_tok = block_counts_blk[:, qb, :]
        else:
            raise ValueError(
                "block_counts must have shape (batch, seq_len, num_kv_heads) or (batch, num_kv_heads, num_blocks). "
                f"Got shape {block_counts_arr.shape} for seq_len={seq_len}, num_kv_heads={num_kv_heads}, "
                f"num_blocks={num_blocks}, block_size={block_size}."
            )

    return _sparse_attention_with_vjp(query, key, value, block_indices_tok, block_counts_tok, block_size, softmax_scale)


@kernel_registry.register("native_sparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
    g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Compute Native Sparse Attention (NSA) using JAX/XLA.

    NSA combines two attention components:

    1. **Compressed attention** (only when ``g_cmp`` is provided): queries
       attend to mean-pooled (block-compressed) key-value representations,
       providing coarse global context.  The log-sum-exp from this step is
       also used to select the most relevant blocks for step 2.

    2. **Selected sparse attention**: queries attend to a small set of
       full-resolution K/V blocks chosen either by top-k block selection
       (when ``g_cmp`` is given) or by the caller-supplied ``block_indices``.

    The final output is computed as::

        o = o_slc * g_slc  (if g_slc provided, else o = o_slc)
        o = o + o_cmp * g_cmp  (only when g_cmp provided)

    where ``g_cmp`` and ``g_slc`` are element-wise gate scalars broadcast over
    ``head_dim``.

    Note:
        On the GPU path, top-k block selection temporarily moves data to CPU to
        work around an XLA:GPU cache-reuse crash in some JAX builds.

    Args:
        query: Query tensor ``[batch, seq_len, num_q_heads, head_dim]``.
        key: Key tensor ``[batch, seq_len, num_kv_heads, head_dim]``.  GQA is
            supported: ``num_q_heads`` must be a multiple of ``num_kv_heads``.
        value: Value tensor ``[batch, seq_len, num_kv_heads, head_dim]``.
        g_cmp: Gate for the compressed attention output.  Shape
            ``[batch, seq_len, num_q_heads]``.  When provided the compressed
            attention path is active and ``block_indices`` is ignored (a
            warning is emitted if it is also passed).  When None, the
            compressed path is skipped entirely and ``block_indices`` is
            required.
        g_slc: Gate for the selected sparse attention output.  Shape
            ``[batch, seq_len, num_q_heads]``.  When None the selected
            output is used unscaled.
        block_indices: Pre-computed block selection indices.  Required when
            ``g_cmp`` is None.  Two layouts accepted:

            - Per-token: ``[batch, seq_len, num_kv_heads, num_selected_blocks]``
            - Per-query-block: ``[batch, num_kv_heads, num_blocks, num_selected_blocks]``

        block_counts: Number of K/V blocks each query attends to.  Can be:

            - ``int``: uniform for all positions (default ``16``).
            - ``[batch, seq_len, num_kv_heads]``: per-token counts.
            - ``[batch, num_kv_heads, num_blocks]``: per-query-block counts.
        block_size: Tokens per K/V block.  Defaults to ``64``.
        softmax_scale: QK scaling factor.  Defaults to ``1 / sqrt(head_dim)``.
        cu_seqlens: Cumulative sequence lengths ``[num_seqs + 1]`` for packed
            variable-length inputs.  When provided a warning is issued if
            ``batch != 1``.  Variable-length support is incomplete in the XLA
            implementation.
        block_k: Accepted for API compatibility with Triton; ignored by XLA.
        block_v: Accepted for API compatibility with Triton; ignored by XLA.
        num_warps: Accepted for API compatibility with Triton; ignored by XLA.
        num_stages: Accepted for API compatibility with Triton; ignored by XLA.

    Returns:
        Attention output ``[batch, seq_len, num_q_heads, head_dim]``.

    Raises:
        ValueError: If neither ``g_cmp`` nor ``block_indices`` is provided.

    Examples:
        >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        >>> block_size = 64
        >>> block_counts = 16
        >>>
        >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>>
        >>>
        >>> g_cmp = jnp.ones((batch, seq_len, num_heads * head_dim))
        >>> output = native_sparse_attention(
        ...     query, key, value, g_cmp=g_cmp, block_counts=block_counts, block_size=block_size
        ... )
        >>> output.shape
        (2, 1024, 8, 64)
        >>>
        >>>
        >>> num_blocks = seq_len // block_size
        >>> block_indices = jnp.tile(
        ...     jnp.arange(block_counts)[None, None, None, :],
        ...     (batch, num_heads, num_blocks, 1)
        ... )
        >>> output = native_sparse_attention(
        ...     query, key, value, block_indices=block_indices, block_counts=block_counts, block_size=block_size
        ... )
        >>> output.shape
        (2, 1024, 8, 64)
    """
    if softmax_scale is None:
        softmax_scale = float(1.0 / jnp.sqrt(query.shape[-1]))
    else:
        softmax_scale = float(softmax_scale)

    if cu_seqlens is not None:
        batch_size = query.shape[0]
        if batch_size != 1:
            warnings.warn(
                "cu_seqlens with batch_size != 1 may not work correctly in XLA implementation. "
                "Consider using batch_size=1 for variable-length sequences.",
                stacklevel=2,
            )

    batch, seq_len, _num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    num_blocks = (seq_len + block_size - 1) // block_size

    pad_len = num_blocks * block_size - seq_len
    if pad_len > 0:
        k_padded = jnp.pad(key, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v_padded = jnp.pad(value, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    else:
        k_padded = key
        v_padded = value

    k_cmp = k_padded.reshape(batch, num_blocks, block_size, num_kv_heads, head_dim).mean(axis=2)
    v_cmp = v_padded.reshape(batch, num_blocks, block_size, num_kv_heads, head_dim).mean(axis=2)
    o_cmp = None

    if g_cmp is not None:
        o_cmp, lse_cmp = _nsa_compression_xla(
            query=query,
            k_cmp=k_cmp,
            v_cmp=v_cmp,
            block_size=block_size,
            softmax_scale=softmax_scale,
        )
        if block_indices is not None:
            warnings.warn(
                "`block_indices` will be ignored when `g_cmp` is provided",
                stacklevel=2,
            )

        block_indices = _nsa_topk_xla(
            query=query,
            k_cmp=k_cmp,
            lse=lse_cmp,
            block_counts=block_counts if isinstance(block_counts, int) else int(block_counts[0, 0, 0]),
            block_size=block_size,
            softmax_scale=softmax_scale,
        )
    if block_indices is None:
        raise ValueError("Either `g_cmp` must be provided or `block_indices` must be passed.")

    o_slc = apply_native_sparse_attention(
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        softmax_scale=softmax_scale,
    )

    o = o_slc
    if g_slc is not None:
        o = o_slc * jnp.expand_dims(g_slc, -1)

    if o_cmp is not None and g_cmp is not None:
        o = o + o_cmp * jnp.expand_dims(g_cmp, -1)

    return o
