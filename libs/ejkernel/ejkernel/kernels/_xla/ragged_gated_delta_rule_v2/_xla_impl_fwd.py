# Copyright 2026 Google LLC
# Copyright EasyDeL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Packed (ragged) Gated Delta Rule kernels for continuous-batching inference.

This module provides the portable JAX/XLA implementation of the Gated Delta
Rule (GDR) recurrence for the eSurge packed-inference path, where many
requests of heterogeneous lengths share a single contiguous token buffer. It exposes:

* :func:`ragged_gated_delta_rule` - the top-level JIT entry point that splits
  an interleaved ``mixed_qkv`` stream into Q/K/V, optionally repeats heads
  for grouped-query layouts, and dispatches to either the decode-only fast
  path or the chunked mixed-prefill branch based on the ``request_distribution``.
* :func:`ragged_gated_delta_rule_decode_only` - per-token JAX update
  used when every active request consumes exactly one new token.
* :func:`ragged_gated_delta_rule_mixed_prefill` - chunked algorithm that
  pads each request's tokens to a multiple of ``chunk_size``, runs the
  intra-chunk attention in parallel, and propagates inter-chunk state via
  ``lax.scan``. Requires the unit lower-triangular inverse provided by
  :class:`TriangleSolverImpl`.
* Helpers for the unit lower-triangular inverse used inside the chunked
  formulation via a portable ``jax.scipy``-based path.

Algorithmic notes:
- All forward arithmetic is performed in float32 for numerical stability and
  cast back to ``mixed_qkv.dtype`` (typically bfloat16) on the way out.
- The chunked path uses an online-style update: per-chunk ``q @ k^T`` is
  weighted by ``exp(g_diff)`` to mix the gated decay into the attention
  pattern, then the inter-chunk recurrence carries the running ``state``
  through a ``lax.scan``.
- The decode-only path uses a pure-JAX gather/compute/scatter implementation.
"""

import enum

import jax
import jax.numpy as jnp
from jax import lax


def _reorder_concatenated_tensor_for_sharding(
    concatenated_tensor: jax.Array,
    split_sizes: tuple[int, ...],
    n_shards: int,
    dim: int,
) -> jax.Array:
    """Reorder a fused feature axis so per-shard slices are interleaved.

    Rearranges a tensor whose ``dim`` axis is the concatenation of several
    logical sub-features ``[A|B|C|...]`` into the per-shard interleaved
    layout ``[A0|B0|C0|A1|B1|C1|...]`` expected by a sharded consumer. Each
    sub-feature is split into ``n_shards`` equal slabs along ``dim`` and the
    slabs are regrouped shard-by-shard.

    Args:
        concatenated_tensor: Source tensor whose ``dim`` axis is the
            concatenation of the ``split_sizes`` sub-features.
        split_sizes: Lengths along ``dim`` of each fused sub-feature, in
            order; must sum to ``concatenated_tensor.shape[dim]`` and each
            entry must be divisible by ``n_shards``.
        n_shards: Number of shards the ``dim`` axis is divided into.
        dim: Axis along which the fused sub-features are concatenated;
            negative values are normalized against ``ndim``.

    Returns:
        jax.Array: Tensor with the same shape and dtype as
        ``concatenated_tensor`` whose ``dim`` axis is reordered into the
        per-shard interleaved layout.
    """
    if dim < 0:
        dim += concatenated_tensor.ndim
    old_shape = concatenated_tensor.shape
    new_shape = (*old_shape[:dim], int(n_shards), -1, *old_shape[dim + 1 :])
    split_tensors = []
    start_offset = 0
    for split_size in split_sizes:
        split_tensor = jax.lax.slice_in_dim(
            concatenated_tensor,
            start_offset,
            start_offset + int(split_size),
            axis=dim,
        )
        split_tensors.append(split_tensor.reshape(new_shape))
        start_offset += int(split_size)
    reordered_tensor = jnp.concatenate(split_tensors, axis=dim + 1)
    return reordered_tensor.reshape(old_shape)


def newton_schulz_inverse_ref(A, n=None):
    """Reference Newton-Schulz inverse for unit lower-triangular matrices.

    Computes :math:`A^{-1}` for a batch of unit lower-triangular ``N x N``
    matrices using the Newton-Schulz iteration
    :math:`S_{k+1} = S_k (2 I - A S_k)`. With :math:`L = A - I` strictly
    lower-triangular, the recurrence is mathematically equivalent to the
    finite product :math:`S_k = (I - L) \\prod_{j=1}^{k}(I + L^{2^j})`,
    which terminates exactly after :math:`\\lceil \\log_2 N \\rceil`
    doublings because :math:`L^{N} = 0`.

    For numerical stability the implementation does *not* materialise the
    closed-form product; instead it iterates :math:`S \\leftarrow S (2 I -
    A S)` with the matmul running at ``Precision.HIGHEST`` so the final
    step is performed at full precision while the loop body remains
    accurate enough for bfloat16/float16 inputs.

    Args:
        A: Array of shape ``(..., N, N)`` whose last two dimensions form a
            unit lower-triangular matrix (1s on the diagonal, zeros above).
            Higher leading dimensions are batched.
        n: Optional iteration upper bound; defaults to ``A.shape[-1]``.
            The loop doubles ``k`` each iteration so any value at or above
            ``ceil(log2(N))`` produces the exact inverse.

    Returns:
        jnp.ndarray: A tensor with the same shape and dtype as ``A``
        containing the inverse of each unit lower-triangular slab.
    """
    if n is None:
        n = A.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=A.dtype), A.shape)
    S = 2 * eye - A
    k = 1
    while k < n:
        precision = jax.lax.Precision.HIGHEST
        k *= 2
        I_plus_error = 2 * eye - jnp.matmul(A, S, precision=precision)
        S = jnp.matmul(S, I_plus_error, precision=precision)
    return S


def triangular_inverse_jax(A):
    """Backend-agnostic unit lower-triangular inverse via ``jax.scipy``.

    Computes :math:`A^{-1}` column-by-column using
    :func:`jax.scipy.linalg.solve_triangular` against an identity batch.
    Acts as the portable path selected by :class:`TriangleSolverImpl`.

    Args:
        A: Tensor of shape ``(..., N, N)`` whose last two dimensions form
            unit lower-triangular matrices. Leading dimensions are
            preserved.

    Returns:
        jnp.ndarray: Inverse with the same shape and dtype as ``A``.
    """
    shape = A.shape
    A_2d = A.reshape(-1, shape[-2], shape[-1])
    N = shape[-1]
    I_batch = jnp.broadcast_to(jnp.eye(N, dtype=A.dtype), A_2d.shape)
    X = jax.scipy.linalg.solve_triangular(A_2d, I_batch, lower=True, unit_diagonal=True)
    return X.reshape(shape)


class TriangleSolverImpl(enum.StrEnum):
    """Selector for the unit lower-triangular inverse implementation.

    A string enum used to choose which backend computes the per-chunk
    unit lower-triangular inverse inside the chunked GDR prefill. Calling
    an instance dispatches to the corresponding implementation, letting the
    enum value double as a callable solver.

    Attributes:
        JAX: The ``"jax"`` member selecting the portable
            :func:`triangular_inverse_jax` path (``jax.scipy``-based).
    """

    JAX = "jax"

    def __call__(self, A):
        """Invoke the selected unit lower-triangular inverse implementation.

        Args:
            A: Tensor of shape ``(..., N, N)`` whose last two dimensions are
                unit lower-triangular matrices to invert.

        Returns:
            jnp.ndarray: The inverse of each slab, same shape and dtype as
            ``A``, computed via :func:`triangular_inverse_jax`.
        """
        return triangular_inverse_jax(A)


def l2norm(x: jnp.ndarray, dim: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    """Normalize ``x`` to unit L2 norm along ``dim``.

    Implementation uses ``rsqrt`` for stability:
    ``inv_norm = rsqrt(sum(x*x, axis=dim) + eps)`` then
    ``x_normalized = x * inv_norm``. The epsilon is added inside the
    rsqrt to keep the operation well-defined for zero-magnitude inputs.

    Args:
        x: Input tensor; any shape and floating dtype is accepted.
        dim: Axis to reduce; defaults to the trailing axis.
        eps: Stability epsilon mixed in before the rsqrt to bound the
            output magnitude when ``x`` is identically zero. Small values
            (~1e-6) preserve numerical fidelity for normal inputs.

    Returns:
        jnp.ndarray: Same shape and dtype as ``x``, with the squared
        elements along ``dim`` summing to approximately 1.
    """
    inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))
    return x * inv_norm


def pack_inputs_single_stream(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    chunk_size: int,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Pads each sequence to multiple of chunk_size and concatenates.

    This function takes ragged sequences and pads each of them so that their
    lengths become a multiple of `chunk_size`. It then concatenates these
    padded sequences into a single continuous stream. This allows for efficient
    chunk-based processing on hardware like TPUs, where fixed-size operations
    are preferred.

    It also computes a `reset_mask` to indicate where a new sequence starts
    (aligned to chunk boundaries), which is used to reset the recurrent state
    during processing.

    Example:
      Original sequences (ragged):
      Seq 1: [A, A, A] (len 3)
      Seq 2: [B, B, B, B, B] (len 5)
      Seq 3: [C, C] (len 2)

      Packed stream (chunk_size=4):
      Chunk 1: [A, A, A, P]  <- Seq 1 padded (New sequence starts)
      Chunk 2: [B, B, B, B]  <- Seq 2 (part 1) (New sequence starts)
      Chunk 3: [B, P, P, P]  <- Seq 2 (part 2) padded
      Chunk 4: [C, C, P, P]  <- Seq 3 padded (New sequence starts)
      (where 'P' denotes padding)

      reset_mask = [True, True, False, True]
      (Indicates whether each chunk starts a new sequence)

    Args:
        query: Ragged queries of shape ``(num_tokens, num_heads, d_k)`` in
            the unpadded original stream.
        key: Ragged keys with the same layout as ``query``.
        value: Ragged values of shape ``(num_tokens, num_heads, d_v)``.
        g: Per-token log-space gate of shape ``(num_tokens, num_heads)``,
            float32.
        beta: Per-token gating coefficient of shape
            ``(num_tokens, num_heads)``.
        query_start_loc: Cumulative per-request token offsets of shape
            ``(num_requests + 1,)`` describing the request boundaries in
            the original stream.
        distribution: ``(decode_end, prefill_end, total)`` int32 triple;
            only the third entry (number of valid requests) is consumed
            here to mask out trailing inactive slots.
        chunk_size: Pad each request to a multiple of this size.
        compute_dtype: Dtype to cast Q/K/V/beta into for the chunked
            kernel (typically ``bfloat16``); ``g`` stays in float32.

    Returns:
        tuple: ``(packed_query, packed_key, packed_value, packed_g,
        packed_beta, reset_mask, new_query_start_loc, padded_indices_valid)``
        where the packed tensors live in the chunked stream of length
        ``num_chunks * chunk_size``, ``reset_mask`` is a boolean array of
        shape ``(num_chunks,)`` true at chunk boundaries that start a
        fresh request, ``new_query_start_loc`` describes request
        boundaries in the packed stream, and ``padded_indices_valid``
        gives per-original-token indices into the packed buffer for use
        when scattering outputs back.
    """
    num_tokens = query.shape[0]
    num_seqs = len(query_start_loc) - 1

    num_valid_seqs = jnp.clip(distribution[2], 0, num_seqs)
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)

    seq_lengths = jnp.maximum(effective_query_start_loc[1:] - effective_query_start_loc[:-1], 0)
    num_chunks = (seq_lengths + chunk_size - 1) // chunk_size
    padded_lengths = num_chunks * chunk_size

    new_query_start_loc = jnp.cumsum(jnp.concatenate([jnp.array([0]), padded_lengths]))
    token_ids = jnp.arange(num_tokens)
    seq_id = jnp.searchsorted(effective_query_start_loc, token_ids, side="right", method="compare_all") - 1
    valid_token_mask = (seq_id >= 0) & (seq_id < num_valid_seqs) & (token_ids < last_valid_loc)
    safe_seq_id = jnp.clip(seq_id, 0, max(num_seqs - 1, 0))
    original_start = effective_query_start_loc[safe_seq_id]
    new_start = new_query_start_loc[safe_seq_id]
    padded_indices_raw = new_start + (token_ids - original_start)

    max_packed_tokens = num_tokens + num_seqs * chunk_size
    max_packed_tokens = (max_packed_tokens + chunk_size - 1) // chunk_size * chunk_size
    scatter_indices = jnp.where(valid_token_mask, padded_indices_raw, max_packed_tokens)
    padded_indices_valid = jnp.where(valid_token_mask, padded_indices_raw, 0)

    # Concatenate by dtype to reduce scatter operations
    beta_expanded = beta[..., None]

    combined_qkvb = jnp.concatenate(
        [
            query.astype(compute_dtype),
            key.astype(compute_dtype),
            value.astype(compute_dtype),
            beta_expanded.astype(compute_dtype),
        ],
        axis=-1,
    )

    output_shape = (max_packed_tokens, *combined_qkvb.shape[1:])
    packed_combined_qkvb = jnp.zeros(output_shape, dtype=compute_dtype)
    packed_combined_qkvb = packed_combined_qkvb.at[scatter_indices].set(
        combined_qkvb,
        mode="drop",
        wrap_negative_indices=False,
    )

    K_dim = query.shape[2]
    V_dim = value.shape[2]
    packed_query = packed_combined_qkvb[..., :K_dim]
    packed_key = packed_combined_qkvb[..., K_dim : 2 * K_dim]
    packed_value = packed_combined_qkvb[..., 2 * K_dim : 2 * K_dim + V_dim]
    packed_beta = packed_combined_qkvb[..., 2 * K_dim + V_dim]

    # For g (float32)
    output_shape_f32 = (max_packed_tokens, *g.shape[1:])
    packed_g = jnp.zeros(output_shape_f32, dtype=jnp.float32)
    packed_g = packed_g.at[scatter_indices].set(
        g.astype(jnp.float32),
        mode="drop",
        wrap_negative_indices=False,
    )

    num_chunks_total = max_packed_tokens // chunk_size
    reset_mask = jnp.zeros((num_chunks_total,), dtype=bool)
    start_chunk_indices = jnp.where(seq_lengths > 0, new_query_start_loc[:-1] // chunk_size, num_chunks_total)
    reset_mask = reset_mask.at[start_chunk_indices].set(True, mode="drop", wrap_negative_indices=False)

    return (
        packed_query,
        packed_key,
        packed_value,
        packed_g,
        packed_beta,
        reset_mask,
        new_query_start_loc,
        padded_indices_valid,
    )


def ragged_gated_delta_rule_mixed_prefill(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    b_reshaped: jnp.ndarray,
    a_reshaped: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    has_initial_state: jnp.ndarray | None,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    mask_initial_state: bool = False,
    compute_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    preferred_element_type: jnp.dtype = jnp.float32,
    triangle_solver_impl: TriangleSolverImpl | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies chunked gated delta rule for mixed prefill case.

    This function handles the case where sequences can have lengths greater than
    1.
    It pads sequences to multiples of `chunk_size` and processes them in parallel
    within chunks, and sequentially across chunks.

    Args:
        query: Ragged queries ``(num_tokens, n_v, d_k)`` already expanded
            from the grouped layout (so head count matches ``n_v``).
        key: Ragged keys with the same shape as ``query``.
        value: Ragged values ``(num_tokens, n_v, d_v)``.
        b_reshaped: Pre-sigmoid beta source ``(num_tokens, n_v)``;
            sigmoided in-place to produce the gating coefficient.
        a_reshaped: Pre-softplus alpha source ``(num_tokens, n_v)``;
            combined with ``A_log`` and ``dt_bias`` to form the log-space
            decay.
        A_log: Per-head log-decay parameter ``(n_v,)``.
        dt_bias: Per-head delta-time bias ``(n_v,)``.
        query_start_loc: Cumulative per-request offsets in the original
            stream, shape ``(num_requests + 1,)``.
        recurrent_state: Global state pool of shape
            ``(num_blocks, n_v, d_k, d_v)`` where ``num_blocks >= max_reqs
            + 1``; the first block is a null block reserved for padded /
            invalid tokens.
        state_indices: Per-request mapping into ``recurrent_state``,
            shape ``(num_requests,)``.
        distribution: Triple ``[decode_end, prefill_end, total]``; the
            third entry bounds the positional row prefix whose slots are
            written. Empty (``q_len == 0``) rows in it keep their slot.
        has_initial_state: Boolean tensor marking requests whose recurrent
            state slot should be used as the initial prefill state. Requests
            without initial state start from zeros even if the slot contains
            stale state from a previous sequence.
        chunk_size: Padding and chunking granularity for the parallel
            intra-chunk attention.
        use_qk_norm_in_gdn: Whether to L2-normalize Q and K before the
            chunked attention.
        mask_initial_state: When ``True``, gate the loaded recurrent state with
            ``has_initial_state``. Keep ``False`` for the common all-valid path
            so the compiled prefill stays on the cheaper state-load lowering.
        compute_dtype: Dtype for the chunked Q/K/V/beta tensors.
        precision: ``lax.Precision`` for the matmul calls; defaults to
            ``HIGHEST`` to keep numerical stability across long
            recurrences.
        preferred_element_type: Accumulation dtype for ``jnp.matmul``.
        triangle_solver_impl: Selector for the unit lower-triangular
            inverse used to solve for the per-chunk attention weights.
            Defaults to the portable JAX path.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` where
        ``updated_recurrent_state`` has shape
        ``(num_blocks, n_v, d_k, d_v)`` and ``output`` has shape
        ``(num_tokens, n_v * d_v)`` cast back to ``query.dtype``.
    """
    if triangle_solver_impl is None:
        triangle_solver_impl = TriangleSolverImpl.JAX

    initial_dtype = query.dtype

    beta = jax.nn.sigmoid(b_reshaped)
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a_reshaped.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )

    (
        packed_query,
        packed_key,
        packed_value,
        packed_g,
        packed_beta,
        _,
        new_query_start_loc,
        padded_indices_valid,
    ) = pack_inputs_single_stream(
        query,
        key,
        value,
        g,
        beta,
        query_start_loc,
        distribution,
        chunk_size,
        compute_dtype=compute_dtype,
    )

    if use_qk_norm_in_gdn:
        packed_query = l2norm(packed_query, dim=-1, eps=1e-6)
        packed_key = l2norm(packed_key, dim=-1, eps=1e-6)

    scale = jax.lax.rsqrt(jnp.array(packed_query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
    packed_query = packed_query * scale

    total_tokens = packed_query.shape[0]
    num_chunks = total_tokens // chunk_size
    H = packed_query.shape[1]
    K_dim = packed_query.shape[2]
    V_dim = packed_value.shape[2]

    def to_chunk(x):
        """Reshape ``[total, H, D]`` into ``[num_chunks, H, chunk_size, D]``.

        Args:
            x: 3-D packed tensor with token-major layout.

        Returns:
            jax.Array: 4-D chunked, head-major view of ``x``.
        """
        return x.reshape(num_chunks, chunk_size, H, -1).transpose(0, 2, 1, 3)

    def to_chunk_scalar(x):
        """Reshape ``[total, H]`` into ``[num_chunks, H, chunk_size]``.

        Per-token scalar variant of :func:`to_chunk` used for ``g`` and
        ``beta``.

        Args:
            x: 2-D packed tensor.

        Returns:
            jax.Array: 3-D chunked, head-major view of ``x``.
        """
        return x.reshape(num_chunks, chunk_size, H).transpose(0, 2, 1)

    q_c = to_chunk(packed_query)
    k_c = to_chunk(packed_key)
    v_c = to_chunk(packed_value)
    g_c = to_chunk_scalar(packed_g)
    beta_c = to_chunk_scalar(packed_beta)

    # STAGE 2: INTRA-CHUNK PRE-COMPUTATION
    g_cumsum = jnp.cumsum(g_c, axis=-1)
    k_beta = k_c * beta_c[..., None]

    S = jnp.matmul(
        k_beta,
        k_c.swapaxes(-1, -2),
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    S = S.astype(jnp.float32)

    g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
    g_diff = jnp.where(mask, g_diff, -1e30)

    S = S * jnp.exp(g_diff)
    S = jnp.where(mask, S, 0.0)

    identity = jnp.eye(chunk_size, dtype=jnp.float32)

    A = triangle_solver_impl(identity + S)

    v_beta = v_c * beta_c[..., None]
    u_chunks = jnp.matmul(
        A,
        v_beta.astype(jnp.float32),
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    u_chunks = u_chunks.astype(compute_dtype)

    k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
    w_chunks = jnp.matmul(
        A,
        k_beta_g,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    w_chunks = w_chunks.astype(compute_dtype)

    attn_chunks = jnp.matmul(
        q_c,
        k_c.swapaxes(-1, -2),
        precision=precision,
        preferred_element_type=preferred_element_type,
    ).astype(jnp.float32)
    g_diff_chunks = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    g_diff_chunks = jnp.where(mask_intra, g_diff_chunks, -1e30)
    attn_i_chunks = jnp.where(mask_intra, attn_chunks * jnp.exp(g_diff_chunks), 0.0).astype(compute_dtype)

    q_g_chunks = (q_c.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]).astype(compute_dtype)
    g_i_last_exp_chunks = jnp.exp(g_cumsum[..., -1, None, None])
    g_diff_exp_state_chunks = jnp.exp(g_cumsum[..., -1, None] - g_cumsum)[..., None]
    k_i_g_diff_chunks = (k_c.astype(jnp.float32) * g_diff_exp_state_chunks).astype(compute_dtype)

    # STAGE 3: INTER-CHUNK RECURRENCE
    w_scan = w_chunks
    u_scan = u_chunks
    q_g_scan = q_g_chunks
    attn_i_scan = attn_i_chunks
    g_i_last_exp_scan = g_i_last_exp_chunks
    k_i_g_diff_scan = k_i_g_diff_chunks

    # Empty (q_len == 0) rows own no chunk: their start chunk aliases the next
    # row's, so they are routed out of range and dropped from the scatters.
    num_seqs = new_query_start_loc.shape[0] - 1
    num_valid_seqs = jnp.clip(distribution[2], 0, num_seqs)
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)
    seq_is_nonempty = new_query_start_loc[1:] > new_query_start_loc[:-1]
    start_chunk_indices = jnp.where(seq_is_nonempty, new_query_start_loc[:-1] // chunk_size, num_chunks)
    reset_mask = jnp.zeros((num_chunks,), dtype=bool).at[start_chunk_indices].set(True, mode="drop")
    init_seq_per_chunk = (
        jnp.zeros((num_chunks,), dtype=jnp.int32)
        .at[start_chunk_indices]
        .set(jnp.arange(num_seqs, dtype=jnp.int32), mode="drop")
    )
    num_slots = recurrent_state.shape[0]
    slot_ok = (state_indices >= 0) & (state_indices < num_slots)
    safe_state_indices = jnp.clip(state_indices, 0, max(num_slots - 1, 0))
    init_states_for_seqs = jnp.where(
        slot_ok[:, None, None, None],
        recurrent_state[safe_state_indices],
        0.0,
    )
    if mask_initial_state and has_initial_state is not None:
        init_states_for_seqs = jnp.where(
            has_initial_state.astype(jnp.bool_)[:, None, None, None],
            init_states_for_seqs,
            jnp.zeros_like(init_states_for_seqs),
        )

    h_init = jnp.zeros((H, K_dim, V_dim), dtype=jnp.float32)

    xs = (
        w_scan,
        u_scan,
        q_g_scan,
        attn_i_scan,
        g_i_last_exp_scan,
        k_i_g_diff_scan,
        reset_mask,
        init_seq_per_chunk,
    )

    def scan_body(h, args):
        """Inter-chunk recurrence body for the chunked GDR prefill scan.

        For each chunk, optionally resets the state to the owning request's
        initial state (``init_states_for_seqs[init_seq]``), applies the
        carried-over recurrent contribution to produce the chunk output, and
        advances the state through the chunk using the accumulated gated decay.

        Args:
            h: Current recurrent state of shape ``(H, K_dim, V_dim)``.
            args: Tuple of per-chunk tensors and flags
                ``(w, u, q_g, attn_i, g_i_last_exp, k_i_g_diff, reset,
                init_seq)`` produced by the surrounding chunked algorithm.

        Returns:
            tuple: ``(h_new, (o_c, h_new))`` — ``h_new`` is the updated
            recurrent state, ``o_c`` is the chunk's output of shape
            ``(H, chunk_size, V_dim)``, and ``h_new`` is also emitted as
            scan output for downstream use.
        """
        w, u, q_g, attn_i, g_i_last_exp, k_i_g_diff, reset, init_seq = args

        h = jnp.where(reset, init_states_for_seqs[init_seq].astype(jnp.float32), h)

        attn_inter = jnp.matmul(
            q_g,
            h,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

        v_prime = jnp.matmul(
            w.astype(jnp.float32),
            h,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        v_new = u.astype(jnp.float32) - v_prime

        term2 = jnp.matmul(
            attn_i,
            v_new,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        o_c = attn_inter + term2

        h_new = h * g_i_last_exp
        update_term = jnp.matmul(
            k_i_g_diff.swapaxes(-1, -2),
            v_new,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        h_new = h_new + update_term

        return h_new, (o_c, h_new)

    _, (o_chunks, h_chunks) = lax.scan(scan_body, h_init, xs)

    # STAGE 4: FINALIZATION
    o = o_chunks.transpose(0, 2, 1, 3)
    o = o.reshape(-1, H, V_dim)

    o = o.astype(initial_dtype)

    packed_output_flat = o.reshape(-1, H * V_dim)
    output = packed_output_flat[padded_indices_valid]

    token_ids = jnp.arange(query.shape[0])
    token_seq_id = jnp.searchsorted(effective_query_start_loc, token_ids, side="right", method="compare_all") - 1
    safe_token_seq_id = jnp.clip(token_seq_id, 0, max(num_seqs - 1, 0))
    valid_token_mask = (
        (token_seq_id >= 0) & (token_seq_id < num_valid_seqs) & (token_ids < last_valid_loc) & slot_ok[safe_token_seq_id]
    )
    output = jnp.where(valid_token_mask[:, None], output, 0.0)

    last_chunk_indices = (new_query_start_loc[1:] // chunk_size) - 1
    final_states = h_chunks[last_chunk_indices]

    # Inactive, empty, and out-of-range-slot rows keep their slot untouched:
    # their writes are routed out of range and dropped.
    valid_seq_mask = (jnp.arange(num_seqs) < num_valid_seqs) & seq_is_nonempty & slot_ok
    write_indices = jnp.where(valid_seq_mask, state_indices, num_slots)
    updated_recurrent_state = recurrent_state.at[write_indices].set(
        final_states.astype(recurrent_state.dtype),
        mode="drop",
        wrap_negative_indices=False,
    )

    return updated_recurrent_state, output


def recurrent_gated_delta_rule_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step recurrent update for gated-delta-rule decode.

    Reference pure-JAX implementation of one decode step of the Gated Delta
    Rule recurrence used by Qwen3-Next style models.

    Args:
        query: Query tensor of shape ``(B, H, d_k)``.
        key: Key tensor of shape ``(B, H, d_k)``.
        value: Value tensor of shape ``(B, H, d_v)``.
        g: Log-space decay of shape ``(B, H)``.
        beta: Gating coefficient of shape ``(B, H)``.
        state: Optional initial recurrent state of shape
            ``(B, H, d_k, d_v)``. Defaults to all-zeros when ``None``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(out, new_state)`` where ``out``
        has shape ``(B, H, d_v)`` and ``new_state`` has shape
        ``(B, H, d_k, d_v)``.
    """
    B, H, d_k = query.shape
    d_v = value.shape[-1]

    if state is None:
        state = jnp.zeros((B, H, d_k, d_v), dtype=query.dtype)

    scale = d_k**-0.5
    query = query * scale

    exp_g = jnp.exp(g)

    k_state = jnp.einsum("bhd, bhdm -> bhm", key, state)
    v_diff = value - exp_g[..., None] * k_state

    v_new = beta[..., None] * v_diff

    q_state = jnp.einsum("bhd, bhdm -> bhm", query, state)
    q_k = jnp.sum(query * key, axis=-1, keepdims=True)

    out = exp_g[..., None] * q_state + q_k * v_new

    # Outer product using broadcasting
    k_v_new = key[..., :, None] * v_new[..., None, :]
    new_state = state * exp_g[..., None, None] + k_v_new

    return out, new_state


def _decode_token_rows_and_slots(
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    num_valid_seqs: jnp.ndarray,
    num_tokens: int,
    num_slots: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map every packed decode token to its owning row and state pool slot."""
    num_valid_seqs = jnp.clip(num_valid_seqs, 0, query_start_loc.shape[0] - 1)
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    effective_query_start_loc = jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)
    token_rows = (
        jnp.searchsorted(effective_query_start_loc, jnp.arange(num_tokens), side="right", method="compare_all") - 1
    )
    valid_mask = (token_rows >= 0) & (token_rows < num_valid_seqs)
    safe_rows = jnp.clip(token_rows, 0, state_indices.shape[0] - 1)
    token_slots = state_indices[safe_rows]
    valid_mask = valid_mask & (token_slots >= 0) & (token_slots < num_slots)
    safe_read_slots = jnp.clip(token_slots, 0, num_slots - 1)
    write_slots = jnp.where(valid_mask, token_slots, num_slots)
    return safe_rows, safe_read_slots, write_slots, valid_mask


def _decode_token_slots(
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    num_valid_seqs: jnp.ndarray,
    num_tokens: int,
    num_slots: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map every packed decode token to the state slot of the row that owns it.

    Rows ``[0, num_valid_seqs)`` are positional and may be empty
    (``q_len == 0``): a token belongs to the last row starting at or before it,
    so empty rows never claim another row's token. Tokens outside
    ``[query_start_loc[0], query_start_loc[num_valid_seqs])`` are padding.

    Args:
        query_start_loc: Cumulative per-row token offsets, shape
            ``(num_rows + 1,)``.
        state_indices: Row-to-slot mapping, shape ``(num_rows,)``.
        num_valid_seqs: Scalar number of positional rows in use.
        num_tokens: Static number of packed tokens.
        num_slots: Static size of the state pool.

    Returns:
        tuple: ``(token_slots, write_slots, valid_mask)`` of shape
        ``(num_tokens,)``. ``token_slots`` is always in range (safe to gather);
        ``write_slots`` is ``num_slots`` for padding tokens so ``mode="drop"``
        scatters never write their state.
    """
    _, safe_read_slots, write_slots, valid_mask = _decode_token_rows_and_slots(
        query_start_loc,
        state_indices,
        num_valid_seqs,
        num_tokens,
        num_slots,
    )
    return safe_read_slots, write_slots, valid_mask


def ragged_gated_delta_rule_decode_only(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    b_reshaped: jnp.ndarray,
    a_reshaped: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    use_qk_norm_in_gdn: bool,
    has_initial_state: jnp.ndarray | None = None,
    mask_initial_state: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies gated delta rule for decode-only case (sequence lengths <= 1).

    Args:
        query: Per-token queries ``(num_tokens, n_v, d_k)`` (already
            expanded to value-head count).
        key: Per-token keys with the same shape as ``query``.
        value: Per-token values ``(num_tokens, n_v, d_v)``.
        b_reshaped: Pre-sigmoid beta source ``(num_tokens, n_v)``.
        a_reshaped: Pre-softplus alpha source ``(num_tokens, n_v)``.
        recurrent_state: Global state pool of shape
            ``(num_blocks, n_v, d_k, d_v)`` where the first block is a
            null block reserved for padding / invalid tokens.
        A_log: Per-head log-decay ``(n_v,)``.
        dt_bias: Per-head delta-time bias ``(n_v,)``.
        query_start_loc: Cumulative per-request offsets, shape
            ``(num_requests + 1,)``.
        state_indices: Request-to-slot mapping, shape ``(num_requests,)``.
        distribution: ``[decode_end, prefill_end, total]`` int32 triple;
            ``distribution[2]`` bounds the positional row prefix. Rows in it
            carry at most one token; empty rows are skipped.
        use_qk_norm_in_gdn: Whether to L2-normalize Q and K before the
            decode update.
        has_initial_state: Optional boolean array ``(num_requests,)`` marking
            which requests carry valid prior state.
        mask_initial_state: When ``True`` and ``has_initial_state`` is
            provided, requests with ``has_initial_state == False`` start from
            zero state instead of reading stale slot contents.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` where the state has
        shape ``(num_blocks, n_v, d_k, d_v)`` and the output has shape
        ``(num_tokens, n_v * d_v)``. Output rows for padding tokens (at or
        past ``query_start_loc[distribution[2]]``) are zeroed; slots of empty,
        inactive, and padding rows are left untouched.
    """
    num_tokens = query.shape[0]
    d_k = query.shape[-1]

    safe_rows, token_slots, write_slots, valid_mask = _decode_token_rows_and_slots(
        query_start_loc,
        state_indices,
        distribution[2],
        num_tokens,
        recurrent_state.shape[0],
    )
    keep_initial = (
        has_initial_state.astype(jnp.bool_)[safe_rows]
        if (mask_initial_state and has_initial_state is not None)
        else None
    )

    # The whole recurrence stays in float32: rounding the per-step decay to a
    # low-precision dtype (e.g. bf16 turns exp(-1e-3) into 1.0) silently
    # removes it and the error compounds across decode steps.
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    if use_qk_norm_in_gdn:
        query = l2norm(query)
        key = l2norm(key)
    query = query * jnp.asarray(d_k**-0.5, dtype=jnp.float32)
    beta = jax.nn.sigmoid(b_reshaped.astype(jnp.float32))
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a_reshaped.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
    )
    exp_g = jnp.exp(g)

    state_f = recurrent_state[token_slots].astype(jnp.float32)
    if keep_initial is not None:
        state_f = jnp.where(keep_initial[:, None, None, None], state_f, 0.0)
    v_f = value.astype(jnp.float32)
    precision = jax.lax.Precision.HIGHEST

    k_state = jnp.einsum("bhd,bhdm->bhm", key, state_f, precision=precision)
    q_state = jnp.einsum("bhd,bhdm->bhm", query, state_f, precision=precision)
    v_new = beta[..., None] * (v_f - exp_g[..., None] * k_state)
    q_k = jnp.sum(query * key, axis=-1, keepdims=True)
    outputs = exp_g[..., None] * q_state + q_k * v_new
    k_v_new = key[..., :, None] * v_new[..., None, :]
    new_states = state_f * exp_g[..., None, None] + k_v_new

    outputs = jnp.where(valid_mask[:, None, None], outputs, 0.0)
    outputs = outputs.reshape(num_tokens, -1)
    updated_recurrent_state = recurrent_state.at[write_slots].set(
        new_states.astype(recurrent_state.dtype),
        mode="drop",
        wrap_negative_indices=False,
    )

    return updated_recurrent_state, outputs


def ragged_gdn_conv_decode_xla(
    conv_input: jnp.ndarray,
    b_raw: jnp.ndarray,
    a_raw: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_pool: jnp.ndarray,
    recurrent_pool: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    token_slots: jnp.ndarray,
    token_init: jnp.ndarray,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_l2norm: bool = True,
    apply_silu: bool = True,
    conv_output_dtype: object | None = None,
    out_dtype: object = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Portable XLA reference for fused causal Conv1D + SiLU + L2Norm + GDN decode.

    Matches :func:`pallas_gdn_conv_decode` semantics: updates ``conv_pool`` and
    ``recurrent_pool`` only for tokens whose ``token_slots[t]`` is in
    ``[0, num_slots)``. Fresh rows (``token_init[t] == False``) start from zero
    conv and recurrent state. Idle tokens output zeros and leave pools untouched.
    """
    n_kq = int(n_kq)
    n_v = int(n_v)
    d_k = int(d_k)
    d_v = int(d_v)
    if n_kq <= 0 or n_v <= 0 or d_k <= 0 or d_v <= 0:
        raise ValueError(
            f"n_kq, n_v, d_k, and d_v must be positive integers, got n_kq={n_kq}, n_v={n_v}, d_k={d_k}, d_v={d_v}."
        )
    if n_v % n_kq != 0:
        raise ValueError(f"n_v ({n_v}) must be divisible by n_kq ({n_kq}).")
    key_dim = n_kq * d_k
    conv_dim = 2 * key_dim + n_v * d_v
    if conv_input.ndim != 2 or int(conv_input.shape[-1]) != conv_dim:
        raise ValueError(f"conv_input must have shape (num_tokens, {conv_dim}), got {tuple(conv_input.shape)}.")
    num_tokens = int(conv_input.shape[0])
    if conv_weight.ndim != 2 or int(conv_weight.shape[0]) != conv_dim:
        raise ValueError(f"conv_weight must have shape ({conv_dim}, d_conv), got {tuple(conv_weight.shape)}.")
    d_conv = int(conv_weight.shape[1])
    if conv_pool.ndim != 3 or tuple(int(s) for s in conv_pool.shape[1:]) != (conv_dim, d_conv):
        raise ValueError(f"conv_pool must have shape (num_slots, {conv_dim}, {d_conv}), got {tuple(conv_pool.shape)}.")
    if recurrent_pool.ndim != 4 or tuple(int(s) for s in recurrent_pool.shape[1:]) != (n_v, d_k, d_v):
        raise ValueError(
            f"recurrent_pool must have shape (num_slots, {n_v}, {d_k}, {d_v}), got {tuple(recurrent_pool.shape)}."
        )
    if int(b_raw.size) != num_tokens * n_v:
        raise ValueError(
            f"b_raw must have {num_tokens * n_v} elements for (num_tokens={num_tokens}, n_v={n_v}), "
            f"got shape {tuple(b_raw.shape)}."
        )
    if int(a_raw.size) != num_tokens * n_v:
        raise ValueError(
            f"a_raw must have {num_tokens * n_v} elements for (num_tokens={num_tokens}, n_v={n_v}), "
            f"got shape {tuple(a_raw.shape)}."
        )
    if A_log.ndim != 1 or int(A_log.shape[0]) != n_v:
        raise ValueError(f"A_log must have shape ({n_v},), got {tuple(A_log.shape)}.")
    if dt_bias.ndim != 1 or int(dt_bias.shape[0]) != n_v:
        raise ValueError(f"dt_bias must have shape ({n_v},), got {tuple(dt_bias.shape)}.")
    if token_slots.ndim != 1 or int(token_slots.shape[0]) != num_tokens:
        raise ValueError(f"token_slots must have shape ({num_tokens},), got {tuple(token_slots.shape)}.")
    if token_init.ndim != 1 or int(token_init.shape[0]) != num_tokens:
        raise ValueError(f"token_init must have shape ({num_tokens},), got {tuple(token_init.shape)}.")

    num_conv_slots = int(conv_pool.shape[0])
    num_rec_slots = int(recurrent_pool.shape[0])
    num_slots = min(num_conv_slots, num_rec_slots)
    expand = n_v // n_kq
    c_dtype = jnp.dtype(conv_output_dtype or conv_input.dtype)
    o_dtype = jnp.dtype(out_dtype)

    if num_slots == 0 or num_tokens == 0:
        return jnp.zeros((num_tokens, n_v, d_v), dtype=o_dtype), conv_pool, recurrent_pool

    slots = token_slots.astype(jnp.int32)
    init = token_init.astype(jnp.bool_)
    live = (slots >= 0) & (slots < num_slots)
    keep = live & init
    conv_read_slots = jnp.clip(slots, 0, num_conv_slots - 1)
    rec_read_slots = jnp.clip(slots, 0, num_rec_slots - 1)
    conv_write_slots = jnp.where(live, slots, num_conv_slots)
    rec_write_slots = jnp.where(live, slots, num_rec_slots)

    old_conv = conv_pool[conv_read_slots].astype(jnp.float32)
    old_conv = jnp.where(keep[:, None, None], old_conv, 0.0)
    new_win = jnp.concatenate([old_conv[:, :, 1:], conv_input.astype(jnp.float32)[:, :, None]], axis=-1)
    mixed = jnp.sum(new_win * conv_weight.astype(jnp.float32)[None, :, :], axis=-1)
    y = jax.nn.silu(mixed) if apply_silu else mixed
    y = y.astype(c_dtype).astype(jnp.float32)
    updated_conv_pool = conv_pool.at[conv_write_slots].set(
        new_win.astype(conv_pool.dtype),
        mode="drop",
        wrap_negative_indices=False,
    )

    q = y[:, :key_dim].reshape(num_tokens, n_kq, d_k)
    k = y[:, key_dim : 2 * key_dim].reshape(num_tokens, n_kq, d_k)
    v = y[:, 2 * key_dim :].reshape(num_tokens, n_v, d_v)
    if use_qk_l2norm:
        q = l2norm(q)
        k = l2norm(k)
    q = q * jnp.asarray(d_k**-0.5, dtype=jnp.float32)
    if expand > 1:
        q = jnp.repeat(q, expand, axis=1)
        k = jnp.repeat(k, expand, axis=1)

    beta = jax.nn.sigmoid(b_raw.reshape(num_tokens, n_v).astype(jnp.float32))
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a_raw.reshape(num_tokens, n_v).astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
    )
    exp_g = jnp.exp(g)

    state_f = jnp.where(keep[:, None, None, None], recurrent_pool[rec_read_slots].astype(jnp.float32), 0.0)
    precision = jax.lax.Precision.HIGHEST
    s_dec = state_f * exp_g[:, :, None, None]
    k_state = jnp.einsum("bhd,bhdm->bhm", k, s_dec, precision=precision)
    delta = beta[:, :, None] * (v - k_state)
    new_state = s_dec + k[:, :, :, None] * delta[:, :, None, :]
    out = jnp.einsum("bhd,bhdm->bhm", q, new_state, precision=precision)
    out = jnp.where(live[:, None, None], out, 0.0).astype(o_dtype)
    updated_rec_pool = recurrent_pool.at[rec_write_slots].set(
        new_state.astype(recurrent_pool.dtype),
        mode="drop",
        wrap_negative_indices=False,
    )
    return out, updated_conv_pool, updated_rec_pool


@jax.jit(
    donate_argnames=("recurrent_state",),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "chunk_size",
        "use_qk_norm_in_gdn",
        "apply_silu_in_gdr",
        "use_recurrent_scan_prefill",
        "mask_initial_state",
    ),
)
@jax.named_scope("ragged_gated_delta_rule_chunked")
def ragged_gated_delta_rule(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    has_initial_state: jnp.ndarray | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = True,
    apply_silu_in_gdr: bool = False,
    use_recurrent_scan_prefill: bool = False,
    mask_initial_state: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies the gated delta rule over ragged seq lengths

    This function separates mixed QKV, handles repeating for multi-query attention
    if needed, and routes to either the decode-only or mixed-prefill branch
    depending on sequence lengths.

    Args:
        mixed_qkv: Interleaved Q/K/V projections in a single flat feature
            dimension, shape ``(num_tokens, 2 * n_kq * d_k + n_v * d_v)``.
            The first ``n_kq * d_k`` features hold queries, the next
            ``n_kq * d_k`` features hold keys, and the remaining
            ``n_v * d_v`` features hold values.
        b: Pre-sigmoid beta source ``(num_tokens, n_v)``.
        a: Pre-softplus alpha source ``(num_tokens, n_v)``.
        recurrent_state: Global state pool ``(num_blocks, n_v, d_k, d_v)``
            with ``num_blocks >= max_reqs + 1``; block 0 is the null
            block reserved for padded slots.
        A_log: Per-head log-decay ``(n_v,)``.
        dt_bias: Per-head delta-time bias ``(n_v,)``.
        query_start_loc: Cumulative per-request offsets,
            shape ``(num_requests + 1,)``.
        state_indices: Request-to-slot mapping, shape ``(num_requests,)``.
        distribution: ``int32[3]`` ``(decode_end, prefill_end, mixed_end)``
            classifying scheduled requests; controls the
            ``decode_only_branch`` / ``mixed_prefill_branch`` selection.
            ``mixed_end`` bounds the positional row prefix; empty
            (``q_len == 0``) rows inside it are skipped by both branches.
        has_initial_state: Optional boolean array of shape
            ``(num_requests,)`` flagging which requests carry a non-empty
            initial recurrent state. Defaults to all-``True`` when ``None``;
            consumed to zero stale slots when ``mask_initial_state=True``.
        n_kq: Number of key/query heads (before head expansion).
        n_v: Number of value heads (after the GQA-style expansion).
        d_k: Per-head key/query dimension.
        d_v: Per-head value dimension.
        chunk_size: Padding granularity used by the mixed-prefill branch.
            Defaults to 64.
        use_qk_norm_in_gdn: Whether to L2-normalize queries and keys before
            the recurrence. Defaults to True.
        apply_silu_in_gdr: When True, applies SiLU to ``mixed_qkv`` before
            splitting it into Q/K/V. Defaults to False.
        use_recurrent_scan_prefill: Static flag accepted for API/signature
            compatibility; not consumed by this implementation. Defaults to
            False.
        mask_initial_state: When True, use ``has_initial_state`` to zero stale
            recurrent slots for fresh prefill requests. Static; defaults to
            False to keep the all-valid prefill path fast.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` with state of shape
        ``(num_blocks, n_v, d_k, d_v)`` and output of shape
        ``(num_tokens, n_v * d_v)`` cast to ``mixed_qkv.dtype``.
    """
    if has_initial_state is None:
        has_initial_state = jnp.ones(state_indices.shape[0], dtype=jnp.bool_)

    num_tokens = mixed_qkv.shape[0]
    mixed_qkv_post_silu = jax.nn.silu(mixed_qkv) if apply_silu_in_gdr else mixed_qkv
    key_dim = n_kq * d_k
    query = mixed_qkv_post_silu[..., :key_dim]
    key = mixed_qkv_post_silu[..., key_dim : key_dim * 2]
    value = mixed_qkv_post_silu[..., key_dim * 2 :]

    q_reshaped = query.reshape(num_tokens, n_kq, d_k)
    k_reshaped = key.reshape(num_tokens, n_kq, d_k)
    v_reshaped = value.reshape(num_tokens, n_v, d_v)

    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
        q_reshaped = jnp.repeat(q_reshaped, repeat_factor, axis=1)
        k_reshaped = jnp.repeat(k_reshaped, repeat_factor, axis=1)
    b_reshaped = b.reshape(num_tokens, n_v)
    a_reshaped = a.reshape(num_tokens, n_v)

    def decode_only_branch(_):
        """Run the decode-only fast path under :func:`lax.cond`.

        Used when every active request consumes exactly one new token.
        The unused operand is required by the ``lax.cond`` signature.

        Args:
            _: Unused operand passed through by ``lax.cond``.

        Returns:
            tuple: ``(updated_recurrent_state, output)`` produced by
            :func:`ragged_gated_delta_rule_decode_only`, with output cast
            back to ``mixed_qkv.dtype``.
        """
        new_state, output = ragged_gated_delta_rule_decode_only(
            query=q_reshaped,
            key=k_reshaped,
            value=v_reshaped,
            b_reshaped=b_reshaped,
            a_reshaped=a_reshaped,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            has_initial_state=has_initial_state,
            mask_initial_state=mask_initial_state,
        )
        return new_state, output.astype(mixed_qkv.dtype)

    def mixed_prefill_branch(_):
        """Run the chunked mixed-prefill path under :func:`lax.cond`.

        Used when at least one scheduled request has more than one new
        token. Pads each request's tokens up to ``chunk_size`` and runs
        the fully-batched chunked GDR algorithm.

        Args:
            _: Unused operand passed through by ``lax.cond``.

        Returns:
            tuple: ``(updated_recurrent_state, output)`` produced by
            :func:`ragged_gated_delta_rule_mixed_prefill`.
        """
        return ragged_gated_delta_rule_mixed_prefill(
            query=q_reshaped,
            key=k_reshaped,
            value=v_reshaped,
            b_reshaped=b_reshaped,
            a_reshaped=a_reshaped,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            recurrent_state=recurrent_state,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            mask_initial_state=mask_initial_state,
            compute_dtype=jnp.float32,
        )

    is_decode_only = distribution[0] == distribution[2]

    return jax.lax.cond(is_decode_only, decode_only_branch, mixed_prefill_branch, operand=None)


def ragged_gated_delta_rule_v2(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    has_initial_state: jnp.ndarray | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = True,
    apply_silu_in_gdr: bool = False,
    use_recurrent_scan_prefill: bool = False,
    mask_initial_state: bool = False,
    runtime_dtype: jnp.dtype | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the unsharded packed-inference GDN v2 XLA kernel.

    Thin entry wrapper around :func:`ragged_gated_delta_rule` that first
    casts the activations to a common ``runtime_dtype`` and defaults
    ``has_initial_state`` before dispatching to the (jitted) chunked kernel.
    The recurrent state keeps its storage dtype in and out, and the gate
    parameters are used in float32.
    Use this for the single-device / unsharded path.

    Args:
        mixed_qkv: Interleaved Q/K/V projections in one flat feature
            dimension, shape ``(num_tokens, 2 * n_kq * d_k + n_v * d_v)``.
        b: Pre-sigmoid beta source ``(num_tokens, n_v)``.
        a: Pre-softplus alpha source ``(num_tokens, n_v)``.
        recurrent_state: Global state pool ``(num_blocks, n_v, d_k, d_v)``
            with ``num_blocks >= max_reqs + 1``; block 0 is the null block.
        A_log: Per-head log-decay ``(n_v,)``.
        dt_bias: Per-head delta-time bias ``(n_v,)``.
        query_start_loc: Cumulative per-request offsets,
            shape ``(num_requests + 1,)``.
        state_indices: Request-to-slot mapping, shape ``(num_requests,)``.
        distribution: ``int32[3]`` ``(decode_end, prefill_end, mixed_end)``
            controlling the decode-only vs mixed-prefill branch selection.
        has_initial_state: Optional boolean array ``(num_requests,)`` marking
            requests with a non-empty initial state; defaults to all-``True``
            when ``None``.
        n_kq: Number of key/query heads (before head expansion).
        n_v: Number of value heads (after the GQA-style expansion).
        d_k: Per-head key/query dimension.
        d_v: Per-head value dimension.
        chunk_size: Padding granularity for the mixed-prefill branch.
            Defaults to 64.
        use_qk_norm_in_gdn: Whether to L2-normalize queries and keys.
            Defaults to True.
        apply_silu_in_gdr: When True, applies SiLU to ``mixed_qkv`` before
            the Q/K/V split. Defaults to False.
        use_recurrent_scan_prefill: Static compatibility flag forwarded
            unchanged; not consumed by the implementation. Defaults to False.
        mask_initial_state: When True, use ``has_initial_state`` to zero stale
            recurrent slots for fresh prefill requests. Defaults to False.
        runtime_dtype: Common dtype for ``mixed_qkv``, ``a`` and ``b``.
            Defaults to ``mixed_qkv.dtype`` when ``None``.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` with state of shape
        ``(num_blocks, n_v, d_k, d_v)`` and output of shape
        ``(num_tokens, n_v * d_v)``, as produced by
        :func:`ragged_gated_delta_rule`.
    """
    runtime_dtype = runtime_dtype or mixed_qkv.dtype
    mixed_qkv = mixed_qkv.astype(runtime_dtype)
    b = b.astype(runtime_dtype)
    a = a.astype(runtime_dtype)
    A_log = A_log.astype(jnp.float32)
    dt_bias = dt_bias.astype(jnp.float32)
    if has_initial_state is None:
        has_initial_state = jnp.ones(state_indices.shape[0], dtype=jnp.bool_)

    return ragged_gated_delta_rule(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        recurrent_state=recurrent_state,
        A_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        has_initial_state=has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        apply_silu_in_gdr=apply_silu_in_gdr,
        use_recurrent_scan_prefill=use_recurrent_scan_prefill,
        mask_initial_state=mask_initial_state,
    )
