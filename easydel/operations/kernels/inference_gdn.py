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

This module provides the JAX/Pallas implementation of the Gated Delta Rule
(GDR) recurrence for the eSurge packed-inference path, where many requests
of heterogeneous lengths share a single contiguous token buffer. It exposes:

* :func:`ragged_gated_delta_rule` - the top-level JIT entry point that splits
  an interleaved ``mixed_qkv`` stream into Q/K/V, optionally repeats heads
  for grouped-query layouts, and dispatches to either the decode-only fast
  path or the chunked mixed-prefill branch based on the ``request_distribution``.
* :func:`ragged_gated_delta_rule_decode_only` - per-token Pallas/JAX update
  used when every active request consumes exactly one new token.
* :func:`ragged_gated_delta_rule_mixed_prefill` - chunked algorithm that
  pads each request's tokens to a multiple of ``chunk_size``, runs the
  intra-chunk attention in parallel, and propagates inter-chunk state via
  ``lax.scan``. Requires the unit lower-triangular inverse provided by
  :class:`TriangleSolverImpl`.
* :class:`RaggedGatedDeltaRule` - first-class :class:`OperationImpl`
  wrapping the kernel in a head-parallel ``shard_map``.
* Helpers for the unit lower-triangular inverse used inside the chunked
  formulation (Newton-Schulz, blockwise Gaussian elimination, and a portable
  ``jax.scipy``-based path), plus a Pallas TPU decode kernel
  (``_pallas_gdn_decode_kernel``) used when the geometry fits in VMEM.

Algorithmic notes:
- All forward arithmetic is performed in float32 for numerical stability and
  cast back to ``mixed_qkv.dtype`` (typically bfloat16) on the way out.
- The chunked path uses an online-style update: per-chunk ``q @ k^T`` is
  weighted by ``exp(g_diff)`` to mix the gated decay into the attention
  pattern, then the inter-chunk recurrence carries the running ``state``
  through a ``lax.scan``.
- The decode-only path tries a Pallas TPU kernel when geometric constraints
  (head dim, Mosaic VMEM budget, identity state mapping) are satisfied,
  falling back to a pure-JAX gather/scatter implementation otherwise.
"""

import enum
import functools
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec

from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)
from ._gdn_policy import normalize_kernel_tile_policy


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


def newton_schulz_inverse_pallas_kernel(A_ref, x_ref):
    """Pallas kernel body that wraps :func:`newton_schulz_inverse_ref`.

    Reads the unit lower-triangular block from ``A_ref`` and writes its
    inverse into ``x_ref``.

    Args:
        A_ref: Pallas reference to a block of unit lower-triangular matrices
            with shape ``(block_size, N, N)``.
        x_ref: Pallas reference to the output block, same shape and dtype as
            ``A_ref``.
    """
    x_ref[...] = newton_schulz_inverse_ref(A_ref[...])


def newton_schulz_inverse_pallas(A, *, block_size=64):
    """Newton-Schulz iteration for unit lower triangular matrices on Pallas.

    Tiles the leading dimensions of ``A`` so each Pallas program inverts a
    ``(block_size, N, N)`` slab using :func:`newton_schulz_inverse_pallas_kernel`.

    Args:
        A: Tensor whose last two dimensions form a unit lower-triangular
            ``N x N`` matrix; the leading dimensions are batched.
        block_size: Number of leading-axis matrices to handle per Pallas
            program. Defaults to 64.

    Returns:
        jnp.ndarray: Tensor with the same shape and dtype as ``A`` containing
        the inverse of each unit lower-triangular slab.
    """

    A_shape = A.shape
    A = A.reshape(-1, *A.shape[-2:])
    N = A.shape[0]
    grid_size = pl.cdiv(N, block_size)
    x = pl.pallas_call(
        newton_schulz_inverse_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_size, A.shape[-2], A.shape[-1]), lambda idx: (idx, 0, 0)),
        ],
        out_specs=pl.BlockSpec((block_size, A.shape[-2], A.shape[-1]), lambda idx: (idx, 0, 0)),
        name="newton_schulz_inverse_kernel",
    )(A)
    return x.reshape(A_shape)


def local_forward_substitution(A, b):
    """Solve :math:`A X = b` row-by-row for unit lower-triangular ``A``.

    Used inside :func:`decompose_triangular_matrix_inverse_pallas_kernel`
    on a single Pallas program block. Iterates over the ``N`` rows of each
    matrix, solving for ``X[:, i, :]`` using the previously-computed
    rows. Because ``A`` has unit diagonal, no division is required:
    ``X[:, i, :] = b[:, i, :] - sum_{j < i} A[:, i, j] * X[:, j, :]``.

    The implementation accumulates ``X`` row-by-row in a Python list (the
    loop is unrolled at trace time) and stacks them at the end. This is
    intended for small ``N`` (typically ``block_size`` of the calling
    Pallas tile, e.g. 16); use a Newton-Schulz or LAPACK-based path for
    large matrices.

    Args:
        A: Batched unit lower-triangular matrix of shape ``(B, N, N)``.
        b: Right-hand side of shape ``(B, N, K)``.

    Returns:
        jnp.ndarray: Solution ``X`` of shape ``(B, N, K)``.
    """
    _B, N, _K = b.shape
    x_list = []
    for i in range(N):
        b_i = b[:, i, :]
        if i == 0:
            x_i = b_i
        else:
            stacked_x = jnp.stack(x_list, axis=1)  # (B, i, K)
            all_prev_A = A[:, i, :i]  # (B, i)
            prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)  # (B, K)
            x_i = b_i - prev_sum  # (B, K) for the row i
        x_list.append(x_i)
    x = jnp.stack(x_list, axis=1)  # (B, N, K)
    return x


def decompose_triangular_matrix_inverse_pallas_kernel(A_ref, x_ref, *, block_size=16):
    """Pallas kernel body for blockwise unit lower-triangular inversion.

    Performs blockwise Gaussian elimination over the trailing ``N x N`` axis
    of ``A_ref``, writing the inverse incrementally into ``x_ref`` in chunks
    of ``block_size`` rows.

    Args:
        A_ref: Pallas reference holding the input matrices,
            shape ``(B, N, N)``.
        x_ref: Pallas reference for the inverse output, same shape as
            ``A_ref``.
        block_size: Row-block size used to partition each ``N x N`` matrix
            during the forward-substitution sweep. ``N`` must be a multiple
            of ``block_size``.
    """
    A = A_ref[...]
    B, N, _ = A.shape
    num_blocks = N // block_size

    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))
        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = x_ref[:, :start, :]
            prev_sum = jnp.matmul(interaction_A, solved_x, precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_substitution(local_A, target_b)
        x_ref[..., start:end, :] = x_block


def decompose_triangular_matrix_inverse_pallas(A, *, n_block_size=64, block_size=16):
    """Pallas TPU kernel that inverts a stack of unit lower-triangular matrices.

    Solves :math:`A X = I` for ``X`` where every ``N x N`` slab of ``A`` is
    unit lower-triangular. The leading dimensions are first squashed into
    a single batch axis, then partitioned along that axis in chunks of
    ``n_block_size`` matrices per Pallas program. Within each program,
    :func:`decompose_triangular_matrix_inverse_pallas_kernel` performs
    blockwise Gaussian elimination, sweeping through ``N // block_size``
    row-blocks and using :func:`local_forward_substitution` for each
    inner block.

    The kernel sets ``vmem_limit_bytes`` to 64 MiB so it can hold the
    full ``(n_block_size, N, N)`` slab in VMEM without spills.

    Args:
        A: Tensor of shape ``(..., N, N)`` whose last two dimensions form
            unit lower-triangular matrices. The leading dimensions are
            arbitrary (e.g. ``(batch, num_chunks, num_heads, N, N)`` for
            the chunked GDR path).
        n_block_size: Number of leading-axis matrices that each Pallas
            program processes. Larger values amortise launch overhead but
            increase per-program VMEM footprint.
        block_size: Inner block size used by the Gaussian elimination
            sweep; ``N`` must be a multiple of ``block_size``.

    Returns:
        jnp.ndarray: Inverse of ``A`` with the same shape and dtype.
    """

    # Squash all the leading dimensions
    A_reshaped = A.reshape(-1, *A.shape[-2:])
    A_shape = A_reshaped.shape
    x_shape = A_shape

    N = A_reshaped.shape[0]
    grid_size = pl.cdiv(N, n_block_size)

    head_dim = A_shape[-1]
    kernel = functools.partial(decompose_triangular_matrix_inverse_pallas_kernel, block_size=block_size)
    x = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x_shape, A.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((n_block_size, head_dim, head_dim), lambda idx: (idx, 0, 0)),
        ],
        out_specs=pl.BlockSpec((n_block_size, head_dim, head_dim), lambda idx: (idx, 0, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=67108864),
        name=f"decompose_triangular_matrix_inverse_pallas_kernel_{n_block_size}_{block_size}",
    )(A_reshaped)

    return x.reshape(A.shape)


def triangular_inverse_jax(A):
    """Backend-agnostic unit lower-triangular inverse via ``jax.scipy``.

    Computes :math:`A^{-1}` column-by-column using
    :func:`jax.scipy.linalg.solve_triangular` against an identity batch.
    Acts as the portable fallback selected by :class:`TriangleSolverImpl`
    when the Pallas TPU paths are not available (non-TPU backend, mixed
    dtypes the Pallas kernel does not support, or unit-test reference
    flows).

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

    Members:
        GAUSSIAN: Blockwise Gaussian elimination via Pallas TPU kernel
            (:func:`decompose_triangular_matrix_inverse_pallas`).
        NEWTON_SCHULZ: Newton-Schulz iteration via Pallas TPU kernel
            (:func:`newton_schulz_inverse_pallas`).
        JAX: Portable pure-JAX path via
            :func:`triangular_inverse_jax`.
    """

    GAUSSIAN = "gaussian"
    NEWTON_SCHULZ = "newton_schulz"
    JAX = "jax"

    def __call__(self, A):
        """Invoke the selected inverse implementation.

        Args:
            A: Tensor whose last two dimensions form unit lower-triangular
                ``N x N`` matrices.

        Returns:
            jnp.ndarray: The inverse of ``A`` along the last two
            dimensions, computed by the chosen backend. Falls back to
            :data:`GAUSSIAN` when the value is unknown.
        """
        if self == TriangleSolverImpl.GAUSSIAN:
            return decompose_triangular_matrix_inverse_pallas(A, n_block_size=min(64, A.shape[-1]))
        elif self == TriangleSolverImpl.NEWTON_SCHULZ:
            return newton_schulz_inverse_pallas(A)
        elif self == TriangleSolverImpl.JAX:
            return triangular_inverse_jax(A)
        else:
            print(f"Unknown solver: {self.value} Using default solver. {TriangleSolverImpl.GAUSSIAN.value}")
            return decompose_triangular_matrix_inverse_pallas(A, n_block_size=min(64, A.shape[-1]))


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
    compute_dtype: jnp.dtype = jnp.bfloat16,
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

    num_valid_seqs = distribution[2]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)

    # Calculate sequence lengths and pad them to multiples of chunk_size.
    seq_lengths = effective_query_start_loc[1:] - effective_query_start_loc[:-1]
    num_chunks = (seq_lengths + chunk_size - 1) // chunk_size
    padded_lengths = num_chunks * chunk_size

    new_query_start_loc = jnp.cumsum(jnp.concatenate([jnp.array([0]), padded_lengths]))
    seq_id = jnp.searchsorted(effective_query_start_loc, jnp.arange(num_tokens), side="right") - 1
    original_start = effective_query_start_loc[seq_id]
    new_start = new_query_start_loc[seq_id]
    padded_indices_valid = new_start + (jnp.arange(num_tokens) - original_start)

    max_packed_tokens = num_tokens + num_seqs * chunk_size
    max_packed_tokens = (max_packed_tokens + chunk_size - 1) // chunk_size * chunk_size

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
    packed_combined_qkvb = packed_combined_qkvb.at[padded_indices_valid].set(combined_qkvb)

    K_dim = query.shape[2]
    V_dim = value.shape[2]
    packed_query = packed_combined_qkvb[..., :K_dim]
    packed_key = packed_combined_qkvb[..., K_dim : 2 * K_dim]
    packed_value = packed_combined_qkvb[..., 2 * K_dim : 2 * K_dim + V_dim]
    packed_beta = packed_combined_qkvb[..., 2 * K_dim + V_dim]

    # For g (float32)
    output_shape_f32 = (max_packed_tokens, *g.shape[1:])
    packed_g = jnp.zeros(output_shape_f32, dtype=jnp.float32)
    packed_g = packed_g.at[padded_indices_valid].set(g.astype(jnp.float32))

    num_chunks_total = max_packed_tokens // chunk_size
    reset_mask = jnp.zeros((num_chunks_total,), dtype=bool)
    start_chunk_indices = new_query_start_loc[:-1] // chunk_size
    reset_mask = reset_mask.at[start_chunk_indices].set(True)

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
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
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
            third entry gates which slots have outputs to write.
        chunk_size: Padding and chunking granularity for the parallel
            intra-chunk attention.
        use_qk_norm_in_gdn: Whether to L2-normalize Q and K before the
            chunked attention.
        compute_dtype: Dtype for the chunked Q/K/V/beta tensors.
        precision: ``lax.Precision`` for the matmul calls; defaults to
            ``HIGHEST`` to keep numerical stability across long
            recurrences.
        preferred_element_type: Accumulation dtype for ``jnp.matmul``.
        triangle_solver_impl: Selector for the unit lower-triangular
            inverse used to solve for the per-chunk attention weights.
            Defaults to the Pallas TPU Gaussian solver on TPU and the
            portable JAX path otherwise.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` where
        ``updated_recurrent_state`` has shape
        ``(num_blocks, n_v, d_k, d_v)`` and ``output`` has shape
        ``(num_tokens, n_v * d_v)`` cast back to ``query.dtype``.
    """
    if triangle_solver_impl is None:
        triangle_solver_impl = TriangleSolverImpl.GAUSSIAN if jax.default_backend() == "tpu" else TriangleSolverImpl.JAX

    initial_dtype = query.dtype

    beta = jax.nn.sigmoid(b_reshaped)
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a_reshaped.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )

    # Pack inputs
    (
        packed_query,
        packed_key,
        packed_value,
        packed_g,
        packed_beta,
        reset_mask,
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
        return x.reshape(num_chunks, chunk_size, H, -1).transpose(0, 2, 1, 3)

    def to_chunk_scalar(x):
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

    # Prepare init_h_per_chunk
    init_h_per_chunk = jnp.zeros((num_chunks, H, K_dim, V_dim), dtype=recurrent_state.dtype)
    start_chunk_indices = new_query_start_loc[:-1] // chunk_size
    init_h_per_chunk = init_h_per_chunk.at[start_chunk_indices].set(recurrent_state[state_indices])

    h_init = jnp.zeros((H, K_dim, V_dim), dtype=jnp.float32)

    xs = (
        w_scan,
        u_scan,
        q_g_scan,
        attn_i_scan,
        g_i_last_exp_scan,
        k_i_g_diff_scan,
        reset_mask,
        init_h_per_chunk,
    )

    def scan_body(h, args):
        w, u, q_g, attn_i, g_i_last_exp, k_i_g_diff, reset, init_h = args

        h = jnp.where(reset, init_h, h)

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

    # Unpack output
    packed_output_flat = o.reshape(-1, H * V_dim)
    output = packed_output_flat[padded_indices_valid]

    # Update recurrent state
    last_chunk_indices = (new_query_start_loc[1:] // chunk_size) - 1
    final_states = h_chunks[last_chunk_indices]

    num_seqs = last_chunk_indices.shape[0]
    valid_seq_mask = jnp.arange(num_seqs) < distribution[2]
    current_states = recurrent_state[state_indices]
    states_to_set = jnp.where(
        valid_seq_mask[:, None, None, None],
        final_states.astype(recurrent_state.dtype),
        current_states,
    )
    updated_recurrent_state = recurrent_state.at[state_indices].set(states_to_set)

    return updated_recurrent_state, output


def _pallas_gdn_decode_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    exp_g_ref,
    state_ref,
    valid_ref,
    out_ref,
    new_state_ref,
):
    """Per-program: process `B_TOK` tokens (unrolled) with one slot's state each.

    Block layout (per program):
        q, k:        (B_TOK, H, D_K)      bf16
        v:           (B_TOK, H, D_V)      bf16
        beta, exp_g: (B_TOK, H, 128)      bf16  (padded — lane-align)
        state:       (B_TOK, H, D_K, D_V) bf16
        valid:       (B_TOK, 1, 128)      int32 (padded)

    Outputs:
        out:         (B_TOK, H, D_V)       bf16
        new_state:   (B_TOK, H, D_K, D_V)  bf16  (for invalid tokens: unchanged)

    Assumes identity state mapping (token i → slot i) — which matches the
    Qwen3-Next ragged decode path where ``state_indices = arange(num_slots)``.
    """
    B = q_ref.shape[0]
    for b_idx in range(B):
        q = q_ref[b_idx]
        k = k_ref[b_idx]
        v = v_ref[b_idx]
        beta_v = beta_ref[b_idx, :, 0]
        exp_g_v = exp_g_ref[b_idx, :, 0].astype(jnp.float32)
        state = state_ref[b_idx]
        valid_bool = valid_ref[b_idx, 0, 0] != 0

        state_f = state.astype(jnp.float32)
        k_f = k.astype(jnp.float32)
        q_f = q.astype(jnp.float32)
        v_f = v.astype(jnp.float32)

        # Pallas TPU matmul only supports 1 batch dim → iterate heads via (H, 1, D_K) @ (H, D_K, D_V).
        k_state = jnp.matmul(k_f[:, None, :], state_f)[:, 0, :]  # (H, D_V)
        q_state = jnp.matmul(q_f[:, None, :], state_f)[:, 0, :]

        v_new = beta_v[..., None].astype(jnp.float32) * (v_f - exp_g_v[..., None] * k_state)
        q_k = jnp.sum(q_f * k_f, axis=-1, keepdims=True)
        out = exp_g_v[..., None] * q_state + q_k * v_new

        k_v_new = k_f[..., None] * v_new[..., None, :]
        new_state = state_f * exp_g_v[..., None, None] + k_v_new

        out_masked = jnp.where(valid_bool, out, 0.0).astype(out_ref.dtype)
        new_state_masked = jnp.where(valid_bool, new_state, state_f).astype(new_state_ref.dtype)

        out_ref[b_idx] = out_masked
        new_state_ref[b_idx] = new_state_masked


_PALLAS_GDN_TILE_POLICY = normalize_kernel_tile_policy(os.environ.get("EASYDEL_GDN_TILE_POLICY", "auto"))
_PALLAS_GDN_BTOK_CANDIDATES = (16, 8, 4)


def set_gdn_kernel_tile_policy(policy: str) -> None:
    """Set the TPU Pallas GDN decode tile policy for future traces.

    Updates the module-level ``_PALLAS_GDN_TILE_POLICY`` consulted by
    :func:`_select_pallas_gdn_btok` when the operation is next traced. Also
    invoked indirectly via the ``EASYDEL_GDN_TILE_POLICY`` environment
    variable.

    Args:
        policy: One of ``"auto"``, ``"b16"``, ``"b8"`` or ``"b4"`` (case
            insensitive). ``"auto"`` lets the kernel pick a tile size based
            on VMEM-window heuristics.

    Raises:
        ValueError: If ``policy`` is not one of the supported variants.
    """

    global _PALLAS_GDN_TILE_POLICY
    _PALLAS_GDN_TILE_POLICY = normalize_kernel_tile_policy(policy)


def _select_pallas_gdn_btok(num_tokens: int, n_v: int, d_k: int, d_v: int, dtype) -> int | None:
    """Choose a TPU tile that keeps Mosaic VMEM windows under control.

    Honours the ``_PALLAS_GDN_TILE_POLICY`` global. In ``"auto"`` mode it
    iterates over ``(16, 8, 4)`` candidates and returns the first that both
    divides ``num_tokens`` and keeps ``2 * b_tok * n_v * d_k * d_v *
    bytes_per`` under a 14 MiB cap (input + output state windows).

    Args:
        num_tokens: Total tokens packed into the decode batch.
        n_v: Number of value heads in the kernel.
        d_k: Key/query head dimension.
        d_v: Value head dimension.
        dtype: Input dtype, used to estimate per-element bytes.

    Returns:
        int | None: The chosen ``b_tok`` value, or ``None`` if no candidate
        fits the constraints (in which case the JAX fallback is used).
    """

    policy = _PALLAS_GDN_TILE_POLICY
    if policy != "auto":
        b_tok = int(policy[1:])
        return b_tok if num_tokens >= b_tok and num_tokens % b_tok == 0 else None

    bytes_per = 2
    try:
        bytes_per = jnp.dtype(dtype).itemsize
    except Exception:
        pass
    # TPU Mosaic may double-buffer these windows and still needs spill space.
    # Keep the raw input+output state window pair small enough that buffering
    # does not push the program over the 64MiB VMEM ceiling.
    max_window_pair_bytes = 14 << 20
    for b_tok in _PALLAS_GDN_BTOK_CANDIDATES:
        if num_tokens < b_tok or num_tokens % b_tok != 0:
            continue
        pair_bytes = 2 * b_tok * int(n_v) * int(d_k) * int(d_v) * int(bytes_per)
        if pair_bytes <= max_window_pair_bytes:
            return b_tok
    return None


def _pallas_gdn_decode_call(q, k, v, beta, exp_g, state, valid, *, b_tok: int):
    """Run the fused Pallas decode kernel and return ``(outputs, new_state)``.

    Wraps :func:`_pallas_gdn_decode_kernel` in a ``pallas_call`` with a
    ``(T // b_tok,)`` grid. ``beta``, ``exp_g`` and ``valid`` are broadcast
    into 128-wide lane-aligned tensors before being handed to the kernel.

    Assumes:
        - ``state_indices`` is identity (token ``i`` corresponds to slot ``i``).
        - ``q.shape[0]`` is divisible by the selected Pallas tile size.

    Args:
        q: Query tensor of shape ``(T, H, D_K)``.
        k: Key tensor of shape ``(T, H, D_K)``.
        v: Value tensor of shape ``(T, H, D_V)``.
        beta: Per-token gating coefficients of shape ``(T, H)``.
        exp_g: Per-token decay (already ``exp``-applied) of shape ``(T, H)``.
        state: Per-slot recurrent state of shape ``(T, H, D_K, D_V)``.
        valid: Boolean validity mask of shape ``(T,)`` indicating which
            tokens are real vs padding.
        b_tok: Pallas tile size along the token axis. Must divide ``T``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(outputs, new_state)`` where
        ``outputs`` has shape ``(T, H, D_V)`` and ``new_state`` has shape
        ``(T, H, D_K, D_V)``.
    """
    T, H, D_K = q.shape
    D_V = v.shape[-1]
    assert T % b_tok == 0, f"num_tokens={T} must be divisible by pallas b_tok={b_tok}"

    beta_p = jnp.broadcast_to(beta[..., None], (T, H, 128)).astype(beta.dtype)
    exp_g_p = jnp.broadcast_to(exp_g[..., None], (T, H, 128)).astype(exp_g.dtype)
    valid_p = jnp.broadcast_to(valid[:, None, None], (T, 1, 128)).astype(jnp.int32)

    return pl.pallas_call(
        _pallas_gdn_decode_kernel,
        grid=(T // b_tok,),
        in_specs=[
            pl.BlockSpec((b_tok, H, D_K), lambda i: (i, 0, 0)),
            pl.BlockSpec((b_tok, H, D_K), lambda i: (i, 0, 0)),
            pl.BlockSpec((b_tok, H, D_V), lambda i: (i, 0, 0)),
            pl.BlockSpec((b_tok, H, 128), lambda i: (i, 0, 0)),
            pl.BlockSpec((b_tok, H, 128), lambda i: (i, 0, 0)),
            pl.BlockSpec((b_tok, H, D_K, D_V), lambda i: (i, 0, 0, 0)),
            pl.BlockSpec((b_tok, 1, 128), lambda i: (i, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((b_tok, H, D_V), lambda i: (i, 0, 0)),
            pl.BlockSpec((b_tok, H, D_K, D_V), lambda i: (i, 0, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((T, H, D_V), q.dtype),
            jax.ShapeDtypeStruct((T, H, D_K, D_V), state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
        name=f"pallas_gdn_decode_step_b{b_tok}",
    )(q, k, v, beta_p, exp_g_p, state, valid_p)


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
    Rule recurrence used by Qwen3-Next style models. Useful as a numerical
    check for the Pallas decode kernel.

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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies gated delta rule for decode-only case (sequence lengths = 1).

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
            ``distribution[2]`` is consulted to mask outputs of inactive
            tokens.
        use_qk_norm_in_gdn: Whether to L2-normalize Q and K before the
            decode update.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` where the state has
        shape ``(num_blocks, n_v, d_k, d_v)`` and the output has shape
        ``(num_tokens, n_v * d_v)``. Output rows for tokens with
        ``token_idx >= distribution[2]`` are zeroed and the corresponding
        state slots are left untouched.
    """
    num_tokens = query.shape[0]
    max_reqs = recurrent_state.shape[0]
    n_v = query.shape[1]
    d_k = query.shape[-1]
    d_v = value.shape[-1]

    token_idx = jnp.arange(num_tokens)
    valid_mask = token_idx < distribution[2]

    # Preprocess on-device (outside the Pallas kernel):
    #   - sigmoid/softplus/exp applied here (Pallas TPU Mosaic struggles with
    #     these on small per-head vectors)
    #   - L2 normalization + scaling
    if use_qk_norm_in_gdn:
        query = l2norm(query)
        key = l2norm(key)
    scale = jnp.asarray(d_k**-0.5, dtype=jnp.float32)
    query = (query.astype(jnp.float32) * scale).astype(query.dtype)
    beta = jax.nn.sigmoid(b_reshaped.astype(jnp.float32)).astype(b_reshaped.dtype)
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a_reshaped.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
    )
    exp_g = jnp.exp(g).astype(query.dtype)

    # Fast path: identity state map (token i → slot i) + shape-compatible.
    # Kernel produces the first `num_tokens` slots of the updated pool; we
    # splice those into the pre-existing pool prefix, leaving slots
    # `[num_tokens, max_reqs)` untouched (they hold stale state for idle
    # requests not scheduled in this step).
    pallas_btok = (
        _select_pallas_gdn_btok(num_tokens, n_v, d_k, d_v, query.dtype)
        if jax.default_backend() == "tpu" and d_k == 128 and d_v == 128 and num_tokens <= max_reqs
        else None
    )
    use_pallas = pallas_btok is not None
    if use_pallas:
        # Kernel expects slot[t] = recurrent_state[t] for t in [0, num_tokens).
        # Slice the prefix for the kernel (and recompose afterwards).
        state_prefix = recurrent_state[:num_tokens]
        outputs_3d, new_state_prefix = _pallas_gdn_decode_call(
            query,
            key,
            value,
            beta,
            exp_g,
            state_prefix,
            valid_mask,
            b_tok=int(pallas_btok),
        )
        outputs = outputs_3d.reshape(num_tokens, -1)
        if num_tokens == max_reqs:
            updated_pool = new_state_prefix.astype(recurrent_state.dtype)
        else:
            updated_pool = recurrent_state.at[:num_tokens].set(new_state_prefix)
        return updated_pool.astype(recurrent_state.dtype), outputs

    # Generic fallback (non-identity state map, smaller D_K/D_V, or non-TPU):
    # original gather-compute-scatter implementation.
    req_indices = jnp.clip(token_idx, 0, max_reqs - 1)
    req_state_indices = state_indices[req_indices]
    current_states = recurrent_state[req_state_indices]

    state_f = current_states.astype(jnp.float32)
    k_f = key.astype(jnp.float32)
    q_f = query.astype(jnp.float32)
    v_f = value.astype(jnp.float32)
    exp_g_f = exp_g.astype(jnp.float32)

    k_state = jnp.einsum("bhd,bhdm->bhm", k_f, state_f)
    q_state = jnp.einsum("bhd,bhdm->bhm", q_f, state_f)
    v_new = beta.astype(jnp.float32)[..., None] * (v_f - exp_g_f[..., None] * k_state)
    q_k = jnp.sum(q_f * k_f, axis=-1, keepdims=True)
    outputs = exp_g_f[..., None] * q_state + q_k * v_new
    k_v_new = k_f[..., :, None] * v_new[..., None, :]
    new_states = state_f * exp_g_f[..., None, None] + k_v_new

    outputs = jnp.where(valid_mask[:, None, None], outputs, 0.0)
    outputs = outputs.reshape(num_tokens, -1)
    states_to_set = jnp.where(valid_mask[:, None, None, None], new_states, state_f)
    updated_recurrent_state = recurrent_state.at[req_state_indices].set(states_to_set)

    return updated_recurrent_state.astype(recurrent_state.dtype), outputs


@jax.jit(
    donate_argnames=("recurrent_state",),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "chunk_size",
        "use_qk_norm_in_gdn",
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
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = True,
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
        n_kq: Number of key/query heads (before head expansion).
        n_v: Number of value heads (after the GQA-style expansion).
        d_k: Per-head key/query dimension.
        d_v: Per-head value dimension.
        chunk_size: Padding granularity used by the mixed-prefill branch.
        use_qk_norm_in_gdn: Whether to L2-normalize queries and keys.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` with state of shape
        ``(num_blocks, n_v, d_k, d_v)`` and output of shape
        ``(num_tokens, n_v * d_v)`` cast to ``mixed_qkv.dtype``.
    """
    num_tokens = mixed_qkv.shape[0]
    key_dim = n_kq * d_k
    query = mixed_qkv[..., :key_dim]
    key = mixed_qkv[..., key_dim : key_dim * 2]
    value = mixed_qkv[..., key_dim * 2 :]

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
        )
        return new_state, output.astype(mixed_qkv.dtype)

    def mixed_prefill_branch(_):
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
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        )

    is_decode_only = distribution[0] == distribution[2]

    return jax.lax.cond(is_decode_only, decode_only_branch, mixed_prefill_branch, operand=None)


@OperationRegistry.register
class RaggedGatedDeltaRule(OperationImpl):
    """Ragged Gated Delta Rule operation wrapping the inlined vLLM kernel.

    Provides a first-class EasyDeL operation for the ragged GDN kernel with
    ``jax.shard_map`` support for head-parallel sharding across mesh axes.

    The operation accepts already-convolved ``mixed_qkv`` (flat feature dim)
    plus raw gating inputs ``b``/``a``, ``A_log``, and ``dt_bias``, then
    internally routes to either the decode-only or mixed-prefill branch.

    Registered under the name ``"ragged_gated_delta_rule_v2"``.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registry name for this operation.

        Returns:
            str: ``"ragged_gated_delta_rule_v2"``.
        """
        return "ragged_gated_delta_rule_v2"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for RaggedGatedDeltaRule.

        Ragged GDR requires sequence metadata for packed batching and
        supports recurrent/hybrid cache types for state persistence.
        """
        return (
            RequirementsBuilder("ragged_gated_delta_rule_v2")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.POSITIONS
                | MetadataField.HAS_INITIAL_STATE
                | MetadataField.STATE_INDICES
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .build()
        )

    @jax.named_scope("easydel-ragged-gated-delta-rule-native")
    def forward_native(
        self,
        mixed_qkv: jnp.ndarray,
        b: jnp.ndarray,
        a: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        A_log: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        chunk_size: int = 64,
        use_qk_norm_in_gdn: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run :func:`ragged_gated_delta_rule` under a head-parallel ``shard_map``.

        Splits the flat ``mixed_qkv`` feature axis into Q / K / V tensors,
        expands key / query heads when ``n_v > n_kq`` (GQA-style), and
        wraps a per-shard call to :func:`ragged_gated_delta_rule` in
        :func:`jax.shard_map`. The mesh axis used for head sharding is
        resolved from the operation metadata's BTHD sharding spec, so
        this op transparently follows the model's TP layout.

        Args:
            mixed_qkv: Post-convolution mixed QKV tensor,
                shape ``[total_tokens, 2*n_kq*d_k + n_v*d_v]``.
            b: Beta gating input, shape ``[total_tokens, n_v]`` or
                ``[total_tokens]`` (will be expanded if 1-D).
            a: Alpha gating input, shape ``[total_tokens, n_v]`` or
                ``[total_tokens]`` (will be expanded if 1-D).
            recurrent_state: Recurrent state pool,
                shape ``[num_slots, n_v, d_k, d_v]``.
            A_log: Log-space decay parameters, shape ``[n_v]``.
            dt_bias: Delta-time bias, shape ``[n_v]``.
            query_start_loc: Cumulative token offsets per request,
                shape ``[num_requests + 1]``.
            state_indices: Request-to-slot mapping, shape ``[num_requests]``.
            distribution: Branch selector ``[decode_end, prefill_end, mixed_end]``,
                shape ``[3]``.
            n_kq: Number of key/query heads.
            n_v: Number of value heads.
            d_k: Key/query head dimension.
            d_v: Value head dimension.
            chunk_size: Chunk size for the mixed-prefill branch.
            use_qk_norm_in_gdn: Whether to apply QK L2 normalization.

        Returns:
            Tuple of:
            - updated_recurrent_state: shape ``[num_slots, n_v, d_k, d_v]``.
            - output: shape ``[total_tokens, n_v, d_v]``.
        """
        runtime_dtype = self.metadata.runtime_dtype
        mixed_qkv = mixed_qkv.astype(runtime_dtype)
        b = b.astype(runtime_dtype)
        a = a.astype(runtime_dtype)

        recurrent_state = recurrent_state.astype(runtime_dtype)
        A_log = A_log.astype(runtime_dtype)
        dt_bias = dt_bias.astype(runtime_dtype)

        mode = self.get_mode(query=jnp.expand_dims(mixed_qkv, 0), BTHD=False)
        shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")
        head_axis = shardings_bthd.query[2] if shardings_bthd.query is not None else None

        token_head_spec = PartitionSpec(None, head_axis, None)
        beta_spec = PartitionSpec(None, head_axis)
        state_spec = PartitionSpec(None, head_axis, None, None)
        head_param_spec = PartitionSpec(head_axis)
        Ps = PartitionSpec

        # Split flat mixed_qkv into separate Q/K/V head tensors so we can
        # shard on the head axis via shard_map.
        num_tokens = mixed_qkv.shape[0]
        key_dim = n_kq * d_k
        q = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
        k = mixed_qkv[..., key_dim : key_dim * 2].reshape(num_tokens, n_kq, d_k)
        v = mixed_qkv[..., key_dim * 2 :].reshape(num_tokens, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q = jnp.repeat(q, repeat_factor, axis=1)
            k = jnp.repeat(k, repeat_factor, axis=1)

        @partial(
            jax.shard_map,
            mesh=self.metadata.mesh,
            in_specs=(
                token_head_spec,  # q
                token_head_spec,  # k
                token_head_spec,  # v
                beta_spec,  # b
                beta_spec,  # a
                state_spec,  # recurrent_state
                head_param_spec,  # A_log
                head_param_spec,  # dt_bias
                Ps(),  # query_start_loc
                Ps(),  # state_indices
                Ps(),  # distribution
            ),
            out_specs=(state_spec, token_head_spec),
            check_vma=False,
        )
        def _mapped(
            local_q,
            local_k,
            local_v,
            local_b,
            local_a,
            local_state,
            local_A_log,
            local_dt_bias,
            local_qsl,
            local_si,
            local_dist,
        ):
            # Derive per-shard head counts from local tensor shapes.
            local_n_kq = local_q.shape[1]
            local_n_v = local_v.shape[1]
            local_d_k = local_q.shape[2]
            local_d_v = local_v.shape[2]

            local_mixed = jnp.concatenate(
                [
                    local_q.reshape(local_q.shape[0], -1),
                    local_k.reshape(local_k.shape[0], -1),
                    local_v.reshape(local_v.shape[0], -1),
                ],
                axis=-1,
            )

            new_state, output = ragged_gated_delta_rule(
                mixed_qkv=local_mixed,
                b=local_b,
                a=local_a,
                recurrent_state=local_state,
                A_log=local_A_log,
                dt_bias=local_dt_bias,
                query_start_loc=local_qsl,
                state_indices=local_si,
                distribution=local_dist,
                n_kq=local_n_kq,
                n_v=local_n_v,
                d_k=local_d_k,
                d_v=local_d_v,
                chunk_size=chunk_size,
                use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            )
            # ragged_gated_delta_rule returns output as 2D [tokens, n_v*d_v];
            # reshape to 3D [tokens, n_v, d_v] so it matches out_specs[1]'s rank.
            output = output.reshape(output.shape[0], local_n_v, local_d_v)
            return new_state, output

        new_state, output = _mapped(
            q,
            k,
            v,
            b,
            a,
            recurrent_state,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
        )
        # Restore the flat [tokens, n_v*d_v] layout expected by callers.
        output = output.reshape(output.shape[0], -1)
        return new_state, output

    def forward_tpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """TPU dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional args (see :meth:`forward_native`).
            **kwargs: Forwarded keyword args (see :meth:`forward_native`).

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: ``(updated_recurrent_state,
            output)`` from :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """GPU dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional args.
            **kwargs: Forwarded keyword args.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Same as :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """CPU dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional args.
            **kwargs: Forwarded keyword args.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Same as :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """CUDA dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional args.
            **kwargs: Forwarded keyword args.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Same as :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ROCm dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional args.
            **kwargs: Forwarded keyword args.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Same as :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)
