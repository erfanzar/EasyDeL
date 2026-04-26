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
"""Ragged gated delta rule packed JAX implementation."""

import enum
import functools
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


def newton_schulz_inverse_ref(A, n=None):
    """Inverse of unit lower triangular matrix using Newton-Schulz iteration.

    Args:
      A: Tensor with last two dimensions representing a square lower triangular
        matrix with unit diagonal.
      n: Number of iterations to run.

    Newton Schulz iteration:
    https://en.wikipedia.org/wiki/Matrix_sign_function#Newton%E2%80%93Schulz_iteration
    S_{k+1} = S_k @ (2 * I - A @ S_k)

    Let L = A - I
    Starting with S_0 = I, this is equivalent mathematically to
    S_k = (I - L) @ (I + L^2) @ (I + L^4)....(I + (L^(2^k))), k > 0

    If L is strictly lower (or upper) triangular, L ^ n == 0.
    So this series converges after log(n) steps.

    We don't directly compute S_k as above to reduce precision loss.
    We run the last step in higher precision to improve the overall estimate.
    Initial steps are kept in lower precision for speed.

    Returns:
      Inverse of A.
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
    x_ref[...] = newton_schulz_inverse_ref(A_ref[...])


def newton_schulz_inverse_pallas(A, *, block_size=64):
    """Newton-Schulz iteration for unit lower triangular matrices on Pallas."""

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
    """Solves A X = B for unit lower triangular matrix A using forward substitution.

    Args:
      A: A tensor of shape (B, N, N) representing a batch of unit lower triangular
        matrices.
      b: A tensor of shape (B, N, K) representing the right-hand side.

    Returns:
      A tensor of shape (B, N, K) representing the solution X.
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
    """Inverts unit lower triangular matrices using a block-wise approach in Pallas.

    This function solves A X = I for X, where A is a unit lower triangular matrix.
    It uses a block-wise Gaussian elimination approach to improve performance.

    Args:
      A: A tensor of shape (batch_size, chunks, heads, head_dim, head_dim) where
        the last two dimensions represent unit lower triangular matrices.
      n_block_size: The block size for Pallas grid execution.
      block_size: The block size for the block-wise inversion algorithm.

    Returns:
      A tensor of the same shape as A, representing the inverse of A.
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
    """Pure-JAX inverse of unit lower-triangular matrices (any backend).

    Uses ``jax.scipy.linalg.solve_triangular`` to solve :math:`A X = I`
    column-by-column.  Works on TPU, GPU and CPU without Pallas.
    """
    shape = A.shape
    A_2d = A.reshape(-1, shape[-2], shape[-1])
    N = shape[-1]
    I_batch = jnp.broadcast_to(jnp.eye(N, dtype=A.dtype), A_2d.shape)
    X = jax.scipy.linalg.solve_triangular(A_2d, I_batch, lower=True, unit_diagonal=True)
    return X.reshape(shape)


class TriangleSolverImpl(enum.StrEnum):
    GAUSSIAN = "gaussian"
    NEWTON_SCHULZ = "newton_schulz"
    JAX = "jax"

    def __call__(self, A):
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
    """Normalizes x along the specified dimension using L2 norm.

    Args:
      x: Input array.
      dim: Dimension along which to normalize.
      eps: Epsilon value to avoid division by zero.

    Returns:
      Normalized array.
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
      query: Query tensor.
      key: Key tensor.
      value: Value tensor.
      g: Gate tensor.
      beta: Beta tensor.
      query_start_loc: Start locations of each sequence in original stream.
      distribution: Distribution tensor containing number of valid sequences at
        index 2.
      chunk_size: Chunk size for padding.
      compute_dtype: Dtype for computation (Q, K, V, beta).

    Returns:
      A tuple containing:
        - packed_query: Packed query tensor.
        - packed_key: Packed key tensor.
        - packed_value: Packed value tensor.
        - packed_g: Packed gate tensor.
        - packed_beta: Packed beta tensor.
        - reset_mask: Mask indicating start of sequences (per chunk).
        - new_query_start_loc: Start locations in packed stream.
        - padded_indices_valid: Indices mapping original to packed.
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
      query: Query tensor.
      key: Key tensor.
      value: Value tensor.
      b_reshaped: Reshaped b tensor (for beta).
      a_reshaped: Reshaped a tensor (for g).
      A_log: A_log tensor.
      dt_bias: dt_bias tensor.
      query_start_loc: Start locations of sequences in original stream.
      recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
        d_v)`. `num_blocks` is always equal or larger than `max_seqs + 1`. The
        first block is a null_block and only used for padded / invalid tokens.
      state_indices: Indices mapping sequences to recurrent state slots.
      distribution: Distribution tensor containing number of valid sequences at
        index 2.
      chunk_size: Chunk size for padding and processing.
      use_qk_norm_in_gdn: Whether to use QK normalization.
      compute_dtype: Dtype for computation.
      precision: Precision for matrix multiplication.
      preferred_element_type: Preferred element type for matrix multiplication.
      triangle_solver_impl: Which triangle solver implementation to use.

    Returns:
      A tuple containing:
        - updated_recurrent_state: Updated recurrent state tensor of shape
          `(num_blocks, n_v, d_k, d_v)`.
        - output: Output tensor.
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


_PALLAS_GDN_BTOK = 16


def _pallas_gdn_decode_call(q, k, v, beta, exp_g, state, valid):
    """Run the fused Pallas kernel. Returns (new_state_pool, outputs_3d).

    Assumes:
        - state_indices is identity: token i corresponds to slot i.
        - ``q.shape[0]`` is divisible by ``_PALLAS_GDN_BTOK``.
    """
    T, H, D_K = q.shape
    D_V = v.shape[-1]
    b_tok = _PALLAS_GDN_BTOK
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
    """Single-step recurrent update for decode."""
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
      query: Query tensor.
      key: Key tensor.
      value: Value tensor.
      b_reshaped: Reshaped b tensor (for beta).
      a_reshaped: Reshaped a tensor (for g).
      recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
        d_v)`. `num_blocks` is always equal or larger than `max_seqs + 1`. The
        first block is a null_block and only used for padded / invalid tokens.
      A_log: A_log tensor.
      dt_bias: dt_bias tensor.
      query_start_loc: Start locations of sequences.
      state_indices: Indices mapping sequences to recurrent state slots.
      distribution: Distribution tensor containing number of valid sequences at
        index 2.
      use_qk_norm_in_gdn: Whether to use QK normalization.

    Returns:
      A tuple containing:
        - updated_recurrent_state: Updated recurrent state tensor of shape
          `(num_blocks, n_v, d_k, d_v)`.
        - output: Output tensor.
    """
    num_tokens = query.shape[0]
    max_reqs = recurrent_state.shape[0]
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
    use_pallas = (
        jax.default_backend() == "tpu"
        and d_k == 128
        and d_v == 128
        and num_tokens <= max_reqs
        and num_tokens >= _PALLAS_GDN_BTOK
        and num_tokens % _PALLAS_GDN_BTOK == 0
    )
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
      mixed_qkv: Mixed query, key, value tensor.
      b: b tensor (for beta).
      a: a tensor (for g).
      recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
        d_v)`. `num_blocks` is always equal or larger than `max_reqs + 1`. The
        first block is a null_block and only used for padded / invalid tokens.
      A_log: A_log tensor.
      dt_bias: dt_bias tensor.
      query_start_loc: Start locations of sequences.
      state_indices: Indices mapping sequences to recurrent state slots.
      distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
        mixed_end)`.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key/query dimension.
      d_v: Value dimension.
      chunk_size: Chunk size for padding in mixed prefill.
      use_qk_norm_in_gdn: Whether to use QK normalization.

    Returns:
      A tuple containing:
        - updated_recurrent_state: Updated recurrent state tensor of shape
          `(num_blocks, n_v, d_k, d_v)`.
        - output: Output tensor.
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
        """Forward pass for ragged gated delta rule.

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
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forward_native(*args, **kwargs)
