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

"""Forward TPU Pallas kernels for Gated Delta Rule (GDR).

Two-phase architecture for high MXU utilization on TPU v4:
  Phase 1 (parallel): Neumann inverse + state-independent quantities for ALL
           chunks simultaneously via a single pallas_call.
  Phase 2 (sequential): Lightweight lax.scan with only 4 matmuls per chunk.

Supports inference=True mode for faster forward-only execution.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm_with_inv, _recurrent_gdr_fwd

_P = lax.Precision.DEFAULT
_N_FUSE = 1


def _dot(a, b):
    """2-D matrix multiply with the module-level ``_P`` precision setting."""
    return lax.dot(a, b, precision=_P)


def _chunk_blockspec(shape: tuple[int, ...]) -> pl.BlockSpec:
    """Create a Pallas BlockSpec indexed by ``(batch, head)`` with remaining axes at 0."""
    return pl.BlockSpec(shape, lambda b, h: (b, h, *([0] * (len(shape) - 2))))


def _neumann_inv(A, C, strict_lower=None, lower_mask=None):
    """Compute ``(I - A)^{-1}`` via repeated squaring (Neumann series).

    ``A`` must be strictly lower-triangular so the series terminates exactly
    after ``C - 1`` terms.  Repeated squaring needs ``ceil(log2(C))``
    iterations to accumulate all terms up to ``A^{C-1}``.

    The computation runs at ``HIGHEST`` precision to minimise rounding error
    in the inverse.  Both ``A`` and the output are clamped with
    ``nan_to_num`` to guard against overflow in extreme inputs.

    Args:
        A: Strict lower-triangular matrix [C, C], float32. Must already be
            sanitized (NaN/Inf replaced with 0).
        C: Chunk size — determines the number of Neumann iterations needed.
        strict_lower: Optional precomputed strict-lower mask [C, C] (1 below
            diagonal, 0 on/above). Computed internally if not provided.
        lower_mask: Optional precomputed lower-triangular mask [C, C]
            (1 on diagonal and below). Computed internally if not provided.

    Returns:
        Approximation to ``(I - A)^{-1}`` as a float32 [C, C] matrix, with
        NaN/Inf entries replaced by 0.
    """
    _hp = lax.Precision.HIGHEST
    num_iters = math.ceil(math.log2(C)) if C > 1 else 0
    if strict_lower is None:
        strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.float32), k=-1)
    if lower_mask is None:
        lower_mask = strict_lower + jnp.eye(C, dtype=jnp.float32)
    S = jnp.eye(C, dtype=jnp.float32)
    P = jnp.where(strict_lower, A, 0.0)
    for _ in range(num_iters):
        S = jnp.where(lower_mask, S + lax.dot(P, S, precision=_hp), 0.0)
        P = jnp.where(strict_lower, lax.dot(P, P, precision=_hp), 0.0)
    return jnp.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)


def _process_one_chunk(q, k, v, beta, decay, state, C):
    """Run the GDR recurrence on a single chunk of length C.

    Used by the backward kernel to re-materialise intermediate states for
    chunks that precede the current one being differentiated.  All
    intermediate values are guarded with ``nan_to_num``.

    Args:
        q: Query slice [C, qk_dim], float32.
        k: Key slice [C, qk_dim], float32.
        v: Value slice [C, v_dim], float32.
        beta: Per-token gate [C], float32.
        decay: Per-token log decay [C], float32.
        state: Recurrent state at the start of this chunk [qk_dim, v_dim].
        C: Chunk size (must match ``q.shape[0]``).

    Returns:
        Tuple ``(core_out, new_state)`` where:
        * ``core_out`` is the attention output for this chunk [C, v_dim].
        * ``new_state`` is the updated recurrent state [qk_dim, v_dim].
    """
    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)
    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]
    g_cumsum = jnp.sum(lower_mask * decay[None, :], axis=1, keepdims=True)
    g_diff = g_cumsum - g_cumsum.T
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask
    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inv = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)
    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1 : C, :]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))
    value_local = jnp.nan_to_num(_dot(attn_inv, v_beta), nan=0.0, posinf=0.0, neginf=0.0)
    k_cumdecay = jnp.nan_to_num(_dot(attn_inv, k_beta * g_cumsum_exp), nan=0.0, posinf=0.0, neginf=0.0)
    attn_qk = jnp.nan_to_num(_dot(q, k.T) * decay_mask, nan=0.0, posinf=0.0, neginf=0.0)
    q_scaled = q * g_cumsum_exp
    v_prime = jnp.nan_to_num(_dot(k_cumdecay, state), nan=0.0, posinf=0.0, neginf=0.0)
    attn_inter = jnp.nan_to_num(_dot(q_scaled, state), nan=0.0, posinf=0.0, neginf=0.0)
    v_new = jnp.nan_to_num(value_local - v_prime, nan=0.0, posinf=0.0, neginf=0.0)
    core_out = attn_inter + _dot(attn_qk, v_new)
    core_out = jnp.nan_to_num(core_out, nan=0.0, posinf=0.0, neginf=0.0)
    k_scaled = k * g_diff_state_exp
    new_state = state * g_end_exp + _dot(k_scaled.T, v_new)
    new_state = jnp.nan_to_num(new_state, nan=0.0, posinf=0.0, neginf=0.0)
    return core_out, new_state


def _phase1_kernel_infer(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    decay_mask_ref,
    g_cumsum_ref,
    value_local_ref,
    k_cumdecay_ref,
    attn_qk_ref,
    q_scaled_ref,
    k_scaled_ref,
    g_end_exp_ref,
):
    """Phase 1 Pallas kernel — inference mode (precomputed decay masks).

    Accepts precomputed ``decay_mask`` and ``g_cumsum`` as inputs to avoid
    recomputing them per-chunk.  Does **not** save ``attn_inv`` in the output
    (not needed at inference time).

    Grid: ``(batch, num_heads, num_chunks)``; all axes are "arbitrary"
    (processed sequentially by the scan in Phase 2).

    BlockSpec shape convention: ``(1, 1, 1, chunk_dim)`` for every array.

    Inputs:
        q_ref: [1,1,1,C,K] query block.
        k_ref: [1,1,1,C,K] key block.
        v_ref: [1,1,1,C,V] value block.
        beta_ref: [1,1,1,1,C] gate vector.
        decay_ref: [1,1,1,1,C] log decay vector (unused — only decay_mask/
            g_cumsum are read).
        decay_mask_ref: [1,1,1,C,C] precomputed decay mask.
        g_cumsum_ref: [1,1,1,1,C] cumulative log decay.

    Outputs written:
        value_local_ref: [1,1,1,C,V] intra-chunk corrected value.
        k_cumdecay_ref: [1,1,1,C,K] decay-weighted key accumulation.
        attn_qk_ref: [1,1,1,C,C] query-key attention matrix.
        q_scaled_ref: [1,1,1,C,K] query scaled by cumulative decay.
        k_scaled_ref: [1,1,1,C,K] key scaled by state-to-end decay.
        g_end_exp_ref: [1,1,1,1,1] exp of the last cumulative decay value.
    """
    C = q_ref.shape[3]
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    beta = beta_ref[0, 0, 0, 0]
    decay_mask = decay_mask_ref[0, 0, 0].astype(jnp.float32)
    g_cumsum = g_cumsum_ref[0, 0, 0, 0]

    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inv = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum[:, None], -20.0, 20.0))
    g_end_val = g_cumsum[-1:]
    g_end_exp = jnp.exp(jnp.clip(g_end_val, -20.0, 20.0)).reshape(1, 1)
    g_diff_state_exp = jnp.exp(jnp.clip(g_end_val[:, None] - g_cumsum[:, None], -20.0, 20.0))

    k_beta_scaled = k_beta * g_cumsum_exp
    combined_rhs = jnp.concatenate([v_beta, k_beta_scaled], axis=-1)
    combined_out = _dot(attn_inv, combined_rhs)
    V = v_beta.shape[-1]
    value_local = combined_out[:, :V]
    k_cumdecay = combined_out[:, V:]

    attn_qk = _dot(q, k.T) * decay_mask
    q_scaled = q * g_cumsum_exp
    k_scaled = k * g_diff_state_exp

    value_local_ref[0, 0, 0] = value_local.astype(value_local_ref.dtype)
    k_cumdecay_ref[0, 0, 0] = k_cumdecay.astype(k_cumdecay_ref.dtype)
    attn_qk_ref[0, 0, 0] = attn_qk.astype(attn_qk_ref.dtype)
    q_scaled_ref[0, 0, 0] = q_scaled.astype(q_scaled_ref.dtype)
    k_scaled_ref[0, 0, 0] = k_scaled.astype(k_scaled_ref.dtype)
    g_end_exp_ref[0, 0, 0] = g_end_exp.astype(g_end_exp_ref.dtype)


def _phase1_kernel_train(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    value_local_ref,
    k_cumdecay_ref,
    attn_qk_ref,
    q_scaled_ref,
    k_scaled_ref,
    g_end_exp_ref,
    attn_inv_ref,
):
    """Phase 1 Pallas kernel — training mode (computes and saves attn_inv).

    Computes all chunk-local intermediates from scratch and writes
    ``attn_inv`` to an output ref so the backward pass can use it without
    re-running the Neumann series.

    Grid: ``(batch, num_heads, num_chunks)``; the chunk axis is "arbitrary".

    BlockSpec shape convention: ``(1, 1, 1, chunk_dim)`` for every array.

    Inputs:
        q_ref: [1,1,1,C,K] query block.
        k_ref: [1,1,1,C,K] key block.
        v_ref: [1,1,1,C,V] value block.
        beta_ref: [1,1,1,1,C] gate vector.
        decay_ref: [1,1,1,1,C] log decay vector.

    Outputs written (same layout as inference kernel plus):
        value_local_ref: [1,1,1,C,V].
        k_cumdecay_ref: [1,1,1,C,K].
        attn_qk_ref: [1,1,1,C,C].
        q_scaled_ref: [1,1,1,C,K].
        k_scaled_ref: [1,1,1,C,K].
        g_end_exp_ref: [1,1,1,1,1].
        attn_inv_ref: [1,1,1,C,C] — ``(I - A)^{-1}`` saved for backward.
    """
    C = q_ref.shape[3]
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    beta = beta_ref[0, 0, 0, 0]
    decay = decay_ref[0, 0, 0, 0]

    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    g_cumsum = jnp.sum(lower_mask * decay[None, :], axis=1, keepdims=True)
    g_diff = g_cumsum - g_cumsum.T
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask

    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inv = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1 : C, :]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))

    value_local = _dot(attn_inv, v_beta)
    k_cumdecay = _dot(attn_inv, k_beta * g_cumsum_exp)
    attn_qk = _dot(q, k.T) * decay_mask
    q_scaled = q * g_cumsum_exp
    k_scaled = k * g_diff_state_exp

    def _s(x):
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    value_local_ref[0, 0, 0] = _s(value_local).astype(value_local_ref.dtype)
    k_cumdecay_ref[0, 0, 0] = _s(k_cumdecay).astype(k_cumdecay_ref.dtype)
    attn_qk_ref[0, 0, 0] = _s(attn_qk).astype(attn_qk_ref.dtype)
    q_scaled_ref[0, 0, 0] = q_scaled.astype(q_scaled_ref.dtype)
    k_scaled_ref[0, 0, 0] = k_scaled.astype(k_scaled_ref.dtype)
    g_end_exp_ref[0, 0, 0] = jnp.broadcast_to(g_end_exp, (1, 1)).astype(g_end_exp_ref.dtype)
    attn_inv_ref[0, 0, 0] = attn_inv.astype(attn_inv_ref.dtype)


def _run_phase1(query_c, key_c, value_c, beta_c, decay_c, *, inference=False):
    """Launch the Phase 1 Pallas kernel over ALL chunks simultaneously.

    Dispatches to either ``_phase1_kernel_infer`` (faster, no ``attn_inv``
    saved) or ``_phase1_kernel_train`` (saves ``attn_inv`` for backward).

    Args:
        query_c: Reshaped query [B, H, NC, C, K].
        key_c: Reshaped key [B, H, NC, C, K].
        value_c: Reshaped value [B, H, NC, C, V].
        beta_c: Reshaped gate [B, H, NC, 1, C].
        decay_c: Reshaped log decay [B, H, NC, 1, C].
        inference: If True, precomputes decay masks in XLA before the Pallas
            call and uses the faster inference kernel.

    Returns:
        Seven-element tuple:
        ``(value_local, k_cumdecay, attn_qk, q_scaled, k_scaled, g_end_exp,
        attn_inv)`` each of shape ``[B, H, NC, ...]``.  ``attn_inv`` is
        ``None`` in inference mode.
    """
    B, H, NC, C, K = query_c.shape
    V = value_c.shape[-1]

    def bs3(shape):
        return pl.BlockSpec((1, 1, 1, *shape), lambda b, h, c: (b, h, c, *([0] * len(shape))))

    if inference:
        decay_flat = decay_c.squeeze(-2)
        g_cumsum = jnp.cumsum(decay_flat, axis=-1)
        g_cs = g_cumsum[..., None]
        lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
        decay_mask = jnp.exp(jnp.clip((g_cs - g_cs.transpose(0, 1, 2, 4, 3)) * lower_mask, -20.0, 20.0)) * lower_mask
        g_cumsum_input = g_cumsum.reshape(B, H, NC, 1, C).astype(jnp.float32)

        call = pl.pallas_call(
            _phase1_kernel_infer,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((C, V)),
                    bs3((1, C)),
                    bs3((1, C)),
                    bs3((C, C)),
                    bs3((1, C)),
                ],
                out_specs=[
                    bs3((C, V)),
                    bs3((C, K)),
                    bs3((C, C)),
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((1, 1)),
                ],
                grid=(B, H, NC),
            ),
            out_shape=[
                jax.ShapeDtypeStruct((B, H, NC, C, V), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, C), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, 1, 1), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
            ),
        )
        results = call(query_c, key_c, value_c, beta_c, decay_c, decay_mask, g_cumsum_input)
        return (*results, None)
    else:
        call = pl.pallas_call(
            _phase1_kernel_train,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((C, V)),
                    bs3((1, C)),
                    bs3((1, C)),
                ],
                out_specs=[
                    bs3((C, V)),
                    bs3((C, K)),
                    bs3((C, C)),
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((1, 1)),
                    bs3((C, C)),
                ],
                grid=(B, H, NC),
            ),
            out_shape=[
                jax.ShapeDtypeStruct((B, H, NC, C, V), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, C), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, 1, 1), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, C), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
            ),
        )
        return call(query_c, key_c, value_c, beta_c, decay_c)


def _phase2_scan_body(state, inputs):
    """Phase 2 scan body (training mode): 4 matmuls + element-wise ops.

    Consumes the precomputed Phase-1 intermediates for one chunk and
    updates the recurrent state.  All intermediate results are sanitized
    with ``nan_to_num`` to ensure stable gradients in training.

    Args:
        state: Recurrent state [B, H, K, V], float32.
        inputs: Six-element tuple per chunk step:
            ``(value_local, k_cumdecay, attn_qk, q_scaled, k_scaled,
            g_end_exp)``.

    Returns:
        Tuple ``(new_state, (core_out, state))`` compatible with
        ``jax.lax.scan``.  ``state`` is the state *before* the update
        (needed by the backward scan).
    """
    value_local, k_cumdecay, attn_qk, q_scaled, k_scaled, g_end_exp = inputs

    def _s(x):
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    v_prime = _s(jnp.einsum("bhck,bhkv->bhcv", k_cumdecay, state))
    attn_inter = _s(jnp.einsum("bhck,bhkv->bhcv", q_scaled, state))
    v_new = _s(value_local - v_prime)
    core_out = _s(attn_inter + jnp.einsum("bhcr,bhrv->bhcv", attn_qk, v_new))

    state_update = jnp.einsum("bhkc,bhcv->bhkv", k_scaled.transpose(0, 1, 3, 2), v_new)
    new_state = _s(state * g_end_exp + state_update)

    return new_state, (core_out, state)


def _phase2_scan_body_infer(state, inputs):
    """Phase 2 scan body (inference mode): faster variant without nan-guards.

    Identical computation to ``_phase2_scan_body`` but skips
    ``nan_to_num`` on intermediate tensors for better throughput.
    Only the new state carry is sanitized to prevent NaN propagation.

    Args:
        state: Recurrent state [B, H, K, V], float32.
        inputs: Six-element tuple ``(value_local, k_cumdecay, attn_qk,
            q_scaled, k_scaled, g_end_exp)`` for the current chunk.

    Returns:
        Tuple ``(new_state, (core_out, state))`` for ``jax.lax.scan``.
    """
    value_local, k_cumdecay, attn_qk, q_scaled, k_scaled, g_end_exp = inputs

    v_prime = jnp.einsum("bhck,bhkv->bhcv", k_cumdecay, state)
    attn_inter = jnp.einsum("bhck,bhkv->bhcv", q_scaled, state)
    v_new = value_local - v_prime
    core_out = attn_inter + jnp.einsum("bhcr,bhrv->bhcv", attn_qk, v_new)

    state_update = jnp.einsum("bhkc,bhcv->bhkv", k_scaled.transpose(0, 1, 3, 2), v_new)
    new_state = jnp.nan_to_num(state * g_end_exp + state_update, nan=0.0, posinf=0.0, neginf=0.0)

    return new_state, (core_out, state)


def _chunk_gdr_fwd_core(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
    *,
    save_residual: bool,
    inference: bool = False,
):
    """Two-phase chunked GDR forward pass (shared by training and inference).

    Phase 1 (parallel Pallas):
        Runs all chunks in parallel to compute intra-chunk intermediates
        (``value_local``, ``k_cumdecay``, ``attn_qk``, ``q_scaled``,
        ``k_scaled``, ``g_end_exp``, and optionally ``attn_inv``).

    Phase 2 (sequential lax.scan):
        Applies a lightweight 4-matmul scan that consumes the Phase-1
        outputs and threads the recurrent state through all chunks.

    Tensor layout (internally): ``[B, H, L, dim]`` where
    ``B=batch, H=num_heads``.

    Args:
        query: [B, H, L, K] float.
        key: [B, H, L, K] float.
        value: [B, H, L, V] float.
        beta: [B, H, L] float — per-token gate.
        decay: [B, H, L] float or None — per-token log decay.
        chunk_size: Number of tokens per chunk. Padded up to a multiple of
            ``chunk_size`` if ``L % chunk_size != 0``.
        initial_state: [B, H, K, V] float or None.
        use_qk_l2norm: Apply L2 normalisation to Q and K before computation.
        save_residual: If True, package and return the residual tuple needed
            by the backward pass.
        inference: If True, use the faster inference kernel (precomputed
            decay masks, no ``attn_inv`` saved, fewer NaN guards).

    Returns:
        Three-element tuple ``(output, final_state, residual)`` where
        ``residual`` is the backward-residual tuple when ``save_residual``
        is True, otherwise ``None``.
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    input_dtype = query.dtype
    decay_was_none = decay is None
    initial_state_was_none = initial_state is None

    q_inv_norm = k_inv_norm = None
    if use_qk_l2norm:
        query, q_inv_norm = _l2norm_with_inv(query, axis=-1, eps=1e-6)
        key, k_inv_norm = _l2norm_with_inv(key, axis=-1, eps=1e-6)

    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=input_dtype)
    else:
        decay = decay.astype(input_dtype)

    pad_size = (chunk_size - L % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))

    num_chunks = (L + pad_size) // chunk_size
    scale = 1.0 / math.sqrt(K_dim)
    query = query * scale

    query_c = query.reshape(B, H, num_chunks, chunk_size, K_dim)
    key_c = key.reshape(B, H, num_chunks, chunk_size, K_dim)
    value_c = value.reshape(B, H, num_chunks, chunk_size, V_dim)
    beta_c = beta.reshape(B, H, num_chunks, 1, chunk_size).astype(jnp.float32)
    decay_c = decay.reshape(B, H, num_chunks, 1, chunk_size).astype(jnp.float32)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    value_local, k_cumdecay, attn_qk, q_scaled, k_scaled, g_end_exp, _attn_inv = _run_phase1(
        query_c,
        key_c,
        value_c,
        beta_c,
        decay_c,
        inference=inference,
    )

    scan_inputs = (
        value_local.transpose(2, 0, 1, 3, 4),
        k_cumdecay.transpose(2, 0, 1, 3, 4),
        attn_qk.transpose(2, 0, 1, 3, 4),
        q_scaled.transpose(2, 0, 1, 3, 4),
        k_scaled.transpose(2, 0, 1, 3, 4),
        g_end_exp.transpose(2, 0, 1, 3, 4),
    )

    scan_fn = _phase2_scan_body_infer if inference else _phase2_scan_body
    final_state, (core_out_tm, state_pre_tm) = lax.scan(
        scan_fn,
        initial_state,
        scan_inputs,
    )

    core_attn_out = core_out_tm.transpose(1, 2, 0, 3, 4)
    core_attn_out = core_attn_out.reshape(B, H, -1, V_dim)[:, :, :L, :]
    final_state_out = final_state.astype(input_dtype)

    if not save_residual:
        return core_attn_out, final_state_out, None

    state_pre_all = state_pre_tm.transpose(1, 2, 0, 3, 4)
    residual = (
        query_c,
        key_c,
        value_c,
        beta_c.squeeze(-2),
        decay_c.squeeze(-2),
        state_pre_all,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        L,
        pad_size,
        decay_was_none,
        initial_state_was_none,
        chunk_size,
    )
    return core_attn_out, final_state_out, residual


def _chunk_gdr_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Inference-only wrapper: calls ``_chunk_gdr_fwd_core`` without saving residuals."""
    output, final_state, _ = _chunk_gdr_fwd_core(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        save_residual=False,
        inference=True,
    )
    return output, final_state


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 7))
def _chunk_gdr_fwd_pallas_chunk(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Chunked forward pass for GDR on TPU via 2-phase Pallas kernel."""
    return _chunk_gdr_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm)


def _chunk_gdr_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Custom-VJP forward rule: training mode — runs Phase 1+2 and saves residuals.

    Returns:
        Tuple ``((output, final_state), residual)`` as required by
        ``jax.custom_vjp``.
    """
    output, final_state, residual = _chunk_gdr_fwd_core(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        save_residual=True,
        inference=False,
    )
    return (output, final_state), residual


def _chunk_gdr_bwd_rule(chunk_size, use_qk_l2norm, res, g):
    """Custom-VJP backward rule: delegates to ``_chunk_gdr_bwd`` in ``_pallas_impl_bwd``."""
    from ._pallas_impl_bwd import _chunk_gdr_bwd

    return _chunk_gdr_bwd(chunk_size, use_qk_l2norm, res, g)


_chunk_gdr_fwd_pallas_chunk.defvjp(_chunk_gdr_fwd_rule, _chunk_gdr_bwd_rule)


def _chunk_gdr_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Exact multi-token chunked GDR forward path.

    Multi-token training/prefill is routed through the XLA recurrent
    implementation (``_recurrent_gdr_fwd``) for numerical stability, while
    the optimised Pallas single-token decode kernel remains active for
    ``seq_len == 1`` decoding.

    Args:
        query: [B, H, L, K].
        key: [B, H, L, K].
        value: [B, H, L, V].
        beta: [B, H, L].
        decay: [B, H, L] or None.
        chunk_size: Passed through to ``_recurrent_gdr_fwd``.
        initial_state: [B, H, K, V] or None.
        use_qk_l2norm: Apply L2 normalisation to Q/K.

    Returns:
        ``(output [B, H, L, V], final_state [B, H, K, V])``.
    """
    return _recurrent_gdr_fwd(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size,
    )


def _gdr_single_step_fwd_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_ref, out_ref, final_state_ref):
    """Pallas kernel for one GDR decode step (seq_len == 1).

    Grid: ``(batch, num_heads)``; both axes are "parallel".

    Inputs (all with BlockSpec ``(1, 1, ..., dim)``):
        q_ref: [B, H, 1, 1, K] — query for the current token.
        k_ref: [B, H, 1, 1, K] — key for the current token.
        v_ref: [B, H, 1, 1, V] — value for the current token.
        beta_ref: [B, H, 1, 1] — gate scalar.
        decay_ref: [B, H, 1, 1] — log decay scalar.
        state_ref: [B, H, 1, K, V] — previous recurrent state.

    Outputs written:
        out_ref: [B, H, 1, 1, V] — attention output for the token.
        final_state_ref: [B, H, 1, K, V] — updated recurrent state.

    Computation:
        state_decayed = state_prev * exp(decay_t)
        kv_mem        = state_decayed @ k_t         (K -> V projection)
        delta         = (v_t - kv_mem) * beta_t
        state_new     = state_decayed + k_t[:, None] * delta[None, :]
        out_t         = state_new @ q_t
    """
    q_t = q_ref[0, 0, 0].astype(jnp.float32)
    k_t = k_ref[0, 0, 0].astype(jnp.float32)
    v_t = v_ref[0, 0, 0].astype(jnp.float32)
    beta_t = beta_ref[0, 0].reshape(())
    g_exp = jnp.exp(jnp.clip(decay_ref[0, 0].reshape(()), -20.0, 20.0))
    state_prev = state_ref[0, 0].astype(jnp.float32)
    state_decayed = state_prev * g_exp
    kv_mem = jnp.sum(state_decayed * k_t[:, None], axis=0)
    delta = (v_t - kv_mem) * beta_t
    state = state_decayed + k_t[:, None] * delta[None, :]
    out = jnp.sum(state * q_t[:, None], axis=0)
    out_ref[0, 0, 0] = out.astype(out_ref.dtype)
    final_state_ref[0, 0] = state.astype(final_state_ref.dtype)


def _gdr_single_step_fwd_dma_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    state_ref,
    out_ref,
    final_state_ref,
    state_tile_ref,
    dma_sem_ref,
):
    """DMA-backed Pallas kernel for one GDR decode step.

    The recurrent state is the only operand large enough to amortize DMA
    setup. It is copied asynchronously while the per-token q/k/v vectors and
    scalar gate/decay values are loaded directly.
    """
    qk_dim = q_ref.shape[3]
    v_dim = v_ref.shape[3]
    state_copy = pltpu.make_async_copy(
        src_ref=state_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, qk_dim), pl.ds(0, v_dim)],
        dst_ref=state_tile_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, qk_dim), pl.ds(0, v_dim)],
        sem=dma_sem_ref.at[0],
    )
    state_copy.start()

    k_t = k_ref[0, 0, 0].astype(jnp.float32)
    v_t = v_ref[0, 0, 0].astype(jnp.float32)
    beta_t = beta_ref[0, 0, 0, 0]
    g_exp = jnp.exp(jnp.clip(decay_ref[0, 0, 0, 0], -20.0, 20.0))
    state_copy.wait()
    state_prev = state_tile_ref[0, 0].astype(jnp.float32)
    state_decayed = state_prev * g_exp
    kv_mem = jnp.sum(state_decayed * k_t[:, None], axis=0)
    delta = (v_t - kv_mem) * beta_t
    state = state_decayed + k_t[:, None] * delta[None, :]

    q_t = q_ref[0, 0, 0].astype(jnp.float32)
    out = jnp.sum(state * q_t[:, None], axis=0)
    out_ref[0, 0, 0] = out.astype(out_ref.dtype)
    final_state_ref[0, 0] = state.astype(final_state_ref.dtype)


def _run_single_step_forward(query, key, value, beta, decay, recurrent_state):
    """Launch the single-step GDR Pallas kernel and return (output, new_state).

    Args:
        query: [B, H, 1, K], already L2-normalised and scaled.
        key: [B, H, 1, K], already L2-normalised.
        value: [B, H, 1, V].
        beta: [B, H, 1, 1] float32 gate.
        decay: [B, H, 1, 1] float32 log decay.
        recurrent_state: [B, H, K, V].

    Returns:
        Tuple ``(output [B, H, 1, V], new_state [B, H, K, V])``.
    """
    bsz, num_heads, _, qk_dim = query.shape
    v_dim = value.shape[-1]
    beta = beta[..., None].astype(jnp.float32)
    decay = decay[..., None].astype(jnp.float32)
    call = pl.pallas_call(
        _gdr_single_step_fwd_dma_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[_chunk_blockspec((1, 1, 1, v_dim)), _chunk_blockspec((1, 1, qk_dim, v_dim))],
            scratch_shapes=[
                pltpu.VMEM((1, 1, qk_dim, v_dim), recurrent_state.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            grid=(bsz, num_heads),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_heads, 1, v_dim), query.dtype),
            jax.ShapeDtypeStruct((bsz, num_heads, qk_dim, v_dim), recurrent_state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )
    return call(query, key, value, beta, decay, recurrent_state)


def _single_step_gdr_fwd_impl(query, key, value, beta, decay, recurrent_state, use_qk_l2norm):
    """Shared forward computation for the single-step GDR path (forward + backward rule).

    Applies optional L2-normalisation, query scaling, and zero-decay fill-in,
    then calls the Pallas kernel.  Packages a residual tuple for the backward rule.

    Args:
        query: [B, H, 1, K].
        key: [B, H, 1, K].
        value: [B, H, 1, V].
        beta: [B, H, 1] gate.
        decay: [B, H, 1] log decay, or None (treated as zeros).
        recurrent_state: [B, H, K, V].
        use_qk_l2norm: Apply L2 normalisation to Q/K.

    Returns:
        Tuple ``(output, final_state, residual)`` where ``residual`` contains
        all tensors needed by the backward rule.
    """
    input_dtype = query.dtype
    decay_was_none = decay is None
    q_inv_norm = k_inv_norm = None
    query = query.astype(input_dtype)
    key = key.astype(input_dtype)
    value = value.astype(input_dtype)
    beta = beta.astype(input_dtype)
    if use_qk_l2norm:
        query, q_inv_norm = _l2norm_with_inv(query, axis=-1, eps=1e-6)
        key, k_inv_norm = _l2norm_with_inv(key, axis=-1, eps=1e-6)
    scale = 1.0 / math.sqrt(query.shape[-1])
    query = query * scale
    if decay is None:
        decay = jnp.zeros(beta.shape, dtype=input_dtype)
    else:
        decay = decay.astype(input_dtype)
    recurrent_state = recurrent_state.astype(input_dtype)
    output, final_state = _run_single_step_forward(query, key, value, beta, decay, recurrent_state)
    residual = (query, key, value, beta, decay, recurrent_state, q_inv_norm, k_inv_norm, decay_was_none)
    return output, final_state, residual


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _single_step_gdr_fwd(query, key, value, beta, decay, recurrent_state, use_qk_l2norm=True):
    """Single-step GDR forward with custom VJP for gradient support.

    Non-diff arg: ``use_qk_l2norm`` (index 6).

    Args:
        query: [B, H, 1, K].
        key: [B, H, 1, K].
        value: [B, H, 1, V].
        beta: [B, H, 1].
        decay: [B, H, 1] or None.
        recurrent_state: [B, H, K, V].
        use_qk_l2norm: Apply L2-normalisation to Q/K.

    Returns:
        ``(output [B, H, 1, V], final_state [B, H, K, V])``.
    """
    output, final_state, _ = _single_step_gdr_fwd_impl(query, key, value, beta, decay, recurrent_state, use_qk_l2norm)
    return output, final_state


def _single_step_gdr_fwd_rule(query, key, value, beta, decay, recurrent_state, use_qk_l2norm):
    """Custom-VJP forward rule for single-step GDR: runs and saves residuals."""
    output, final_state, residual = _single_step_gdr_fwd_impl(
        query, key, value, beta, decay, recurrent_state, use_qk_l2norm
    )
    return (output, final_state), residual


def _single_step_gdr_bwd_rule(use_qk_l2norm, res, g):
    """Custom-VJP backward rule for single-step GDR: delegates to Pallas bwd kernel."""
    from ._pallas_impl_bwd import _single_step_gdr_bwd

    return _single_step_gdr_bwd(use_qk_l2norm, res, g)


_single_step_gdr_fwd.defvjp(_single_step_gdr_fwd_rule, _single_step_gdr_bwd_rule)
