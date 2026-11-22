# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""XLA implementation for ragged causal depthwise conv1d."""

import jax
import jax.numpy as jnp


def _fix_query_start_loc(query_start_loc: jnp.ndarray, num_valid_seqs: jnp.ndarray) -> jnp.ndarray:
    """Clamp trailing padding entries of ``query_start_loc`` to a sentinel value.

    The schedule sometimes carries inactive trailing slots whose
    ``query_start_loc`` entries are not monotone. This helper rewrites all
    entries past ``num_valid_seqs`` to ``query_start_loc[num_valid_seqs]``
    so downstream length / boundary arithmetic produces zero-length
    sequences for those slots.

    Args:
        query_start_loc: Cumulative per-request token offsets, shape
            ``(num_slots + 1,)``.
        num_valid_seqs: Scalar number of valid (non-padding) sequences.

    Returns:
        jnp.ndarray: A copy of ``query_start_loc`` with padding entries
        clamped to the last valid offset.
    """
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    return jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)


def _depthwise_conv1d_flat(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Run a depthwise causal conv over the flat packed token stream.

    Performs ``out[t, :] = sum_k padded_x[t + k, :] * kernel[:, k]`` with
    a ``(d_conv - 1)``-wide left pad of zeros, in float32. The output
    boundary tokens at the seam between requests are *not* yet correct —
    they are rewritten by the boundary fix-up in
    :func:`_ragged_causal_conv1d_impl`.

    Args:
        x: Packed input tokens, shape ``(num_tokens, conv_dim)``.
        kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``.

    Returns:
        jnp.ndarray: Conv output in float32, shape
        ``(num_tokens, conv_dim)``.
    """
    num_tokens = x.shape[0]
    d_conv = kernel.shape[-1]
    padded_x = jnp.pad(x.astype(jnp.float32), ((d_conv - 1, 0), (0, 0)))
    kernel = kernel.astype(jnp.float32)
    out = jnp.zeros((num_tokens, x.shape[-1]), dtype=jnp.float32)
    for k in range(d_conv):
        out = out + padded_x[k : k + num_tokens, :] * kernel[None, :, k]
    return out


def _get_boundary_indices(
    starts: jnp.ndarray,
    lengths: jnp.ndarray,
    d_conv: int,
    num_valid_seqs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the gather/scatter indices used to fix per-request boundary tokens.

    For each request, the first ``d_conv - 1`` output tokens depend on
    historical state from the previous step rather than the zeros that the
    flat conv produced. This helper materialises:

    * ``gather_indices`` — token positions in ``x`` to read the head of
      each request from (clamped to the request length so we never index
      past its end).
    * ``scatter_indices`` — destination output positions to rewrite with
      the corrected boundary values, with ``-1`` for invalid (padding)
      entries.

    Args:
        starts: Per-request start offsets, shape ``(num_slots,)``.
        lengths: Per-request lengths, shape ``(num_slots,)``.
        d_conv: Convolution window size.
        num_valid_seqs: Scalar number of valid sequences.

    Returns:
        tuple: ``(gather_indices, scatter_indices)`` with shape
        ``(num_slots, d_conv - 1)`` each, both ``int32``.
    """
    valid_mask = jnp.arange(starts.shape[0]) < num_valid_seqs
    starts = jnp.where(valid_mask, starts, 1)[:, None]
    lengths = lengths[:, None]
    k_range = jnp.arange(d_conv - 1)[None, :]
    gather_indices = starts + jnp.minimum(k_range, lengths - 1)
    scatter_indices = jnp.where(
        (k_range < lengths) & valid_mask[:, None],
        starts + k_range,
        -1,
    )
    return gather_indices, scatter_indices


def _ragged_causal_conv1d_impl(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    *,
    d_conv: int,
    apply_silu: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies causal depthwise conv1d over ragged sequences with rolling state.

    Single-pass kernel that computes one depthwise-conv output per token and
    refreshes each slot's rolling state to the last ``d_conv`` tokens of its
    request. Handles decode (single-token) and prefill (multi-token) slots
    uniformly in the same packed batch. SiLU is fused in by default.

    Algorithm
    ---------
    For each packed token at global index ``t`` belonging to request ``r`` at
    local offset ``local = t - q_start[r]``, the output is

        output[t, :] = silu( sum_{k = 0..d_conv - 1}
                             kernel[:, d_conv - 1 - k] * tok(local - k) )

    where ``tok(local - k)`` is resolved as:

        * ``x[t - k, :]``            if ``local - k >= 0``  (same request)
        * ``conv_state[r, :, j]``    if ``local - k <  0``  (historical), with
                                     ``j = d_conv + local - k``

    The kernel orientation follows the depthwise-conv convention: summing
    over the last axis of ``state * kernel`` is equivalent to a PyTorch-style
    causal depthwise conv where ``kernel[:, d_conv - 1]`` is the "current" tap
    and ``kernel[:, 0]`` is the ``d_conv - 1``-step-back tap.

    The state update produces, for each slot ``r``,

        new_state[r, :, j] = tok at seq pos ``(L_r - d_conv + j)``

    where ``L_r`` is the slot's request length. For positions that fall before
    the start of the current request (``L_r + j < d_conv``), the value is
    carried over from the incoming ``conv_state``; otherwise it is sourced from
    ``x``. For sequences with ``L_r >= d_conv`` the new state is exactly the
    trailing ``d_conv`` tokens of ``x`` for that slot.

    Conventions
    -----------
    EasyDeL's state convention stores the *full* ``d_conv``-wide window (channels
    first): after processing token ``N``, ``conv_state[:, :, d_conv - 1]`` holds
    token ``N`` and ``conv_state[:, :, 0]`` holds token ``N - d_conv + 1``. The
    next conv step operates on a shifted version of this window. Note that
    position ``0`` is never read while computing outputs for a subsequent
    request (the oldest entry falls outside the ``d_conv``-wide window) but it
    is carried through so sub-``d_conv`` request lengths keep continuity.

    Low-precision inputs (fp8 / fp4) are promoted to float32 for the
    accumulation; the final cast returns to ``x.dtype``.

    Args:
        x: Packed input stream, shape ``(num_tokens, conv_dim)``.
        conv_state: Per-slot rolling state,
            shape ``(num_slots, conv_dim, d_conv)``. Position ``d_conv - 1`` is
            the most recent historical token, position ``0`` is ``d_conv``
            tokens back.
        kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``. Same layout as
            Qwen3-Next / GatedDeltaNet causal depthwise-conv convention.
        query_start_loc: Cumulative token offsets per request,
            shape ``(num_slots + 1,)``. ``query_start_loc[-1]`` must equal the
            number of valid tokens; any trailing "inactive" slots can be
            encoded by setting their length to 0 and/or using ``distribution``.
        state_indices: Request-to-state-slot mapping, shape ``(num_slots,)``.
            Used to gather the incoming state for each request and to scatter
            the updated state back into the pool.
        distribution: ``(decode_end, prefill_end, mixed_end)`` tensor of
            shape ``(3,)`` int32. Only ``distribution[2]`` is consumed here
            (number of valid sequences); trailing slots beyond that index keep
            their existing state unchanged and contribute no output updates.
        d_conv: Convolution kernel / state window size. Must match
            ``kernel.shape[-1]`` and ``conv_state.shape[-1]``.
        apply_silu: If True (default), applies ``jax.nn.silu`` after the
            accumulation, matching Qwen3-Next / GatedDeltaNet conventions.
            Pass False to get the raw linear convolution output.

    Returns:
        A tuple ``(output, updated_conv_state)``:

        - ``output``: Per-token conv output, shape ``(num_tokens, conv_dim)``,
          dtype matches ``x``.
        - ``updated_conv_state``: Conv-state pool with the slots indexed by
          ``state_indices`` refreshed, shape and dtype match ``conv_state``.
          The input buffer is donated (``donate_argnames=("conv_state",)``)
          to avoid an XLA copy.

    Notes:
        * The function is JIT-compiled with ``d_conv`` and ``apply_silu``
          marked static, so passing different values for these triggers
          recompilation.
        * The ``for k in range(d_conv)`` loop is Python-level and unrolled at
          trace time; pick ``d_conv`` such that the unroll is reasonable
          (typically 4, the Qwen3-Next / GDR default).
    """
    num_tokens, dim = x.shape
    max_reqs = state_indices.shape[0]

    num_valid_seqs = distribution[2]
    effective_query_start_loc = _fix_query_start_loc(query_start_loc, num_valid_seqs)
    lengths = effective_query_start_loc[1:] - effective_query_start_loc[:-1]

    gathered_state = conv_state[state_indices]

    out = _depthwise_conv1d_flat(x, kernel)

    starts = effective_query_start_loc[:-1]
    gather_indices, scatter_indices = _get_boundary_indices(starts, lengths, d_conv, num_valid_seqs)
    x_first = x[gather_indices]
    history = gathered_state[:, :, 1:].transpose(0, 2, 1)
    combined_tokens = jnp.concatenate([history, x_first], axis=1)
    boundary_out = jax.lax.conv_general_dilated(
        combined_tokens.astype(jnp.float32),
        kernel[:, None, :].astype(jnp.float32),
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "OIW", "NWC"),
        feature_group_count=dim,
        precision=jax.lax.Precision.HIGHEST,
    ).reshape(-1, dim)
    out = out.at[scatter_indices.flatten()].set(
        boundary_out.astype(out.dtype),
        mode="drop",
        wrap_negative_indices=False,
    )
    total_valid_tokens = effective_query_start_loc[num_valid_seqs]
    valid_token_mask = jnp.arange(num_tokens) < total_valid_tokens
    out = jnp.where(valid_token_mask[:, None], out, 0.0)

    if apply_silu:
        out = jax.nn.silu(out)

    padded_lengths = jnp.zeros(max_reqs, dtype=jnp.int32).at[: lengths.shape[0]].set(lengths)
    padded_q_end = jnp.zeros(max_reqs, dtype=jnp.int32).at[: lengths.shape[0]].set(effective_query_start_loc[1:])

    r_grid = jnp.arange(max_reqs)[:, None]
    j_grid = jnp.arange(d_conv)[None, :]
    is_from_old_state = (padded_lengths[:, None] + j_grid) < d_conv

    idx_state_new = jnp.where(is_from_old_state, padded_lengths[:, None] + j_grid, 0)
    idx_x_new = jnp.clip(padded_q_end[:, None] - d_conv + j_grid, 0, num_tokens - 1)

    new_state_hist = gathered_state[r_grid, :, idx_state_new]
    new_state_from_x = x[idx_x_new.reshape(-1)].reshape(max_reqs, d_conv, dim)

    new_state_jchw = jnp.where(
        is_from_old_state[..., None],
        new_state_hist,
        new_state_from_x.astype(new_state_hist.dtype),
    )
    new_state = new_state_jchw.transpose(0, 2, 1).astype(conv_state.dtype)

    true_valid_seq_mask = jnp.arange(max_reqs) < num_valid_seqs
    updated_conv_state = conv_state.at[state_indices].set(
        jnp.where(
            true_valid_seq_mask[:, None, None],
            new_state,
            conv_state[state_indices],
        )
    )

    return out.astype(x.dtype), updated_conv_state


@jax.jit(
    donate_argnames=("conv_state",),
    static_argnames=("d_conv", "apply_silu"),
)
@jax.named_scope("ragged_causal_conv1d_jax")
def ragged_causal_conv1d(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    *,
    d_conv: int,
    apply_silu: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled entry point for the ragged causal depthwise conv1d.

    Thin wrapper that forwards to :func:`_ragged_causal_conv1d_impl` after
    being jit-compiled with ``d_conv`` and ``apply_silu`` marked static and
    ``conv_state`` donated.

    Args:
        x: Packed input stream, shape ``(num_tokens, conv_dim)``.
        conv_state: Per-slot rolling state, shape
            ``(num_slots, conv_dim, d_conv)``. Donated.
        kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``.
        query_start_loc: Cumulative per-request token offsets, shape
            ``(num_slots + 1,)``.
        state_indices: Request-to-slot mapping, shape ``(num_slots,)``.
        distribution: ``(decode_end, prefill_end, mixed_end)`` int32
            triple of shape ``(3,)``.
        d_conv: Convolution window size (static).
        apply_silu: Whether to fuse SiLU into the output (static).

    Returns:
        tuple: ``(output, updated_conv_state)`` matching the return
        contract of :func:`_ragged_causal_conv1d_impl`.
    """
    return _ragged_causal_conv1d_impl(
        x=x,
        conv_state=conv_state,
        kernel=kernel,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        d_conv=d_conv,
        apply_silu=apply_silu,
    )
