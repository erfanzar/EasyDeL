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

"""Recurrent attention interface for linear-time sequence processing.

This module provides the public API for recurrent linear attention with
O(N) complexity. Supports various gating mechanisms (GLA, Lightning)
and provides custom VJP for efficient gradient computation.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.interpreters import ad
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _recurrent_attention_bwd
from ._xla_impl_fwd import _recurrent_attention_fwd


def _pack_varlen_inputs(
    query: Array,
    key: Array,
    value: Array,
    g: Array | None,
    gk: Array | None,
    gv: Array | None,
) -> tuple[Array, Array, Array, Array | None, Array | None, Array | None, bool]:
    """Normalise packed varlen inputs to 3-D token-major layout.

    Callers may pass rank-4 tensors ``[1, total_tokens, H, D]`` (batch=1
    wrapping a packed token stream) or rank-3 tensors ``[total_tokens, H, D]``.
    This function strips the leading batch dimension when present and returns
    a flag indicating whether it was added.

    Args:
        query: Query tensor, rank 3 or 4 with batch size 1 if rank 4.
        key: Key tensor, same rank as ``query``.
        value: Value tensor, same rank as ``query``.
        g: Optional GLA gate, same rank as ``query`` (or ``None``).
        gk: Optional key gate, same rank as ``query`` (or ``None``).
        gv: Optional value gate, same rank as ``value`` (or ``None``).

    Returns:
        A 7-tuple ``(query_tok, key_tok, value_tok, g_tok, gk_tok, gv_tok,
        add_batch_dim)`` where the tensors are 3-D token-major and
        ``add_batch_dim`` is ``True`` if a leading batch dim was removed
        (the caller must re-add it to the output).

    Raises:
        ValueError: If ranks mismatch, or if batch size > 1 is encountered
            for rank-4 inputs.
    """
    if query.ndim != key.ndim or query.ndim != value.ndim:
        raise ValueError(f"query/key/value must share rank in varlen mode, got {query.ndim}, {key.ndim}, {value.ndim}.")

    if query.ndim == 4:
        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError(
                "Packed varlen recurrent expects batch size 1 when using rank-4 inputs; "
                f"got query/key/value batch sizes {query.shape[0]}/{key.shape[0]}/{value.shape[0]}."
            )
        if g is not None and g.shape[0] != 1:
            raise ValueError(f"g must have batch size 1 in varlen packed mode, got {g.shape[0]}.")
        if gk is not None and gk.shape[0] != 1:
            raise ValueError(f"gk must have batch size 1 in varlen packed mode, got {gk.shape[0]}.")
        if gv is not None and gv.shape[0] != 1:
            raise ValueError(f"gv must have batch size 1 in varlen packed mode, got {gv.shape[0]}.")
        return (
            query[0],
            key[0],
            value[0],
            None if g is None else g[0],
            None if gk is None else gk[0],
            None if gv is None else gv[0],
            True,
        )

    if query.ndim == 3:
        return query, key, value, g, gk, gv, False

    raise ValueError(f"Varlen recurrent expects rank-3 or rank-4 inputs, got rank {query.ndim}.")


def _normalize_varlen_initial_state(
    initial_state: Array | None,
    *,
    num_seqs: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[Array, bool]:
    """Build or validate the per-sequence initial hidden state.

    Accepts ``None`` (allocate zeros), a 4-D ``[num_seqs, H, K, V]`` array,
    or a 5-D ``[1, num_seqs, H, K, V]`` array (strips the leading batch dim).

    Args:
        initial_state: Optional initial hidden state.  ``None`` creates a
            zero-initialised state.
        num_seqs: Number of sequences; must match ``initial_state.shape[0]``.
        num_heads: Number of attention heads.
        key_dim: Key/query head dimension.
        value_dim: Value head dimension.

    Returns:
        Tuple ``(state, state_was_none)`` where ``state`` has shape
        ``[num_seqs, H, key_dim, value_dim]`` and ``state_was_none`` is
        ``True`` if the input was ``None``.

    Raises:
        ValueError: On shape mismatches.
    """
    if initial_state is None:
        return jnp.zeros((num_seqs, num_heads, key_dim, value_dim), dtype=jnp.float32), True

    state = jnp.asarray(initial_state)
    if state.ndim == 5:
        if state.shape[0] != 1:
            raise ValueError(
                f"Varlen recurrent initial_state rank-5 form must have leading batch size 1, got {state.shape[0]}."
            )
        state = state[0]
    if state.ndim != 4:
        raise ValueError(f"Varlen recurrent initial_state must be rank-4 [N,H,K,V], got shape {state.shape}.")
    if state.shape[0] != num_seqs:
        raise ValueError(f"Varlen recurrent initial_state must have N={num_seqs}, got shape {state.shape}.")
    return state, False


def _varlen_sequence_ids(
    cu_seqlens: Int[Array, "num_seqs_plus_one"],
    total_tokens: int,
) -> Int[Array, "total_tokens"]:
    """Map each token position to its sequence index.

    Args:
        cu_seqlens: Cumulative sequence lengths, shape ``[num_seqs + 1]``.
        total_tokens: Total number of tokens in the packed stream.

    Returns:
        Integer array of shape ``[total_tokens]`` where element ``i`` is the
        0-based index of the sequence that token ``i`` belongs to.
    """
    if cu_seqlens.shape[0] <= 1:
        return jnp.zeros((total_tokens,), dtype=jnp.int32)
    token_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    boundaries = jnp.asarray(cu_seqlens[1:-1], dtype=jnp.int32)
    return jnp.sum(token_idx[:, None] >= boundaries[None, :], axis=1).astype(jnp.int32)


def _varlen_reverse_permutation(
    cu_seqlens: Int[Array, "num_seqs_plus_one"],
    total_tokens: int,
) -> Int[Array, "total_tokens"]:
    """Build a permutation that reverses each sequence independently.

    For each sequence ``[start, end)``, token at position ``i`` is mapped to
    ``start + end - 1 - i``.  Tokens from different sequences are not
    interleaved.

    Args:
        cu_seqlens: Cumulative sequence lengths, shape ``[num_seqs + 1]``.
        total_tokens: Total number of tokens.

    Returns:
        Integer permutation array of shape ``[total_tokens]``.
    """
    idx = jnp.arange(total_tokens, dtype=jnp.int32)
    num_seqs = int(cu_seqlens.shape[0] - 1)

    def _body(seq_idx: int, perm: Int[Array, "total_tokens"]) -> Int[Array, "total_tokens"]:
        start = jnp.asarray(cu_seqlens[seq_idx], dtype=jnp.int32)
        end = jnp.asarray(cu_seqlens[seq_idx + 1], dtype=jnp.int32)
        in_seq = (idx >= start) & (idx < end)
        rev_idx = start + end - 1 - idx
        return jnp.where(in_seq, rev_idx, perm)

    return jax.lax.fori_loop(0, num_seqs, _body, idx)


def _normalize_g_gamma_per_seq(
    g_gamma: Array | None,
    *,
    num_seqs: int,
    num_heads: int,
) -> Float[Array, "num_seqs num_heads"]:
    """Broadcast ``g_gamma`` to shape ``[num_seqs, num_heads]``.

    Accepted input shapes:

    * ``None``: returns zeros ``[num_seqs, num_heads]``.
    * 1-D ``[num_heads]``: broadcast over sequences.
    * 2-D ``[1, num_heads]``: broadcast over sequences.
    * 2-D ``[num_seqs, num_heads]``: returned as-is.

    Args:
        g_gamma: Per-head decay factor (or ``None``).
        num_seqs: Number of sequences.
        num_heads: Number of attention heads.

    Returns:
        Float array of shape ``[num_seqs, num_heads]``.

    Raises:
        ValueError: On incompatible shapes.
    """
    if g_gamma is None:
        return jnp.zeros((num_seqs, num_heads), dtype=jnp.float32)

    g_gamma = jnp.asarray(g_gamma, dtype=jnp.float32)
    if g_gamma.ndim == 1:
        if g_gamma.shape[0] != num_heads:
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({num_seqs}, {num_heads})")
        return jnp.broadcast_to(g_gamma[None, :], (num_seqs, num_heads))
    if g_gamma.ndim == 2:
        if g_gamma.shape[1] != num_heads:
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({num_seqs}, {num_heads})")
        if g_gamma.shape[0] == 1:
            return jnp.broadcast_to(g_gamma, (num_seqs, num_heads))
        if g_gamma.shape[0] == num_seqs:
            return g_gamma
        raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({num_seqs}, {num_heads})")
    raise ValueError(f"g_gamma.ndim={g_gamma.ndim} must be 1 or 2")


def _recurrent_varlen_forward(
    query: Array,
    key: Array,
    value: Array,
    cu_seqlens: Int[Array, "num_seqs_plus_one"],
    g: Array | None,
    g_gamma: Array | None,
    gk: Array | None,
    gv: Array | None,
    softmax_scale: float | None,
    initial_state: Array | None,
    reverse: bool,
) -> tuple[Array, Array, Array, bool]:
    """Run the varlen recurrent forward pass.

    Adapts public packed-input tensors to the internal token-major layout
    (via ``_pack_varlen_inputs``), builds per-sequence initial states, and
    runs a single ``lax.scan`` over all tokens in the packed stream.
    Sequence state updates are scattered back via indexed assignment at
    ``h_all[seq_t]``.

    Args:
        query: Query tensor (rank 3 or 4).
        key: Key tensor (same rank as ``query``).
        value: Value tensor (same rank as ``query``).
        cu_seqlens: Cumulative sequence lengths, shape ``[num_seqs + 1]``.
        g: Optional GLA gate (same shape as ``query``).
        g_gamma: Optional per-head decay.
        gk: Optional key gate.
        gv: Optional value gate.
        softmax_scale: Query scaling factor.
        initial_state: Optional per-sequence initial hidden state.
        reverse: If ``True``, process each sequence in reverse token order.

    Returns:
        A 4-tuple ``(output, final_state, initial_state, state_was_none)``
        where ``output`` and ``final_state`` match the layout expected by
        the public ``recurrent`` API.
    """
    query_tok, key_tok, value_tok, g_tok, gk_tok, gv_tok, add_batch_dim = _pack_varlen_inputs(
        query, key, value, g, gk, gv
    )
    if g_tok is None:
        g_tok = jnp.zeros_like(query_tok)
    if gk_tok is None:
        gk_tok = jnp.zeros_like(query_tok)
    if gv_tok is None:
        gv_tok = jnp.zeros_like(value_tok)
    num_seqs = int(cu_seqlens.shape[0] - 1)
    state_tok, state_was_none = _normalize_varlen_initial_state(
        initial_state,
        num_seqs=num_seqs,
        num_heads=query_tok.shape[1],
        key_dim=query_tok.shape[2],
        value_dim=value_tok.shape[2],
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(query_tok.shape[-1]).astype(jnp.float32)
    else:
        softmax_scale = jnp.asarray(softmax_scale, dtype=jnp.float32)

    g_gamma_by_seq = _normalize_g_gamma_per_seq(g_gamma, num_seqs=num_seqs, num_heads=query_tok.shape[1])

    total_tokens = int(query_tok.shape[0])
    seq_ids = _varlen_sequence_ids(cu_seqlens, total_tokens)
    if reverse:
        perm = _varlen_reverse_permutation(cu_seqlens, total_tokens)
    else:
        perm = jnp.arange(total_tokens, dtype=jnp.int32)

    q_proc = query_tok[perm].astype(jnp.float32)
    k_proc = key_tok[perm].astype(jnp.float32)
    v_proc = value_tok[perm].astype(jnp.float32)
    g_proc = g_tok[perm].astype(jnp.float32)
    gk_proc = gk_tok[perm].astype(jnp.float32)
    gv_proc = gv_tok[perm].astype(jnp.float32)
    seq_proc = seq_ids[perm]

    def _step(
        h_all: Float[Array, "num_seqs num_heads qk_head_dim v_head_dim"],
        xs: tuple[
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
            Int[Array, ""],
        ],
    ) -> tuple[Float[Array, "num_seqs num_heads qk_head_dim v_head_dim"], Float[Array, "num_heads v_head_dim"]]:
        q_t, k_t, v_t, g_t, gk_t, gv_t, seq_t = xs
        h_prev = h_all[seq_t]
        h_cur = h_prev * jnp.exp(g_t)[:, :, None]
        h_cur = h_cur * jnp.exp(g_gamma_by_seq[seq_t])[:, None, None]
        h_cur = h_cur * jnp.exp(gk_t)[:, :, None]
        h_cur = h_cur * jnp.exp(gv_t)[:, None, :]
        h_cur = h_cur + k_t[:, :, None] * v_t[:, None, :]
        out_t = jnp.sum(h_cur * (q_t * softmax_scale)[:, :, None], axis=1)
        h_all = h_all.at[seq_t].set(h_cur)
        return h_all, out_t

    final_state, out_proc = jax.lax.scan(
        _step,
        state_tok.astype(jnp.float32),
        (q_proc, k_proc, v_proc, g_proc, gk_proc, gv_proc, seq_proc),
    )
    out_tok = jnp.zeros_like(value_tok, dtype=out_proc.dtype).at[perm].set(out_proc)
    out = out_tok[None, ...] if add_batch_dim else out_tok
    return out, final_state, state_tok, state_was_none


@partial(jax.custom_vjp, nondiff_argnums=(4, 7, 9, 10))
def _recurrent_core(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "... num_heads head_dim head_dim"]]:
    """Core recurrent attention computation with custom VJP for gradients.

    This is the main computational entry point that dispatches to either
    variable-length or fixed-length implementations based on cu_seqlens.
    Custom VJP enables efficient gradient computation for training.

    Args:
        query, key, value: Standard attention tensors [batch, seq_len, heads, dim]
        g: GLA-style gate tensor for gated linear attention
        g_gamma: Per-head decay factor for Lightning attention (nondiff)
        gk: Gate applied to keys
        gv: Gate applied to values
        softmax_scale: Query scaling factor (nondiff)
        initial_state: Initial hidden state for continuation
        reverse: Process sequence in reverse order (nondiff)
        cu_seqlens: Cumulative sequence lengths for packed mode (nondiff)

    Returns:
        Tuple of (output, final_state)
    """
    if cu_seqlens is not None:
        out, final_state, _, _ = _recurrent_varlen_forward(
            query, key, value, cu_seqlens, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse
        )
        return out, final_state
    else:
        output, _, final_state = _recurrent_attention_fwd(
            query, key, value, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse
        )
        return output, final_state


def _recurrent_fwd(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch num_heads head_dim head_dim"]],
    tuple,
]:
    """Forward rule for recurrent attention VJP.

    Computes the recurrent attention output and saves residuals needed
    for the backward pass. Handles default initialization for optional
    parameters.

    Args:
        query, key, value: Input tensors for attention
        g, g_gamma, gk, gv: Optional gating parameters
        softmax_scale: Query scaling (defaults to 1/sqrt(head_dim))
        initial_state: Starting hidden state (defaults to zeros)
        reverse: Process sequence backwards
        cu_seqlens: For variable-length mode

    Returns:
        Tuple of ((output, final_state), residuals)
    """

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(query.shape[-1]).astype(jnp.float32)
    if g is None:
        g = jnp.zeros_like(query)
    if g_gamma is None:
        g_gamma = jnp.zeros((query.shape[-2],))
    if gk is None:
        gk = jnp.zeros_like(query)
    if gv is None:
        gv = jnp.zeros_like(query)
    state_was_none = initial_state is None

    if cu_seqlens is not None:
        o, final_state, initial_state, state_was_none = _recurrent_varlen_forward(
            query, key, value, cu_seqlens, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse
        )
        hidden_states = None
    else:
        if initial_state is None:
            initial_state = jnp.zeros((query.shape[0], query.shape[2], query.shape[3], value.shape[3]))
        o, hidden_states, final_state = _recurrent_attention_fwd(
            query, key, value, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse
        )

    residuals = (query, key, value, g, g_gamma, gk, gv, hidden_states, initial_state, state_was_none)
    return (o, final_state), residuals


def _recurrent_bwd(
    g_gamma_nondiff: Float[Array, "... num_heads"],
    scale_nondiff: float | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    residuals: tuple,
    grads: tuple,
) -> tuple:
    """Backward rule for recurrent attention VJP.

    Computes gradients with respect to query, key, value, gates, and
    initial state using the custom backward implementation.

    Args:
        g_gamma_nondiff: Per-head decay (nondiff, from closure)
        scale_nondiff: Softmax scale (nondiff, from closure)
        reverse: Whether forward was reversed (nondiff)
        cu_seqlens: Sequence lengths (nondiff)
        residuals: Saved tensors from forward pass
        grads: (do, dfinal_state) gradients from downstream

    Returns:
        Tuple of gradients (dq, dk, dv, dg, dgk, dgv, dinitial_state)
    """
    query, key, value, g, g_gamma, gk, gv, hidden_states, initial_state, state_was_none = residuals
    do, dfinal_state = grads

    do = ad.instantiate_zeros(do)
    dfinal_state = ad.instantiate_zeros(dfinal_state)
    if do is None:
        do = jnp.zeros_like(value)
    if dfinal_state is None:
        dfinal_state = jnp.zeros_like(initial_state)

    if scale_nondiff is None:
        softmax_scale = 1.0 / jnp.sqrt(query.shape[-1]).astype(jnp.float32)
    else:
        softmax_scale = scale_nondiff

    if cu_seqlens is None:
        dq, dk, dv, dg, dgk, dgv, dinitial_state = _recurrent_attention_bwd(
            query,
            key,
            value,
            g,
            g_gamma,
            gk,
            gv,
            hidden_states,
            do,
            dfinal_state,
            softmax_scale,
            initial_state,
            reverse,
        )
    else:
        del g_gamma_nondiff

        query_tok, key_tok, value_tok, g_tok, gk_tok, gv_tok, add_batch_dim = _pack_varlen_inputs(
            query, key, value, g, gk, gv
        )
        do_tok = do[0] if add_batch_dim else do

        num_seqs = int(cu_seqlens.shape[0] - 1)
        g_gamma_by_seq = _normalize_g_gamma_per_seq(g_gamma, num_seqs=num_seqs, num_heads=query_tok.shape[1])
        total_tokens = int(query_tok.shape[0])
        seq_ids = _varlen_sequence_ids(cu_seqlens, total_tokens)
        if reverse:
            perm = _varlen_reverse_permutation(cu_seqlens, total_tokens)
        else:
            perm = jnp.arange(total_tokens, dtype=jnp.int32)

        q_proc = query_tok[perm].astype(jnp.float32)
        k_proc = key_tok[perm].astype(jnp.float32)
        v_proc = value_tok[perm].astype(jnp.float32)
        g_proc = g_tok[perm].astype(jnp.float32)
        gk_proc = gk_tok[perm].astype(jnp.float32)
        gv_proc = gv_tok[perm].astype(jnp.float32)
        do_proc = do_tok[perm].astype(jnp.float32)
        seq_proc = seq_ids[perm]
        h0_all = initial_state.astype(jnp.float32)
        dht_all = dfinal_state.astype(jnp.float32)

        def _fwd_hist_step(
            h_all: Float[Array, "num_seqs num_heads qk_head_dim v_head_dim"],
            xs: tuple[
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads v_head_dim"],
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads v_head_dim"],
                Int[Array, ""],
            ],
        ):
            q_t, k_t, v_t, g_t, gk_t, gv_t, seq_t = xs
            del q_t
            h_prev = h_all[seq_t]
            h1 = h_prev * jnp.exp(g_t)[:, :, None]
            h2 = h1 * jnp.exp(g_gamma_by_seq[seq_t])[:, None, None]
            h3 = h2 * jnp.exp(gk_t)[:, :, None]
            h4 = h3 * jnp.exp(gv_t)[:, None, :]
            h_out = h4 + k_t[:, :, None] * v_t[:, None, :]
            h_all = h_all.at[seq_t].set(h_out)
            return h_all, (h_prev, h_out)

        _, (h_prev_hist, h_out_hist) = jax.lax.scan(
            _fwd_hist_step,
            h0_all,
            (q_proc, k_proc, v_proc, g_proc, gk_proc, gv_proc, seq_proc),
        )

        def _bwd_step(
            dh_all: Float[Array, "num_seqs num_heads qk_head_dim v_head_dim"],
            xs: tuple[
                Float[Array, "num_heads qk_head_dim v_head_dim"],
                Float[Array, "num_heads qk_head_dim v_head_dim"],
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads v_head_dim"],
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads qk_head_dim"],
                Float[Array, "num_heads v_head_dim"],
                Float[Array, "num_heads v_head_dim"],
                Int[Array, ""],
            ],
        ):
            h_prev, h_out, q_t, k_t, v_t, g_t, gk_t, gv_t, do_t, seq_t = xs
            q_scaled = q_t * softmax_scale

            dh_next = dh_all[seq_t]
            dq_t = jnp.sum(h_out * do_t[:, None, :], axis=-1) * softmax_scale
            dh_t = dh_next + q_scaled[:, :, None] * do_t[:, None, :]

            dk_t = jnp.sum(dh_t * v_t[:, None, :], axis=-1)
            dv_t = jnp.sum(dh_t * k_t[:, :, None], axis=-2)

            h1 = h_prev * jnp.exp(g_t)[:, :, None]
            h2 = h1 * jnp.exp(g_gamma_by_seq[seq_t])[:, None, None]
            h3 = h2 * jnp.exp(gk_t)[:, :, None]
            h4 = h3 * jnp.exp(gv_t)[:, None, :]

            dgv_t = jnp.sum(dh_t * h4, axis=1)
            dh3 = dh_t * jnp.exp(gv_t)[:, None, :]

            dgk_t = jnp.sum(dh3 * h3, axis=-1)
            dh2 = dh3 * jnp.exp(gk_t)[:, :, None]

            dh1 = dh2 * jnp.exp(g_gamma_by_seq[seq_t])[:, None, None]
            dg_t = jnp.sum(dh1 * h1, axis=-1)
            dh_prev = dh1 * jnp.exp(g_t)[:, :, None]

            dh_all = dh_all.at[seq_t].set(dh_prev)
            return dh_all, (dq_t, dk_t, dv_t, dg_t, dgk_t, dgv_t)

        rev_inputs = (
            jnp.flip(h_prev_hist, axis=0),
            jnp.flip(h_out_hist, axis=0),
            jnp.flip(q_proc, axis=0),
            jnp.flip(k_proc, axis=0),
            jnp.flip(v_proc, axis=0),
            jnp.flip(g_proc, axis=0),
            jnp.flip(gk_proc, axis=0),
            jnp.flip(gv_proc, axis=0),
            jnp.flip(do_proc, axis=0),
            jnp.flip(seq_proc, axis=0),
        )
        dinitial_state, grads_rev = jax.lax.scan(_bwd_step, dht_all, rev_inputs)
        dq_proc_rev, dk_proc_rev, dv_proc_rev, dg_proc_rev, dgk_proc_rev, dgv_proc_rev = grads_rev

        dq_proc = jnp.flip(dq_proc_rev, axis=0)
        dk_proc = jnp.flip(dk_proc_rev, axis=0)
        dv_proc = jnp.flip(dv_proc_rev, axis=0)
        dg_proc = jnp.flip(dg_proc_rev, axis=0)
        dgk_proc = jnp.flip(dgk_proc_rev, axis=0)
        dgv_proc = jnp.flip(dgv_proc_rev, axis=0)

        dq_tok = jnp.zeros_like(query_tok, dtype=jnp.float32).at[perm].set(dq_proc)
        dk_tok = jnp.zeros_like(key_tok, dtype=jnp.float32).at[perm].set(dk_proc)
        dv_tok = jnp.zeros_like(value_tok, dtype=jnp.float32).at[perm].set(dv_proc)
        dg_tok = jnp.zeros_like(g_tok, dtype=jnp.float32).at[perm].set(dg_proc)
        dgk_tok = jnp.zeros_like(gk_tok, dtype=jnp.float32).at[perm].set(dgk_proc)
        dgv_tok = jnp.zeros_like(gv_tok, dtype=jnp.float32).at[perm].set(dgv_proc)

        if add_batch_dim:
            dq = dq_tok[None, ...]
            dk = dk_tok[None, ...]
            dv = dv_tok[None, ...]
            dg = dg_tok[None, ...]
            dgk = dgk_tok[None, ...]
            dgv = dgv_tok[None, ...]
        else:
            dq, dk, dv, dg, dgk, dgv = dq_tok, dk_tok, dv_tok, dg_tok, dgk_tok, dgv_tok

    dq = dq.astype(query.dtype)
    dk = dk.astype(key.dtype)
    dv = dv.astype(value.dtype)
    dg = dg.astype(g.dtype)
    dgk = dgk.astype(gk.dtype)
    dgv = dgv.astype(gv.dtype)

    if state_was_none:
        dinitial_state = None
    else:
        dinitial_state = dinitial_state.astype(initial_state.dtype)

    return (dq, dk, dv, dg, dgk, dgv, dinitial_state)


_recurrent_core.defvjp(_recurrent_fwd, _recurrent_bwd)


@kernel_registry.register("recurrent", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Recurrent linear attention with O(N) complexity using JAX/XLA.

    Implements linear attention as a recurrent process that maintains a hidden
    state ``h`` of shape ``[num_heads, qk_head_dim, v_head_dim]`` per sequence.
    Each step applies optional gated decay before accumulating a new key-value
    outer product:

    .. code-block:: text

        h_t = decay_t * h_{t-1} + k_t[:, :, None] * v_t[:, None, :]
        o_t = sum(h_t * (q_t * scale)[:, :, None], axis=1)

    The decay is the product of any active gating exponentials:
    ``exp(g) * exp(g_gamma) * exp(gk) (applied per key-dim) * exp(gv)
    (applied per value-dim)``.

    Registered under ``"recurrent"`` for ``Platform.XLA``, ``Backend.ANY``.
    Uses a custom VJP (``_recurrent_core``) for gradient computation.

    Normal (padded-batch) mode:
        Inputs have shape ``[batch, seq_len, num_heads, head_dim]``.
        Each batch element is processed independently via ``jax.vmap``.

    Varlen (packed-stream) mode:
        When ``cu_seqlens`` is provided, inputs should have shape
        ``[total_tokens, num_heads, head_dim]`` (rank 3) or
        ``[1, total_tokens, num_heads, head_dim]`` (rank 4, batch=1).
        A single ``lax.scan`` processes all tokens; ``final_state`` has shape
        ``[num_seqs, num_heads, qk_head_dim, v_head_dim]``.

    Args:
        query: Shape ``[batch, seq_len, num_heads, qk_head_dim]``.
        key: Shape ``[batch, seq_len, num_kv_heads, qk_head_dim]``.
            GQA is not supported: ``num_kv_heads`` must equal ``num_heads``.
        value: Shape ``[batch, seq_len, num_kv_heads, v_head_dim]``.
        g: Optional GLA-style element-wise gate applied to the full state,
            shape ``[batch, seq_len, num_heads, qk_head_dim]``.
        g_gamma: Optional scalar per-head decay for Lightning-attention style.
            Shape ``[num_heads]`` or ``[batch, num_heads]`` or
            ``[1, num_heads]``.
        gk: Optional gate applied per key dimension (left index of state),
            shape ``[batch, seq_len, num_heads, qk_head_dim]``.
        gv: Optional gate applied per value dimension (right index of state),
            shape ``[batch, seq_len, num_heads, v_head_dim]``.
        softmax_scale: Multiplicative scale for query vectors.
            Defaults to ``1 / sqrt(qk_head_dim)``.
        initial_state: Optional starting hidden state.
            Shape ``[batch, num_heads, qk_head_dim, v_head_dim]``.
            Defaults to zeros.
        reverse: If ``True``, process the sequence in reverse temporal order.
            Useful for bidirectional models.
        cu_seqlens: Cumulative sequence lengths for packed-stream mode,
            shape ``[num_seqs + 1]``.  When ``None``, uses padded-batch mode.
        block_k: Accepted for API compatibility with Triton; ignored by XLA.
        block_v: Accepted for API compatibility with Triton; ignored by XLA.
        num_warps: Accepted for API compatibility with Triton; ignored by XLA.
        num_stages: Accepted for API compatibility with Triton; ignored by XLA.

    Returns:
        A tuple ``(output, final_state)`` where:

        * ``output`` has the same shape as ``query`` (batch or packed-stream).
        * ``final_state`` has shape ``[batch, num_heads, qk_head_dim,
          v_head_dim]`` (batch mode) or ``[num_seqs, num_heads, qk_head_dim,
          v_head_dim]`` (varlen mode).

    Note:
        In padded-batch mode all hidden states at every timestep are
        materialised in memory during the forward pass for use by the custom
        backward rule.  For long sequences this can be memory-intensive.

    Example:
        >>> query = jnp.ones((2, 100, 8, 64))
        >>> key = jnp.ones((2, 100, 8, 64))
        >>> value = jnp.ones((2, 100, 8, 64))
        >>> output, final_state = recurrent(query, key, value)
        >>> output.shape
        (2, 100, 8, 64)
        >>> final_state.shape
        (2, 8, 64, 64)

        >>> # Varlen packed-stream mode
        >>> query = jnp.ones((1, 150, 8, 64))
        >>> cu_seqlens = jnp.array([0, 50, 100, 150])
        >>> output, final_state = recurrent(query, query, query,
        ...                                 cu_seqlens=cu_seqlens)
        >>> final_state.shape
        (3, 8, 64, 64)
    """
    return _recurrent_core(query, key, value, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse, cu_seqlens)
