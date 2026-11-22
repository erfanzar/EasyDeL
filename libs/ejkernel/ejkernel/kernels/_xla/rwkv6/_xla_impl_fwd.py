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

"""RWKV-6 recurrent kernel (XLA).

This module provides a pure JAX/XLA implementation of the RWKV-6 time-mix
recurrence. RWKV-6 extends RWKV-4/5 with multi-head attention and a more
expressive state transition that uses per-timestep decay and a bonus term.

RWKV-6 introduces several architectural improvements over RWKV-4:
1. Multi-head structure: State is per-head [H, K, V] instead of per-channel
2. Per-timestep decay: w is [B, T, H, K] instead of global [C]
3. Bonus term: u provides per-head key dimension bonuses
4. Receptance: r (query) is separate from k (key)

The core recurrence is:
    kv_t = k_t^T ⊗ v_t  (outer product, [H, K, V])
    o_t = r_t^T @ (h_{t-1} + kv_t * u)  (bonus-enhanced attention)
    h_t = h_{t-1} * exp(w_t) + kv_t  (exponential decay update)

This matches the semantics of Flash-Linear-Attention's RWKV-6 fused recurrent op:
    - inputs in `[B, T, H, ...]` (batch, time, head) format
    - optional packed variable-length mode via `cu_seqlens`
    - optional reverse recurrence for bidirectional processing
    - returns `(o, final_state)` where final_state is float32

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.rwkv6 import rwkv6
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 100, 8, 64
    >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))  # receptance
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))  # key
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))  # value
    >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))  # log decay
    >>> u = jnp.zeros((num_heads, head_dim))  # bonus
    >>>
    >>> output, final_state = rwkv6(r, k, v, w, u)
    >>> output.shape
    (2, 100, 8, 64)

Reference:
    RWKV-6: Linear Transformers with Enhanced Expressivity
    https://github.com/BlinkDL/RWKV-LM
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _rwkv6_update(
    h: Array,
    q_t: Array,
    k_t: Array,
    v_t: Array,
    w_t: Array,
    u: Array,
) -> tuple[Array, Array]:
    """Single step of the RWKV-6 recurrence with bonus-enhanced attention.

    Computes one timestep of the RWKV-6 mechanism with multi-head state
    and per-timestep exponential decay.

    Args:
        h: Hidden state [B, H, K, V] - key-value memory matrix per head
        q_t: Query/receptance at time t [B, H, K]
        k_t: Key at time t [B, H, K]
        v_t: Value at time t [B, H, V]
        w_t: Log decay at time t [B, H, K]
        u: Bonus term for current token [H, K]

    Returns:
        Tuple of:
            - h_next: Updated hidden state [B, H, K, V]
            - o_t: Output at time t [B, H, V]
    """
    kv = k_t[..., :, None] * v_t[..., None, :]
    o_t = jnp.einsum("bhk,bhkv->bhv", q_t, h + kv * u[None, :, :, None])
    h_next = h * jnp.exp(w_t)[..., :, None] + kv
    return h_next, o_t


def _rwkv6_scan(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    u: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    """Sequential scan for RWKV-6 over full sequence.

    Args:
        r: Receptance/query [B, T, H, K]
        k: Key [B, T, H, K]
        v: Value [B, T, H, V]
        w: Log decay [B, T, H, K]
        u: Bonus term [H, K]
        softmax_scale: Scale factor for receptance
        initial_state: Initial hidden state [B, H, K, V]
        reverse: If True, process sequence in reverse

    Returns:
        Tuple of (output [B, T, H, V], final_state [B, H, K, V])
    """
    if reverse:
        r = r[:, ::-1, :, :]
        k = k[:, ::-1, :, :]
        v = v[:, ::-1, :, :]
        w = w[:, ::-1, :, :]

    r = r * softmax_scale

    def step(h, xs):
        """Single RWKV-6 recurrence step for ``jax.lax.scan``.

        Unpacks the per-timestep inputs and delegates to ``_rwkv6_update``
        to advance the hidden state by one position.

        Args:
            h: Current hidden state [B, H, K, V].
            xs: Tuple of (q_t, k_t, v_t, w_t), each [B, H, *] for one timestep.

        Returns:
            Tuple of (h_next, o_t) where h_next is the updated state and
            o_t is the output [B, H, V] for this timestep.
        """
        q_t, k_t, v_t, w_t = xs
        h_next, o_t = _rwkv6_update(h, q_t, k_t, v_t, w_t, u)
        return h_next, o_t

    xs = (jnp.swapaxes(r, 0, 1), jnp.swapaxes(k, 0, 1), jnp.swapaxes(v, 0, 1), jnp.swapaxes(w, 0, 1))
    h_final, oT = jax.lax.scan(step, initial_state, xs)
    o = jnp.swapaxes(oT, 0, 1)
    if reverse:
        o = o[:, ::-1, :, :]
    return o, h_final


def _validate_rwkv6_inputs(r: Array, k: Array, v: Array, w: Array, u: Array) -> None:
    """Validate RWKV-6 input tensor shapes and dimensions.

    Args:
        r: Receptance tensor, expected [B, T, H, K]
        k: Key tensor, expected [B, T, H, K]
        v: Value tensor, expected [B, T, H, V]
        w: Log decay tensor, expected [B, T, H, K]
        u: Bonus tensor, expected [H, K]

    Raises:
        ValueError: If any tensor has incorrect rank or mismatched shapes.
    """
    if r.ndim != 4 or k.ndim != 4 or w.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected r,k,w rank-4 [B,T,H,K] and v rank-4 [B,T,H,V], got "
            f"r={r.shape}, k={k.shape}, v={v.shape}, w={w.shape}."
        )
    if r.shape != k.shape or r.shape != w.shape:
        raise ValueError(f"`r`, `k`, and `w` must have the same shape, got r={r.shape}, k={k.shape}, w={w.shape}.")
    if v.shape[:3] != r.shape[:3]:
        raise ValueError(f"`v` must match [B,T,H,*], got v={v.shape}, r={r.shape}.")
    if u.ndim != 2 or u.shape[0] != r.shape[2] or u.shape[1] != r.shape[3]:
        raise ValueError(f"`u` must have shape [H,K]={(r.shape[2], r.shape[3])}, got {u.shape}.")


def _rwkv6_varlen(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    u: Array,
    cu_seqlens: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    """RWKV-6 scan for variable-length packed sequences.

    Processes multiple sequences packed into a single tensor, using
    cumulative sequence lengths to determine boundaries.

    Args:
        r: Receptance [1, total_tokens, H, K]
        k: Key [1, total_tokens, H, K]
        v: Value [1, total_tokens, H, V]
        w: Log decay [1, total_tokens, H, K]
        u: Bonus [H, K]
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1]
        softmax_scale: Scale for receptance
        initial_state: Initial states [num_seqs, H, K, V]
        reverse: If True, process each sequence in reverse

    Returns:
        Tuple of (output [1, total_tokens, H, V], final_states [num_seqs, H, K, V])

    Raises:
        ValueError: If batch size is not 1 or initial_state count mismatches.
    """
    if r.shape[0] != 1:
        raise ValueError(f"Packed mode expects batch size 1, got {r.shape[0]}.")
    total_tokens = r.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    if initial_state.shape[0] != num_seqs:
        raise ValueError(f"`initial_state` must have shape [N,H,K,V] with N={num_seqs}, got {initial_state.shape}.")

    idx = jnp.arange(total_tokens, dtype=cu_seqlens.dtype)
    seq_id = jnp.searchsorted(cu_seqlens[1:], idx, side="right")
    starts = cu_seqlens[seq_id]
    ends = cu_seqlens[seq_id + 1] - 1
    is_start = idx == starts
    is_end = idx == ends

    if reverse:
        r = r[:, ::-1, :, :]
        k = k[:, ::-1, :, :]
        v = v[:, ::-1, :, :]
        w = w[:, ::-1, :, :]
        seq_id = seq_id[::-1]
        is_start, is_end = is_end[::-1], is_start[::-1]

    r = r * softmax_scale

    def step(carry, xs):
        """Single RWKV-6 step for variable-length packed-token scan.

        Handles sequence boundaries by conditionally resetting the hidden
        state at sequence starts and saving final states at sequence ends.

        Args:
            carry: Tuple of (h, finals) where h is the running hidden state
                [1, H, K, V] and finals accumulates per-sequence final states
                [num_seqs, H, K, V].
            xs: Tuple of (q_t, k_t, v_t, w_t, sid, start, end) where sid
                is the sequence index, start/end are boolean flags indicating
                sequence boundaries.

        Returns:
            Tuple of ((h_next, finals_updated), o_t) where o_t is the
            output [H, V] for this token.
        """
        h, finals = carry
        q_t, k_t, v_t, w_t, sid, start, end = xs

        h = jax.lax.cond(start, lambda _: initial_state[sid][None, ...], lambda _: h, operand=None)
        h, o_t = _rwkv6_update(h, q_t[None, ...], k_t[None, ...], v_t[None, ...], w_t[None, ...], u)
        finals = jax.lax.cond(end, lambda f: f.at[sid].set(h[0]), lambda f: f, finals)
        return (h, finals), o_t[0]

    xs = (
        r[0],
        k[0],
        v[0],
        w[0],
        seq_id.astype(jnp.int32),
        is_start,
        is_end,
    )
    h_init = jnp.zeros((1, *initial_state[0].shape), dtype=initial_state.dtype)
    (h_last, finals), o = jax.lax.scan(step, (h_init, initial_state), xs)
    del h_last
    o = o[None, ...]
    if reverse:
        o = o[:, ::-1, :, :]
    return o, finals


def rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-6 recurrent time-mix attention using XLA backend.

    Implements the RWKV-6 multi-head recurrent mechanism with O(N) complexity.
    Each head maintains a [K, V] state matrix that is updated via exponential
    decay and key-value outer products. The receptance (r) queries against
    this state with a bonus term (u) for the current token.

    The recurrence is:
        kv_t = k_t^T ⊗ v_t  (outer product)
        o_t = r_t^T @ (h_{t-1} + kv_t * u)  (query with bonus)
        h_t = h_{t-1} * exp(w_t) + kv_t  (decay and accumulate)

    Args:
        r: Receptance/query tensor for attention retrieval.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        k: Key tensor for memory addressing.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        v: Value tensor for memory content.
            Shape: [batch, seq_len, num_heads, v_head_dim]
        w: Log decay tensor controlling how fast history fades.
            Negative values mean slower decay (longer memory).
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        u: Bonus tensor that boosts the current token's contribution.
            Shape: [num_heads, qk_head_dim]
        softmax_scale: Optional scale for receptance. If None, uses K^-0.5.
        initial_state: Optional initial hidden state for continuation.
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
            or [num_seqs, num_heads, qk_head_dim, v_head_dim] for packed mode.
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed variable-length
            sequences (FlashAttention-style). Shape: [num_seqs + 1]

    Returns:
        Tuple of:
            - output: Attention output matching input dtype.
                Shape: [batch, seq_len, num_heads, v_head_dim]
            - final_state: Final hidden state in float32.
                Shape: [batch, num_heads, qk_head_dim, v_head_dim]
                or [num_seqs, num_heads, qk_head_dim, v_head_dim] for packed mode.

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> batch, seq_len, num_heads, head_dim = 2, 100, 8, 64
        >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> w = -jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.1
        >>> u = jnp.zeros((num_heads, head_dim))
        >>>
        >>> output, state = rwkv6(r, k, v, w, u)
        >>> output.shape
        (2, 100, 8, 64)
        >>>
        >>> # Variable-length sequences
        >>> total_tokens = 300
        >>> r_packed = jnp.ones((1, total_tokens, num_heads, head_dim))
        >>> k_packed = jnp.ones((1, total_tokens, num_heads, head_dim))
        >>> v_packed = jnp.ones((1, total_tokens, num_heads, head_dim))
        >>> w_packed = jnp.zeros((1, total_tokens, num_heads, head_dim))
        >>> cu_seqlens = jnp.array([0, 100, 200, 300])
        >>> output, states = rwkv6(r_packed, k_packed, v_packed, w_packed, u, cu_seqlens=cu_seqlens)
    """
    _validate_rwkv6_inputs(r, k, v, w, u)

    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5

    out_dtype = v.dtype
    r_f32 = r.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    w_f32 = w.astype(jnp.float32)
    u_f32 = u.astype(jnp.float32)

    if cu_seqlens is None:
        B, _, H, K = r.shape
        V = v.shape[-1]
        if initial_state is None:
            initial_state_f32 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
        else:
            initial_state_f32 = initial_state.astype(jnp.float32)
        o_f32, final_state = _rwkv6_scan(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            u_f32,
            softmax_scale=float(softmax_scale),
            initial_state=initial_state_f32,
            reverse=reverse,
        )
    else:
        num_seqs = cu_seqlens.shape[0] - 1
        H, K = r.shape[2], r.shape[3]
        V = v.shape[-1]
        if initial_state is None:
            initial_state_f32 = jnp.zeros((num_seqs, H, K, V), dtype=jnp.float32)
        else:
            initial_state_f32 = initial_state.astype(jnp.float32)
        o_f32, final_state = _rwkv6_varlen(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            u_f32,
            cu_seqlens.astype(jnp.int32),
            softmax_scale=float(softmax_scale),
            initial_state=initial_state_f32,
            reverse=reverse,
        )

    return o_f32.astype(out_dtype), final_state
