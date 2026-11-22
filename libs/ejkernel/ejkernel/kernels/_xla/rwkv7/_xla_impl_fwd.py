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

"""RWKV-7 recurrent kernel (XLA).

This module provides a pure JAX/XLA implementation of the RWKV-7 time-mix
recurrence using a DPLR (Diagonal + Low-Rank) state transition. RWKV-7
is the latest evolution of the RWKV architecture with enhanced expressivity
through low-rank state updates.

RWKV-7 extends RWKV-6 with a more expressive state transition:
1. Diagonal term: exp(w_t) * h_{t-1} (exponential decay)
2. Low-rank term: a_t * (b_t^T @ h_{t-1}) (rank-1 read-write)
3. Key-value term: k_t ⊗ v_t (standard outer product)

The DPLR formulation allows the model to:
- Selectively read from previous state (via b_t)
- Selectively write to state (via a_t)
- Forget with per-dimension rates (via w_t)
- Store new information (via k_t, v_t)

Core recurrence:
    h_t = diag(exp(w_t)) @ h_{t-1} + a_t @ (b_t^T @ h_{t-1}) + k_t @ v_t^T
    o_t = r_t^T @ h_t

Where:
    - h is the state matrix [K, V] per head
    - w controls diagonal decay (log space)
    - a, b provide low-rank read-write mechanism
    - k, v add new key-value pairs
    - r queries the accumulated state

This file provides two parameterizations:
    - `rwkv7`: Standard (a, b) parameterization
    - `rwkv7_mul`: Multiplicative (kk, a) parameterization where b = -kk

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.rwkv7 import rwkv7
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 100, 8, 64
    >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> a = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> b = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>>
    >>> output, final_state = rwkv7(r, w, k, v, a, b)
    >>> output.shape
    (2, 100, 8, 64)

Reference:
    RWKV-7: https://github.com/BlinkDL/RWKV-LM
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _rwkv7_update(
    h: Array,
    r_t: Array,
    k_t: Array,
    v_t: Array,
    w_t: Array,
    a_t: Array,
    b_t: Array,
) -> tuple[Array, Array]:
    """Single step of the RWKV-7 DPLR recurrence.

    Computes one timestep of the RWKV-7 mechanism using the Diagonal + Low-Rank
    state transition that enables selective reading and writing to memory.

    The update consists of three components:
    1. Diagonal decay: h * exp(w_t) - controls forgetting
    2. Low-rank update: a_t @ (b_t^T @ h) - selective read-write
    3. Key-value: k_t @ v_t^T - new information injection

    Args:
        h: Hidden state [B, H, K, V] - key-value memory matrix per head
        r_t: Receptance/query at time t [B, H, K]
        k_t: Key at time t [B, H, K]
        v_t: Value at time t [B, H, V]
        w_t: Log decay at time t [B, H, K]
        a_t: Low-rank write vector [B, H, K]
        b_t: Low-rank read vector [B, H, K]

    Returns:
        Tuple of:
            - h_next: Updated hidden state [B, H, K, V]
            - o_t: Output at time t [B, H, V]
    """
    hb = jnp.einsum("bhk,bhkv->bhv", b_t, h)
    h = h * jnp.exp(w_t)[..., :, None] + a_t[..., :, None] * hb[..., None, :] + k_t[..., :, None] * v_t[..., None, :]
    o_t = jnp.einsum("bhk,bhkv->bhv", r_t, h)
    return h, o_t


def _validate_rwkv7_inputs(r: Array, k: Array, v: Array, w: Array, a: Array, b: Array) -> None:
    """Validate RWKV-7 input tensor shapes and dimensions.

    Args:
        r: Receptance tensor, expected [B, T, H, K]
        k: Key tensor, expected [B, T, H, K]
        v: Value tensor, expected [B, T, H, V]
        w: Log decay tensor, expected [B, T, H, K]
        a: Low-rank write tensor, expected [B, T, H, K]
        b: Low-rank read tensor, expected [B, T, H, K]

    Raises:
        ValueError: If any tensor has incorrect rank or mismatched shapes.
    """
    if r.ndim != 4 or k.ndim != 4 or w.ndim != 4 or a.ndim != 4 or b.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "Expected r,k,w,a,b rank-4 [B,T,H,K] and v rank-4 [B,T,H,V], "
            f"got r={r.shape}, k={k.shape}, v={v.shape}, w={w.shape}, a={a.shape}, b={b.shape}."
        )
    if r.shape != k.shape or r.shape != w.shape or r.shape != a.shape or r.shape != b.shape:
        raise ValueError(
            f"`r`, `k`, `w`, `a`, and `b` must have identical shapes, "
            f"got r={r.shape}, k={k.shape}, w={w.shape}, a={a.shape}, b={b.shape}."
        )
    if v.shape[:3] != r.shape[:3]:
        raise ValueError(f"`v` must match [B,T,H,*], got v={v.shape}, r={r.shape}.")


def _rwkv7_scan(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    a: Array,
    b: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    """Sequential scan for RWKV-7 over full sequence.

    Args:
        r: Receptance/query [B, T, H, K]
        k: Key [B, T, H, K]
        v: Value [B, T, H, V]
        w: Log decay [B, T, H, K]
        a: Low-rank write [B, T, H, K]
        b: Low-rank read [B, T, H, K]
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
        a = a[:, ::-1, :, :]
        b = b[:, ::-1, :, :]

    r = r * softmax_scale

    def step(h, xs):
        """Single RWKV-7 DPLR recurrence step for ``jax.lax.scan``.

        Unpacks the per-timestep inputs and delegates to ``_rwkv7_update``
        to advance the hidden state by one position using the diagonal +
        low-rank state transition.

        Args:
            h: Current hidden state [B, H, K, V].
            xs: Tuple of (r_t, k_t, v_t, w_t, a_t, b_t), each [B, H, *]
                for one timestep.

        Returns:
            Tuple of (h_next, o_t) where h_next is the updated state and
            o_t is the output [B, H, V] for this timestep.
        """
        r_t, k_t, v_t, w_t, a_t, b_t = xs
        return _rwkv7_update(h, r_t, k_t, v_t, w_t, a_t, b_t)

    xs = (
        jnp.swapaxes(r, 0, 1),
        jnp.swapaxes(k, 0, 1),
        jnp.swapaxes(v, 0, 1),
        jnp.swapaxes(w, 0, 1),
        jnp.swapaxes(a, 0, 1),
        jnp.swapaxes(b, 0, 1),
    )
    h_final, oT = jax.lax.scan(step, initial_state, xs)
    o = jnp.swapaxes(oT, 0, 1)
    if reverse:
        o = o[:, ::-1, :, :]
    return o, h_final


def _rwkv7_varlen(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    a: Array,
    b: Array,
    cu_seqlens: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    """RWKV-7 scan for variable-length packed sequences.

    Processes multiple sequences packed into a single tensor, using
    cumulative sequence lengths to determine boundaries.

    Args:
        r: Receptance [1, total_tokens, H, K]
        k: Key [1, total_tokens, H, K]
        v: Value [1, total_tokens, H, V]
        w: Log decay [1, total_tokens, H, K]
        a: Low-rank write [1, total_tokens, H, K]
        b: Low-rank read [1, total_tokens, H, K]
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
        a = a[:, ::-1, :, :]
        b = b[:, ::-1, :, :]
        seq_id = seq_id[::-1]
        is_start, is_end = is_end[::-1], is_start[::-1]

    r = r * softmax_scale

    def step(carry, xs):
        """Single RWKV-7 DPLR step for variable-length packed-token scan.

        Handles sequence boundaries by conditionally resetting the hidden
        state at sequence starts and saving final states at sequence ends.
        Uses the full DPLR update (diagonal + low-rank + key-value) at
        each token position.

        Args:
            carry: Tuple of (h, finals) where h is the running hidden state
                [1, H, K, V] and finals accumulates per-sequence final states
                [num_seqs, H, K, V].
            xs: Tuple of (r_t, k_t, v_t, w_t, a_t, b_t, sid, start, end)
                where sid is the sequence index, start/end are boolean flags
                indicating sequence boundaries.

        Returns:
            Tuple of ((h_next, finals_updated), o_t) where o_t is the
            output [H, V] for this token.
        """
        h, finals = carry
        r_t, k_t, v_t, w_t, a_t, b_t, sid, start, end = xs
        h = jax.lax.cond(start, lambda _: initial_state[sid][None, ...], lambda _: h, operand=None)
        h, o_t = _rwkv7_update(
            h,
            r_t[None, ...],
            k_t[None, ...],
            v_t[None, ...],
            w_t[None, ...],
            a_t[None, ...],
            b_t[None, ...],
        )
        finals = jax.lax.cond(end, lambda f: f.at[sid].set(h[0]), lambda f: f, finals)
        return (h, finals), o_t[0]

    xs = (
        r[0],
        k[0],
        v[0],
        w[0],
        a[0],
        b[0],
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


def rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-7 DPLR recurrent attention using XLA backend.

    Implements the RWKV-7 Diagonal + Low-Rank state update with O(N) complexity.
    The DPLR formulation provides enhanced expressivity by allowing the model
    to selectively read from and write to memory through low-rank projections.

    The recurrence is:
        hb_t = b_t^T @ h_{t-1}  (low-rank read)
        h_t = exp(w_t) * h_{t-1} + a_t @ hb_t^T + k_t @ v_t^T  (DPLR update)
        o_t = r_t^T @ h_t  (query)

    Args:
        r: Receptance/query tensor for attention retrieval.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        w: Log decay tensor controlling diagonal forgetting.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        k: Key tensor for memory addressing.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        v: Value tensor for memory content.
            Shape: [batch, seq_len, num_heads, v_head_dim]
        a: Low-rank write vector controlling what to write.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        b: Low-rank read vector controlling what to read from previous state.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        softmax_scale: Optional scale for receptance. If None, uses K^-0.5.
        initial_state: Optional initial hidden state for continuation.
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
            or [num_seqs, num_heads, qk_head_dim, v_head_dim] for packed mode.
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed variable-length
            sequences. Shape: [num_seqs + 1]

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
        >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> a = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>> b = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>>
        >>> output, state = rwkv7(r, w, k, v, a, b)
        >>> output.shape
        (2, 100, 8, 64)
    """
    _validate_rwkv7_inputs(r, k, v, w, a, b)

    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5

    out_dtype = v.dtype
    r_f32 = r.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    w_f32 = w.astype(jnp.float32)
    a_f32 = a.astype(jnp.float32)
    b_f32 = b.astype(jnp.float32)

    if cu_seqlens is None:
        B, _, H, K = r.shape
        V = v.shape[-1]
        if initial_state is None:
            initial_state_f32 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
        else:
            initial_state_f32 = initial_state.astype(jnp.float32)
        o_f32, final_state = _rwkv7_scan(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            a_f32,
            b_f32,
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
        o_f32, final_state = _rwkv7_varlen(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            a_f32,
            b_f32,
            cu_seqlens.astype(jnp.int32),
            softmax_scale=float(softmax_scale),
            initial_state=initial_state_f32,
            reverse=reverse,
        )

    return o_f32.astype(out_dtype), final_state


def rwkv7_mul(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    kk: Float[Array, "batch seq_len num_heads qk_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-7 multiplicative parameterization using XLA backend.

    Alternative parameterization of RWKV-7 DPLR that uses a multiplicative
    form for the low-rank components. This is equivalent to the standard
    (a, b) parameterization but with different learned parameters.

    The transformation from (kk, a) to (a', b') is:
        a' = kk * a  (element-wise multiplication)
        b' = -kk     (negated kk)

    This parameterization is used by some optimized kernel implementations
    and may provide different training dynamics.

    Args:
        r: Receptance/query tensor.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        w: Log decay tensor.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        k: Key tensor.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        v: Value tensor.
            Shape: [batch, seq_len, num_heads, v_head_dim]
        kk: Key-key tensor used to compute both a' and b'.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        a: Scaling tensor multiplied with kk to get a'.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        softmax_scale: Optional scale for receptance. If None, uses K^-0.5.
        initial_state: Optional initial hidden state.
        reverse: If True, process sequence in reverse.
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.

    Returns:
        Tuple of:
            - output: Attention output [batch, seq_len, num_heads, v_head_dim]
            - final_state: Final hidden state [batch, num_heads, qk_head_dim, v_head_dim]

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> batch, seq_len, num_heads, head_dim = 2, 100, 8, 64
        >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> kk = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> a = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>>
        >>> output, state = rwkv7_mul(r, w, k, v, kk, a)
    """
    return rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        a=kk * a,
        b=-kk,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
