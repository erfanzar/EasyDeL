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

"""RWKV-6 linear attention recurrence implementation using Triton.

This module provides a GPU-accelerated implementation of the RWKV-6 linear
attention mechanism. RWKV-6 extends RWKV-4 with a multi-head architecture
and per-timestep time-decay parameters, enabling more expressive sequence
modeling while maintaining O(N) complexity.

Key improvements over RWKV-4:
- Multi-head architecture for parallel processing of different subspaces
- Data-dependent decay (w) that varies per timestep and head
- Separate query-key (K) and value (V) head dimensions
- Support for grouped-query attention patterns

The RWKV-6 recurrence computes:
    h_t = diag(exp(w_t)) * h_{t-1} + k_t^T * v_t
    o_t = softmax_scale * (r_t * (u * (k_t^T * v_t) + h_{t-1}))

Key components:
- Receptance (r): Analogous to query in standard attention
- Key (k): Used with value to form outer product updates
- Value (v): The values to be aggregated
- Log-decay (w): Per-timestep, per-head decay rates in log-space
- Bonus (u): Direct contribution from current timestep

Key features:
- O(N) time complexity for sequence processing
- Multi-head attention with independent decay per head
- Variable sequence length support via cu_seqlens
- Bidirectional processing via reverse flag
- Custom Triton kernel for GPU acceleration

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.rwkv6 import rwkv6
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> u = jnp.zeros((num_heads, head_dim))
    >>>
    >>> output, final_state = rwkv6(r, k, v, w, u)

Reference:
    Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence
    https://arxiv.org/abs/2404.05892
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.interpreters import ad
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import fwd_triton_impl


def _rwkv6_bwd_sequence(
    r_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    k_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    v_ord: Float[Array, "seq_len num_heads v_head_dim"],
    w_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    do_ord: Float[Array, "seq_len num_heads v_head_dim"],
    dht: Float[Array, "num_heads qk_head_dim v_head_dim"],
    h0: Float[Array, "num_heads qk_head_dim v_head_dim"],
    softmax_scale: float,
) -> tuple[
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "seq_len num_heads v_head_dim"],
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "num_heads qk_head_dim"],
    Float[Array, "num_heads qk_head_dim v_head_dim"],
]:
    """Compute RWKV-6 backward pass gradients for a single sequence using JAX scan.

    Implements the backward pass in pure JAX (no Triton kernel) for a single
    sequence of length ``seq_len`` with a given initial state ``h0``. Used
    by the custom VJP to compute gradients over batches via ``jax.vmap`` or
    sequentially over varlen segments.

    The function performs two scans:
    1. **Forward scan** (``_hist_step``): replays the recurrence to collect the
       pre-update hidden states ``h_prev_hist[t]`` needed for the backward pass.
    2. **Backward scan** (``_bwd_step``): sweeps the sequence in reverse to
       accumulate gradients for each timestep.

    The RWKV-6 recurrence used in the forward scan:
        ``h_t = exp(w_t)[:, :, None] * h_{t-1} + k_t^T * v_t``

    Args:
        r_ord: Receptance tensor for this sequence, shape ``(T, H, K)``.
        k_ord: Key tensor, shape ``(T, H, K)``.
        v_ord: Value tensor, shape ``(T, H, V)``.
        w_ord: Log-decay tensor, shape ``(T, H, K)``.
        u: Bonus tensor shared across the sequence, shape ``(H, K)``.
        do_ord: Output gradient for this sequence, shape ``(T, H, V)``.
        dht: Gradient of the final hidden state, shape ``(H, K, V)``.
        h0: Initial hidden state, shape ``(H, K, V)``.
        softmax_scale: Scaling factor applied to receptance during the forward pass.

    Returns:
        Tuple ``(dr, dk, dv, dw, du, dh0)`` gradients for each input:
            - ``dr``: shape ``(T, H, K)``
            - ``dk``: shape ``(T, H, K)``
            - ``dv``: shape ``(T, H, V)``
            - ``dw``: shape ``(T, H, K)``
            - ``du``: shape ``(H, K)`` (summed over sequence)
            - ``dh0``: shape ``(H, K, V)``
    """
    r_f = r_ord.astype(jnp.float32)
    k_f = k_ord.astype(jnp.float32)
    v_f = v_ord.astype(jnp.float32)
    w_f = w_ord.astype(jnp.float32)
    do_f = do_ord.astype(jnp.float32)
    u_f = u.astype(jnp.float32)
    h0_f = h0.astype(jnp.float32)
    dht_f = dht.astype(jnp.float32)

    def _hist_step(
        h_prev: Float[Array, "num_heads qk_head_dim v_head_dim"],
        xs: tuple[
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
        ],
    ):
        k_t, v_t, w_t = xs
        decay = jnp.exp(w_t)[:, :, None]
        kv = k_t[:, :, None] * v_t[:, None, :]
        h_next = h_prev * decay + kv
        return h_next, h_prev

    _, h_prev_hist = jax.lax.scan(_hist_step, h0_f, (k_f, v_f, w_f))

    def _bwd_step(
        dh_next: Float[Array, "num_heads qk_head_dim v_head_dim"],
        xs: tuple[
            Float[Array, "num_heads qk_head_dim v_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
        ],
    ):
        h_prev, r_t, k_t, v_t, w_t, do_t = xs

        q_t = r_t * softmax_scale
        s_t = jnp.sum(q_t * u_f * k_t, axis=-1)
        ds_t = jnp.sum(do_t * v_t, axis=-1)

        dq_t = jnp.sum(h_prev * do_t[:, None, :], axis=-1) + ds_t[:, None] * (u_f * k_t)
        dr_t = dq_t * softmax_scale

        dk_t = ds_t[:, None] * (q_t * u_f)
        du_t = ds_t[:, None] * (q_t * k_t)
        dv_t = do_t * s_t[:, None]

        decay = jnp.exp(w_t)
        dh_prev = dh_next * decay[:, :, None]
        dw_t = jnp.sum(dh_next * (h_prev * decay[:, :, None]), axis=-1)

        dk_t = dk_t + jnp.sum(dh_next * v_t[:, None, :], axis=-1)
        dv_t = dv_t + jnp.sum(dh_next * k_t[:, :, None], axis=-2)
        dh_prev = dh_prev + q_t[:, :, None] * do_t[:, None, :]

        return dh_prev, (dr_t, dk_t, dv_t, dw_t, du_t)

    xs = (h_prev_hist, r_f, k_f, v_f, w_f, do_f)
    xs_rev = jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=0), xs)
    dh0, grads_rev = jax.lax.scan(_bwd_step, dht_f, xs_rev)
    dr_rev, dk_rev, dv_rev, dw_rev, du_seq_rev = grads_rev

    dr = jnp.flip(dr_rev, axis=0)
    dk = jnp.flip(dk_rev, axis=0)
    dv = jnp.flip(dv_rev, axis=0)
    dw = jnp.flip(dw_rev, axis=0)
    du = jnp.sum(jnp.flip(du_seq_rev, axis=0), axis=0)

    return dr, dk, dv, dw, du, dh0


def _normalize_rwkv6_varlen_state(
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    *,
    num_seqs: int,
    num_heads: int,
    qk_dim: int,
    v_dim: int,
) -> tuple[Float[Array, "num_seqs num_heads qk_head_dim v_head_dim"], bool]:
    """Normalise an optional initial state for RWKV-6 variable-length mode.

    Validates and reshapes the ``initial_state`` argument into a contiguous
    ``(num_seqs, H, K, V)`` float32 array required by the varlen forward/backward
    paths. Supports rank-4 ``[N, H, K, V]`` and rank-5 ``[1, N, H, K, V]`` inputs
    (the latter arising when a batched model wraps the state with a leading
    batch-size-1 dimension).

    Args:
        initial_state: Optional initial hidden state. Accepted shapes:
            - ``None``: initialises to zeros.
            - ``(num_seqs, H, K, V)``: used directly.
            - ``(1, num_seqs, H, K, V)``: squeezed along the leading dim.
        num_seqs: Number of sequences in the packed batch.
        num_heads: Number of attention heads (``H``).
        qk_dim: Key/query head dimension (``K``).
        v_dim: Value head dimension (``V``).

    Returns:
        Tuple ``(state, state_was_none)`` where:
            - ``state``: float32 array of shape ``(num_seqs, H, K, V)``.
            - ``state_was_none``: True if the caller passed ``None``.

    Raises:
        ValueError: If the state shape is not rank-4 or rank-5 with
            leading size 1, or if ``state.shape[0] != num_seqs``.
    """
    if initial_state is None:
        return jnp.zeros((num_seqs, num_heads, qk_dim, v_dim), dtype=jnp.float32), True

    state = jnp.asarray(initial_state)
    if state.ndim == 5:
        if state.shape[0] != 1:
            raise ValueError(
                f"Packed varlen RWKV6 initial_state with rank-5 must have leading batch size 1, got shape {state.shape}."
            )
        state = state[0]
    if state.ndim != 4:
        raise ValueError(f"RWKV6 varlen initial_state must be rank-4 [N,H,K,V], got shape {state.shape}.")
    if state.shape[0] != num_seqs:
        raise ValueError(f"RWKV6 varlen initial_state must have N={num_seqs}, got shape {state.shape}.")
    return state.astype(jnp.float32), False


def _fwd_call(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    softmax_scale: float | None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
):
    """Forward pass for RWKV-6 recurrence in a custom VJP.

    Computes the RWKV-6 linear attention and saves residuals for backward pass.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        u: Bonus tensor of shape `[H, K]`.
        softmax_scale: Scaling factor for attention computation.
        initial_state: Optional initial hidden state of shape `[B, H, K, V]`.
        reverse: Whether to process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.

    Returns:
        A tuple containing (output, final_state) and residuals for backward.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5

    out, final_state = fwd_triton_impl(
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    residual = (r, k, v, w, u, softmax_scale, initial_state, reverse, cu_seqlens)
    return (out, final_state), residual


def _bwd_call(
    softmax_scale: float | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    residual,
    grads,
):
    """Backward pass for RWKV-6 recurrence in a custom VJP.

    Args:
        softmax_scale: Non-differentiable scaling factor.
        reverse: Non-differentiable reverse flag.
        cu_seqlens: Non-differentiable cumulative sequence lengths.
        residual: Tensors saved from the forward pass.
        grads: A tuple containing gradients (do, dht) of output and final state.
    """
    r, k, v, w, u, _, initial_state, _, _ = residual
    do, dht = grads

    do = ad.instantiate_zeros(do)
    dht = ad.instantiate_zeros(dht)

    if do is None:
        do = jnp.zeros_like(v)

    if softmax_scale is None:
        softmax_scale = float(r.shape[-1] ** -0.5)

    if cu_seqlens is None:
        batch_size, _, num_heads, qk_dim = r.shape
        v_dim = v.shape[-1]
        if initial_state is None:
            h0 = jnp.zeros((batch_size, num_heads, qk_dim, v_dim), dtype=jnp.float32)
            state_was_none = True
        else:
            h0 = initial_state.astype(jnp.float32)
            state_was_none = False

        if dht is None:
            dht = jnp.zeros_like(h0)
        else:
            dht = dht.astype(jnp.float32)

        r_ord = jnp.flip(r, axis=1) if reverse else r
        k_ord = jnp.flip(k, axis=1) if reverse else k
        v_ord = jnp.flip(v, axis=1) if reverse else v
        w_ord = jnp.flip(w, axis=1) if reverse else w
        do_ord = jnp.flip(do, axis=1) if reverse else do

        dr_ord, dk_ord, dv_ord, dw_ord, du_bt, dh0 = jax.vmap(
            _rwkv6_bwd_sequence,
            in_axes=(0, 0, 0, 0, None, 0, 0, 0, None),
            out_axes=(0, 0, 0, 0, 0, 0),
        )(r_ord, k_ord, v_ord, w_ord, u, do_ord, dht, h0, float(softmax_scale))

        dr = jnp.flip(dr_ord, axis=1) if reverse else dr_ord
        dk = jnp.flip(dk_ord, axis=1) if reverse else dk_ord
        dv = jnp.flip(dv_ord, axis=1) if reverse else dv_ord
        dw = jnp.flip(dw_ord, axis=1) if reverse else dw_ord
        du = jnp.sum(du_bt, axis=0)
        dinitial_state = None if state_was_none else dh0.astype(initial_state.dtype)
    else:
        if r.ndim == 4:
            if r.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1 or w.shape[0] != 1:
                raise ValueError(
                    "RWKV6 packed varlen mode expects rank-4 inputs with batch size 1; "
                    f"got r/k/v/w batch sizes {r.shape[0]}/{k.shape[0]}/{v.shape[0]}/{w.shape[0]}."
                )
            r_tok, k_tok, v_tok, w_tok = r[0], k[0], v[0], w[0]
            do_tok = do[0]
            add_batch_dim = True
        elif r.ndim == 3:
            r_tok, k_tok, v_tok, w_tok = r, k, v, w
            do_tok = do
            add_batch_dim = False
        else:
            raise ValueError(f"RWKV6 varlen expects rank-3 or rank-4 inputs, got rank {r.ndim}.")

        num_seqs = int(cu_seqlens.shape[0] - 1)
        h0_all, state_was_none = _normalize_rwkv6_varlen_state(
            initial_state,
            num_seqs=num_seqs,
            num_heads=r_tok.shape[1],
            qk_dim=r_tok.shape[2],
            v_dim=v_tok.shape[2],
        )
        if dht is None:
            dht_all = jnp.zeros_like(h0_all)
        else:
            dht_all = dht.astype(jnp.float32)

        dr_tok = jnp.zeros_like(r_tok, dtype=jnp.float32)
        dk_tok = jnp.zeros_like(k_tok, dtype=jnp.float32)
        dv_tok = jnp.zeros_like(v_tok, dtype=jnp.float32)
        dw_tok = jnp.zeros_like(w_tok, dtype=jnp.float32)
        du = jnp.zeros_like(u, dtype=jnp.float32)
        dh0_all = jnp.zeros_like(h0_all, dtype=jnp.float32)

        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx])
            end = int(cu_seqlens[seq_idx + 1])

            r_seq = r_tok[start:end]
            k_seq = k_tok[start:end]
            v_seq = v_tok[start:end]
            w_seq = w_tok[start:end]
            do_seq = do_tok[start:end]

            if reverse:
                r_seq = jnp.flip(r_seq, axis=0)
                k_seq = jnp.flip(k_seq, axis=0)
                v_seq = jnp.flip(v_seq, axis=0)
                w_seq = jnp.flip(w_seq, axis=0)
                do_seq = jnp.flip(do_seq, axis=0)

            dr_seq, dk_seq, dv_seq, dw_seq, du_seq, dh0_seq = _rwkv6_bwd_sequence(
                r_seq,
                k_seq,
                v_seq,
                w_seq,
                u,
                do_seq,
                dht_all[seq_idx],
                h0_all[seq_idx],
                float(softmax_scale),
            )

            if reverse:
                dr_seq = jnp.flip(dr_seq, axis=0)
                dk_seq = jnp.flip(dk_seq, axis=0)
                dv_seq = jnp.flip(dv_seq, axis=0)
                dw_seq = jnp.flip(dw_seq, axis=0)

            dr_tok = dr_tok.at[start:end].set(dr_seq)
            dk_tok = dk_tok.at[start:end].set(dk_seq)
            dv_tok = dv_tok.at[start:end].set(dv_seq)
            dw_tok = dw_tok.at[start:end].set(dw_seq)
            du = du + du_seq
            dh0_all = dh0_all.at[seq_idx].set(dh0_seq)

        if add_batch_dim:
            dr = dr_tok[None, ...]
            dk = dk_tok[None, ...]
            dv = dv_tok[None, ...]
            dw = dw_tok[None, ...]
        else:
            dr, dk, dv, dw = dr_tok, dk_tok, dv_tok, dw_tok

        dinitial_state = None if state_was_none else dh0_all.astype(initial_state.dtype)

    return (
        dr.astype(r.dtype),
        dk.astype(k.dtype),
        dv.astype(v.dtype),
        dw.astype(w.dtype),
        du.astype(u.dtype),
        dinitial_state,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 7, 8))
@partial(jax.jit, static_argnums=(5, 7))
def _rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Core JIT-compiled RWKV-6 function with custom VJP.

    This internal function directly calls the Triton implementation and is
    registered with JAX's custom differentiation system for memory-efficient
    gradient computation.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        u: Bonus tensor of shape `[H, K]`.
        softmax_scale: Scaling factor (static argument).
        initial_state: Optional initial hidden state of shape `[B, H, K, V]`.
        reverse: Process in reverse order (static argument).
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.

    Returns:
        A tuple containing:
            - output: The attention output tensor of shape `[B, T, H, V]`.
            - final_state: The final hidden state of shape `[B, H, K, V]`.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5
    return fwd_triton_impl(
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


_rwkv6.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("rwkv6", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
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
    """RWKV-6 linear attention recurrence (Triton GPU implementation).

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        w: Log decay tensor `[B, T, H, K]`.
        u: Bonus tensor `[H, K]`.
        softmax_scale: Optional scale for receptance.
        initial_state: Optional initial state `[B, H, K, V]`.
        reverse: Process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.

    Returns:
        Tuple of (output `[B, T, H, V]`, final_state `[B, H, K, V]`).
    """
    return _rwkv6(
        r,
        k,
        v,
        w,
        u,
        softmax_scale,
        initial_state,
        reverse,
        cu_seqlens,
    )
