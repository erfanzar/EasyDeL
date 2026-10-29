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

"""RWKV-7 Diagonal Plus Low Rank (DPLR) recurrence implementation using Triton.

This module provides a GPU-accelerated implementation of the RWKV-7 linear
attention mechanism. RWKV-7 introduces a Diagonal Plus Low-Rank (DPLR)
parameterization for the state transition matrix, enabling more expressive
state dynamics while maintaining O(N) complexity.

Key improvements over RWKV-6:
- DPLR state transition: Combines diagonal decay with low-rank updates
- Enhanced expressiveness: Can model richer state transitions
- Flexible parameterization: Supports both additive (a, b) and multiplicative
  (kk, a) forms

The RWKV-7 DPLR recurrence computes:
    h_t = diag(exp(w_t)) * (h_{t-1} + a_t^T * (b_t @ h_{t-1})) + k_t^T * v_t
    o_t = softmax_scale * (r_t @ h_{t-1})

The low-rank update (a, b) allows the model to selectively modify the hidden
state based on learned projections, providing more flexible information routing
than simple diagonal decay.

Key components:
- Receptance (r): Query-like projection for reading from state
- Log-decay (w): Per-timestep, per-head diagonal decay rates
- Key (k): Used with value for rank-1 state updates
- Value (v): The values to be accumulated in state
- Low-rank a: First component of DPLR update (rank-1 outer product)
- Low-rank b: Second component of DPLR update (projection vector)

Alternative multiplicative parameterization (rwkv7_mul):
- kk: Multiplicative scaling factor
- a: Base update vector
- Computes a' = kk * a, b' = -kk internally

Key features:
- O(N) time complexity for sequence processing
- DPLR state dynamics for enhanced expressiveness
- Variable sequence length support via cu_seqlens
- Bidirectional processing via reverse flag
- Custom Triton kernel for GPU acceleration

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.rwkv7 import rwkv7
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> a = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> b = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>>
    >>> output, final_state = rwkv7(r, w, k, v, a, b)

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


def _rwkv7_bwd_sequence(
    r_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    w_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    k_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    v_ord: Float[Array, "seq_len num_heads v_head_dim"],
    a_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    b_ord: Float[Array, "seq_len num_heads qk_head_dim"],
    do_ord: Float[Array, "seq_len num_heads v_head_dim"],
    dht: Float[Array, "num_heads qk_head_dim v_head_dim"],
    h0: Float[Array, "num_heads qk_head_dim v_head_dim"],
    softmax_scale: float,
) -> tuple[
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "seq_len num_heads v_head_dim"],
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "seq_len num_heads qk_head_dim"],
    Float[Array, "num_heads qk_head_dim v_head_dim"],
]:
    """Compute RWKV-7 DPLR backward gradients for a single sequence using JAX scan.

    Implements the backward pass in pure JAX (no Triton kernel) for a single
    sequence with initial state ``h0``. Used by the custom VJP via ``jax.vmap``
    (batched mode) or sequential iteration (varlen mode).

    Two scans are performed:
    1. **Forward scan** (``_hist_step``): replays the DPLR recurrence to collect
       ``h_prev_hist[t]`` (the hidden state *before* token ``t`` is processed).
    2. **Backward scan** (``_bwd_step``): sweeps the sequence in reverse to
       accumulate gradients for each timestep.

    The RWKV-7 DPLR recurrence used in the forward scan:
        ``hb_t = sum(b_t[:, :, None] * h, axis=1)``  (project state via b)
        ``h_mid = exp(w_t)[:, :, None] * h + a_t[:, :, None] * hb_t[:, None, :]``
        ``h_t = h_mid + k_t[:, :, None] * v_t[:, None, :]``

    Args:
        r_ord: Receptance tensor for this sequence, shape ``(T, H, K)``.
        w_ord: Log-decay tensor, shape ``(T, H, K)``.
        k_ord: Key tensor, shape ``(T, H, K)``.
        v_ord: Value tensor, shape ``(T, H, V)``.
        a_ord: Low-rank update vector, shape ``(T, H, K)``.
        b_ord: Low-rank projection vector, shape ``(T, H, K)``.
        do_ord: Output gradient for this sequence, shape ``(T, H, V)``.
        dht: Gradient of the final hidden state, shape ``(H, K, V)``.
        h0: Initial hidden state, shape ``(H, K, V)``.
        softmax_scale: Scaling factor applied to receptance during the forward pass.

    Returns:
        Tuple ``(dr, dw, dk, dv, da, db, dh0)`` gradients for each input:
            - ``dr``, ``dw``, ``dk``, ``da``, ``db``: shape ``(T, H, K)``
            - ``dv``: shape ``(T, H, V)``
            - ``dh0``: shape ``(H, K, V)``
    """
    r_f = r_ord.astype(jnp.float32)
    w_f = w_ord.astype(jnp.float32)
    k_f = k_ord.astype(jnp.float32)
    v_f = v_ord.astype(jnp.float32)
    a_f = a_ord.astype(jnp.float32)
    b_f = b_ord.astype(jnp.float32)
    do_f = do_ord.astype(jnp.float32)
    h0_f = h0.astype(jnp.float32)
    dht_f = dht.astype(jnp.float32)

    def _hist_step(
        h_prev: Float[Array, "num_heads qk_head_dim v_head_dim"],
        xs: tuple[
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
        ],
    ):
        w_t, k_t, v_t, a_t, b_t = xs
        hb_t = jnp.sum(b_t[:, :, None] * h_prev, axis=1)
        decay = jnp.exp(w_t)[:, :, None]
        h_mid = decay * h_prev + a_t[:, :, None] * hb_t[:, None, :]
        h_next = h_mid + k_t[:, :, None] * v_t[:, None, :]
        return h_next, h_prev

    _, h_prev_hist = jax.lax.scan(_hist_step, h0_f, (w_f, k_f, v_f, a_f, b_f))

    def _bwd_step(
        dh_next: Float[Array, "num_heads qk_head_dim v_head_dim"],
        xs: tuple[
            Float[Array, "num_heads qk_head_dim v_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads qk_head_dim"],
            Float[Array, "num_heads v_head_dim"],
        ],
    ):
        h_prev, r_t, w_t, k_t, v_t, a_t, b_t, do_t = xs

        q_t = r_t * softmax_scale
        hb_t = jnp.sum(b_t[:, :, None] * h_prev, axis=1)
        decay = jnp.exp(w_t)[:, :, None]
        h_mid = decay * h_prev + a_t[:, :, None] * hb_t[:, None, :]
        h_t = h_mid + k_t[:, :, None] * v_t[:, None, :]

        dq_t = jnp.sum(h_t * do_t[:, None, :], axis=-1)
        dr_t = dq_t * softmax_scale
        dh_t = dh_next + q_t[:, :, None] * do_t[:, None, :]

        dk_t = jnp.sum(dh_t * v_t[:, None, :], axis=-1)
        dv_t = jnp.sum(dh_t * k_t[:, :, None], axis=-2)

        dh_mid = dh_t
        dw_t = jnp.sum(dh_mid * (h_prev * decay), axis=-1)
        dh_prev = dh_mid * decay

        da_t = jnp.sum(dh_mid * hb_t[:, None, :], axis=-1)
        dhb_t = jnp.sum(dh_mid * a_t[:, :, None], axis=-2)
        db_t = jnp.sum(dhb_t[:, None, :] * h_prev, axis=-1)
        dh_prev = dh_prev + b_t[:, :, None] * dhb_t[:, None, :]

        return dh_prev, (dr_t, dw_t, dk_t, dv_t, da_t, db_t)

    xs = (h_prev_hist, r_f, w_f, k_f, v_f, a_f, b_f, do_f)
    xs_rev = jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=0), xs)
    dh0, grads_rev = jax.lax.scan(_bwd_step, dht_f, xs_rev)
    dr_rev, dw_rev, dk_rev, dv_rev, da_rev, db_rev = grads_rev

    return (
        jnp.flip(dr_rev, axis=0),
        jnp.flip(dw_rev, axis=0),
        jnp.flip(dk_rev, axis=0),
        jnp.flip(dv_rev, axis=0),
        jnp.flip(da_rev, axis=0),
        jnp.flip(db_rev, axis=0),
        dh0,
    )


def _normalize_rwkv7_varlen_state(
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    *,
    num_seqs: int,
    num_heads: int,
    qk_dim: int,
    v_dim: int,
) -> tuple[Float[Array, "num_seqs num_heads qk_head_dim v_head_dim"], bool]:
    """Normalise an optional initial state for RWKV-7 variable-length mode.

    Validates and reshapes ``initial_state`` into a ``(num_seqs, H, K, V)`` float32
    array required by the RWKV-7 varlen forward/backward paths. Mirrors
    ``_normalize_rwkv6_varlen_state`` for the RWKV-7 kernel.

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
                f"Packed varlen RWKV7 initial_state with rank-5 must have leading batch size 1, got shape {state.shape}."
            )
        state = state[0]
    if state.ndim != 4:
        raise ValueError(f"RWKV7 varlen initial_state must be rank-4 [N,H,K,V], got shape {state.shape}.")
    if state.shape[0] != num_seqs:
        raise ValueError(f"RWKV7 varlen initial_state must have N={num_seqs}, got shape {state.shape}.")
    return state.astype(jnp.float32), False


def _fwd_call(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    softmax_scale: float | None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
):
    """Forward pass for RWKV-7 DPLR recurrence in a custom VJP.

    Computes the RWKV-7 recurrence with DPLR state transition and saves
    residuals for backward pass.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        a: Low-rank update vector of shape `[B, T, H, K]`.
        b: Low-rank projection vector of shape `[B, T, H, K]`.
        softmax_scale: Scaling factor for receptance.
        initial_state: Optional initial hidden state `[B, H, K, V]`.
        reverse: If True, process sequence in reverse.
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        A tuple containing (output, final_state) and residuals for backward.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5
    out, final_state = fwd_triton_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    residual = (r, w, k, v, a, b, softmax_scale, initial_state, reverse, cu_seqlens)
    return (out, final_state), residual


def _bwd_call(
    softmax_scale: float | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    block_v: int,
    num_warps: int,
    num_stages: int,
    residual,
    grads,
):
    """Backward pass for RWKV-7 DPLR recurrence in a custom VJP.

    Args:
        softmax_scale: Non-differentiable scaling factor.
        reverse: Non-differentiable reverse flag.
        cu_seqlens: Non-differentiable cumulative sequence lengths.
        block_v: Forward value tile size.
        num_warps: Forward Triton warp count.
        num_stages: Forward Triton pipeline stage count.
        residual: Tensors saved from the forward pass.
        grads: A tuple containing gradients (do, dht) of output and final state.
    """
    r, w, k, v, a, b, _, initial_state, _, _ = residual
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
        w_ord = jnp.flip(w, axis=1) if reverse else w
        k_ord = jnp.flip(k, axis=1) if reverse else k
        v_ord = jnp.flip(v, axis=1) if reverse else v
        a_ord = jnp.flip(a, axis=1) if reverse else a
        b_ord = jnp.flip(b, axis=1) if reverse else b
        do_ord = jnp.flip(do, axis=1) if reverse else do

        dr_ord, dw_ord, dk_ord, dv_ord, da_ord, db_ord, dh0 = jax.vmap(
            _rwkv7_bwd_sequence,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None),
            out_axes=(0, 0, 0, 0, 0, 0, 0),
        )(r_ord, w_ord, k_ord, v_ord, a_ord, b_ord, do_ord, dht, h0, float(softmax_scale))

        dr = jnp.flip(dr_ord, axis=1) if reverse else dr_ord
        dw = jnp.flip(dw_ord, axis=1) if reverse else dw_ord
        dk = jnp.flip(dk_ord, axis=1) if reverse else dk_ord
        dv = jnp.flip(dv_ord, axis=1) if reverse else dv_ord
        da = jnp.flip(da_ord, axis=1) if reverse else da_ord
        db = jnp.flip(db_ord, axis=1) if reverse else db_ord
        dinitial_state = None if state_was_none else dh0.astype(initial_state.dtype)
    else:
        if r.ndim == 4:
            if r.shape[0] != 1 or w.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
                raise ValueError(
                    "RWKV7 packed varlen mode expects rank-4 inputs with batch size 1; "
                    f"got r/w/k/v batch sizes {r.shape[0]}/{w.shape[0]}/{k.shape[0]}/{v.shape[0]}."
                )
            r_tok, w_tok, k_tok, v_tok, a_tok, b_tok = r[0], w[0], k[0], v[0], a[0], b[0]
            do_tok = do[0]
            add_batch_dim = True
        elif r.ndim == 3:
            r_tok, w_tok, k_tok, v_tok, a_tok, b_tok = r, w, k, v, a, b
            do_tok = do
            add_batch_dim = False
        else:
            raise ValueError(f"RWKV7 varlen expects rank-3 or rank-4 inputs, got rank {r.ndim}.")

        num_seqs = int(cu_seqlens.shape[0] - 1)
        h0_all, state_was_none = _normalize_rwkv7_varlen_state(
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
        dw_tok = jnp.zeros_like(w_tok, dtype=jnp.float32)
        dk_tok = jnp.zeros_like(k_tok, dtype=jnp.float32)
        dv_tok = jnp.zeros_like(v_tok, dtype=jnp.float32)
        da_tok = jnp.zeros_like(a_tok, dtype=jnp.float32)
        db_tok = jnp.zeros_like(b_tok, dtype=jnp.float32)
        dh0_all = jnp.zeros_like(h0_all, dtype=jnp.float32)

        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx])
            end = int(cu_seqlens[seq_idx + 1])

            r_seq = r_tok[start:end]
            w_seq = w_tok[start:end]
            k_seq = k_tok[start:end]
            v_seq = v_tok[start:end]
            a_seq = a_tok[start:end]
            b_seq = b_tok[start:end]
            do_seq = do_tok[start:end]

            if reverse:
                r_seq = jnp.flip(r_seq, axis=0)
                w_seq = jnp.flip(w_seq, axis=0)
                k_seq = jnp.flip(k_seq, axis=0)
                v_seq = jnp.flip(v_seq, axis=0)
                a_seq = jnp.flip(a_seq, axis=0)
                b_seq = jnp.flip(b_seq, axis=0)
                do_seq = jnp.flip(do_seq, axis=0)

            dr_seq, dw_seq, dk_seq, dv_seq, da_seq, db_seq, dh0_seq = _rwkv7_bwd_sequence(
                r_seq,
                w_seq,
                k_seq,
                v_seq,
                a_seq,
                b_seq,
                do_seq,
                dht_all[seq_idx],
                h0_all[seq_idx],
                float(softmax_scale),
            )

            if reverse:
                dr_seq = jnp.flip(dr_seq, axis=0)
                dw_seq = jnp.flip(dw_seq, axis=0)
                dk_seq = jnp.flip(dk_seq, axis=0)
                dv_seq = jnp.flip(dv_seq, axis=0)
                da_seq = jnp.flip(da_seq, axis=0)
                db_seq = jnp.flip(db_seq, axis=0)

            dr_tok = dr_tok.at[start:end].set(dr_seq)
            dw_tok = dw_tok.at[start:end].set(dw_seq)
            dk_tok = dk_tok.at[start:end].set(dk_seq)
            dv_tok = dv_tok.at[start:end].set(dv_seq)
            da_tok = da_tok.at[start:end].set(da_seq)
            db_tok = db_tok.at[start:end].set(db_seq)
            dh0_all = dh0_all.at[seq_idx].set(dh0_seq)

        if add_batch_dim:
            dr = dr_tok[None, ...]
            dw = dw_tok[None, ...]
            dk = dk_tok[None, ...]
            dv = dv_tok[None, ...]
            da = da_tok[None, ...]
            db = db_tok[None, ...]
        else:
            dr, dw, dk, dv, da, db = dr_tok, dw_tok, dk_tok, dv_tok, da_tok, db_tok

        dinitial_state = None if state_was_none else dh0_all.astype(initial_state.dtype)

    return (
        dr.astype(r.dtype),
        dw.astype(w.dtype),
        dk.astype(k.dtype),
        dv.astype(v.dtype),
        da.astype(a.dtype),
        db.astype(b.dtype),
        dinitial_state,
    )


@partial(jax.custom_vjp, nondiff_argnums=(6, 8, 9, 10, 11, 12))
@partial(jax.jit, static_argnums=(6, 8, 10, 11, 12))
def _rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Core JIT-compiled RWKV-7 function with a custom VJP.

    This is an internal function that directly calls the Triton implementation
    and is registered with JAX's custom differentiation system.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        a: Low-rank update vector of shape `[B, T, H, K]`.
        b: Low-rank projection vector of shape `[B, T, H, K]`.
        softmax_scale: Scaling factor for receptance (static argument).
        initial_state: Optional initial hidden state `[B, H, K, V]`.
        reverse: If True, process sequence in reverse (static argument).
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        A tuple containing:
            - output: The attention output tensor of shape `[B, T, H, V]`.
            - final_state: The final hidden state of shape `[B, H, K, V]`.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5
    return fwd_triton_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )


_rwkv7.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("rwkv7", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
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
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-7 DPLR recurrence (a,b) (Triton GPU implementation).

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        a: Low-rank update vector `[B, T, H, K]`.
        b: Low-rank projection vector `[B, T, H, K]`.
        softmax_scale: Optional scale for receptance.
        initial_state: Optional initial state `[B, H, K, V]`.
        reverse: Process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Tuple of (output `[B, T, H, V]`, final_state `[B, H, K, V]`).
    """
    return _rwkv7(r, w, k, v, a, b, softmax_scale, initial_state, reverse, cu_seqlens, block_v, num_warps, num_stages)


@kernel_registry.register("rwkv7_mul", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
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
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-7 multiplicative (kk, a) parameterization (Triton GPU implementation).

    Converts (kk, a) to standard DPLR form: a' = kk * a, b' = -kk.

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        kk: Multiplicative factor `[B, T, H, K]`.
        a: Low-rank update base `[B, T, H, K]`.
        softmax_scale: Optional scale for receptance.
        initial_state: Optional initial state `[B, H, K, V]`.
        reverse: Process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Tuple of (output `[B, T, H, V]`, final_state `[B, H, K, V]`).
    """
    return _rwkv7(
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
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
