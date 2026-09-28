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

"""Recurrent linear attention backward pass implementation using XLA/JAX.

Custom backward rule for ``_recurrent_core`` (via ``jax.custom_vjp``).
Avoids re-running the forward pass by using the hidden states materialised
during ``_recurrent_attention_fwd``.

Algorithm (per batch element, run via ``jax.vmap``):
    Reverse-time ``lax.scan`` over ``(q, k, v, g, gk, gv, h, do)``:

    1. ``dq_t = sum(h_t * do_t[:, None, :], axis=-1) * scale``
    2. ``dh_t  = dh_next + (q_t * scale)[:, :, None] * do_t[:, None, :]``
    3. ``dk_t  = einsum("nkv, nv -> nk", dh_t, v_t)``
    4. ``dv_t  = einsum("nkv, nk -> nv", dh_t, k_t)``
    5. Backward through each active gate in reverse application order:
       ``gv → gk → g_gamma → g``.
    6. ``dh_prev = dh_current * product_of_decay_gradients``

Gradient equations:
    - ``dL/dq_t = softmax_scale * h_t @ do_t``
    - ``dL/dk_t = dh_t @ v_t``
    - ``dL/dv_t = k_t^T @ dh_t``
    - ``dL/dg_t  = sum_v(dh1_t * h1_t)``  with ``h1_t = h_{t-1} * exp(g_t)`` (GLA gate)
    - ``dL/dgk_t = sum_v(dh3_t * h3_t)``  with ``h3_t = h2_t * exp(gk_t)`` (key gate)
    - ``dL/dgv_t = sum_k(dh_t  * h4_t)``  with ``h4_t = h3_t * exp(gv_t)`` (value gate)
    - ``d_initial_state = dh after all reverse steps``

Note:
    Gates that were not provided are zero-filled by the forward rule, so
    ``exp(0) == 1`` makes their decay an exact no-op; the gate chain is
    applied unconditionally to keep the rule traceable under ``jax.jit``.
    ``g_gamma`` gradient is not produced (it is a nondiff argument handled
    in ``_recurrent_bwd`` in ``_interface.py``).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _recurrent_attention_bwd(
    q: Float[Array, "batch seq_len num_heads head_dim"],
    k: Float[Array, "batch seq_len num_heads head_dim"],
    v: Float[Array, "batch seq_len num_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"],
    g_gamma: Float[Array, "... num_heads"],
    gk: Float[Array, "batch seq_len num_heads head_dim"],
    gv: Float[Array, "batch seq_len num_heads head_dim"],
    hidden_states: Float[Array, "batch seq_len num_heads head_dim head_dim"],
    do: Float[Array, "batch seq_len num_heads head_dim"],
    dfinal_state: Float[Array, "batch num_heads head_dim head_dim"],
    softmax_scale: float,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"],
    reverse: bool,
) -> tuple:
    """Backward pass for recurrent linear attention.

    Uses hidden states saved during the forward pass to avoid re-running the
    recurrence, then runs a reverse-time scan to accumulate gradients.

    Args:
        q: Query tensor, shape ``[batch, seq_len, num_heads, head_dim]``.
        k: Key tensor, same shape as ``q``.
        v: Value tensor, same shape as ``q``.
        g: GLA gate, shape ``[batch, seq_len, num_heads, head_dim]``.
            May be all-zero if GLA gating was inactive.
        g_gamma: Per-head decay, shape ``[num_heads]`` or
            ``[batch, num_heads]``.
        gk: Key gate, same shape as ``q``.
        gv: Value gate, same shape as ``q``.
        hidden_states: All hidden states from the forward pass,
            shape ``[batch, seq_len, num_heads, head_dim, head_dim]``.
            ``hidden_states[:, t, ...]`` is ``h_t`` (the state *after* step ``t``).
        do: Gradient of the output, shape ``[batch, seq_len, num_heads, head_dim]``.
        dfinal_state: Gradient of the final hidden state,
            shape ``[batch, num_heads, head_dim, head_dim]``.
        softmax_scale: Query scaling factor used in forward pass.
        initial_state: Initial hidden state used in forward pass,
            shape ``[batch, num_heads, head_dim, head_dim]``.
        reverse: Whether the forward pass processed the sequence in reverse.
            If ``True``, inputs/outputs are flipped before/after the scan.

    Returns:
        Tuple of ``(dq, dk, dv, dg, dgk, dgv, dinitial_state)`` with the
        same shapes as the corresponding forward inputs.
    """
    batch, seq_len, num_heads, _head_dim = q.shape

    if g_gamma.ndim == 1:
        if g_gamma.shape != (num_heads,):
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({batch}, {num_heads})")
        g_gamma_batch = jnp.broadcast_to(g_gamma, (batch, num_heads))
    elif g_gamma.ndim == 2:
        if g_gamma.shape[1] != num_heads:
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({batch}, {num_heads})")
        if g_gamma.shape[0] == 1 and batch != 1:
            g_gamma_batch = jnp.broadcast_to(g_gamma, (batch, num_heads))
        elif g_gamma.shape[0] == batch:
            g_gamma_batch = g_gamma
        else:
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({batch}, {num_heads})")
    else:
        raise ValueError(f"g_gamma.ndim={g_gamma.ndim} must be 1 or 2")

    if reverse:
        do = jnp.flip(do, axis=1)
        q = jnp.flip(q, axis=1)
        k = jnp.flip(k, axis=1)
        v = jnp.flip(v, axis=1)
        g = jnp.flip(g, axis=1)
        gk = jnp.flip(gk, axis=1)
        gv = jnp.flip(gv, axis=1)
        hidden_states = jnp.flip(hidden_states, axis=1)

    # State entering each step in the (possibly flipped) processing order:
    # hprev[:, t] = h_{t-1}, with h_{-1} = initial_state.  This must be built
    # AFTER the reverse flip above: the forward applies ``initial_state`` at
    # the start of the flipped scan, so flipping a pre-built hprev would put
    # ``initial_state`` at the wrong end and shift every entry by one step.
    h_prev_states = jnp.concatenate([initial_state[:, None], hidden_states[:, :-1]], axis=1)

    def process_batch(q_b, k_b, v_b, g_b, g_gamma_b, gk_b, gv_b, hidden_b, hprev_b, do_b, dfinal_h):
        """Process backward pass for a single batch element.

        Runs the reverse-time scan to compute gradients for all recurrent
        attention parameters. Uses lax.scan in reverse order and accumulates
        per-step contributions for gates and initial state.

        Args:
            q_b: Query for this batch [seq_len, num_heads, head_dim].
            k_b: Key for this batch [seq_len, num_heads, head_dim].
            v_b: Value for this batch [seq_len, num_heads, head_dim].
            g_b: GLA gate for this batch [seq_len, num_heads, head_dim].
            g_gamma_b: Per-head decay for this batch [num_heads].
            gk_b: Key gate for this batch [seq_len, num_heads, head_dim].
            gv_b: Value gate for this batch [seq_len, num_heads, head_dim].
            hidden_b: Saved hidden states [seq_len, num_heads, head_dim, head_dim].
            hprev_b: States entering each step [seq_len, num_heads, head_dim, head_dim].
            do_b: Output gradient [seq_len, num_heads, head_dim].
            dfinal_h: Gradient of final hidden state [num_heads, head_dim, head_dim].

        Returns:
            Tuple of (dq_b, dk_b, dv_b, dg_b, dgk_b, dgv_b, dh_initial).
        """

        dh = dfinal_h

        def backward_step(carry, inputs):
            """Compute one reverse-time backward step for recurrent attention.

            Computes gradients dq, dk, dv, dg, dgk, dgv for a single timestep
            and propagates dh to the previous step through the decay chain.

            Args:
                carry: Gradient w.r.t. hidden state from future steps
                    [num_heads, head_dim, head_dim].
                inputs: Tuple of (t_idx, q_t, k_t, v_t, g_t, gk_t, gv_t, h_t, hprev_t, do_t)
                    for the current (reversed) timestep.

            Returns:
                Tuple of (dh_prev, (dq_t, dk_t, dv_t, dg_t, dgk_t, dgv_t)).
            """
            dh_next = carry
            _t_idx, q_t, k_t, v_t, g_t, gk_t, gv_t, h_t, hprev_t, do_t = inputs

            dq_t = jnp.sum(do_t[:, None, :] * h_t, axis=-1) * softmax_scale

            dh_from_output = do_t[:, None, :] * (q_t * softmax_scale)[:, :, None]
            dh_current = dh_next + dh_from_output

            dk_t = jnp.einsum("nkv,nv->nk", dh_current, v_t)

            dv_t = jnp.einsum("nkv,nk->nv", dh_current, k_t)

            # Forward gate chain (gates are zero-filled when inactive, so
            # exp(0) == 1 keeps every stage an exact no-op):
            #   h1 = h_{t-1} * exp(g_t)
            #   h2 = h1 * exp(g_gamma)
            #   h3 = h2 * exp(gk_t)
            #   h4 = h3 * exp(gv_t)
            #   h_t = h4 + k_t ⊗ v_t
            # Backward applies the transposed chain in reverse order.
            h1 = hprev_t * jnp.exp(g_t)[:, :, None]
            h2 = h1 * jnp.exp(g_gamma_b)[:, None, None]
            h3 = h2 * jnp.exp(gk_t)[:, :, None]
            h4 = h3 * jnp.exp(gv_t)[:, None, :]

            dgv_t = jnp.sum(dh_current * h4, axis=1)
            dh3 = dh_current * jnp.exp(gv_t)[:, None, :]

            dgk_t = jnp.sum(dh3 * h3, axis=-1)
            dh2 = dh3 * jnp.exp(gk_t)[:, :, None]

            dh1 = dh2 * jnp.exp(g_gamma_b)[:, None, None]
            dg_t = jnp.sum(dh1 * h1, axis=-1)
            dh_prev = dh1 * jnp.exp(g_t)[:, :, None]

            outputs = (dq_t, dk_t, dv_t, dg_t, dgk_t, dgv_t)
            return dh_prev, outputs

        scan_inputs = (
            jnp.arange(seq_len)[::-1],
            q_b[::-1],
            k_b[::-1],
            v_b[::-1],
            g_b[::-1],
            gk_b[::-1],
            gv_b[::-1],
            hidden_b[::-1],
            hprev_b[::-1],
            do_b[::-1],
        )

        dh_initial, outputs = jax.lax.scan(backward_step, dh, scan_inputs)

        dq_b, dk_b, dv_b, dg_b, dgk_b, dgv_b = outputs
        dq_b = dq_b[::-1]
        dk_b = dk_b[::-1]
        dv_b = dv_b[::-1]
        dg_b = dg_b[::-1]
        dgk_b = dgk_b[::-1]
        dgv_b = dgv_b[::-1]

        return dq_b, dk_b, dv_b, dg_b, dgk_b, dgv_b, dh_initial

    dq, dk, dv, dg, dgk, dgv, dinitial_state = jax.vmap(process_batch)(
        q, k, v, g, g_gamma_batch, gk, gv, hidden_states, h_prev_states, do, dfinal_state
    )

    if reverse:
        dq = jnp.flip(dq, axis=1)
        dk = jnp.flip(dk, axis=1)
        dv = jnp.flip(dv, axis=1)
        dg = jnp.flip(dg, axis=1)
        dgk = jnp.flip(dgk, axis=1)
        dgv = jnp.flip(dgv, axis=1)

    return dq, dk, dv, dg, dgk, dgv, dinitial_state
