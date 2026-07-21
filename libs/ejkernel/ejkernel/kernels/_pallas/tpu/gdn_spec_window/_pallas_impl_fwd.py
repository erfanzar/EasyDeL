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

"""TPU Pallas implementation of the GDN speculative-window state scan.

The XLA reference expresses the window as ``num_steps`` sequential
einsum/elementwise groups, each of which round-trips the full recurrent state
pool through HBM — the dominant cost of a speculative verify forward on GDN
hybrids. This kernel keeps one ``(key_dim, value_dim)`` state tile resident in
VMEM per flattened ``(batch, value_head)`` program, walks all window steps
in-register, and streams out only the per-step state stack, so the pool is
read once and each candidate row written once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Bool, Float

from ...._xla.gdn_spec_window._xla_impl_fwd import _prep_window_inputs


def _gdn_spec_window_kernel(
    key_ref,
    value_ref,
    beta_ref,
    gate_ref,
    valid_ref,
    state_ref,
    out_ref,
):
    """Pallas kernel body: one flattened ``(batch, value_head)`` state tile, all steps.

    Runs on a ``(batch * num_v_heads,)`` grid. Each program instance loads its
    ``[key_dim, value_dim]`` float32 state tile once, applies the gated
    delta-rule recurrence for every window step (skipping invalid steps by
    carrying the state unchanged), and writes the post-step state into the
    step-major output stack.

    Args:
        key_ref: ``[1, num_steps, key_dim]`` L2-normalised keys (repeated to
            value heads by the caller). Read only.
        value_ref: ``[1, num_steps, value_dim]`` values. Read only.
        beta_ref: ``[1, num_steps, 1]`` float32 sigmoid betas. Read only.
        gate_ref: ``[1, num_steps, 1]`` float32 multiplicative decays
            (``exp(g)``). Read only.
        valid_ref: ``[1, num_steps, 1]`` int32 step-validity mask. Read only.
        state_ref: ``[1, key_dim, value_dim]`` window-entry state. Read only.
        out_ref: ``[num_steps, 1, key_dim, value_dim]`` output stack,
            written once per step.
    """
    num_steps = key_ref.shape[1]
    state = state_ref[0, :, :].astype(jnp.float32)
    for t in range(num_steps):
        key_t = key_ref[0, t, :].astype(jnp.float32)
        value_t = value_ref[0, t, :].astype(jnp.float32)
        beta_t = beta_ref[0, t, 0]
        gate_t = gate_ref[0, t, 0]
        valid_t = valid_ref[0, t, 0]
        k_mem = jax.lax.dot_general(
            key_t[None, :],
            state,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[0]
        v_new = beta_t * (value_t - gate_t * k_mem)
        state_new = state * gate_t + key_t[:, None] * v_new[None, :]
        state = jnp.where(valid_t > 0, state_new, state)
        out_ref[t, 0, :, :] = state.astype(out_ref.dtype)


def gdn_spec_window_states(
    mixed_qkv: Float[Array, "batch num_steps qkv_dim"],
    b: Float[Array, "batch num_steps num_v_heads"],
    a: Float[Array, "batch num_steps num_v_heads"],
    A_log: Float[Array, "num_v_heads"],
    dt_bias: Float[Array, "num_v_heads"],
    step_valid: Bool[Array, "batch num_steps"],
    recurrent_state: Float[Array, "batch num_v_heads key_dim value_dim"],
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    apply_silu: bool = False,
) -> Float[Array, "num_steps batch num_v_heads key_dim value_dim"]:
    """Pallas-accelerated GDN speculative-window state scan for TPU.

    Prepares the per-step ``(key, value, beta, gate)`` tensors with the shared
    XLA prep (bit-identical inputs to the reference), flattens ``(batch,
    num_v_heads)`` into the grid axis, and launches one fused kernel that
    advances the recurrence entirely in VMEM and emits the step-major state
    stack.

    Args:
        mixed_qkv: Post-conv packed rows ``[batch, num_steps, qkv_dim]``.
        b: Raw beta gate rows ``[batch, num_steps, n_v]``.
        a: Raw alpha gate rows ``[batch, num_steps, n_v]``.
        A_log: Per-value-head log-decay parameter ``[n_v]``.
        dt_bias: Per-value-head delta-time bias ``[n_v]``.
        step_valid: ``[batch, num_steps]`` per-step validity mask.
        recurrent_state: Window-entry state ``[batch, n_v, d_k, d_v]``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key/query head dim.
        d_v: Value head dim.
        apply_silu: Apply SiLU to ``mixed_qkv`` before slicing.

    Returns:
        ``[num_steps, batch, n_v, d_k, d_v]`` per-prefix recurrent states in
        ``recurrent_state.dtype``.
    """
    key, value, beta, gate = _prep_window_inputs(
        mixed_qkv, b, a, A_log, dt_bias, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, apply_silu=apply_silu
    )
    batch, num_steps = mixed_qkv.shape[:2]
    bh = batch * n_v

    key_f = key.transpose(0, 2, 1, 3).reshape(bh, num_steps, d_k)
    value_f = value.transpose(0, 2, 1, 3).reshape(bh, num_steps, d_v)
    beta_f = beta.transpose(0, 2, 1).reshape(bh, num_steps, 1)
    gate_f = gate.transpose(0, 2, 1).reshape(bh, num_steps, 1)
    valid_f = (
        jnp.broadcast_to(step_valid[:, None, :], (batch, n_v, num_steps))
        .reshape(bh, num_steps, 1)
        .astype(jnp.int32)
    )
    state_f = recurrent_state.reshape(bh, d_k, d_v)

    out_shape = jax.ShapeDtypeStruct((num_steps, bh, d_k, d_v), recurrent_state.dtype)
    stacked = pl.pallas_call(
        _gdn_spec_window_kernel,
        grid=(bh,),
        in_specs=[
            pl.BlockSpec((1, num_steps, d_k), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, num_steps, d_v), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, num_steps, 1), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, num_steps, 1), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, num_steps, 1), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, d_k, d_v), lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((num_steps, 1, d_k, d_v), lambda i: (0, i, 0, 0)),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(key_f, value_f, beta_f, gate_f, valid_f, state_f)
    return stacked.reshape(num_steps, batch, n_v, d_k, d_v)
