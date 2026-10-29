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

"""Pallas TPU kernels for ragged gated delta rule decode.

Uses a ``(num_tokens,)`` grid where each program processes one token
across all heads. States are pre-gathered from the global pool before
the kernel call and scattered back afterward.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_F32 = jnp.float32


def _gdr_decode_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    state_ref,
    out_ref,
    state_out_ref,
    *,
    num_heads: int,
):
    """Pallas kernel for single-token ragged GDR decode.

    Grid: ``(num_tokens,)``. Each program processes one token across
    all heads via a static for loop.

    Args:
        q_ref: Query block ``(1, H, D_K)``.
        k_ref: Key block ``(1, H, D_K)``.
        v_ref: Value block ``(1, H, D_V)``.
        beta_ref: Gating ``(1, H, 1)``.
        decay_ref: Decay ``(1, H, 1)``.
        state_ref: State block ``(1, H, D_K, D_V)``.
        out_ref: Output ``(1, H, D_V)``.
        state_out_ref: Updated state ``(1, H, D_K, D_V)``.
        num_heads: Number of attention heads.
    """
    D_K = q_ref.shape[2]
    scale = D_K**-0.5

    for h in range(num_heads):
        q = q_ref[0, h, :].astype(_F32) * scale
        k = k_ref[0, h, :].astype(_F32)
        v = v_ref[0, h, :].astype(_F32)
        beta_val = beta_ref[0, h, 0].astype(_F32)
        g_val = decay_ref[0, h, 0].astype(_F32)
        g_exp = jnp.exp(jnp.clip(g_val, -20.0, 20.0))

        state_h = state_ref[0, h].astype(_F32)

        k_state = jnp.sum(state_h * k[:, None], axis=0)
        v_diff = v - g_exp * k_state
        v_new = beta_val * v_diff

        q_state = jnp.sum(state_h * q[:, None], axis=0)
        q_k = jnp.sum(q * k)
        output_h = g_exp * q_state + q_k * v_new

        new_state_h = state_h * g_exp + k[:, None] * v_new[None, :]

        out_ref[0, h, :] = output_h.astype(out_ref.dtype)
        state_out_ref[0, h, :, :] = new_state_h.astype(state_out_ref.dtype)


def run_ragged_gdr_decode_pallas(query, key, value, beta, decay, gathered_state, *, use_l2norm=False):
    """Launch the Pallas decode kernel on pre-gathered per-token states.

    Args:
        query: Queries ``(num_tokens, H, D_K)``.
        key: Keys ``(num_tokens, H, D_K)``.
        value: Values ``(num_tokens, H, D_V)``.
        beta: Gating ``(num_tokens, H)``.
        decay: Decay ``(num_tokens, H)``.
        gathered_state: Per-token states ``(num_tokens, H, D_K, D_V)``.
        use_l2norm: Unused (kept for API compat).

    Returns:
        ``(output, updated_state)`` with shapes
        ``(num_tokens, H, D_V)`` and ``(num_tokens, H, D_K, D_V)``.
    """
    num_tokens, H, D_K = query.shape
    D_V = value.shape[2]

    kernel_fn = functools.partial(_gdr_decode_kernel, num_heads=H)

    call = pl.pallas_call(
        kernel_fn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, H, D_K), lambda t: (t, 0, 0)),
                pl.BlockSpec((1, H, D_K), lambda t: (t, 0, 0)),
                pl.BlockSpec((1, H, D_V), lambda t: (t, 0, 0)),
                pl.BlockSpec((1, H, 1), lambda t: (t, 0, 0)),
                pl.BlockSpec((1, H, 1), lambda t: (t, 0, 0)),
                pl.BlockSpec((1, H, D_K, D_V), lambda t: (t, 0, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, H, D_V), lambda t: (t, 0, 0)),
                pl.BlockSpec((1, H, D_K, D_V), lambda t: (t, 0, 0, 0)),
            ],
            grid=(num_tokens,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, H, D_V), query.dtype),
            jax.ShapeDtypeStruct((num_tokens, H, D_K, D_V), gathered_state.dtype),
        ],
        input_output_aliases={5: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
        ),
    )

    beta_3d = beta[:, :, None].astype(_F32)
    decay_3d = decay[:, :, None].astype(_F32)
    return call(query, key, value, beta_3d, decay_3d, gathered_state)
