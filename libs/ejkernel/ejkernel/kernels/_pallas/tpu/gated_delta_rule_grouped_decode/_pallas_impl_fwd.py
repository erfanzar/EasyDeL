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

"""TPU Pallas implementation for grouped GDR single-step decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float


def _gated_delta_rule_grouped_decode_kernel(
    query_ref,
    key_ref,
    value_ref,
    beta_ref,
    decay_ref,
    state_ref,
    output_ref,
    state_out_ref,
):
    """Pallas kernel body for a single grouped GDR (gated delta-rule) decode step.

    This kernel runs on TPU inside a ``pallas_call`` with grid ``(batch,)``.
    Each program instance processes one batch element entirely in VMEM,
    iterating over all key heads ``kh`` and, for each, over the
    ``expand_ratio`` value heads it expands into (value head index
    ``kh * expand_ratio + e``).

    Per (key head, expansion) pair the recurrence applies log-space decay to
    the previous state, computes the delta-rule update
    ``delta = (value - k @ state) * beta``, rebuilds the state column-by-column
    as ``state + outer(k, delta)`` (Mosaic lacks scatter, so columns are
    stacked), and reads out ``output = (scale * q) @ state``. The query is
    scaled by ``head_dim ** -0.5``. Matmuls use float32 accumulation as
    required by the TPU matrix unit; the working dtype is the state ref dtype.

    Args:
        query_ref: Input ref of shape ``[1, num_k_heads, head_dim]``. Read only.
        key_ref: Input ref of shape ``[1, num_k_heads, head_dim]``. Read only.
        value_ref: Input ref of shape
            ``[1, num_k_heads, expand_ratio, value_dim]``. Read only.
        beta_ref: Input ref of gating coefficients of shape
            ``[1, num_k_heads, expand_ratio]``. Read only.
        decay_ref: Input ref of log-space decay factors of shape
            ``[1, num_k_heads, expand_ratio]``. Read only.
        state_ref: Input ref of the incoming recurrent state of shape
            ``[1, num_v_heads, head_dim, value_dim]``. Read only.
        output_ref: Output ref written with the decode output of shape
            ``[1, num_v_heads, value_dim]``.
        state_out_ref: Output ref written with the updated recurrent state of
            shape ``[1, num_v_heads, head_dim, value_dim]``.

    Returns:
        None. Results are written in place to ``output_ref`` and
        ``state_out_ref``.
    """
    num_k_heads = query_ref.shape[1]
    head_dim = query_ref.shape[2]
    expand_ratio = value_ref.shape[2]
    compute_dtype = state_ref.dtype
    scale = jnp.asarray(head_dim**-0.5, dtype=compute_dtype)

    for kh in range(num_k_heads):
        q = query_ref[0, kh, :].astype(compute_dtype) * scale
        k = key_ref[0, kh, :].astype(compute_dtype)

        vh_start = kh * expand_ratio
        s_all = state_ref[0, vh_start : vh_start + expand_ratio, :, :].astype(compute_dtype)
        d_all = decay_ref[0, kh, :].astype(jnp.float32)
        beta_all = beta_ref[0, kh, :].astype(jnp.float32)
        v_all = value_ref[0, kh, :, :].astype(compute_dtype)

        for e in range(expand_ratio):
            s_e = s_all[e, :, :]
            v_e = v_all[e, :]
            beta_e = beta_all[e].astype(compute_dtype)
            decay_e = d_all[e]

            s_e = s_e * jnp.exp(decay_e).astype(compute_dtype)
            # Rank-2 dot with fp32 accumulation (required by the TPU matmul unit).
            kv_mem = jax.lax.dot_general(
                k[None, :],
                s_e,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )[0].astype(compute_dtype)
            delta_e = (v_e - kv_mem) * beta_e
            # Mosaic can only squeeze 32-bit arrays to scalars, and does not
            # support scatter updates, so rebuild the updated state by stacking
            # per-column rank-1 updates.
            delta_e_fp32 = delta_e.astype(jnp.float32)
            updated_cols = []
            for v in range(s_e.shape[1]):
                updated_cols.append(s_e[:, v] + k * delta_e_fp32[v].astype(compute_dtype))
            s_e = jnp.stack(updated_cols, axis=1)
            out_e = jax.lax.dot_general(
                q[None, :],
                s_e,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )[0].astype(compute_dtype)

            state_out_ref[0, vh_start + e, :, :] = s_e.astype(state_out_ref.dtype)
            output_ref[0, vh_start + e, :] = out_e.astype(output_ref.dtype)


def gated_delta_rule_grouped_decode(
    query: Float[Array, "batch num_k_heads head_dim"],
    key: Float[Array, "batch num_k_heads head_dim"],
    value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
    beta: Float[Array, "batch num_k_heads expand_ratio"],
    decay: Float[Array, "batch num_k_heads expand_ratio"],
    recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
) -> tuple[
    Float[Array, "batch num_v_heads value_dim"],
    Float[Array, "batch num_v_heads head_dim value_dim"],
]:
    """Pallas-accelerated grouped GDR (gated delta-rule) decode step targeting TPU.

    Builds and launches the ``pallas_call`` over a ``(batch,)`` grid, with the
    value-head count derived as ``num_v_heads = num_k_heads * expand_ratio``.
    Outputs and the updated state are allocated with the dtype of
    ``recurrent_state``.

    Args:
        query: Query tensor ``[batch, num_k_heads, head_dim]``.
        key: Key tensor ``[batch, num_k_heads, head_dim]``.
        value: Value tensor ``[batch, num_k_heads, expand_ratio, value_dim]``.
        beta: Gating coefficients ``[batch, num_k_heads, expand_ratio]``.
        decay: Log-space decay factors ``[batch, num_k_heads, expand_ratio]``.
        recurrent_state: Current recurrent state
            ``[batch, num_v_heads, head_dim, value_dim]``.

    Returns:
        A tuple ``(output, new_state)`` where ``output`` is the decode output
        ``[batch, num_v_heads, value_dim]`` and ``new_state`` is the updated
        recurrent state ``[batch, num_v_heads, head_dim, value_dim]``, both with
        the dtype of ``recurrent_state``.
    """
    batch, num_k_heads, head_dim = query.shape
    expand_ratio = value.shape[2]
    value_dim = value.shape[3]
    num_v_heads = num_k_heads * expand_ratio

    out_output = jax.ShapeDtypeStruct((batch, num_v_heads, value_dim), dtype=recurrent_state.dtype)
    out_state = jax.ShapeDtypeStruct((batch, num_v_heads, head_dim, value_dim), dtype=recurrent_state.dtype)

    output, new_state = pl.pallas_call(
        _gated_delta_rule_grouped_decode_kernel,
        grid=(batch,),
        in_specs=[
            pl.BlockSpec((1, num_k_heads, head_dim), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, num_k_heads, head_dim), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, num_k_heads, expand_ratio, value_dim), lambda b: (b, 0, 0, 0)),
            pl.BlockSpec((1, num_k_heads, expand_ratio), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, num_k_heads, expand_ratio), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, num_v_heads, head_dim, value_dim), lambda b: (b, 0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, num_v_heads, value_dim), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, num_v_heads, head_dim, value_dim), lambda b: (b, 0, 0, 0)),
        ],
        out_shape=[out_output, out_state],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(query, key, value, beta, decay, recurrent_state)

    return output, new_state
