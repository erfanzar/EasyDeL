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

"""Pallas TPU kernel: one per-channel KDA decode token per row, updating the recurrent-state pool in place.

Row ``r`` owns pool slot ``r`` (identity layout). The grid walks the rows in blocks of ``block_rows``; each step
DMAs those rows' ``[H, K, V]`` states into VMEM (pipelined by Pallas), advances the rows that take a token in fp32,
and writes the block back to the same slots (the pool is an input/output alias, so nothing else is copied). Rows that
do not advance write their state back unchanged (bit for bit) and output zeros. With explicit ``state_indices`` the
rows' states are gathered first and the advanced ones scattered back (rows that do not advance drop their write).

Per row and head, with ``q`` / ``k`` L2-normalised in their input dtype and ``q`` scaled by ``K**-0.5`` (the same
XLA ops as the ``ragged_kda`` decode phase, so these inputs are bit-identical to the reference)::

    S = S * exp(g)[:, None];   delta = beta * (v - k^T S);   S = S + k delta^T;   o = q^T S

The products and reductions run on the VPU in fp32 (``reduce="vpu"``, the reference arithmetic; only the summation
order differs), or as batched MXU matvecs at ``HIGHEST`` precision (``reduce="mxu"``).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ...._xla.kda_chunk._xla_impl import l2norm

REDUCE_MODES = ("vpu", "mxu")
# MXU matvecs at HIGHEST precision: measured 0.161 ms vs 0.220 ms (VPU) per 128-row, 16-head bf16-pool call on v5p,
# outputs within 1.2e-7 relative of the VPU / XLA reference arithmetic.
DEFAULT_REDUCE = "mxu"


def supported(head_dim: int, v_dim: int) -> bool:
    """Whether the kernel tiles these head sizes (both multiples of the 128 TPU lanes)."""
    return int(head_dim) % 128 == 0 and int(v_dim) % 128 == 0


def pick_block_rows(num_rows: int, requested: int = 0) -> int:
    """Rows per grid step: ``requested`` when given (must divide ``num_rows``), else 2 when it divides, else 1."""
    if requested:
        if num_rows % requested:
            raise ValueError(f"block_rows={requested} must divide the row count {num_rows}")
        return int(requested)
    return 2 if num_rows % 2 == 0 else 1


def pick_block_h(num_heads: int, requested: int = 0) -> int:
    """Heads per inner step: ``requested`` when given (must divide ``num_heads``), else 8 when it divides, else all."""
    if requested:
        if num_heads % requested:
            raise ValueError(f"block_h={requested} must divide the head count {num_heads}")
        return int(requested)
    return 8 if num_heads % 8 == 0 else int(num_heads)


def _vmem_limit_bytes() -> int:
    try:
        capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # pragma: no cover - only reached off TPU
        capacity = 64 * 1024 * 1024
    return int(capacity * 0.75)


def _decode_kernel(
    valid_ref,
    init_ref,
    q_ref,
    k_ref,
    v_ref,
    b_ref,
    eg_ref,
    state_ref,
    o_ref,
    new_state_ref,
    *,
    block_rows: int,
    num_heads: int,
    block_h: int,
    reduce: str,
):
    """Advance ``block_rows`` rows; blocks: q/k/eg ``[br, H, K]``, v/b/o ``[br, H, V]``, states ``[br, H, K, V]``."""
    step = pl.program_id(0)
    f32 = jnp.float32
    head_dim = q_ref.shape[-1]
    v_dim = v_ref.shape[-1]

    def contract(vec, s):
        # sum_k vec[h, k] * s[h, k, v] -> [h, v]
        if reduce == "mxu":
            return lax.dot_general(
                vec.reshape(block_h, 1, head_dim),
                s,
                (((2,), (1,)), ((0,), (0,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=f32,
            ).reshape(block_h, v_dim)
        return jnp.sum(s * vec[:, :, None], axis=1)

    def advance(r: int, fresh: bool):
        for h0 in range(0, num_heads, block_h):
            hs = slice(h0, h0 + block_h)
            k = k_ref[r, hs, :]
            if fresh:
                s = jnp.zeros((block_h, head_dim, v_dim), f32)
            else:
                s = state_ref[r, hs, :, :].astype(f32) * eg_ref[r, hs, :][:, :, None]
            delta = (v_ref[r, hs, :] - contract(k, s)) * b_ref[r, hs, :]
            s = s + k[:, :, None] * delta[:, None, :]
            o_ref[r, hs, :] = contract(q_ref[r, hs, :], s).astype(o_ref.dtype)
            new_state_ref[r, hs, :, :] = s.astype(new_state_ref.dtype)

    for r in range(block_rows):
        row = step * block_rows + r
        live = valid_ref[row] != 0
        keep = init_ref[row] != 0

        @pl.when(jnp.logical_and(live, keep))
        def _carry(r=r):
            advance(r, fresh=False)

        @pl.when(jnp.logical_and(live, jnp.logical_not(keep)))
        def _fresh(r=r):
            advance(r, fresh=True)

        @pl.when(jnp.logical_not(live))
        def _hold(r=r):
            # The output block is its own VMEM buffer: copy the row's state through unchanged (bit for bit).
            for h0 in range(0, num_heads, block_h):
                hs = slice(h0, h0 + block_h)
                new_state_ref[r, hs, :, :] = state_ref[r, hs, :, :]
            o_ref[r] = jnp.zeros(o_ref.shape[1:], o_ref.dtype)


def ragged_kda_decode_pallas(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    recurrent_state: Array,
    state_indices: Array | None,
    valid: Array,
    has_initial_state: Array | None = None,
    *,
    use_qk_l2norm: bool = True,
    block_rows: int = 0,
    block_h: int = 0,
    reduce: str = DEFAULT_REDUCE,
) -> tuple[Array, Array]:
    """One KDA decode token per row over a recurrent-state pool (Pallas TPU; pool updated in place).

    Args:
        query: Post-conv queries ``[R, H, K]``.
        key: Post-conv keys ``[R, H, K]``.
        value: Post-conv values ``[R, H, V]``.
        beta: Write strengths ``[R, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[R, H, K]``.
        recurrent_state: State pool ``[S, H, K, V]`` (fp32 / bf16 / fp16); the step runs in fp32.
        state_indices: Pool slot of every row ``[R]`` (out-of-range: the row does not advance), or ``None`` (row
            ``r`` owns slot ``r``: the kernel reads and writes the leading ``R`` slots in place).
        valid: ``[R]`` rows that advance.
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state.
        use_qk_l2norm: L2-normalise queries and keys.
        block_rows: Rows per grid step (``0``: automatic).
        block_h: Heads per inner step (``0``: automatic).
        reduce: ``"vpu"`` or ``"mxu"`` contraction of the state with ``k`` and ``q``.

    Returns:
        ``(out [R, H, V] fp32, new_recurrent_state [S, H, K, V])``; rows that do not advance output zeros.
    """
    if reduce not in REDUCE_MODES:
        raise ValueError(f"reduce must be one of {REDUCE_MODES}, got {reduce!r}")
    num_rows, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    num_slots = recurrent_state.shape[0]
    f32 = jnp.float32
    # Same XLA ops as the reference decode step: these inputs are bit-identical to it.
    if use_qk_l2norm:
        query = l2norm(query)
        key = l2norm(key)
    q = query.astype(f32) * (1.0 / (head_dim**0.5))
    k = key.astype(f32)
    v = value.astype(f32)
    eg = jnp.exp(decay.astype(f32))
    b = jnp.broadcast_to(beta.astype(f32)[:, :, None], (num_rows, num_heads, v_dim))
    live = jnp.asarray(valid).astype(jnp.bool_)
    if state_indices is None:
        pool = recurrent_state
    else:
        slots = jnp.asarray(state_indices).astype(jnp.int32)
        live = live & (slots >= 0) & (slots < num_slots)
        pool = recurrent_state[jnp.clip(slots, 0, num_slots - 1)]
    if has_initial_state is None:
        init = jnp.ones((num_rows,), jnp.int32)
    else:
        init = jnp.asarray(has_initial_state).astype(jnp.int32)

    br = pick_block_rows(num_rows, block_rows)
    bh = pick_block_h(num_heads, block_h)
    tok_spec = pl.BlockSpec((br, num_heads, head_dim), lambda i, *_: (i, 0, 0))
    val_spec = pl.BlockSpec((br, num_heads, v_dim), lambda i, *_: (i, 0, 0))
    state_spec = pl.BlockSpec((br, num_heads, head_dim, v_dim), lambda i, *_: (i, 0, 0, 0))
    out, new_pool = pl.pallas_call(
        functools.partial(_decode_kernel, block_rows=br, num_heads=num_heads, block_h=bh, reduce=reduce),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(num_rows // br,),
            in_specs=[tok_spec, tok_spec, val_spec, val_spec, tok_spec, state_spec],
            out_specs=[val_spec, state_spec],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_rows, num_heads, v_dim), f32),
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        ],
        # operands: valid, init, q, k, v, b, eg, pool -> the pool (7) is updated in place as output 1.
        input_output_aliases={7: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_vmem_limit_bytes(),
        ),
        name=f"ejkernel_ragged_kda_decode_br{br}_bh{bh}_{reduce}",
    )(live.astype(jnp.int32), init, q, k, v, b, eg, pool)
    if state_indices is not None:
        new_pool = recurrent_state.at[jnp.where(live, slots, num_slots)].set(new_pool, mode="drop")
    return out, new_pool


__all__ = ("REDUCE_MODES", "pick_block_h", "pick_block_rows", "ragged_kda_decode_pallas", "supported")
