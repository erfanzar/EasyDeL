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


"""Pallas TPU decode MoE segment in one kernel (``moe_decode_fused_w8a8``).

One ``pallas_call`` runs, for a few decode token rows, everything an expert shard does between the router's top-k and
the MoE output reduction:

* it lists the shard's active local experts (scalar pass over the ``[tokens * k]`` expert ids in SMEM),
* quantizes the token rows once (per-row int8, ejkernel ``quantize_rows`` rule),
* streams each active expert's int8 gate, up and down slabs HBM -> VMEM through a ring of ``nbuf`` slots with
  ``nbuf - 1`` experts of prefetch,
* computes gate/up (int8 x int8 -> int32, ``(acc * row_scale) * channel_scale``, rounded to the rows' dtype), the
  two-branch combine, the per-row int8 quantization of the hidden, the down projection (same epilogue, rounded to the
  rows' dtype) for every token row, and adds each token's rows weighted by its combine weight for that expert (zero
  for tokens not routed there) into a float32 ``[tokens, H]`` sum,
* and optionally the folded W8A8 shared expert on this shard's own token rows (float32 output, as
  ``shared_expert_w8a8``).

Every expert runs on all token rows (decode: the MXU pads the rows anyway), so there is no row gather, no sort, no
group sizes and no dispatch tail. The result is APPROXIMATE against the XLA dispatch: the expert sum order and the
in-kernel f32 lowering of divide/silu differ. Tiles and the ring depth come from the VMEM capacity.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..shared_expert_w8a8._pallas_impl import _codes

F32 = jnp.float32
I32 = jnp.int32
_LANES = 128
_INT8_SUBLANES = 32
_MAX_NBUF = 8

#: Token-row bound of the kernel (decode buckets; larger steps belong to the XLA dispatch).
MAX_TOKENS = 256


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _tpu_info() -> int:
    """VMEM bytes of one TensorCore (16 MiB when unknown)."""
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # no TPU info on this backend
        return 16 << 20


def _plan(t: int, h: int, m: int, e: int, inter: int, rows_itemsize: int, vmem: int) -> tuple[int, int] | None:
    """``(nbuf, vmem bytes)``: the deepest weight ring (2..8 slots) whose working set fits 5/8 of VMEM, or ``None``."""
    tp = _round_up(t, _INT8_SUBLANES)
    rows8 = _round_up(t, 8)
    blocks = rows8 * h * rows_itemsize + 2 * rows8 * _LANES * 4 + (2 * _round_up(e, 8) * m + _round_up(e, 8) * h) * 4
    blocks += rows8 * h * 4
    fixed = 2 * blocks + tp * h * 5 + rows8 * _LANES * 4 + tp * m * 4 + 4 * tp * h * 4 + 8 * tp * m * 4
    if inter:
        fixed += 3 * h * inter + 2 * 8 * (2 * inter + h) * 4 + tp * inter * 4 + 6 * tp * inter * 4
    per = 3 * h * m
    for nbuf in range(_MAX_NBUF, 1, -1):
        need = fixed + nbuf * per
        if need <= vmem * 5 // 8:
            return nbuf, need
    return None


def unsupported_reason(
    x,
    expert_ids,
    combine_weights,
    gate_codes,
    gate_scales,
    up_codes,
    up_scales,
    down_codes,
    down_scales,
    control,
    shared_gate_up_codes=None,
    shared_gate_up_scales=None,
    shared_down_codes=None,
    shared_down_scales=None,
) -> str | None:
    """Why the Pallas kernel cannot take these operands (``None``: supported)."""
    if x.ndim != 2 or not jnp.issubdtype(x.dtype, jnp.floating):
        return "x must be a floating [tokens, H] array"
    t, h = x.shape
    if t < 1 or t > MAX_TOKENS or t % 8:
        return f"{t} token rows (supported: multiples of 8 up to {MAX_TOKENS})"
    if expert_ids.ndim != 2 or expert_ids.shape[0] != t or combine_weights.shape != expert_ids.shape:
        return "expert_ids and combine_weights must be [tokens, k]"
    if not jnp.issubdtype(expert_ids.dtype, jnp.integer):
        return "expert_ids must be integer"
    k = expert_ids.shape[1]
    if k > _LANES:
        return f"top-k {k} above {_LANES}"
    if any(c.ndim != 3 or c.dtype != jnp.int8 for c in (gate_codes, up_codes, down_codes)):
        return "expert codes must be int8 [E, K, N]"
    e, kh, m = gate_codes.shape
    if kh != h or up_codes.shape != (e, h, m) or down_codes.shape != (e, m, h):
        return f"expected gate/up [E, H, M] and down [E, M, H], got {gate_codes.shape}, {up_codes.shape}, " + str(
            down_codes.shape
        )
    if gate_scales.size != e * m or up_scales.size != e * m or down_scales.size != e * h:
        return "scales must hold one value per expert and output channel"
    if h % _LANES or m % _LANES:
        return f"H ({h}) and M ({m}) must be multiples of {_LANES}"
    if control.shape != (3,):
        return "control must be int32 [3] (expert offset, shared row start, shared row end)"
    shared = (shared_gate_up_codes, shared_gate_up_scales, shared_down_codes, shared_down_scales)
    inter = 0
    if any(s is not None for s in shared):
        if any(s is None for s in shared):
            return "the shared expert needs all four operands"
        if shared_gate_up_codes.dtype != jnp.int8 or shared_down_codes.dtype != jnp.int8:
            return "shared codes must be int8"
        inter = shared_down_codes.shape[0]
        if shared_gate_up_codes.shape != (h, 2 * inter) or shared_down_codes.shape != (inter, h):
            return "expected shared gate_up [H, 2I] and down [I, H]"
        if shared_gate_up_scales.size != 2 * inter or shared_down_scales.size != h:
            return "shared scales must hold one value per output channel"
        if inter % _LANES:
            return f"shared I ({inter}) must be a multiple of {_LANES}"
    if _plan(t, h, m, e, inter, jnp.dtype(x.dtype).itemsize, _tpu_info()) is None:
        return "the weight ring does not fit VMEM"
    return None


def _make_kernel(
    *,
    t: int,
    e: int,
    m: int,
    inter: int,
    nbuf: int,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
    shared_combine: Callable[[jax.Array, jax.Array], jax.Array] | None,
    rows_dtype,
    n_pairs: int,
):
    """Build the kernel body for one static configuration (see the module docstring)."""
    tp = _round_up(t, _INT8_SUBLANES)
    has_shared = shared_combine is not None

    def kernel(pairs_ref, ctrl_ref, x_ref, ids_ref, w_ref, gs_ref, us_ref, ds_ref, g_hbm, u_hbm, d_hbm, *rest):
        if has_shared:
            sgu_hbm, sgus_ref, sdn_hbm, sdns_ref, o_ref, *scratch = rest
        else:
            o_ref, *scratch = rest
        gbuf, ubuf, dbuf, sem, flags, active, xpad, xq, xs_ref, hpad, *shared_scratch = scratch
        offset = ctrl_ref[0]

        if has_shared:
            sgu, sdn, ssem, spad = shared_scratch
            cp_sgu = pltpu.make_async_copy(sgu_hbm, sgu, ssem.at[0])
            cp_sdn = pltpu.make_async_copy(sdn_hbm, sdn, ssem.at[1])
            cp_sgu.start()
            cp_sdn.start()

        # Active local experts in ascending id order.
        def _zero(i, c):
            flags[i] = jnp.int32(0)
            return c

        lax.fori_loop(0, e, _zero, 0)

        def _mark(p, c):
            lid = pairs_ref[p] - offset

            @pl.when((lid >= 0) & (lid < e))
            def _():
                flags[lid] = jnp.int32(1)

            return c

        lax.fori_loop(0, n_pairs, _mark, 0)

        def _compact(i, n):
            f = flags[i]

            @pl.when(f != 0)
            def _():
                active[n] = i

            return n + f

        n_active = lax.fori_loop(0, e, _compact, jnp.int32(0))

        def _copies(i, slot):
            ex = active[i]
            return (
                pltpu.make_async_copy(g_hbm.at[ex], gbuf.at[slot], sem.at[0, slot]),
                pltpu.make_async_copy(u_hbm.at[ex], ubuf.at[slot], sem.at[1, slot]),
                pltpu.make_async_copy(d_hbm.at[ex], dbuf.at[slot], sem.at[2, slot]),
            )

        for j in range(nbuf - 1):

            @pl.when(j < n_active)
            def _(j=j):
                for cp in _copies(j, j):
                    cp.start()

        # Token rows -> int8 codes (rows padded to the int8 sublane tile with zero codes) and row scales.
        x32 = x_ref[...].astype(F32)
        codes, xs = _codes(x32, x32)
        if tp != t:
            xpad[...] = jnp.zeros(xpad.shape, F32)
            hpad[...] = jnp.zeros(hpad.shape, F32)
        xpad[pl.ds(0, t), :] = codes
        xq[...] = xpad[...].astype(I32).astype(jnp.int8)
        xs_ref[...] = xs

        if has_shared:
            cp_sgu.wait()
            cp_sdn.wait()
            if tp != t:
                spad[...] = jnp.zeros(spad.shape, F32)
            q = xq[...]
            gate = jnp.dot(q, sgu[:, pl.ds(0, inter)], preferred_element_type=I32)[:t]
            up = jnp.dot(q, sgu[:, pl.ds(inter, inter)], preferred_element_type=I32)[:t]
            gate = ((gate.astype(F32) * xs) * sgus_ref[:, pl.ds(0, inter)]).astype(rows_dtype).astype(F32)
            up = ((up.astype(F32) * xs) * sgus_ref[:, pl.ds(inter, inter)]).astype(rows_dtype).astype(F32)
            h32 = shared_combine(gate, up).astype(F32)
            hcodes, hs = _codes(h32.astype(rows_dtype).astype(F32), h32)
            spad[pl.ds(0, t), :] = hcodes
            hq = spad[...].astype(I32).astype(jnp.int8)
            out = jnp.dot(hq, sdn[...], preferred_element_type=I32)[:t]
            out = (out.astype(F32) * hs) * sdns_ref[...]
            row = lax.broadcasted_iota(I32, (t, 1), 0)
            own = (row >= ctrl_ref[1]) & (row < ctrl_ref[2])
            o_ref[...] = jnp.where(own, out, 0.0)
        else:
            o_ref[...] = jnp.zeros(o_ref.shape, F32)

        def _expert(i, c):
            slot = lax.rem(i, nbuf)
            for cp in _copies(i, slot):
                cp.wait()

            @pl.when(i + nbuf - 1 < n_active)
            def _():
                nxt = i + nbuf - 1
                for cp in _copies(nxt, lax.rem(nxt, nbuf)):
                    cp.start()

            ex = active[i]
            q = xq[...]
            s = xs_ref[...]
            gate = jnp.dot(q, gbuf[slot], preferred_element_type=I32)[:t]
            up = jnp.dot(q, ubuf[slot], preferred_element_type=I32)[:t]
            gate = ((gate.astype(F32) * s) * gs_ref[pl.ds(ex, 1), :]).astype(rows_dtype).astype(F32)
            up = ((up.astype(F32) * s) * us_ref[pl.ds(ex, 1), :]).astype(rows_dtype).astype(F32)
            hb = combine(gate, up).astype(rows_dtype).astype(F32)
            hcodes, hs = _codes(hb, hb)
            hpad[pl.ds(0, t), :] = hcodes
            hq = hpad[...].astype(I32).astype(jnp.int8)
            down = jnp.dot(hq, dbuf[slot], preferred_element_type=I32)[:t]
            down = ((down.astype(F32) * hs) * ds_ref[pl.ds(ex, 1), :]).astype(rows_dtype).astype(F32)
            wsel = jnp.sum(jnp.where(ids_ref[...] == ex + offset, w_ref[...], 0.0), axis=1, keepdims=True)
            o_ref[...] += wsel * down
            return c

        lax.fori_loop(0, n_active, _expert, 0)

    return kernel


def moe_decode_fused_w8a8_pallas(
    x: jax.Array,
    expert_ids: jax.Array,
    combine_weights: jax.Array,
    gate_codes: jax.Array,
    gate_scales: jax.Array,
    up_codes: jax.Array,
    up_scales: jax.Array,
    down_codes: jax.Array,
    down_scales: jax.Array,
    control: jax.Array,
    shared_gate_up_codes: jax.Array | None,
    shared_gate_up_scales: jax.Array | None,
    shared_down_codes: jax.Array | None,
    shared_down_scales: jax.Array | None,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
    shared_combine: Callable[[jax.Array, jax.Array], jax.Array] | None,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Run the fused decode MoE segment; returns float32 ``[tokens, H]`` (see the module docstring).

    Args:
        x: Token rows ``[T, H]`` (floating; the rows' dtype sets the intermediate roundings).
        expert_ids: Global expert id per (token, slot), ``[T, k]``.
        combine_weights: Combine weight per (token, slot), ``[T, k]``.
        gate_codes: int8 ``[E, H, M]`` of this shard's experts; ``gate_scales`` ``[E, 1, M]``.
        up_codes: int8 ``[E, H, M]``; ``up_scales`` ``[E, 1, M]``.
        down_codes: int8 ``[E, M, H]``; ``down_scales`` ``[E, 1, H]``.
        control: int32 ``[3]``: global id of local expert 0, shared-expert row start and end.
        shared_gate_up_codes: Optional shared expert int8 ``[H, 2I]`` (``[gate | up]``) and its scales ``[1, 2I]``.
        shared_down_codes: Optional shared expert int8 ``[I, H]`` and its scales ``[1, H]``.
        combine: Routed ``(gate, up) -> hidden`` combine (float32 in and out).
        shared_combine: Shared-expert combine, or ``None`` without a shared expert.
        interpret: Pallas interpret mode (CPU logic checks only).

    Returns:
        float32 ``[T, H]``: this shard's weighted expert rows summed per token plus its shared-expert rows.

    Raises:
        ValueError: Unsupported operands (see :func:`unsupported_reason`).
    """
    reason = unsupported_reason(
        x,
        expert_ids,
        combine_weights,
        gate_codes,
        gate_scales,
        up_codes,
        up_scales,
        down_codes,
        down_scales,
        control,
        shared_gate_up_codes,
        shared_gate_up_scales,
        shared_down_codes,
        shared_down_scales,
    )
    if reason is not None:
        raise ValueError(f"moe_decode_fused_w8a8 (Pallas TPU): {reason}")
    has_shared = shared_gate_up_codes is not None
    if has_shared != (shared_combine is not None):
        raise ValueError("shared_combine must be given exactly when the shared expert operands are")
    t, h = x.shape
    k = expert_ids.shape[1]
    e, _, m = gate_codes.shape
    inter = int(shared_down_codes.shape[0]) if has_shared else 0
    vmem = _tpu_info()
    nbuf, need = _plan(t, h, m, e, inter, jnp.dtype(x.dtype).itemsize, vmem)
    tp = _round_up(t, _INT8_SUBLANES)
    k_pad = _LANES
    ids = expert_ids.astype(I32)
    pairs = ids.reshape(-1)
    ids_p = jnp.pad(ids, ((0, 0), (0, k_pad - k)), constant_values=-1)
    w_p = jnp.pad(combine_weights.astype(F32), ((0, 0), (0, k_pad - k)))

    def full(shape):
        nd = len(shape)
        return pl.BlockSpec(tuple(shape), lambda i, *_: (0,) * nd)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    operands = [
        x,
        ids_p,
        w_p,
        gate_scales.reshape(e, m).astype(F32),
        up_scales.reshape(e, m).astype(F32),
        down_scales.reshape(e, h).astype(F32),
        gate_codes,
        up_codes,
        down_codes,
    ]
    in_specs = [full((t, h)), full((t, k_pad)), full((t, k_pad)), full((e, m)), full((e, m)), full((e, h))]
    in_specs += [any_spec, any_spec, any_spec]
    scratch = [
        pltpu.VMEM((nbuf, h, m), jnp.int8),
        pltpu.VMEM((nbuf, h, m), jnp.int8),
        pltpu.VMEM((nbuf, m, h), jnp.int8),
        pltpu.SemaphoreType.DMA((3, nbuf)),
        pltpu.SMEM((e,), I32),
        pltpu.SMEM((e,), I32),
        pltpu.VMEM((tp, h), F32),
        pltpu.VMEM((tp, h), jnp.int8),
        pltpu.VMEM((t, 1), F32),
        pltpu.VMEM((tp, m), F32),
    ]
    if has_shared:
        operands += [
            shared_gate_up_codes,
            shared_gate_up_scales.reshape(1, 2 * inter).astype(F32),
            shared_down_codes,
            shared_down_scales.reshape(1, h).astype(F32),
        ]
        in_specs += [any_spec, full((1, 2 * inter)), any_spec, full((1, h))]
        scratch += [
            pltpu.VMEM((h, 2 * inter), jnp.int8),
            pltpu.VMEM((inter, h), jnp.int8),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((tp, inter), F32),
        ]
    kernel = _make_kernel(
        t=t,
        e=e,
        m=m,
        inter=inter,
        nbuf=nbuf,
        combine=combine,
        shared_combine=shared_combine,
        rows_dtype=x.dtype,
        n_pairs=t * k,
    )
    limit = int(min(vmem * 15 // 16, max(32 << 20, need + (16 << 20))))
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=in_specs,
            out_specs=full((t, h)),
            scratch_shapes=scratch,
        ),
        out_shape=jax.ShapeDtypeStruct((t, h), F32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=limit),
        interpret=interpret,
        name="ejkernel_moe_decode_fused_w8a8",
    )(pairs, control.astype(I32), *operands)


__all__ = ("MAX_TOKENS", "moe_decode_fused_w8a8_pallas", "unsupported_reason")
