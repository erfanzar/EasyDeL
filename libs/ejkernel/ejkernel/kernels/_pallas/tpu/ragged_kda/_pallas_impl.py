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

"""Pallas TPU ``ragged_kda``: the multi-token chunk path of ragged KDA in one ``pallas_call``, pool updated in place.

Single-token rows take the XLA reference's vectorised decode phase unchanged. Every row with two or more tokens is
cut into ``chunk_size`` chunks aligned to its first token, exactly as in the XLA reference (``_xla.ragged_kda``):
chunk ``c`` reads the ``chunk_size``-token window of the flat stream that starts at the chunk's first token (clamped
back near the end of the buffer), and tokens of the window outside the chunk are zeroed, which makes them inert.
The live chunks of a step are laid out back to back (row by row) in a static number of chunk slots::

    num_chunks = max_live_chunks(T, R, cs)           # static bound on the live chunks (see that function)

    XLA:     windows of q, k, v, beta, decay -> normalised, masked, head-major [num_chunks, H, cs, D] fp32
    kernel:  grid = (H // block_h "parallel", num_chunks "arbitrary")
             state [block_h, K, V] fp32 in a VMEM scratch, carried along the chunk axis
             a row's first chunk seeds it from the row's pool slot (zero for a fresh row), its last chunk stores it
             back; the pool block follows the slot table (scalar prefetch), so each live row's slot is DMAed in once
             and out once, and the pool is an input/output alias (no other slot is read or written)
             per chunk: g_cumsum on the MXU and the shared ``chunk_core`` math vmapped over the head block (the
             training ``kda_chunk`` kernel's factored form)
             chunk slots past the live ones revisit the last live chunk's blocks: no DMA, no compute
    XLA:     chunk outputs gathered back to the flat stream, merged with the decode rows' outputs

The kernel needs the factored pairwise decay (a ``decay_lower_bound`` that allows it), the masked chunk inverse and
lane-aligned head sizes; :func:`supported` says when, and the registry entry runs the XLA reference otherwise. Rows
with tokens must own distinct pool slots (as eSurge schedules them).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ...._xla.kda_chunk._xla_impl import LANES, SUBLANES, chunk_core, l2norm, pairwise_sub_block, resolve_precision
from ...._xla.ragged_kda._xla_impl import _decode_phase, resolve_inverse
from ..kda_chunk._pallas_impl_fwd import chunk_cumsum

_NUM_TABLES = 6  # scalar-prefetch operands: live count, chunk map, slot, first, last, init


def supported(head_dim: int, v_dim: int, chunk_size: int, sub: int, inverse: str) -> bool:
    """Whether the Pallas kernel runs these options (otherwise the XLA reference does).

    Args:
        head_dim: Key / query head size.
        v_dim: Value head size.
        chunk_size: Chunk length.
        sub: Factored-decay sub-block size from ``pairwise_sub_block`` (``0``: dense form, not in this kernel).
        inverse: Resolved chunk inverse (only ``"masked"`` lowers on TPU).

    Returns:
        ``True`` when both head sizes are multiples of the 128 lanes, the chunk is a power of two in ``[8, 128]``, the
        pairwise decay is factored and the inverse is the masked one.
    """
    cs = int(chunk_size)
    return (
        int(head_dim) % LANES == 0
        and int(v_dim) % LANES == 0
        and SUBLANES <= cs <= LANES
        and not cs & (cs - 1)
        and int(sub) > 0
        and inverse == "masked"
    )


def pick_block_h(num_heads: int, requested: int = 0) -> int:
    """Heads per grid step.

    Measured on one v5p chip (16 local heads, one 2,048-token row, bf16 pool; ms per call, XLA window gathers
    included): block_h 8 0.817, 4 0.838, 2 0.918, 16 1.190, 1 1.226. Larger blocks batch more heads into each
    chunk's short chain of dependent matmuls. One block of all 16 heads leaves the "parallel" grid axis a single
    step, which a megacore chip cannot split over its two TensorCores (the likely cause of its slowdown).

    Args:
        num_heads: Local head count.
        requested: Explicit choice (must divide ``num_heads``), or ``0``.

    Returns:
        ``requested`` when given, else the largest of 8, 4, 2, 1 that divides ``num_heads`` and leaves at least two
        head blocks, else ``num_heads``.

    Raises:
        ValueError: If ``requested`` does not divide ``num_heads``.
    """
    if requested:
        if num_heads % requested:
            raise ValueError(f"block_h={requested} must divide the head count {num_heads}")
        return int(requested)
    for bh in (8, 4, 2, 1):
        if num_heads % bh == 0 and num_heads // bh >= 2:
            return bh
    return int(num_heads)


def max_live_chunks(num_tokens: int, num_rows: int, chunk_size: int) -> int:
    """Static bound on the live chunks of a step of ``T`` tokens over ``R`` rows.

    Rows with ``n >= 2`` tokens number at most ``min(R, T // 2)``, and each has ``ceil(n / cs) < n / cs + 1`` chunks,
    so all of them have at most ``cdiv(T, cs) + min(R, T // 2)``. For ``cs >= 8`` also ``ceil(n / cs) <= n // 2``, so
    at most ``T // 2``. The grid gets the smaller bound (at least one slot).

    Args:
        num_tokens: Length of the flat stream (before any padding to one chunk).
        num_rows: Row count ``R``.
        chunk_size: Chunk length (``>= 8``).

    Returns:
        The chunk-slot count of the kernel's grid.
    """
    t, cs = int(num_tokens), int(chunk_size)
    return max(min(t // 2, -(-t // cs) + min(int(num_rows), t // 2)), 1)


def _vmem_limit_bytes() -> int:
    try:
        capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # pragma: no cover - only reached off TPU
        capacity = 64 * 1024 * 1024
    return int(capacity * 0.75)


def _chunk_tables(
    qsl: Array,
    lens: Array,
    state_indices: Array | None,
    has_init: Array | None,
    *,
    num_tokens: int,
    num_slots: int,
    chunk_size: int,
    num_chunks: int,
) -> dict[str, Array]:
    """Per-chunk-slot tables: row, window start, valid lanes, pool slot and first / last / init flags."""
    i32 = jnp.int32
    cs = chunk_size
    num_rows = lens.shape[0]
    n_chunks = jnp.where(lens >= 2, (lens + cs - 1) // cs, 0).astype(i32)
    chunk_end = jnp.cumsum(n_chunks).astype(i32)
    chunk_begin = chunk_end - n_chunks
    n_live = chunk_end[-1]
    c = jnp.arange(num_chunks, dtype=i32)
    live = c < n_live
    # Slots past the live chunks revisit the last live chunk (same blocks: the pipeline issues no DMA for them).
    cmap = jnp.minimum(c, jnp.maximum(n_live - 1, 0))
    row = jnp.minimum(jnp.sum(chunk_end[None, :] <= cmap[:, None], axis=1), num_rows - 1).astype(i32)
    local = cmap - chunk_begin[row]
    start = qsl[row] + local * cs
    win_start = jnp.clip(start, 0, num_tokens - cs)
    lo = jnp.where(live, start - win_start, 0)
    hi = jnp.where(live, jnp.minimum(start + cs, qsl[row + 1]) - win_start, 0)
    slot = row if state_indices is None else state_indices[row]
    init = jnp.ones((num_chunks,), i32) if has_init is None else has_init[row].astype(i32)
    return {
        "n_live": n_live.reshape(1),
        "cmap": cmap,
        "slot": jnp.clip(slot, 0, num_slots - 1).astype(i32),
        "first": (live & (local == 0)).astype(i32),
        "last": (live & (local == n_chunks[row] - 1)).astype(i32),
        "init": init,
        "win_start": win_start,
        "lo": lo,
        "hi": hi,
        "chunk_begin": chunk_begin,
    }


def _chunk_inputs(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    tables: dict[str, Array],
    *,
    chunk_size: int,
    use_qk_l2norm: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    """Head-major chunk windows ``[C, H, cs, D]`` fp32 of ``q * K**-0.5``, ``k``, ``k * beta``, ``v * beta``, ``g``.

    Same per-token arithmetic and masking as the XLA reference's chunk loop: lanes outside ``[lo, hi)`` are zero.
    """
    cs = chunk_size
    f32 = jnp.float32
    scale = 1.0 / (query.shape[-1] ** 0.5)
    lane = jnp.arange(cs, dtype=jnp.int32)
    valid = (lane[None, :] >= tables["lo"][:, None]) & (lane[None, :] < tables["hi"][:, None])  # [C, cs]
    vmask = valid[:, :, None, None]
    starts = tables["win_start"]

    def windows(x: Array) -> Array:
        return jax.vmap(lambda s: lax.dynamic_slice_in_dim(x, s, cs, axis=0))(starts)  # [C, cs, ...]

    q = windows(query).astype(f32)
    k = windows(key).astype(f32)
    if use_qk_l2norm:
        q = l2norm(q)
        k = l2norm(k)
    q = jnp.where(vmask, q * scale, 0.0)
    k = jnp.where(vmask, k, 0.0)
    v = jnp.where(vmask, windows(value).astype(f32), 0.0)
    g = jnp.where(vmask, windows(decay).astype(f32), 0.0)
    b = jnp.where(valid[:, :, None], windows(beta).astype(f32), 0.0)[..., None]
    return tuple(jnp.swapaxes(x, 1, 2) for x in (q, k, k * b, v * b, g))


def _chunk_kernel(
    n_live_ref,
    cmap_ref,
    slot_ref,
    first_ref,
    last_ref,
    init_ref,
    q_ref,
    k_ref,
    kb_ref,
    vb_ref,
    g_ref,
    pool_ref,
    out_ref,
    new_pool_ref,
    state_scr,
    gc_scr,
    *,
    chunk_size: int,
    sub: int,
    precision,
):
    """Grid step ``(hb, c)``: advance ``block_h`` heads of chunk slot ``c``'s row through that chunk."""
    del cmap_ref, slot_ref  # consumed by the index maps
    c = pl.program_id(1)
    n_live = n_live_ref[0]
    live = c < n_live
    first = jnp.logical_and(live, first_ref[c] != 0)
    keep = init_ref[c] != 0

    @pl.when(jnp.logical_and(first, keep))
    def _seed():
        state_scr[...] = pool_ref[...].astype(jnp.float32)

    @pl.when(jnp.logical_and(first, jnp.logical_not(keep)))
    def _fresh():
        state_scr[...] = jnp.zeros(state_scr.shape, jnp.float32)

    @pl.when(live)
    def _advance():
        gc_scr[...] = chunk_cumsum(g_ref[...])

        def one_head(st, q, k, kb, vb, gc, gl):
            return chunk_core(st, q, k, kb, vb, gc, gl, None, precision, inverse="masked", sub=sub)

        new_state, core_out = jax.vmap(one_head)(
            state_scr[...],
            q_ref[...],
            k_ref[...],
            kb_ref[...],
            vb_ref[...],
            gc_scr[...],
            gc_scr[:, pl.ds(chunk_size - 1, 1), :],
        )
        out_ref[...] = core_out
        state_scr[...] = new_state

    @pl.when(jnp.logical_and(live, last_ref[c] != 0))
    def _store():
        new_pool_ref[...] = state_scr[...].astype(new_pool_ref.dtype)

    @pl.when(n_live == 0)
    def _hold():
        # No live chunk: the visited pool block is written back unchanged.
        new_pool_ref[...] = pool_ref[...]


def _flat_output(
    chunk_out: Array,
    qsl: Array,
    lens: Array,
    tables: dict[str, Array],
    *,
    num_tokens: int,
    chunk_size: int,
) -> tuple[Array, Array]:
    """Gather the chunk outputs ``[C, H, cs, V]`` back to the flat stream.

    Returns:
        ``(in_multi_token_row [T] bool, out [T, H, V])``; ``out`` is meaningful only where the flag is set.
    """
    i32 = jnp.int32
    cs = chunk_size
    num_rows = lens.shape[0]
    num_chunks = chunk_out.shape[0]
    t = jnp.arange(num_tokens, dtype=i32)
    row = jnp.sum(qsl[None, 1:] <= t[:, None], axis=1).astype(i32)
    row_c = jnp.minimum(row, num_rows - 1)
    owned = (row < num_rows) & (t >= qsl[row_c])
    multi = owned & (lens[row_c] >= 2)
    chunk = jnp.clip(tables["chunk_begin"][row_c] + (t - qsl[row_c]) // cs, 0, num_chunks - 1)
    lane = jnp.clip(t - tables["win_start"][chunk], 0, cs - 1)
    return multi, chunk_out[chunk, :, lane, :]


def ragged_kda_pallas(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    recurrent_state: Array,
    query_start_loc: Array,
    state_indices: Array | None,
    has_initial_state: Array | None = None,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    inverse: str = "auto",
    pairwise: str = "auto",
    block_h: int = 0,
) -> tuple[Array, Array]:
    """Ragged per-channel KDA over a packed token stream and a recurrent-state pool (Pallas TPU chunk path).

    Same contract as the XLA reference ``ragged_kda_xla``; the caller checks :func:`supported` first.

    Args:
        query: Post-conv queries ``[T, H, K]``.
        key: Post-conv keys ``[T, H, K]``.
        value: Post-conv values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``).
        recurrent_state: State pool ``[S, H, K, V]``; computed in fp32, returned in its own dtype.
        query_start_loc: Cumulative row offsets ``[R + 1]`` (nondecreasing).
        state_indices: Pool slot of every row ``[R]`` (out of range: the row is empty), or ``None`` (row ``r`` owns
            slot ``r``). Rows with tokens must own distinct slots.
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state.
        chunk_size: Chunk length of the multi-token path.
        use_qk_l2norm: L2-normalise queries and keys.
        decay_lower_bound: Promised lower bound of every log-decay (selects the factored pairwise decay).
        precision: Precision of the non-inverse chunk dots.
        inverse: ``"auto"`` or ``"masked"``.
        pairwise: ``"auto"`` (the factored form must result).
        block_h: Heads per grid step (``0``: :func:`pick_block_h`).

    Returns:
        ``(out [T, H, V] fp32, new_recurrent_state [S, H, K, V])``.

    Raises:
        ValueError: If the options do not resolve to the factored form with the masked inverse, or for a bad
            ``block_h``.
    """
    num_tokens, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    cs = int(chunk_size)
    num_slots = recurrent_state.shape[0]
    sub = pairwise_sub_block(decay_lower_bound, cs) if pairwise == "auto" else 0
    inverse = resolve_inverse(inverse, cs)
    if not supported(head_dim, v_dim, cs, sub, inverse):
        raise ValueError(
            f"ragged_kda Pallas TPU kernel: unsupported options (head_dim={head_dim}, v_dim={v_dim}, "
            f"chunk_size={cs}, factored sub-block={sub}, inverse={inverse!r})"
        )
    qsl = query_start_loc.astype(jnp.int32)
    num_rows = qsl.shape[0] - 1
    lens = jnp.maximum(qsl[1:] - qsl[:-1], 0)
    if state_indices is not None:
        state_indices = state_indices.astype(jnp.int32)
        lens = jnp.where((state_indices >= 0) & (state_indices < num_slots), lens, 0)
    has_init = None if has_initial_state is None else jnp.asarray(has_initial_state).astype(jnp.bool_)

    pad = max(cs - num_tokens, 0)  # a chunk window must fit in the buffer (as in the reference)
    if pad:
        query, key, value, decay = (jnp.pad(x, ((0, pad), (0, 0), (0, 0))) for x in (query, key, value, decay))
        beta = jnp.pad(beta, ((0, pad), (0, 0)))
    padded = num_tokens + pad
    out = jnp.zeros((padded, num_heads, v_dim), jnp.float32)
    out, pool = _decode_phase(
        out,
        recurrent_state,
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        starts=qsl[:-1],
        lens=lens,
        state_indices=state_indices,
        has_init=has_init,
        num_tokens=num_tokens,
        use_qk_l2norm=use_qk_l2norm,
    )

    num_chunks = max_live_chunks(num_tokens, num_rows, cs)
    tables = _chunk_tables(
        qsl,
        lens,
        state_indices,
        has_init,
        num_tokens=padded,
        num_slots=num_slots,
        chunk_size=cs,
        num_chunks=num_chunks,
    )
    q, k, kb, vb, g = _chunk_inputs(query, key, value, beta, decay, tables, chunk_size=cs, use_qk_l2norm=use_qk_l2norm)

    bh = pick_block_h(num_heads, int(block_h))

    def chunk_map(hb, c, n_live, cmap, *_):
        return (cmap[c], hb, 0, 0)

    def slot_map(hb, c, n_live, cmap, slot, *_):
        return (slot[c], hb, 0, 0)

    k_spec = pl.BlockSpec((None, bh, cs, head_dim), chunk_map)
    v_spec = pl.BlockSpec((None, bh, cs, v_dim), chunk_map)
    pool_spec = pl.BlockSpec((None, bh, head_dim, v_dim), slot_map)
    f32 = jnp.float32
    kernel = functools.partial(_chunk_kernel, chunk_size=cs, sub=int(sub), precision=resolve_precision(precision))
    chunk_out, new_pool = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=_NUM_TABLES,
            grid=(num_heads // bh, num_chunks),
            in_specs=[k_spec, k_spec, k_spec, v_spec, k_spec, pool_spec],
            out_specs=[v_spec, pool_spec],
            scratch_shapes=[pltpu.VMEM((bh, head_dim, v_dim), f32), pltpu.VMEM((bh, cs, head_dim), f32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_chunks, num_heads, cs, v_dim), f32),
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        ],
        # operands: 6 tables, q, k, k_beta, v_beta, g, pool -> the pool (11) is updated in place as output 1.
        input_output_aliases={_NUM_TABLES + 5: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=_vmem_limit_bytes(),
        ),
        name=f"ejkernel_ragged_kda_chunks_bh{bh}_cs{cs}",
    )(
        tables["n_live"],
        tables["cmap"],
        tables["slot"],
        tables["first"],
        tables["last"],
        tables["init"],
        q,
        k,
        kb,
        vb,
        g,
        pool,
    )
    multi, flat = _flat_output(chunk_out, qsl, lens, tables, num_tokens=num_tokens, chunk_size=cs)
    return jnp.where(multi[:, None, None], flat, out[:num_tokens]), new_pool


__all__ = ("max_live_chunks", "pick_block_h", "ragged_kda_pallas", "supported")
