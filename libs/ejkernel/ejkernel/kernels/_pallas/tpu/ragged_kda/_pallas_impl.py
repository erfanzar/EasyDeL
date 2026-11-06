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

"""Pallas TPU ``ragged_kda``: chunk rows in one ``pallas_call`` (pool updated in place), short rows token by token.

Row planning, the causal conv and the short rows (at most ``short_row_max`` tokens, captured per token into the row's
slot group) follow the XLA reference (``_xla.ragged_kda``). With the conv arrays every short row runs in one Pallas
kernel (``ragged_kda_short``: conv fused, state in VMEM, captures written into the pools in place, the serving decode
step's arithmetic); on post-conv inputs the short rows run the reference's decode step.

Every chunk row (more than ``short_row_max`` tokens) is cut into ``chunk_size`` chunks aligned to its first token,
exactly as in the XLA reference: chunk ``c`` reads the ``chunk_size``-token window of the flat (post-conv) stream that
starts at the chunk's first token (clamped back near the end of the buffer), and tokens of the window outside the
chunk are zeroed, which makes them inert. The live chunks of a step are laid out back to back (row by row) in a static
number of chunk slots::

    num_chunks = max_live_chunks(T, R, cs)           # static bound on the live chunks (see that function)

    XLA:     windows of q, k, v, beta, decay -> normalised, masked, head-major [num_chunks, H, cs, D] fp32
             (beta [T, H] has no lane axis: its windows come from a row gather of cs-token blocks plus static
             shifts, because XLA lowers a window gather of it to a loop with one iteration per chunk slot)
    kernel:  grid = (H // block_h "parallel", num_chunks "arbitrary")
             state [block_h, K, V] fp32 in a VMEM scratch, carried along the chunk axis
             a row's first chunk seeds it from the row's read slot g + a (zero for a fresh row), its last chunk
             stores it into the base slot g; the pool blocks follow the read / write slot tables (scalar prefetch), so
             each live row's slots are DMAed in once and out once, and the pool is an input/output alias (no other
             slot is read or written)
             per chunk: g_cumsum on the MXU and the shared ``chunk_core`` math vmapped over the head block (the
             training ``kda_chunk`` kernel's factored form)
             chunk slots past the live ones revisit the last live chunk's blocks: no DMA, no compute
    XLA:     chunk outputs gathered back to the flat stream, merged with the short rows' outputs

The kernel needs the factored pairwise decay (a ``decay_lower_bound`` that allows it), the masked chunk inverse and
lane-aligned head sizes; :func:`supported` says when, and the registry entry runs the XLA reference otherwise. Rows
with tokens must own disjoint slot groups (as eSurge schedules them).
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
from ...._xla.ragged_kda._xla_impl import conv_and_short_rows, plan_rows, resolve_inverse
from ..kda_chunk._pallas_impl_fwd import chunk_cumsum
from ..ragged_kda_short._interface import ragged_kda_short

_NUM_TABLES = 7  # scalar-prefetch operands: live count, chunk map, read slot, write slot, first, last, init


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
    read_slot: Array,
    write_slot: Array,
    has_init: Array,
    *,
    num_tokens: int,
    num_slots: int,
    chunk_size: int,
    num_chunks: int,
) -> dict[str, Array]:
    """Per-chunk-slot tables: row, window start, valid lanes, read / write pool slots and first / last / init flags.

    ``lens`` holds the chunk rows' token counts (zero for every other row).
    """
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
    wslot = jnp.clip(write_slot[row], 0, num_slots - 1).astype(i32)
    rslot = jnp.clip(read_slot[row], 0, num_slots - 1).astype(i32)
    # Without a live chunk the kernel copies the visited block onto itself: read and write the same slot.
    rslot = jnp.where(n_live > 0, rslot, wslot)
    return {
        "n_live": n_live.reshape(1),
        "cmap": cmap,
        "rslot": rslot,
        "wslot": wslot,
        "first": (live & (local == 0)).astype(i32),
        "last": (live & (local == n_chunks[row] - 1)).astype(i32),
        "init": has_init[row].astype(i32),
        "win_start": win_start,
        "lo": lo,
        "hi": hi,
        "chunk_begin": chunk_begin,
    }


def _beta_windows(beta: Array, starts: Array, chunk_size: int) -> Array:
    """``beta[s : s + cs]`` for every window start ``s`` of ``starts`` ``[C]``: ``[C, cs, H]``, without a window gather.

    ``beta`` ``[T, H]`` has no 128-lane axis (in serving it leaves the narrow beta projection token-minor), and a
    vmapped ``dynamic_slice`` of it lowers to an XLA while loop with one index fusion and one dynamic-update-slice per
    window (544 iterations per call at 32,768 tokens and 32 rows). Instead ``beta`` is viewed as rows of ``cs``
    tokens; each window lies in the two consecutive rows starting at row ``s // cs`` (one plain row gather), and
    ``log2(cs)`` static selects shift that pair left by ``s % cs``. Pure data movement: the result equals the window
    slices bit for bit (non-finite values included).

    Args:
        beta: ``[T, H]`` with ``T >= cs``.
        starts: Window starts ``[C]`` int32 in ``[0, T - cs]``.
        chunk_size: Window length ``cs`` (a power of two).

    Returns:
        ``[C, cs, H]`` in ``beta``'s dtype.
    """
    cs = int(chunk_size)
    num_tokens, num_heads = beta.shape
    num_blocks = -(-num_tokens // cs)
    if num_blocks * cs != num_tokens:
        beta = jnp.pad(beta, ((0, num_blocks * cs - num_tokens), (0, 0)))
    blocks = beta.reshape(num_blocks, cs * num_heads)
    starts = starts.astype(jnp.int32)
    first = starts // cs
    shift = starts - first * cs
    # A window that ends inside its first row (shift 0 at the last row) never reads the clamped second row.
    rows = jnp.stack([first, jnp.minimum(first + 1, num_blocks - 1)], axis=1)
    pair = jnp.take(blocks, rows, axis=0, mode="clip").reshape(starts.shape[0], 2 * cs, num_heads)
    for bit in range(cs.bit_length() - 1):
        step = 1 << bit
        moved = jnp.concatenate([pair[:, step:], pair[:, :step]], axis=1)
        pair = jnp.where(((shift >> bit) & 1)[:, None, None] == 1, moved, pair)
    return pair[:, :cs]


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
    The beta windows come from :func:`_beta_windows` (equal to window slices bit for bit, without a per-window loop).
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
    b = jnp.where(valid[:, :, None], _beta_windows(beta, starts, cs).astype(f32), 0.0)[..., None]
    return tuple(jnp.swapaxes(x, 1, 2) for x in (q, k, k * b, v * b, g))


def _chunk_kernel(
    n_live_ref,
    cmap_ref,
    rslot_ref,
    wslot_ref,
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
    del cmap_ref, rslot_ref, wslot_ref  # consumed by the index maps
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
        # No live chunk: the visited pool block (read slot == write slot) is written back unchanged.
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
    """Gather the chunk outputs ``[C, H, cs, V]`` back to the flat stream (``lens``: the chunk rows' token counts).

    Returns:
        ``(in_chunk_row [T] bool, out [T, H, V])``; ``out`` is meaningful only where the flag is set.
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
    read_offset: Array | None = None,
    commit_lens: Array | None = None,
    q_conv_weight: Array | None = None,
    k_conv_weight: Array | None = None,
    v_conv_weight: Array | None = None,
    q_conv_state: Array | None = None,
    k_conv_state: Array | None = None,
    v_conv_state: Array | None = None,
    *,
    short_row_max: int = 1,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    inverse: str = "auto",
    pairwise: str = "auto",
    block_h: int = 0,
) -> tuple[Array, ...]:
    """Ragged per-channel KDA with the causal conv fused and per-token state capture (Pallas TPU).

    Same contract as the XLA reference ``ragged_kda_xla``; the caller checks :func:`supported` first.

    Args:
        query: Queries ``[T, H, K]``: raw (pre-conv) with the six conv arrays, post-conv without them.
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``).
        recurrent_state: State pool ``[S, H, K, V]``; computed in fp32, returned in its own dtype.
        query_start_loc: Cumulative row offsets ``[R + 1]`` (nondecreasing).
        state_indices: Base slot of every row ``[R]``, or ``None`` (``g = r``). Rows must own disjoint slot groups.
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from zero windows and a zero state.
        read_offset: Optional ``[R]`` read offsets (zero when ``None``).
        commit_lens: Optional ``[R]`` commit lengths (``lens`` when ``None``).
        q_conv_weight: Depthwise causal conv taps ``[H * K, d_conv]``, or ``None`` (post-conv inputs).
        k_conv_weight: ``[H * K, d_conv]`` or ``None``.
        v_conv_weight: ``[H * V, d_conv]`` or ``None``.
        q_conv_state: Conv-window pool ``[S, H * K, d_conv]``, or ``None``.
        k_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        v_conv_state: ``[S, H * V, d_conv]`` or ``None``.
        short_row_max: ``W``: rows of at most ``W`` tokens take the short-row path (per-token capture).
        chunk_size: Chunk length of the chunk path.
        use_qk_l2norm: L2-normalise queries and keys.
        decay_lower_bound: Promised lower bound of every log-decay (selects the factored pairwise decay).
        precision: Precision of the non-inverse chunk dots.
        inverse: ``"auto"`` or ``"masked"``.
        pairwise: ``"auto"`` (the factored form must result).
        block_h: Heads per grid step of the chunk kernel (``0``: :func:`pick_block_h`).

    Returns:
        With the conv arrays ``(out [T, H, V] fp32, q_conv_state, k_conv_state, v_conv_state, recurrent_state)``,
        without them ``(out, recurrent_state)``.

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
    fused = q_conv_weight is not None
    qsl = query_start_loc.astype(jnp.int32)
    num_rows = qsl.shape[0] - 1
    plan = plan_rows(
        qsl,
        state_indices,
        has_initial_state,
        read_offset,
        commit_lens,
        num_slots=num_slots,
        short_row_max=int(short_row_max),
    )
    weights = (q_conv_weight, k_conv_weight, v_conv_weight) if fused else None
    pools_in = (q_conv_state, k_conv_state, v_conv_state, recurrent_state)

    def all_rows(pools_in):
        return _all_rows(
            query,
            key,
            value,
            beta,
            decay,
            weights,
            pools_in,
            qsl,
            plan,
            short_row_max=int(short_row_max),
            use_qk_l2norm=use_qk_l2norm,
            chunk_size=cs,
            sub=int(sub),
            precision=precision,
            block_h=int(block_h),
        )

    def short_rows_only(pools_in):
        out = jnp.zeros((num_tokens, num_heads, v_dim), jnp.float32)
        out, pools = ragged_kda_short(
            out,
            pools_in,
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            weights=weights,
            plan=plan,
            short_row_max=int(short_row_max),
            use_qk_l2norm=use_qk_l2norm,
        )
        return (out, *pools)

    if int(short_row_max) > 1 and num_tokens <= num_rows * int(short_row_max):
        # A step whose rows all have at most W tokens (a speculative verify or decode step) fits a bucket of R * W
        # tokens. Without a chunk row it runs the short rows alone: no flat conv, chunk tables or chunk kernel.
        out, q_pool, k_pool, v_pool, new_pool = lax.cond(jnp.any(plan.chunk), all_rows, short_rows_only, pools_in)
    else:
        out, q_pool, k_pool, v_pool, new_pool = all_rows(pools_in)
    if fused:
        return out, q_pool, k_pool, v_pool, new_pool
    return out, new_pool


def _all_rows(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    weights: tuple[Array, Array, Array] | None,
    pools_in: tuple[Array | None, Array | None, Array | None, Array],
    qsl: Array,
    plan,
    *,
    short_row_max: int,
    use_qk_l2norm: bool,
    chunk_size: int,
    sub: int,
    precision: str,
    block_h: int,
) -> tuple[Array | None, ...]:
    """The flat conv, the short rows and the chunk kernel over the chunk rows' live chunks.

    Returns:
        ``(out [T, H, V] fp32, q_conv_state, k_conv_state, v_conv_state, recurrent_state)``; the conv pools are
        ``None`` for post-conv inputs.
    """
    num_tokens, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    cs = int(chunk_size)
    num_slots = pools_in[3].shape[0]
    num_rows = qsl.shape[0] - 1
    out, pools, (query, key, value) = conv_and_short_rows(
        query,
        key,
        value,
        beta,
        decay,
        weights,
        pools_in,
        qsl,
        plan,
        short_row_max=int(short_row_max),
        use_qk_l2norm=use_qk_l2norm,
        short_fn=ragged_kda_short,
    )
    chunk_lens = jnp.where(plan.chunk, plan.lens, 0)

    pad = max(cs - num_tokens, 0)  # a chunk window must fit in the buffer (as in the reference)
    if pad:
        query, key, value, decay = (jnp.pad(x, ((0, pad), (0, 0), (0, 0))) for x in (query, key, value, decay))
        beta = jnp.pad(beta, ((0, pad), (0, 0)))
    padded = num_tokens + pad
    num_chunks = max_live_chunks(num_tokens, num_rows, cs)
    tables = _chunk_tables(
        qsl,
        chunk_lens,
        plan.read_slot,
        plan.base_slot,
        plan.init,
        num_tokens=padded,
        num_slots=num_slots,
        chunk_size=cs,
        num_chunks=num_chunks,
    )
    q, k, kb, vb, g = _chunk_inputs(query, key, value, beta, decay, tables, chunk_size=cs, use_qk_l2norm=use_qk_l2norm)

    bh = pick_block_h(num_heads, int(block_h))

    def chunk_map(hb, c, n_live, cmap, *_):
        return (cmap[c], hb, 0, 0)

    def read_map(hb, c, n_live, cmap, rslot, *_):
        return (rslot[c], hb, 0, 0)

    def write_map(hb, c, n_live, cmap, rslot, wslot, *_):
        return (wslot[c], hb, 0, 0)

    k_spec = pl.BlockSpec((None, bh, cs, head_dim), chunk_map)
    v_spec = pl.BlockSpec((None, bh, cs, v_dim), chunk_map)
    f32 = jnp.float32
    pool = pools[3]
    kernel = functools.partial(_chunk_kernel, chunk_size=cs, sub=int(sub), precision=resolve_precision(precision))
    chunk_out, new_pool = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=_NUM_TABLES,
            grid=(num_heads // bh, num_chunks),
            in_specs=[
                k_spec,
                k_spec,
                k_spec,
                v_spec,
                k_spec,
                pl.BlockSpec((None, bh, head_dim, v_dim), read_map),
            ],
            out_specs=[v_spec, pl.BlockSpec((None, bh, head_dim, v_dim), write_map)],
            scratch_shapes=[pltpu.VMEM((bh, head_dim, v_dim), f32), pltpu.VMEM((bh, cs, head_dim), f32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_chunks, num_heads, cs, v_dim), f32),
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        ],
        # operands: 7 tables, q, k, k_beta, v_beta, g, pool -> the pool (12) is updated in place as output 1.
        input_output_aliases={_NUM_TABLES + 5: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=_vmem_limit_bytes(),
        ),
        name=f"ejkernel_ragged_kda_chunks_bh{bh}_cs{cs}",
    )(
        tables["n_live"],
        tables["cmap"],
        tables["rslot"],
        tables["wslot"],
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
    multi, flat = _flat_output(chunk_out, qsl, chunk_lens, tables, num_tokens=num_tokens, chunk_size=cs)
    out = jnp.where(multi[:, None, None], flat, out)
    return out, pools[0], pools[1], pools[2], new_pool


__all__ = ("max_live_chunks", "pick_block_h", "ragged_kda_pallas", "supported")
