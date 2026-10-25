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

"""Pallas TPU forward of the chunked per-channel-decay KDA: the whole inter-chunk recurrence in one ``pallas_call``.

::

    grid = (batch, num_heads // block_h, num_chunks)      # ("parallel", "parallel", "arbitrary")

    per (b, hb):     state [block_h, K, V] fp32 in a VMEM scratch, carried along the chunk axis
    per (b, hb, c):  q, k, k_beta, v_beta, g_cumsum tiles of this chunk  [block_h, cs, 128*]  (BlockSpec DMA)
                     meta tile of this (b, c)                             [8, 128]  (packed rows only; row 0 holds
                                                                          the local run index in lanes < cs)
                     for h in range(block_h): chunk_core(...) on one head's tiles (shared with the XLA module)
                     c == 0 seeds the state from initial_state; c == last writes final_state.

Every chunk-local term (the ``[cs, cs, K]`` pairwise decay, both contractions over it, the block-recursive chunk
inverse, the seven small matmuls and the state update) runs on VMEM tiles; HBM sees each input once.

Layout rule: every BlockSpec block and scratch has (8, 128)-aligned trailing dims, including dims equal to the full
array dim -- Mosaic's memref layout pass aborts the process (``layout.h: Check failed: rem == 0``) otherwise. That is
why ``beta`` is never an operand (the wrapper passes ``k * beta`` and ``v * beta``) and the packed-row metadata
travels as one ``[8, 128]`` tile per ``(b, c)``. :func:`check_tile_alignment` enforces the rule at trace time. The
per-chunk prefix sum of ``g`` is taken outside the kernel (Mosaic has no ``cumsum`` lowering).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ...._xla.kda_chunk._xla_impl import LANES, META_ROW_RUN, SUBLANES, chunk_core, resolve_precision, segment_masks

DEFAULT_FWD_VMEM_LIMIT = 32 * 2**20


def pack_segment_meta(local_run: Array) -> Array:
    """``[B, C, cs]`` local run indices -> one (8, 128)-aligned fp32 tile per ``(b, c)``: ``[B, C, 8, 128]``.

    Args:
        local_run: Local run index of every token, ``cs <= 128``.

    Returns:
        Metadata tiles; row ``META_ROW_RUN`` holds the indices in lanes ``< cs``, everything else is zero.
    """
    B, C, cs = local_run.shape
    if cs > LANES:
        raise ValueError(f"chunk_size {cs} exceeds the {LANES}-lane metadata tile")
    meta = jnp.zeros((B, C, SUBLANES, LANES), jnp.float32)
    return meta.at[:, :, META_ROW_RUN, :cs].set(local_run.astype(jnp.float32))


def check_tile_alignment(layout: dict[str, tuple[int | None, ...]]) -> None:
    """Raise ``ValueError`` unless every block / scratch shape has (8, 128)-aligned trailing dims.

    Args:
        layout: Operand name -> block shape (``None`` = squeezed dim).

    Raises:
        ValueError: Naming every offending block.
    """
    bad = []
    for name, shape in layout.items():
        dims = [d for d in shape if d is not None]
        if len(dims) < 2:
            bad.append(f"{name}: block {shape} has fewer than two non-squeezed dims")
            continue
        if dims[-2] % SUBLANES or dims[-1] % LANES:
            bad.append(f"{name}: block {shape} -> trailing dims {tuple(dims[-2:])} not ({SUBLANES}, {LANES})-aligned")
    if bad:
        raise ValueError("kda_chunk: unaligned Pallas TPU block/scratch shapes:\n  " + "\n  ".join(bad))


def fwd_block_layout(
    block_h: int, chunk_size: int, k_dim: int, v_dim: int, segmented: bool
) -> dict[str, tuple[int | None, ...]]:
    """Every block / scratch shape the forward launcher uses, by operand name."""
    cs = chunk_size
    lay: dict[str, tuple[int | None, ...]] = {
        "q": (None, block_h, cs, k_dim),
        "k": (None, block_h, cs, k_dim),
        "k_beta": (None, block_h, cs, k_dim),
        "v_beta": (None, block_h, cs, v_dim),
        "g_cumsum": (None, block_h, cs, k_dim),
        "initial_state": (None, block_h, k_dim, v_dim),
    }
    if segmented:
        lay["segment_meta"] = (None, None, SUBLANES, LANES)
    lay["core_out"] = (None, block_h, cs, v_dim)
    lay["final_state"] = (None, block_h, k_dim, v_dim)
    lay["chunk_states_in"] = (None, block_h, None, k_dim, v_dim)
    lay["scratch:state"] = (block_h, k_dim, v_dim)
    return lay


def _fwd_kernel(*refs, block_h: int, num_chunks: int, segmented: bool, precision):
    """Grid step ``(b, hb, c)``: advance ``block_h`` heads through chunk ``c``."""
    q_ref, k_ref, kb_ref, vb_ref, gc_ref, init_ref = refs[:6]
    pos = 6
    meta_ref = None
    if segmented:
        meta_ref = refs[pos]
        pos += 1
    out_ref, final_ref, states_ref, state_scr = refs[pos : pos + 4]

    cs = q_ref.shape[1]
    c = pl.program_id(2)

    @pl.when(c == 0)
    def _seed():
        state_scr[...] = init_ref[...].astype(jnp.float32)

    seg = None
    if segmented:
        seg = segment_masks(meta_ref[pl.ds(META_ROW_RUN, 1), pl.ds(0, cs)], cs)

    for h in range(block_h):
        state = state_scr[h]
        states_ref[h] = state
        g_last = gc_ref[h, pl.ds(cs - 1, 1), :]
        new_state, core_out = chunk_core(
            state, q_ref[h], k_ref[h], kb_ref[h], vb_ref[h], gc_ref[h], g_last, seg, precision, inverse="masked"
        )
        out_ref[h] = core_out
        state_scr[h] = new_state

    @pl.when(c == num_chunks - 1)
    def _finish():
        final_ref[...] = state_scr[...]


def kda_chunk_pallas_fwd(
    q: Array,
    k: Array,
    k_beta: Array,
    v_beta: Array,
    g_cumsum: Array,
    initial_state: Array,
    segment_meta: Array | None,
    *,
    chunk_size: int,
    block_h: int = 1,
    precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_FWD_VMEM_LIMIT,
) -> tuple[Array, Array, Array]:
    """Launch the fused chunked-KDA forward.

    Args:
        q: Scaled, normalised queries ``[B, H, Lp, K]`` fp32, ``Lp % chunk_size == 0``.
        k: Keys ``[B, H, Lp, K]`` fp32.
        k_beta: ``k * beta`` ``[B, H, Lp, K]`` fp32.
        v_beta: ``v * beta`` ``[B, H, Lp, V]`` fp32.
        g_cumsum: Per-chunk prefix sums of the log-decay ``[B, H, Lp, K]`` fp32.
        initial_state: ``[B, H, K, V]`` fp32.
        segment_meta: ``[B, C, 8, 128]`` from :func:`pack_segment_meta`, or ``None``.
        chunk_size: ``cs`` (power of two, ``8 <= cs <= 128``).
        block_h: Heads per grid step.
        precision: Precision of the non-inverse dots.
        vmem_limit_bytes: Scoped VMEM requested from Mosaic.

    Returns:
        ``(core_out [B, H, Lp, V], final_state [B, H, K, V], chunk_states_in [B, H, C, K, V])`` fp32; the last is the
        state ENTERING every chunk (the backward's residual).
    """
    B, H, Lp, K = q.shape
    V = v_beta.shape[-1]
    cs = chunk_size
    C = Lp // cs
    segmented = segment_meta is not None
    lay = fwd_block_layout(block_h, cs, K, V, segmented)
    check_tile_alignment(lay)

    def chunk_spec(name):
        return pl.BlockSpec(lay[name], lambda b, hb, c: (b, hb, c, 0))

    def state_spec(name):
        return pl.BlockSpec(lay[name], lambda b, hb, c: (b, hb, 0, 0))

    in_specs = [
        chunk_spec("q"),
        chunk_spec("k"),
        chunk_spec("k_beta"),
        chunk_spec("v_beta"),
        chunk_spec("g_cumsum"),
        state_spec("initial_state"),
    ]
    args = [q, k, k_beta, v_beta, g_cumsum, initial_state]
    if segmented:
        in_specs.append(pl.BlockSpec(lay["segment_meta"], lambda b, hb, c: (b, c, 0, 0)))
        args.append(segment_meta)
    out_specs = [
        chunk_spec("core_out"),
        state_spec("final_state"),
        pl.BlockSpec(lay["chunk_states_in"], lambda b, hb, c: (b, hb, c, 0, 0)),
    ]
    f32 = jnp.float32
    out_shape = [
        jax.ShapeDtypeStruct((B, H, Lp, V), f32),
        jax.ShapeDtypeStruct((B, H, K, V), f32),
        jax.ShapeDtypeStruct((B, H, C, K, V), f32),
    ]
    kernel = functools.partial(
        _fwd_kernel, block_h=block_h, num_chunks=C, segmented=segmented, precision=resolve_precision(precision)
    )
    out, final, states_in = pl.pallas_call(
        kernel,
        grid=(B, H // block_h, C),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=[pltpu.VMEM(lay["scratch:state"], f32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_bytes),
        ),
    )(*args)
    return out, final, states_in
