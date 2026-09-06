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

"""Pallas TPU kernel for compressed-window full-sequence attention (DeepSeek-V4).

This is the compute-bound sibling of the decode kernel: it serves DeepSeek-V4's
training / prefill forward (and the eSurge prefill), where the KV axis (the
sliding-window token KVs concatenated with the compressed entries) can be long
and therefore cannot all sit in VMEM. The kernel is a flash-attention-style
tiled scan over the KV axis with an explicit ``pltpu.make_async_copy``
double-buffer (issue the DMA for KV tile ``N+1`` while the MXU computes tile
``N`` — the gmm-v3 / ragged-page-attention idiom) and an online (flash) softmax
that folds the per-head learnable sink into the running normaliser::

    grid = (batch, 1, num_q_blocks)       # all heads per program; axes parallel

    per (b, qb):
        q_b   [num_heads, block_q, head_dim]  (VMEM resident, one q block)
        acc, m, l  online-softmax state (registers), seeded by the sink
        for p in range(n_tok_iter + n_entry_tiles):   # double-buffered HBM->VMEM
            tile = actual_tile(p)         # band-anchored token tile, or entry tile
            wait(tile); prefetch(next)
            kv_t   [block_kv, head_dim]
            bias_t [block_q, block_kv]
            logits = scale * q_b @ kv_t^T + bias_t
            m, l, acc  <- online update (K == V: kv_t is both key and value)
        out = acc / l                          # sink mass already in l

**Sliding-window band-skip** (``window > 0``): the token region is padded to a
tile boundary and the compressed entries placed after it; each query block only
visits the ``ceil((block_q+window)/block_kv)+1`` token tiles that can hold a
visible key (a contiguous window anchored at a q-block-dependent, clamped
``tile_lo``) plus every entry tile — O(S*window) rather than O(S^2), a measured
~1.35-2.71x forward win that scales with S. Skipped token tiles are provably
``_MASK_MIN``-masked. Fully masked sink-free rows use a once-per-batch uniform
weighted sum over all real KVs, so skipping and padding preserve the dense
reference's uniform-average behavior.

``K == V`` (the same ``kv`` tensor is both the key in the ``q @ kv^T`` score
and the value in ``p @ kv``), exactly as the dense reference and the decode
kernel. Masking uses a large *finite* sentinel (``finfo(f32).min``) rather than
``-inf`` so the online recurrence never produces ``inf - inf`` NaNs; a
fully-masked no-sink query row degenerates to a uniform average, matching the
XLA reference. Matmul precision is dtype-aware: f32 -> ``Precision.HIGHEST``
(multi-pass MXU, matching the HIGHEST XLA reference), bf16 -> native single-pass.

The forward Pallas kernel is a Mosaic custom call and is therefore not
auto-differentiable. :func:`compressed_window_attention_tpu` wraps it in a
``jax.custom_jvp`` whose derivative uses checkpointed query tiles of the
reference math. This supports forward and reverse AD without retaining the
full head-expanded score matrix. Dense bias storage and unrestricted-mask
quadratic arithmetic remain; this is not a flash-style band-skipping backward.
A structured sparse-bias interface and band-skipping derivative remain follow-ups.
"""

from __future__ import annotations

import functools
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ejkernel.callib import ejit
from ejkernel.ops import FwdParams

# Large finite sentinel used for masked / padded logits. Finite (not -inf) so
# the online-softmax recurrence (max/exp/rescale) never evaluates inf - inf.
_MASK_MIN = float(jnp.finfo(jnp.float32).min)

#: TPU lane-tile width. Mosaic tiles a 2-D+ HBM memref ``(1, 128)`` over its
#: last two dims, so any partial slice of the last dim must be a multiple of
#: this. See :func:`_legal_block_kv`.
_LANE = 128


def _round_up(x: int, multiple: int) -> int:
    """Round ``x`` up to the nearest multiple of ``multiple``."""
    return ((x + multiple - 1) // multiple) * multiple


def _vmem_safe_block_q(block_q: int, num_heads: int, d_pad: int, cap_elems: int = 180_000) -> int:
    """Shrink ``block_q`` so the ``[num_heads, block_q, d_pad]`` buffers fit VMEM.

    All heads are processed per program, so the query / accumulator / output
    buffers are ``num_heads * block_q * d_pad`` fp32 elements each; a handful of
    them must stay under the ~16 MB scoped VMEM budget. ``cap_elems`` bounds one
    such buffer's element count. The result is a multiple of 8 (VPU sublane) and
    at least 8.

    Args:
        block_q: Requested query tile size.
        num_heads: Query head count (per device).
        d_pad: Padded head dimension.
        cap_elems: Element cap for a single ``[num_heads, block_q, d_pad]`` buffer.

    Returns:
        A VMEM-safe query tile size (multiple of 8, in ``[8, block_q]``).
    """
    max_bq = max(8, (cap_elems // (num_heads * d_pad)) // 8 * 8)
    return max(8, min(block_q, max_bq))


def _dense_reference(
    query: Array,
    kv: Array,
    bias: Array,
    softmax_aux: Array | None,
    softmax_scale: float,
) -> Array:
    """Dense sink-augmented shared-KV attention (differentiable, fp32 softmax).

    Numerically identical to the registered XLA fallback
    (``kernels/_xla/compressed_window_attention``); duplicated here so the
    Pallas backend directory stays self-contained (no cross-backend import) and
    the tiled differentiation rule can recompute reference gradients. The
    gradient-parity test enforces that this stays in sync with the XLA fallback.

    Args:
        query: Queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared KV axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive fp32 bias ``[batch, q_len, kv_len]`` (head-broadcast).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    logits = jnp.einsum("bhsd,bld->bhsl", query, kv, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)
    # Keep the bf16 dot/f32 softmax boundary explicit under compiled AD.
    # Without this barrier TPU fusion produces NaN bf16 JVPs for masked rows;
    # the same expression evaluated eagerly remains finite.
    logits = jax.lax.optimization_barrier(logits)
    logits = logits * softmax_scale + bias[:, None, :, :].astype(jnp.float32)
    kv_len = logits.shape[-1]
    if softmax_aux is not None:
        num_heads = logits.shape[1]
        sinks = jnp.broadcast_to(
            softmax_aux.astype(jnp.float32).reshape(1, num_heads, 1, 1),
            (*logits.shape[:3], 1),
        )
        combined = jnp.concatenate([logits, sinks], axis=-1)
    else:
        combined = logits
    combined = combined - jnp.max(combined, axis=-1, keepdims=True)
    probs = jax.nn.softmax(combined, axis=-1)
    scores = probs[..., :kv_len].astype(kv.dtype)
    return jnp.einsum("bhsl,bld->bhsd", scores, kv, precision=jax.lax.Precision.HIGHEST)


def _tiled_reference(
    query: Array,
    kv: Array,
    bias: Array,
    softmax_aux: Array | None,
    softmax_scale: float,
    *,
    block_q: int = 64,
) -> Array:
    """Exact reference with rematerialized, bounded-query score buffers.

    Query rows are independent. Processing and checkpointing each tile avoids
    retaining a full ``[batch, heads, q_len, kv_len]`` score/probability tensor
    in either AD direction. Peak per-tile scores scale with ``block_q*kv_len``.
    This does not remove the caller's dense ``[batch, q_len, kv_len]`` bias or
    its cotangent, nor the quadratic arithmetic of an unrestricted mask.

    Args:
        query: Queries ``[batch, heads, q_len, head_dim]``.
        kv: Shared keys/values ``[batch, kv_len, head_dim]``.
        bias: Head-broadcast additive bias ``[batch, q_len, kv_len]``.
        softmax_aux: Optional per-head sink logits.
        softmax_scale: Query/key dot-product scale.
        block_q: Maximum query rows in one rematerialized tile.

    Returns:
        Attention outputs with the query shape.
    """
    batch, heads, q_len, dim = query.shape
    if q_len == 0:
        return jnp.zeros_like(query)
    tile = min(max(1, block_q), q_len)
    tiles = (q_len + tile - 1) // tile
    pad = tiles * tile - q_len
    q_padded = jnp.pad(query, ((0, 0), (0, 0), (0, pad), (0, 0)))
    # Finite dummy rows avoid NaNs in masked-out padding cotangents when
    # sinks are disabled. The dummy outputs are discarded below.
    b_padded = jnp.pad(bias, ((0, 0), (0, pad), (0, 0)))

    @jax.checkpoint
    def attend_tile(index):
        q = jax.lax.dynamic_slice_in_dim(q_padded, index * tile, tile, axis=2)
        b = jax.lax.dynamic_slice_in_dim(b_padded, index * tile, tile, axis=1)
        return _dense_reference(q, kv, b, softmax_aux, softmax_scale)

    blocks = jax.lax.map(attend_tile, jnp.arange(tiles, dtype=jnp.int32))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(batch, heads, tiles * tile, dim)[:, :, :q_len]


def _flash_fwd_kernel(
    q_ref,
    kv_hbm_ref,
    bias_hbm_ref,
    sink_ref,
    uniform_ref,
    o_ref,
    kv_x2,
    bias_x2,
    kv_sem,
    bias_sem,
    *,
    softmax_scale: float,
    use_sink: bool,
    block_kv: int,
    n_tok_iter: int,
    n_token_tiles: int,
    n_entry_tiles: int,
    band_skip: bool,
    window: int,
):
    """Tiled online-softmax attention for one ``(batch, head_block, q_block)`` program.

    Streams the KV axis in ``block_kv`` tiles from HBM into a double-buffered
    VMEM scratch via ``pltpu.make_async_copy`` (prefetch tile ``p + 1`` while
    computing tile ``p``) and accumulates a flash-style online softmax whose
    running normaliser is seeded by the per-head sink. A small block of heads is
    processed per program (batched over the head axis); the shared KV / bias are
    head-broadcast and re-streamed once per head block.

    Sliding-window band-skip: when ``band_skip`` the loop visits only the
    ``n_tok_iter`` token tiles that can hold a visible key for this q-block (a
    contiguous window anchored at a q-block-dependent ``tile_lo``) instead of all
    ``n_token_tiles`` token tiles, turning the token cost from O(S) to O(W) per
    q-block. The ``n_entry_tiles`` compressed-entry tiles (which sit after the
    token region and are always potentially visible) are always scanned. Skipped
    token tiles are provably fully masked (their bias is the ``_MASK_MIN``
    sentinel). Fully masked sink-free rows select ``uniform_ref`` to include
    skipped real KVs and exclude physical padding, matching the dense scan.

    Args:
        q_ref: Query block VMEM ref ``[block_h, block_q, head_dim]``.
        kv_hbm_ref: Full KV tensor in HBM ``[batch, kv_pad, head_dim]`` (token
            region padded to a tile boundary, then the compressed entries).
        bias_hbm_ref: Full bias tensor in HBM ``[batch, q_pad, kv_pad]``.
        sink_ref: This head block's sink logits VMEM ref ``[block_h]`` (fp32).
        uniform_ref: Uniform weighted sum of all real KV rows ``[1, head_dim]``;
            used only for fully masked sink-free rows (including skipped tiles).
        o_ref: Output block VMEM ref ``[block_h, block_q, head_dim]``.
        kv_x2: Double-buffered KV VMEM scratch ``[2, block_kv, head_dim]``.
        bias_x2: Double-buffered bias VMEM scratch ``[2, block_q, block_kv]``.
        kv_sem: DMA semaphore array ``[2]`` for the KV copies.
        bias_sem: DMA semaphore array ``[2]`` for the bias copies.
        softmax_scale: Logit scale.
        use_sink: Whether to seed the softmax with the per-head sink column.
        block_kv: KV tile width.
        n_tok_iter: Number of token tiles visited per q-block (``n_token_tiles``
            without band-skip, else the static visible-window width).
        n_token_tiles: Total token-region tiles (entries start after these).
        n_entry_tiles: Number of compressed-entry tiles (always scanned).
        band_skip: Whether the token window is anchored at a dynamic ``tile_lo``.
        window: Sliding-window size (used to anchor ``tile_lo``).
    """
    b = pl.program_id(0)
    qb = pl.program_id(2)
    q = q_ref[...]  # [block_h, block_q, D] (batch None squeezed), kept in input dtype
    block_h, block_q, head_dim = q.shape
    # Matmul precision policy: for f32 inputs use HIGHEST (multi-pass bf16 MXU ->
    # accurate f32, matching the HIGHEST XLA reference); for bf16 inputs the data
    # is already bf16 so the native single-pass MXU (DEFAULT) is both correct at
    # the bf16 floor and far cheaper. The softmax/accumulation stay f32 either way.
    mm_dtype = q.dtype
    prec = jax.lax.Precision.HIGHEST if mm_dtype == jnp.float32 else jax.lax.Precision.DEFAULT

    # First visible token tile for this q-block. The window [tile_lo, tile_lo +
    # n_tok_iter) is clamped to stay within [0, n_token_tiles) and always covers
    # every visible token key (oldest = qb*block_q - window + 1); tiles past the
    # visible range are masked by the bias.
    if band_skip:
        raw_lo = (qb * block_q - window + 1) // block_kv
        tile_lo = jnp.clip(raw_lo, 0, n_token_tiles - n_tok_iter)
    else:
        tile_lo = 0

    total = n_tok_iter + n_entry_tiles  # static number of scanned tiles

    def _actual_tile(p: int):
        """Map scan position ``p`` (static) to its KV tile index (maybe dynamic)."""
        if p < n_tok_iter:
            return tile_lo + p  # token window (dynamic anchor when band_skip)
        return n_token_tiles + (p - n_tok_iter)  # compressed-entry tile (static)

    def _start(tile, buf):
        """Issue the async HBM->VMEM copies for KV ``tile`` into buffer ``buf``."""
        pltpu.make_async_copy(
            kv_hbm_ref.at[b, pl.ds(tile * block_kv, block_kv), :],
            kv_x2.at[buf],
            kv_sem.at[buf],
        ).start()
        pltpu.make_async_copy(
            bias_hbm_ref.at[b, pl.ds(qb * block_q, block_q), pl.ds(tile * block_kv, block_kv)],
            bias_x2.at[buf],
            bias_sem.at[buf],
        ).start()

    def _wait(tile, buf):
        """Wait for KV ``tile``'s copies into buffer ``buf`` to complete."""
        pltpu.make_async_copy(
            kv_hbm_ref.at[b, pl.ds(tile * block_kv, block_kv), :],
            kv_x2.at[buf],
            kv_sem.at[buf],
        ).wait()
        pltpu.make_async_copy(
            bias_hbm_ref.at[b, pl.ds(qb * block_q, block_q), pl.ds(tile * block_kv, block_kv)],
            bias_x2.at[buf],
            bias_sem.at[buf],
        ).wait()

    if use_sink:
        sink = sink_ref[...].astype(jnp.float32)  # [block_h] (static slice, no dynamic index)
        m = jnp.broadcast_to(sink[:, None], (block_h, block_q))  # seed max at the sink
        ell = jnp.ones((block_h, block_q), jnp.float32)  # exp(sink - sink) = 1
    else:
        m = jnp.full((block_h, block_q), _MASK_MIN, jnp.float32)
        ell = jnp.zeros((block_h, block_q), jnp.float32)
    acc = jnp.zeros((block_h, block_q, head_dim), jnp.float32)

    _start(_actual_tile(0), 0)
    for p in range(total):
        cur = p % 2
        _wait(_actual_tile(p), cur)
        if p + 1 < total:
            _start(_actual_tile(p + 1), (p + 1) % 2)
        kv_t = kv_x2[cur]  # [block_kv, D] (input dtype)
        bias_t = bias_x2[cur].astype(jnp.float32)  # [block_q, block_kv]

        # logits[h, sq, sk] = scale * sum_d q[h, sq, d] * kv_t[sk, d] + bias_t[sq, sk].
        logits = jax.lax.dot_general(
            q, kv_t, (((2,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=prec
        )
        logits = logits * softmax_scale + bias_t[None, :, :]  # [block_h, block_q, block_kv]

        tile_max = jnp.max(logits, axis=-1)  # [block_h, block_q]
        m_new = jnp.maximum(m, tile_max)
        corr = jnp.exp(m - m_new)  # finite operands only -> no NaN
        p = jnp.exp(logits - m_new[..., None])  # [block_h, block_q, block_kv]
        ell = ell * corr + jnp.sum(p, axis=-1)
        pv = jax.lax.dot_general(
            p.astype(mm_dtype), kv_t, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32, precision=prec
        )
        acc = acc * corr[..., None] + pv
        m = m_new

    ell = jnp.where(ell == 0.0, 1.0, ell)
    out = acc / ell[..., None]
    if not use_sink:
        # Finite masking makes an all-masked row uniform over *real* KVs.
        # Its online state instead includes physical zero padding and omits
        # band-skipped real KVs. Correct only these rows, using a single O(KD)
        # per-batch reduction rather than rescanning all KVs for every query.
        out = jnp.where(m[..., None] == _MASK_MIN, uniform_ref[...][None, :, :], out)
    o_ref[...] = out.astype(o_ref.dtype)


def _flash_forward(
    query: Array,
    kv: Array,
    bias: Array,
    sink: Array,
    softmax_scale: float,
    use_sink: bool,
    block_q: int,
    block_kv: int,
    window: int,
    token_kv_len: int,
) -> Array:
    """Set up and launch the tiled-flash forward Pallas kernel (pure forward).

    Pads the head dim to a 128 multiple (zeros — inert in the dot products), the
    query length to ``block_q`` (garbage rows sliced off), and the KV length to
    ``block_kv`` (pad rows carry a ``_MASK_MIN`` bias). Fully masked sink-free
    rows select a separate uniform weighted sum that excludes physical padding.

    When ``window > 0`` the sliding-window token region ``kv[:, :token_kv_len]``
    is padded to a tile boundary and the compressed entries ``kv[:, token_kv_len:]``
    placed after it, so the kernel can skip token tiles outside each q-block's
    window (see ``_flash_fwd_kernel``). ``window == 0`` disables band-skip and
    scans the whole (contiguous) KV axis — the generic, mask-agnostic path.

    Args:
        query: Queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared KV axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive fp32 bias ``[batch, q_len, kv_len]`` (head-broadcast).
        sink: Per-head sink logits ``[num_heads]`` (fp32); ignored when
            ``use_sink`` is False.
        softmax_scale: Logit scale.
        use_sink: Whether to include the sink softmax column.
        block_q: Query tile size.
        block_kv: KV tile size.
        window: Sliding-window size; ``0`` disables band-skip.
        token_kv_len: Length of the sliding-window token region at the front of
            the KV axis (the rest are always-visible compressed entries). Only
            used when ``window > 0``.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    batch, num_heads, q_len, head_dim = query.shape
    kv_len = kv.shape[1]

    d_pad = _round_up(head_dim, 128)
    # Process all heads in one program: the shared KV / bias are read once per
    # q-block (not re-streamed per head) and no in-kernel head selection is
    # needed (the full sink array is resident). Bound the ~16 MB scoped VMEM by
    # shrinking block_q for wide head counts.
    block_h = num_heads
    block_q = _vmem_safe_block_q(block_q, num_heads, d_pad)
    q_pad = _round_up(q_len, block_q)

    q_p = query
    if d_pad != head_dim:
        q_p = jnp.pad(q_p, ((0, 0), (0, 0), (0, 0), (0, d_pad - head_dim)))
    if q_pad != q_len:
        q_p = jnp.pad(q_p, ((0, 0), (0, 0), (0, q_pad - q_len), (0, 0)))

    band = window > 0 and 0 < token_kv_len <= kv_len
    tok_len = token_kv_len if band else kv_len  # token-region length
    n_entries = kv_len - tok_len  # compressed-entry rows

    token_kv_pad = _round_up(tok_len, block_kv)
    entries_pad = _round_up(n_entries, block_kv)
    n_token_tiles = token_kv_pad // block_kv
    n_entry_tiles = entries_pad // block_kv

    def _pad_d(x):
        """Pad the trailing feature dim to ``d_pad`` (zeros; inert in dots)."""
        return x if d_pad == head_dim else jnp.pad(x, ((0, 0), (0, 0), (0, d_pad - head_dim)))

    # Token region padded to a tile boundary, then the (tile-boundary-padded)
    # entries -> token tiles and entry tiles are cleanly separated.
    token_kv = _pad_d(kv[:, :tok_len, :])
    token_bias = bias[:, :, :tok_len]
    token_kv = jnp.pad(token_kv, ((0, 0), (0, token_kv_pad - tok_len), (0, 0)))
    token_bias = jnp.pad(token_bias, ((0, 0), (0, 0), (0, token_kv_pad - tok_len)), constant_values=_MASK_MIN)
    if n_entry_tiles:
        entry_kv = _pad_d(kv[:, tok_len:, :])
        entry_bias = bias[:, :, tok_len:]
        entry_kv = jnp.pad(entry_kv, ((0, 0), (0, entries_pad - n_entries), (0, 0)))
        entry_bias = jnp.pad(entry_bias, ((0, 0), (0, 0), (0, entries_pad - n_entries)), constant_values=_MASK_MIN)
        kv_p = jnp.concatenate([token_kv, entry_kv], axis=1)
        bias_p = jnp.concatenate([token_bias, entry_bias], axis=2)
    else:
        kv_p, bias_p = token_kv, token_bias
    if q_pad != q_len:
        bias_p = jnp.pad(bias_p, ((0, 0), (0, q_pad - q_len), (0, 0)))

    # Static visible token-tile window width: enough tiles to cover any q-block's
    # (block_q + window)-key span at any alignment. Band-skip only pays off once
    # there are more token tiles than that.
    n_vis_tok = _round_up(block_q + window, block_kv) // block_kv + 1
    band_skip = band and n_token_tiles > n_vis_tok
    n_tok_iter = n_vis_tok if band_skip else n_token_tiles

    sink_p = sink.astype(jnp.float32).reshape(num_heads)

    # The finite-sentinel dense softmax is uniform on fully masked sink-free
    # rows. Compute its value over the original (unpadded, unskipped) KV axis.
    # Cast probabilities before the dot just like _dense_reference, notably
    # for bf16 where mean(kv) need not equal this quantized weighted sum.
    if not use_sink:
        weights = jnp.full((1, kv_len), 1.0 / kv_len, jnp.float32).astype(kv.dtype)
        uniform = jnp.einsum("sl,bld->bsd", weights, kv, precision=jax.lax.Precision.HIGHEST)
        uniform_p = _pad_d(uniform)
    else:
        uniform_p = jnp.zeros((batch, 1, d_pad), kv.dtype)

    # Auto-interpret off TPU: the Mosaic lowering only exists on TPU, so on
    # CPU/GPU the kernel runs in Pallas interpret mode (emulated VMEM + DMA
    # semaphores). This lets the numerics be validated host-side; production
    # only routes here on TPU (the model defaults to the XLA reference).
    interpret = jax.default_backend() != "tpu"

    # One program per (batch, head_block, q_block): a small block of heads is
    # resident (VMEM-safe on the 16 MB scoped budget), the shared KV / bias are
    # head-broadcast and re-streamed once per head block (MQA), and all grid axes
    # are independent -> "parallel" (megacore split). ``block_h`` divides
    # ``num_heads`` so no head padding is needed.
    grid = (batch, num_heads // block_h, q_pad // block_q)
    out = pl.pallas_call(
        functools.partial(
            _flash_fwd_kernel,
            softmax_scale=softmax_scale,
            use_sink=use_sink,
            block_kv=block_kv,
            n_tok_iter=n_tok_iter,
            n_token_tiles=n_token_tiles,
            n_entry_tiles=n_entry_tiles,
            band_skip=band_skip,
            window=window,
        ),
        grid=grid,
        in_specs=[
            pl.BlockSpec((None, block_h, block_q, d_pad), lambda b, hb, qb: (b, hb, qb, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv: full tensor in HBM (streamed)
            pl.BlockSpec(memory_space=pltpu.HBM),  # bias: full tensor in HBM (streamed)
            pl.BlockSpec((block_h,), lambda b, hb, qb: (hb,)),  # this head block's sinks
            pl.BlockSpec((None, 1, d_pad), lambda b, hb, qb: (b, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, block_h, block_q, d_pad), lambda b, hb, qb: (b, hb, qb, 0)),
        out_shape=jax.ShapeDtypeStruct((batch, num_heads, q_pad, d_pad), kv.dtype),
        scratch_shapes=[
            pltpu.VMEM((2, block_kv, d_pad), kv_p.dtype),
            pltpu.VMEM((2, block_q, block_kv), bias_p.dtype),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=interpret,
    )(q_p, kv_p, bias_p, sink_p, uniform_p)

    out = out[:, :, :q_len, :]
    if d_pad != head_dim:
        out = out[..., :head_dim]
    return out


@partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _cwa_pallas(
    query: Array,
    kv: Array,
    bias: Array,
    sink: Array,
    softmax_scale: float,
    use_sink: bool,
    block_q: int,
    block_kv: int,
    window: int,
    token_kv_len: int,
) -> Array:
    """Pallas primal with an exact XLA differentiation rule.

    The custom JVP makes both forward-mode (``jax.jvp``/``jacfwd``) and
    reverse-mode (the transpose of this linear tangent rule) available.  The
    primal remains the tiled Pallas kernel; only the tangent computation uses
    the numerically equivalent reference expression.
    """
    return _flash_forward(query, kv, bias, sink, softmax_scale, use_sink, block_q, block_kv, window, token_kv_len)


@_cwa_pallas.defjvp
def _cwa_pallas_jvp(
    softmax_scale,
    use_sink,
    block_q,
    block_kv,
    window,
    token_kv_len,
    primals,
    tangents,
):
    """Differentiate every trainable operand, including indexer bias and sink."""
    query, kv, bias, sink = primals
    dquery, dkv, dbias, dsink = tangents
    out = _flash_forward(
        query,
        kv,
        bias,
        sink,
        softmax_scale,
        use_sink,
        block_q,
        block_kv,
        window,
        token_kv_len,
    )

    def _ref(q, k, b, s):
        return _tiled_reference(q, k, b, s if use_sink else None, softmax_scale)

    _, tangent = jax.jvp(_ref, (query, kv, bias, sink), (dquery, dkv, dbias, dsink))
    return out, tangent


def _default_blocks(q_len: int, kv_len: int, fwd_params: FwdParams | None) -> tuple[int, int]:
    """Pick ``(block_q, block_kv)`` from ``fwd_params`` or adaptive defaults.

    Defaults cap tiles at 128 (a friendly MXU/VPU width) but shrink for short
    axes so tiny shapes are not padded to 128 unnecessarily.

    Args:
        q_len: Query length.
        kv_len: KV axis length.
        fwd_params: Optional tiling hints (``q_blocksize`` / ``kv_blocksize``).

    Returns:
        Tuple ``(block_q, block_kv)``.
    """
    block_q = None if fwd_params is None else getattr(fwd_params, "q_blocksize", None)
    block_kv = None if fwd_params is None else getattr(fwd_params, "kv_blocksize", None)
    if block_q is None:
        block_q = min(128, _round_up(q_len, 8))
    if block_kv is None:
        block_kv = min(128, _round_up(kv_len, 8))
    return int(block_q), int(block_kv)


def _legal_block_kv(block_kv: int) -> int:
    """Round ``block_kv`` up to the lane tile, which the bias DMA requires.

    The per-tile bias copy is
    ``bias_hbm_ref.at[b, ds(qb * block_q, block_q), ds(tile * block_kv, block_kv)]``.
    Mosaic tiles that HBM buffer ``(1, 128)`` over its last two dims and
    rejects a ``memref_slice`` whose extent along the *lane* (last) dim is not
    a multiple of 128 -- "Slice shape along dimension 2 must be aligned to
    tiling (128), but is 24". That holds even when the slice spans the whole
    dimension, so the tile width, not just its alignment, has to be a lane
    multiple; and because the padded KV extents are built as multiples of
    ``block_kv``, fixing the tile fixes every offset too.

    This overrides the adaptive default's "don't pad tiny axes to 128" rule
    (``min(128, round_up(kv_len, 8))``), which produced tile widths like 24
    and 40 that cannot lower on TPU at all. A user-supplied
    ``kv_blocksize`` is rounded up for the same reason.

    Args:
        block_kv: Proposed KV tile width.

    Returns:
        ``block_kv`` rounded up to a multiple of :data:`_LANE`.
    """
    return _round_up(block_kv, _LANE)


@ejit(static_argnames=["softmax_scale", "fwd_params", "window", "token_kv_len"])
def compressed_window_attention_tpu(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len kv_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    window: int = 0,
    token_kv_len: int = 0,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Pallas TPU compressed-window full-sequence attention (differentiable).

    Args:
        query: Queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared KV axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive fp32 bias ``[batch, q_len, kv_len]`` (head-broadcast).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Optional tiling hints (``q_blocksize`` / ``kv_blocksize``).
        window: Sliding-window size of the token-KV region. ``0`` (default)
            disables band-skip and scans the whole (contiguous) KV axis. When
            ``> 0`` the forward skips out-of-window token tiles (O(S*window)) and
            the backward recomputes through the matching windowed reference.
        token_kv_len: Length of the sliding-window token region at the front of
            the KV axis (the rest are always-visible compressed entries). Only
            used when ``window > 0``; defaults to ``kv_len`` (no entries).

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    num_heads = query.shape[1]
    head_dim = query.shape[-1]
    q_len = query.shape[2]
    kv_len = kv.shape[1]
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    block_q, block_kv = _default_blocks(q_len, kv_len, fwd_params)
    tkl = kv_len if (window <= 0 or token_kv_len <= 0) else int(token_kv_len)
    block_kv = _legal_block_kv(block_kv)

    use_sink = softmax_aux is not None
    sink = jnp.zeros((num_heads,), jnp.float32) if softmax_aux is None else softmax_aux.astype(jnp.float32)
    return _cwa_pallas(query, kv, bias, sink, scale, use_sink, block_q, block_kv, int(window), tkl)
