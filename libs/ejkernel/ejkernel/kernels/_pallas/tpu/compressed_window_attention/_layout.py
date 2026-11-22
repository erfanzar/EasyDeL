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

"""Tile layout shared by the compressed-window attention forward and backward Pallas kernels.

The KV axis is ``[token region | compressed entries]``. Both regions are padded to a ``block_kv`` boundary
separately, so token tiles and entry tiles never mix; the query axis is padded to ``block_q`` and the head dim to a
lane multiple. :class:`Tiling` is the static (hashable) description every kernel launch derives its grid, index
maps and visit pattern from:

* **band skip** (``window > 0``): a q-block visits only the ``n_tok_iter`` token tiles that can hold a key of its
  ``(block_q + window)``-wide window; every entry tile is always visited.
* **causal skip** (``causal=True``, dense bias only): the caller promises the bias closes every key
  ``j > s + diag_offset``; token tiles entirely above a q-block's diagonal are skipped.

Skipped tiles hold only the finite ``MASK_MIN`` sentinel, so with a sink (or any visible key) the skip is exact.

Tile sizes follow the local VMEM: on a 16 MiB part (v4, or off TPU) the historical 180k-element query-block cap
applies; on larger parts the tiles are kept inside an envelope measured on a v5p (64 MiB VMEM, 16 heads x 512,
bf16): every tiling with a ``[H, block_q, d]`` query block of at most 2 MiB and an fp32 ``[H, block_q, block_kv]``
intermediate of at most ``16 * 128 * 512`` elements compiled and ran under a 48 MiB scoped limit, while
``(256, 128)`` overflowed 48 MiB and ``(512, 512)`` needed 96 MiB. Measured per DSA-like layer at 32k tokens
(fwd+bwd): ``(16, 128)`` 2,025 ms, ``(128, 128)`` 467 ms, ``(128, 256)`` 380 ms, ``(128, 512)`` 363 ms.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ejkernel.ops import FwdParams

MASK_MIN = float(jnp.finfo(jnp.float32).min)
MASKED_ROW_MAX = 0.5 * MASK_MIN
LANE = 128
DEFAULT_VMEM_BYTES = 16 * 2**20

_LEGACY_BLOCK_Q_CAP_ELEMS = 180_000
_Q_BLOCK_BYTES_MAX = 16 * 128 * 512 * 2
_INTERMEDIATE_ELEMS_MAX = 16 * 128 * 512
_LARGE_PART_VMEM_LIMIT = 64 * 2**20


def round_up(x: int, multiple: int) -> int:
    """Round ``x`` up to the nearest multiple of ``multiple``."""
    return ((x + multiple - 1) // multiple) * multiple


@functools.lru_cache(maxsize=1)
def tpu_vmem_capacity_bytes() -> int:
    """Physical VMEM of the local TPU generation (``pltpu.get_tpu_info()``), or 16 MiB off TPU / on failure."""
    if jax.default_backend() != "tpu":
        return DEFAULT_VMEM_BYTES
    try:
        capacity = int(getattr(pltpu.get_tpu_info(), "vmem_capacity_bytes", 0) or 0)
    except Exception:
        return DEFAULT_VMEM_BYTES
    return capacity if capacity > 0 else DEFAULT_VMEM_BYTES


def vmem_safe_block_q(block_q: int, num_heads: int, d_pad: int, cap_elems: int = _LEGACY_BLOCK_Q_CAP_ELEMS) -> int:
    """Shrink ``block_q`` so one ``[num_heads, block_q, d_pad]`` fp32 buffer stays under ``cap_elems`` (16 MiB parts).

    Args:
        block_q: Requested query tile size.
        num_heads: Query head count (per device).
        d_pad: Padded head dimension.
        cap_elems: Element cap for a single ``[num_heads, block_q, d_pad]`` buffer.

    Returns:
        A query tile size that is a multiple of 8 in ``[8, block_q]``.
    """
    max_bq = max(8, (cap_elems // (num_heads * d_pad)) // 8 * 8)
    return max(8, min(block_q, max_bq))


def default_blocks(q_len: int, kv_len: int, fwd_params: FwdParams | None, capacity: int) -> tuple[int, int]:
    """Requested ``(block_q, block_kv)`` from ``fwd_params`` or the defaults for this VMEM capacity.

    ``block_q`` defaults to ``min(128, round_up(q_len, 8))``. ``block_kv`` defaults to 512 on parts with more than
    16 MiB of VMEM when the KV axis has at least 512 rows, else 128; any value is rounded up to the 128-lane tile,
    because Mosaic rejects HBM slices whose lane extent is not a multiple of 128 (a short prefill would otherwise
    fail to compile).

    Args:
        q_len: Query length.
        kv_len: KV axis length.
        fwd_params: Optional tiling hints (``q_blocksize`` / ``kv_blocksize``).
        capacity: VMEM capacity in bytes.

    Returns:
        Tuple ``(block_q, block_kv)`` before VMEM fitting.
    """
    block_q = None if fwd_params is None else getattr(fwd_params, "q_blocksize", None)
    block_kv = None if fwd_params is None else getattr(fwd_params, "kv_blocksize", None)
    if block_q is None:
        block_q = min(128, round_up(q_len, 8))
    if block_kv is None:
        block_kv = 512 if (capacity > DEFAULT_VMEM_BYTES and kv_len >= 512) else 128
    return int(block_q), round_up(int(block_kv), LANE)


def vmem_safe_blocks(
    block_q: int,
    block_kv: int,
    num_heads: int,
    d_pad: int,
    itemsize: int,
    capacity: int,
) -> tuple[int, int, int | None]:
    """``(block_q, block_kv, vmem_limit)``: tiles that fit ``capacity`` and the scoped VMEM limit to request.

    On a 16 MiB part the legacy rule applies (``block_q`` capped by :func:`vmem_safe_block_q`, ``block_kv`` kept,
    default scoped limit). On larger parts ``block_q`` is halved (floor 16) until the ``[H, block_q, d]`` query block
    is within 2 MiB and ``block_kv`` halved (floor 128) until the fp32 ``[H, block_q, block_kv]`` intermediate is
    within ``16 * 128 * 512`` elements, and Mosaic is asked for ``min(capacity - 1 MiB, 64 MiB)``. ``block_q`` is
    always a multiple of 16 (bf16 sublane packing).

    Args:
        block_q: Requested query tile.
        block_kv: Requested KV tile (a multiple of 128).
        num_heads: Query heads per program.
        d_pad: Padded head dim.
        itemsize: Query element size in bytes.
        capacity: VMEM capacity in bytes.

    Returns:
        Tuple ``(block_q, block_kv, vmem_limit_bytes or None)``.
    """
    if capacity <= DEFAULT_VMEM_BYTES:
        block_q = vmem_safe_block_q(block_q, num_heads, d_pad)
        return max(16, (block_q // 16) * 16), block_kv, None
    block_q = max(16, (block_q // 16) * 16)
    block_kv = max(LANE, round_up(block_kv, LANE))
    while num_heads * block_q * d_pad * itemsize > _Q_BLOCK_BYTES_MAX and block_q > 16:
        block_q = max(16, ((block_q // 2) // 16) * 16)
    while num_heads * block_q * block_kv > _INTERMEDIATE_ELEMS_MAX and block_kv > LANE:
        block_kv = max(LANE, round_up(block_kv // 2, LANE))
    return block_q, block_kv, int(min(capacity - 2**20, _LARGE_PART_VMEM_LIMIT))


@dataclasses.dataclass(frozen=True)
class Tiling:
    """Static tile layout of one compressed-window attention call (hashable; used as a static kernel argument)."""

    block_q: int
    block_kv: int
    q_len: int
    q_pad: int
    head_dim: int
    d_pad: int
    kv_len: int
    tok_len: int
    n_entries: int
    token_kv_pad: int
    entries_pad: int
    n_token_tiles: int
    n_entry_tiles: int
    n_tok_iter: int
    band: bool
    band_skip: bool
    causal: bool
    diag_offset: int
    window: int
    implicit: bool
    vmem_limit: int | None

    @property
    def n_qb(self) -> int:
        """Number of query blocks."""
        return self.q_pad // self.block_q


def make_tiling(
    *,
    q_len: int,
    kv_len: int,
    head_dim: int,
    block_q: int,
    block_kv: int,
    vmem_limit: int | None,
    window: int,
    token_kv_len: int,
    implicit: bool,
    causal: bool,
) -> Tiling:
    """Build the :class:`Tiling` for one call.

    Args:
        q_len: Query length.
        kv_len: KV axis length.
        head_dim: Head dim.
        block_q: Query tile (VMEM-fitted).
        block_kv: KV tile (VMEM-fitted, multiple of 128).
        vmem_limit: Scoped VMEM to request (``None`` = Mosaic default).
        window: Sliding-window size; ``0`` disables the band.
        token_kv_len: Token-region length (entries follow); only used with ``window``.
        implicit: The token-region mask is built in-kernel from positions and per-key validity.
        causal: The caller promised a causal dense bias (ignored with ``implicit``, which is causal by construction).

    Returns:
        The tiling.
    """
    band = window > 0 and 0 < token_kv_len <= kv_len
    tok_len = token_kv_len if band else kv_len
    n_entries = kv_len - tok_len
    token_kv_pad = round_up(tok_len, block_kv)
    entries_pad = round_up(n_entries, block_kv)
    n_token_tiles = token_kv_pad // block_kv
    n_entry_tiles = entries_pad // block_kv
    # Enough token tiles to cover any q-block's (block_q + window)-key span at any alignment.
    n_vis_tok = round_up(block_q + window, block_kv) // block_kv + 1
    band_skip = band and n_token_tiles > n_vis_tok
    return Tiling(
        block_q=block_q,
        block_kv=block_kv,
        q_len=q_len,
        q_pad=round_up(q_len, block_q),
        head_dim=head_dim,
        d_pad=round_up(head_dim, LANE),
        kv_len=kv_len,
        tok_len=tok_len,
        n_entries=n_entries,
        token_kv_pad=token_kv_pad,
        entries_pad=entries_pad,
        n_token_tiles=n_token_tiles,
        n_entry_tiles=n_entry_tiles,
        n_tok_iter=n_vis_tok if band_skip else n_token_tiles,
        band=band,
        band_skip=band_skip,
        causal=bool(causal) and not implicit,
        diag_offset=max(int(tok_len) - int(q_len), 0),
        window=int(window),
        implicit=bool(implicit),
        vmem_limit=vmem_limit,
    )


def without_skips(t: Tiling) -> Tiling:
    """The same layout visiting every token tile (no band or causal skip)."""
    return dataclasses.replace(t, band_skip=False, causal=False, n_tok_iter=t.n_token_tiles)


def tile_lo(qb, t: Tiling):
    """First token tile visited by q-block ``qb`` (``0`` without band skip)."""
    if not t.band_skip:
        return 0
    raw_lo = (qb * t.block_q - t.window + 1) // t.block_kv
    return jnp.clip(raw_lo, 0, t.n_token_tiles - t.n_tok_iter)


def tile_hi(qb, t: Tiling):
    """Last token tile q-block ``qb`` must visit under the causal skip (``n_token_tiles - 1`` without it).

    Uses ``lax.div`` on non-negative operands (truncation equals floor there) instead of ``//``.
    """
    if not t.causal:
        return t.n_token_tiles - 1
    raw_hi = jax.lax.div((qb + 1) * t.block_q - 1 + t.diag_offset, t.block_kv)
    return jnp.clip(raw_hi, 0, t.n_token_tiles - 1)


def n_visit(qb, t: Tiling):
    """Token tiles q-block ``qb`` visits: the band window, cut at the diagonal under the causal skip."""
    if not t.causal:
        return t.n_tok_iter
    return jnp.clip(tile_hi(qb, t) - tile_lo(qb, t) + 1, 1, t.n_tok_iter)


def qb_lo(tile, t: Tiling):
    """First q-block that can see token tile ``tile`` under the causal skip (``0`` without it)."""
    if not t.causal:
        return 0
    num = jnp.maximum(tile * t.block_kv - t.diag_offset, 0)
    return jnp.clip(jax.lax.div(num, t.block_q), 0, t.n_qb - 1)


def tile_of(qb, step, t: Tiling):
    """KV tile streamed at scan ``step`` of q-block ``qb``; token steps past the causal cut repeat the last tile."""
    tok = tile_lo(qb, t) + jnp.minimum(step, n_visit(qb, t) - 1)
    return jnp.where(step < t.n_tok_iter, tok, t.n_token_tiles + (step - t.n_tok_iter))


def token_mask_bias(tv_ref, qb, tile, t: Tiling):
    """In-kernel ``[block_q, block_kv]`` fp32 bias of token ``tile`` for q-block ``qb`` (window + causal + validity).

    Token key ``k`` is visible to query ``s`` iff ``k <= s``, ``k > s - window`` and ``token_valid[k]``.

    Args:
        tv_ref: Per-key validity ref ``[1, token_kv_pad]`` (fp32 1/0; padded keys are 0).
        qb: Query block index.
        tile: Token tile index (static or traced).
        t: Tiling.

    Returns:
        ``0`` on visible keys, ``MASK_MIN`` elsewhere.
    """
    bq, bkv = t.block_q, t.block_kv
    start = tile * bkv
    if not isinstance(start, int):
        start = pl.multiple_of(start, bkv)
    q_pos = qb * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bkv), 0)
    k_pos = tile * bkv + jax.lax.broadcasted_iota(jnp.int32, (bq, bkv), 1)
    key_ok = tv_ref[:, pl.ds(start, bkv)] > 0.5
    visible = (k_pos <= q_pos) & (k_pos > q_pos - t.window) & key_ok
    return jnp.where(visible, 0.0, MASK_MIN).astype(jnp.float32)


def compiler_params(t: Tiling, semantics: tuple[str, ...]) -> pltpu.CompilerParams:
    """Mosaic parameters: grid semantics plus the tiling's scoped-VMEM request when it has one."""
    if t.vmem_limit is None:
        return pltpu.CompilerParams(dimension_semantics=semantics)
    return pltpu.CompilerParams(dimension_semantics=semantics, vmem_limit_bytes=t.vmem_limit)


def pad_operands(
    query: Array,
    kv: Array,
    bias: Array,
    token_valid: Array | None,
    t: Tiling,
) -> tuple[Array, Array, Array, Array]:
    """Pad the operands to the tile grid.

    The head dim is zero-padded (inert in the dots), the query axis padded to ``q_pad``, the token region and the
    entries padded to tile boundaries separately; pad bias entries carry the bias dtype's minimum so padded keys never
    contribute. The bias keeps the caller's dtype (fp32 or bf16); kernels upcast every tile to fp32.

    Args:
        query: ``[B, H, q_len, D]``.
        kv: ``[B, kv_len, D]``.
        bias: Dense ``[B, q_len, kv_len]`` bias, or the entry columns ``[B, q_len, n_entries]`` when implicit.
        token_valid: ``[B, tok_len]`` fp32 validity when implicit, else ``None``.
        t: Tiling.

    Returns:
        ``(q_p [B, H, q_pad, d_pad], kv_p [B, kv_pad, d_pad], bias_p, tv_p [B, 1, tv_len])``; when implicit
        ``bias_p`` is ``[B, q_pad, entries_pad]`` (a ``[B, block_q, block_kv]`` placeholder without entries) and
        otherwise ``[B, q_pad, token_kv_pad + entries_pad]``; ``tv_p`` is a placeholder for dense biases.
    """
    batch = query.shape[0]
    q_extra, d_extra = t.q_pad - t.q_len, t.d_pad - t.head_dim
    q_p = jnp.pad(query, ((0, 0), (0, 0), (0, q_extra), (0, d_extra)))
    token_kv = jnp.pad(kv[:, : t.tok_len], ((0, 0), (0, t.token_kv_pad - t.tok_len), (0, d_extra)))
    if t.n_entry_tiles:
        entry_kv = jnp.pad(kv[:, t.tok_len :], ((0, 0), (0, t.entries_pad - t.n_entries), (0, d_extra)))
        kv_p = jnp.concatenate([token_kv, entry_kv], axis=1)
    else:
        kv_p = token_kv
    pad_min = MASK_MIN if bias.dtype == jnp.float32 else float(jnp.finfo(bias.dtype).min)
    if t.implicit:
        tv_p = jnp.pad(token_valid.astype(jnp.float32), ((0, 0), (0, t.token_kv_pad - t.tok_len)))[:, None, :]
        if t.n_entry_tiles:
            widths = ((0, 0), (0, q_extra), (0, t.entries_pad - t.n_entries))
            bias_p = jnp.pad(bias, widths, constant_values=pad_min)
        else:
            bias_p = jnp.zeros((batch, t.block_q, t.block_kv), jnp.float32)
        return q_p, kv_p, bias_p, tv_p
    tv_p = jnp.zeros((batch, 1, LANE), jnp.float32)
    token_bias = jnp.pad(
        bias[:, :, : t.tok_len], ((0, 0), (0, q_extra), (0, t.token_kv_pad - t.tok_len)), constant_values=pad_min
    )
    if t.n_entry_tiles:
        entry_bias = jnp.pad(
            bias[:, :, t.tok_len :], ((0, 0), (0, q_extra), (0, t.entries_pad - t.n_entries)), constant_values=pad_min
        )
        bias_p = jnp.concatenate([token_bias, entry_bias], axis=2)
    else:
        bias_p = token_bias
    return q_p, kv_p, bias_p, tv_p
