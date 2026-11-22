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


"""Fused Lightning-indexer scoring (Pallas TPU, differentiable).

The indexer scores every query against every candidate key:

    scores[b, s, t] = sum_h  w[b, h, s] * relu( q[b, h, s, :] . k[b, t, :] )

Written as XLA ops this materialises the ``[B, H, S, T]`` fp32 logits (8.6 GB for ``H = 32``, ``S = T = 16384``),
writes them, reads them back for the relu / weight / head-sum and does it all again for the backward. The kernel
here walks ``(q tile, entry tile)`` pairs and keeps the per-head logits in VMEM: the head-summed ``[bq, tk]`` fp32
scores are the only thing that leaves the chip.

Backward (``custom_vjp``), given ``g = d scores [B, S, T]`` and with ``a = q . k`` recomputed per tile,
``r = 1[a > 0]``:

    d q[h, s, :]  = sum_t  g[s, t] w[h, s] r[h, s, t]  k[t, :]           (dq kernel, entry tiles inner)
    d w[h, s]     = sum_t  g[s, t] relu(a[h, s, t])                      (same kernel)
    d k[t, :]     = sum_s sum_h g[s, t] w[h, s] r[h, s, t]  q[h, s, :]    (dk kernel, q tiles inner)

Matmuls contract in the operands' dtype with fp32 accumulation by default (one MXU pass, what a default-precision
fp32 einsum also does on TPU); ``precision="high"`` / ``"highest"`` casts the tiles to fp32 in-kernel and contracts
at that precision. Masking of the scores (causality, reach) stays with the caller. Off TPU the kernels run in Pallas
interpret mode.
"""

from __future__ import annotations

import functools
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ejkernel.callib import ejit
from ejkernel.ops import FwdParams

_LANES = 128
_SUBLANES = 16  # bf16 packs two rows per sublane, so the query tile is a multiple of 16
_DEFAULT_VMEM = 16 * 2**20  # Mosaic's default scoped VMEM limit (and a v4 core's whole VMEM)
_DEFAULT_BLOCK_Q = 128
_DEFAULT_BLOCK_T = 1024  # measured best entry tile (see IndexerScoresConfig)

_PRECISION_ALIASES = {
    "default": "default",
    "fastest": "default",
    "bfloat16": "default",
    "high": "high",
    "tensorfloat32": "high",
    "highest": "highest",
    "float32": "highest",
}


def _round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def precision_key(precision: tp.Any) -> str:
    """Normalise ``precision`` (``None`` / str / ``jax.lax.Precision`` / an equal pair) to ``default|high|highest``.

    Args:
        precision: Caller precision.

    Returns:
        The normalised static precision name.

    Raises:
        ValueError: For an unknown spelling or a pair of different precisions.
    """
    if precision is None:
        return "default"
    if isinstance(precision, (tuple, list)):
        keys = {precision_key(p) for p in precision}
        if len(keys) != 1:
            raise ValueError(f"indexer_scores: per-operand precision {precision!r} is not supported")
        return keys.pop()
    if isinstance(precision, jax.lax.Precision):
        return _PRECISION_ALIASES[precision.name.lower()]
    if isinstance(precision, str) and precision.lower() in _PRECISION_ALIASES:
        return _PRECISION_ALIASES[precision.lower()]
    raise ValueError(f"precision must be None, 'default', 'high', 'highest' or a jax.lax.Precision, got {precision!r}")


def _mm_dtype(operand_dtype, precision: str):
    """``high`` / ``highest`` contract fp32 tiles; ``default`` contracts in the operand dtype."""
    return jnp.float32 if precision in ("high", "highest") else operand_dtype


def _lax_precision(precision: str) -> jax.lax.Precision:
    return {
        "default": jax.lax.Precision.DEFAULT,
        "high": jax.lax.Precision.HIGH,
        "highest": jax.lax.Precision.HIGHEST,
    }[precision]


def _dot(lhs: Array, rhs: Array, dims, precision: str) -> Array:
    """``dot_general`` with fp32 accumulation; operands cast to fp32 first for ``high`` / ``highest``."""
    dt = _mm_dtype(lhs.dtype, precision)
    return jax.lax.dot_general(
        lhs.astype(dt), rhs.astype(dt), dims, preferred_element_type=jnp.float32, precision=_lax_precision(precision)
    )


def _qk(q_h: Array, k_t: Array, precision: str) -> Array:
    """``[bq, d] x [tk, d] -> [bq, tk]`` fp32 logits for one head."""
    return _dot(q_h, k_t, (((1,), (1,)), ((), ())), precision)


class _Layout(tp.NamedTuple):
    block_q: int
    block_t: int
    q_pad: int
    t_pad: int
    q_p: Array  # [B, H, q_pad, d]
    k_p: Array  # [B, t_pad, d]
    w_p: Array  # [B, H, q_pad, 1] fp32
    vmem_limit: int | None  # bytes to request from Mosaic; None keeps the default scoped limit


def vmem_capacity() -> int:
    """Physical VMEM per core of the local TPU (from ``pltpu.get_tpu_info()``; 16 MiB off TPU or if unknown)."""
    try:
        cap = int(getattr(pltpu.get_tpu_info(), "vmem_capacity_bytes", 0) or 0)
    except Exception:
        return _DEFAULT_VMEM
    return cap if cap > 0 else _DEFAULT_VMEM


def program_bytes(block_q: int, block_t: int, num_heads: int, head_dim: int, itemsize: int) -> int:
    """VMEM working set of the heaviest program (the dQ / dW step), double-buffered inputs and outputs included.

    The ``[H, bq, 1]`` weight and ``dW`` tiles are lane-padded to 128 in VMEM, which dominates at many heads; the
    per-head ``[bq, tk]`` fp32 intermediates (logits, gate, product) are counted once since heads are looped.
    """
    q_tile = num_heads * block_q * head_dim * itemsize
    k_tile = block_t * head_dim * itemsize
    w_tile = num_heads * block_q * _LANES * 4
    g_tile = block_q * block_t * 4
    dq_tile = num_heads * block_q * head_dim * 4
    dw_tile = num_heads * block_q * _LANES * 4
    intermediates = 4 * block_q * block_t * 4
    return 2 * (q_tile + k_tile + w_tile + g_tile) + 2 * dq_tile + dq_tile + 3 * dw_tile + intermediates


def vmem_safe_blocks(
    block_q: int, block_t: int, num_heads: int, head_dim: int, itemsize: int, capacity: int | None = None
) -> tuple[int, int, int | None]:
    """``(block_q, block_t, vmem_limit)``: tiles that fit the local VMEM with headroom, and the limit to request.

    Requested tiles are rounded UP to the sublane / lane multiples Mosaic needs (16 for the query tile, 128 for the
    entry tile), then shrunk -- the query tile first, then the entry tile -- until the working set is under half the
    physical VMEM. On a 16 MiB part the default scoped limit is kept; larger parts get a limit sized to the working set.

    Args:
        block_q: Requested query tile.
        block_t: Requested entry tile.
        num_heads: Index heads.
        head_dim: Index head width.
        itemsize: Bytes per operand element.
        capacity: Physical VMEM in bytes; ``None`` queries the local TPU.

    Returns:
        The fitted ``(block_q, block_t)`` and the Mosaic ``vmem_limit_bytes`` (``None`` = default).
    """
    block_q = max(_SUBLANES, _round_up(int(block_q), _SUBLANES))
    block_t = max(_LANES, _round_up(int(block_t), _LANES))
    capacity = vmem_capacity() if capacity is None else int(capacity)
    budget = capacity // 2
    while program_bytes(block_q, block_t, num_heads, head_dim, itemsize) > budget and block_q > _SUBLANES:
        block_q = max(_SUBLANES, _round_up(block_q // 2, _SUBLANES))
    while program_bytes(block_q, block_t, num_heads, head_dim, itemsize) > budget and block_t > _LANES:
        block_t = max(_LANES, _round_up(block_t // 2, _LANES))
    need = program_bytes(block_q, block_t, num_heads, head_dim, itemsize)
    vmem_limit = None if capacity <= _DEFAULT_VMEM else int(min(capacity - 2**20, max(_DEFAULT_VMEM, 2 * need)))
    return block_q, block_t, vmem_limit


def _padded_layout(query: Array, index_k: Array, weights: Array, block_q: int, block_t: int) -> _Layout:
    """Fit the tiles, then pad the query axis to ``block_q`` and the entry axis to ``block_t``.

    Padded query rows carry zero weights and padded entry rows zero keys, so they score 0 and are sliced off.
    """
    _batch, num_heads, q_len, head_dim = query.shape
    n_entries = index_k.shape[1]
    block_q, block_t, vmem_limit = vmem_safe_blocks(block_q, block_t, num_heads, head_dim, query.dtype.itemsize)
    block_q = min(block_q, _round_up(q_len, _SUBLANES))
    block_t = min(block_t, _round_up(n_entries, _LANES))
    q_pad, t_pad = _round_up(q_len, block_q), _round_up(n_entries, block_t)
    q_p = query if q_pad == q_len else jnp.pad(query, ((0, 0), (0, 0), (0, q_pad - q_len), (0, 0)))
    k_p = index_k if t_pad == n_entries else jnp.pad(index_k, ((0, 0), (0, t_pad - n_entries), (0, 0)))
    w_p = weights.astype(jnp.float32)
    if q_pad != q_len:
        w_p = jnp.pad(w_p, ((0, 0), (0, 0), (0, q_pad - q_len), (0, 0)))
    return _Layout(block_q, block_t, q_pad, t_pad, q_p, k_p.astype(query.dtype), w_p, vmem_limit)


def _compiler_params(lay: _Layout, semantics: tuple[str, ...]) -> pltpu.CompilerParams:
    if lay.vmem_limit is None:
        return pltpu.CompilerParams(dimension_semantics=semantics)
    return pltpu.CompilerParams(dimension_semantics=semantics, vmem_limit_bytes=lay.vmem_limit)


def _fwd_kernel(q_ref, k_ref, w_ref, out_ref, *, num_heads: int, precision: str):
    """One ``(batch, q tile, entry tile)``: head-summed weighted relu logits; the logits never leave VMEM."""
    k_t = k_ref[...]
    acc = jnp.zeros(out_ref.shape, jnp.float32)
    for h in range(num_heads):
        acc = acc + jnp.maximum(_qk(q_ref[h], k_t, precision), 0.0) * w_ref[h]  # w[h]: [bq, 1] along lanes
    out_ref[...] = acc.astype(out_ref.dtype)


def _forward(query: Array, index_k: Array, weights: Array, block_q: int, block_t: int, precision: str) -> Array:
    batch, num_heads, q_len, head_dim = query.shape
    n_entries = index_k.shape[1]
    lay = _padded_layout(query, index_k, weights, block_q, block_t)
    block_q, block_t = lay.block_q, lay.block_t
    grid = (batch, lay.q_pad // block_q, lay.t_pad // block_t)
    scores = pl.pallas_call(
        functools.partial(_fwd_kernel, num_heads=num_heads, precision=precision),
        grid=grid,
        in_specs=[
            pl.BlockSpec((None, num_heads, block_q, head_dim), lambda b, qb, tb: (b, 0, qb, 0)),
            pl.BlockSpec((None, block_t, head_dim), lambda b, qb, tb: (b, tb, 0)),
            pl.BlockSpec((None, num_heads, block_q, 1), lambda b, qb, tb: (b, 0, qb, 0)),
        ],
        out_specs=pl.BlockSpec((None, block_q, block_t), lambda b, qb, tb: (b, qb, tb)),
        out_shape=jax.ShapeDtypeStruct((batch, lay.q_pad, lay.t_pad), jnp.float32),
        compiler_params=_compiler_params(lay, ("parallel", "parallel", "parallel")),
        interpret=jax.default_backend() != "tpu",
    )(lay.q_p, lay.k_p, lay.w_p)
    return scores[:, :q_len, :n_entries]


def _bwd_dq_dw_kernel(
    q_ref, k_ref, w_ref, g_ref, dq_ref, dw_ref, dq_acc, dw_acc, *, num_heads: int, n_tiles: int, precision: str
):
    """dQ and dW for one ``(batch, q tile)``, accumulated over the entry tiles (innermost grid axis)."""
    step = pl.program_id(2)

    @pl.when(step == 0)
    def _init():
        dq_acc[...] = jnp.zeros(dq_acc.shape, dq_acc.dtype)
        dw_acc[...] = jnp.zeros(dw_acc.shape, dw_acc.dtype)

    k_t = k_ref[...]
    g = g_ref[...].astype(jnp.float32)
    for h in range(num_heads):
        a = _qk(q_ref[h], k_t, precision)
        grad_logit = jnp.where(a > 0.0, g * w_ref[h], 0.0)  # d a[h] = g * w[h] * relu'(a)
        dq_acc[h] += _dot(grad_logit.astype(_mm_dtype(k_t.dtype, precision)), k_t, (((1,), (0,)), ((), ())), precision)
        dw_acc[h] += jnp.sum(g * jnp.maximum(a, 0.0), axis=-1, keepdims=True)

    @pl.when(step == n_tiles - 1)
    def _finish():
        dq_ref[...] = dq_acc[...].astype(dq_ref.dtype)
        dw_ref[...] = dw_acc[...].astype(dw_ref.dtype)


def _bwd_dk_kernel(q_ref, k_ref, w_ref, g_ref, dk_ref, dk_acc, *, num_heads: int, n_tiles: int, precision: str):
    """dK for one ``(batch, entry tile)``, accumulated over the q tiles (innermost grid axis)."""
    step = pl.program_id(2)

    @pl.when(step == 0)
    def _init():
        dk_acc[...] = jnp.zeros(dk_acc.shape, dk_acc.dtype)

    k_t = k_ref[...]
    g = g_ref[...].astype(jnp.float32)
    for h in range(num_heads):
        q_h = q_ref[h]
        a = _qk(q_h, k_t, precision)
        grad_logit = jnp.where(a > 0.0, g * w_ref[h], 0.0)  # [bq, tk]
        # TN dot: contract over the q rows.
        dk_acc[...] += _dot(grad_logit.astype(_mm_dtype(q_h.dtype, precision)), q_h, (((0,), (0,)), ((), ())), precision)

    @pl.when(step == n_tiles - 1)
    def _finish():
        dk_ref[...] = dk_acc[...].astype(dk_ref.dtype)


def _backward(query: Array, index_k: Array, weights: Array, grad: Array, block_q: int, block_t: int, precision: str):
    batch, num_heads, q_len, head_dim = query.shape
    n_entries = index_k.shape[1]
    lay = _padded_layout(query, index_k, weights, block_q, block_t)
    block_q, block_t = lay.block_q, lay.block_t
    n_qb, n_tb = lay.q_pad // block_q, lay.t_pad // block_t
    interpret = jax.default_backend() != "tpu"
    g_p = grad.astype(jnp.float32)
    if (lay.q_pad, lay.t_pad) != (q_len, n_entries):  # padded rows / columns carry zero cotangent
        g_p = jnp.pad(g_p, ((0, 0), (0, lay.q_pad - q_len), (0, lay.t_pad - n_entries)))
    cparams = _compiler_params(lay, ("parallel", "parallel", "arbitrary"))

    dq_p, dw_p = pl.pallas_call(
        functools.partial(_bwd_dq_dw_kernel, num_heads=num_heads, n_tiles=n_tb, precision=precision),
        grid=(batch, n_qb, n_tb),
        in_specs=[
            pl.BlockSpec((None, num_heads, block_q, head_dim), lambda b, qb, tb: (b, 0, qb, 0)),
            pl.BlockSpec((None, block_t, head_dim), lambda b, qb, tb: (b, tb, 0)),
            pl.BlockSpec((None, num_heads, block_q, 1), lambda b, qb, tb: (b, 0, qb, 0)),
            pl.BlockSpec((None, block_q, block_t), lambda b, qb, tb: (b, qb, tb)),
        ],
        out_specs=[
            pl.BlockSpec((None, num_heads, block_q, head_dim), lambda b, qb, tb: (b, 0, qb, 0)),
            pl.BlockSpec((None, num_heads, block_q, 1), lambda b, qb, tb: (b, 0, qb, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((batch, num_heads, lay.q_pad, head_dim), jnp.float32),
            jax.ShapeDtypeStruct((batch, num_heads, lay.q_pad, 1), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((num_heads, block_q, head_dim), jnp.float32),
            pltpu.VMEM((num_heads, block_q, 1), jnp.float32),
        ],
        compiler_params=cparams,
        interpret=interpret,
    )(lay.q_p, lay.k_p, lay.w_p, g_p)

    dk_p = pl.pallas_call(
        functools.partial(_bwd_dk_kernel, num_heads=num_heads, n_tiles=n_qb, precision=precision),
        grid=(batch, n_tb, n_qb),
        in_specs=[
            pl.BlockSpec((None, num_heads, block_q, head_dim), lambda b, tb, qb: (b, 0, qb, 0)),
            pl.BlockSpec((None, block_t, head_dim), lambda b, tb, qb: (b, tb, 0)),
            pl.BlockSpec((None, num_heads, block_q, 1), lambda b, tb, qb: (b, 0, qb, 0)),
            pl.BlockSpec((None, block_q, block_t), lambda b, tb, qb: (b, qb, tb)),
        ],
        out_specs=pl.BlockSpec((None, block_t, head_dim), lambda b, tb, qb: (b, tb, 0)),
        out_shape=jax.ShapeDtypeStruct((batch, lay.t_pad, head_dim), jnp.float32),
        scratch_shapes=[pltpu.VMEM((block_t, head_dim), jnp.float32)],
        compiler_params=cparams,
        interpret=interpret,
    )(lay.q_p, lay.k_p, lay.w_p, g_p)

    dq = dq_p[:, :, :q_len, :].astype(query.dtype)
    dw = dw_p[:, :, :q_len, :].astype(weights.dtype)
    dk = dk_p[:, :n_entries, :].astype(index_k.dtype)
    return dq, dk, dw


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _indexer_scores_pallas(query, index_k, weights, block_q, block_t, precision):
    return _forward(query, index_k, weights, block_q, block_t, precision)


def _indexer_scores_fwd(query, index_k, weights, block_q, block_t, precision):
    return _forward(query, index_k, weights, block_q, block_t, precision), (query, index_k, weights)


def _indexer_scores_bwd(block_q, block_t, precision, residuals, grad):
    query, index_k, weights = residuals
    return _backward(query, index_k, weights, grad, block_q, block_t, precision)


_indexer_scores_pallas.defvjp(_indexer_scores_fwd, _indexer_scores_bwd)


def requested_blocks(fwd_params: FwdParams | None) -> tuple[int, int]:
    """Requested ``(block_q, block_t)`` (``q_blocksize`` / ``kv_blocksize``) before :func:`vmem_safe_blocks`."""
    block_q = getattr(fwd_params, "q_blocksize", None) if fwd_params is not None else None
    block_t = getattr(fwd_params, "kv_blocksize", None) if fwd_params is not None else None
    return int(block_q or _DEFAULT_BLOCK_Q), int(block_t or _DEFAULT_BLOCK_T)


def indexer_scores_tpu(
    query: Float[Array, "batch num_heads q_len head_dim"],
    index_k: Float[Array, "batch num_entries head_dim"],
    weights: Float[Array, "batch num_heads q_len 1"],
    fwd_params: FwdParams | None = None,
    precision: tp.Any = None,
) -> Float[Array, "batch q_len num_entries"]:
    """Head-summed weighted-relu indexer scores ``[B, S, T]`` (fp32) without the ``[B, H, S, T]`` intermediate.

    Args:
        query: Index queries ``[B, H, S, d]``.
        index_k: Index keys ``[B, T, d]``; cast to the query dtype for the contraction, so with a bf16 query and an
            fp32 key the key cotangent is bf16-valued (returned in the key's dtype), as in the XLA reference.
        weights: Per-head, per-query weights ``[B, H, S, 1]`` (applied in fp32).
        fwd_params: Optional tiling (``q_blocksize`` = query tile, ``kv_blocksize`` = entry tile); rounded up to the
            sublane / lane multiples and shrunk to fit the local VMEM.
        precision: ``None`` / ``"default"`` contracts in the operand dtype (one MXU pass for bf16); ``"high"`` /
            ``"highest"`` (or the ``jax.lax.Precision``) casts the tiles to fp32 in-kernel.

    Returns:
        fp32 ``[B, S, T]`` scores, differentiable in ``query``, ``index_k`` and ``weights``.
    """
    block_q, block_t = requested_blocks(fwd_params)
    return _indexer_scores_tpu(
        query, index_k, weights, block_q=block_q, block_t=block_t, precision=precision_key(precision)
    )


@ejit(static_argnames=["block_q", "block_t", "precision"])
def _indexer_scores_tpu(query, index_k, weights, block_q: int, block_t: int, precision: str):
    return _indexer_scores_pallas(query, index_k.astype(query.dtype), weights, block_q, block_t, precision)


__all__ = ("indexer_scores_tpu", "precision_key", "program_bytes", "vmem_capacity", "vmem_safe_blocks")
