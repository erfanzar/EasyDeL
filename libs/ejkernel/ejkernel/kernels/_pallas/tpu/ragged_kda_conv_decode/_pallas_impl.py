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

"""Pallas TPU kernel: the whole KDA decode step of one token per row in one ``pallas_call``, pools updated in place.

Token-major: grid ``(cores, steps)``; every grid step owns ``block_tokens`` consecutive tokens of the packed stream, so
the raw q / k / v tokens, beta, the log-decay and the output are plain pipelined blocks (no gather, no scatter). Token
``t`` advances the row whose single token it is: ``token_code[t] = 2 * slot + has_initial_state`` (``-1``: the token
outputs zeros and touches nothing). Per advancing token:

* its three conv windows and its recurrent state are DMAed from pool slot ``slot`` (rows that start a request read
  nothing and start from zeros); all reads of a grid step are issued up front and waited per token;
* depthwise causal conv: window shifted by the raw token, ``d_conv`` taps mixed in fp32 in the reference order
  (``t0 * w0``, then ``+= t_j * w_j``), SiLU, rounded to the compute and conv dtypes;
* L2 norm of q and k per head, q scaled by ``K**-0.5``, ``exp`` of the log-decay, beta broadcast over ``V``;
* the decode step of ``ragged_kda_decode`` (same expressions, same ``HIGHEST`` MXU matvecs on the same head blocks)::

      S = S * exp(g);   delta = beta * (v - k^T S);   S = S + k delta^T;   o = q^T S

* the new windows and state are DMAed back into slot ``slot`` (the pools are input/output aliases).

Every op runs inside the kernel: Mosaic's ``exp``, ``rsqrt``, ``logistic`` / SiLU, division and the fp32 conv mix were
measured bit-identical to XLA's on TPU v5p. What is not fixed by the source is where XLA rounds the bf16 L2-norm
intermediates (XLA keeps bf16 arithmetic chains in fp32 inside a fusion and rounds at fusion boundaries); ``rounding``
names, for each such point, whether the value is rounded to the conv dtype. :data:`EXACT_ROUNDING` is the recipe that
reproduces the serving path (XLA conv glue + the ``ragged_kda_decode`` kernel) bit for bit in a standalone, unsharded
program. The reference's rounding points are a property of the graph XLA fuses, not of the source: in a 16-chip
serve (heads split under ``shard_map``) the same kernel was close (bf16 quality gate passed) but not bitwise.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ..ragged_kda_decode._pallas_impl import pick_block_h

L2_EPS = 1e-6
"""Epsilon of the L2 norm (``kda_chunk._xla_impl.l2norm``), added in the conv dtype as the reference does."""

ROUNDING_POINTS: tuple[str, ...] = (
    "q_sq",
    "q_prod",
    "q_sum",
    "q_den",
    "q_rsqrt",
    "q_mul",
    "q_out",
    "k_sq",
    "k_prod",
    "k_sum",
    "k_den",
    "k_rsqrt",
    "k_mul",
    "k_out",
    "v",
)
"""Rounding points of the L2 norm and the value. ``*_sq`` / ``*_mul`` / ``v``: the squares / the final product / the
value read the conv output rounded to the conv dtype (else the unrounded fp32 SiLU); ``*_prod``, ``*_sum``, ``*_den``,
``*_rsqrt``, ``*_out``: the squares, their sum, ``sum + eps``, its ``rsqrt`` and the normalised vector are rounded to
the conv dtype."""

EXACT_ROUNDING: tuple[bool, ...] = (
    False, False, True, False, True, True, False,
    False, False, True, False, True, True, False,
    False,
)  # fmt: skip
"""Recipe of the serving path, found on TPU v5p (jax 0.11.2) against XLA conv glue + ``ragged_kda_decode``: XLA drops the
bf16 round trips the squares, the denominator, the normalised vector and the value see (``xla_allow_excess_precision``)
and keeps those of the sum of squares, the ``rsqrt`` and the conv output the final product reads."""

_F32 = jnp.dtype(jnp.float32)
_SUPPORTED_FLOATS = (jnp.dtype(jnp.bfloat16), _F32)


def _round_up(x: int, m: int) -> int:
    return -(-int(x) // m) * m


def _tile_bytes(rows: int, cols: int, dtype: tp.Any) -> int:
    """VMEM bytes of a ``[rows, cols]`` tile, padded to native ``(8 * packing, 128)`` tiles."""
    itemsize = jnp.dtype(dtype).itemsize
    return _round_up(rows, 8 * max(1, 4 // itemsize)) * _round_up(cols, 128) * itemsize


def vmem_capacity_bytes() -> int:
    """VMEM capacity of one TensorCore (64 MiB off TPU)."""
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # pragma: no cover - only reached off TPU
        return 64 * 1024 * 1024


def tensor_cores() -> int:
    """TensorCores per chip that a ``"parallel"`` grid axis is split over (1 off TPU)."""
    try:
        return max(1, int(pltpu.get_tpu_info().num_cores))
    except Exception:  # pragma: no cover - only reached off TPU
        return 1


def plan_blocks(
    num_tokens: int,
    per_token_bytes: int,
    fixed_bytes: int,
    *,
    budget_bytes: int,
    num_cores: int,
    requested: int = 0,
) -> tuple[int, int] | None:
    """``(cores, block_tokens)`` for a token count, from the VMEM budget.

    Tokens split evenly over the cores (else one core); a block is the largest divisor of a core's tokens that fits
    ``fixed_bytes + block * per_token_bytes <= budget_bytes`` and is a multiple of 16 (the bf16 sublane tile) or the
    whole stream.

    Args:
        num_tokens: Tokens of the step's packed stream.
        per_token_bytes: VMEM bytes one token of a block needs.
        fixed_bytes: VMEM bytes independent of the block size.
        budget_bytes: VMEM bytes available to the kernel.
        num_cores: TensorCores per chip.
        requested: Block size to use (``0``: automatic); must satisfy the same rules.

    Returns:
        ``(cores, block_tokens)``, or ``None`` when no block fits.
    """
    for cores in sorted({max(1, int(num_cores)), 1}, reverse=True):
        if num_tokens % cores:
            continue
        per_core = num_tokens // cores
        sizes = [int(requested)] if requested else range(per_core, 0, -1)
        for bt in sizes:
            if bt <= 0 or per_core % bt or (bt % 16 and bt != num_tokens):
                continue
            if fixed_bytes + bt * per_token_bytes <= budget_bytes:
                return cores, bt
    return None


def block_plan(
    query: tp.Any,
    key: tp.Any,
    value: tp.Any,
    beta: tp.Any,
    decay: tp.Any,
    conv_weights: tp.Sequence[tp.Any],
    conv_pools: tp.Sequence[tp.Any],
    recurrent_state: tp.Any,
    *,
    num_cores: int = 0,
    block_tokens: int = 0,
) -> tuple[int, int, int] | None:
    """``(cores, block_tokens, vmem_limit_bytes)`` of a call, from the VMEM capacity (``None``: nothing fits).

    Arguments are arrays or ``jax.ShapeDtypeStruct`` with the shapes of the kernel call (per device under
    ``shard_map``). The VMEM need is the per-token buffers (windows, state, token rows) times the block plus the
    block-independent buffers; the budget is 60% of :func:`vmem_capacity_bytes`.

    Args:
        query: ``[T, H*K]``.
        key: ``[T, H*K]``.
        value: ``[T, H*V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        conv_weights: ``(q, k, v)`` ``[dim, d_conv]``.
        conv_pools: ``(q, k, v)`` ``[S, dim, d_conv]``.
        recurrent_state: ``[S, H, K, V]``.
        num_cores: TensorCores to split the tokens over (``0``: the chip's).
        block_tokens: Requested tokens per grid step (``0``: automatic).

    Returns:
        The plan, or ``None``.
    """
    f32 = jnp.float32
    num_tokens, key_dim = (int(d) for d in query.shape)
    value_dim = int(value.shape[-1])
    num_heads = int(beta.shape[-1])
    head_dim, v_dim = key_dim // num_heads, value_dim // num_heads
    d_conv = int(conv_weights[0].shape[-1])
    dims = (key_dim, key_dim, value_dim)
    stream_dtypes = (query.dtype, key.dtype, value.dtype)
    per_token = (
        num_heads * _tile_bytes(head_dim, v_dim, recurrent_state.dtype)
        + sum(_tile_bytes(d_conv, d, p.dtype) for d, p in zip(dims, conv_pools, strict=True))
        + 4 * (2 * key_dim + value_dim + 128)
        + 2 * sum(d * jnp.dtype(s).itemsize for d, s in zip(dims, stream_dtypes, strict=True))
        + 2 * _tile_bytes(num_heads, head_dim, decay.dtype)
        + 2 * 128 * jnp.dtype(beta.dtype).itemsize
        + 2 * _tile_bytes(num_heads, v_dim, f32)
    )
    bh = pick_block_h(num_heads)
    fixed = (
        2 * sum(_tile_bytes(d_conv, d, w.dtype) for d, w in zip(dims, conv_weights, strict=True))
        + 3 * _tile_bytes(d_conv, max(key_dim, value_dim), f32)
        + 6 * _tile_bytes(num_heads, max(head_dim, v_dim), f32)
        + 8 * bh * head_dim * v_dim * 4
    )
    capacity = vmem_capacity_bytes()
    plan = plan_blocks(
        num_tokens,
        per_token,
        fixed,
        budget_bytes=int(capacity * 0.6),
        num_cores=int(num_cores) or tensor_cores(),
        requested=int(block_tokens),
    )
    return None if plan is None else (*plan, int(capacity * 0.75))


def unsupported_reason(
    query: tp.Any,
    key: tp.Any,
    value: tp.Any,
    beta: tp.Any,
    decay: tp.Any,
    conv_weights: tp.Sequence[tp.Any],
    conv_pools: tp.Sequence[tp.Any],
    recurrent_state: tp.Any,
    conv_dtype: tp.Any,
    *,
    block_tokens: int = 0,
) -> str | None:
    """Why the fused kernel cannot run this call, or ``None`` when it can (arrays or ``ShapeDtypeStruct``).

    Args:
        query: ``[T, H*K]``.
        key: ``[T, H*K]``.
        value: ``[T, H*V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        conv_weights: ``(q, k, v)`` ``[dim, d_conv]``.
        conv_pools: ``(q, k, v)`` ``[S, dim, d_conv]``.
        recurrent_state: ``[S, H, K, V]``.
        conv_dtype: Dtype of the conv outputs.
        block_tokens: Requested tokens per grid step (``0``: automatic).

    Returns:
        A short reason, or ``None``.
    """
    num_heads = int(beta.shape[-1])
    head_dim, v_dim = int(query.shape[-1]) // num_heads, int(value.shape[-1]) // num_heads
    if head_dim % 128 or v_dim % 128:
        return f"head sizes {head_dim} / {v_dim} are not multiples of the 128 TPU lanes"
    dtypes = (
        query.dtype,
        key.dtype,
        value.dtype,
        beta.dtype,
        decay.dtype,
        *(w.dtype for w in conv_weights),
        *(p.dtype for p in conv_pools),
        recurrent_state.dtype,
        conv_dtype,
    )
    for d in dtypes:
        if jnp.dtype(d) not in _SUPPORTED_FLOATS:
            return f"dtype {jnp.dtype(d).name} is not bfloat16 / float32"
    if (
        block_plan(query, key, value, beta, decay, conv_weights, conv_pools, recurrent_state, block_tokens=block_tokens)
        is None
    ):
        return f"no block of the {int(query.shape[0])} tokens fits VMEM (block_tokens={block_tokens})"
    return None


def _rnd(x: Array, dtype: tp.Any) -> Array:
    """``x`` (fp32) rounded through ``dtype``."""
    return x if jnp.dtype(dtype) == _F32 else x.astype(dtype).astype(jnp.float32)


def _kernel(
    code_ref,
    flags_ref,
    xq_ref,
    xk_ref,
    xv_ref,
    beta_ref,
    decay_ref,
    wq_ref,
    wk_ref,
    wv_ref,
    cq_hbm,
    ck_hbm,
    cv_hbm,
    rec_hbm,
    o_ref,
    cq_out,
    ck_out,
    cv_out,
    rec_out,
    sxq,
    sxk,
    sxv,
    sbeta,
    wbq,
    wbk,
    wbv,
    stb,
    hq,
    hk,
    hv,
    qr,
    qu,
    kr,
    ku,
    vr,
    vu,
    sems,
    *,
    block_tokens: int,
    steps: int,
    num_heads: int,
    head_dim: int,
    v_dim: int,
    d_conv: int,
    block_h: int,
    computes: tuple,
    conv_dtype,
    pool_dtypes: tuple,
    rec_dtype,
    eps: float,
    rounding: tuple[bool, ...] | None,
    use_qk_l2norm: bool,
):
    """One grid step: ``block_tokens`` tokens, their pool slots DMAed in and out by hand (see the module docstring)."""
    f32 = jnp.float32
    base = (pl.program_id(0) * steps + pl.program_id(1)) * block_tokens
    scale = 1.0 / (head_dim**0.5)
    pools_in = (cq_hbm, ck_hbm, cv_hbm, rec_hbm)
    pools_out = (cq_out, ck_out, cv_out, rec_out)
    bufs = (wbq, wbk, wbv, stb)

    # Token rows are read at a dynamic sublane: keep fp32 copies (bf16 rows are packed two per sublane).
    sxq[...] = xq_ref[...].astype(f32)
    sxk[...] = xk_ref[...].astype(f32)
    sxv[...] = xv_ref[...].astype(f32)
    sbeta[...] = beta_ref[...].astype(f32)

    def decode(code):
        live = code >= 0
        keep = jnp.logical_and(live, jnp.bitwise_and(code, 1) == 1)
        return live, keep, jnp.right_shift(jnp.maximum(code, 0), 1)

    def reads(j, slot):
        return [pltpu.make_async_copy(pools_in[a].at[slot], bufs[a].at[j], sems.at[a, j]) for a in range(4)]

    def writes(j, slot):
        return [pltpu.make_async_copy(bufs[a].at[j], pools_out[a].at[slot], sems.at[a, j]) for a in range(4)]

    def pick(name: str, rounded: Array, raw: Array) -> Array:
        i = ROUNDING_POINTS.index(name)
        if rounding is not None:
            return rounded if rounding[i] else raw
        return jnp.where(jnp.full(rounded.shape, flags_ref[i], jnp.int32) != 0, rounded, raw)

    def maybe_rnd(name: str, x: Array) -> Array:
        return pick(name, _rnd(x, conv_dtype), x)

    def l2(prefix: str, xr: Array, xu: Array) -> Array:
        sq = pick(f"{prefix}_sq", xr, xu)
        prod = maybe_rnd(f"{prefix}_prod", sq * sq)
        total = maybe_rnd(f"{prefix}_sum", jnp.sum(prod, axis=-1, keepdims=True))
        den = maybe_rnd(f"{prefix}_den", total + eps)
        r = maybe_rnd(f"{prefix}_rsqrt", lax.rsqrt(den))
        return maybe_rnd(f"{prefix}_out", pick(f"{prefix}_mul", xr, xu) * r)

    def contract(vec, s):
        # sum_k vec[h, k] * s[h, k, v] -> [h, v]: the ragged_kda_decode kernel's MXU matvec.
        return lax.dot_general(
            vec.reshape(block_h, 1, head_dim),
            s,
            (((2,), (1,)), ((0,), (0,))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=f32,
        ).reshape(block_h, v_dim)

    def issue(j, carry):
        live, keep, slot = decode(code_ref[base + j])
        del live

        @pl.when(keep)
        def _start():
            for cp in reads(j, slot):
                cp.start()

        return carry

    lax.fori_loop(0, block_tokens, issue, 0)

    def conv(sx, wb, w_ref, hist, compute, pool_dtype, j):
        x = _rnd(sx[pl.ds(j, 1), :], compute)  # [1, D]
        old = wb[j].astype(f32)  # [d_conv, D]; zeros for a fresh row
        taps = [old[t : t + 1, :] for t in range(1, d_conv)] + [x]
        wts = w_ref[...].astype(f32)  # [d_conv, D]
        mixed = taps[0] * wts[0:1, :]
        for t in range(1, d_conv):
            mixed = mixed + taps[t] * wts[t : t + 1, :]
        y = jax.nn.silu(mixed)
        for t in range(d_conv):
            hist[t : t + 1, :] = taps[t]
        wb[j] = hist[...].astype(pool_dtype)
        return _rnd(_rnd(y, compute), conv_dtype), y

    def split_heads(y: Array, dst, width: int) -> None:
        for h in range(num_heads):
            dst[h : h + 1, :] = y[:, h * width : (h + 1) * width]

    def body(j, carry):
        live, keep, slot = decode(code_ref[base + j])

        @pl.when(keep)
        def _wait():
            for cp in reads(j, slot):
                cp.wait()

        @pl.when(jnp.logical_and(live, jnp.logical_not(keep)))
        def _fresh():
            for a in range(4):
                bufs[a][j] = jnp.zeros(bufs[a].shape[1:], bufs[a].dtype)

        @pl.when(live)
        def _advance():
            yq_r, yq = conv(sxq, wbq, wq_ref, hq, computes[0], pool_dtypes[0], j)
            yk_r, yk = conv(sxk, wbk, wk_ref, hk, computes[1], pool_dtypes[1], j)
            yv_r, yv = conv(sxv, wbv, wv_ref, hv, computes[2], pool_dtypes[2], j)
            split_heads(yq_r, qr, head_dim)
            split_heads(yq, qu, head_dim)
            split_heads(yk_r, kr, head_dim)
            split_heads(yk, ku, head_dim)
            split_heads(yv_r, vr, v_dim)
            split_heads(yv, vu, v_dim)
            if use_qk_l2norm:
                q_all = l2("q", qr[...], qu[...])
                k_all = l2("k", kr[...], ku[...])
            else:
                q_all = pick("q_mul", qr[...], qu[...])
                k_all = pick("k_mul", kr[...], ku[...])
            q_all = q_all * scale
            v_all = pick("v", vr[...], vu[...])
            # beta [1, H] -> column [H, 1] (one nonzero per row: exact) -> [H, V].
            row = sbeta[pl.ds(j, 1), :]
            eye = lax.broadcasted_iota(jnp.int32, (num_heads, num_heads), 0) == lax.broadcasted_iota(
                jnp.int32, (num_heads, num_heads), 1
            )
            col = jnp.sum(jnp.where(eye, jnp.broadcast_to(row, (num_heads, num_heads)), 0.0), axis=1, keepdims=True)
            b_all = jnp.broadcast_to(col, (num_heads, v_dim))
            for h0 in range(0, num_heads, block_h):
                hs = slice(h0, h0 + block_h)
                k = k_all[hs]
                eg = jnp.exp(decay_ref[j, hs, :].astype(f32))
                s = stb[j, hs, :, :].astype(f32) * eg[:, :, None]
                delta = (v_all[hs] - contract(k, s)) * b_all[hs]
                s = s + k[:, :, None] * delta[:, None, :]
                o_ref[j, hs, :] = contract(q_all[hs], s).astype(o_ref.dtype)
                stb[j, hs, :, :] = s.astype(rec_dtype)
            for cp in writes(j, slot):
                cp.start()

        @pl.when(jnp.logical_not(live))
        def _idle():
            o_ref[j] = jnp.zeros(o_ref.shape[1:], o_ref.dtype)

        return carry

    lax.fori_loop(0, block_tokens, body, 0)

    def drain(j, carry):
        live, keep, slot = decode(code_ref[base + j])
        del keep

        @pl.when(live)
        def _wait():
            for cp in writes(j, slot):
                cp.wait()

        return carry

    lax.fori_loop(0, block_tokens, drain, 0)


def ragged_kda_conv_decode_pallas(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    conv_weights: tuple[Array, Array, Array],
    conv_pools: tuple[Array, Array, Array],
    recurrent_state: Array,
    token_slots: Array,
    token_init: Array,
    *,
    conv_dtype: tp.Any,
    use_qk_l2norm: bool = True,
    block_tokens: int = 0,
    rounding: tuple[bool, ...] | None = EXACT_ROUNDING,
    rounding_flags: Array | None = None,
    num_cores: int = 0,
    interpret: bool = False,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Fused KDA decode step over the packed token stream (Pallas TPU; pools updated in place).

    Args:
        query: Raw (pre-conv) queries ``[T, H*K]``.
        key: Raw keys ``[T, H*K]``.
        value: Raw values ``[T, H*V]``.
        beta: Write strengths ``[T, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[T, H, K]``.
        conv_weights: Depthwise conv kernels ``(q, k, v)``, each ``[dim, d_conv]`` (last tap = current token).
        conv_pools: Conv windows ``(q [S, H*K, d_conv], k [S, H*K, d_conv], v [S, H*V, d_conv])``.
        recurrent_state: State pool ``[S, H, K, V]``.
        token_slots: ``[T]`` pool slot of the row whose single token is ``t`` (``< 0`` or ``>= S``: the token outputs
            zeros and changes nothing).
        token_init: ``[T]`` whether that row continues its state (``False``: zero window and state).
        conv_dtype: Dtype of the conv outputs (the reference rounds the SiLU output to it).
        use_qk_l2norm: L2-normalise queries and keys.
        block_tokens: Tokens per grid step (``0``: from the VMEM budget, :func:`plan_blocks`).
        rounding: Static rounding recipe over :data:`ROUNDING_POINTS`, or ``None`` to read ``rounding_flags``.
        rounding_flags: ``[len(ROUNDING_POINTS)]`` int32 flags read at run time when ``rounding`` is ``None``.
        num_cores: TensorCores to split the tokens over (``0``: the chip's).
        interpret: Run the kernel in the TPU interpreter (logic checks only).

    Returns:
        ``(out [T, H, V] fp32, (q_pool, k_pool, v_pool, recurrent_state))``; non-advancing tokens output zeros.

    Raises:
        ValueError: If no block of tokens fits the VMEM budget, or ``block_tokens`` breaks the block rules.
    """
    num_tokens, key_dim = query.shape
    num_heads = int(beta.shape[-1])
    head_dim = key_dim // num_heads
    value_dim = int(value.shape[-1])
    v_dim = value_dim // num_heads
    num_slots = int(recurrent_state.shape[0])
    d_conv = int(conv_weights[0].shape[-1])
    i32 = jnp.int32
    f32 = jnp.float32
    rec_dtype = recurrent_state.dtype
    pool_dtypes = tuple(p.dtype for p in conv_pools)
    stream_dtypes = (query.dtype, key.dtype, value.dtype)
    computes = tuple(jnp.promote_types(s, p) for s, p in zip(stream_dtypes, pool_dtypes, strict=True))
    bh = pick_block_h(num_heads)
    eps = float(np.asarray(L2_EPS, dtype=jnp.dtype(conv_dtype)).astype(np.float32))

    slots = jnp.asarray(token_slots).astype(i32).reshape(-1)
    live = (slots >= 0) & (slots < num_slots)
    code = jnp.where(live, slots * 2 + jnp.asarray(token_init).reshape(-1).astype(i32), -1)
    if rounding_flags is None:
        flags = jnp.zeros((len(ROUNDING_POINTS),), i32)
    else:
        flags = jnp.asarray(rounding_flags).astype(i32)

    plan = block_plan(
        query, key, value, beta, decay, conv_weights, conv_pools, recurrent_state,
        num_cores=num_cores, block_tokens=block_tokens,
    )  # fmt: skip
    if plan is None:
        raise ValueError(
            f"ragged_kda_conv_decode: no block of the {num_tokens} tokens fits VMEM (block_tokens={block_tokens})"
        )
    cores, bt, vmem_limit = plan
    steps = num_tokens // (cores * bt)

    def tok_map(c, i, *_):
        return (c * steps + i, 0)

    def tok3_map(c, i, *_):
        return (c * steps + i, 0, 0)

    def whole(c, i, *_):
        return (0, 0)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    views = [jnp.swapaxes(p, 1, 2) for p in conv_pools]  # [S, d_conv, dim]: the pools' physical layout
    wts = [jnp.swapaxes(w, 0, 1) for w in conv_weights]  # [d_conv, dim]
    kernel = functools.partial(
        _kernel,
        block_tokens=bt,
        steps=steps,
        num_heads=num_heads,
        head_dim=head_dim,
        v_dim=v_dim,
        d_conv=d_conv,
        block_h=bh,
        computes=computes,
        conv_dtype=jnp.dtype(conv_dtype),
        pool_dtypes=pool_dtypes,
        rec_dtype=rec_dtype,
        eps=eps,
        rounding=None if rounding is None else tuple(bool(r) for r in rounding),
        use_qk_l2norm=bool(use_qk_l2norm),
    )
    tag = "rt" if rounding is None else "".join("1" if r else "0" for r in rounding)
    out, nq, nk, nv, new_rec = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(cores, steps),
            in_specs=[
                pl.BlockSpec((bt, key_dim), tok_map),
                pl.BlockSpec((bt, key_dim), tok_map),
                pl.BlockSpec((bt, value_dim), tok_map),
                pl.BlockSpec((bt, num_heads), tok_map),
                pl.BlockSpec((bt, num_heads, head_dim), tok3_map),
                pl.BlockSpec((d_conv, key_dim), whole),
                pl.BlockSpec((d_conv, key_dim), whole),
                pl.BlockSpec((d_conv, value_dim), whole),
                any_spec,
                any_spec,
                any_spec,
                any_spec,
            ],
            out_specs=[pl.BlockSpec((bt, num_heads, v_dim), tok3_map), any_spec, any_spec, any_spec, any_spec],
            scratch_shapes=[
                pltpu.VMEM((bt, key_dim), f32),
                pltpu.VMEM((bt, key_dim), f32),
                pltpu.VMEM((bt, value_dim), f32),
                pltpu.VMEM((bt, num_heads), f32),
                pltpu.VMEM((bt, d_conv, key_dim), pool_dtypes[0]),
                pltpu.VMEM((bt, d_conv, key_dim), pool_dtypes[1]),
                pltpu.VMEM((bt, d_conv, value_dim), pool_dtypes[2]),
                pltpu.VMEM((bt, num_heads, head_dim, v_dim), rec_dtype),
                pltpu.VMEM((d_conv, key_dim), f32),
                pltpu.VMEM((d_conv, key_dim), f32),
                pltpu.VMEM((d_conv, value_dim), f32),
                pltpu.VMEM((num_heads, head_dim), f32),
                pltpu.VMEM((num_heads, head_dim), f32),
                pltpu.VMEM((num_heads, head_dim), f32),
                pltpu.VMEM((num_heads, head_dim), f32),
                pltpu.VMEM((num_heads, v_dim), f32),
                pltpu.VMEM((num_heads, v_dim), f32),
                pltpu.SemaphoreType.DMA((4, bt)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, num_heads, v_dim), f32),
            jax.ShapeDtypeStruct(views[0].shape, views[0].dtype),
            jax.ShapeDtypeStruct(views[1].shape, views[1].dtype),
            jax.ShapeDtypeStruct(views[2].shape, views[2].dtype),
            jax.ShapeDtypeStruct(recurrent_state.shape, rec_dtype),
        ],
        # operands: code, flags, xq, xk, xv, beta, decay, wq, wk, wv, cq (10), ck, cv, rec (13) -> outputs 1..4.
        input_output_aliases={10: 1, 11: 2, 12: 3, 13: 4},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"ejkernel_ragged_kda_conv_decode_c{cores}_bt{bt}_bh{bh}_r{tag}",
    )(code, flags, query, key, value, beta, decay, *wts, *views, recurrent_state)
    nq, nk, nv = (jnp.swapaxes(p, 1, 2) for p in (nq, nk, nv))
    return out, (nq, nk, nv, new_rec)


__all__ = (
    "EXACT_ROUNDING",
    "ROUNDING_POINTS",
    "block_plan",
    "plan_blocks",
    "ragged_kda_conv_decode_pallas",
    "tensor_cores",
    "unsupported_reason",
    "vmem_capacity_bytes",
)
