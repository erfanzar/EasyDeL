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

"""Flash-style backward Pallas kernels of the compressed-window attention (no dense logits).

The probabilities are recomputed tile by tile from the forward's per-row statistics,
``p = exp((logits - m) - logl)`` (``m`` and ``log(ell)`` kept apart so a ``MASK_MIN`` max cannot round the log term
away), with ``delta = rowsum(dO * O)`` and ``ds = p * (dO @ kv^T - delta)``:

* the **dQ pass** walks, per ``(batch, q_block)``, exactly the KV tiles the forward visited (band window cut at the
  diagonal under the causal skip, then every entry tile), accumulates ``dq`` in fp32 scratch and writes the
  head-summed bias cotangent tile by tile into a zero-initialised aliased buffer (tiles never visited keep zeros);
* the **dKV pass** walks, per ``(batch, kv_tile)``, the q-blocks that can see the tile (the band's q window, the
  q-blocks at or below the diagonal, or all of them for entry tiles) and accumulates ``dkv`` with the key and value
  roles summed (``K == V``).

Memory is tile-sized and the work ``O(S * (window + entries))`` with band skip. Measured on one v5p chip (bf16,
fwd+bwd of the attention core) against a dense-recompute backward: H16 S4096 T2048 12.46 -> 3.31 ms (4.20 -> 0.27
GiB temporaries); H16 S16384 T16384 276.0 -> 77.5 ms (88.3 -> 4.3 GiB).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ._flash_fwd import matmul_precision
from ._layout import Tiling, compiler_params, n_visit, qb_lo, tile_lo, tile_of, token_mask_bias


def _recompute(q, do, kv_t, bias_t, m, logl, delta, softmax_scale: float):
    """Probabilities ``p`` and ``ds = d loss / d logits`` (both fp32 ``[block_h, block_q, block_kv]``) of one tile."""
    prec = matmul_precision(q.dtype)
    logits = jax.lax.dot_general(q, kv_t, (((2,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=prec)
    logits = logits * softmax_scale + bias_t[None, :, :]
    p = jnp.exp((logits - m) - logl)
    dp = jax.lax.dot_general(do, kv_t, (((2,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=prec)
    return p, p * (dp - delta)


def _dq_kernel(
    q_ref,
    do_ref,
    m_ref,
    logl_ref,
    delta_ref,
    kv_ref,
    bias_ref,
    tv_ref,
    _dbias_init_ref,
    dq_ref,
    dbias_ref,
    dq_acc,
    *,
    softmax_scale: float,
    t: Tiling,
):
    """``dq`` and the head-summed bias cotangent of one ``(batch, q_block)`` over its visited tiles."""
    qb = pl.program_id(1)
    step = pl.program_id(2)
    total = t.n_tok_iter + t.n_entry_tiles

    @pl.when(step == 0)
    def _init():
        dq_acc[...] = jnp.zeros(dq_acc.shape, dq_acc.dtype)

    def _accumulate(bias_t, write_dbias: bool):
        q = q_ref[...]
        kv_t = kv_ref[...]
        _, ds = _recompute(q, do_ref[...], kv_t, bias_t, m_ref[...], logl_ref[...], delta_ref[...], softmax_scale)
        dq_acc[...] += (
            jax.lax.dot_general(
                ds.astype(q.dtype),
                kv_t,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
                precision=matmul_precision(q.dtype),
            )
            * softmax_scale
        )
        if write_dbias:
            dbias_ref[...] = jnp.sum(ds, axis=0).astype(dbias_ref.dtype)

    is_tok = step < t.n_tok_iter

    @pl.when(jnp.logical_and(is_tok, step < n_visit(qb, t)))
    def _token():
        if t.implicit:
            _accumulate(token_mask_bias(tv_ref, qb, tile_lo(qb, t) + step, t), write_dbias=False)
        else:
            _accumulate(bias_ref[...].astype(jnp.float32), write_dbias=True)

    if t.n_entry_tiles:

        @pl.when(jnp.logical_not(is_tok))
        def _entry():
            _accumulate(bias_ref[...].astype(jnp.float32), write_dbias=True)

    @pl.when(step == total - 1)
    def _finish():
        dq_ref[...] = dq_acc[...].astype(dq_ref.dtype)


def _dkv_kernel(
    q_ref,
    do_ref,
    m_ref,
    logl_ref,
    delta_ref,
    kv_ref,
    bias_ref,
    tv_ref,
    dkv_ref,
    dkv_acc,
    *,
    softmax_scale: float,
    t: Tiling,
    n_inner: int,
    tile_base: int,
    token_region: bool,
    qblk,
    causal_lo: bool,
):
    """``dkv`` (key and value roles summed) of one ``(batch, kv_tile)`` over its visiting q-blocks."""
    tt = pl.program_id(1)
    j = pl.program_id(2)
    tile = tile_base + tt

    @pl.when(j == 0)
    def _init():
        dkv_acc[...] = jnp.zeros(dkv_acc.shape, dkv_acc.dtype)

    def _compute():
        q = q_ref[...]
        do = do_ref[...]
        kv_t = kv_ref[...]
        if token_region and t.implicit:
            bias_t = token_mask_bias(tv_ref, qblk(tt, j), tile, t)
        else:
            bias_t = bias_ref[...].astype(jnp.float32)
        p, ds = _recompute(q, do, kv_t, bias_t, m_ref[...], logl_ref[...], delta_ref[...], softmax_scale)
        prec = matmul_precision(q.dtype)
        acc = dkv_acc[...]
        for h in range(q.shape[0]):  # static head loop: two "TN" matmuls per head, no [H, bkv, d] intermediate
            dv_h = jax.lax.dot_general(
                p[h].astype(q.dtype), do[h], (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32,
                precision=prec,
            )  # fmt: skip
            dk_h = jax.lax.dot_general(
                ds[h].astype(q.dtype), q[h], (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32,
                precision=prec,
            )  # fmt: skip
            acc = acc + dv_h + dk_h * softmax_scale
        dkv_acc[...] = acc

    if causal_lo:
        pl.when(j >= qb_lo(tile, t))(_compute)
    else:
        _compute()

    @pl.when(j == n_inner - 1)
    def _finish():
        dkv_ref[...] = dkv_acc[...].astype(dkv_ref.dtype)


def flash_backward(
    q_p: Array,
    kv_p: Array,
    bias_p: Array,
    tv_p: Array,
    do_p: Array,
    m_p: Array,
    logl_p: Array,
    delta_p: Array,
    *,
    softmax_scale: float,
    t: Tiling,
) -> tuple[Array, Array, Array | None, Array]:
    """Launch the dQ and dKV passes over padded operands.

    Args:
        q_p: ``[B, H, q_pad, d_pad]`` queries.
        kv_p: ``[B, kv_pad, d_pad]`` KV.
        bias_p: Padded bias (see :func:`._layout.pad_operands`).
        tv_p: ``[B, 1, tv_len]`` token validity.
        do_p: ``[B, H, q_pad, d_pad]`` output cotangent in the query dtype (zero on padded / excluded rows).
        m_p: ``[B, H, q_pad, 1]`` forward running max.
        logl_p: ``[B, H, q_pad, 1]`` ``log`` of the forward normaliser.
        delta_p: ``[B, H, q_pad, 1]`` ``rowsum(dO * O)`` (fp32).
        softmax_scale: Logit scale.
        t: Static tiling.

    Returns:
        ``(dq_p, dkv_tok, dkv_ent, dbias_p)``: fp32 ``[B, H, q_pad, d_pad]``, fp32 ``[B, token_kv_pad, d_pad]``,
        fp32 ``[B, entries_pad, d_pad]`` or ``None``, and the bias cotangent shaped like ``bias_p``.
    """
    batch, num_heads, q_pad, d_pad = q_p.shape
    bq, bkv, n_qb = t.block_q, t.block_kv, t.n_qb
    interpret = jax.default_backend() != "tpu"
    cparams = compiler_params(t, ("parallel", "parallel", "arbitrary"))
    total = t.n_tok_iter + t.n_entry_tiles
    has_bias_cols = not (t.implicit and not t.n_entry_tiles)  # False only for the implicit-mask placeholder

    def _bias_col(qb, p):
        """Bias column tile streamed at dQ scan step ``p`` (entry-relative for an implicit mask)."""
        if not t.implicit:
            return tile_of(qb, p, t)
        return jnp.maximum(p - t.n_tok_iter, 0)

    def _bias_index(b, qb, p):
        if not has_bias_cols:
            return (b, 0, 0)
        return (b, qb, _bias_col(qb, p))

    def row(b, qb, p):
        return (b, 0, qb, 0)

    dq_p, dbias_p = pl.pallas_call(
        functools.partial(_dq_kernel, softmax_scale=softmax_scale, t=t),
        grid=(batch, n_qb, total),
        in_specs=[
            pl.BlockSpec((None, num_heads, bq, d_pad), row),  # q
            pl.BlockSpec((None, num_heads, bq, d_pad), row),  # dO
            pl.BlockSpec((None, num_heads, bq, 1), row),  # m
            pl.BlockSpec((None, num_heads, bq, 1), row),  # logl
            pl.BlockSpec((None, num_heads, bq, 1), row),  # delta
            pl.BlockSpec((None, bkv, d_pad), lambda b, qb, p: (b, tile_of(qb, p, t), 0)),  # kv tile
            pl.BlockSpec((None, bq, bkv), _bias_index),  # bias tile
            pl.BlockSpec((None, 1, tv_p.shape[-1]), lambda b, qb, p: (b, 0, 0)),  # token validity
            pl.BlockSpec(memory_space=pltpu.HBM),  # zero dbias buffer, aliased to output 1 and never read
        ],
        out_specs=[
            pl.BlockSpec((None, num_heads, bq, d_pad), row),
            pl.BlockSpec((None, bq, bkv), _bias_index),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((batch, num_heads, q_pad, d_pad), jnp.float32),
            jax.ShapeDtypeStruct(bias_p.shape, bias_p.dtype),
        ],
        scratch_shapes=[pltpu.VMEM((num_heads, bq, d_pad), jnp.float32)],
        input_output_aliases={8: 1},
        compiler_params=cparams,
        interpret=interpret,
    )(q_p, do_p, m_p, logl_p, delta_p, kv_p, bias_p, tv_p, jnp.zeros(bias_p.shape, bias_p.dtype))

    def _dkv_call(n_tiles: int, tile_base: int, n_inner: int, qblk, token_region: bool, causal_lo: bool = False):
        def qrow(b, tt, j):
            return (b, 0, qblk(tt, j), 0)

        def bias_index(b, tt, j):
            if token_region and t.implicit:
                return (b, 0, 0)  # unused: the token mask is built in-kernel
            col = tile_base + tt if not t.implicit else tt  # implicit: the bias holds only the entry columns
            return (b, qblk(tt, j), col)

        return pl.pallas_call(
            functools.partial(
                _dkv_kernel,
                softmax_scale=softmax_scale,
                t=t,
                n_inner=n_inner,
                tile_base=tile_base,
                token_region=token_region,
                qblk=qblk,
                causal_lo=causal_lo,
            ),
            grid=(batch, n_tiles, n_inner),
            in_specs=[
                pl.BlockSpec((None, num_heads, bq, d_pad), qrow),  # q
                pl.BlockSpec((None, num_heads, bq, d_pad), qrow),  # dO
                pl.BlockSpec((None, num_heads, bq, 1), qrow),  # m
                pl.BlockSpec((None, num_heads, bq, 1), qrow),  # logl
                pl.BlockSpec((None, num_heads, bq, 1), qrow),  # delta
                pl.BlockSpec((None, bkv, d_pad), lambda b, tt, j: (b, tile_base + tt, 0)),  # kv tile
                pl.BlockSpec((None, bq, bkv), bias_index),  # bias tile
                pl.BlockSpec((None, 1, tv_p.shape[-1]), lambda b, tt, j: (b, 0, 0)),  # token validity
            ],
            out_specs=pl.BlockSpec((None, bkv, d_pad), lambda b, tt, j: (b, tt, 0)),
            out_shape=jax.ShapeDtypeStruct((batch, n_tiles * bkv, d_pad), jnp.float32),
            scratch_shapes=[pltpu.VMEM((bkv, d_pad), jnp.float32)],
            compiler_params=cparams,
            interpret=interpret,
        )(q_p, do_p, m_p, logl_p, delta_p, kv_p, bias_p, tv_p)

    # q-blocks whose queries can see token tile tt under the band: keys [tt*bkv, (tt+1)*bkv) are visible from
    # queries s in [key, key + window), i.e. blocks floor(tt*bkv/bq) .. floor(((tt+1)*bkv + window - 2)/bq).
    n_q_iter = (bkv + t.window - 2) // bq + 2
    if t.band_skip and n_q_iter < n_qb:
        dkv_tok = _dkv_call(
            t.n_token_tiles, 0, n_q_iter, lambda tt, j: jnp.clip((tt * bkv) // bq, 0, n_qb - n_q_iter) + j, True
        )
    elif t.causal:
        dkv_tok = _dkv_call(t.n_token_tiles, 0, n_qb, lambda tt, j: jnp.maximum(j, qb_lo(tt, t)), True, True)
    else:
        dkv_tok = _dkv_call(t.n_token_tiles, 0, n_qb, lambda tt, j: j, True)
    dkv_ent = None
    if t.n_entry_tiles:
        dkv_ent = _dkv_call(t.n_entry_tiles, t.n_token_tiles, n_qb, lambda tt, j: j, False)
    return dq_p, dkv_tok, dkv_ent, dbias_p
