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

"""Decode-size mHC read (gates + collapse) and residual write as two Pallas TPU kernels. APPROXIMATE class.

Serving decode only (forward, at most 128 tokens per device). Off by default; callers opt in explicitly.

* :func:`mhc_decode_read_pallas` runs the packed coefficient kernel's arithmetic verbatim on ``logits.T`` (one lane per
  token), moves the gates to a token-major table ``[splits, T, lanes]`` with an in-kernel transpose (data movement
  only) and computes the collapse ``sum_h pre_h x_h`` with XLA's four-term tree ``(x0 + x2) + (x1 + x3)``.
* :func:`mhc_decode_write_pallas` computes ``post_i * out + sum_j comb_ji x_j`` from that table with the gates rounded to
  the stream dtype, a sequential four-term sum in float32 and one final rounding, and writes the streams flattened
  ``[T, hc * D]``.

Measured on TPU v5p against the unfused XLA path: the read (gates and collapse) matches bit for bit. The write is
within one bf16 ulp of ``manifold_residual_write``, whose own rounding depends on the fusion XLA places it in
(standalone, 0 to 1 of 524288 elements differed; 6 of 2097152 at 128 tokens). A projection dot that reads the
kernel's flat output also accumulates in another order (119 of 768 logits differ, max 4.8e-7). Over 45
GLM-5.3-Flash-shaped layers (32 rows, 4 x 4096 bf16 streams, random weights): after one layer 464 of 131072 sublayer
inputs differ (max 1.6e-2, about one bf16 ulp) and 97722 of 524288 stream elements (max 0.125); after 45 layers most
elements differ. In the same chain the two kernels cut the mHC glue from 39.3 to 25.8 us per layer of device time
(tmp-files/glm53_serve/notes/t1_mhc.md).
"""

from __future__ import annotations

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...._xla.mhc_decode_fused._xla_impl import LANES as _LANES
from ...._xla.mhc_decode_fused._xla_impl import gate_table_layout, supports


def _tpu_field(name: str, default: int) -> int:
    try:
        return int(getattr(pltpu.get_tpu_info(), name))
    except Exception:  # not a TPU (interpret mode) or an older runtime
        return default


def _params(estimate: int) -> pltpu.CompilerParams:
    capacity = _tpu_field("vmem_capacity_bytes", 64 << 20)
    limit = int(min(capacity - (2 << 20), max(16 << 20, 2 * estimate)))
    return pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=limit)


def _chunk(width: int) -> int:
    for size in (512, 256, 128):
        if width % size == 0:
            return size
    return width


def _pick(table, lanes, k):
    """Column ``k`` of ``table`` as ``[T, 1]``: one value summed with zeros, so exact."""
    return jnp.sum(jnp.where(lanes == k, table, 0.0), axis=1, keepdims=True)


def _read_kernel(l_ref, b_ref, s_ref, *refs, hc, n_iters, eps, tokens, slab, chunk):
    x_refs = refs[:hc]
    table_ref, out_ref, scratch = refs[hc:]
    logits, base, scale = l_ref[...], b_ref[...], s_ref[...]
    pre = jax.nn.sigmoid(logits[:hc, :] * scale[0, 0] + base[:hc, :]) + eps
    post = 2 * jax.nn.sigmoid(logits[hc : 2 * hc, :] * scale[1, 0] + base[hc : 2 * hc, :])
    mix = (logits[2 * hc :, :] * scale[2, 0] + base[2 * hc :, :]).reshape(hc, hc, tokens)
    matrix = jnp.exp(mix - jnp.max(mix, axis=1, keepdims=True))
    matrix = matrix / jnp.sum(matrix, axis=1, keepdims=True) + eps
    matrix = matrix / (jnp.sum(matrix, axis=0, keepdims=True) + eps)

    def body(_, value):
        value = value / (jnp.sum(value, axis=1, keepdims=True) + eps)
        return value / (jnp.sum(value, axis=0, keepdims=True) + eps)

    matrix = lax.fori_loop(0, n_iters - 1, body, matrix)
    scratch[...] = jnp.zeros(scratch.shape, jnp.float32)
    scratch[0:hc, 0:tokens] = pre
    scratch[slab : slab + hc, 0:tokens] = post
    for row in range(hc):
        scratch[(2 + row) * slab : (2 + row) * slab + hc, 0:tokens] = matrix[row]
    table = scratch[...].T[:tokens]
    table_ref[0] = table
    lanes = lax.broadcasted_iota(jnp.int32, table.shape, 1)
    gates = [_pick(table, lanes, h) for h in range(hc)]
    for k in range(out_ref.shape[1] // chunk):
        cols = pl.ds(k * chunk, chunk)
        terms = [gates[h] * x_refs[h][:, cols].astype(jnp.float32) for h in range(hc)]
        while len(terms) > 1:
            half = len(terms) // 2
            terms = [terms[i] + terms[i + half] for i in range(half)]
        out_ref[:, cols] = terms[0].astype(out_ref.dtype)


def mhc_decode_read_pallas(
    logits: jax.Array,
    base: jax.Array,
    scale: jax.Array,
    streams: jax.Array,
    *,
    n_iters: int = 20,
    eps: float = 1e-6,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Gates and collapse for a decode token block.

    Args:
        logits: Float32 projection logits ``[T, (2 + hc) * hc]`` (RMS scale applied), ``T <= 128``.
        base: Gate offsets ``[(2 + hc) * hc]``.
        scale: Gate/mixer multipliers ``[3]``.
        streams: Streams flattened per token ``[T, hc * D]`` (stream ``h`` in columns ``[h D, (h + 1) D)``).
        n_iters: Sinkhorn iterations (positive).
        eps: Gate floor and normalization offset.
        interpret: Pallas interpret mode (default: off TPU only).

    Returns:
        ``table [splits, T, lanes]`` float32 (see :func:`gate_table_layout`; every split holds the same gates) and
        ``collapsed [T, D]`` in ``streams.dtype``.

    Raises:
        ValueError: Unsupported shapes.
    """
    tokens, mix = logits.shape
    hc = round((-2 + (4 + 4 * mix) ** 0.5) / 2)
    if (2 + hc) * hc != mix or streams.shape[0] != tokens or streams.shape[1] % hc or base.shape != (mix,):
        raise ValueError(f"mhc_decode_read: logits {logits.shape}, base {base.shape}, streams {streams.shape}")
    width = streams.shape[1] // hc
    if not supports(tokens, hc, width) or scale.shape != (3,) or n_iters < 1:
        raise ValueError(f"mhc_decode_read: unsupported tokens={tokens} hc={hc} width={width} n_iters={n_iters}")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    slab, lanes = gate_table_layout(hc)
    cores = _tpu_field("num_cores", 1)
    splits = cores if width % (cores * _LANES) == 0 else 1
    part = width // splits
    item = jnp.dtype(streams.dtype).itemsize
    estimate = 2 * (hc + 1) * tokens * part * item + 6 * lanes * _LANES * 4 + 8 * tokens * _chunk(part) * 4

    def kernel(*refs):
        _read_kernel(*refs, hc=hc, n_iters=int(n_iters), eps=float(eps), tokens=tokens, slab=slab, chunk=_chunk(part))

    return pl.pallas_call(
        kernel,
        grid=(splits,),
        in_specs=[
            pl.BlockSpec((mix, tokens), lambda c: (0, 0)),
            pl.BlockSpec((mix, 1), lambda c: (0, 0)),
            pl.BlockSpec((3, 1), lambda c: (0, 0)),
        ]
        + [pl.BlockSpec((tokens, part), lambda c, h=h: (0, h * splits + c)) for h in range(hc)],
        out_specs=(
            pl.BlockSpec((1, tokens, lanes), lambda c: (c, 0, 0)),
            pl.BlockSpec((tokens, part), lambda c: (0, c)),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((splits, tokens, lanes), jnp.float32),
            jax.ShapeDtypeStruct((tokens, width), streams.dtype),
        ),
        scratch_shapes=[pltpu.VMEM((lanes, _LANES), jnp.float32)],
        compiler_params=_params(estimate),
        interpret=interpret,
        name="mhc_decode_read",
    )(
        logits.astype(jnp.float32).T,
        base.astype(jnp.float32)[:, None],
        scale.astype(jnp.float32)[:, None],
        *([streams] * hc),
    )


def _write_kernel(x_ref, o_ref, t_ref, y_ref, *, hc, slab, width, per, chunk):
    table = t_ref[0]
    lanes = lax.broadcasted_iota(jnp.int32, table.shape, 1)
    dtype = y_ref.dtype

    def gate(k):
        return _pick(table, lanes, k).astype(dtype).astype(jnp.float32)

    for local in range(per):
        i = pl.program_id(0) * per + local
        post = gate(slab + i)
        comb = [gate((2 + j) * slab + i) for j in range(hc)]
        for k in range(width // chunk):
            out = o_ref[:, pl.ds(k * chunk, chunk)].astype(jnp.float32)
            acc = comb[0] * x_ref[:, pl.ds(k * chunk, chunk)].astype(jnp.float32)
            for j in range(1, hc):
                acc = acc + comb[j] * x_ref[:, pl.ds(j * width + k * chunk, chunk)].astype(jnp.float32)
            y_ref[:, pl.ds(local * width + k * chunk, chunk)] = (post * out + acc).astype(dtype)


def mhc_decode_write_pallas(
    streams: jax.Array, out: jax.Array, table: jax.Array, *, interpret: bool | None = None
) -> jax.Array:
    """Residual write ``post_i * out + sum_j comb_ji x_j`` from a :func:`mhc_decode_read_pallas` gate table.

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        out: Sub-layer output ``[T, D]``.
        table: Gate table ``[splits, T, lanes]`` from :func:`mhc_decode_read_pallas` on the same streams.
        interpret: Pallas interpret mode (default: off TPU only).

    Returns:
        New streams ``[T, hc * D]`` in ``streams.dtype``.

    Raises:
        ValueError: Shapes that do not match.
    """
    tokens, total = streams.shape
    width = out.shape[-1]
    hc = total // width if width else 0
    if hc * width != total or out.shape != (tokens, width) or table.ndim != 3 or table.shape[1] != tokens:
        raise ValueError(f"mhc_decode_write: streams {streams.shape}, out {out.shape}, table {table.shape}")
    slab, lanes = gate_table_layout(hc)
    if table.shape[2] != lanes or not supports(tokens, hc, width):
        raise ValueError(f"mhc_decode_write: unsupported streams {streams.shape} / table {table.shape}")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    cores = _tpu_field("num_cores", 1)
    splits = cores if hc % cores == 0 else 1
    per = hc // splits
    item = jnp.dtype(streams.dtype).itemsize
    estimate = 2 * (tokens * total * item + tokens * width * item + tokens * lanes * 4 + tokens * per * width * item)

    def kernel(*refs):
        _write_kernel(*refs, hc=hc, slab=slab, width=width, per=per, chunk=_chunk(width))

    return pl.pallas_call(
        kernel,
        grid=(splits,),
        in_specs=[
            pl.BlockSpec((tokens, total), lambda c: (0, 0)),
            pl.BlockSpec((tokens, width), lambda c: (0, 0)),
            pl.BlockSpec((1, tokens, lanes), lambda c: (0, 0, 0)),
        ],
        out_specs=pl.BlockSpec((tokens, per * width), lambda c: (0, c)),
        out_shape=jax.ShapeDtypeStruct((tokens, total), streams.dtype),
        compiler_params=_params(estimate + 8 * tokens * _chunk(width) * 4),
        interpret=interpret,
        name="mhc_decode_write",
    )(streams, out, table)


__all__ = ("mhc_decode_read_pallas", "mhc_decode_write_pallas")
