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

"""Gather DSA attention, XLA reference: gather each query's open keys, fp32 softmax, differentiable by autodiff."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from ._layout import default_positions, selection_layout

NEG = -1e30


def dsa_gather_attention_xla(
    query: Array,
    latent: Array,
    indices: Array,
    positions: Array | None = None,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    include_tail: bool = True,
    query_chunk: int = 128,
) -> Array:
    """Attention of each query over its open keys (see ``_layout``), ``query_chunk`` queries at a time.

    Args:
        query: ``[B, N, H, D]`` queries.
        latent: ``[B, S, D]`` shared ``K == V`` latent.
        indices: ``[B, N, K]`` selected pool ids (``-1`` = none).
        positions: ``[B, N]`` absolute query positions (negative = invalid); ``None`` = ``0 .. N - 1``.
        kpool: Keys per pool.
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        include_tail: Open each query's own incomplete pool.
        query_chunk: Queries gathered per step (bounds the ``[B, C, (K + 1) * kpool, D]`` transient).

    Returns:
        ``[B, N, H, D]`` in ``latent``'s dtype; queries without an open key give 0.
    """
    bsz, n, h, d = query.shape
    s_len = latent.shape[1]
    k_sel = indices.shape[-1]
    if positions is None:
        positions = default_positions(bsz, n)
    scale = d**-0.5 if softmax_scale is None else float(softmax_scale)
    selected_ok, tail_len, own = selection_layout(indices, positions, kpool=kpool, include_tail=include_tail)
    rr = jnp.arange(kpool, dtype=jnp.int32)
    pools = jnp.concatenate([own[..., None], jnp.where(selected_ok, indices, 0).astype(jnp.int32)], axis=-1)
    key_ok = jnp.concatenate(
        [
            (rr < tail_len[..., None])[..., None, :],
            jnp.broadcast_to(selected_ok[..., None], (*selected_ok.shape, kpool)),
        ],
        axis=-2,
    )
    width = (k_sel + 1) * kpool
    key_ids = jnp.clip(pools[..., None] * kpool + rr, 0, s_len - 1).reshape(bsz, n, width)
    key_ok = key_ok.reshape(bsz, n, width)

    chunk = max(1, min(int(query_chunk), n))
    n_chunks = -(-n // chunk)
    pad = n_chunks * chunk - n
    if pad:
        query = jnp.pad(query, ((0, 0), (0, pad), (0, 0), (0, 0)))
        key_ids = jnp.pad(key_ids, ((0, 0), (0, pad), (0, 0)))
        key_ok = jnp.pad(key_ok, ((0, 0), (0, pad), (0, 0)))

    def to_chunks(x: Array) -> Array:
        return jnp.moveaxis(x.reshape(bsz, n_chunks, chunk, *x.shape[2:]), 1, 0)

    lat32 = latent.astype(jnp.float32)
    hi = lax.Precision.HIGHEST

    def one(args):
        qc, ic, oc = args  # [B, C, H, D], [B, C, T], [B, C, T]
        keys = jax.vmap(lambda lat, ids: lat[ids])(lat32, ic)  # [B, C, T, D]
        sc = jnp.einsum("bchd,bctd->bcht", qc.astype(jnp.float32), keys, precision=hi) * scale
        open_ = oc[:, :, None, :]
        sc = jnp.where(open_, sc, NEG)
        m = lax.stop_gradient(jnp.max(sc, axis=-1, keepdims=True))
        p = jnp.where(open_, jnp.exp(sc - m), 0.0)
        denom = jnp.sum(p, axis=-1, keepdims=True)
        p = p / jnp.where(denom > 0, denom, 1.0)
        return jnp.einsum("bcht,bctd->bchd", p, keys, precision=hi)

    out = lax.map(one, (to_chunks(query), to_chunks(key_ids), to_chunks(key_ok)))
    out = jnp.moveaxis(out, 0, 1).reshape(bsz, n_chunks * chunk, h, d)[:, :n]
    return out.astype(latent.dtype)
