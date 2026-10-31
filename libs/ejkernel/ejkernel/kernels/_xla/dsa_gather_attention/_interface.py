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

"""Gather DSA attention interface (XLA reference, mandatory fallback)."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._layout import validate
from ._xla_impl import dsa_gather_attention_xla


@kernel_registry.register("dsa_gather_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def dsa_gather_attention(
    query: Float[Array, "batch seq num_heads head_dim"],
    latent: Float[Array, "batch kv_seq head_dim"],
    indices: Int[Array, "batch seq num_selected"],
    positions: Int[Array, "batch seq"] | None = None,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    include_tail: bool = True,
    block_q: int = 16,
    pools_per_step: int | None = None,
    num_buffers: int | None = None,
    num_cores: int | None = None,
    fused_backward: bool | None = None,
    interpret: bool | None = None,
) -> Float[Array, "batch seq num_heads head_dim"]:
    """Attention of each query over its selected pools plus its own-pool tail (XLA reference).

    Registered under ``"dsa_gather_attention"`` for ``Platform.XLA`` / ``Backend.ANY``; defines the value and
    gradient contract of the Pallas TPU kernel. ``block_q``, ``pools_per_step``, ``num_buffers``, ``num_cores``,
    ``fused_backward`` and ``interpret`` only shape the kernel and are ignored here.

    Args:
        query: ``[B, N, H, D]`` queries (absorbed-MLA queries for GLM-5-style DSA), query-major.
        latent: ``[B, S, D]`` shared ``K == V`` latent.
        indices: ``[B, N, K]`` int selected pool ids (pool ``p`` = keys ``p*kpool .. p*kpool + kpool - 1``,
            ``-1`` = none). Only complete causal pools are attended; valid ids must be distinct.
        positions: ``[B, N]`` absolute query positions into ``latent`` (negative = invalid query, output 0);
            ``None`` = ``0 .. N - 1``.
        kpool: Keys per pool.
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        include_tail: Open each query's own incomplete pool ``kpool * (pos // kpool) .. pos``.
        block_q: Kernel query block (ignored).
        pools_per_step: Kernel pools per DMA group (ignored).
        num_buffers: Kernel DMA ring depth (ignored).
        num_cores: Kernel TensorCore split of the backward (ignored).
        fused_backward: Kernel fused-row backward choice (ignored).
        interpret: Pallas interpret mode (ignored).

    Returns:
        ``[B, N, H, D]`` in ``latent``'s dtype.
    """
    validate(query.shape, latent.shape, indices.shape, None if positions is None else positions.shape, int(kpool))
    _ = (block_q, pools_per_step, num_buffers, num_cores, fused_backward, interpret)
    return dsa_gather_attention_xla(
        query,
        latent,
        indices,
        positions,
        kpool=int(kpool),
        softmax_scale=softmax_scale,
        include_tail=bool(include_tail),
    )
