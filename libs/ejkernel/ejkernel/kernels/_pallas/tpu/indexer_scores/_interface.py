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

"""Fused indexer scoring interface (Pallas TPU)."""

import typing as tp

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.ops import FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import indexer_scores_tpu


@kernel_registry.register("indexer_scores", Platform.PALLAS, Backend.TPU, priority=10)
@jaxtyping.jaxtyped(typechecker=beartype)
def indexer_scores(
    query: Float[Array, "batch num_heads q_len head_dim"],
    index_k: Float[Array, "batch num_entries head_dim"],
    weights: Float[Array, "batch num_heads q_len 1"],
    fwd_params: FwdParams | None = None,
    precision: tp.Any = None,
) -> Float[Array, "batch q_len num_entries"]:
    """``sum_h weights[b, h, s] * relu(query[b, h, s] . index_k[b, t])`` without the ``[B, H, S, T]`` logits.

    Registered under ``"indexer_scores"`` for ``Platform.PALLAS`` / ``Backend.TPU``; same signature and math as the
    XLA reference, differentiable in all three operands through tiled backward kernels.

    Args:
        query: Index queries ``[batch, num_heads, q_len, head_dim]``.
        index_k: Index keys ``[batch, num_entries, head_dim]``.
        weights: Per-head, per-query weights ``[batch, num_heads, q_len, 1]``.
        fwd_params: Optional tiling (``q_blocksize``, ``kv_blocksize`` = entry tile); rounded and fitted to VMEM.
        precision: ``None`` / ``"default"`` = operand-dtype MXU pass; ``"high"`` / ``"highest"`` / a
            ``jax.lax.Precision`` = fp32 tiles in-kernel.

    Returns:
        fp32 ``[batch, q_len, num_entries]`` scores.
    """
    return indexer_scores_tpu(query, index_k, weights, fwd_params=fwd_params, precision=precision)


__all__ = ("indexer_scores",)
