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

"""Registry entry for the XLA ``indexer_scores`` reference."""

import typing as tp

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import indexer_scores_xla


@kernel_registry.register("indexer_scores", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def indexer_scores(
    query: Float[Array, "batch num_heads q_len head_dim"],
    index_k: Float[Array, "batch num_entries head_dim"],
    weights: Float[Array, "batch num_heads q_len 1"],
    fwd_params: FwdParams | None = None,
    precision: tp.Any = None,
) -> Float[Array, "batch q_len num_entries"]:
    """Head-summed weighted-relu indexer scores, XLA reference (materialises the ``[B, H, S, T]`` logits).

    Registered under ``"indexer_scores"`` for ``Platform.XLA`` / ``Backend.ANY`` (the mandatory fallback).
    Differentiable by autodiff; the Pallas TPU kernel matches it in value and gradient.

    Args:
        query: ``[batch, num_heads, q_len, head_dim]`` index queries.
        index_k: ``[batch, num_entries, head_dim]`` index keys.
        weights: ``[batch, num_heads, q_len, 1]`` per-head, per-query weights.
        fwd_params: Ignored (accepted for signature parity with the Pallas kernel).
        precision: ``None`` / ``"default"`` = operand dtype; ``"high"`` / ``"highest"`` = fp32 operands.

    Returns:
        fp32 ``[batch, q_len, num_entries]``.
    """
    return indexer_scores_xla(query, index_k, weights, fwd_params, precision)


__all__ = ("indexer_scores",)
