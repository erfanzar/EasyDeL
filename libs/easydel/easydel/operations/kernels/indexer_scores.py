# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Fused Lightning-indexer scoring operation (adapter over ejkernel ``indexer_scores``).

``scores[b, s, t] = sum_h weights[b, h, s] * relu(query[b, h, s] . index_k[b, t])`` as fp32 ``[B, S, T]``. On TPU
the ejkernel Pallas kernel keeps the per-head ``[B, H, S, T]`` logits in VMEM (forward and both backward passes);
elsewhere the ejkernel XLA reference runs and GSPMD partitions it. A Mosaic call cannot be GSPMD-partitioned, so on
a multi-device mesh the TPU path takes explicit ``mesh`` / ``in_specs`` / ``out_specs`` and ejkernel runs it under
``jax.shard_map``; the caller resolves those specs (see ``easydel.layers.indexer``).
"""

from __future__ import annotations

import functools
import typing as tp

import jax
from eformer.pytree import auto_pytree
from ejkernel.modules import indexer_scores as ejk_indexer_scores  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

from .._operation_impl import OperationImpl, OperationOutput, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder

__all__ = ("IndexerScoresOp", "IndexerScoresOutput", "indexer_scores_op")


@auto_pytree
class IndexerScoresOutput(OperationOutput):
    """Result of one :class:`IndexerScoresOp` call.

    Attributes:
        scores: fp32 head-summed scores ``[B, S, T]`` (differentiable in all three operands).
    """

    scores: Array | None = None


@OperationRegistry.register
class IndexerScoresOp(OperationImpl):
    """Head-summed weighted-relu indexer scoring (DSA / Lightning indexers, dense training and prefill).

    Stateless and cache-agnostic: every operand is passed explicitly. ``forward_tpu`` runs the ejkernel Pallas
    kernel (under ``shard_map`` when a mesh is given); every other backend runs the ejkernel XLA reference.
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the operation registry key."""
        return "indexer_scores"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """No metadata fields and no cache view: every input is passed explicitly."""
        del mode
        return RequirementsBuilder("indexer_scores").require_metadata(MetadataField.NONE).no_cache_required().build()

    @jax.named_scope("easydel-indexer_scores")
    def forward_native(
        self,
        query: Float[Array, "batch heads seq dim"],
        index_k: Float[Array, "batch entries dim"],
        weights: Float[Array, "batch heads seq 1"],
        *,
        precision: jax.lax.PrecisionLike = None,
        cfg: tp.Any = None,
        **ignores,
    ) -> IndexerScoresOutput:
        """XLA reference (ejkernel ``platform="xla"``); partitioned by GSPMD, so mesh arguments are ignored.

        Args:
            query: Index queries ``[B, H, S, d]``.
            index_k: Index keys ``[B, T, d]``.
            weights: Per-head weights ``[B, H, S, 1]`` (all scaling folded in).
            precision: Contraction precision (``None`` = operand dtype, ``high`` / ``highest`` = fp32 operands).
            cfg: Optional ejkernel ``IndexerScoresConfig``.
            **ignores: Ignored (``mesh`` / ``in_specs`` / ``out_specs`` and interface parity).

        Returns:
            IndexerScoresOutput: the fp32 ``[B, S, T]`` scores.
        """
        del ignores
        return IndexerScoresOutput(
            scores=ejk_indexer_scores(query, index_k, weights, precision=precision, platform="xla", cfg=cfg)
        )

    @jax.named_scope("easydel-indexer_scores")
    def forward_tpu(
        self,
        query: Float[Array, "batch heads seq dim"],
        index_k: Float[Array, "batch entries dim"],
        weights: Float[Array, "batch heads seq 1"],
        *,
        precision: jax.lax.PrecisionLike = None,
        cfg: tp.Any = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, PartitionSpec, PartitionSpec] | None = None,
        out_specs: PartitionSpec | None = None,
        **ignores,
    ) -> IndexerScoresOutput:
        """Pallas TPU kernel (ejkernel ``platform="pallas"``), per device under ``shard_map`` when ``mesh`` is set.

        Args:
            query: Index queries ``[B, H, S, d]``.
            index_k: Index keys ``[B, T, d]``.
            weights: Per-head weights ``[B, H, S, 1]``.
            precision: Contraction precision.
            cfg: Optional ejkernel ``IndexerScoresConfig`` (tile requests).
            mesh: Optional device mesh; required on a multi-device mesh outside a fully manual region.
            in_specs: Specs of ``(query, index_k, weights)`` (reduced dims replicated; checked by ejkernel).
            out_specs: Spec of the ``[B, S, T]`` output.
            **ignores: Ignored (interface parity).

        Returns:
            IndexerScoresOutput: the fp32 ``[B, S, T]`` scores.
        """
        del ignores
        scores = ejk_indexer_scores(
            query,
            index_k,
            weights,
            precision=precision,
            platform="pallas",
            cfg=cfg,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )
        return IndexerScoresOutput(scores=scores)


@functools.lru_cache(maxsize=1)
def indexer_scores_op() -> IndexerScoresOp:
    """Process-wide :class:`IndexerScoresOp` (stateless; built without a mesh so any layer may call it)."""
    from .._operation_meta import OperationMetadata

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=None,
        platform=None,
        backend=None,
    )
    return IndexerScoresOp(metadata)
