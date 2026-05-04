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

"""Topology and KV-cache planning utilities for eSurge pipeline-parallel inference.

Owns :class:`PipelineInferencePlan`, the frozen snapshot every PP-aware
component reads to learn (a) whether PP is active for the current model/mesh,
(b) which transformer layer lives on which stage rank, (c) which subset of
those layers carry KV state, and (d) how many KV pages the cache pool may
hold given the user's budget knobs.

The plan is built at engine startup by :func:`build_pipeline_inference_plan`
from the model's mesh and layer metadata, and is then forwarded to
:class:`ExecutionManager`, :class:`ModelStepExecutor`, and
:class:`PipelineStageRuntime`. The plan is intentionally immutable so
downstream code can rely on ``layer_to_stage`` / ``stage_to_layers``
remaining stable for the lifetime of the engine.
"""

from __future__ import annotations

import dataclasses
import math
import typing as tp

from eformer.loggings import get_logger

from easydel.caching import (
    KDACacheView,
    LightningCacheView,
    MLARaggedPagesCacheView,
    ParallelHybridCacheView,
    RaggedPagesCacheView,
    RecurrentCacheView,
    TransformerCacheView,
    UnifiedAttentionCacheView,
)
from easydel.infra.sharding import is_mpmd_mesh

from ..config import KernelTilePolicy, normalize_kernel_tile_policy

logger = get_logger("eSurge-PipelinePlan")


@dataclasses.dataclass(frozen=True)
class PipelineInferencePlan:
    """Frozen topology + KV-cache plan for one eSurge engine instance.

    Built once by :func:`build_pipeline_inference_plan` and treated as
    read-only thereafter. Disabled plans (``enabled=False``) still carry the
    normalized ``kernel_tile_policy`` so SPMD callers can consult them
    uniformly.

    Attributes:
        enabled: ``True`` iff PP inference is active for this engine. False
            when the model's mesh is SPMD. Most other fields are empty /
            trivial when this is False.
        mpmd_dim: Number of pipeline stages (i.e. ``mpmd_mesh.mpmd_dim``).
            ``1`` on disabled plans.
        final_stage: Rank of the stage that produces the final hidden
            states / logits. Equal to ``layer_to_stage[num_layers - 1]``;
            consulted by :class:`ModelStepExecutor` to place the LM-head
            executable.
        stage_meshes: Tuple of submesh objects, one per rank, in rank order.
            Empty tuple when ``enabled=False``.
        layer_to_stage: ``layer_idx -> stage_rank`` for *every* transformer
            layer (cache-bearing or not). Filled by the model's
            ``_layer_physical_stage_assignment`` when present; otherwise a
            round-robin fallback is used.
        stage_to_layers: Reverse mapping; ``stage_rank ->
            tuple[layer_idx, ...]`` listing all layers on that stage. Empty
            tuples are kept for stages that received no layers.
        cache_layer_to_stage: Subset of ``layer_to_stage`` restricted to
            layer indices whose cache view is one of the recognized KV /
            recurrent cache classes (see
            :func:`_cache_bearing_layer_indices`).
        stage_to_cache_layers: Reverse of ``cache_layer_to_stage``. Used by
            ``max_stage_cache_layers`` to size per-stage cache allocations.
        max_cache_tokens: Optional absolute upper bound on total cached
            tokens across all pages, forwarded from
            :data:`eSurgeCacheRuntimeConfig.max_cache_tokens`. ``None`` lets
            HBM utilization decide.
        cache_capacity_margin: Multiplicative shrink factor in ``(0, 1]``
            applied to the page count by :func:`cap_metadata_pages` to leave
            headroom for non-KV allocations.
        kernel_tile_policy: Normalized Pallas/GDN tile-selection policy
            ultimately passed to
            :func:`set_gdn_kernel_tile_policy` so the inference kernels
            agree on tiling with the plan.
    """

    enabled: bool
    mpmd_dim: int
    final_stage: int
    stage_meshes: tuple[tp.Any, ...]
    layer_to_stage: dict[int, int]
    stage_to_layers: dict[int, tuple[int, ...]]
    cache_layer_to_stage: dict[int, int]
    stage_to_cache_layers: dict[int, tuple[int, ...]]
    max_cache_tokens: int | None
    cache_capacity_margin: float
    kernel_tile_policy: KernelTilePolicy

    @property
    def is_enabled(self) -> bool:
        """Boolean wrapper of :attr:`enabled` for cleaner call-site reads.

        Equivalent to ``bool(plan.enabled)``; provided so callers can write
        ``if plan.is_enabled:`` without having to remember the field name.
        """
        return bool(self.enabled)

    @property
    def max_stage_cache_layers(self) -> int:
        """Maximum cache-layer count across all stages.

        Used by KV-cache allocation (e.g. ``cap_metadata_pages``) to size
        per-stage page tables: every stage gets enough pages for the *worst*
        stage's number of cache-bearing layers. Returns ``0`` when no stage
        owns any cache-bearing layer (typical of disabled plans).
        """
        if not self.stage_to_cache_layers:
            return 0
        return max((len(v) for v in self.stage_to_cache_layers.values()), default=0)


def _cache_bearing_layer_indices(model: tp.Any) -> set[int]:
    """Return the indices of layers whose cache view stores KV state.

    Args:
        model (Any): EasyDeL model exposing ``get_operations_cache_view``.

    Returns:
        set[int]: Layer indices whose mapped cache view is one of the
        recognized KV-cache view classes.
    """
    cache_views = {
        KDACacheView,
        LightningCacheView,
        MLARaggedPagesCacheView,
        ParallelHybridCacheView,
        RaggedPagesCacheView,
        RecurrentCacheView,
        TransformerCacheView,
        UnifiedAttentionCacheView,
    }
    mapping = model.get_operations_cache_view()
    return {int(idx) for idx, view_cls in mapping.items() if view_cls in cache_views}


def build_pipeline_inference_plan(
    *,
    model: tp.Any,
    max_cache_tokens: int | None = None,
    cache_capacity_margin: float = 0.92,
    kernel_tile_policy: KernelTilePolicy | None = "auto",
) -> PipelineInferencePlan:
    """Build a dynamic PP inference plan from the model mesh and layer metadata.

    Inspects the model's mesh to decide whether PP is enabled, walks layer
    indices to assign each to a stage (via the model's
    ``_layer_physical_stage_assignment`` if available, else round-robin),
    and locates cache-bearing layers.

    Args:
        model (Any): Loaded EasyDeL model.
        max_cache_tokens (int | None): Optional global cap on cached tokens.
        cache_capacity_margin (float): Safety margin in ``(0, 1]``.
        kernel_tile_policy (KernelTilePolicy | None): Pallas/GDN tile policy.

    Returns:
        PipelineInferencePlan: A plan with ``is_enabled`` reflecting whether
        PP is active. Disabled plans still carry cache caps and kernel policy
        for downstream consumers.

    Raises:
        ValueError: If ``max_cache_tokens`` is non-positive, or
            ``cache_capacity_margin`` is outside ``(0, 1]``.
    """

    tile_policy = normalize_kernel_tile_policy(kernel_tile_policy)
    mesh = getattr(model, "mesh", None)
    is_mpmd = is_mpmd_mesh(mesh)
    enabled = is_mpmd

    if max_cache_tokens is not None and int(max_cache_tokens) <= 0:
        raise ValueError(f"max_cache_tokens must be positive when provided; got {max_cache_tokens}.")
    if not (0.0 < float(cache_capacity_margin) <= 1.0):
        raise ValueError(f"cache_capacity_margin must be in (0, 1]; got {cache_capacity_margin}.")

    if not enabled:
        return PipelineInferencePlan(
            enabled=False,
            mpmd_dim=1,
            final_stage=0,
            stage_meshes=(),
            layer_to_stage={},
            stage_to_layers={},
            cache_layer_to_stage={},
            stage_to_cache_layers={},
            max_cache_tokens=None if max_cache_tokens is None else int(max_cache_tokens),
            cache_capacity_margin=float(cache_capacity_margin),
            kernel_tile_policy=tile_policy,
        )

    mpmd_mesh = mesh.mpmd_mesh
    mpmd_dim = int(mpmd_mesh.mpmd_dim)
    stage_meshes = tuple(mpmd_mesh.submesh(rank) for rank in range(mpmd_dim))
    text_config = model.config.get_text_config()
    cache_view_mapping = model.get_operations_cache_view()
    total_layers = int(getattr(text_config, "num_hidden_layers", max(cache_view_mapping.keys(), default=-1) + 1))

    layer_to_stage: dict[int, int] = {}
    stage_to_layers: dict[int, list[int]] = {rank: [] for rank in range(mpmd_dim)}
    for layer_idx in range(total_layers):
        if hasattr(model, "_layer_physical_stage_assignment"):
            rank, _ = model._layer_physical_stage_assignment(layer_idx, total_layers)
        else:
            rank = min(mpmd_dim - 1, (layer_idx * mpmd_dim) // max(1, total_layers))
        rank = int(rank)
        layer_to_stage[layer_idx] = rank
        stage_to_layers.setdefault(rank, []).append(layer_idx)

    cache_layers = _cache_bearing_layer_indices(model)
    cache_layer_to_stage = {idx: layer_to_stage[idx] for idx in cache_layers if idx in layer_to_stage}
    stage_to_cache_layers: dict[int, list[int]] = {rank: [] for rank in range(mpmd_dim)}
    for idx, rank in cache_layer_to_stage.items():
        stage_to_cache_layers.setdefault(rank, []).append(idx)

    final_stage = layer_to_stage.get(max(total_layers - 1, 0), mpmd_dim - 1)
    plan = PipelineInferencePlan(
        enabled=True,
        mpmd_dim=mpmd_dim,
        final_stage=int(final_stage),
        stage_meshes=stage_meshes,
        layer_to_stage=layer_to_stage,
        stage_to_layers={rank: tuple(vals) for rank, vals in stage_to_layers.items()},
        cache_layer_to_stage=cache_layer_to_stage,
        stage_to_cache_layers={rank: tuple(vals) for rank, vals in stage_to_cache_layers.items()},
        max_cache_tokens=None if max_cache_tokens is None else int(max_cache_tokens),
        cache_capacity_margin=float(cache_capacity_margin),
        kernel_tile_policy=tile_policy,
    )
    logger.info(
        "Enabled PP inference plan: mpmd_dim=%s final_stage=%s max_stage_cache_layers=%s cache_layers=%s",
        plan.mpmd_dim,
        plan.final_stage,
        plan.max_stage_cache_layers,
        {rank: len(layers) for rank, layers in plan.stage_to_cache_layers.items()},
    )
    return plan


def cap_metadata_pages(metadata: tp.Any, plan: PipelineInferencePlan | None) -> tp.Any:
    """Apply user/runtime cache token caps to cache metadata in-place.

    Args:
        metadata (Any): Cache-metadata object exposing ``num_pages`` and
            ``page_size`` (and optionally ``_mixed_layer_configs``).
        plan (PipelineInferencePlan | None): Plan whose
            ``max_cache_tokens`` and ``cache_capacity_margin`` drive the cap.
            A ``None`` or disabled plan is a no-op.

    Returns:
        Any: The same ``metadata`` object (mutated in place).
    """

    if plan is None or not plan.is_enabled or not hasattr(metadata, "num_pages") or not hasattr(metadata, "page_size"):
        return metadata

    num_pages = int(metadata.num_pages)
    if plan.max_cache_tokens is not None:
        token_pages = max(1, math.ceil(int(plan.max_cache_tokens) / max(1, int(metadata.page_size))))
        num_pages = min(num_pages, int(token_pages))
    num_pages = max(1, int(num_pages * float(plan.cache_capacity_margin)))
    set_metadata_num_pages(metadata, num_pages)
    return metadata


def set_metadata_num_pages(metadata: tp.Any, num_pages: int) -> None:
    """Update ``num_pages`` on representative and mixed per-layer metadata.

    Args:
        metadata (Any): Metadata object whose ``num_pages`` (and any nested
            ``_mixed_layer_configs[*].num_pages``) are overwritten.
        num_pages (int): New page count (lower-bounded to 1).
    """

    if not hasattr(metadata, "num_pages"):
        return
    metadata.num_pages = max(1, int(num_pages))
    mixed = getattr(metadata, "_mixed_layer_configs", None)
    if isinstance(mixed, dict):
        for cfg in mixed.values():
            if hasattr(cfg, "num_pages"):
                cfg.num_pages = max(1, int(num_pages))


def metadata_num_pages(metadata: tp.Any) -> int | None:
    """Read ``num_pages`` off a cache-metadata object if present.

    Args:
        metadata (Any): Cache metadata object.

    Returns:
        int | None: ``num_pages`` cast to ``int``, or ``None`` if the
        attribute is absent.
    """
    if not hasattr(metadata, "num_pages"):
        return None
    return int(metadata.num_pages)
