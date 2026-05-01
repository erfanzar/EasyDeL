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

"""Dynamic planning utilities for eSurge pipeline-parallel inference."""

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

from ..config import (
    KernelTilePolicy,
    PipelineInferenceMode,
    normalize_kernel_tile_policy,
    normalize_pipeline_inference_mode,
)

logger = get_logger("eSurge-PipelinePlan")


@dataclasses.dataclass(frozen=True)
class PipelineInferencePlan:
    """Topology and cache-capacity plan for true PP eSurge inference."""

    enabled: bool
    mode: PipelineInferenceMode
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
        return bool(self.enabled)

    @property
    def max_stage_cache_layers(self) -> int:
        if not self.stage_to_cache_layers:
            return 0
        return max((len(v) for v in self.stage_to_cache_layers.values()), default=0)


def _cache_bearing_layer_indices(model: tp.Any) -> set[int]:
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
    mpmd_scheduler: tp.Any,
    pipeline_inference: PipelineInferenceMode | None = "auto",
    max_cache_tokens: int | None = None,
    cache_capacity_margin: float = 0.92,
    kernel_tile_policy: KernelTilePolicy | None = "auto",
) -> PipelineInferencePlan:
    """Build a dynamic PP inference plan from the model mesh and layer metadata."""

    mode = normalize_pipeline_inference_mode(pipeline_inference)
    tile_policy = normalize_kernel_tile_policy(kernel_tile_policy)
    mesh = getattr(model, "mesh", None)
    is_mpmd = is_mpmd_mesh(mesh)
    del mpmd_scheduler
    enabled = mode == "on" or (mode == "auto" and is_mpmd)
    if mode == "on" and not is_mpmd:
        raise ValueError("pipeline_inference='on' requires a SpectraX MPMD mesh.")

    if max_cache_tokens is not None and int(max_cache_tokens) <= 0:
        raise ValueError(f"max_cache_tokens must be positive when provided; got {max_cache_tokens}.")
    if not (0.0 < float(cache_capacity_margin) <= 1.0):
        raise ValueError(f"cache_capacity_margin must be in (0, 1]; got {cache_capacity_margin}.")

    if not enabled:
        return PipelineInferencePlan(
            enabled=False,
            mode=mode,
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
        mode=mode,
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
    """Apply user/runtime cache token caps to cache metadata in-place."""

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
    """Update ``num_pages`` on representative and mixed per-layer metadata."""

    if not hasattr(metadata, "num_pages"):
        return
    metadata.num_pages = max(1, int(num_pages))
    mixed = getattr(metadata, "_mixed_layer_configs", None)
    if isinstance(mixed, dict):
        for cfg in mixed.values():
            if hasattr(cfg, "num_pages"):
                cfg.num_pages = max(1, int(num_pages))


def metadata_num_pages(metadata: tp.Any) -> int | None:
    if not hasattr(metadata, "num_pages"):
        return None
    return int(metadata.num_pages)
