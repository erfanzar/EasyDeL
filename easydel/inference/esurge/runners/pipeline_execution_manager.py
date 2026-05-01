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

"""Pipeline-parallel eSurge execution manager."""

from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger

from .execution_manager import ExecutionManager
from .pipeline_plan import PipelineInferencePlan

logger = get_logger("eSurge-PipelineExecutionManager")


class PipelineExecutionManager(ExecutionManager):
    """Execution manager for SpectraX-backed PP inference.

    The backbone is still compiled through ``spx.jit`` in ``ModelStepExecutor``;
    this manager is the eSurge-level PP path that guarantees a dynamic topology
    plan is present, sampler state is final-stage placed, and cache allocation
    uses the PP-aware bounded retry path from ``ExecutionManager``.
    """

    def __init__(self, *args: tp.Any, pipeline_plan: PipelineInferencePlan | None = None, **kwargs: tp.Any) -> None:
        if pipeline_plan is None or not pipeline_plan.is_enabled:
            raise ValueError("PipelineExecutionManager requires an enabled PipelineInferencePlan.")
        logger.info(
            "Using true PP eSurge execution path: mpmd_dim=%s final_stage=%s max_stage_cache_layers=%s",
            pipeline_plan.mpmd_dim,
            pipeline_plan.final_stage,
            pipeline_plan.max_stage_cache_layers,
        )
        super().__init__(*args, pipeline_plan=pipeline_plan, **kwargs)
