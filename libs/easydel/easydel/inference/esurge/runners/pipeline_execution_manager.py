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

"""Pipeline-parallel-aware execution manager for eSurge.

Holds the single subclass of :class:`ExecutionManager` selected when the
runner detects an enabled :class:`PipelineInferencePlan`. The subclass
exists purely to enforce a hard invariant — *if PP is on, the plan must
travel with the manager* — at construction time. All actual pipeline
mechanics (per-stage compile, queue-backed dispatch, final-stage sampler
placement) live in :class:`ExecutionManager` and
:class:`ModelStepExecutor`; this file is a thin guard.
"""

from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger

from .execution_manager import ExecutionManager
from .pipeline_plan import PipelineInferencePlan

logger = get_logger("eSurge-PipelineExecutionManager")


class PipelineExecutionManager(ExecutionManager):
    """:class:`ExecutionManager` specialization gated on an enabled PP plan.

    Selected by :class:`eSurgeRunner` when
    :func:`build_pipeline_inference_plan` returns an enabled plan. The class
    contributes no new behaviour beyond the base manager — the actual PP
    work is done by :class:`ModelStepExecutor` (per-stage backbone compile),
    :class:`PipelineStageRuntime` (resident worker dispatch), and the
    sampler-placement logic in :class:`SamplerExecutor`. The subclass
    exists so that:

    1. Construction fails loudly when an instance is created without a
       valid plan, instead of silently falling back to SPMD execution.
    2. The selection branch in the runner reduces to ``manager_cls = ...``
       and stays a one-line decision.
    3. Logging at construction surfaces the resolved topology
       (``mpmd_dim`` / ``final_stage`` / ``max_stage_cache_layers``) so
       startup logs make the active PP shape easy to verify.
    """

    def __init__(self, *args: tp.Any, pipeline_plan: PipelineInferencePlan | None = None, **kwargs: tp.Any) -> None:
        """Validate the plan, log topology, and delegate to :class:`ExecutionManager`.

        Args:
            *args: Positional arguments forwarded to
                :meth:`ExecutionManager.__init__` unchanged.
            pipeline_plan (PipelineInferencePlan | None): Enabled pipeline
                plan describing stage meshes, layer assignment, and cache
                caps. Forwarded as the ``pipeline_plan=`` keyword to the
                base manager so :class:`ModelStepExecutor` can pick it up
                from there.
            **kwargs: Remaining keyword arguments forwarded to the base
                manager (``mpmd_scheduler``, ``metadata``, ``model``, …).

        Raises:
            ValueError: When ``pipeline_plan`` is ``None`` or its
                ``is_enabled`` flag is ``False``. The whole point of this
                subclass is the PP guarantee, so we refuse to construct
                without one.
        """
        if pipeline_plan is None or not pipeline_plan.is_enabled:
            raise ValueError("PipelineExecutionManager requires an enabled PipelineInferencePlan.")
        logger.info(
            "Using true PP eSurge execution path: mpmd_dim=%s final_stage=%s max_stage_cache_layers=%s",
            pipeline_plan.mpmd_dim,
            pipeline_plan.final_stage,
            pipeline_plan.max_stage_cache_layers,
        )
        super().__init__(*args, pipeline_plan=pipeline_plan, **kwargs)
