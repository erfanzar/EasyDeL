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

"""Runner execution-manager lifecycle and pipeline-plan validation.

:class:`RunnerExecutionManager` exposes idempotent cache allocation to the
runner while retaining the base manager's bounded allocation retries.
:class:`PipelineExecutionManager` adds the construction-time invariant that
an enabled pipeline plan must travel with a PP manager. Pipeline mechanics
remain in :class:`ExecutionManager` and :class:`ModelStepExecutor`.
"""

from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger

from .execution_manager import ExecutionManager
from .pipeline_plan import PipelineInferencePlan

logger = get_logger("eSurge-PipelineExecutionManager")


class RunnerExecutionManager(ExecutionManager):
    """Execution manager with a public, idempotent cache-allocation boundary.

    Both SPMD and PP runners use this lifecycle API. The inherited allocation
    implementation remains the single owner of cache layout, speculative
    scratch sizing and bounded PP HBM-OOM shrink/retry behavior.
    """

    def execute_model(
        self, num_tokens: int, padded_num_reqs: int, inputs: tp.Any, use_pipeline_runtime: bool = True
    ) -> tp.Any:
        """Retain the actual model request bucket, including PP prefix compaction."""
        output = super().execute_model(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            inputs=inputs,
            use_pipeline_runtime=use_pipeline_runtime,
        )
        self._last_model_request_bucket = int(padded_num_reqs)
        return output

    def hidden_state_layout(
        self, hidden_states: tp.Any, *, num_tokens: int, padded_num_reqs: int
    ) -> tp.Literal["tokens", "requests"]:
        """Identify the executed bucket's token-major or gathered hidden output.

        Active token counts cannot discriminate these layouts: a compact output
        can contain more padded request rows than the replay has active tokens.
        Use the actual token/request buckets, and the executor's compiled-output
        policy when those bucket sizes coincide. No device data is read.

        Args:
            hidden_states: Hidden output returned by this manager's execution.
            num_tokens: Static token bucket used by that execution.
            padded_num_reqs: Static model request bucket used by that execution.

        Returns:
            ``"tokens"`` for full backbone rows, ``"requests"`` for last-token
            rows gathered by request.

        Raises:
            ValueError: If neither output layout matches the executed buckets.
        """
        hidden_len = int(hidden_states.shape[0])
        executor = self._model_executor
        if executor._uses_mpmd_mesh():
            # PP may narrow a prefix-contiguous request bucket. Its fused step
            # returns gathered rows; the stitched backbone path returns full rows.
            model_request_bucket = int(getattr(self, "_last_model_request_bucket", padded_num_reqs))
            fused_key = (int(num_tokens), model_request_bucket, "model_step", executor._compile_mode())
            if hidden_len == model_request_bucket and fused_key in executor._cache:
                return "requests"
            if hidden_len == int(num_tokens):
                return "tokens"
        else:
            token_rows = hidden_len == int(num_tokens)
            request_rows = hidden_len == int(padded_num_reqs)
            if token_rows != request_rows:
                return "tokens" if token_rows else "requests"
            if token_rows:
                return "tokens" if 0 < int(num_tokens) <= self.full_hidden_state_max_tokens else "requests"
        raise ValueError("Hidden output does not match the executed token or request bucket.")

    def initialize_kv_cache(self, *, quantizer: tp.Any, masking_details: tp.Any = None) -> tp.Any:
        """Return existing cache pages or allocate them through bounded retries.

        Args:
            quantizer: Quantizer built from the live model's KV-cache config.
                Passed unchanged to the inherited allocation implementation.
            masking_details: Optional live model mask details.

        Returns:
            Existing or newly allocated operations cache.

        Raises:
            Exception: Propagates allocation failures after the inherited retry
                policy is exhausted, leaving ``kv_pages`` unset for a later retry.
        """
        if self.kv_pages is None:
            self.kv_pages = self._init_operations_cache_with_retry(quantizer=quantizer, masking_details=masking_details)
        return self.kv_pages


class PipelineExecutionManager(RunnerExecutionManager):
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
