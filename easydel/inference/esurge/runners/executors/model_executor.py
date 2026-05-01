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

"""Model-step compilation/execution for eSurge.

This module isolates the model forward pass (KV-cache update + logits/hidden-states)
from the rest of the execution manager. The ModelStepExecutor is responsible for
building and caching compiled variants of the model step for different token/batch
bucket combinations.

The model step performs:
    1. Reconstruct the full model from split graph components
    2. Prepare cache metadata for paged attention
    3. Execute the transformer forward pass with paged KV cache
    4. Extract hidden states and compute logits for sampling positions
    5. Return updated KV cache and outputs

This separation from the execution manager allows:
    - Independent optimization of the model forward pass
    - Cleaner caching strategy with well-defined keys
    - Easier testing and profiling of model execution
    - Support for different model architectures via the graphdef

Classes:
    ModelStepExecutor: Manages compilation, caching, and execution of the
        model forward pass function.

Example:
    >>> executor = ModelStepExecutor(
    ...     model=model,
    ...     mesh=mesh,
    ...     metadata=cache_config,
    ...     kv_pages_template=kv_cache,
    ...     graphstate_template=graphstate,
    ...     graphother_template=graphother,
    ...     max_num_reqs=32,
    ...     graphdef=graphdef,
    ...     empty_sharding=sharding,
    ...     use_aot_forward=True,
    ... )
    >>>
    >>> # Compile for expected configurations
    >>> executor.compile(
    ...     num_tokens=256,
    ...     padded_num_reqs=16,
    ...     graphdef=graphdef,
    ...     graphstate=graphstate,
    ...     graphother=graphother,
    ...     inputs=dummy_inputs,
    ... )
    >>>
    >>> # Execute model step
    >>> model_fn = executor.get_compiled(num_tokens=256, padded_num_reqs=16)
    >>> outputs = model_fn(graphstate, graphother, kv_pages, batch_metadata)
"""

from __future__ import annotations

import typing as tp
from collections import OrderedDict

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.caching import (
    HybridCache,
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RaggedPagesMetadata,
    UnifiedAttentionCache,
    UnifiedAttentionCacheConfig,
)
from easydel.infra.sharding import MeshLike, replicate_on_array_mesh, resolve_stage_mesh
from easydel.utils import set_inference_mode

from ..execution_types import BackboneOutputs, BatchMetadata, ModelStepOutputs, StepFunctionInputs
from ..pipeline_plan import PipelineInferencePlan
from ..pipeline_runtime import PipelineStageRuntime

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule
    from easydel.infra.etils import MpMdSchedulers


class ModelStepExecutor:
    """Compile, cache, and execute the model forward step.

    The ModelStepExecutor manages compilation and execution of the model
    forward pass, which computes hidden states and logits while updating
    the paged KV cache. It retains compiled function variants for different
    input dimensions until the cache is explicitly cleared.

    The executor separates graph definition (static model structure) from
    graph state (weights) and graph other (auxiliary data), allowing weight
    updates without recompilation.

    Attributes:
        model (EasyDeLBaseModule): The EasyDeL model instance.
        mesh (Any): JAX sharding mesh for distributed execution.
        metadata (RaggedPagesCacheConfig | UnifiedAttentionCacheConfig):
            KV cache configuration.
        max_num_reqs (int): Maximum number of concurrent requests.
        graphdef (Any): Model graph definition (static structure).
        use_aot_forward (bool): Whether to use AOT compilation.
        bind_graphstate_for_aot (bool): Whether AOT-compiled model steps
            close over graphstate/graphother as compile-time constants.

    Example:
        >>> executor = ModelStepExecutor(
        ...     model=model,
        ...     mesh=mesh,
        ...     metadata=cache_config,
        ...     kv_pages_template=kv_cache,
        ...     graphstate_template=graphstate,
        ...     graphother_template=graphother,
        ...     max_num_reqs=32,
        ...     graphdef=graphdef,
        ...     empty_sharding=sharding,
        ...     use_aot_forward=True,
        ... )
        >>> outputs = executor.get_compiled(num_tokens=256, padded_num_reqs=16)(
        ...     graphstate, graphother, kv_pages, batch_metadata
        ... )
    """

    def __init__(
        self,
        *,
        model: "EasyDeLBaseModule",
        mesh: MeshLike,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
        max_num_reqs: int,
        graphdef: tp.Any,
        empty_sharding: jax.sharding.Sharding,
        use_aot_forward: bool,
        bind_graphstate_for_aot: bool = False,
        cache_capacity: int = 64,
        mpmd_scheduler: MpMdSchedulers | None = None,
        pipeline_plan: PipelineInferencePlan | None = None,
    ) -> None:
        """Initialize the ModelStepExecutor.

        Args:
            model: The EasyDeL model instance.
            mesh: JAX sharding mesh for distributed execution.
            metadata: KV cache configuration (ragged pages or unified attention).
            kv_pages_template: Template KV cache for shape/sharding inference.
            graphstate_template: Template graph state for sharding inference.
            graphother_template: Template graph other for sharding inference.
            max_num_reqs: Maximum number of concurrent requests.
            graphdef: Model graph definition (static structure).
            empty_sharding: Default JAX sharding for replicated arrays.
            use_aot_forward: If True, use AOT compilation via lower().compile().
                If False, use JIT compilation on first call.
            bind_graphstate_for_aot: When True in AOT mode, compile model-step
                variants with graphstate/graphother closed over as constants
                (runtime call signature is preserved). This enables weight-
                concrete kernel policies (e.g. TPU predecode-once). Default: False.
            cache_capacity: Deprecated compatibility argument. Compiled
                variants are retained until ``clear_cache()`` is called.
            mpmd_scheduler: Optional ``spectrax.runtime.schedules.Schedule``
                forwarded to ``spx.jit(schedule=...)`` when the mesh is MPMD.
                ``None`` ⇒ forward-only marker-cluster MPMD path. Ignored on SPMD.
        """
        self.model = model
        self.mesh = mesh
        self.metadata = metadata
        self.max_num_reqs = int(max_num_reqs)
        self.graphdef = graphdef
        self._metadata_version = metadata.version
        self._use_slot_mapping = self._metadata_version == "v2"
        self._empty_sharding = empty_sharding
        self.use_aot_forward = bool(use_aot_forward)
        self.bind_graphstate_for_aot = bool(bind_graphstate_for_aot)
        self.mpmd_scheduler = mpmd_scheduler
        self.pipeline_plan = pipeline_plan
        self._pipeline_runtime = (
            PipelineStageRuntime(plan=pipeline_plan) if pipeline_plan is not None and pipeline_plan.is_enabled else None
        )
        self._lm_head_uses_tied_projection = self._uses_tied_lm_head() if self._uses_mpmd_mesh() else False
        del cache_capacity

        self._backbone_fn = self._build_backbone_fn(
            kv_pages_template=kv_pages_template,
            graphstate_template=graphstate_template,
            graphother_template=graphother_template,
        )
        self._lm_head_fn = self._build_lm_head_fn(
            graphstate_template=graphstate_template,
            graphother_template=graphother_template,
        )
        # Backbone cache: keyed by (num_tokens, "backbone", mode)
        self._backbone_cache: OrderedDict[tuple[int, str, str], tp.Any] = OrderedDict()
        # LM-head cache: keyed by (padded_num_reqs, "lm_head", mode)
        self._lm_head_cache: OrderedDict[tuple[int, str, str], tp.Any] = OrderedDict()
        # Legacy combined cache (kept for backward compat with tests)
        self._cache: OrderedDict[tuple[int, int, str, str], tp.Any] = OrderedDict()
        self._lm_head_graphdef: tp.Any | None = None
        self._lm_head_state: tp.Any | None = None
        self._lm_head_clip_cap: float | None = None
        self._lm_head_logit_scale: float | None = None
        self._lm_head_soft_cap: float | None = None
        if self._uses_mpmd_mesh():
            self.refresh_lm_head_state(
                graphdef=graphdef,
                graphstate=graphstate_template,
                graphother=graphother_template,
            )

    def clear_cache(self) -> None:
        """Clear all cached compiled functions."""
        self._backbone_cache.clear()
        self._lm_head_cache.clear()
        self._cache.clear()

    def shutdown(self) -> None:
        """Release resident PP stage workers."""
        if self._pipeline_runtime is not None:
            self._pipeline_runtime.shutdown()

    @staticmethod
    def _cache_store(cache: OrderedDict, key, value) -> None:
        cache[key] = value
        cache.move_to_end(key)

    @staticmethod
    def _cache_lookup(cache: OrderedDict, key):
        value = cache[key]
        cache.move_to_end(key)
        return value

    def _uses_mpmd_mesh(self) -> bool:
        return self.mesh.is_mpmd

    def _final_stage_mesh(self):
        if not self._uses_mpmd_mesh():
            return None
        if self.pipeline_plan is not None and self.pipeline_plan.is_enabled and self.pipeline_plan.stage_meshes:
            return self.pipeline_plan.stage_meshes[int(self.pipeline_plan.final_stage)]
        text_config = self.model.config.get_text_config()
        total_layers = int(getattr(text_config, "num_hidden_layers", 1))
        if hasattr(self.model, "_layer_physical_stage_assignment"):
            stage = self.model._layer_physical_stage_assignment(total_layers - 1, total_layers)
        else:
            mpmd_dim = int(getattr(self.mesh, "mpmd_dim", 1))
            stage = (mpmd_dim - 1, mpmd_dim)
        return resolve_stage_mesh(self.mesh, stage=stage)

    def _embedding_stage_mesh(self):
        if not self._uses_mpmd_mesh():
            return None
        if self.pipeline_plan is not None and self.pipeline_plan.is_enabled and self.pipeline_plan.stage_meshes:
            return self.pipeline_plan.stage_meshes[0]
        mpmd_dim = int(getattr(self.mesh, "mpmd_dim", 1))
        return resolve_stage_mesh(self.mesh, stage=(0, mpmd_dim))

    def _lm_head_stage_mesh(self):
        if self._lm_head_uses_tied_projection:
            return self._embedding_stage_mesh()
        return self._final_stage_mesh()

    def _lm_head_replicated_sharding(self) -> NamedSharding | None:
        stage_mesh = self._lm_head_stage_mesh()
        if stage_mesh is None:
            return None
        return NamedSharding(stage_mesh, PartitionSpec())

    def _uses_tied_lm_head(self) -> bool:
        configs = [self.model.config]
        get_text_config = getattr(self.model.config, "get_text_config", None)
        if callable(get_text_config):
            text_config = get_text_config()
            if text_config is not self.model.config:
                configs.append(text_config)

        for config in configs:
            for key in ("tie_word_embeddings", "share_input_output_layers"):
                if hasattr(config, key) and bool(getattr(config, key)):
                    return True
        return False

    def _lm_head_hidden_sharding(self, shape: tuple[int, ...]) -> NamedSharding | None:
        stage_mesh = self._lm_head_stage_mesh()
        if stage_mesh is None:
            return None
        resolver = self.model.config.runtime_sharding_resolver.with_mesh(stage_mesh)
        spec = resolver.resolve(
            axes=[spx.common_types.EMPTY, spx.common_types.EMBED],
            mode=spx.common_types.MODE_DECODE,
            shape=shape,
        )
        return NamedSharding(stage_mesh, spec)

    def _place_lm_head_hidden(self, hidden_states: jax.Array) -> jax.Array:
        if not self._uses_mpmd_mesh():
            return hidden_states
        sharding = self._lm_head_hidden_sharding(tuple(hidden_states.shape))
        if sharding is None:
            return hidden_states
        current = getattr(hidden_states, "sharding", None)
        if (
            isinstance(current, NamedSharding)
            and getattr(current, "mesh", None) == sharding.mesh
            and getattr(current, "spec", None) == sharding.spec
        ):
            return hidden_states
        return jax.device_put(hidden_states, sharding)

    def _stage_local_lm_head_state(self, state: tp.Any) -> tp.Any:
        if not self._uses_mpmd_mesh():
            return state
        stage_mesh = self._lm_head_stage_mesh()
        if stage_mesh is None:
            return state
        shardings = spx.extract_sharding_structure(state, mesh=self.mesh, stage_mesh=stage_mesh)
        return jax.tree_util.tree_map(
            lambda x, s: jax.device_put(x, s) if s is not None and hasattr(x, "dtype") else x,
            state,
            shardings,
        )

    def _refresh_lm_head_postprocess(self, model: "EasyDeLBaseModule") -> None:
        clip_feature = getattr(model, "_logit_cap_feature", None)
        self._lm_head_clip_cap = getattr(clip_feature, "cap_value", None)

        scale = None
        for attr in ("logit_scale", "output_multiplier_scale"):
            value = getattr(model, attr, None)
            if value is not None:
                scale = float(value) if scale is None else scale * float(value)
        base_model = getattr(model, "base_model", None)
        multiplier = getattr(base_model, "lm_head_multiplier", None)
        if multiplier is not None:
            scale = float(multiplier) if scale is None else scale * float(multiplier)
        self._lm_head_logit_scale = scale

        configs = [getattr(model, "config", None)]
        config = configs[0]
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            configs.append(get_text_config())
        elif hasattr(config, "text_config"):
            configs.append(config.text_config)
        self._lm_head_soft_cap = next(
            (
                float(cap)
                for cfg in configs
                if cfg is not None
                for cap in (getattr(cfg, "final_logit_softcapping", None),)
                if cap is not None
            ),
            None,
        )

    def refresh_lm_head_state(self, *, graphdef: tp.Any, graphstate: tp.Any, graphother: tp.Any) -> None:
        """Refresh the stage-local LM-head state without touching backbone weights.

        PP inference runs the transformer backbone through ``spx.jit`` and then
        projects only sampled rows on the final stage.  The projection must not
        close over a full bound model: that makes the head executable compile
        too much state and can leave it stale after graph updates.
        """
        if not self._uses_mpmd_mesh():
            return
        self._lm_head_uses_tied_projection = self._uses_tied_lm_head()
        stage_mesh = self._lm_head_stage_mesh()
        mesh_context = stage_mesh if stage_mesh is not None else self.model.mesh
        with mesh_context:
            model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=False))
            self._refresh_lm_head_postprocess(model)
            if self._lm_head_uses_tied_projection:
                projection = model.get_embedding()
            else:
                projection = model.get_lm_head()
            lm_head_graphdef, lm_head_state = spx.export(projection)
        self._lm_head_graphdef = lm_head_graphdef
        self._lm_head_state = self._stage_local_lm_head_state(lm_head_state)

    def _compile_mode(self) -> str:
        if self._uses_mpmd_mesh():
            return "mpmd"
        return "aot" if self.use_aot_forward else "jit"

    def _cache_put(self, key: tuple[int, int, str, str], value: tp.Any) -> None:
        self._cache_store(self._cache, key, value)

    def _cache_get(self, key: tuple[int, int, str, str]) -> tp.Any:
        return self._cache_lookup(self._cache, key)

    def cache_keys(self) -> list:
        """Get all keys currently in backbone + lm_head caches."""
        return list(self._backbone_cache.keys()) + list(self._lm_head_cache.keys())

    def has(self, key: tuple[int, int, str, str]) -> bool:
        """Check if a (num_tokens, padded_num_reqs) pair is fully compiled."""
        mode = key[3] if len(key) == 4 else self._compile_mode()
        backbone_key = (key[0], "backbone", mode)
        lm_head_key = (key[1], "lm_head", mode)
        return backbone_key in self._backbone_cache and lm_head_key in self._lm_head_cache

    def has_backbone(self, num_tokens: int) -> bool:
        mode = self._compile_mode()
        return (int(num_tokens), "backbone", mode) in self._backbone_cache

    def has_lm_head(self, padded_num_reqs: int) -> bool:
        mode = self._compile_mode()
        return (int(padded_num_reqs), "lm_head", mode) in self._lm_head_cache

    def get_compiled(self, *, num_tokens: int, padded_num_reqs: int) -> tp.Any:
        """Retrieve pre-compiled backbone + lm_head as a combined callable.

        Returns a cached wrapper that calls backbone then lm_head, producing
        a ``ModelStepOutputs`` — same interface as before the split.
        """
        mode = self._compile_mode()
        backbone_fn = self._cache_lookup(self._backbone_cache, (int(num_tokens), "backbone", mode))
        lm_head_fn = self._cache_lookup(self._lm_head_cache, (int(padded_num_reqs), "lm_head", mode))
        _pnr = int(padded_num_reqs)

        def _combined(graphstate_, graphother_, kv_pages_, metadata_):
            backbone_out = backbone_fn(graphstate_, graphother_, kv_pages_, metadata_)
            # Gather outside the compiled lm_head so it always sees
            # [padded_num_reqs, hidden_dim] regardless of num_tokens.
            logits_indices = metadata_.logits_indices[:_pnr]
            if self._uses_mpmd_mesh():
                logits_indices = replicate_on_array_mesh(logits_indices, backbone_out.hidden_states)
            gathered_hs = backbone_out.hidden_states[logits_indices]
            gathered_hs = self._place_lm_head_hidden(gathered_hs)
            logits = lm_head_fn(graphstate_, graphother_, gathered_hs)
            return ModelStepOutputs(
                kv_pages=backbone_out.kv_pages,
                hidden_states=backbone_out.hidden_states,
                logits=logits,
            )

        return _combined

    def get_backbone(self, *, num_tokens: int) -> tp.Any:
        """Retrieve a pre-compiled backbone function."""
        mode = self._compile_mode()
        return self._cache_lookup(self._backbone_cache, (int(num_tokens), "backbone", mode))

    def get_lm_head(self, *, padded_num_reqs: int) -> tp.Any:
        """Retrieve a pre-compiled lm_head function."""
        mode = self._compile_mode()
        return self._cache_lookup(self._lm_head_cache, (int(padded_num_reqs), "lm_head", mode))

    def compile(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        graphdef: tp.Any,
        graphstate: tp.Any,
        graphother: tp.Any,
        inputs: StepFunctionInputs,
    ) -> ModelStepOutputs | None:
        """Compile backbone and lm_head for the given dimensions.

        The backbone is keyed by ``num_tokens`` only; the lm_head by
        ``padded_num_reqs`` only.  Each is skipped if already cached.
        """
        self.graphdef = graphdef
        backbone_out = self.compile_backbone(
            num_tokens=num_tokens,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )
        self.compile_lm_head(
            padded_num_reqs=padded_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )
        return backbone_out

    def compile_backbone(
        self,
        *,
        num_tokens: int,
        graphdef: tp.Any,
        graphstate: tp.Any,
        graphother: tp.Any,
        inputs: StepFunctionInputs,
    ) -> BackboneOutputs | None:
        """Compile the backbone (transformer forward) for a token bucket.

        Keyed by ``num_tokens`` only — independent of ``padded_num_reqs``.
        """
        self.graphdef = graphdef
        mode = self._compile_mode()
        key = (int(num_tokens), "backbone", mode)
        if key in self._backbone_cache:
            return None

        if self._uses_mpmd_mesh():

            def wrapped_backbone(graphstate_, graphother_, kv_pages_, metadata_):
                return self._dispatch_pipeline_backbone(graphstate_, graphother_, kv_pages_, metadata_)

            out = wrapped_backbone(graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
            self._cache_store(self._backbone_cache, key, wrapped_backbone)
            return out

        if self.use_aot_forward:
            if self.bind_graphstate_for_aot:

                def _bound_backbone(kv_pages_, metadata_):
                    return self._backbone_fn(graphdef, graphstate, graphother, kv_pages_, metadata_)

                compiled_bound = spx.jit(_bound_backbone).lower(inputs.kv_pages, inputs.batch_metadata).compile()

                def _wrapped_bound_backbone(graphstate_, graphother_, kv_pages_, metadata_):
                    del graphstate_, graphother_
                    return compiled_bound(kv_pages_, metadata_)

                self._cache_store(self._backbone_cache, key, _wrapped_bound_backbone)
            else:
                compiled = self._backbone_fn.lower(  # pyright: ignore[reportFunctionMemberAccess]
                    *(graphdef, graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
                ).compile()
                self._cache_store(self._backbone_cache, key, compiled)
            return None

        def wrapped_backbone(graphstate_, graphother_, kv_pages_, metadata_):
            return self._backbone_fn(self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)

        out = wrapped_backbone(graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
        self._cache_store(self._backbone_cache, key, wrapped_backbone)
        return out

    def _dispatch_pipeline_backbone(self, graphstate_, graphother_, kv_pages_, metadata_) -> BackboneOutputs:
        pipeline_runtime = getattr(self, "_pipeline_runtime", None)
        if pipeline_runtime is None:
            return self._backbone_fn(self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)
        return pipeline_runtime.dispatch(
            self._backbone_fn,
            self.graphdef,
            graphstate_,
            graphother_,
            kv_pages_,
            metadata_,
        )

    def last_pipeline_stats(self) -> dict[str, float | int]:
        if self._pipeline_runtime is None:
            state = getattr(self._backbone_fn, "_mpmd_state", {})
            launches = int(state.get("forward_stage_launches", 0) or 0)
            if launches <= 0:
                return {}
            return {"pp_stage_launches": launches}
        stats = self._pipeline_runtime.last_stats
        return {
            "pp_stage_launches": int(stats.stage_launches),
            "pp_stage_dispatch_time": float(stats.stage_dispatch_time),
            "pp_queue_wait_time": float(stats.queue_wait_time),
        }

    def compile_lm_head(
        self,
        *,
        padded_num_reqs: int,
        graphdef: tp.Any,
        graphstate: tp.Any,
        graphother: tp.Any,
        inputs: StepFunctionInputs,
        hidden_dim: int | None = None,
        dtype: tp.Any = None,
    ) -> None:
        """Compile the lm_head for a request bucket.

        Keyed by ``padded_num_reqs`` only — independent of ``num_tokens``.
        """
        self.graphdef = graphdef
        mode = self._compile_mode()
        key = (int(padded_num_reqs), "lm_head", mode)
        if key in self._lm_head_cache:
            return

        if hidden_dim is None:
            hidden_dim = int(self.model.config.get_text_config().hidden_size)
        if dtype is None:
            dtype = self.model.dtype
        # Input is pre-gathered: [padded_num_reqs, hidden_dim]
        dummy_hs = jnp.zeros((int(padded_num_reqs), int(hidden_dim)), dtype=dtype)

        if self._uses_mpmd_mesh():
            self.refresh_lm_head_state(graphdef=graphdef, graphstate=graphstate, graphother=graphother)
            if self._lm_head_graphdef is None or self._lm_head_state is None:
                raise ValueError("eSurge PP lm_head state was not initialized.")
            stage_mesh = self._lm_head_stage_mesh()
            hidden_sharding = self._lm_head_hidden_sharding(tuple(dummy_hs.shape))
            if hidden_sharding is not None:
                dummy_hs = jax.device_put(dummy_hs, hidden_sharding)
            logits_sharding = self._lm_head_replicated_sharding()
            lm_head_state = self._lm_head_state
            lm_head_state_sharding = spx.extract_sharding_structure(
                lm_head_state,
                mesh=self.mesh,
                stage_mesh=stage_mesh,
            )

            jit_kwargs = {}
            if hidden_sharding is not None:
                jit_kwargs["in_shardings"] = (lm_head_state_sharding, hidden_sharding)
            if logits_sharding is not None:
                jit_kwargs["out_shardings"] = logits_sharding

            mesh_context = stage_mesh if stage_mesh is not None else self.model.mesh
            lm_head_graphdef = self._lm_head_graphdef
            clip_cap = self._lm_head_clip_cap
            logit_scale = self._lm_head_logit_scale
            soft_cap = self._lm_head_soft_cap
            uses_tied_projection = self._lm_head_uses_tied_projection

            @spx.jit(**jit_kwargs)  # pyright: ignore[reportUntypedFunctionDecorator]
            def _stage_lm_head(head_state_, hs_):
                with mesh_context:
                    projection = spx.bind(lm_head_graphdef, head_state_)
                    if uses_tied_projection:
                        attend = getattr(projection, "attend", None)
                        if attend is None:
                            weight = projection.weight.value
                            logits = jnp.dot(hs_, weight.T)
                        else:
                            logits = attend(hs_)
                    else:
                        native_forward = getattr(projection, "native_forward", None)
                        if native_forward is None:
                            logits = projection(hs_)
                        else:
                            logits = native_forward(hs_)
                    if clip_cap is not None:
                        cap = jnp.array(clip_cap, dtype=logits.dtype)
                        logits = jnp.clip(logits, -cap, cap)
                    if logit_scale is not None:
                        logits = logits * jnp.array(logit_scale, dtype=logits.dtype)
                    if soft_cap is not None:
                        cap = jnp.array(soft_cap, dtype=logits.dtype)
                        logits = cap * jax.nn.tanh(logits / cap)
                    return logits

            _ = _stage_lm_head(lm_head_state, dummy_hs)

            def wrapped_lm_head(graphstate_, graphother_, hs_):
                del graphstate_, graphother_
                if self._lm_head_state is None:
                    raise ValueError("eSurge PP lm_head state was not initialized.")
                return _stage_lm_head(self._lm_head_state, self._place_lm_head_hidden(hs_))

            self._cache_store(self._lm_head_cache, key, wrapped_lm_head)
            return

        if self.use_aot_forward:
            if self.bind_graphstate_for_aot:

                def _bound_lm_head(hs_):
                    return self._lm_head_fn(graphdef, graphstate, graphother, hs_)

                compiled_bound = spx.jit(_bound_lm_head).lower(dummy_hs).compile()

                def _wrapped_bound_lm_head(graphstate_, graphother_, hs_):
                    del graphstate_, graphother_
                    return compiled_bound(hs_)

                self._cache_store(self._lm_head_cache, key, _wrapped_bound_lm_head)
            else:
                compiled = self._lm_head_fn.lower(  # pyright: ignore[reportFunctionMemberAccess]
                    *(graphdef, graphstate, graphother, dummy_hs)
                ).compile()
                self._cache_store(self._lm_head_cache, key, compiled)
            return

        def wrapped_lm_head(graphstate_, graphother_, hs_):
            return self._lm_head_fn(self.graphdef, graphstate_, graphother_, hs_)

        _ = wrapped_lm_head(graphstate, graphother, dummy_hs)
        self._cache_store(self._lm_head_cache, key, wrapped_lm_head)

    def _build_backbone_fn(
        self,
        *,
        kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
    ) -> tp.Callable[..., BackboneOutputs]:
        """Build the JIT-compiled backbone function (forward pass without lm_head).

        The backbone is compiled once per ``num_tokens`` bucket because its
        input shapes depend only on the token-budget and fixed ``max_num_reqs``
        arrays — *not* on ``padded_num_reqs``.

        Returns:
            JIT-decorated backbone function with signature:
            ``(graphdef, graphstate, graphother, kv_pages, metadata)``
            → ``BackboneOutputs``
        """
        max_num_reqs = int(self.max_num_reqs)
        num_reqs_max_model_len = min(int(self.metadata.get_max_num_seqs()), max_num_reqs)

        metadata_sharding = BatchMetadata(
            packed_qsl_seqlens=self._empty_sharding,
            packed_i32_padded=self._empty_sharding,
            packed_f32_padded=self._empty_sharding,
            packed_misc_i32=self._empty_sharding,
            pages_tables=self._empty_sharding,
            input_ids_buf=self._empty_sharding,
            position_ids_buf=self._empty_sharding,
            slot_mapping=self._empty_sharding if self._use_slot_mapping else None,
            num_kv_update_slices=self._empty_sharding if self._use_slot_mapping else None,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
        )

        kv_pages_sharding = spx.extract_sharding_structure(kv_pages_template, mesh=self.mesh)

        hidden_out_sharding = self._empty_sharding
        if self._uses_mpmd_mesh():
            hidden_size = int(self.model.config.get_text_config().hidden_size)
            hidden_out_sharding = self._lm_head_hidden_sharding((1, hidden_size)) or hidden_out_sharding

        backbone_out_shardings = BackboneOutputs(
            kv_pages=spx.extract_sharding_structure(kv_pages_template, mesh=self.mesh),
            hidden_states=hidden_out_sharding,
        )

        # @erfanzar NOTE:
        #   The MPMD and SPMD JIT paths used to diverge here: SPMD declared
        #   ``in_shardings``/``out_shardings`` while MPMD passed neither.
        #   spectrax's ``sxjit`` *does* honor ``in_shardings`` (see the
        #   placement priority in ``runtime/mpmd/runtime.py``: explicit >
        #   already-on-correct-rank > auto-inferred > replicated), so
        #   omitting them on MPMD just left layout/sharding decisions to
        #   the auto-inference path, which is fine when leaves are already
        #   correctly placed but doesn't pin layouts the way SPMD does. We
        #   now share the same input/output sharding declarations across
        #   both paths; the only real difference is the ``mesh=`` arg
        #   (required by sxjit) and the ``donate_*`` flavor (sxjit rejects
        #   ``donate_argnames``, so we use ``donate_argnums`` everywhere).
        in_shardings = (
            spx.extract_sharding_structure(graphstate_template, mesh=self.mesh),
            spx.extract_sharding_structure(graphother_template, mesh=self.mesh),
            kv_pages_sharding,
            metadata_sharding,
        )
        common_jit_kwargs = {
            "static_argnums": (0,),
            "donate_argnums": (3,),
            "in_shardings": in_shardings,
            "out_shardings": backbone_out_shardings,
        }
        if self._uses_mpmd_mesh():
            jit_kwargs = {**common_jit_kwargs, "mesh": self.mesh}
            if self.mpmd_scheduler is not None:
                jit_kwargs["schedule"] = self.mpmd_scheduler
        else:
            jit_kwargs = dict(common_jit_kwargs)

        @spx.jit(**jit_kwargs)  # pyright: ignore[reportUntypedFunctionDecorator]
        def _backbone_step(
            graphdef,
            graphstate,
            graphother,
            kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
            metadata: BatchMetadata,
        ) -> BackboneOutputs:
            with self.model.mesh:
                model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=False))
                input_ids_view = metadata.input_ids_buf
                position_ids_view = metadata.position_ids_buf

                cache_metadata = RaggedPagesMetadata(
                    pages_tables=metadata.pages_tables,
                    context_lens=metadata.seq_lens[:num_reqs_max_model_len],
                    query_start_loc=metadata.query_start_loc[: num_reqs_max_model_len + 1],
                    num_seqs=jnp.array([metadata.num_requests], dtype=jnp.int32),
                    num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
                    page_size=self.metadata.page_size,
                    request_distribution=metadata.request_distribution,
                    slot_mapping=metadata.slot_mapping,
                    num_kv_update_slices=metadata.num_kv_update_slices,
                    version=self._metadata_version,
                )

                external_inputs: dict[str, tp.Any] = {}
                if metadata.pixel_values is not None or metadata.pixel_values_videos is not None:
                    external_inputs.update(
                        dict(
                            pixel_values=metadata.pixel_values,
                            image_grid_thw=metadata.image_grid_thw,
                            pixel_values_videos=metadata.pixel_values_videos,
                            video_grid_thw=metadata.video_grid_thw,
                        )
                    )
                if metadata.deepstack_visual_embeds is not None:
                    if metadata.visual_pos_masks is None:
                        raise ValueError("`visual_pos_masks` must be provided when `deepstack_visual_embeds` is set.")
                    external_inputs.update(
                        dict(
                            visual_pos_masks=jnp.expand_dims(metadata.visual_pos_masks, 0),
                            deepstack_visual_embeds=list(metadata.deepstack_visual_embeds),
                        )
                    )

                use_prefill_embeds = metadata.prefill_embeds is not None and metadata.prefill_embeds_mask is not None
                use_mrope = metadata.mrope_position_ids is not None

                if use_mrope:
                    position_ids = jnp.expand_dims(metadata.mrope_position_ids, 1)
                else:
                    position_ids = jnp.expand_dims(position_ids_view, 0)

                model_inputs: dict[str, tp.Any]
                if use_prefill_embeds:
                    base_embeds = model.compute_embedding(jnp.expand_dims(input_ids_view, 0))
                    override = jnp.expand_dims(metadata.prefill_embeds, 0).astype(base_embeds.dtype)
                    mask = jnp.expand_dims(metadata.prefill_embeds_mask, 0)[..., None]
                    inputs_embeds = jnp.where(mask, override, base_embeds)
                    model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
                else:
                    model_inputs = {"input_ids": jnp.expand_dims(input_ids_view, 0)}
                with set_inference_mode():
                    output = model(
                        **model_inputs,
                        position_ids=position_ids,
                        past_key_values=kv_pages,
                        cache_metadata=cache_metadata,
                        apply_lm_head=False,
                        **external_inputs,
                    )
                hs = output.last_hidden_state.squeeze(0)

                return BackboneOutputs(
                    kv_pages=output.past_key_values,
                    hidden_states=hs,
                )

        return _backbone_step

    def _build_lm_head_fn(
        self,
        *,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
    ) -> tp.Callable[..., jax.Array]:
        """Build the JIT-compiled lm_head function.

        The lm_head gathers hidden states by ``logits_indices`` and projects
        to vocab logits.  It is compiled per ``padded_num_reqs`` (cheap —
        just a gather + matmul) while the backbone is compiled per
        ``num_tokens`` (expensive — full transformer).

        Returns:
            JIT-decorated lm_head function with signature:
            ``(graphdef, graphstate, graphother, gathered_hidden_states)``
            → ``logits [padded_num_reqs, vocab_size]``

        Note:
            The gather ``hidden_states[logits_indices]`` is done *outside*
            this function (in the ``_combined`` wrapper) so that the compiled
            lm_head only sees ``[padded_num_reqs, hidden_dim]`` input —
            independent of ``num_tokens``.
        """

        hidden_in_sharding = None if self._uses_mpmd_mesh() else self._empty_sharding
        out_sharding = None if self._uses_mpmd_mesh() else self._empty_sharding

        @spx.jit(  # pyright: ignore[reportUntypedFunctionDecorator]
            static_argnums=(0,),
            in_shardings=(
                spx.extract_sharding_structure(graphstate_template, mesh=self.mesh),
                spx.extract_sharding_structure(graphother_template, mesh=self.mesh),
                hidden_in_sharding,
            ),
            out_shardings=out_sharding,
        )
        def _lm_head_step(
            graphdef,
            graphstate,
            graphother,
            gathered_hidden_states: jax.Array,
        ) -> jax.Array:
            with self.model.mesh:
                # spx.bind only runs at trace/compile time (inside @spx.jit),
                # not at inference runtime; XLA sees through it.
                model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=False))
                return model.apply_lm_head(gathered_hidden_states)

        return _lm_head_step
