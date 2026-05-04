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
        self.mpmd_scheduler = mpmd_scheduler
        self.pipeline_plan = pipeline_plan
        self._pipeline_runtime = (
            PipelineStageRuntime(plan=pipeline_plan) if pipeline_plan is not None and pipeline_plan.is_enabled else None
        )
        self._lm_head_uses_tied_projection = self._uses_tied_lm_head() if self._uses_mpmd_mesh() else False
        self._lm_head_hidden_sharding_cache: dict[tuple[int, ...], NamedSharding | None] = {}
        del cache_capacity

        self._backbone_fn = self._build_backbone_fn(
            kv_pages_template=kv_pages_template,
            graphstate_template=graphstate_template,
            graphother_template=graphother_template,
        )
        self._model_step_fn = (
            None
            if self._uses_mpmd_mesh()
            else self._build_model_step_fn(
                kv_pages_template=kv_pages_template,
                graphstate_template=graphstate_template,
                graphother_template=graphother_template,
            )
        )
        self._can_fuse_pipeline_model_step = self._uses_mpmd_mesh() and not self._lm_head_uses_tied_projection
        self._pipeline_model_step_templates = (kv_pages_template, graphstate_template, graphother_template)
        self._lm_head_fn = self._build_lm_head_fn(
            graphstate_template=graphstate_template,
            graphother_template=graphother_template,
        )
        # Backbone cache: keyed by (num_tokens, "backbone", mode)
        self._backbone_cache: OrderedDict[tuple[int, str, str], tp.Any] = OrderedDict()
        # LM-head cache: keyed by (padded_num_reqs, "lm_head", mode)
        self._lm_head_cache: OrderedDict[tuple[int, str, str], tp.Any] = OrderedDict()
        self._pipeline_model_step_fn_cache: OrderedDict[tuple[int, str], tp.Any] = OrderedDict()
        self._pipeline_lm_head_cache: OrderedDict[tuple[int, tuple[int, ...], str, str], tp.Any] = OrderedDict()
        # Legacy combined cache (kept for backward compat with tests)
        self._cache: OrderedDict[tuple[int, int, str, str], tp.Any] = OrderedDict()
        self._pipeline_kv_carry_map_cache: dict[int, dict[int, dict[int, int]]] = {}
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

    @property
    def pipeline_runtime(self) -> PipelineStageRuntime | None:
        """Resident SpectraX PP runtime used by the MPMD backbone path.

        ``None`` means this executor is running either SPMD or the direct
        ``sxjit`` MPMD dispatcher. eSurge uses the property instead of reaching
        into ``_pipeline_runtime`` so the ownership boundary is clear: this
        class owns compiled model functions and the optional resident runtime.
        """
        return self._pipeline_runtime

    @property
    def supports_pipeline_model_step(self) -> bool:
        """Whether this MPMD executor can fuse backbone gather + LM head.

        The fused PP model step is intentionally disabled for tied LM-head
        models because the projection weights live with the embedding stage,
        not the final transformer stage. Callers use this property to avoid
        repeatedly asking for a fused bucket that cannot exist for the model.
        """
        return self._can_fuse_pipeline_model_step

    def clear_cache(self) -> None:
        """Drop every compiled backbone / LM-head / legacy variant.

        Called when the user explicitly asks for a recompile or during engine
        teardown. Does *not* shut down PP workers — see
        :meth:`shutdown` for that.
        """
        self._backbone_cache.clear()
        self._lm_head_cache.clear()
        self._pipeline_model_step_fn_cache.clear()
        self._pipeline_lm_head_cache.clear()
        self._cache.clear()
        self._pipeline_kv_carry_map_cache.clear()
        self._lm_head_hidden_sharding_cache.clear()
        if self._pipeline_runtime is not None:
            self._pipeline_runtime.clear_prepare_cache()

    def shutdown(self) -> None:
        """Join the resident PP-stage worker threads, if any.

        No-op when running SPMD (``_pipeline_runtime is None``). Called
        from the engine's ``terminate`` / ``release_model_state`` paths so
        the daemon threads do not survive the engine that owns them.
        """
        if self._pipeline_runtime is not None:
            self._pipeline_runtime.shutdown()

    @staticmethod
    def _cache_store(cache: OrderedDict, key, value) -> None:
        """Store ``value`` under ``key`` and mark it most-recently-used.

        Args:
            cache (OrderedDict): LRU cache to mutate in place.
            key: Hashable cache key.
            value: Cached payload.
        """
        cache[key] = value
        cache.move_to_end(key)

    @staticmethod
    def _cache_lookup(cache: OrderedDict, key):
        """Look up ``key`` and mark it most-recently-used.

        Args:
            cache (OrderedDict): LRU cache to read.
            key: Hashable cache key.

        Returns:
            The cached value associated with ``key``.

        Raises:
            KeyError: If ``key`` is not in the cache.
        """
        value = cache[key]
        cache.move_to_end(key)
        return value

    def _uses_mpmd_mesh(self) -> bool:
        """Whether this executor's mesh requires the MPMD compile path.

        Drives every PP/SPMD branch in this class: when ``True``, backbone
        and LM head are compiled with ``mesh=`` and (optionally) a
        SpectraX schedule, hidden states are explicitly placed on the
        final-stage submesh, and the resident pipeline runtime is used in
        place of ``spx.jit``'s default dispatcher.
        """
        return self.mesh.is_mpmd

    def _final_stage_mesh(self):
        """Resolve the submesh that owns the final transformer layer.

        Three resolution strategies, in priority order:

        1. The pipeline plan's recorded ``stage_meshes[final_stage]``,
           when an enabled :class:`PipelineInferencePlan` is attached.
        2. ``model._layer_physical_stage_assignment(N-1, N)`` when the
           model exposes the per-layer assignment helper.
        3. Heuristic fallback assuming the last layer lives on the last
           pipeline stage (``stage = (mpmd_dim - 1, mpmd_dim)``).

        Returns ``None`` on SPMD meshes (no PP topology to resolve).
        """
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
        """Resolve the submesh that owns the input embedding layer.

        Mirror of :meth:`_final_stage_mesh` but always points at stage 0
        (the embedding always lives on the first pipeline stage). Used
        for tied-embedding LM-head placement, where the projection
        weights are physically the same as the input embedding's. Returns
        ``None`` on SPMD meshes.
        """
        if not self._uses_mpmd_mesh():
            return None
        if self.pipeline_plan is not None and self.pipeline_plan.is_enabled and self.pipeline_plan.stage_meshes:
            return self.pipeline_plan.stage_meshes[0]
        mpmd_dim = int(getattr(self.mesh, "mpmd_dim", 1))
        return resolve_stage_mesh(self.mesh, stage=(0, mpmd_dim))

    def _lm_head_stage_mesh(self):
        """Pick the embedding-stage or final-stage mesh based on tied-embedding policy.

        For tied-embedding models the LM head reuses the embedding
        weights and must therefore run on the same stage as the embedding
        (stage 0). For untied heads it lives on the same stage as the
        final transformer layer. Centralizing the choice here keeps every
        LM-head sharding helper consistent.
        """
        if self._lm_head_uses_tied_projection:
            return self._embedding_stage_mesh()
        return self._final_stage_mesh()

    def _lm_head_replicated_sharding(self) -> NamedSharding | None:
        """``NamedSharding`` that fully replicates over the LM-head stage.

        Used as the ``out_shardings`` of the LM-head jit so the resulting
        ``[padded_num_reqs, vocab]`` logits are replicated across every
        device in the head stage — convenient because the sampler then
        sees a tensor that doesn't need any cross-stage gather. Returns
        ``None`` on SPMD or when no LM-head stage exists.
        """
        stage_mesh = self._lm_head_stage_mesh()
        if stage_mesh is None:
            return None
        return NamedSharding(stage_mesh, PartitionSpec())

    def _uses_tied_lm_head(self) -> bool:
        """Check whether the loaded model exposes a tied-embedding LM head.

        Inspects ``self.model.config`` (and its nested text config when
        present) for either of the standard tied-embedding flags
        ``tie_word_embeddings`` / ``share_input_output_layers``. Both
        sub-configs are scanned because some HF wrappers only set the
        flag on the inner text config.

        Returns:
            ``True`` when at least one matching flag is truthy on either
            config; ``False`` otherwise (untied head).
        """
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
        """Sharding for the LM head's input ``[padded_num_reqs, hidden_dim]`` tensor.

        Asks the model's runtime sharding resolver (rebound to the
        LM-head stage mesh) for the ``[EMPTY, EMBED]`` decode-mode
        layout for the given shape, then wraps it in a
        :class:`NamedSharding`. Used both as the ``in_shardings`` for
        the head jit and as the placement target for the gathered hidden
        states fed to it. Returns ``None`` on SPMD or when no LM-head
        stage exists.

        Args:
            shape: Concrete shape of the hidden-state tensor at the call
                site; passed through to ``runtime_sharding_resolver``
                because some resolvers specialize on shape.
        """
        shape = tuple(int(dim) for dim in shape)
        cached = self._lm_head_hidden_sharding_cache.get(shape)
        if shape in self._lm_head_hidden_sharding_cache:
            return cached
        stage_mesh = self._lm_head_stage_mesh()
        if stage_mesh is None:
            self._lm_head_hidden_sharding_cache[shape] = None
            return None
        resolver = self.model.config.runtime_sharding_resolver.with_mesh(stage_mesh)
        spec = resolver.resolve(
            axes=[spx.common_types.EMPTY, spx.common_types.EMBED],
            mode=spx.common_types.MODE_DECODE,
            shape=shape,
        )
        sharding = NamedSharding(stage_mesh, spec)
        self._lm_head_hidden_sharding_cache[shape] = sharding
        return sharding

    def _place_lm_head_hidden(self, hidden_states: jax.Array) -> jax.Array:
        """Move ``hidden_states`` to the LM-head sharding when needed; pass through otherwise.

        Skips the move when (a) the executor is on an SPMD mesh, (b) no
        LM-head sharding can be resolved, or (c) ``hidden_states`` is
        already on the matching mesh+spec. Otherwise issues
        ``jax.device_put`` to satisfy the LM-head jit's
        ``in_shardings`` invariant. The cheap fast path matters because
        every step gathers from the backbone's ``hidden_states`` and
        feeds the result here.
        """
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
        """Pin every leaf of an exported LM-head state onto the head-stage mesh.

        After :func:`spx.export` produces the LM-head sub-graph state,
        leaves still carry shardings from the full model mesh. This
        helper rebinds them onto just the LM-head stage submesh via
        :func:`spx.extract_sharding_structure` followed by
        :func:`jax.device_put` per leaf. Result is a state pytree whose
        compile-time placement matches the dedicated head jit and which
        does not retain references to devices on other stages. SPMD and
        no-stage-mesh cases pass through unchanged.

        Args:
            state: LM-head state pytree as returned by
                :func:`spx.export`.

        Returns:
            New pytree with the same structure but stage-local leaves.
        """
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
        """Capture model-level logit post-processing parameters into instance state.

        Compile-time inputs to the dedicated LM-head jit. Reads three
        scalars off the live bound model (re-derived after every weight
        refresh because they may depend on weights):

        * ``_lm_head_clip_cap`` — symmetric ``±cap`` clip applied to
          logits, when the model exposes a ``_logit_cap_feature``.
        * ``_lm_head_logit_scale`` — multiplicative scaling, taken as
          the product of any of ``logit_scale``,
          ``output_multiplier_scale``, and ``base_model.lm_head_multiplier``.
        * ``_lm_head_soft_cap`` — Gemma-style ``cap * tanh(logits / cap)``
          soft cap, sourced from ``config.final_logit_softcapping`` (or
          its nested text config).

        These three are then closed over by the :meth:`compile_lm_head`
        jit so the head's compiled output already includes the
        post-processing — no extra eager pass is needed during sampling.

        Args:
            model: Live :class:`EasyDeLBaseModule` bound from the
                current ``(graphdef, graphstate, graphother)``.
        """
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
        """Re-export the LM-head sub-graph and place it on the final-stage mesh.

        Pipeline inference compiles the transformer backbone via
        ``spx.jit`` and only projects the sampled-row hidden states
        through the LM head on the final stage. To keep that head
        executable small and reusable across hot weight updates, the
        head graph is re-exported separately each time backbone weights
        change. This method:

        1. Resolves whether the model uses tied embeddings (and if so,
           targets the embedding-stage mesh; otherwise the final-stage
           mesh).
        2. Builds a transient bound model from ``(graphdef, graphstate,
           graphother)``, invokes :meth:`_refresh_lm_head_postprocess`
           to capture clip / scale / soft-cap policy from the live
           model, and selects either ``get_embedding()`` (tied) or
           ``get_lm_head()`` as the projection module.
        3. Exports that projection through :func:`spx.export` and pins
           the resulting state onto the LM-head stage mesh via
           :meth:`_stage_local_lm_head_state` so the per-stage compile
           in :meth:`compile_lm_head` can close over a small, correctly-
           sharded state.

        No-op on SPMD meshes — there's no "final stage" to materialize
        the head on.
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
        """Return the cache-key suffix that identifies the active compile path.

        Three values, mutually exclusive:

        * ``"mpmd"`` — pipeline-parallel MPMD compile (forced when the
          executor's mesh is MPMD, regardless of ``use_aot_forward``).
        * ``"aot"`` — non-PP ahead-of-time compile via
          ``spx.jit(...).lower(...).compile()``.
        * ``"jit"`` — non-PP lazy JIT compile.

        Embedded into every backbone / LM-head / sampler cache key so
        a single :class:`ModelStepExecutor` instance can hold compiled
        variants for several modes simultaneously (useful when toggling
        AOT off for debugging without invalidating MPMD entries).
        """
        if self._uses_mpmd_mesh():
            return "mpmd"
        return "aot" if self.use_aot_forward else "jit"

    def _cache_put(self, key: tuple[int, int, str, str], value: tp.Any) -> None:
        self._cache_store(self._cache, key, value)

    def _cache_get(self, key: tuple[int, int, str, str]) -> tp.Any:
        return self._cache_lookup(self._cache, key)

    def cache_keys(self) -> list:
        """Concatenated key list across the backbone and LM-head caches.

        Useful for diagnostics and tests that want to assert which buckets
        have been compiled. The two caches are keyed differently
        (``(num_tokens, "backbone", mode)`` vs
        ``(padded_num_reqs, "lm_head", mode)``) so the returned list is
        heterogeneous.
        """
        return list(self._cache.keys()) + list(self._backbone_cache.keys()) + list(self._lm_head_cache.keys())

    def has(self, key: tuple[int, int, str, str]) -> bool:
        """Whether *both* halves of a ``(num_tokens, padded_num_reqs)`` are cached.

        Convenience for legacy callers that still treat the two-stage
        compile as a single (num_tokens, padded_num_reqs, "model_step",
        mode) key. Splits the input into the backbone and LM-head sub-keys
        and tests cache membership for each.

        Args:
            key: Four-tuple ``(num_tokens, padded_num_reqs, _, mode)``;
                the third element is ignored (kept for backward compat),
                and ``mode`` is auto-resolved from
                :meth:`_compile_mode` when only three elements are
                provided.

        Returns:
            ``True`` iff a backbone has been compiled for ``num_tokens``
            *and* an LM head has been compiled for ``padded_num_reqs``
            under the same compile mode.
        """
        mode = key[3] if len(key) == 4 else self._compile_mode()
        if not self._uses_mpmd_mesh():
            return (int(key[0]), int(key[1]), "model_step", mode) in self._cache
        backbone_key = (key[0], "backbone", mode)
        lm_head_key = (key[1], "lm_head", mode)
        return backbone_key in self._backbone_cache and lm_head_key in self._lm_head_cache

    def has_model_step(self, num_tokens: int, padded_num_reqs: int) -> bool:
        """Whether the fused SPMD model-step variant is compiled."""
        mode = self._compile_mode()
        return (int(num_tokens), int(padded_num_reqs), "model_step", mode) in self._cache

    def has_backbone(self, num_tokens: int) -> bool:
        """Whether a backbone variant has been compiled for the given token bucket.

        Args:
            num_tokens: Token-axis bucket size to look up.

        Returns:
            ``True`` if the backbone cache already contains an entry for
            ``num_tokens`` under the executor's current compile mode.
        """
        if not self._uses_mpmd_mesh():
            return any(key[0] == int(num_tokens) and key[2] == "model_step" for key in self._cache)
        mode = self._compile_mode()
        return (int(num_tokens), "backbone", mode) in self._backbone_cache

    def has_lm_head(self, padded_num_reqs: int) -> bool:
        """Whether an LM-head variant has been compiled for the given request bucket.

        Args:
            padded_num_reqs: Request-axis bucket size to look up.

        Returns:
            ``True`` if the LM-head cache already contains an entry for
            ``padded_num_reqs`` under the executor's current compile mode.
        """
        if not self._uses_mpmd_mesh():
            return any(key[1] == int(padded_num_reqs) and key[2] == "model_step" for key in self._cache)
        mode = self._compile_mode()
        return (int(padded_num_reqs), "lm_head", mode) in self._lm_head_cache

    def get_compiled(self, *, num_tokens: int, padded_num_reqs: int) -> tp.Any:
        """Stitch backbone + LM head into a single combined callable.

        Looks both compiled halves up under the executor's current compile
        mode and returns a closure that mimics the pre-split unified
        ``model_step`` signature: ``(graphstate, graphother, kv_pages,
        metadata) -> ModelStepOutputs``. The closure performs the
        ``hidden_states[logits_indices]`` gather *outside* the LM-head
        executable so that compiled variant sees a ``[padded_num_reqs,
        hidden_dim]`` input regardless of ``num_tokens``.

        Args:
            num_tokens: Token-axis bucket size; selects the backbone
                variant.
            padded_num_reqs: Request-axis bucket size; selects the LM-head
                variant.

        Returns:
            Callable producing :class:`ModelStepOutputs` from
            ``(graphstate, graphother, kv_pages, metadata)``. The returned
            ``ModelStepOutputs`` carries the updated KV pages, the full
            backbone hidden states (so callers may reuse them for spec
            decoding etc.), and the gathered logits.
        """
        mode = self._compile_mode()
        if not self._uses_mpmd_mesh():
            return self._cache_lookup(self._cache, (int(num_tokens), int(padded_num_reqs), "model_step", mode))
        fused_key = (int(num_tokens), int(padded_num_reqs), "model_step", mode)
        if fused_key in self._cache:
            return self._cache_lookup(self._cache, fused_key)
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
        """Look up the compiled backbone for ``num_tokens``.

        Args:
            num_tokens: Token-axis bucket size identifying the variant.

        Returns:
            The compiled backbone callable. Has signature
            ``(graphstate, graphother, kv_pages, metadata) -> BackboneOutputs``
            (or, in the AOT-bound path, ``(kv_pages, metadata)`` with
            graphstate / graphother captured as constants).

        Raises:
            KeyError: If no backbone has been compiled for that bucket
                under the current compile mode.
        """
        mode = self._compile_mode()
        return self._cache_lookup(self._backbone_cache, (int(num_tokens), "backbone", mode))

    def get_lm_head(self, *, padded_num_reqs: int) -> tp.Any:
        """Look up the compiled LM head for ``padded_num_reqs``.

        Args:
            padded_num_reqs: Request-axis bucket size identifying the
                variant.

        Returns:
            The compiled LM-head callable producing ``logits`` of shape
            ``[padded_num_reqs, vocab_size]`` from
            ``(graphstate, graphother, gathered_hidden_states)``.

        Raises:
            KeyError: If no LM head has been compiled for that bucket
                under the current compile mode.
        """
        mode = self._compile_mode()
        return self._cache_lookup(self._lm_head_cache, (int(padded_num_reqs), "lm_head", mode))

    def get_pipeline_lm_head(self, *, padded_num_reqs: int, part_rows: tuple[int, ...]) -> tp.Any:
        """Return a PP decode LM-head that gathers/concats microbatch hidden rows.

        The normal PP microbatch path used to launch a separate combine kernel
        to gather hidden rows from each microbatch into a full
        ``[padded_num_reqs, hidden]`` tensor, then launch the LM head.  This
        variant folds that gather/concat directly into the final-stage LM-head
        executable, which removes one device dispatch from every decode step.
        """
        mode = self._compile_mode()
        part_rows = tuple(int(x) for x in part_rows)
        key = (int(padded_num_reqs), part_rows, "pipeline_lm_head", mode)
        cached = self._pipeline_lm_head_cache.get(key)
        if cached is not None:
            return cached
        if not self._uses_mpmd_mesh():
            return self.get_lm_head(padded_num_reqs=padded_num_reqs)
        if self._lm_head_graphdef is None or self._lm_head_state is None:
            raise ValueError("eSurge PP lm_head state was not initialized.")

        hidden_dim = int(self.model.config.get_text_config().hidden_size)
        dtype = self.model.dtype
        dummy_parts = []
        dummy_indices = []
        for rows in part_rows:
            rows = max(1, int(rows))
            dummy_hs = jnp.zeros((rows, hidden_dim), dtype=dtype)
            part_sharding = self._lm_head_hidden_sharding(tuple(dummy_hs.shape))
            if part_sharding is not None:
                dummy_hs = jax.device_put(dummy_hs, part_sharding)
            dummy_parts.append(dummy_hs)
            dummy_indices.append(jnp.arange(rows, dtype=jnp.int32))

        stage_mesh = self._lm_head_stage_mesh()
        logits_sharding = self._lm_head_replicated_sharding()
        lm_head_state = self._lm_head_state
        lm_head_graphdef = self._lm_head_graphdef
        clip_cap = self._lm_head_clip_cap
        logit_scale = self._lm_head_logit_scale
        soft_cap = self._lm_head_soft_cap
        uses_tied_projection = self._lm_head_uses_tied_projection
        mesh_context = stage_mesh if stage_mesh is not None else self.model.mesh

        jit_kwargs = {}
        if logits_sharding is not None:
            jit_kwargs["out_shardings"] = logits_sharding

        @spx.jit(**jit_kwargs)  # pyright: ignore[reportUntypedFunctionDecorator]
        def _stage_lm_head_from_parts(head_state_, hidden_parts_, index_parts_):
            with mesh_context:
                with jax.named_scope("easydel/esurge/pp_lm_head_from_parts"):
                    with jax.named_scope("easydel/esurge/pp_lm_head_from_parts/gather_hidden_rows"):
                        gathered = tuple(
                            part_hidden[part_indices]
                            for part_hidden, part_indices in zip(hidden_parts_, index_parts_, strict=True)
                        )
                        hs_ = jnp.concatenate(gathered, axis=0)
                        pad_rows = int(padded_num_reqs) - int(hs_.shape[0])
                        if pad_rows > 0:
                            hs_ = jnp.pad(hs_, ((0, pad_rows), (0, 0)))
                        hs_ = hs_[: int(padded_num_reqs)]

                    with jax.named_scope("easydel/esurge/pp_lm_head_from_parts/project"):
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
                    with jax.named_scope("easydel/esurge/pp_lm_head_from_parts/logit_postprocess"):
                        if clip_cap is not None:
                            cap = jnp.array(clip_cap, dtype=logits.dtype)
                            logits = jnp.clip(logits, -cap, cap)
                        if logit_scale is not None:
                            logits = logits * jnp.array(logit_scale, dtype=logits.dtype)
                        if soft_cap is not None:
                            cap = jnp.array(soft_cap, dtype=logits.dtype)
                            logits = cap * jax.nn.tanh(logits / cap)
                    return logits

        _ = _stage_lm_head_from_parts(lm_head_state, tuple(dummy_parts), tuple(dummy_indices))

        def wrapped_lm_head_from_parts(graphstate_, graphother_, hidden_parts_, index_parts_):
            _ = (graphstate_, graphother_)
            if self._lm_head_state is None:
                raise ValueError("eSurge PP lm_head state was not initialized.")
            return _stage_lm_head_from_parts(self._lm_head_state, hidden_parts_, index_parts_)

        self._cache_store(self._pipeline_lm_head_cache, key, wrapped_lm_head_from_parts)
        return wrapped_lm_head_from_parts

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
        """Compile (or no-op-if-cached) the backbone and the LM head.

        Backbone is keyed solely by ``num_tokens`` because its compute
        cost dominates and depends only on the token-axis padding;
        request-axis padding is absorbed by the LM-head executable, which
        is keyed solely by ``padded_num_reqs``. This split keeps the
        compile-cache product small (``|num_tokens|`` + ``|padded_num_reqs|``
        instead of their cartesian product).

        Args:
            num_tokens: Token-axis bucket size to compile for.
            padded_num_reqs: Request-axis bucket size to compile for.
            graphdef: Static model graph definition. Stashed on ``self`` so
                wrapper closures can see fresh weights after a hot-swap.
            graphstate: Live mutable parameters used as the trace template.
            graphother: Live auxiliary buffers used as the trace template.
            inputs: Dummy ``StepFunctionInputs`` shaped for ``num_tokens``
                used by the AOT path's ``lower(...).compile()`` call.

        Returns:
            :class:`BackboneOutputs` produced by the eager pre-touch when
            the backbone path is JIT (lazy) — useful for warming caches.
            ``None`` if either half was already compiled or when the AOT
            path is taken (no eager call is made).
        """
        if not self._uses_mpmd_mesh():
            return self.compile_model_step(
                num_tokens=num_tokens,
                padded_num_reqs=padded_num_reqs,
                graphdef=graphdef,
                graphstate=graphstate,
                graphother=graphother,
                inputs=inputs,
            )
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

    def compile_model_step(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        graphdef: tp.Any,
        graphstate: tp.Any,
        graphother: tp.Any,
        inputs: StepFunctionInputs,
    ) -> ModelStepOutputs | None:
        """Compile the fused SPMD model step for one token/request bucket pair.

        The MPMD path intentionally keeps the backbone and LM head separate so
        stage-local placement can be controlled. On SPMD there is no stage
        boundary, so splitting the decode step into backbone -> host-side gather
        -> LM-head creates extra device dispatches on every generated token.
        This fused executable restores the old single-step shape while keeping
        the same ``ModelStepOutputs`` contract.
        """
        if self._uses_mpmd_mesh():
            raise RuntimeError("compile_model_step is only used on SPMD meshes.")
        if self._model_step_fn is None:
            raise RuntimeError("SPMD model-step function was not initialized.")

        self.graphdef = graphdef
        mode = self._compile_mode()
        key = (int(num_tokens), int(padded_num_reqs), "model_step", mode)
        if key in self._cache:
            return None

        if self.use_aot_forward:
            compiled = self._model_step_fn.lower(  # pyright: ignore[reportFunctionMemberAccess]
                *(
                    graphdef,
                    graphstate,
                    graphother,
                    inputs.kv_pages,
                    inputs.batch_metadata,
                    int(padded_num_reqs),
                )
            ).compile()
            self._cache_store(self._cache, key, compiled)
            return None

        def wrapped_model_step(graphstate_, graphother_, kv_pages_, metadata_):
            return self._model_step_fn(
                self.graphdef,
                graphstate_,
                graphother_,
                kv_pages_,
                metadata_,
                padded_num_reqs,
            )

        out = wrapped_model_step(graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
        self._cache_store(self._cache, key, wrapped_model_step)
        return out

    def compile_pipeline_model_step(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        graphdef: tp.Any,
        graphstate: tp.Any,
        graphother: tp.Any,
        inputs: StepFunctionInputs,
    ) -> ModelStepOutputs | None:
        """Compile the fused PP model step for one token/request bucket pair.

        The regular MPMD decode path historically launched the pipeline
        backbone, then gathered sampled hidden rows and ran a separate final-
        stage LM-head executable. That split is useful for PP microbatch
        wavefronts, but it is expensive for the latency path where each decode
        step contains a single logical batch: the extra LM-head launch lands on
        every generated token.

        This variant keeps the backbone split across SpectraX pipeline stages
        while placing the sampled-row gather and LM-head projection in the same
        final-stage ``sxjit`` plan. It is only enabled for untied LM heads; tied
        projection models still use the split path so their embedding-stage
        weights are not pulled across the PP boundary.
        """
        if not self._uses_mpmd_mesh():
            raise RuntimeError("compile_pipeline_model_step is only used on MPMD meshes.")
        if not self.supports_pipeline_model_step:
            return None

        self.graphdef = graphdef
        mode = self._compile_mode()
        key = (int(num_tokens), int(padded_num_reqs), "model_step", mode)
        if key in self._cache:
            return None

        pnr = int(padded_num_reqs)
        pipeline_model_step_fn = self._get_pipeline_model_step_fn(pnr, graphdef=self.graphdef)

        def wrapped_pipeline_model_step(graphstate_, graphother_, kv_pages_, metadata_):
            return self._dispatch_pipeline_model_step(
                pipeline_model_step_fn,
                graphstate_,
                graphother_,
                kv_pages_,
                metadata_,
                padded_num_reqs=pnr,
            )

        out = wrapped_pipeline_model_step(graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
        self._cache_store(self._cache, key, wrapped_pipeline_model_step)
        return out

    def compile_backbone(
        self,
        *,
        num_tokens: int,
        graphdef: tp.Any,
        graphstate: tp.Any,
        graphother: tp.Any,
        inputs: StepFunctionInputs,
    ) -> BackboneOutputs | None:
        """Compile the transformer forward for one token bucket.

        Selects between three implementations depending on the active
        compile mode (set by :meth:`_compile_mode`):

        * **MPMD** — wraps the call in a closure that defers to
          :meth:`_dispatch_pipeline_backbone`, which routes through the
          resident :class:`PipelineStageRuntime` when available; eagerly
          warms once with the dummy inputs.
        * **AOT** — lowers and compiles the backbone jit through
          ``spx.jit(...).lower(...).compile()`` with graphstate / graphother
          as normal dynamic inputs.
        * **JIT** — stores a closure around the lazy ``spx.jit`` so the
          first dispatch traces.

        Caches the resulting callable in ``self._backbone_cache`` keyed
        by ``(num_tokens, "backbone", mode)``.

        Args:
            num_tokens: Token-axis bucket to compile for.
            graphdef: Updated graphdef pinned on ``self`` for closure use.
            graphstate, graphother: Trace templates / capture constants.
            inputs: Dummy ``StepFunctionInputs`` providing concrete shapes
                for the lower step.

        Returns:
            :class:`BackboneOutputs` from the eager warm-up call when the
            MPMD or non-bound JIT paths are taken; ``None`` for the AOT
            paths (no eager call) and when the bucket was already cached.
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
        """Run the backbone through the resident PP runtime, falling back to direct call.

        Called from the MPMD-mode wrapper installed by
        :meth:`compile_backbone`. When :class:`PipelineStageRuntime` is
        available (it is, whenever the executor was built with an
        enabled :class:`PipelineInferencePlan`), the call is routed
        through :meth:`PipelineStageRuntime.dispatch` so the
        per-stage worker threads service the SpectraX compile plan and
        last-call dispatch stats are surfaced via
        :meth:`last_pipeline_stats`. With no runtime configured (single-
        stage MPMD topology), the fallback path invokes ``self._backbone_fn``
        directly, leaving SpectraX's default in-line dispatcher in
        charge.

        Args:
            graphstate_, graphother_, kv_pages_, metadata_: Same step
                inputs the compiled wrapper received from the runner.

        Returns:
            :class:`BackboneOutputs` for this step.
        """
        pipeline_runtime = self.pipeline_runtime
        if pipeline_runtime is None:
            return self._backbone_fn(self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)
        return pipeline_runtime.dispatch(
            self._backbone_fn,
            self.graphdef,
            graphstate_,
            graphother_,
            kv_pages_,
            metadata_,
            prepare_cache_key=(
                "backbone",
                int(metadata_.input_ids_buf.shape[0]),
                id(self.graphdef),
            ),
            runtime_static_argnums=(0, 1, 2),
        )

    def _dispatch_pipeline_model_step(
        self,
        pipeline_model_step_fn: tp.Any,
        graphstate_,
        graphother_,
        kv_pages_,
        metadata_,
        *,
        padded_num_reqs: int,
    ) -> ModelStepOutputs:
        """Run the fused PP model step through the resident pipeline runtime.

        This mirrors :meth:`_dispatch_pipeline_backbone`, but the prepared
        ``sxjit`` function includes the sampled-row hidden gather and LM-head
        projection. The prepare-cache key includes both token and request
        buckets because the LM-head gather has a static ``padded_num_reqs``.
        """
        pipeline_runtime = self.pipeline_runtime
        if pipeline_runtime is None:
            return pipeline_model_step_fn(
                graphstate_,
                graphother_,
                kv_pages_,
                metadata_,
            )
        return pipeline_runtime.dispatch(
            pipeline_model_step_fn,
            graphstate_,
            graphother_,
            kv_pages_,
            metadata_,
            prepare_cache_key=(
                "model_step",
                int(metadata_.input_ids_buf.shape[0]),
                int(padded_num_reqs),
                id(self.graphdef),
            ),
            runtime_static_argnums=(0, 1),
        )

    def _build_pipeline_kv_carry_map(
        self,
        graphstate_,
        graphother_,
        kv_pages_,
        metadata_,
        *,
        num_tokens: int,
    ) -> dict[int, dict[int, int]]:
        """Map KV input leaves to same-stage KV outputs for PP microbatches.

        ``BackboneOutputs`` flattens as ``(kv_pages leaves..., hidden_states)``.
        The sxjit plan tells us which physical stage produced each output leaf;
        pairing those first ``len(kv_pages)`` output leaves with the
        corresponding ``kv_pages`` input leaves gives SpectraX enough
        information to carry each stage's local cache from microbatch N to
        microbatch N+1 while the other stages continue their own wavefront.
        """
        cache_key = int(num_tokens)
        cached = self._pipeline_kv_carry_map_cache.get(cache_key)
        if cached is not None:
            return cached

        prepare = getattr(self._backbone_fn, "_mpmd_prepare", None)
        if prepare is None:
            return {}
        state = prepare(self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)
        fn_outvar_map = state.get("fn_outvar_map")
        if not fn_outvar_map:
            return {}

        kv_flat_count = len(jax.tree.leaves(kv_pages_))
        if kv_flat_count <= 0:
            return {}
        if len(fn_outvar_map) < kv_flat_count:
            raise RuntimeError("MPMD backbone output map is missing KV-cache leaves.")

        kv_flat_start = len(jax.tree.leaves((self.graphdef, graphstate_, graphother_)))
        carry_map: dict[int, dict[int, int]] = {}
        for leaf_offset in range(kv_flat_count):
            mapping = fn_outvar_map[leaf_offset]
            if len(mapping) < 2:
                continue
            stage_idx, stage_out_pos = mapping[:2]
            if not isinstance(stage_idx, int):
                continue
            carry_map.setdefault(int(stage_idx), {})[kv_flat_start + leaf_offset] = int(stage_out_pos)
        self._pipeline_kv_carry_map_cache[cache_key] = carry_map
        return carry_map

    def execute_many(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        input_batches: tp.Sequence[tuple[tp.Any, tp.Any, tp.Any, BatchMetadata]],
    ) -> list[ModelStepOutputs]:
        """Execute same-shaped model-step microbatches through the PP wavefront.

        This is used by decode-time pipeline parallelism. The transformer
        backbone runs via SpectraX's stage-local carry executor; the LM head
        remains the small final-stage projection already used by
        :meth:`get_compiled`.
        """
        if len(input_batches) == 0:
            return []
        if not self._uses_mpmd_mesh() or self._pipeline_runtime is None or len(input_batches) == 1:
            model_fn = self.get_compiled(num_tokens=num_tokens, padded_num_reqs=padded_num_reqs)
            return [model_fn(*batch) for batch in input_batches]

        backbone_outputs = self.execute_backbones_many(num_tokens=num_tokens, input_batches=input_batches)
        lm_head_fn = self.get_lm_head(padded_num_reqs=padded_num_reqs)

        outputs: list[ModelStepOutputs] = []
        _pnr = int(padded_num_reqs)
        for (graphstate_, graphother_, _, metadata_), backbone_out in zip(
            input_batches,
            backbone_outputs,
            strict=True,
        ):
            logits_indices = metadata_.logits_indices[:_pnr]
            logits_indices = replicate_on_array_mesh(logits_indices, backbone_out.hidden_states)
            gathered_hs = backbone_out.hidden_states[logits_indices]
            gathered_hs = self._place_lm_head_hidden(gathered_hs)
            logits = lm_head_fn(graphstate_, graphother_, gathered_hs)
            outputs.append(
                ModelStepOutputs(
                    kv_pages=backbone_out.kv_pages,
                    hidden_states=backbone_out.hidden_states,
                    logits=logits,
                )
            )
        return outputs

    def execute_backbones_many(
        self,
        *,
        num_tokens: int,
        input_batches: tp.Sequence[tuple[tp.Any, tp.Any, tp.Any, BatchMetadata]],
    ) -> list[BackboneOutputs]:
        """Run same-shaped backbone microbatches through SpectraX PP wavefront."""
        if len(input_batches) == 0:
            return []
        if not self._uses_mpmd_mesh() or self._pipeline_runtime is None or len(input_batches) == 1:
            backbone_fn = self.get_backbone(num_tokens=num_tokens)
            return [backbone_fn(*batch) for batch in input_batches]

        self.get_backbone(num_tokens=num_tokens)
        graphstate0, graphother0, kv_pages0, metadata0 = input_batches[0]
        carry_map = self._build_pipeline_kv_carry_map(
            graphstate0,
            graphother0,
            kv_pages0,
            metadata0,
            num_tokens=num_tokens,
        )
        backbone_arg_batches = [
            (self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)
            for graphstate_, graphother_, kv_pages_, metadata_ in input_batches
        ]
        return self._pipeline_runtime.dispatch_many(
            self._backbone_fn,
            backbone_arg_batches,
            carry_input_output_map=carry_map,
            prepare_cache_key=("backbone", int(num_tokens), id(self.graphdef)),
            runtime_static_argnums=(0, 1, 2),
        )

    def last_pipeline_stats(self) -> dict[str, float | int]:
        """Per-step pipeline-dispatch counters for the runner perf log.

        Surfaced through ``ExecutionManager.execute`` and merged into the
        per-step metrics dict that :class:`eSurgeRunner` formats into the
        ``[perf]`` log line. Two reporting paths:

        * **Resident runtime present** — pulls
          :attr:`PipelineStageRuntime.last_stats` and reports
          ``pp_stage_launches``, ``pp_stage_dispatch_time``, and
          ``pp_queue_wait_time`` (the only counters worth tracking from
          the dispatcher side).
        * **Fallback to SpectraX dispatcher** — peeks at the sxjit
          function's private ``_mpmd_state`` and, if it has recorded a
          forward-only stage-launch count, returns just
          ``pp_stage_launches``; returns an empty dict when no PP work
          ran this step (e.g. SPMD path).

        Returns:
            Mapping from metric name to value, suitable for direct
            inclusion in the runner's per-step metrics dict.
        """
        if self._pipeline_runtime is None:
            state = getattr(self._backbone_fn, "_mpmd_state", {})
            launches = int(state.get("forward_stage_launches", 0) or 0)
            if launches <= 0:
                return {}
            return {"pp_stage_launches": launches}
        stats = self._pipeline_runtime.last_stats
        result = {
            "pp_stage_launches": int(stats.stage_launches),
            "pp_stage_dispatch_time": float(stats.stage_dispatch_time),
            "pp_queue_wait_time": float(stats.queue_wait_time),
            "pp_prepare_time": float(stats.prepare_time),
            "pp_assemble_time": float(stats.assemble_time),
            "pp_submit_time": float(stats.submit_time),
        }
        for stage_idx, submit_ms in enumerate(stats.stage_submit_times_ms):
            result[f"pp_stage_{stage_idx}_submit_time"] = float(submit_ms) / 1000.0
        for stage_idx, assemble_ms in enumerate(stats.stage_assemble_times_ms):
            result[f"pp_stage_{stage_idx}_assemble_time"] = float(assemble_ms) / 1000.0
        for stage_idx, execute_ms in enumerate(stats.stage_execute_times_ms):
            result[f"pp_stage_{stage_idx}_execute_time"] = float(execute_ms) / 1000.0
        return result

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
        """Compile the LM-head projection for one request-axis bucket.

        Mirrors :meth:`compile_backbone` but operates on a much smaller
        slice of work: a gather-then-matmul over the ``padded_num_reqs``
        sampled rows. Three branches are taken based on the compile mode:

        * **MPMD** — refreshes the stage-local LM-head state, places dummy
          hidden states on the LM-head stage submesh, and compiles a
          dedicated ``_stage_lm_head`` jit that runs ``projection(...)``
          (or the tied-embedding ``attend``) plus any logit clip / scale
          / soft-cap post-processing on the final-stage submesh.
        * **AOT** — lowers and compiles ``self._lm_head_fn`` with graphstate /
          graphother as normal dynamic inputs.
        * **JIT** — caches a closure around the lazy ``spx.jit`` form for
          on-demand tracing.

        Stores the resulting callable in ``self._lm_head_cache`` keyed by
        ``(padded_num_reqs, "lm_head", mode)``. Idempotent.

        Args:
            padded_num_reqs: Request-axis bucket to compile for.
            graphdef: Updated graphdef pinned on ``self`` so wrappers see
                fresh weights after a hot-swap.
            graphstate, graphother: Trace templates / capture constants.
            inputs: Reserved for symmetry with :meth:`compile_backbone`;
                the LM head only consumes pre-gathered hidden states, so
                ``inputs`` is unused here.
            hidden_dim: Override for the hidden-size dimension of the
                dummy input. ``None`` reads it from the model's text
                config.
            dtype: dtype of the dummy hidden states. ``None`` reads
                ``self.model.dtype``.
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
                    with jax.named_scope("easydel/esurge/pp_lm_head"):
                        with jax.named_scope("easydel/esurge/pp_lm_head/project"):
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
                        with jax.named_scope("easydel/esurge/pp_lm_head/logit_postprocess"):
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
            input_token_handoff_positions=self._empty_sharding,
            input_token_handoff_ids=self._empty_sharding,
            input_token_handoff_count=self._empty_sharding,
            input_token_handoff_offset=self._empty_sharding,
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
                with jax.named_scope("easydel/esurge/backbone_step"):
                    model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=False))
                    input_ids_view = metadata.model_input_ids
                    position_ids_view = metadata.position_ids_buf

                    with jax.named_scope("easydel/esurge/backbone_step/cache_metadata"):
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
                            raise ValueError(
                                "`visual_pos_masks` must be provided when `deepstack_visual_embeds` is set."
                            )
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

                    with jax.named_scope("easydel/esurge/backbone_step/prepare_inputs"):
                        model_inputs: dict[str, tp.Any]
                        if use_prefill_embeds:
                            base_embeds = model.compute_embedding(jnp.expand_dims(input_ids_view, 0))
                            override = jnp.expand_dims(metadata.prefill_embeds, 0).astype(base_embeds.dtype)
                            mask = jnp.expand_dims(metadata.prefill_embeds_mask, 0)[..., None]
                            inputs_embeds = jnp.where(mask, override, base_embeds)
                            model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
                        else:
                            model_inputs = {"input_ids": jnp.expand_dims(input_ids_view, 0)}

                    with jax.named_scope("easydel/esurge/backbone_step/model_forward"):
                        with set_inference_mode():
                            output = model(
                                **model_inputs,
                                position_ids=position_ids,
                                past_key_values=kv_pages,
                                cache_metadata=cache_metadata,
                                apply_lm_head=False,
                                **external_inputs,
                            )
                    with jax.named_scope("easydel/esurge/backbone_step/output"):
                        hs = output.last_hidden_state.squeeze(0)
                        return BackboneOutputs(kv_pages=output.past_key_values, hidden_states=hs)

        return _backbone_step

    def _get_pipeline_model_step_fn(self, padded_num_reqs: int, *, graphdef: tp.Any) -> tp.Any:
        """Return or build the fused PP model-step function for one request bucket."""
        key = (int(padded_num_reqs), f"{self._compile_mode()}:{id(graphdef)}")
        cached = self._pipeline_model_step_fn_cache.get(key)
        if cached is not None:
            return cached
        kv_pages_template, graphstate_template, graphother_template = self._pipeline_model_step_templates
        fn = self._build_pipeline_model_step_fn(
            kv_pages_template=kv_pages_template,
            graphstate_template=graphstate_template,
            graphother_template=graphother_template,
            padded_num_reqs=int(padded_num_reqs),
            graphdef=graphdef,
        )
        self._cache_store(self._pipeline_model_step_fn_cache, key, fn)
        return fn

    def _build_pipeline_model_step_fn(
        self,
        *,
        kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
        padded_num_reqs: int,
        graphdef: tp.Any,
    ) -> tp.Callable[..., ModelStepOutputs]:
        """Build the fused MPMD PP model step.

        The function has the same model semantics as the SPMD fused step, but
        it is compiled with ``spx.jit(mesh=self.mesh)`` so SpectraX still splits
        transformer layers across physical PP stages. The sampled-row gather
        and LM-head projection are intentionally after the transformer forward,
        which places them in the final logical stage for untied LM-head models.
        """
        padded_num_reqs = int(padded_num_reqs)
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
            input_token_handoff_positions=self._empty_sharding,
            input_token_handoff_ids=self._empty_sharding,
            input_token_handoff_count=self._empty_sharding,
            input_token_handoff_offset=self._empty_sharding,
            slot_mapping=self._empty_sharding if self._use_slot_mapping else None,
            num_kv_update_slices=self._empty_sharding if self._use_slot_mapping else None,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
        )

        kv_pages_sharding = spx.extract_sharding_structure(kv_pages_template, mesh=self.mesh)
        hidden_size = int(self.model.config.get_text_config().hidden_size)
        hidden_out_sharding = self._lm_head_hidden_sharding((max(1, max_num_reqs), hidden_size)) or self._empty_sharding
        logits_out_sharding = self._lm_head_replicated_sharding() or self._empty_sharding
        model_step_out_shardings = ModelStepOutputs(
            kv_pages=kv_pages_sharding,
            hidden_states=hidden_out_sharding,
            logits=logits_out_sharding,
        )

        jit_kwargs = {
            "donate_argnums": (2,),
            "mesh": self.mesh,
            "in_shardings": (
                spx.extract_sharding_structure(graphstate_template, mesh=self.mesh),
                spx.extract_sharding_structure(graphother_template, mesh=self.mesh),
                kv_pages_sharding,
                metadata_sharding,
            ),
            "out_shardings": model_step_out_shardings,
        }
        if self.mpmd_scheduler is not None:
            jit_kwargs["schedule"] = self.mpmd_scheduler

        @spx.jit(**jit_kwargs)  # pyright: ignore[reportUntypedFunctionDecorator]
        def _pipeline_model_step(
            graphstate,
            graphother,
            kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
            metadata: BatchMetadata,
        ) -> ModelStepOutputs:
            with self.model.mesh:
                with jax.named_scope("easydel/esurge/pp_model_step"):
                    model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=False))
                    input_ids_view = metadata.model_input_ids
                    position_ids_view = metadata.position_ids_buf

                    with jax.named_scope("easydel/esurge/pp_model_step/cache_metadata"):
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
                            raise ValueError(
                                "`visual_pos_masks` must be provided when `deepstack_visual_embeds` is set."
                            )
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

                    with jax.named_scope("easydel/esurge/pp_model_step/prepare_inputs"):
                        model_inputs: dict[str, tp.Any]
                        if use_prefill_embeds:
                            base_embeds = model.compute_embedding(jnp.expand_dims(input_ids_view, 0))
                            override = jnp.expand_dims(metadata.prefill_embeds, 0).astype(base_embeds.dtype)
                            mask = jnp.expand_dims(metadata.prefill_embeds_mask, 0)[..., None]
                            inputs_embeds = jnp.where(mask, override, base_embeds)
                            model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
                        else:
                            model_inputs = {"input_ids": jnp.expand_dims(input_ids_view, 0)}

                    with jax.named_scope("easydel/esurge/pp_model_step/model_forward"):
                        with set_inference_mode():
                            output = model(
                                **model_inputs,
                                position_ids=position_ids,
                                past_key_values=kv_pages,
                                cache_metadata=cache_metadata,
                                apply_lm_head=False,
                                **external_inputs,
                            )
                    with jax.named_scope("easydel/esurge/pp_model_step/gather_hidden"):
                        hidden_states = output.last_hidden_state.squeeze(0)
                        gathered_hidden_states = hidden_states[metadata.logits_indices[:padded_num_reqs]]
                    with jax.named_scope("easydel/esurge/pp_model_step/lm_head"):
                        logits = model.apply_lm_head(gathered_hidden_states)
                    return ModelStepOutputs(
                        kv_pages=output.past_key_values,
                        hidden_states=gathered_hidden_states,
                        logits=logits,
                    )

        return _pipeline_model_step

    def _build_model_step_fn(
        self,
        *,
        kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
    ) -> tp.Callable[..., ModelStepOutputs]:
        """Build the fused SPMD model step.

        This is the same transformer forward as :meth:`_build_backbone_fn`,
        but it also gathers the sampled rows and applies the LM head inside the
        compiled executable. Keeping those operations in one SPMD executable
        avoids the per-token dispatch/materialization overhead that the split
        PP path must pay for stage-local LM-head placement.
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
            input_token_handoff_positions=self._empty_sharding,
            input_token_handoff_ids=self._empty_sharding,
            input_token_handoff_count=self._empty_sharding,
            input_token_handoff_offset=self._empty_sharding,
            slot_mapping=self._empty_sharding if self._use_slot_mapping else None,
            num_kv_update_slices=self._empty_sharding if self._use_slot_mapping else None,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
        )

        kv_pages_sharding = spx.extract_sharding_structure(kv_pages_template, mesh=self.mesh)
        model_step_out_shardings = ModelStepOutputs(
            kv_pages=kv_pages_sharding,
            hidden_states=self._empty_sharding,
            logits=self._empty_sharding,
        )

        @jax.jit(
            static_argnums=(0, 5),
            donate_argnums=(3,),
            in_shardings=(
                spx.extract_sharding_structure(graphstate_template, mesh=self.mesh),
                spx.extract_sharding_structure(graphother_template, mesh=self.mesh),
                kv_pages_sharding,
                metadata_sharding,
            ),
            out_shardings=model_step_out_shardings,
        )
        def _model_step(
            graphdef,
            graphstate,
            graphother,
            kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache,
            metadata: BatchMetadata,
            padded_num_reqs: int,
        ) -> ModelStepOutputs:
            with self.model.mesh:
                with jax.named_scope("easydel/esurge/spmd_model_step"):
                    model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=False))
                    input_ids_view = metadata.model_input_ids
                    position_ids_view = metadata.position_ids_buf

                    with jax.named_scope("easydel/esurge/spmd_model_step/cache_metadata"):
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
                            raise ValueError(
                                "`visual_pos_masks` must be provided when `deepstack_visual_embeds` is set."
                            )
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

                    with jax.named_scope("easydel/esurge/spmd_model_step/prepare_inputs"):
                        if use_prefill_embeds:
                            base_embeds = model.compute_embedding(jnp.expand_dims(input_ids_view, 0))
                            override = jnp.expand_dims(metadata.prefill_embeds, 0).astype(base_embeds.dtype)
                            mask = jnp.expand_dims(metadata.prefill_embeds_mask, 0)[..., None]
                            inputs_embeds = jnp.where(mask, override, base_embeds)
                            model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
                        else:
                            model_inputs = {"input_ids": jnp.expand_dims(input_ids_view, 0)}

                    with jax.named_scope("easydel/esurge/spmd_model_step/model_forward"):
                        with set_inference_mode():
                            output = model(
                                **model_inputs,
                                position_ids=position_ids,
                                past_key_values=kv_pages,
                                cache_metadata=cache_metadata,
                                apply_lm_head=False,
                                **external_inputs,
                            )
                    with jax.named_scope("easydel/esurge/spmd_model_step/gather_hidden"):
                        hidden_states = output.last_hidden_state.squeeze(0)
                        gathered_hidden_states = hidden_states[metadata.logits_indices[:padded_num_reqs]]
                    with jax.named_scope("easydel/esurge/spmd_model_step/lm_head"):
                        logits = model.apply_lm_head(gathered_hidden_states)
                    return ModelStepOutputs(
                        kv_pages=output.past_key_values,
                        hidden_states=gathered_hidden_states,
                        logits=logits,
                    )

        return _model_step

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
                with jax.named_scope("easydel/esurge/lm_head_step"):
                    model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=False))
                    with jax.named_scope("easydel/esurge/lm_head_step/project"):
                        return model.apply_lm_head(gathered_hidden_states)

        return _lm_head_step
