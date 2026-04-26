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

from easydel.caching import (
    HybridCache,
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RaggedPagesMetadata,
    UnifiedAttentionCache,
    UnifiedAttentionCacheConfig,
)
from easydel.utils import set_inference_mode

from ..execution_types import BackboneOutputs, BatchMetadata, ModelStepOutputs, StepFunctionInputs

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


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
        mesh: tp.Any,
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

    def clear_cache(self) -> None:
        """Clear all cached compiled functions."""
        self._backbone_cache.clear()
        self._lm_head_cache.clear()
        self._cache.clear()

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
            gathered_hs = backbone_out.hidden_states[metadata_.logits_indices[:_pnr]]
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
                return self._backbone_fn(self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)

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

            def wrapped_lm_head(graphstate_, graphother_, hs_):
                return self._lm_head_fn(self.graphdef, graphstate_, graphother_, hs_)

            _ = wrapped_lm_head(graphstate, graphother, dummy_hs)
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

        backbone_out_shardings = BackboneOutputs(
            kv_pages=spx.extract_sharding_structure(kv_pages_template, mesh=self.mesh),
            hidden_states=self._empty_sharding,
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
                model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=True))
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

        @spx.jit(  # pyright: ignore[reportUntypedFunctionDecorator]
            static_argnums=(0,),
            in_shardings=(
                spx.extract_sharding_structure(graphstate_template, mesh=self.mesh),
                spx.extract_sharding_structure(graphother_template, mesh=self.mesh),
                self._empty_sharding,
            ),
            out_shardings=self._empty_sharding,
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
                model: "EasyDeLBaseModule" = spx.bind(graphdef, graphstate.merge(graphother, copy=True))
                return model.apply_lm_head(gathered_hidden_states)

        return _lm_head_step
