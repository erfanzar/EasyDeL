# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from the rest of the execution manager. The resulting executor is responsible for
building and caching compiled variants of the model step for different token/batch
bucket combinations.
"""

from __future__ import annotations

import typing as tp
from collections import OrderedDict

import jax
from eformer import escale as es
from flax import nnx as nn
from jax import numpy as jnp

from easydel.layers.caching import RaggedPagesCache, RaggedPagesCacheView, RaggedPagesMetadata
from easydel.utils import ejit

from ..execution_types import BatchMetadata, ModelStepOutputs, StepFunctionInputs

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


class ModelStepExecutor:
    """Compile/cache and execute the model-only forward step."""

    def __init__(
        self,
        *,
        model: "EasyDeLBaseModule",
        mesh: tp.Any,
        metadata: RaggedPagesCacheView,
        kv_pages_template: RaggedPagesCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
        max_num_reqs: int,
        graphdef: tp.Any,
        empty_sharding: jax.sharding.Sharding,
        use_aot_forward: bool,
        cache_capacity: int = 64,
        maybe_implicit: tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]] | None = None,
    ) -> None:
        self.model = model
        self.mesh = mesh
        self.metadata = metadata
        self.max_num_reqs = int(max_num_reqs)
        self.graphdef = graphdef
        self._metadata_version = metadata.version
        self._use_slot_mapping = self._metadata_version == "v2"
        self._empty_sharding = empty_sharding
        self.use_aot_forward = bool(use_aot_forward)
        self._cache_capacity = int(cache_capacity)

        self._maybe_implicit = maybe_implicit or (lambda f: f)

        self._model_step_fn = self._build_model_step_fn(
            kv_pages_template=kv_pages_template,
            graphstate_template=graphstate_template,
            graphother_template=graphother_template,
        )
        self._cache: OrderedDict[tuple[int, int, str, str], tp.Any] = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cache_put(self, key: tuple[int, int, str, str], value: tp.Any) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)

    def _cache_get(self, key: tuple[int, int, str, str]) -> tp.Any:
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def cache_keys(self) -> list[tuple[int, int, str, str]]:
        return list(self._cache.keys())

    def has(self, key: tuple[int, int, str, str]) -> bool:
        return key in self._cache

    def get_compiled(self, *, num_tokens: int, padded_num_reqs: int) -> tp.Any:
        mode = "aot" if self.use_aot_forward else "jit"
        key = (int(num_tokens), int(padded_num_reqs), "model", mode)
        return self._cache_get(key)

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
        # Keep graphdef as a mutable attribute so JIT-mode wrappers can follow updates.
        self.graphdef = graphdef
        mode = "aot" if self.use_aot_forward else "jit"
        key = (int(num_tokens), int(padded_num_reqs), "model", mode)
        if key in self._cache:
            return None

        if self.use_aot_forward:
            compiled = self._model_step_fn.lower(
                *(graphdef, graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
            ).compile()
            self._cache_put(key, compiled)
            return None

        def wrapped(graphstate_, graphother_, kv_pages_, metadata_):
            return self._model_step_fn(self.graphdef, graphstate_, graphother_, kv_pages_, metadata_)

        out = wrapped(graphstate, graphother, inputs.kv_pages, inputs.batch_metadata)
        self._cache_put(key, wrapped)
        return out

    def _build_model_step_fn(
        self,
        *,
        kv_pages_template: RaggedPagesCache,
        graphstate_template: tp.Any,
        graphother_template: tp.Any,
    ) -> tp.Callable[..., ModelStepOutputs]:
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

        kv_pages_sharding = es.extract_shardings(kv_pages_template, self.mesh)

        outputs_shardings = ModelStepOutputs(
            kv_pages=es.extract_shardings(kv_pages_template, self.mesh),
            hidden_states=self._empty_sharding,
            logits=self._empty_sharding,
        )

        @ejit(
            static_argnums=(0,),
            donate_argnames=["kv_pages"],
            in_shardings=(
                es.extract_shardings(graphstate_template, self.mesh),
                es.extract_shardings(graphother_template, self.mesh),
                kv_pages_sharding,
                metadata_sharding,
            ),
            out_shardings=outputs_shardings,
        )
        @self._maybe_implicit
        def _model_step(
            graphdef, graphstate, graphother, kv_pages: RaggedPagesCache, metadata: BatchMetadata
        ) -> ModelStepOutputs:
            with self.model.mesh:
                model: "EasyDeLBaseModule" = nn.merge(graphdef, graphstate, graphother)
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

                output = model(
                    **model_inputs,
                    position_ids=position_ids,
                    past_key_values=kv_pages,
                    cache_metadata=cache_metadata,
                    apply_lm_head=False,
                    **external_inputs,
                )
                hs = output.last_hidden_state.squeeze(0)
                logits = model.apply_lm_head(hs[metadata.logits_indices])

                return ModelStepOutputs(
                    kv_pages=output.past_key_values,
                    hidden_states=hs,
                    logits=logits,
                )

        return _model_step
