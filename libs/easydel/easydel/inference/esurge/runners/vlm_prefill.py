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

"""VLM prompt-prefill helpers for the eSurge model runner.

Owns the eager/compiled vision-feature precompute that must run outside the
compiled eSurge step (mRoPE index construction and vision towers with
data-dependent control flow are not JIT/AOT friendly), the bucketed vision
JIT helpers used to precompile the vision encoder, and the host-side CPU
scratch buffers for VLM embedding overrides.

The runner accesses the model through a getter so hot weight updates
(:meth:`eSurgeRunner.update_model_weights`) are observed without re-wiring
the helper; the runner resets the cached JIT closures on weight swaps via its
``_vlm_*`` passthrough properties.
"""

from __future__ import annotations

import typing
from functools import partial

import jax
import numpy as np
import spectrax as spx
from eformer.loggings import get_logger
from jax import numpy as jnp

from ..utils import model_uses_mrope

if typing.TYPE_CHECKING:
    from .states import CachedRequestState

logger = get_logger("eSurge")


class VlmPrefillHelper:
    """Vision-language prefill precompute and vision-JIT bucketing.

    Attributes:
        model_getter: Zero-arg callable returning the runner's current model
            (kept as a getter so hot weight updates are observed).
        metadata: Paged-attention cache metadata; only
            ``data_parallel_size`` is read.
        compile_vision_encoder: Whether to precompile and use a bucketed JIT
            helper for VLM vision features.
        vision_patch_buckets: Optional raw patch-count buckets used by the
            vision precompile helper.
        max_num_batched_tokens: Per-step scheduler token budget (bucket
            derivation input).
        max_num_tokens: Largest token compile bucket (bucket derivation
            fallback).
        image_features_jit: Cached JIT closure for image features, or None.
        video_features_jit: Cached JIT closure for video features, or None.
        vision_jit_disabled: Sticky flag set when the vision JIT path failed
            and the helper fell back to eager vision precompute.
        cpu_buffers: Host-side scratch buffers keyed by ``num_tokens_static``.
    """

    def __init__(
        self,
        *,
        model_getter: typing.Callable[[], typing.Any],
        metadata: typing.Any,
        compile_vision_encoder: bool,
        vision_patch_buckets: list[int] | None,
        max_num_batched_tokens: int,
        max_num_tokens: int,
    ) -> None:
        """Initialize the helper and normalize the vision knobs.

        Args:
            model_getter: Zero-arg callable returning the current model.
            metadata: Cache metadata carrying ``data_parallel_size``.
            compile_vision_encoder: Whether the bucketed vision JIT path is
                enabled.
            vision_patch_buckets: Optional explicit raw patch-count buckets.
            max_num_batched_tokens: Per-step scheduler token budget.
            max_num_tokens: Largest token compile bucket.
        """
        self.model_getter = model_getter
        self.metadata = metadata
        self.compile_vision_encoder = bool(compile_vision_encoder)
        self.vision_patch_buckets = (
            [int(bucket) for bucket in vision_patch_buckets] if vision_patch_buckets is not None else None
        )
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.max_num_tokens = int(max_num_tokens)
        self.image_features_jit: typing.Callable | None = None
        self.video_features_jit: typing.Callable | None = None
        self.vision_jit_disabled = False
        # VLM host-side scratch buffers keyed by `num_tokens_static` (avoid repeated
        # large allocations while keeping the step-function input pytree stable).
        self.cpu_buffers: dict[
            int,
            tuple[
                np.ndarray | None,  # prefill_embeds_cpu
                np.ndarray | None,  # prefill_embeds_mask_cpu
                np.ndarray | None,  # mrope_position_ids_cpu
                np.ndarray | None,  # visual_pos_masks_cpu
                list[np.ndarray] | None,  # deepstack_visual_embeds_cpu
            ],
        ] = {}

    def reset_cpu_buffers(self) -> None:
        """Drop the cached host-side scratch buffers (fresh dict)."""
        self.cpu_buffers = {}

    def get_cpu_buffers(
        self,
        *,
        num_tokens_static: int,
        uses_mrope_model: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[np.ndarray] | None]:
        """Get or create cached CPU buffers for VLM prefill data.

        Retrieves pre-allocated CPU buffers for VLM embedding overrides,
        keyed by num_tokens_static to avoid repeated allocations while
        keeping the step function input shape stable.

        Args:
            num_tokens_static: Token count bucket size for buffer sizing.
            uses_mrope_model: Whether the model uses mRoPE positions.

        Returns:
            Tuple of (prefill_embeds_cpu, prefill_embeds_mask_cpu,
            mrope_position_ids_cpu, visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu) where optional arrays are
            None if not applicable.

        Side Effects:
            - Creates and caches buffers in self.cpu_buffers.
            - Clears masks/position buffers each call (filled by caller).
        """
        num_tokens_static = int(num_tokens_static)
        cached = self.cpu_buffers.get(num_tokens_static)
        if cached is None:
            model = self.model_getter()
            hidden_size = int(getattr(model.config.get_text_config(), "hidden_size", 0) or 1)

            prefill_embeds_cpu = np.zeros((num_tokens_static, hidden_size), dtype=np.float16)
            prefill_embeds_mask_cpu = np.zeros((num_tokens_static,), dtype=bool)

            mrope_position_ids_cpu = None
            visual_pos_masks_cpu = None
            deepstack_visual_embeds_cpu = None
            if uses_mrope_model:
                mrope_position_ids_cpu = np.zeros((3, num_tokens_static), dtype=np.int32)
                deepstack_indexes = getattr(
                    getattr(model.config, "vision_config", None),
                    "deepstack_visual_indexes",
                    None,
                )
                deepstack_layers = len(deepstack_indexes) if deepstack_indexes else 0
                if deepstack_layers:
                    visual_pos_masks_cpu = np.zeros((num_tokens_static,), dtype=bool)
                    deepstack_visual_embeds_cpu = [
                        np.zeros((num_tokens_static, hidden_size), dtype=np.float16) for _ in range(deepstack_layers)
                    ]

            cached = (
                prefill_embeds_cpu,
                prefill_embeds_mask_cpu,
                mrope_position_ids_cpu,
                visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu,
            )
            self.cpu_buffers[num_tokens_static] = cached

        (
            prefill_embeds_cpu,
            prefill_embeds_mask_cpu,
            mrope_position_ids_cpu,
            visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu,
        ) = cached

        # Clear masks/position ids each step; large embed buffers are overwritten only
        # for the masked regions, and ignored otherwise.
        prefill_embeds_mask_cpu.fill(False)
        if mrope_position_ids_cpu is not None:
            mrope_position_ids_cpu.fill(0)
        if visual_pos_masks_cpu is not None:
            visual_pos_masks_cpu.fill(False)

        return cached

    @staticmethod
    def static_grid_thw(grid: np.ndarray | jax.Array | tuple | list | None) -> tuple[tuple[int, int, int], ...] | None:
        """Convert a processor grid into a hashable static JIT argument."""
        if grid is None:
            return None
        arr = np.asarray(grid, dtype=np.int64)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                return None
            arr = arr.reshape(1, 3)
        if arr.ndim != 2 or arr.shape[-1] != 3:
            return None
        return tuple(tuple(int(v) for v in row) for row in arr.tolist())

    @staticmethod
    def max_grid_size(grid: tuple[tuple[int, int, int], ...] | None) -> int | None:
        """Return the static max spatial grid size used by Qwen-style vision towers."""
        if not grid:
            return None
        return max(max(int(row[1]), int(row[2])) for row in grid)

    @staticmethod
    def block_tree_until_ready(value: typing.Any) -> None:
        """Block until every device-array leaf in ``value`` is ready."""
        for leaf in jax.tree_util.tree_leaves(value):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    def uses_deepstack_visuals(self) -> bool:
        """Whether the model's vision config declares deepstack visual layers."""
        model = self.model_getter()
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        return bool(deepstack_indexes)

    def vision_patch_input_dim(self) -> int | None:
        """Return the flattened per-patch input dim, or None when unknown."""
        model = self.model_getter()
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        if vision_config is None:
            return None
        try:
            in_channels = int(vision_config.in_channels)
            temporal_patch_size = int(vision_config.temporal_patch_size)
            patch_size = int(vision_config.patch_size)
        except (TypeError, ValueError, AttributeError):
            return None
        return in_channels * temporal_patch_size * patch_size * patch_size

    def vision_spatial_merge_size(self) -> int:
        """Return the vision tower's spatial merge size (>= 1)."""
        model = self.model_getter()
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        try:
            return max(1, int(getattr(vision_config, "spatial_merge_size", 1) or 1))
        except (TypeError, ValueError):
            return 1

    def vision_dtype(self) -> jnp.dtype:
        """Return the dtype the vision tower expects for pixel inputs."""
        model = self.model_getter()
        base_model = getattr(model, "base_model", model)
        visual = getattr(base_model, "visual", getattr(model, "visual", None))
        if visual is not None and callable(getattr(visual, "get_dtype", None)):
            dtype = visual.get_dtype()
        else:
            dtype = getattr(model, "dtype", jnp.bfloat16)
        try:
            return jnp.dtype(dtype)
        except TypeError:
            return jnp.bfloat16

    def vision_grid_for_patch_bucket(self, raw_patches: int) -> tuple[int, tuple[int, int, int]]:
        """Round a raw patch count up to a merge-aligned count and derive a grid."""
        spatial_merge = self.vision_spatial_merge_size()
        merge_unit = spatial_merge * spatial_merge
        raw_patches = max(merge_unit, int(raw_patches))
        if raw_patches % merge_unit:
            raw_patches = ((raw_patches + merge_unit - 1) // merge_unit) * merge_unit

        merged_tokens = max(1, raw_patches // merge_unit)
        h_merged = max(1, int(np.sqrt(merged_tokens)))
        while h_merged > 1 and merged_tokens % h_merged:
            h_merged -= 1
        w_merged = max(1, merged_tokens // h_merged)
        return raw_patches, (1, h_merged * spatial_merge, w_merged * spatial_merge)

    def vision_patch_buckets_for_compile(self, *, max_num_batched_tokens: int | None) -> list[int]:
        """Derive the raw patch-count buckets to precompile the vision tower for."""
        if self.vision_patch_buckets is not None:
            return sorted({max(1, int(bucket)) for bucket in self.vision_patch_buckets})

        spatial_merge = self.vision_spatial_merge_size()
        merge_unit = spatial_merge * spatial_merge
        token_budget = max(1, int(max_num_batched_tokens or self.max_num_batched_tokens or self.max_num_tokens))
        max_patches = max(16, token_budget // merge_unit)
        min_shift = 4
        max_shift = max(min_shift, (max_patches - 1).bit_length())
        return [1 << shift for shift in range(min_shift, max_shift + 1)]

    def get_image_features_jit(self) -> typing.Callable:
        """Return (building if needed) the bucketed image-features JIT closure.

        The model's weights are threaded through the jit as a ``spx.State``
        INPUT (``spx.export`` outside, ``spx.bind`` inside) instead of being
        reached through a closure. A closure would bake the full vision tower
        into the compiled program as captured constants — for real VLMs that is
        gigabytes per vision bucket, which slows compilation, pushes persistent
        compilation-cache entries past protobuf's 2 GB parse limit (poisoning
        the cache for every later process), and duplicates weights in HBM.
        ``spx.export`` only traverses the module pytree and references the
        already-resident device buffers; it copies nothing.
        """
        if self.image_features_jit is None:
            model = self.model_getter()
            graphdef, _ = spx.export(model)

            @partial(jax.jit, static_argnames=("image_grid_thw", "image_max_grid_size"))
            def _image_features(
                state: spx.State,
                pixel_values: jax.Array,
                *,
                image_grid_thw: tuple[tuple[int, int, int], ...],
                image_max_grid_size: int | None,
            ):
                bound = spx.bind(graphdef, state)
                image_embeds_tuple, deepstack_image_embeds = bound.get_image_features(
                    pixel_values,
                    image_grid_thw,
                    image_max_grid_size,
                )
                deepstack = () if deepstack_image_embeds is None else tuple(deepstack_image_embeds)
                return jnp.concatenate(image_embeds_tuple, axis=0), deepstack

            def _call(
                pixel_values: jax.Array,
                *,
                image_grid_thw: tuple[tuple[int, int, int], ...],
                image_max_grid_size: int | None,
            ):
                _, state = spx.export(self.model_getter())
                return _image_features(
                    state,
                    pixel_values,
                    image_grid_thw=image_grid_thw,
                    image_max_grid_size=image_max_grid_size,
                )

            _call._inner_jit = _image_features
            self.image_features_jit = _call
        return self.image_features_jit

    def get_video_features_jit(self) -> typing.Callable:
        """Return (building if needed) the bucketed video-features JIT closure.

        Weights are threaded as a ``spx.State`` jit input rather than captured
        via closure — see :meth:`get_image_features_jit` for why.
        """
        if self.video_features_jit is None:
            model = self.model_getter()
            graphdef, _ = spx.export(model)

            @partial(jax.jit, static_argnames=("video_grid_thw", "video_max_grid_size"))
            def _video_features(
                state: spx.State,
                pixel_values_videos: jax.Array,
                *,
                video_grid_thw: tuple[tuple[int, int, int], ...],
                video_max_grid_size: int | None,
            ):
                bound = spx.bind(graphdef, state)
                video_embeds_tuple, deepstack_video_embeds = bound.get_video_features(
                    pixel_values_videos,
                    video_grid_thw,
                    video_max_grid_size,
                )
                deepstack = () if deepstack_video_embeds is None else tuple(deepstack_video_embeds)
                return jnp.concatenate(video_embeds_tuple, axis=0), deepstack

            def _call(
                pixel_values_videos: jax.Array,
                *,
                video_grid_thw: tuple[tuple[int, int, int], ...],
                video_max_grid_size: int | None,
            ):
                _, state = spx.export(self.model_getter())
                return _video_features(
                    state,
                    pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    video_max_grid_size=video_max_grid_size,
                )

            _call._inner_jit = _video_features
            self.video_features_jit = _call
        return self.video_features_jit

    def compute_embedding_with_info_single_pass(
        self,
        input_ids: jax.Array,
        embed_kwargs: dict[str, typing.Any],
    ) -> tuple[jax.Array, typing.Any]:
        """Compute VLM embeddings while avoiding wrapper-level duplicate vision work."""
        model = self.model_getter()
        base_model = getattr(model, "base_model", None)
        base_fn = getattr(base_model, "compute_embedding_with_info", None) if base_model is not None else None
        if callable(base_fn):
            try:
                return base_fn(input_ids, **embed_kwargs)
            except TypeError:
                pass
        return model.compute_embedding_with_info(input_ids, **embed_kwargs)

    def compiled_vision_embed_kwargs(self, req_state: CachedRequestState) -> dict[str, typing.Any] | None:
        """Build embed kwargs from the compiled vision path, or None to go eager."""
        if (
            not self.compile_vision_encoder
            or self.vision_jit_disabled
            or self.uses_deepstack_visuals()
            or self.model_getter() is None
        ):
            return None
        if int(getattr(self.metadata, "data_parallel_size", 1) or 1) > 1:
            return None

        embed_kwargs: dict[str, typing.Any] = {}
        try:
            if req_state.pixel_values is not None:
                image_grid = self.static_grid_thw(req_state.image_grid_thw)
                if image_grid is None:
                    return None
                image_embeds, deepstack_image_embeds = self.get_image_features_jit()(
                    jnp.asarray(req_state.pixel_values),
                    image_grid_thw=image_grid,
                    image_max_grid_size=self.max_grid_size(image_grid),
                )
                if deepstack_image_embeds:
                    return None
                embed_kwargs["image_embeds"] = image_embeds
                embed_kwargs["image_grid_thw"] = image_grid

            if req_state.pixel_values_videos is not None:
                video_grid = self.static_grid_thw(req_state.video_grid_thw)
                if video_grid is None:
                    return None
                video_embeds, deepstack_video_embeds = self.get_video_features_jit()(
                    jnp.asarray(req_state.pixel_values_videos),
                    video_grid_thw=video_grid,
                    video_max_grid_size=self.max_grid_size(video_grid),
                )
                if deepstack_video_embeds:
                    return None
                embed_kwargs["video_embeds"] = video_embeds
                embed_kwargs["video_grid_thw"] = video_grid
        except Exception as exc:
            self.vision_jit_disabled = True
            logger.warning(
                "VLM vision JIT helper failed; falling back to eager vision precompute: %s",
                exc,
            )
            return None

        return embed_kwargs or None

    def compute_prefill_with_compiled_vision(
        self,
        req_state: CachedRequestState,
        input_ids: jax.Array,
        attention_mask: jax.Array,
    ) -> tuple[jax.Array, typing.Any] | None:
        """Run the prefill embedding pass with compiled vision features, if usable."""
        vision_kwargs = self.compiled_vision_embed_kwargs(req_state)
        if vision_kwargs is None:
            return None
        embed_kwargs: dict[str, typing.Any] = {"attention_mask": attention_mask}
        embed_kwargs.update(vision_kwargs)
        return self.compute_embedding_with_info_single_pass(input_ids, embed_kwargs)

    def precompile_vision_helpers(self, *, max_num_batched_tokens: int | None) -> None:
        """Precompile the bucketed vision-features JIT closure for every bucket."""
        model = self.model_getter()
        if (
            not self.compile_vision_encoder
            or self.vision_jit_disabled
            or self.uses_deepstack_visuals()
            or model is None
            or not callable(getattr(model, "get_image_features", None))
        ):
            return
        if int(getattr(self.metadata, "data_parallel_size", 1) or 1) > 1:
            return

        patch_input_dim = self.vision_patch_input_dim()
        if patch_input_dim is None:
            return

        buckets = self.vision_patch_buckets_for_compile(max_num_batched_tokens=max_num_batched_tokens)
        if not buckets:
            return

        image_features = self.get_image_features_jit()
        dtype = self.vision_dtype()
        logger.info("Precompiling VLM vision feature buckets: %s raw patches", buckets)
        for bucket in buckets:
            raw_patches, grid = self.vision_grid_for_patch_bucket(bucket)
            dummy_pixels = jnp.ones((raw_patches, patch_input_dim), dtype=dtype)
            try:
                out = image_features(
                    dummy_pixels,
                    image_grid_thw=(grid,),
                    image_max_grid_size=self.max_grid_size((grid,)),
                )
                self.block_tree_until_ready(out)
            except Exception as exc:
                self.vision_jit_disabled = True
                logger.warning(
                    "VLM vision JIT precompile failed at raw_patches=%d; falling back to eager vision precompute: %s",
                    raw_patches,
                    exc,
                )
                return
        logger.info("VLM vision feature precompilation finished")

    def precompute_prefill(self, req_state: CachedRequestState) -> None:
        """Precompute prompt embeddings (+ optional mRoPE indices) for VLM requests.

        Some VLM base models compute mRoPE indices via NumPy/data-dependent control-flow
        which is not compatible with JIT/AOT inside the compiled eSurge step. We run
        those parts eagerly here and store host-side arrays for later reuse.
        """
        if req_state.vision_processed:
            return

        uses_mrope = model_uses_mrope(self.model_getter())

        # If raw vision inputs are missing but the request is marked as "has_vision"
        # (e.g. only cached mm_features remain), skip precompute and treat it as processed.
        if req_state.pixel_values is None and req_state.pixel_values_videos is None:
            req_state._vision_processed = True
            return

        if req_state.prefill_inputs_embeds is not None and (
            not uses_mrope or req_state.prefill_position_ids is not None
        ):
            req_state.clear_vision_data()
            return

        prompt_ids = np.asarray(req_state.prompt_token_ids, dtype=np.int32)[None, :]
        input_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
        attention_mask = jnp.ones(input_ids.shape, dtype=jnp.int32)

        try:
            embed_kwargs: dict[str, typing.Any] = {"attention_mask": attention_mask}
            if req_state.pixel_values is not None:
                embed_kwargs["pixel_values"] = req_state.pixel_values
                if req_state.image_grid_thw is not None:
                    embed_kwargs["image_grid_thw"] = req_state.image_grid_thw
            if req_state.pixel_values_videos is not None:
                embed_kwargs["pixel_values_videos"] = req_state.pixel_values_videos
                if req_state.video_grid_thw is not None:
                    embed_kwargs["video_grid_thw"] = req_state.video_grid_thw

            compiled_result = self.compute_prefill_with_compiled_vision(
                req_state=req_state,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if compiled_result is None:
                inputs_embeds, info = self.compute_embedding_with_info_single_pass(input_ids, embed_kwargs)
            else:
                inputs_embeds, info = compiled_result
        except Exception as exc:
            # A vision-encode failure here is NOT silently recoverable. If we
            # swallow it and return, the compiled step runs against the raw
            # prompt token ids (image/video placeholder tokens are never
            # replaced by visual embeddings), which produces coherent-looking
            # but semantically garbage output. Fail loudly so the request is
            # rejected instead of served corrupted generations.
            logger.error(
                "VLM vision encode failed for req_id=%s; rejecting request to avoid serving garbage output. Error: %s",
                req_state.req_id,
                exc,
            )
            raise RuntimeError(f"VLM vision encode failed for req_id={req_state.req_id}: {exc}") from exc

        # Store host-side views (keeps compiled step free of vision preprocessing).
        embeds_host = np.asarray(jax.device_get(inputs_embeds))
        req_state.prefill_inputs_embeds = embeds_host[0]

        if getattr(info, "position_ids", None) is not None:
            pos_host = np.asarray(jax.device_get(info.position_ids))
            if pos_host.ndim == 3:
                pos_host = pos_host[:, 0, :]
            req_state.prefill_position_ids = pos_host.astype(np.int32, copy=False)

        if getattr(info, "rope_deltas", None) is not None:
            req_state.prefill_rope_deltas = np.asarray(jax.device_get(info.rope_deltas)).astype(np.int32, copy=False)

        if getattr(info, "visual_pos_masks", None) is not None:
            mask_host = np.asarray(jax.device_get(info.visual_pos_masks))
            if mask_host.ndim == 2:
                mask_host = mask_host[0]
            req_state.prefill_visual_pos_masks = mask_host.astype(bool, copy=False)

        deepstack_visual_embeds = getattr(info, "deepstack_visual_embeds", None)
        if deepstack_visual_embeds is not None:
            ds_list = []
            for arr in deepstack_visual_embeds:
                ds_list.append(np.asarray(jax.device_get(arr)))
            req_state.prefill_deepstack_visual_embeds = ds_list

        # Raw vision tensors are no longer needed once embeddings are cached.
        req_state.clear_vision_data()
