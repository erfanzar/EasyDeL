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

import inspect
import typing
import uuid
import weakref

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from easydel.utils.graph_jit import module_jit

from ..multimodal.cache import VisionEncoderCache
from ..multimodal.side_inputs import model_needs_token_types, prefill_embeds_dtype
from ..multimodal.types import compute_content_hash
from ..utils import model_uses_mrope

if typing.TYPE_CHECKING:
    from .states import CachedRequestState

logger = get_logger("eSurge")


def _accepts_keyword(fn: typing.Callable, name: str) -> bool:
    """Whether ``fn`` takes ``name`` as a keyword (explicitly or through ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return params[name].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _declares_parameter(fn: typing.Callable | None, name: str) -> bool:
    """Whether ``fn`` names ``name`` explicitly (``**kwargs`` does not count: it may silently drop the value)."""
    if fn is None:
        return False
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _call_visual_features(fn: typing.Callable, pixels: jax.Array, grid, max_grid_size, *, prefix: str):
    """Call a family's ``get_{image,video}_features`` and normalize its result.

    Families disagree on the contract: Qwen3-VL / Qwen3.5 / Qwen4-Exp take ``*_max_grid_size`` and return
    ``(embeds_tuple, deepstack)``; most others (Qwen2-VL, GLM-4V, HunYuan-VL, MiniMax, ...) take only the grid and
    return one merged ``(tokens, hidden)`` array. Grid and max size are passed by keyword, the max size only when the
    method accepts it, so a two-argument signature is never handed a third positional argument.

    Returns:
        ``(merged_embeds, deepstack_tuple)``.
    """
    kwargs = {f"{prefix}_grid_thw": grid} if _accepts_keyword(fn, f"{prefix}_grid_thw") else {}
    if max_grid_size is not None and _accepts_keyword(fn, f"{prefix}_max_grid_size"):
        kwargs[f"{prefix}_max_grid_size"] = max_grid_size
    result = fn(pixels, **kwargs)
    # ``(embeds_tuple, deepstack)`` pairs are told apart from a per-image tuple of arrays (Qwen2-VL / GLM-4V split
    # by image) by their first element, which is itself a sequence of arrays rather than an array.
    if isinstance(result, (tuple, list)) and len(result) == 2 and isinstance(result[0], (tuple, list)):
        embeds, deepstack = result
    else:
        embeds, deepstack = result, None
    if isinstance(embeds, (tuple, list)):
        embeds = jnp.concatenate(tuple(embeds), axis=0)
    elif embeds.ndim > 2:
        embeds = embeds.reshape(-1, embeds.shape[-1])
    return embeds, () if deepstack is None else tuple(deepstack)


def _image_features_body(
    model: typing.Any,
    pixel_values: jax.Array,
    *,
    image_grid_thw: tuple[tuple[int, int, int], ...],
    image_max_grid_size: int | None,
):
    """Traced body: vision tower -> concatenated image embeds (+ deepstack)."""
    return _call_visual_features(
        model.get_image_features, pixel_values, image_grid_thw, image_max_grid_size, prefix="image"
    )


def _video_features_body(
    model: typing.Any,
    pixel_values_videos: jax.Array,
    *,
    video_grid_thw: tuple[tuple[int, int, int], ...],
    video_max_grid_size: int | None,
):
    """Traced body: vision tower -> concatenated video embeds (+ deepstack)."""
    return _call_visual_features(
        model.get_video_features, pixel_values_videos, video_grid_thw, video_max_grid_size, prefix="video"
    )


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
        vision_cache: VisionEncoderCache | None = None,
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
            vision_cache: Optional shared bounded cache for encoder outputs. Pass
                the multimodal manager's cache to share capacity, clearing and statistics.
        """
        self.model_getter = model_getter
        self.vision_cache = vision_cache
        self._vision_cache_model_ref = None
        self._vision_cache_namespace = uuid.uuid4().hex
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

    def reset_vision_cache(self) -> None:
        """Invalidate encoder results after an in-place weight update without clearing other cache users."""
        self._vision_cache_namespace = uuid.uuid4().hex
        model = self.model_getter()
        self._vision_cache_model_ref = weakref.ref(model) if model is not None else None

    def _vision_cache_identity(self) -> str:
        """Use a fresh namespace on model replacement without retaining released model weights."""
        previous = self._vision_cache_model_ref() if self._vision_cache_model_ref is not None else None
        if self.model_getter() is not previous:
            self.reset_vision_cache()
        return self._vision_cache_namespace

    def _encoded_vision_features(self, pixels, grid, *, modality: str, encode: typing.Callable):
        """Cache actual encoder outputs, including every deepstack leaf, not prompt-dependent merged embeddings."""
        key = None
        if self.vision_cache is not None:
            key = (
                f"encoder:{self._vision_cache_identity()}:{modality}:"
                f"{compute_content_hash(pixels)}:{compute_content_hash(grid) if grid is not None else 'none'}"
            )
            cached = self.vision_cache.get(key)
            if cached is not None:
                return cached
        result = encode()
        if key is not None:
            # Host arrays keep a bounded cache from pinning device memory between requests.
            result = (np.asarray(jax.device_get(result[0])), tuple(np.asarray(jax.device_get(x)) for x in result[1]))
            self.vision_cache.put(key, result)
        return result

    def _attach_feature_embeddings(self, req_state, modality: str, embeddings, deepstack=()) -> None:
        """Retain correctly split encoder rows and auxiliary layers when the split is known."""
        features = [f for f in getattr(req_state, "mm_features", ()) if f.modality == modality]
        if not features:
            return
        if len(features) == 1:
            lengths = [len(embeddings)]
        elif all(getattr(f, "placeholder_range", None) is not None for f in features):
            lengths = [f.placeholder_range.length for f in features]
        elif all(f.grid_thw is not None for f in features):
            merge = self.vision_spatial_merge_size() ** 2
            lengths = [int(np.prod(f.grid_thw)) // merge for f in features]
        else:
            return
        if sum(lengths) != len(embeddings):
            return
        offset = 0
        for feature, length in zip(features, lengths, strict=True):
            feature.set_cached_embeddings(embeddings[offset : offset + length])
            feature._esurge_deepstack_embeddings = tuple(layer[offset : offset + length] for layer in deepstack)
            feature._esurge_vision_cache_identity = self._vision_cache_identity()
            feature._esurge_vision_grid_thw = None if feature.grid_thw is None else np.array(feature.grid_thw, copy=True)
            offset += length

    def _feature_embeddings(self, req_state, modality: str):
        """Consume request-owned features across turns, independently of bounded shared-cache eviction."""
        features = [f for f in getattr(req_state, "mm_features", ()) if f.modality == modality]
        if not features or not all(f.has_cached_embeddings for f in features):
            return None
        for feature in features:
            self._validate_cached_feature(feature)
        embeddings = [f.cached_embeddings for f in features]
        return embeddings[0] if len(embeddings) == 1 else jnp.concatenate(embeddings, axis=0)

    def _validate_cached_feature(self, feature) -> None:
        """Reject request-owned rows retained for other weights or a different media layout."""
        identity = self._vision_cache_identity()
        if getattr(feature, "_esurge_vision_cache_identity", identity) != identity:
            raise ValueError("Cached vision embeddings belong to an earlier model; resubmit the raw media.")
        cached_grid = getattr(feature, "_esurge_vision_grid_thw", None)
        if feature.grid_thw is not None and cached_grid is not None:
            if not np.array_equal(np.asarray(feature.grid_thw).reshape(-1, 3), np.asarray(cached_grid).reshape(-1, 3)):
                raise ValueError("Cached vision embeddings have a different grid; resubmit the raw media.")

    def _mixed_feature_embeddings(self, features, modality: str, encode_fn: typing.Callable):
        """Assemble cached-only and raw media of one modality in feature order.

        Aggregate request pixels/grids cover only raw media. Encoding that aggregate
        and assigning it to every feature drops cached rows or poisons the per-feature
        cache, so this path encodes raw features separately before concatenation.
        """
        rows = []
        layers = []
        grids = []
        expected_layers = (
            len(self.model_getter().config.vision_config.deepstack_visual_indexes)
            if self.uses_deepstack_visuals()
            else 0
        )
        for feature in features:
            grid = feature.grid_thw
            if grid is None:
                grid = getattr(feature, "_esurge_vision_grid_thw", None)
            if grid is not None:
                grid = np.asarray(grid)
                # Features accept a single [3] triple; encoders expect [N, 3].
                if grid.ndim == 1 and grid.shape[0] == 3:
                    grid = grid.reshape(1, 3)
            if feature.pixel_values is None:
                self._validate_cached_feature(feature)
                embeddings = feature.cached_embeddings
                if embeddings is None:
                    raise ValueError(f"Missing raw {modality} data and cached embeddings.")
                deepstack = getattr(feature, "_esurge_deepstack_embeddings", ())
            else:
                pixels = feature.pixel_values
                embeddings, deepstack = self._encoded_vision_features(
                    pixels,
                    grid,
                    modality=modality,
                    encode=lambda pv=pixels, g=grid: _call_visual_features(
                        encode_fn, jnp.asarray(pv), g, self.max_grid_size(self.static_grid_thw(g)), prefix=modality
                    ),
                )
            if len(deepstack) != expected_layers or any(layer.shape != embeddings.shape for layer in deepstack):
                raise ValueError(
                    f"Cached {modality} embeddings are missing DeepStack residuals; resubmit the raw media."
                )
            rows.append(embeddings)
            layers.append(deepstack)
            grids.append(grid)
        if any(grid is not None for grid in grids) and not all(grid is not None for grid in grids):
            raise ValueError(f"Missing grid metadata for cached {modality} embeddings; resubmit the raw media.")
        # Validate all rows before updating any feature, including when one cached
        # image is stale or an encoder returned incomplete auxiliary layers.
        for feature, embeddings, deepstack, grid in zip(features, rows, layers, grids, strict=True):
            feature.set_cached_embeddings(embeddings)
            feature._esurge_deepstack_embeddings = tuple(deepstack)
            feature._esurge_vision_cache_identity = self._vision_cache_identity()
            feature._esurge_vision_grid_thw = None if grid is None else np.array(grid, copy=True)
        grid = np.concatenate([np.asarray(g).reshape(-1, 3) for g in grids], axis=0) if grids[0] is not None else None
        return (
            jnp.concatenate(rows, axis=0),
            tuple(jnp.concatenate([value[i] for value in layers], axis=0) for i in range(expected_layers)),
            grid,
        )

    def _accepts_deepstack(self, modality: str) -> bool:
        """Opt into auxiliary caching only when the actual embedding-info entry point declares it."""
        model = self.model_getter()
        owner = getattr(model, "base_model", None)
        fn = getattr(owner, "compute_embedding_with_info", None)
        if not callable(fn):
            fn = getattr(model, "compute_embedding_with_info", None)
        return _declares_parameter(fn, f"deepstack_{modality}_embeds")

    def _feature_deepstack(self, req_state, modality: str):
        """Reassemble retained per-media residuals without caching prompt-dependent masks."""
        features = [f for f in getattr(req_state, "mm_features", ()) if f.modality == modality]
        layers = [getattr(f, "_esurge_deepstack_embeddings", ()) for f in features]
        expected = len(self.model_getter().config.vision_config.deepstack_visual_indexes)
        if not layers or any(len(value) != expected for value in layers):
            raise ValueError(f"Cached {modality} embeddings are missing DeepStack residuals; resubmit the raw media.")
        return tuple(jnp.concatenate(tuple(value[i] for value in layers), axis=0) for i in range(expected))

    def eager_vision_embed_kwargs(self, req_state: CachedRequestState) -> dict[str, typing.Any] | None:
        """Reuse encoder outputs through declared inputs, including all DeepStack residuals.

        Masks, merged prompt embeddings and position ids are rebuilt by the model for
        each prompt. Families without an explicit auxiliary injection contract retain
        their native path rather than silently dropping their visual layers.
        """
        model = self.model_getter()
        owner = getattr(model, "base_model", model)
        embed_fn = getattr(owner, "compute_embedding", None)
        result = {}
        for modality, pixel_name, grid_name in (
            ("image", "pixel_values", "image_grid_thw"),
            ("video", "pixel_values_videos", "video_grid_thw"),
        ):
            pixels = getattr(req_state, pixel_name, None)
            features = [f for f in getattr(req_state, "mm_features", ()) if f.modality == modality]
            if pixels is None and not features:
                continue
            if self.uses_deepstack_visuals() and not self._accepts_deepstack(modality):
                return None
            parameter = next(
                (p for p in (f"{modality}_embeds", f"{modality}_features") if _declares_parameter(embed_fn, p)),
                None,
            )
            if parameter is None:
                return None
            grid = getattr(req_state, grid_name, None)
            feature_grids = [
                f.grid_thw if f.grid_thw is not None else getattr(f, "_esurge_vision_grid_thw", None) for f in features
            ]
            if grid is None and feature_grids and all(g is not None for g in feature_grids):
                grid = np.concatenate([np.asarray(g).reshape(-1, 3) for g in feature_grids], axis=0)
            mixed_features = (
                features
                and any(f.pixel_values is not None for f in features)
                and (pixels is None or any(f.pixel_values is None for f in features))
            )
            if mixed_features:
                encode_fn = getattr(model, f"get_{modality}_features", None)
                if not callable(encode_fn):
                    return None
                embeddings, deepstack, grid = self._mixed_feature_embeddings(features, modality, encode_fn)
            elif pixels is None:
                embeddings = self._feature_embeddings(req_state, modality)
                if embeddings is None:
                    raise ValueError(f"Missing raw {modality} data and cached embeddings.")
                deepstack = self._feature_deepstack(req_state, modality) if self.uses_deepstack_visuals() else ()
            else:
                if self.vision_cache is None:
                    return None
                encode_fn = getattr(model, f"get_{modality}_features", None)
                if not callable(encode_fn):
                    return None
                embeddings, deepstack = self._encoded_vision_features(
                    pixels,
                    grid,
                    modality=modality,
                    encode=lambda fn=encode_fn, pv=pixels, g=grid, m=modality: _call_visual_features(
                        fn, jnp.asarray(pv), g, self.max_grid_size(self.static_grid_thw(g)), prefix=m
                    ),
                )
                if deepstack and not self._accepts_deepstack(modality):
                    return None
                self._attach_feature_embeddings(req_state, modality, embeddings, deepstack)
            if deepstack:
                result[f"deepstack_{modality}_embeds"] = list(deepstack)
            result[parameter] = embeddings
            if grid is not None:
                result[grid_name] = grid
            # Keep raw leaves too: some families use their presence when computing position ids.
            if pixels is not None:
                result[pixel_name] = pixels
        return result or None

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

            prefill_embeds_cpu = np.zeros((num_tokens_static, hidden_size), dtype=prefill_embeds_dtype(model))
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
        INPUT via :func:`module_jit` (``spx.export`` outside, ``spx.bind``
        inside) instead of being reached through a closure. A closure would bake
        the full vision tower into the compiled program as captured constants —
        for real VLMs that is gigabytes per vision bucket, which slows
        compilation, pushes persistent compilation-cache entries past protobuf's
        2 GB parse limit (poisoning the cache for every later process), and
        duplicates weights in HBM.
        """
        if self.image_features_jit is None:
            self.image_features_jit = module_jit(
                _image_features_body,
                self.model_getter,
                static_argnames=("image_grid_thw", "image_max_grid_size"),
            )
        return self.image_features_jit

    def get_video_features_jit(self) -> typing.Callable:
        """Return (building if needed) the bucketed video-features JIT closure.

        Weights are threaded as a ``spx.State`` jit input via :func:`module_jit`
        rather than captured via closure — see :meth:`get_image_features_jit`.
        """
        if self.video_features_jit is None:
            self.video_features_jit = module_jit(
                _video_features_body,
                self.model_getter,
                static_argnames=("video_grid_thw", "video_max_grid_size"),
            )
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
        if not self.compile_vision_encoder or self.vision_jit_disabled or self.model_getter() is None:
            return None
        if int(getattr(self.metadata, "data_parallel_size", 1) or 1) > 1:
            return None
        # Aggregate pixels contain only raw media: let the eager assembler interleave
        # cached-only rows even when there are raw pixels of the same modality.
        for feature in getattr(req_state, "mm_features", ()):
            pixels = req_state.pixel_values if feature.modality == "image" else req_state.pixel_values_videos
            if pixels is None or feature.pixel_values is None:
                return None

        # The compiled path hands precomputed features over as ``image_embeds`` / ``video_embeds``. A family whose
        # ``compute_embedding`` names them differently (``image_features`` in MiniMax / LLaVA / PaliGemma / ...)
        # would swallow the keyword through ``**kwargs`` and silently drop the images, so it stays eager.
        model = self.model_getter()
        embed_owner = getattr(model, "base_model", None)
        if embed_owner is None:
            embed_owner = model
        embed_fn = getattr(embed_owner, "compute_embedding", None)
        if req_state.pixel_values is not None and not _declares_parameter(embed_fn, "image_embeds"):
            return None
        if req_state.pixel_values_videos is not None and not _declares_parameter(embed_fn, "video_embeds"):
            return None
        if self.uses_deepstack_visuals():
            for modality, pixels in (("image", req_state.pixel_values), ("video", req_state.pixel_values_videos)):
                if pixels is not None and not self._accepts_deepstack(modality):
                    return None

        embed_kwargs: dict[str, typing.Any] = {}
        try:
            if req_state.pixel_values is not None:
                image_grid = self.static_grid_thw(req_state.image_grid_thw)
                if image_grid is None:
                    return None
                image_embeds, deepstack_image_embeds = self._encoded_vision_features(
                    req_state.pixel_values,
                    req_state.image_grid_thw,
                    modality="image",
                    encode=lambda: self.get_image_features_jit()(
                        jnp.asarray(req_state.pixel_values),
                        image_grid_thw=image_grid,
                        image_max_grid_size=self.max_grid_size(image_grid),
                    ),
                )
                if deepstack_image_embeds:
                    if not self._accepts_deepstack("image"):
                        return None
                    embed_kwargs["deepstack_image_embeds"] = list(deepstack_image_embeds)
                self._attach_feature_embeddings(req_state, "image", image_embeds, deepstack_image_embeds)
                embed_kwargs["image_embeds"] = image_embeds
                embed_kwargs["image_grid_thw"] = image_grid

            if req_state.pixel_values_videos is not None:
                video_grid = self.static_grid_thw(req_state.video_grid_thw)
                if video_grid is None:
                    return None
                video_embeds, deepstack_video_embeds = self._encoded_vision_features(
                    req_state.pixel_values_videos,
                    req_state.video_grid_thw,
                    modality="video",
                    encode=lambda: self.get_video_features_jit()(
                        jnp.asarray(req_state.pixel_values_videos),
                        video_grid_thw=video_grid,
                        video_max_grid_size=self.max_grid_size(video_grid),
                    ),
                )
                if deepstack_video_embeds:
                    if not self._accepts_deepstack("video"):
                        return None
                    embed_kwargs["deepstack_video_embeds"] = list(deepstack_video_embeds)
                self._attach_feature_embeddings(req_state, "video", video_embeds, deepstack_video_embeds)
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

    def _token_type_encoder_cache(self, feature, encode: typing.Callable):
        """Provide prompt-independent aligner rows with weight-version and content-safe reuse."""
        identity = self._vision_cache_identity()
        grid = feature.grid_thw
        if grid is None:
            grid = getattr(feature, "_esurge_vision_grid_thw", None)
        if feature.pixel_values is None:
            self._validate_cached_feature(feature)
            embeddings = feature.cached_embeddings
            if embeddings is None:
                raise ValueError("Image feature has neither cached encoder rows nor raw pixels.")
        else:
            embeddings, _ = self._encoded_vision_features(
                feature.pixel_values, grid, modality=feature.modality, encode=lambda: (encode(), ())
            )
        embeddings = np.asarray(jax.device_get(embeddings))
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        feature.set_cached_embeddings(embeddings)
        feature._esurge_vision_cache_identity = identity
        feature._esurge_vision_grid_thw = None if grid is None else np.array(grid, copy=True)
        return embeddings

    def _precompute_token_type_prefill(self, req_state: CachedRequestState) -> None:
        """Prepare the model-specific token-type image prefill using its declared hook.

        The scheduler attaches image features with their placeholder ranges. This hook builds merged prompt embeddings
        and token types, then the request state carries them through the regular prefill path.

        Raises:
            RuntimeError: If encoding fails (a request must never be served with raw placeholder embeddings).
        """
        model = self.model_getter()
        compute_prefill_embeddings = getattr(type(model), "esurge_compute_prefill_embeddings", None)
        if compute_prefill_embeddings is None:
            raise RuntimeError("The model requires token-type prefill but declares no eSurge prefill hook.")
        features = list(req_state.mm_features)
        if not features:
            req_state._vision_processed = True
            return
        kwargs = {}
        if _declares_parameter(compute_prefill_embeddings, "encoder_cache_provider"):
            kwargs["encoder_cache_provider"] = self._token_type_encoder_cache
        try:
            embeds, types = compute_prefill_embeddings(model, req_state.prompt_token_ids, features, **kwargs)
        except Exception as exc:
            logger.error("Image encode failed for req_id=%s: %s", req_state.req_id, exc)
            raise RuntimeError(f"Image encode failed for req_id={req_state.req_id}: {exc}") from exc
        req_state.prefill_inputs_embeds = embeds
        req_state.token_types = types
        # The patches stay on the shared features so a request re-created after preemption can re-encode.
        req_state._vision_processed = True

    def precompute_prefill(self, req_state: CachedRequestState) -> None:
        """Precompute prompt embeddings (+ optional mRoPE indices) for VLM requests.

        Some VLM base models compute mRoPE indices via NumPy/data-dependent control-flow
        which is not compatible with JIT/AOT inside the compiled eSurge step. We run
        those parts eagerly here and store host-side arrays for later reuse.
        """
        if req_state.vision_processed:
            return

        if model_needs_token_types(self.model_getter()):
            self._precompute_token_type_prefill(req_state)
            return

        uses_mrope = model_uses_mrope(self.model_getter())

        if (
            req_state.pixel_values is None
            and req_state.pixel_values_videos is None
            and not getattr(req_state, "mm_features", ())
        ):
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
                cached_vision_kwargs = self.eager_vision_embed_kwargs(req_state)
                if cached_vision_kwargs is not None:
                    embed_kwargs.update(cached_vision_kwargs)
                elif req_state.pixel_values is None and req_state.pixel_values_videos is None:
                    raise ValueError("This model needs raw media to rebuild vision embeddings.")
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
