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

"""Multimodal processing manager for vision-language models.

This module provides the MultiModalManager class, the central component for
handling image and video preprocessing in the eSurge inference engine. It
manages resolution bucketing, integration with HuggingFace processors, and
coordination with the vision encoder cache.

The manager is designed to work with various VLM architectures:
    - GLM4V/GLM46V models (flat-patch vision towers)
    - Qwen2-VL/Qwen3-VL models (flat-patch with spatial merge)
    - Standard CLIP-based VLMs

Key Features:
    Resolution Bucketing:
        Images are resized to predefined resolution buckets to minimize JAX
        recompilation. This trades off some image quality for significantly
        improved inference throughput in production settings.

    Flat-Patch Support:
        For models like GLM and Qwen that use flattened patch representations,
        the manager handles spatiotemporal patchification with proper grid
        alignment for spatial merge operations.

    HuggingFace Integration:
        Seamlessly integrates with HuggingFace AutoProcessor instances for
        tokenization and preprocessing, with fallback paths for models that
        don't have full processor support.

    Message Parsing:
        Extracts images and videos from OpenAI-style chat messages, supporting
        various formats including PIL images, file paths, and data URLs.

Classes:
    MultiModalManager: Central manager for vision data processing and caching.

Module Constants:
    CLIP_IMAGE_MEAN: Mean values for CLIP-style image normalization.
    CLIP_IMAGE_STD: Standard deviation values for CLIP-style normalization.
    DEFAULT_RESOLUTION_BUCKETS: Default list of (H, W) resolution tuples.

Example:
    Basic usage::

        >>> from easydel.inference.esurge.multimodal import MultiModalManager
        >>> from transformers import AutoProcessor
        >>>
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
        >>> manager = MultiModalManager(processor=processor, model=model)
        >>>
        >>> # Process images with automatic resolution bucketing
        >>> pixel_values, grid_thw = manager.process_images([image1, image2])
        >>>
        >>> # Extract media from OpenAI-style messages
        >>> images, videos = manager.extract_media_from_messages(messages)
        >>>
        >>> # Tokenize with proper placeholder insertion
        >>> token_ids = manager.tokenize_multimodal(
        ...     messages, images=images, image_grid_thw=grid_thw
        ... )

See Also:
    VisionEncoderCache: Cache used internally for vision encoder outputs
    MultiModalFeature: Features created by process_images_to_features()
    BatchedMultiModalInputs: Batched outputs from multiple features
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .cache import VisionEncoderCache
from .types import MultiModalFeature

if TYPE_CHECKING:
    pass


# CLIP-style normalization constants used by GLM/Qwen vision towers.
# These values are the ImageNet-derived mean and std used by OpenAI CLIP.
CLIP_IMAGE_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
"""np.ndarray: Per-channel mean values for CLIP image normalization (RGB order)."""

CLIP_IMAGE_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
"""np.ndarray: Per-channel standard deviation values for CLIP image normalization (RGB order)."""


# Default resolution buckets for compilation efficiency.
# Images are resized to the nearest bucket to minimize JAX recompilation
# while maintaining reasonable image quality across common input sizes.
DEFAULT_RESOLUTION_BUCKETS = [
    (32, 32),
    (64, 64),
    (128, 128),
    (384, 384),
    (512, 512),
    (768, 768),
    (1024, 1024),
]
"""list[tuple[int, int]]: Default (height, width) resolution buckets for image resizing."""


class MultiModalManager:
    """Manager for multimodal (vision-language) processing.

    Handles image and video preprocessing with resolution bucketing to minimize
    JAX recompilation. Integrates with HuggingFace processors for tokenization
    and vision encoding preparation.

    Attributes:
        processor: HuggingFace processor for the vision-language model.
        resolution_buckets: List of (height, width) tuples for bucketing.
        cache: Vision encoder output cache.

    Example:
        >>> manager = MultiModalManager(processor=processor)
        >>> # Process images with automatic resolution bucketing
        >>> pixel_values, grid_thw = manager.process_images(images)
        >>> # Process OpenAI-style messages
        >>> images, videos = manager.extract_media_from_messages(messages)
    """

    def __init__(
        self,
        processor: Any | None = None,
        model: Any | None = None,
        resolution_buckets: list[tuple[int, int]] | None = None,
        cache_capacity_mb: int = 1024,
        enable_cache: bool = True,
    ):
        """Initialize MultiModalManager.

        Creates a new manager instance configured for the given processor
        and model. The manager will use resolution bucketing to minimize
        JAX recompilation and optionally cache vision encoder outputs.

        Args:
            processor (Any | None): HuggingFace processor (AutoProcessor) for
                the VLM. Used for tokenization and image preprocessing. Can be
                None if only using fallback preprocessing paths.
            model (Any | None): The VLM model instance. Used to access vision
                config for flat-patch models. Can be None if processor handles
                all preprocessing. Defaults to None.
            resolution_buckets (list[tuple[int, int]] | None): List of (H, W)
                resolution tuples for image bucketing. Images are resized to
                the bucket with total pixels closest to the original. If None,
                uses DEFAULT_RESOLUTION_BUCKETS.
            cache_capacity_mb (int): Maximum capacity for the vision encoder
                cache in megabytes. Defaults to 1024 MB (1 GB).
            enable_cache (bool): Whether to enable vision encoder output
                caching. Set to False to disable caching (e.g., for debugging
                or memory-constrained environments). Defaults to True.

        Example:
            >>> from transformers import AutoProcessor
            >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
            >>> manager = MultiModalManager(
            ...     processor=processor,
            ...     model=model,
            ...     cache_capacity_mb=2048,
            ...     enable_cache=True
            ... )
        """
        self.processor = processor
        self.model = model
        self.resolution_buckets = resolution_buckets or DEFAULT_RESOLUTION_BUCKETS
        self.cache = VisionEncoderCache(cache_capacity_mb) if enable_cache else None

    def resize_to_bucket(self, image: Image.Image) -> Image.Image:
        """Resize image to nearest resolution bucket.

        Selects the bucket with total pixels closest to the original image
        to minimize quality loss while enabling compilation reuse.

        Args:
            image: PIL Image to resize.

        Returns:
            Resized PIL Image at bucket resolution.
        """
        w, h = image.size
        original_pixels = h * w

        # Buckets are stored as (H, W).
        target_h, target_w = min(self.resolution_buckets, key=lambda b: abs(b[0] * b[1] - original_pixels))

        if (h, w) == (target_h, target_w):
            return image

        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _align_dim_to_multiple(dim: int, multiple: int) -> int:
        """Align a dimension to the nearest multiple.

        Rounds a dimension value to the nearest multiple of the given value,
        preferring to round up on ties. This ensures spatial dimensions are
        compatible with patch-based vision architectures.

        Args:
            dim (int): The dimension value to align.
            multiple (int): The multiple to align to.

        Returns:
            int: The aligned dimension, always >= 1 and divisible by multiple
                (when multiple > 1).
        """
        dim = int(dim)
        multiple = int(multiple)
        if multiple <= 1:
            return max(1, dim)
        lower = (dim // multiple) * multiple
        upper = lower + multiple
        if lower <= 0:
            lower = multiple
        # Nearest, but prefer rounding up on ties.
        return lower if (dim - lower) < (upper - dim) else upper

    def _get_resize_buckets_for_model(self) -> list[tuple[int, int]]:
        """Return effective resize buckets for the current model/processor.

        Computes the final set of resolution buckets to use, taking into
        account model-specific requirements. For flat-patch VLMs, buckets
        are aligned to `patch_size * spatial_merge_size` to ensure
        `(grid_h, grid_w)` are divisible by `spatial_merge_size`.

        Also adds the model's configured image_size to the bucket list if
        available from the vision config.

        Returns:
            list[tuple[int, int]]: Sorted, deduplicated list of (height, width)
                resolution buckets aligned to model requirements.
        """
        buckets: list[tuple[int, int]] = []
        for h, w in self.resolution_buckets:
            buckets.append((int(h), int(w)))

        vision_cfg = self._get_vision_config()
        image_size = getattr(vision_cfg, "image_size", None) if vision_cfg is not None else None
        if isinstance(image_size, (int, float)) and int(image_size) > 0:
            sz = int(image_size)
            buckets.append((sz, sz))

        if self._supports_flat_patch_inputs() and vision_cfg is not None:
            patch_size = int(getattr(vision_cfg, "patch_size", 1) or 1)
            spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 1) or 1)
            multiple = patch_size * max(1, spatial_merge_size)
            if multiple > 1:
                aligned: list[tuple[int, int]] = []
                for h, w in buckets:
                    aligned.append(
                        (
                            self._align_dim_to_multiple(h, multiple),
                            self._align_dim_to_multiple(w, multiple),
                        )
                    )
                buckets = aligned

        # Deduplicate while preserving determinism.
        buckets = sorted(set(buckets))
        return buckets

    def _resize_to_buckets(self, image: Image.Image, buckets: list[tuple[int, int]]) -> Image.Image:
        """Resize image to the nearest resolution bucket.

        Selects the bucket with total pixel count closest to the original
        image and resizes using LANCZOS interpolation for quality.

        Args:
            image (Image.Image): PIL Image to resize.
            buckets (list[tuple[int, int]]): List of (height, width) buckets.

        Returns:
            Image.Image: Resized image at bucket resolution, or original
                if already at a bucket size.
        """
        w, h = image.size
        original_pixels = h * w
        target_h, target_w = min(buckets, key=lambda b: abs(b[0] * b[1] - original_pixels))
        if (h, w) == (target_h, target_w):
            return image
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    def _token_str_for_id(self, tokenizer: Any, token_id: int | None) -> str | None:
        """Convert a token ID to its string representation.

        Tries multiple methods to convert the token ID: first using
        convert_ids_to_tokens(), then falling back to decode().

        Args:
            tokenizer (Any): Tokenizer instance with convert_ids_to_tokens
                or decode methods.
            token_id (int | None): Token ID to convert.

        Returns:
            str | None: Token string if conversion succeeded, None otherwise.
        """
        if token_id is None:
            return None
        token_id = int(token_id)

        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            try:
                tok = convert(token_id)
                if isinstance(tok, str) and tok:
                    return tok
            except Exception:
                pass

        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            try:
                tok = decode([token_id], skip_special_tokens=False)
                if isinstance(tok, str) and tok:
                    return tok
            except Exception:
                pass

        return None

    def _placeholder_text(self, cfg: Any, tokenizer: Any, kind: str) -> str | None:
        """Build a textual placeholder sequence for an image/video item.

        Constructs the special token sequence (e.g., <|image_start|><|image|><|image_end|>)
        used to represent multimodal content in the tokenized prompt.

        Args:
            cfg (Any): Model config with token ID attributes.
            tokenizer (Any): Tokenizer for converting IDs to strings.
            kind (str): Either "image" or "video".

        Returns:
            str | None: Placeholder string like "<start><token><end>", or None
                if required token IDs are not available in the config.
        """
        if kind == "image":
            token_id = getattr(cfg, "image_token_id", None)
            start_id = getattr(cfg, "image_start_token_id", None) or getattr(cfg, "vision_start_token_id", None)
            end_id = getattr(cfg, "image_end_token_id", None) or getattr(cfg, "vision_end_token_id", None)
        elif kind == "video":
            token_id = getattr(cfg, "video_token_id", None)
            start_id = getattr(cfg, "video_start_token_id", None) or getattr(cfg, "vision_start_token_id", None)
            end_id = getattr(cfg, "video_end_token_id", None) or getattr(cfg, "vision_end_token_id", None)
        else:
            return None

        tok = self._token_str_for_id(tokenizer, token_id)
        start = self._token_str_for_id(tokenizer, start_id)
        end = self._token_str_for_id(tokenizer, end_id)
        if tok is None or start is None or end is None:
            return None
        # NOTE: many VLM chat templates emit these special tokens with no
        # separating whitespace (e.g. `<|begin_of_image|><|image|><|end_of_image|>`).
        # Adding spaces here can insert extra whitespace tokens and break
        # placeholder pattern matching in tokenizer-only fallback expansion.
        return f"{start}{tok}{end}"

    def _normalize_messages_for_chat_template(self, messages: list[dict], cfg: Any, tokenizer: Any) -> list[dict]:
        """Convert OpenAI-style multimodal content into template-friendly text.

        Many tokenizer chat templates don't understand OpenAI-style content arrays
        with structured items like {"type": "image", ...}. This method converts
        each message's content array into a plain string containing the correct
        special tokens for the model.

        Args:
            messages (list[dict]): OpenAI-style messages with content arrays.
            cfg (Any): Model config with token ID attributes.
            tokenizer (Any): Tokenizer for converting IDs to strings.

        Returns:
            list[dict]: Messages with content converted to plain strings
                containing appropriate placeholder tokens.
        """
        image_placeholder = self._placeholder_text(cfg, tokenizer, "image")
        video_placeholder = self._placeholder_text(cfg, tokenizer, "video")

        if image_placeholder is None and video_placeholder is None:
            return messages

        out: list[dict] = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, list):
                out.append(msg)
                continue

            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    if item:
                        parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "text":
                    text = item.get("text", "")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif item_type in ("image", "image_url") or "image" in item or "image_url" in item:
                    if image_placeholder is not None:
                        parts.append(image_placeholder)
                elif item_type == "video" or "video" in item:
                    if video_placeholder is not None:
                        parts.append(video_placeholder)

            new_msg = dict(msg)
            # Keep behavior close to native chat templates that concatenate
            # multimodal parts without inserting extra separators.
            new_msg["content"] = "".join(parts)
            out.append(new_msg)

        return out

    def _get_vision_config(self) -> Any | None:
        """Get the vision config from the model.

        Attempts to retrieve the vision configuration using multiple
        strategies: first trying get_vision_config() method, then
        falling back to vision_config attribute.

        Returns:
            Any | None: Vision config object if available, None otherwise.
        """
        if self.model is None:
            return None
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return None
        if hasattr(cfg, "get_vision_config"):
            try:
                return cfg.get_vision_config()
            except Exception:
                pass
        return getattr(cfg, "vision_config", None)

    def _get_text_tokenizer(self) -> Any:
        """Return a tokenizer-like object for chat templating and text tokenization.

        HuggingFace ProcessorMixin instances often expose the tokenizer on a
        `.tokenizer` attribute, while tokenizer-only flows pass a
        PreTrainedTokenizerBase directly. This method handles both cases.

        Returns:
            Any: A tokenizer object that supports apply_chat_template().

        Raises:
            ValueError: If processor is not configured.
        """
        if self.processor is None:
            raise ValueError("Processor not configured for tokenization")
        tok = getattr(self.processor, "tokenizer", None)
        # `PreTrainedTokenizerFast` also has a `.tokenizer` attribute, but that
        # is the low-level `tokenizers.Tokenizer` backend (no chat templating).
        if tok is not None and hasattr(tok, "apply_chat_template"):
            return tok
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor
        return tok if tok is not None else self.processor

    def _supports_flat_patch_inputs(self) -> bool:
        """Best-effort detection for models expecting flattened patch tokens.

        GLM4V/GLM46V and Qwen VL models in EasyDeL expect `pixel_values` shaped
        as flattened spatio-temporal patches: [num_patches_total, patch_features].
        This method checks the model_type to determine if flat-patch format
        is expected.

        Returns:
            bool: True if the model expects flat-patch inputs, False otherwise.
        """
        if self.model is None:
            return False
        model_type = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        return model_type in {
            "glm4v",
            "glm4v_moe",
            "glm46v",
            "qwen2_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
        }

    def _pad_flat_patches_for_merge(
        self,
        pixel_values: np.ndarray,
        grid_thw: np.ndarray,
        *,
        spatial_merge_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure flat-patch inputs are compatible with spatial merge.

        Some vision towers (GLM/Qwen) require grid height/width to be divisible
        by `spatial_merge_size`. When upstream preprocessing produces odd grids,
        this method pads the last row/col of patches to the nearest divisible
        size and updates `grid_thw` accordingly.

        Args:
            pixel_values (np.ndarray): Flat patches with shape [num_patches, patch_dim].
            grid_thw (np.ndarray): Grid info with shape [num_images, 3], each row (T, H, W).
            spatial_merge_size (int): The spatial merge factor requiring divisibility.

        Returns:
            tuple[np.ndarray, np.ndarray]: Padded pixel_values and updated grid_thw.
                Returns inputs unchanged if no padding is needed.
        """
        spatial_merge_size = int(spatial_merge_size or 1)
        if spatial_merge_size <= 1:
            return pixel_values, grid_thw

        if pixel_values.ndim != 2:
            return pixel_values, grid_thw
        grid_thw = np.asarray(grid_thw, dtype=np.int64)
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            return pixel_values, grid_thw

        sizes = (grid_thw.prod(axis=1)).astype(int)
        total = int(sizes.sum()) if sizes.size else 0
        if total != int(pixel_values.shape[0]):
            return pixel_values, grid_thw

        patch_dim = int(pixel_values.shape[1])
        out_chunks: list[np.ndarray] = []
        out_grid: list[np.ndarray] = []
        offset = 0
        for (t, h, w), n in zip(grid_thw, sizes, strict=False):
            t_i, h_i, w_i = int(t), int(h), int(w)
            n_i = int(n)
            chunk = pixel_values[offset : offset + n_i]
            offset += n_i

            if h_i <= 0 or w_i <= 0 or t_i <= 0:
                continue

            new_h = ((h_i + spatial_merge_size - 1) // spatial_merge_size) * spatial_merge_size
            new_w = ((w_i + spatial_merge_size - 1) // spatial_merge_size) * spatial_merge_size

            if new_h == h_i and new_w == w_i:
                out_chunks.append(chunk)
                out_grid.append(np.asarray([t_i, h_i, w_i], dtype=np.int64))
                continue

            try:
                reshaped = chunk.reshape(t_i, h_i, w_i, patch_dim)
            except Exception:
                # If ordering/shape doesn't match, keep original data.
                out_chunks.append(chunk)
                out_grid.append(np.asarray([t_i, h_i, w_i], dtype=np.int64))
                continue

            pad_h = new_h - h_i
            pad_w = new_w - w_i
            padded = np.pad(
                reshaped,
                ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
                mode="edge",
            )
            out_chunks.append(padded.reshape(t_i * new_h * new_w, patch_dim))
            out_grid.append(np.asarray([t_i, new_h, new_w], dtype=np.int64))

        if not out_chunks:
            return pixel_values, grid_thw
        return np.concatenate(out_chunks, axis=0), np.stack(out_grid, axis=0)

    def _patchify_spatiotemporal(
        self,
        frames: np.ndarray,
        *,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert frames into flattened patches for GLM/Qwen-style vision towers.

        Performs spatiotemporal patchification: divides video frames into
        3D patches (temporal x height x width) and flattens them into the
        format expected by flat-patch vision encoders.

        Args:
            frames (np.ndarray): Video frames with shape [T, H, W, C] where
                T is number of frames, H/W are spatial dimensions, C=3 (RGB).
            patch_size (int): Spatial patch size (typically 14).
            temporal_patch_size (int): Temporal patch size (typically 2).
            spatial_merge_size (int): Spatial merge factor for grid alignment.
                Defaults to 1 (no alignment requirement).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - flat: Flattened patches [num_patches, C * temporal * patch * patch]
                - grid_thw: Grid shape [t_groups, grid_h, grid_w]

        Raises:
            ValueError: If frames don't have expected 4D shape [T, H, W, C]
                or if C != 3, or if patch sizes are invalid.
        """
        if frames.ndim != 4:
            raise ValueError(f"Expected frames with shape [T,H,W,C], got {frames.shape}.")
        t_total, height, width, channels = frames.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB input, got channels={channels}.")
        patch_size = int(patch_size)
        temporal_patch_size = int(temporal_patch_size)
        spatial_merge_size = int(spatial_merge_size or 1)
        if patch_size <= 0 or temporal_patch_size <= 0 or spatial_merge_size <= 0:
            raise ValueError(f"Invalid patch sizes: patch_size={patch_size}, temporal_patch_size={temporal_patch_size}.")

        spatial_multiple = patch_size * spatial_merge_size

        t_pad = (-t_total) % temporal_patch_size
        h_pad = (-height) % spatial_multiple
        w_pad = (-width) % spatial_multiple

        if t_total + t_pad <= 0 or height + h_pad <= 0 or width + w_pad <= 0:
            raise ValueError(
                f"Input too small after padding: T={t_total},H={height},W={width}, "
                f"patch_size={patch_size}, temporal_patch_size={temporal_patch_size}."
            )

        if t_pad or h_pad or w_pad:
            frames = np.pad(
                frames,
                ((0, t_pad), (0, h_pad), (0, w_pad), (0, 0)),
                mode="edge",
            )

        t_padded, h_padded, w_padded, _c = frames.shape
        t_groups = t_padded // temporal_patch_size
        grid_h = h_padded // patch_size
        grid_w = w_padded // patch_size

        # [t_groups, temporal_patch_size, grid_h, patch_size, grid_w, patch_size, C]
        patches = frames.reshape(t_groups, temporal_patch_size, grid_h, patch_size, grid_w, patch_size, channels)
        # [t_groups, grid_h, grid_w, temporal_patch_size, patch_size, patch_size, C]
        patches = patches.transpose(0, 2, 4, 1, 3, 5, 6)
        # [num_patches, temporal_patch_size, patch_size, patch_size, C]
        patches = patches.reshape(t_groups * grid_h * grid_w, temporal_patch_size, patch_size, patch_size, channels)
        # [num_patches, C, temporal_patch_size, patch_size, patch_size]
        patches = patches.transpose(0, 4, 1, 2, 3)
        flat = patches.reshape(patches.shape[0], -1).astype(np.float32, copy=False)

        grid_thw = np.asarray([t_groups, grid_h, grid_w], dtype=np.int64)
        return flat, grid_thw

    def _normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Apply CLIP-style normalization to RGB values.

        Normalizes pixel values from [0, 255] to the standardized range
        expected by CLIP-based vision encoders using ImageNet-derived
        mean and standard deviation values.

        Args:
            rgb (np.ndarray): RGB image or video frames with values in [0, 255].
                Can be any shape as long as the last dimension is 3 (RGB).

        Returns:
            np.ndarray: Normalized float32 array with approximately zero mean
                and unit variance per channel.
        """
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        rgb = rgb / 255.0
        rgb = (rgb - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
        return rgb

    def process_images(
        self,
        images: list[Image.Image] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Process images with resolution bucketing.

        Resizes images to bucket resolutions and processes them using
        the configured processor. For flat-patch models without processor
        support, falls back to manual patchification.

        Args:
            images (list[Image.Image] | None): List of PIL Images to process.
                Can be None or empty, in which case (None, None) is returned.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: A tuple containing:
                - pixel_values: Processed pixel values. Shape depends on model:
                  - Flat-patch: [total_patches, patch_dim]
                  - Standard: [num_images, C, H, W]
                - image_grid_thw: Grid shapes [num_images, 3] for flat-patch
                  models, or None for standard models.

        Raises:
            ValueError: If processor doesn't support image preprocessing and
                the model doesn't expose a flat-patch vision interface for
                fallback processing.

        Example:
            >>> pixel_values, grid_thw = manager.process_images([img1, img2])
            >>> if pixel_values is not None:
            ...     print(f"Processed {len(images)} images")
        """
        if not images:
            return None, None

        resize_buckets = self._get_resize_buckets_for_model()
        bucketed_images = [self._resize_to_buckets(img.convert("RGB"), resize_buckets) for img in images]

        vision_cfg = self._get_vision_config()
        spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 1) or 1) if vision_cfg is not None else 1

        if self.processor is not None:
            try:
                processed = self.processor(images=bucketed_images, return_tensors="np")
            except Exception as exc:
                # Some VLM processors require `text` even when only image tensors are needed.
                if isinstance(exc, ValueError) and "either `text` or `text_target`" in str(exc):
                    try:
                        processed = self.processor(
                            text=[""] * len(bucketed_images), images=bucketed_images, return_tensors="np"
                        )
                    except Exception:
                        processed = None
                else:
                    processed = None

            if isinstance(processed, dict):
                pixel_values = processed.get("pixel_values")
                image_grid_thw = processed.get("image_grid_thw")
                if pixel_values is not None:
                    if (
                        image_grid_thw is not None
                        and self._supports_flat_patch_inputs()
                        and spatial_merge_size > 1
                        and isinstance(pixel_values, np.ndarray)
                    ):
                        try:
                            pixel_values, image_grid_thw = self._pad_flat_patches_for_merge(
                                pixel_values,
                                image_grid_thw,
                                spatial_merge_size=spatial_merge_size,
                            )
                        except Exception:
                            pass
                    return pixel_values, image_grid_thw

        # Fallback: patchify images for models that consume flattened patch tokens.
        if not self._supports_flat_patch_inputs():
            raise ValueError(
                f"Processor {type(self.processor).__name__ if self.processor is not None else None} "
                "does not support image preprocessing, and this model does not expose a compatible "
                "flat-patch vision interface for fallback preprocessing."
            )

        if vision_cfg is None:
            raise ValueError("Vision config not available for image preprocessing fallback.")
        patch_size = int(getattr(vision_cfg, "patch_size", 14))
        temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2) or 1)

        pixel_values_list: list[np.ndarray] = []
        grids: list[np.ndarray] = []
        for img in bucketed_images:
            rgb = np.asarray(img.convert("RGB"))
            rgb = self._normalize_rgb(rgb)
            frames = np.repeat(rgb[None, ...], temporal_patch_size, axis=0)
            flat, grid = self._patchify_spatiotemporal(
                frames,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                spatial_merge_size=spatial_merge_size,
            )
            pixel_values_list.append(flat)
            grids.append(grid[None, :])

        pixel_values = np.concatenate(pixel_values_list, axis=0) if pixel_values_list else None
        image_grid_thw = np.concatenate(grids, axis=0).astype(np.int64) if grids else None
        if pixel_values is not None and image_grid_thw is not None and spatial_merge_size > 1:
            pixel_values, image_grid_thw = self._pad_flat_patches_for_merge(
                pixel_values, image_grid_thw, spatial_merge_size=spatial_merge_size
            )
        return pixel_values, image_grid_thw

    def process_videos(
        self,
        videos: list[np.ndarray] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Process videos with spatial resolution bucketing.

        Resizes video frames to bucket resolutions and processes them
        using the configured processor. For flat-patch models without
        processor support, falls back to manual spatiotemporal patchification.

        Args:
            videos (list[np.ndarray] | None): List of video arrays, each with
                shape (T, H, W, C) where T is frame count, H/W are spatial
                dimensions, and C=3 (RGB). Can be None or empty.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: A tuple containing:
                - pixel_values_videos: Processed pixel values. Shape depends on model:
                  - Flat-patch: [total_patches, patch_dim]
                  - Standard: [num_videos, T, C, H, W]
                - video_grid_thw: Grid shapes [num_videos, 3] for flat-patch
                  models (T=temporal groups), or None for standard models.

        Raises:
            ValueError: If processor doesn't support video preprocessing and
                the model doesn't expose a flat-patch vision interface for
                fallback processing, or if video arrays have wrong shape.

        Example:
            >>> video = np.random.rand(16, 224, 224, 3).astype(np.uint8)
            >>> pixel_values, grid_thw = manager.process_videos([video])
        """
        if not videos:
            return None, None

        vision_cfg = self._get_vision_config()
        spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 1) or 1) if vision_cfg is not None else 1
        resize_buckets = self._get_resize_buckets_for_model()

        if self.processor is not None:
            try:
                processed = self.processor(videos=videos, return_tensors="np")
            except Exception as exc:
                if isinstance(exc, ValueError) and "either `text` or `text_target`" in str(exc):
                    try:
                        processed = self.processor(text=[""] * len(videos), videos=videos, return_tensors="np")
                    except Exception:
                        processed = None
                else:
                    processed = None

            if isinstance(processed, dict):
                pixel_values_videos = processed.get("pixel_values_videos")
                video_grid_thw = processed.get("video_grid_thw")
                if pixel_values_videos is not None:
                    if (
                        video_grid_thw is not None
                        and self._supports_flat_patch_inputs()
                        and spatial_merge_size > 1
                        and isinstance(pixel_values_videos, np.ndarray)
                    ):
                        try:
                            pixel_values_videos, video_grid_thw = self._pad_flat_patches_for_merge(
                                pixel_values_videos,
                                video_grid_thw,
                                spatial_merge_size=spatial_merge_size,
                            )
                        except Exception:
                            pass
                    return pixel_values_videos, video_grid_thw

        # Fallback: patchify videos for models that consume flattened patch tokens.
        if not self._supports_flat_patch_inputs():
            raise ValueError(
                f"Processor {type(self.processor).__name__ if self.processor is not None else None} "
                "does not support video preprocessing, and this model does not expose a compatible "
                "flat-patch vision interface for fallback preprocessing."
            )

        if vision_cfg is None:
            raise ValueError("Vision config not available for video preprocessing fallback.")
        patch_size = int(getattr(vision_cfg, "patch_size", 14))
        temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2) or 1)

        pixel_values_list: list[np.ndarray] = []
        grids: list[np.ndarray] = []

        for video in videos:
            if video.ndim != 4:
                raise ValueError(f"Video must have shape [T,H,W,C], got {video.shape}.")
            _t_total, height, width, _c = video.shape

            # Resize each frame to a bucket resolution.
            target_h, target_w = min(resize_buckets, key=lambda b: abs(b[0] * b[1] - height * width))
            if (height, width) != (target_h, target_w):
                resized = np.stack(
                    [
                        np.asarray(
                            Image.fromarray(frame).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                        )
                        for frame in video
                    ],
                    axis=0,
                )
            else:
                resized = np.asarray(video)

            resized = self._normalize_rgb(resized.astype(np.float32, copy=False))
            flat, grid = self._patchify_spatiotemporal(
                resized,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                spatial_merge_size=spatial_merge_size,
            )
            pixel_values_list.append(flat)
            grids.append(grid[None, :])

        pixel_values_videos = np.concatenate(pixel_values_list, axis=0) if pixel_values_list else None
        video_grid_thw = np.concatenate(grids, axis=0).astype(np.int64) if grids else None
        if pixel_values_videos is not None and video_grid_thw is not None and spatial_merge_size > 1:
            pixel_values_videos, video_grid_thw = self._pad_flat_patches_for_merge(
                pixel_values_videos, video_grid_thw, spatial_merge_size=spatial_merge_size
            )
        return pixel_values_videos, video_grid_thw

    def extract_media_from_messages(
        self,
        messages: list[dict],
    ) -> tuple[list[Image.Image], list[np.ndarray]]:
        """Extract images and videos from OpenAI-style messages.

        Parses messages with content arrays containing image/video/text items
        and extracts the media for processing.

        Args:
            messages: List of message dicts in OpenAI format with content arrays.

        Returns:
            Tuple of (images, videos) lists.

        Example:
            >>> messages = [
            ...     {"role": "user", "content": [
            ...         {"type": "image", "image": pil_image},
            ...         {"type": "text", "text": "Describe this"}
            ...     ]}
            ... ]
            >>> images, videos = manager.extract_media_from_messages(messages)
        """
        images = []
        videos = []

        for message in messages:
            content = message.get("content", [])

            if isinstance(content, str):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")

                def _append_image(img: Any) -> None:
                    if img is None:
                        return
                    if isinstance(img, Image.Image):
                        images.append(img)
                        return
                    if isinstance(img, (bytes, bytearray)):
                        import io

                        images.append(Image.open(io.BytesIO(img)))
                        return
                    if isinstance(img, str):
                        path = img
                        if path.startswith("file://"):
                            path = path[len("file://") :]
                        if os.path.exists(path):
                            images.append(Image.open(path))
                            return
                    raise ValueError(
                        "Unsupported image payload; use a PIL.Image, local path, file:// path, or data: URL."
                    )

                def _append_image_url(image_url_payload: Any) -> None:
                    url = None
                    if isinstance(image_url_payload, dict):
                        url = image_url_payload.get("url") or image_url_payload.get("image_url")
                    elif isinstance(image_url_payload, str):
                        url = image_url_payload
                    if not isinstance(url, str) or not url:
                        raise ValueError("image_url must be a string or a dict with a non-empty `url` field.")

                    if url.startswith("data:"):
                        import base64
                        import io

                        _header, data = url.split(",", 1)
                        image_data = base64.b64decode(data)
                        images.append(Image.open(io.BytesIO(image_data)))
                        return

                    path = url
                    if path.startswith("file://"):
                        path = path[len("file://") :]
                    if os.path.exists(path):
                        images.append(Image.open(path))
                        return

                    raise ValueError(
                        "Unsupported image_url. Provide a `data:` URL (base64) or a local file path. "
                        "Remote http(s) URLs are not fetched by the server."
                    )

                if item_type in ("image", "input_image") or "image" in item:
                    image_payload = item.get("image")
                    if image_payload is not None:
                        _append_image(image_payload)
                        continue

                if item_type in ("image_url", "input_image") or "image_url" in item:
                    _append_image_url(item.get("image_url"))
                    continue

                if item_type in ("video", "input_video") or "video" in item:
                    video = item.get("video")
                    if video is not None:
                        videos.append(video)
                    continue
                if item_type in ("video_url", "input_video") or "video_url" in item:
                    # No URL fetching for videos right now.
                    raise ValueError("video_url is not supported. Provide raw video arrays via `video` or inline data.")

        return images, videos

    def tokenize_multimodal(
        self,
        messages: list[dict],
        images: list[Image.Image] | None = None,
        videos: list[np.ndarray] | None = None,
        image_grid_thw: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
    ) -> list[int]:
        """Tokenize multimodal messages with placeholder insertion.

        Uses the processor's chat template to convert messages to token IDs,
        inserting appropriate placeholder tokens for images and videos. The
        number of placeholder tokens is determined by the grid_thw arrays
        for flat-patch models.

        For flat-patch VLMs with pre-computed grid_thw, uses a tokenizer-only
        fallback to ensure placeholder counts match the actual processed
        pixel values.

        Args:
            messages (list[dict]): OpenAI-style messages list with content
                arrays that may include image/video/text items.
            images (list[Image.Image] | None): Preprocessed images. If None,
                images are extracted from messages. Defaults to None.
            videos (list[np.ndarray] | None): Preprocessed videos. If None,
                videos are extracted from messages. Defaults to None.
            image_grid_thw (np.ndarray | None): Pre-computed grid shapes for
                images, used to determine placeholder counts. Shape [N, 3].
                Defaults to None.
            video_grid_thw (np.ndarray | None): Pre-computed grid shapes for
                videos, used to determine placeholder counts. Shape [N, 3].
                Defaults to None.

        Returns:
            list[int]: List of token IDs with image/video placeholders
                properly expanded to match the vision encoder output size.

        Raises:
            ValueError: If tokenizer-only fallback is needed but model is
                not provided, or if placeholder patterns are not found in
                the tokenized output for all media items.

        Example:
            >>> token_ids = manager.tokenize_multimodal(
            ...     messages=[{"role": "user", "content": [
            ...         {"type": "image", "image": img},
            ...         {"type": "text", "text": "Describe this image"}
            ...     ]}],
            ...     image_grid_thw=grid_thw
            ... )
        """
        if images is None and videos is None:
            images, videos = self.extract_media_from_messages(messages)

        text_tokenizer = self._get_text_tokenizer()
        messages_for_template = messages
        if self.model is not None:
            cfg_for_template = getattr(self.model, "config", None)
            if cfg_for_template is not None:
                messages_for_template = self._normalize_messages_for_chat_template(
                    messages_for_template,
                    cfg_for_template,
                    text_tokenizer,
                )

        # Flat-patch VLMs (GLM/Qwen) must keep placeholder counts consistent with
        # the already-processed grid_thw/pixel_values; using an HF processor here
        # can re-run image resizing and drift the placeholder expansion.
        force_grid_fallback = (
            self.model is not None
            and self._supports_flat_patch_inputs()
            and (image_grid_thw is not None or video_grid_thw is not None)
        )
        if force_grid_fallback:
            cfg = getattr(self.model, "config", None)
            if cfg is None:
                raise ValueError("Tokenizer-only multimodal fallback requires model.config.")
        else:
            cfg = None

        # Fast-path: HF processors that support multimodal tokenization.
        if cfg is None:
            try:
                text = text_tokenizer.apply_chat_template(
                    messages_for_template, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=text,
                    images=images if images else None,
                    videos=videos if videos else None,
                    return_tensors="np",
                    padding=False,
                )
                input_ids = inputs.get("input_ids")
                if input_ids is not None:
                    return input_ids[0].tolist()
            except Exception:
                pass

        # Fallback: tokenize with a tokenizer-only processing class and expand
        # placeholder runs based on computed grids.
        if self.model is None:
            raise ValueError("Tokenizer-only multimodal fallback requires `model` to be provided.")

        cfg = cfg or getattr(self.model, "config", None)
        if cfg is None:
            raise ValueError("Tokenizer-only multimodal fallback requires model.config.")

        prompt = text_tokenizer.apply_chat_template(messages_for_template, tokenize=False, add_generation_prompt=True)
        encoded = text_tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)
        ids = encoded.get("input_ids", [])
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        token_ids: list[int] = [int(t) for t in ids]

        spatial_merge = int(getattr(getattr(cfg, "vision_config", None), "spatial_merge_size", 1) or 1)
        spatial_div = int(spatial_merge * spatial_merge)

        # Token IDs (model-dependent).
        image_token_id = getattr(cfg, "image_token_id", None)
        video_token_id = getattr(cfg, "video_token_id", None)

        # Start/end markers (GLM uses per-modality markers; Qwen uses vision_start/end).
        image_start_token_id = getattr(cfg, "image_start_token_id", None) or getattr(cfg, "vision_start_token_id", None)
        image_end_token_id = getattr(cfg, "image_end_token_id", None) or getattr(cfg, "vision_end_token_id", None)
        video_start_token_id = getattr(cfg, "video_start_token_id", None) or getattr(cfg, "vision_start_token_id", None)
        video_end_token_id = getattr(cfg, "video_end_token_id", None) or getattr(cfg, "vision_end_token_id", None)

        def _counts(grid: np.ndarray | None) -> list[int]:
            if grid is None:
                return []
            grid = np.asarray(grid, dtype=np.int64)
            if grid.ndim != 2 or grid.shape[1] != 3:
                return []
            return [int(np.prod(row) // spatial_div) for row in grid]

        image_counts = _counts(image_grid_thw)
        video_counts = _counts(video_grid_thw)
        img_idx = 0
        vid_idx = 0

        out: list[int] = []
        i = 0
        while i < len(token_ids):
            # Image pattern: <start> <image_token> <end>
            if (
                image_token_id is not None
                and image_start_token_id is not None
                and image_end_token_id is not None
                and i + 2 < len(token_ids)
                and token_ids[i] == int(image_start_token_id)
                and token_ids[i + 1] == int(image_token_id)
                and token_ids[i + 2] == int(image_end_token_id)
                and img_idx < len(image_counts)
            ):
                out.append(int(image_start_token_id))
                out.extend([int(image_token_id)] * max(int(image_counts[img_idx]), 0))
                out.append(int(image_end_token_id))
                img_idx += 1
                i += 3
                continue

            # Video pattern: <start> <video_token> <end>
            if (
                video_token_id is not None
                and video_start_token_id is not None
                and video_end_token_id is not None
                and i + 2 < len(token_ids)
                and token_ids[i] == int(video_start_token_id)
                and token_ids[i + 1] == int(video_token_id)
                and token_ids[i + 2] == int(video_end_token_id)
                and vid_idx < len(video_counts)
            ):
                out.append(int(video_start_token_id))
                out.extend([int(video_token_id)] * max(int(video_counts[vid_idx]), 0))
                out.append(int(video_end_token_id))
                vid_idx += 1
                i += 3
                continue

            out.append(int(token_ids[i]))
            i += 1

        if image_counts and img_idx != len(image_counts):
            raise ValueError(
                "Multimodal image inputs provided, but the chat template did not emit the expected "
                "`<start> <image_token> <end>` sequence for all images. "
                "Ensure the tokenizer/processor chat template supports images, or use a processor that "
                "can generate multimodal `input_ids`."
            )
        if video_counts and vid_idx != len(video_counts):
            raise ValueError(
                "Multimodal video inputs provided, but the chat template did not emit the expected "
                "`<start> <video_token> <end>` sequence for all videos. "
                "Ensure the tokenizer/processor chat template supports videos, or use a processor that "
                "can generate multimodal `input_ids`."
            )

        return out

    def clear_cache(self) -> None:
        """Clear the vision encoder cache.

        Removes all cached vision encoder outputs and resets cache statistics.
        Has no effect if caching is disabled.

        Example:
            >>> manager.clear_cache()
            >>> stats = manager.get_cache_stats()
            >>> assert stats['num_entries'] == 0 if stats else True
        """
        if self.cache is not None:
            self.cache.clear()

    def process_images_to_features(
        self,
        images: list[Image.Image] | None,
        request_idx: int = 0,
    ) -> list[MultiModalFeature]:
        """Process images and create MultiModalFeature objects.

        Processes images with resolution bucketing and creates feature objects
        with content-based hashing for cache lookups. Each image becomes a
        separate MultiModalFeature with its own hash and pixel values.

        For flat-patch models, properly slices the concatenated pixel_values
        using the grid_thw information to create per-image features.

        Args:
            images (list[Image.Image] | None): List of PIL Images to process.
                Can be None or empty, in which case an empty list is returned.
            request_idx (int): Index of the request in a batch. Used for
                tracking which request each feature belongs to. Defaults to 0.

        Returns:
            list[MultiModalFeature]: List of feature objects, one per input
                image. Each feature has pixel_values, grid_thw (if applicable),
                mm_hash for caching, and the request_idx.

        Example:
            >>> features = manager.process_images_to_features([img1, img2], request_idx=0)
            >>> for feat in features:
            ...     print(f"Hash: {feat.mm_hash}, Shape: {feat.pixel_values.shape}")
        """
        if not images:
            return []

        pixel_values, image_grid_thw = self.process_images(images)

        if pixel_values is None:
            return []

        features = []
        num_images = len(images)
        patch_offsets: list[int] | None = None
        if (
            image_grid_thw is not None
            and pixel_values.ndim == 2
            and image_grid_thw.ndim == 2
            and image_grid_thw.shape[1] == 3
        ):
            sizes = (np.asarray(image_grid_thw, dtype=np.int64).prod(axis=1)).astype(int).tolist()
            patch_offsets = [0]
            for s in sizes[:-1]:
                patch_offsets.append(patch_offsets[-1] + int(s))

        for i in range(num_images):
            if patch_offsets is not None:
                assert image_grid_thw is not None
                size = int(np.asarray(image_grid_thw[i]).prod())
                start = int(patch_offsets[i])
                single_pv = pixel_values[start : start + size]
            elif pixel_values.ndim > 3:
                single_pv = pixel_values[i : i + 1]
            else:
                single_pv = pixel_values

            single_grid = None
            if image_grid_thw is not None and i < len(image_grid_thw):
                single_grid = image_grid_thw[i : i + 1]

            feature = MultiModalFeature.from_image(
                pixel_values=single_pv,
                grid_thw=single_grid,
                request_idx=request_idx,
            )
            features.append(feature)

        return features

    def process_videos_to_features(
        self,
        videos: list[np.ndarray] | None,
        request_idx: int = 0,
    ) -> list[MultiModalFeature]:
        """Process videos and create MultiModalFeature objects.

        Processes videos with resolution bucketing and creates feature objects
        with content-based hashing for cache lookups. Each video becomes a
        separate MultiModalFeature with its own hash and pixel values.

        For flat-patch models, properly slices the concatenated pixel_values
        using the grid_thw information to create per-video features.

        Args:
            videos (list[np.ndarray] | None): List of video arrays, each with
                shape (T, H, W, C). Can be None or empty, in which case an
                empty list is returned.
            request_idx (int): Index of the request in a batch. Used for
                tracking which request each feature belongs to. Defaults to 0.

        Returns:
            list[MultiModalFeature]: List of feature objects, one per input
                video. Each feature has pixel_values, grid_thw (if applicable),
                mm_hash for caching, and the request_idx.

        Example:
            >>> video = np.random.rand(16, 224, 224, 3).astype(np.uint8)
            >>> features = manager.process_videos_to_features([video], request_idx=1)
            >>> print(f"Created {len(features)} video features")
        """
        if not videos:
            return []

        pixel_values_videos, video_grid_thw = self.process_videos(videos)

        if pixel_values_videos is None:
            return []

        features = []
        num_videos = len(videos)
        patch_offsets: list[int] | None = None
        if (
            video_grid_thw is not None
            and pixel_values_videos.ndim == 2
            and video_grid_thw.ndim == 2
            and video_grid_thw.shape[1] == 3
        ):
            sizes = (np.asarray(video_grid_thw, dtype=np.int64).prod(axis=1)).astype(int).tolist()
            patch_offsets = [0]
            for s in sizes[:-1]:
                patch_offsets.append(patch_offsets[-1] + int(s))

        for i in range(num_videos):
            if patch_offsets is not None:
                assert video_grid_thw is not None
                size = int(np.asarray(video_grid_thw[i]).prod())
                start = int(patch_offsets[i])
                single_pv = pixel_values_videos[start : start + size]
            elif pixel_values_videos.ndim > 4:
                single_pv = pixel_values_videos[i : i + 1]
            else:
                single_pv = pixel_values_videos

            single_grid = None
            if video_grid_thw is not None and i < len(video_grid_thw):
                single_grid = video_grid_thw[i : i + 1]

            feature = MultiModalFeature.from_video(
                pixel_values=single_pv,
                grid_thw=single_grid,
                request_idx=request_idx,
            )
            features.append(feature)

        return features

    def get_cache_stats(self) -> dict | None:
        """Get vision encoder cache statistics.

        Retrieves performance metrics from the internal vision encoder cache,
        including hit rate, memory usage, and entry count.

        Returns:
            dict | None: Dictionary with cache statistics if caching is
                enabled, None if caching is disabled. When enabled, contains:
                - hits (int): Number of cache hits
                - misses (int): Number of cache misses
                - hit_rate (float): Ratio of hits to total lookups (0.0-1.0)
                - size_mb (float): Current cache size in megabytes
                - num_entries (int): Number of cached entries
                - capacity_mb (float): Maximum cache capacity in megabytes

        Example:
            >>> stats = manager.get_cache_stats()
            >>> if stats:
            ...     print(f"Cache hit rate: {stats['hit_rate']:.1%}")
            ...     print(f"Memory usage: {stats['size_mb']:.1f} MB")
        """
        if self.cache is not None:
            return self.cache.get_stats()
        return None
