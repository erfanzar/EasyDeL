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

"""PIL-free image preprocessing for DeepSeek-V4.1 vision-language serving.

Port of DeepSeek's reference ``inference/image_processor.py`` working on NumPy ``uint8 [H, W, 3]`` arrays. An image
becomes an ``n_vit_h x n_vit_w`` patch grid for the ViT and an ``n_llm_h x n_llm_w`` token grid after the aligner's
3x3 downsample; the language model sees

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

with ``image_token_id`` at every one of those positions in ``input_ids`` and the real role carried by ``token_types``.
Engram hashing must treat every span position as dead (``dead_mask = token_types >= 0``).

Deviation from the reference: the resize uses bilinear sampling (PIL's ``ImageOps.pad`` uses bicubic) and patches stay
float32 (the reference casts to bfloat16); the grid planning, padding colour, normalisation and patch layout are
identical, so token counts, spans and grids match the reference exactly while pixel values differ by resampling noise.
"""

from __future__ import annotations

import dataclasses
import math
import typing as tp

import numpy as np

from .modeling_deepseek_v41_vision import (
    TOKEN_TYPE_IMAGE,
    TOKEN_TYPE_IMAGE_END,
    TOKEN_TYPE_IMAGE_NEW_LINE,
    TOKEN_TYPE_IMAGE_START,
    TOKEN_TYPE_TEXT,
)

_PAD_COLOR = 127


@dataclasses.dataclass(frozen=True)
class DeepseekV41ImageSpec:
    """Static image-preprocessing parameters (``DeepseekV41Config.vision_params()``).

    Attributes:
        patch_size: ViT patch side in pixels.
        downsample_ratio: Aligner fold size ``r`` (a ``r x r`` patch block becomes one LLM token).
        max_image_tokens: Upper bound of LLM tokens one image span may occupy (delimiters included).
        min_pixels: Images with fewer pixels are scaled up to at least this many.
        max_wh_ratio: Optional width/height clamp; wider images are stretched instead of padded.
    """

    patch_size: int = 14
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | None = None

    @classmethod
    def from_config(cls, config: tp.Any) -> DeepseekV41ImageSpec:
        """Build the spec from a :class:`DeepseekV41Config` with an enabled vision tower.

        Args:
            config: Config exposing ``vision_params()``.

        Returns:
            The spec.
        """
        params = config.vision_params()
        ratio = params.get("max_wh_ratio")
        return cls(
            patch_size=int(params["patch_size"]),
            downsample_ratio=int(params["downsample_ratio"]),
            max_image_tokens=int(params["max_image_tokens"]),
            min_pixels=int(params["min_pixels"]),
            max_wh_ratio=None if ratio is None else float(ratio),
        )


@dataclasses.dataclass(frozen=True)
class PreparedImage:
    """One preprocessed image.

    Attributes:
        patches: ``float32 [n_vit_h * n_vit_w, 3, patch, patch]`` in row-major patch order, normalised to ``[-1, 1]``.
        grid: ``(n_vit_h, n_vit_w)`` ViT patch grid.
        llm_grid: ``(n_llm_h, n_llm_w)`` token grid after the aligner.
        token_types: ``int64 [num_image_tokens]`` span layout (``START, (IMAGE*w, NEW_LINE)*h, END``).
    """

    patches: np.ndarray
    grid: tuple[int, int]
    llm_grid: tuple[int, int]
    token_types: np.ndarray

    @property
    def num_tokens(self) -> int:
        """Number of prompt positions the image span occupies."""
        return int(self.token_types.shape[0])


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Span length for a token grid: ``h * (w + 1) + 2``."""
    return n_llm_h * (n_llm_w + 1) + 2


def llm_token_grid(height: int, width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
    """Token grid the aligner produces for an image of ``height x width`` pixels (ceil division)."""
    return math.ceil((height // patch_size) / downsample_ratio), math.ceil((width // patch_size) / downsample_ratio)


def solve_resize_ratio(
    height: int, width: int, patch_size: int, downsample_ratio: int, max_n_token: int
) -> tuple[int, int]:
    """Largest aspect-preserving pixel size whose token grid still fits in ``max_n_token`` tokens."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def plan_image_grid(width: int, height: int, spec: DeepseekV41ImageSpec) -> tuple[int, int, int, int]:
    """Resize plan for an image of the given original size (pure function).

    Args:
        width: Original width in pixels.
        height: Original height in pixels.
        spec: Preprocessing parameters.

    Returns:
        ``(n_llm_h, n_llm_w, best_height, best_width)``; the pixel size is a multiple of ``patch_size``.
    """
    p = spec.patch_size
    if spec.max_wh_ratio is not None and width > height * spec.max_wh_ratio:
        width = height * spec.max_wh_ratio
    if 0 < width * height < spec.min_pixels:
        ratio = (spec.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w = llm_token_grid(best_height, best_width, p, spec.downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > spec.max_image_tokens:
        best_height, best_width = solve_resize_ratio(height, width, p, spec.downsample_ratio, spec.max_image_tokens)
        n_llm_h, n_llm_w = llm_token_grid(best_height, best_width, p, spec.downsample_ratio)
        if num_image_tokens(n_llm_h, n_llm_w) > spec.max_image_tokens:  # pragma: no cover - guarded by the solver
            raise ValueError(f"cannot fit a {width}x{height} image into {spec.max_image_tokens} tokens.")
    return n_llm_h, n_llm_w, best_height, best_width


def image_token_types(n_llm_h: int, n_llm_w: int) -> np.ndarray:
    """Span layout ``[START] + ([IMAGE] * w + [NEW_LINE]) * h + [END]`` as ``int64``."""
    types = [TOKEN_TYPE_IMAGE_START]
    types += ([TOKEN_TYPE_IMAGE] * n_llm_w + [TOKEN_TYPE_IMAGE_NEW_LINE]) * n_llm_h
    types.append(TOKEN_TYPE_IMAGE_END)
    return np.asarray(types, dtype=np.int64)


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of ``[H, W, C]`` to ``[height, width, C]`` (float32, half-pixel centres)."""
    src = image.astype(np.float32)
    src_h, src_w = src.shape[:2]
    if (src_h, src_w) == (height, width):
        return src
    ys = np.clip((np.arange(height, dtype=np.float32) + 0.5) * (src_h / height) - 0.5, 0.0, src_h - 1.0)
    xs = np.clip((np.arange(width, dtype=np.float32) + 0.5) * (src_w / width) - 0.5, 0.0, src_w - 1.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def _pad_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Fit ``image`` inside ``height x width`` keeping its aspect ratio and centre it on grey (``ImageOps.pad``)."""
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, min(width, round(src_w * scale)))
    new_h = max(1, min(height, round(src_h * scale)))
    fitted = _resize_bilinear(image, new_h, new_w)
    canvas = np.full((height, width, image.shape[2]), float(_PAD_COLOR), dtype=np.float32)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = fitted
    return canvas


def preprocess_image(image: np.ndarray, spec: DeepseekV41ImageSpec) -> PreparedImage:
    """Resize, pad, normalise and patchify one image.

    Args:
        image: ``uint8 [H, W, 3]`` RGB array (grayscale ``[H, W]`` / RGBA are rejected, convert first).
        spec: Preprocessing parameters.

    Returns:
        The :class:`PreparedImage`.

    Raises:
        ValueError: On an array that is not ``uint8 [H, W, 3]`` or has an empty side.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be uint8 [H, W, 3]; got shape {image.shape} dtype {image.dtype}.")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("image has an empty side.")
    p = spec.patch_size
    n_llm_h, n_llm_w, best_h, best_w = plan_image_grid(width, height, spec)
    n_vit_h, n_vit_w = best_h // p, best_w // p
    if spec.max_wh_ratio is not None and width >= spec.max_wh_ratio * height:
        pixels = _resize_bilinear(image, best_h, best_w)
    else:
        pixels = _pad_to(image, best_h, best_w)
    x = (pixels / 255.0 - 0.5) / 0.5  # [H, W, 3] in [-1, 1]
    x = x.transpose(2, 0, 1)  # [3, H, W]
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).transpose(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return PreparedImage(
        patches=np.ascontiguousarray(patches, dtype=np.float32),
        grid=(n_vit_h, n_vit_w),
        llm_grid=(n_llm_h, n_llm_w),
        token_types=image_token_types(n_llm_h, n_llm_w),
    )


def expand_image_placeholders(
    prompt_token_ids: tp.Sequence[int],
    images: tp.Sequence[np.ndarray],
    spec: DeepseekV41ImageSpec,
    image_token_id: int,
) -> tuple[list[int], np.ndarray, list[PreparedImage]]:
    """Expand every ``image_token_id`` placeholder of a tokenized prompt into its image span.

    Args:
        prompt_token_ids: Tokenized prompt holding exactly one ``image_token_id`` per image.
        images: ``uint8 [H, W, 3]`` arrays, consumed in order of appearance.
        spec: Preprocessing parameters.
        image_token_id: ``config.image_token_id``.

    Returns:
        ``(token_ids, token_types, prepared)``: expanded ids (span positions carry ``image_token_id``), ``int64``
        token types (``-1`` text) of the same length, and the preprocessed images in order. The engram dead mask is
        ``token_types >= 0``.

    Raises:
        ValueError: If the number of placeholders differs from the number of images.
    """
    placeholders = sum(1 for token in prompt_token_ids if int(token) == int(image_token_id))
    if placeholders != len(images):
        raise ValueError(f"Found {placeholders} image tokens but got {len(images)} images.")
    ids: list[int] = []
    types: list[int] = []
    prepared: list[PreparedImage] = []
    image_iter = iter(images)
    for token in prompt_token_ids:
        if int(token) != int(image_token_id):
            ids.append(int(token))
            types.append(TOKEN_TYPE_TEXT)
            continue
        item = preprocess_image(next(image_iter), spec)
        prepared.append(item)
        ids.extend([int(image_token_id)] * item.num_tokens)
        types.extend(item.token_types.tolist())
    return ids, np.asarray(types, dtype=np.int64), prepared


def image_span_bounds(token_types: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` position ranges of each image span in a ``[S]`` token-type row.

    A span begins at ``IMAGE_START`` and ends after ``IMAGE_END``; the prefill chunking rule is that no chunk boundary
    may fall strictly inside any returned range.

    Args:
        token_types: ``int [S]`` token types.

    Returns:
        One ``(start, end)`` per span, in order.

    Raises:
        ValueError: On a malformed span (missing start / end).
    """
    types = np.asarray(token_types)
    bounds: list[tuple[int, int]] = []
    starts = np.flatnonzero(types == TOKEN_TYPE_IMAGE_START)
    ends = np.flatnonzero(types == TOKEN_TYPE_IMAGE_END)
    if starts.shape != ends.shape or np.any(ends < starts):
        raise ValueError("token_types hold unbalanced IMAGE_START / IMAGE_END markers.")
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        bounds.append((start, end + 1))
    return bounds


@dataclasses.dataclass(frozen=True)
class ExpandedImagePrompt:
    """A prompt with its image placeholders expanded.

    Attributes:
        token_ids: Expanded prompt ids (every span position carries ``image_token_id``).
        token_types: ``int64 [len(token_ids)]`` token types (``-1`` text, ``0..3`` span positions).
        spans: Half-open ``(start, stop)`` position range of each image span.
        features: One eSurge ``MultiModalFeature`` per image (patches, ``[[1, n_vit_h, n_vit_w]]`` grid, placeholder
            range = the span).
    """

    token_ids: list[int]
    token_types: np.ndarray
    spans: list[tuple[int, int]]
    features: list[tp.Any]


class DeepseekV41ImageExpander:
    """Expands ``image_token_id`` placeholders of a tokenized prompt (eSurge scheduler admission)."""

    def __init__(self, config: tp.Any) -> None:
        """Build the expander from a vision-enabled ``DeepseekV41Config``.

        Raises:
            ValueError: If the config has no vision tower.
        """
        if not config.vision_enabled:
            raise ValueError("DeepSeek-V4.1 image serving needs a config with an enabled `vision_config`.")
        self.spec = DeepseekV41ImageSpec.from_config(config)
        self.image_token_id = int(config.image_token_id)

    def expand(self, prompt_token_ids: tp.Sequence[int], images: tp.Sequence[np.ndarray]) -> ExpandedImagePrompt:
        """Expand placeholders, preprocess the images and wrap each into a ``MultiModalFeature``.

        Args:
            prompt_token_ids: Prompt holding exactly one ``image_token_id`` per image.
            images: ``uint8 [H, W, 3]`` arrays in order of appearance.

        Raises:
            ValueError: On a placeholder/image count mismatch or an invalid image.
        """
        from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange

        ids, types, prepared = expand_image_placeholders(prompt_token_ids, images, self.spec, self.image_token_id)
        spans = image_span_bounds(types)
        features = [
            MultiModalFeature.from_image(
                item.patches,
                grid_thw=np.asarray([[1, item.grid[0], item.grid[1]]], dtype=np.int32),
                placeholder_range=PlaceholderRange(offset=start, length=stop - start, modality="image"),
            )
            for (start, stop), item in zip(spans, prepared, strict=True)
        ]
        return ExpandedImagePrompt(token_ids=ids, token_types=types, spans=spans, features=features)


def _feature_grid_thw(feature: tp.Any) -> np.ndarray:
    """Read the image layout retained alongside pixel-cleared encoder results."""
    grid = feature.grid_thw
    if grid is None:
        grid = getattr(feature, "_esurge_vision_grid_thw", None)
    if grid is None:
        raise ValueError("Image feature needs its patch grid or retained encoder-cache grid metadata.")
    grid = np.asarray(grid).reshape(-1)
    if grid.shape != (3,) or int(grid[0]) != 1 or np.any(grid[1:] <= 0):
        raise ValueError(f"Image feature must have one positive [1, height, width] patch grid; got {grid}.")
    return grid


def token_types_from_features(features: tp.Sequence[tp.Any], num_tokens: int, downsample_ratio: int) -> np.ndarray:
    """Rebuild the prompt token types from image features (placeholder offset + patch grid).

    Raises:
        ValueError: If a feature's span length disagrees with its grid.
    """
    types = np.full((int(num_tokens),), TOKEN_TYPE_TEXT, dtype=np.int64)
    for feature in features:
        grid = _feature_grid_thw(feature)
        n_llm_h, n_llm_w = -(-int(grid[1]) // downsample_ratio), -(-int(grid[2]) // downsample_ratio)
        span = image_token_types(n_llm_h, n_llm_w)
        placeholder = feature.placeholder_range
        if placeholder is None or placeholder.length != span.shape[0]:
            raise ValueError("image feature span length disagrees with its patch grid.")
        types[placeholder.offset : placeholder.end] = span
    return types


def compute_prefill_embeddings(
    model: tp.Any,
    prompt_token_ids: tp.Sequence[int],
    features: tp.Sequence[tp.Any],
    encoder_cache_provider: tp.Callable[[tp.Any, tp.Callable[[], tp.Any]], tp.Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode or reuse image rows, then merge them into this prompt's fresh embeddings.

    Args:
        model: ``DeepseekV41ForConditionalGeneration``.
        prompt_token_ids: Expanded prompt ids.
        features: Image features in any order, carrying pixels or cached aligner rows.
        encoder_cache_provider: Optional ``provider(feature, encode)`` returning the
            feature's aligner rows, not prompt-dependent merged embeddings. ``encode``
            is a zero-argument lazy fallback requiring raw pixels. The provider owns
            cache identity/weight-version validation and may reuse cached-only features.
            Without a provider, ``feature.cached_embeddings`` is used when present;
            otherwise the image is encoded and its rows retained on the feature.

    Returns:
        ``(embeds, token_types)``: ``float32 [S, hidden]`` merged embeddings and ``int64 [S]`` token types.

    Raises:
        ValueError: If neither encoder rows nor raw pixels are available, or an
            encoder result's row count disagrees with its image grid.
    """
    import jax
    from jax import numpy as jnp

    from easydel.utils import set_inference_mode

    base = getattr(model, "base_model", model)
    ordered = sorted(features, key=lambda f: f.placeholder_range.offset)
    num_tokens = len(prompt_token_ids)
    ratio = int(model.config.vision_params()["downsample_ratio"])
    types = token_types_from_features(ordered, num_tokens, ratio)
    dtype = getattr(model, "dtype", jnp.float32)
    with model.mesh, set_inference_mode():
        embeds = model.compute_embedding(jnp.asarray(np.asarray(prompt_token_ids, dtype=np.int32))[None, :])
        rows = []
        for feature in ordered:
            grid = _feature_grid_thw(feature)
            image_grid = (int(grid[1]), int(grid[2]))

            def encode(feature=feature, image_grid=image_grid):
                if feature.pixel_values is None:
                    raise ValueError("Image feature has neither cached encoder rows nor raw pixels.")
                patches = jnp.asarray(feature.pixel_values, dtype=dtype)[None]
                return base.get_image_features(patches, image_grid=image_grid)

            cache_rows = False
            if encoder_cache_provider is not None:
                out = encoder_cache_provider(feature, encode)
            else:
                out = getattr(feature, "cached_embeddings", None)
                if out is None:
                    out = encode()
                    cache_rows = True
            out = jnp.asarray(out, dtype=dtype)
            expected_rows = math.ceil(image_grid[0] / ratio) * math.ceil(image_grid[1] / ratio)
            if out.ndim < 2 or math.prod(out.shape[:-1]) != expected_rows:
                raise ValueError(f"Image encoder result must provide {expected_rows} rows; got shape {out.shape}.")
            out = out.reshape(-1, out.shape[-1])
            # MultiModalFeature.clear_pixel_values also clears grid_thw. Keep the
            # small layout metadata needed to rebuild token types on later turns.
            feature._esurge_vision_grid_thw = np.array(grid, copy=True)
            if cache_rows and callable(getattr(feature, "set_cached_embeddings", None)):
                feature.set_cached_embeddings(out)
            rows.append(out)
        merged = (
            base.merge_image_embeddings(embeds, jnp.asarray(types)[None, :], jnp.concatenate(rows, axis=0)[None])
            if rows
            else embeds
        )
        host = np.asarray(jax.device_get(merged.astype(jnp.float32)))
    return host[0], types


__all__ = [
    "DeepseekV41ImageExpander",
    "DeepseekV41ImageSpec",
    "ExpandedImagePrompt",
    "PreparedImage",
    "compute_prefill_embeddings",
    "expand_image_placeholders",
    "image_span_bounds",
    "image_token_types",
    "llm_token_grid",
    "num_image_tokens",
    "plan_image_grid",
    "preprocess_image",
    "solve_resize_ratio",
    "token_types_from_features",
]
