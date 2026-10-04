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

"""Per-token side inputs eSurge feeds DeepSeek-V4.1 (engram hash ids, image token types, image embeddings).

Nothing here is imported by eSurge itself: the model's serving hooks (``_esurge_hooks``) reach it through the generic
side-input / image-expander contracts of :mod:`easydel.inference.esurge.multimodal.side_inputs`.

* :class:`EngramWindowHasher` hashes the scheduled token window ``[start, end)`` of one request. The
  :class:`~easydel.modules.deepseek_v41._engram_hash.EngramHasher` state is rebuilt from the request's own token
  history (the last ``max_ngram_size - 1`` compressed ids, image-span positions dead), which is exactly the state an
  incremental hasher would carry. Chunked prefill, decode, preemption/resume and speculative rollback therefore all
  hash identically to one stateless pass over the sequence, with no per-request hasher state to reset.
* :class:`DeepseekV41ImageExpander` expands image placeholders of a tokenized prompt at admission and wraps each
  prepared image into a :class:`MultiModalFeature` that travels through the normal request plumbing.
* :func:`compute_prefill_embeddings` encodes the images eagerly and merges them into the prompt embeddings, which the
  runner then streams through the existing ``prefill_embeds`` channel.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange

from ._image_processing import DeepseekV41ImageSpec, expand_image_placeholders, image_span_bounds, image_token_types


def build_engram_hasher(config: tp.Any, token_map: np.ndarray | None = None) -> tp.Any:
    """Build the :class:`EngramHasher` of a config.

    Args:
        config: ``DeepseekV41Config``.
        token_map: Compressed token map (``build_engram_token_map(tokenizer)``). ``None`` uses the identity map, which
            is only valid when ``config.engram_compressed_vocab_size == config.vocab_size``.

    Returns:
        The hasher.

    Raises:
        ValueError: If no map is given and the identity map does not match the config.
    """
    from easydel.modules.deepseek_v41._engram_hash import EngramHasher

    if token_map is None:
        vocab = int(config.vocab_size)
        if int(config.engram_compressed_vocab_size) != vocab:
            raise ValueError(
                f"DeepSeek-V4.1 engram needs the tokenizer's compressed token map (config.engram_compressed_vocab_size="
                f"{config.engram_compressed_vocab_size} != vocab_size={vocab}). Build it with "
                "easydel.modules.deepseek_v41._engram_hash.build_engram_token_map(tokenizer) and pass it to "
                "eSurgeRunner.set_side_input_tokenizer_state(...) before serving (eSurge does this from its tokenizer)."
            )
        token_map = np.arange(vocab, dtype=np.int64)
    return EngramHasher.from_config(config, token_map)


class EngramWindowHasher:
    """Hash ids of a scheduled token window, rebuilt from the request's token history (see module docstring)."""

    def __init__(self, hasher: tp.Any) -> None:
        """Wrap an :class:`EngramHasher`.

        Args:
            hasher: The configured hasher.
        """
        self.hasher = hasher
        self.lookback = int(hasher.max_ngram_size) - 1

    @property
    def n_layers(self) -> int:
        """Number of engram layers."""
        return int(self.hasher.n_layers)

    @property
    def n_columns(self) -> int:
        """Hash columns per (position, layer)."""
        return int(self.hasher.n_columns)

    def hash_window(
        self,
        token_ids_row: np.ndarray,
        token_types: np.ndarray | None,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Hash positions ``[start, end)`` of one request.

        Args:
            token_ids_row: The request's token ids (prompt followed by generated tokens); positions
                ``max(0, start - lookback) .. end`` must be real tokens.
            token_types: Prompt token types (``>= 0`` marks an image-span position, which is dead), or ``None`` for a
                text-only request. Positions past its end are text.
            start: First position of the window.
            end: One past the last position of the window.

        Returns:
            ``int32 [end - start, n_layers, n_columns]``.
        """
        hasher = self.hasher
        lo = max(0, int(start) - self.lookback)
        # negative ids are speculative-draft placeholders; their rows are replaced downstream, hash them as id 0
        ids = np.maximum(np.asarray(token_ids_row[lo:end], dtype=np.int64), 0)
        dead = np.zeros(ids.shape, dtype=bool)
        if token_types is not None:
            hi = min(int(end), int(token_types.shape[0]))
            if hi > lo:
                dead[: hi - lo] = np.asarray(token_types[lo:hi]) >= 0
        from easydel.modules.deepseek_v41._engram_hash import DEAD, EngramHashState

        history = np.full((1, self.lookback), DEAD, dtype=np.int64)
        n_history = int(start) - lo
        if n_history:
            history[0, self.lookback - n_history :] = np.where(dead[:n_history], DEAD, hasher.token_map[ids[:n_history]])
        state = EngramHashState(history=history, position=int(start))
        out, _ = hasher.step(ids[None, n_history:], state, dead[None, n_history:])
        return out[0]


@dataclasses.dataclass(frozen=True)
class ExpandedImagePrompt:
    """A prompt with its image placeholders expanded.

    Attributes:
        token_ids: Expanded prompt ids (every span position carries ``image_token_id``).
        token_types: ``int64 [len(token_ids)]`` token types (``-1`` text, ``0..3`` span positions).
        spans: Half-open ``(start, stop)`` position range of each image span.
        features: One :class:`MultiModalFeature` per image (patches, ``[[1, n_vit_h, n_vit_w]]`` grid, placeholder
            range = the span).
    """

    token_ids: list[int]
    token_types: np.ndarray
    spans: list[tuple[int, int]]
    features: list[MultiModalFeature]


class DeepseekV41ImageExpander:
    """Expands ``image_token_id`` placeholders of a tokenized prompt (scheduler admission)."""

    def __init__(self, config: tp.Any) -> None:
        """Build the expander from a vision-enabled ``DeepseekV41Config``.

        Args:
            config: Config with an enabled vision tower.

        Raises:
            ValueError: If the config has no vision tower.
        """
        if not config.vision_enabled:
            raise ValueError("DeepSeek-V4.1 image serving needs a config with an enabled `vision_config`.")
        self.spec = DeepseekV41ImageSpec.from_config(config)
        self.image_token_id = int(config.image_token_id)

    def expand(self, prompt_token_ids: tp.Sequence[int], images: tp.Sequence[np.ndarray]) -> ExpandedImagePrompt:
        """Expand and preprocess.

        Args:
            prompt_token_ids: Prompt holding exactly one ``image_token_id`` per image.
            images: ``uint8 [H, W, 3]`` arrays in order of appearance.

        Returns:
            The :class:`ExpandedImagePrompt`.

        Raises:
            ValueError: On a placeholder/image count mismatch or an invalid image.
        """
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


def token_types_from_features(
    features: tp.Sequence[MultiModalFeature], num_tokens: int, downsample_ratio: int
) -> np.ndarray:
    """Rebuild the prompt token types from image features (offset, patch grid).

    Args:
        features: Image features with a placeholder range and ``grid_thw``.
        num_tokens: Prompt length.
        downsample_ratio: Aligner fold size.

    Returns:
        ``int64 [num_tokens]`` token types.

    Raises:
        ValueError: If a feature's span length disagrees with its grid.
    """
    types = np.full((int(num_tokens),), -1, dtype=np.int64)
    for feature in features:
        grid = np.asarray(feature.grid_thw).reshape(-1)
        n_llm_h, n_llm_w = -(-int(grid[1]) // downsample_ratio), -(-int(grid[2]) // downsample_ratio)
        span = image_token_types(n_llm_h, n_llm_w)
        placeholder = feature.placeholder_range
        if placeholder is None or placeholder.length != span.shape[0]:
            raise ValueError("image feature span length disagrees with its patch grid.")
        types[placeholder.offset : placeholder.end] = span
    return types


def compute_prefill_embeddings(
    model: tp.Any, prompt_token_ids: tp.Sequence[int], features: tp.Sequence[MultiModalFeature]
) -> tuple[np.ndarray, np.ndarray]:
    """Eagerly encode the images and merge them into the prompt embeddings.

    Args:
        model: ``DeepseekV41ForConditionalGeneration``.
        prompt_token_ids: Expanded prompt ids.
        features: Image features (pixel values still present), any order.

    Returns:
        ``(embeds, token_types)``: ``float32 [S, hidden]`` merged embeddings (delimiters and aligner rows in the image
        spans) and ``int64 [S]`` token types.
    """
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
            grid = np.asarray(feature.grid_thw).reshape(-1)
            patches = jnp.asarray(feature.pixel_values, dtype=dtype)[None]
            out = base.get_image_features(patches, image_grid=(int(grid[1]), int(grid[2])))
            rows.append(out.reshape(-1, out.shape[-1]))
        merged = base.merge_image_embeddings(embeds, jnp.asarray(types)[None, :], jnp.concatenate(rows, axis=0)[None])
        host = np.asarray(jax.device_get(merged.astype(jnp.float32)))
    return host[0], types


__all__ = [
    "DeepseekV41ImageExpander",
    "EngramWindowHasher",
    "ExpandedImagePrompt",
    "build_engram_hasher",
    "compute_prefill_embeddings",
    "token_types_from_features",
]
