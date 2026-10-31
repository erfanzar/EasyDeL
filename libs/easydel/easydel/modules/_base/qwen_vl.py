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

"""Parameter-free Qwen-style vision embedding and DeepStack infrastructure.

Put ``QwenVLMultimodalMixin`` before the model base in the MRO. It expects the
existing ``visual``, ``config``, ``get_input_embeddings`` and ``get_rope_index``
interfaces and creates no parameters or module attributes. Text decoders can
use ``DeepStackVisualMixin`` independently of the multimodal embedding API.
"""

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from easydel.infra.modeling_outputs import EmbeddingInfo


class DeepStackVisualMixin:
    """Add compact visual residuals without changing a decoder's parameter tree."""

    def _deepstack_process(self, hidden_states: Array, visual_pos_masks: Array, visual_embeds: Array) -> Array:
        """Inject one layer's visual residuals at flattened placeholder positions.

        Args:
            hidden_states: Language activations, shaped ``[batch, sequence, hidden]``.
            visual_pos_masks: Boolean placeholder mask, shaped ``[batch, sequence]``.
            visual_embeds: Compact residual table, shaped ``[visual_tokens, hidden]``.
                Preprocessing must validate its row count against the mask.

        Returns:
            Activations with residuals added only at visual positions, preserving dtype.
        """
        hidden_dim = hidden_states.shape[-1]
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        flat_mask = visual_pos_masks.reshape(-1)
        # A real pad row also makes a zero-visual-token batch a valid no-op.
        visual_embeds = visual_embeds.astype(hidden_states.dtype)
        padded = jnp.concatenate([jnp.zeros((1, hidden_dim), dtype=hidden_states.dtype), visual_embeds], axis=0)
        updates = padded[jnp.cumsum(flat_mask)]
        return jnp.where(flat_mask[:, None], flat_hidden + updates, flat_hidden).reshape(hidden_states.shape)

    @staticmethod
    def merge_deepstack_features(
        image_mask: Array | None,
        video_mask: Array | None,
        deepstack_image_embeds: list[Array] | None,
        deepstack_video_embeds: list[Array] | None,
    ) -> tuple[Array | None, list[Array] | None]:
        """Interleave per-media DeepStack tables in sequence placeholder order.

        Args:
            image_mask: Image placeholder mask, or None when no image table is present.
            video_mask: Video placeholder mask, or None when no video table is present.
            deepstack_image_embeds: Per-layer compact image residual tables.
            deepstack_video_embeds: Per-layer compact video residual tables.

        Returns:
            Joint visual mask and per-layer residual tables. A single modality is
            returned unchanged. With both modalities, the shared number of layers
            is used, matching the existing Qwen dense/MoE zip semantics. Static
            table lengths determine the joint size, so no boolean-indexed dynamic
            array shapes are required under JIT. Counts must be validated before
            tracing, as for the multimodal embedding merge.
        """
        if image_mask is None:
            return video_mask, deepstack_video_embeds
        if video_mask is None:
            return image_mask, deepstack_image_embeds
        joint_mask = image_mask | video_mask
        joint_embeds = []
        for image, video in zip(deepstack_image_embeds or [], deepstack_video_embeds or [], strict=False):
            count = image.shape[0] + video.shape[0]
            positions = jnp.nonzero(joint_mask.reshape(-1), size=count, fill_value=0)[0]
            image_positions = image_mask.reshape(-1)[positions]
            # The compact concatenated table is gathered in flattened sequence order.
            image_indices = jnp.cumsum(image_positions) - 1
            video_indices = image.shape[0] + jnp.cumsum(~image_positions) - 1
            indices = jnp.where(image_positions, image_indices, video_indices)
            joint_embeds.append(jnp.concatenate([image, video.astype(image.dtype)], axis=0)[indices])
        return joint_mask, joint_embeds


class QwenVLMultimodalMixin(DeepStackVisualMixin):
    """Share vision encoding, placeholder validation and embedding preparation.

    This mixin intentionally does not implement ``forward`` or construct outputs:
    each family retains its own decoder call, output type and MoE router fields.
    Both ``forward`` and ``compute_embedding_with_info`` can consume the same
    ``prepare_multimodal_embeddings`` result, including precomputed DeepStack data.
    """

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode images and split merged tokens using host grid metadata.

        Args:
            pixel_values: Flattened image patches accepted by the vision tower.
            image_grid_thw: Concrete pre-merge temporal/height/width triples.
            image_max_grid_size: Optional maximum spatial grid dimension.

        Returns:
            Per-image merged embeddings and per-layer DeepStack residuals.

        Raises:
            ValueError: If grid metadata is absent or is not a positive ``[N, 3]`` grid.
        """
        if image_grid_thw is None:
            raise ValueError("`image_grid_thw` must be provided to encode vision patches.")
        grid_thw = np.asarray(image_grid_thw)
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or not grid_thw.size or np.any(grid_thw <= 0):
            raise ValueError("`image_grid_thw` must contain positive temporal/height/width triples.")
        if image_max_grid_size is None:
            image_max_grid_size = int(grid_thw[:, 1:].max())
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values.astype(self.visual.get_dtype()), grid_thw=grid_thw, max_grid_size=image_max_grid_size
        )
        split_sizes = np.prod(grid_thw, axis=-1) // (self.visual.spatial_merge_size**2)
        split_points = np.cumsum(split_sizes[:-1]).tolist()
        return tuple(jnp.split(image_embeds, split_points, axis=0)), deepstack_image_embeds

    def get_video_features(
        self,
        pixel_values_videos: Array,
        video_grid_thw: Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode video patches with the same grid and DeepStack semantics as images.

        Args:
            pixel_values_videos: Flattened video patches.
            video_grid_thw: Concrete pre-merge grid triples.
            video_max_grid_size: Optional maximum spatial grid dimension.

        Returns:
            Per-video merged embeddings and per-layer DeepStack residuals.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_placeholder_mask(
        self,
        input_ids: Array | None,
        inputs_embeds: Array,
        image_features: Array | None = None,
        video_features: Array | None = None,
    ) -> tuple[Array, Array]:
        """Find image/video placeholders and validate concrete feature counts.

        Args:
            input_ids: Token IDs, or None to identify placeholders by embedding equality.
            inputs_embeds: Original text embeddings, before visual replacement.
            image_features: Optional image feature table to validate.
            video_features: Optional video feature table to validate.

        Returns:
            Boolean image and video placeholder masks.

        Raises:
            ValueError: If concrete token counts and supplied feature sizes disagree.

        Note:
            Data-dependent counts under JIT must be validated by preprocessing;
            this method introduces no host callbacks into compiled model execution.
        """
        if input_ids is None:
            embedding = self.get_input_embeddings()
            image_ref = embedding(jnp.array(self.config.image_token_id, dtype=jnp.int32))
            video_ref = embedding(jnp.array(self.config.video_token_id, dtype=jnp.int32))
            image_mask = jnp.all(inputs_embeds == image_ref, axis=-1)
            video_mask = jnp.all(inputs_embeds == video_ref, axis=-1)
        else:
            image_mask = input_ids == self.config.image_token_id
            video_mask = input_ids == self.config.video_token_id
        for name, mask, features in (("Image", image_mask, image_features), ("Video", video_mask, video_features)):
            if features is not None:
                count = mask.sum()
                if not isinstance(count, jax.core.Tracer) and int(count) * inputs_embeds.shape[-1] != features.size:
                    raise ValueError(
                        f"{name} features and {name.lower()} tokens do not match: "
                        f"tokens: {count}, features {features.shape[0]}"
                    )
        return image_mask, video_mask

    def prepare_multimodal_embeddings(
        self,
        input_ids: Array | None,
        *,
        inputs_embeds: Array | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        visual_pos_masks: Array | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
    ) -> tuple[Array, EmbeddingInfo]:
        """Prepare fused embeddings and DeepStack data for either public entry point.

        Args:
            input_ids: Token IDs, or None with supplied text embeddings.
            inputs_embeds: Optional precomputed text embeddings.
            pixel_values: Optional image patches; skipped when image_embeds is provided.
            pixel_values_videos: Optional video patches; skipped when video_embeds is provided.
            image_grid_thw: Concrete image grid metadata.
            video_grid_thw: Concrete video grid metadata.
            image_max_grid_size: Optional static image grid bound.
            video_max_grid_size: Optional static video grid bound.
            image_embeds: Optional precomputed merged image features.
            video_embeds: Optional precomputed merged video features.
            visual_pos_masks: Optional precomputed joint DeepStack placeholder mask.
            deepstack_visual_embeds: Optional precomputed per-layer joint residuals.

        Returns:
            Fused embeddings and EmbeddingInfo containing visual masks/residuals.
            Position IDs and RoPE deltas are left to the caller's position policy.
            No model state or checkpoint paths are changed.

        Raises:
            ValueError: If neither IDs nor embeddings is provided, or precomputed
                DeepStack residuals lack a mask and no pixel-derived mask is available.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")
            inputs_embeds = super().compute_embedding(input_ids)
        image_deepstack = video_deepstack = None
        if image_embeds is None and pixel_values is not None:
            images, image_deepstack = self.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)
            image_embeds = jnp.concatenate(images, axis=0)
        if video_embeds is None and pixel_values_videos is not None:
            videos, video_deepstack = self.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)
            video_embeds = jnp.concatenate(videos, axis=0)
        image_mask = video_mask = None
        if image_embeds is not None or video_embeds is not None:
            image_mask, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_embeds, video_embeds)
            for features, mask in ((image_embeds, image_mask), (video_embeds, video_mask)):
                if features is not None:
                    hidden = inputs_embeds.shape[-1]
                    padded = jnp.concatenate(
                        [jnp.zeros((1, hidden), dtype=inputs_embeds.dtype), features.astype(inputs_embeds.dtype)], axis=0
                    )
                    updates = padded[jnp.cumsum(mask.reshape(-1))].reshape(inputs_embeds.shape)
                    inputs_embeds = jnp.where(mask[..., None], updates, inputs_embeds)
        computed_mask, computed_deepstack = self.merge_deepstack_features(
            image_mask if image_deepstack else None,
            video_mask if video_deepstack else None,
            image_deepstack,
            video_deepstack,
        )
        if computed_mask is not None:
            visual_pos_masks, deepstack_visual_embeds = computed_mask, computed_deepstack
        elif deepstack_visual_embeds is not None and visual_pos_masks is None:
            raise ValueError("`visual_pos_masks` must be provided when `deepstack_visual_embeds` is not None.")
        return inputs_embeds, EmbeddingInfo(
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            deepstack_image_embeds=image_deepstack,
            deepstack_video_embeds=video_deepstack,
        )

    def compute_embedding(
        self,
        input_ids: Array | None,
        *,
        inputs_embeds: Array | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute text/vision embeddings, preserving the Qwen family signature.

        Args:
            input_ids: Token IDs, optional when inputs_embeds is supplied.
            inputs_embeds: Optional precomputed text embeddings.
            pixel_values: Optional image patches.
            pixel_values_videos: Optional video patches.
            image_grid_thw: Image grid metadata.
            video_grid_thw: Video grid metadata.
            image_max_grid_size: Optional image grid bound.
            video_max_grid_size: Optional video grid bound.
            image_embeds: Optional precomputed image features, taking priority over pixels.
            video_embeds: Optional precomputed video features, taking priority over pixels.
            **kwargs: Additional compatibility arguments.

        Returns:
            Fused text and visual embeddings.
        """
        embeddings, _ = self.prepare_multimodal_embeddings(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
        )
        return embeddings

    def compute_embedding_with_info(
        self,
        input_ids: Array | None,
        *,
        inputs_embeds: Array | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        attention_mask: Array | None = None,
        **kwargs,
    ) -> tuple[Array, EmbeddingInfo]:
        """Compute fused embeddings, RoPE positions and DeepStack metadata together.

        Args:
            input_ids: Token IDs required by the family RoPE indexing interface.
            inputs_embeds: Optional precomputed text embeddings.
            pixel_values: Optional image patches.
            pixel_values_videos: Optional video patches.
            image_grid_thw: Image grid metadata.
            video_grid_thw: Video grid metadata.
            image_max_grid_size: Optional image grid bound.
            video_max_grid_size: Optional video grid bound.
            image_embeds: Optional precomputed image features.
            video_embeds: Optional precomputed video features.
            attention_mask: Optional text attention mask for RoPE indexing.
            **kwargs: Additional compatibility arguments.

        Returns:
            Fused embeddings and metadata containing positions, deltas and visual residuals.

        Raises:
            ValueError: If input_ids is absent.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding_with_info`.")
        embeddings, info = self.prepare_multimodal_embeddings(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
        )
        info.position_ids, info.rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return embeddings, info
