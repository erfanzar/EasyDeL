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

"""Mistral3 (Pixtral) vision-language model implementation.

Combines a Pixtral vision encoder, a spatial-merge + GELU multimodal projector,
and a Mistral text decoder. Visual features are extracted from a configurable
vision-encoder layer, downsampled by ``spatial_merge_size``, projected into the
text embedding space, and merged into the input sequence at positions marked
by the ``image_token_index``.

Exports:
    - ``Mistral3ModelOutput``, ``Mistral3CausalLMOutputWithPast``: structured outputs.
    - ``Mistral3PatchMerger``: spatial-merge module.
    - ``Mistral3MultiModalProjector``: vision-to-text projector.
    - ``Mistral3Model``: base multimodal model returning hidden states.
    - ``Mistral3ForConditionalGeneration``: full model with LM head for generation.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput, VLMCausalLMOutput
from easydel.infra.utils import ACT2FN
from easydel.layers import ParallelLinear, RMSNorm, RowParallelLinear
from easydel.modules._base import BaseVisionLanguageModule
from easydel.modules.auto.auto_modeling import AutoEasyDeLModel, AutoEasyDeLVisionModel

from .mistral3_configuration import Mistral3Config

logger = get_logger(__name__)


@auto_pytree
class Mistral3ModelOutput(BaseModelOutput):
    """Model output carrying text hidden states and optional projected image embeddings."""

    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


@dataclass
class _Mistral3LegacyOutputFields:
    """Seed dataclass field order for the public loss-first output contract."""

    loss: Array | None = None
    logits: Array | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None
    image_hidden_states: Array | None = None


@auto_pytree
class Mistral3CausalLMOutputWithPast(VLMCausalLMOutput, _Mistral3LegacyOutputFields):
    """Mistral3 autoregressive output, compatible with the shared VLM output.

    Retains the public loss-first positional constructor while supporting
    the shared output's additional fields, including last_hidden_state when
    the language-model head is disabled. Dictionary and iterable construction
    follow the inherited ModelOutput contract.
    Returned by :class:`Mistral3ForConditionalGeneration`.
    """

    # Dataclasses gather base fields in reverse MRO: the legacy mixin seeds
    # their order, then the shared VLM base supplies their current definitions
    # and appends extra fields. ModelOutput therefore sees loss as its first
    # field too, without a constructor or post-init compatibility adapter.


class Mistral3PatchMerger(spx.Module):
    """Concatenates neighbouring patches into channels, then projects back to ``hidden_size``.

    Mistral-3 / Pixtral patch tokens come in with one token per ``patch_size``
    pixels of the original image. This module reduces the token count by a
    factor of ``spatial_merge_size**2`` (typically 4× — every ``2×2`` patch
    block becomes one token) while preserving information by concatenating
    the ``k*k`` patches along the channel dimension and projecting back to
    ``hidden_size`` with a learnable bias-free linear. Critically, the
    operation is *aspect-ratio aware*: ``image_sizes`` carries the per-image
    pixel dimensions so the merge is computed on each image's actual
    rectangular patch grid (``H/patch_size`` × ``W/patch_size``) rather than
    a fixed square assumption. The split-merge-project loop iterates over
    images one at a time because each may have a different token count.

    Attributes:
        spatial_merge_size (int): Side of the spatial merge window
            (``k`` in the description above).
        patch_size (int): Vision tower's pixel patch size, used to convert
            ``image_sizes`` to grid coordinates.
        merging_layer (ParallelLinear): Concatenation projection
            ``hidden_size * k**2 -> hidden_size``.
    """

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mistral3 patch merger.

        Args:
            config (Mistral3Config): Model configuration with vision parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        hidden_size = config.vision_config.hidden_size * num_feature_layers
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = ParallelLinear(
            hidden_size * self.spatial_merge_size**2,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
        )

    def forward(
        self,
        image_features: jax.Array,
        image_sizes: jax.Array,
        *,
        patch_lengths: jax.Array | None = None,
        patch_grid_sizes: jax.Array | None = None,
        patch_attention_mask: jax.Array | None = None,
        return_attention_mask: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Merge complete spatial blocks in a packed, possibly padded patch stream.

        Args:
            image_features: Flattened image patches, shape (capacity, hidden_dim).
                Genuine patches form an image-ordered, row-major prefix.
            image_sizes: Original (height, width) pixel pairs. Concrete sizes retain
                the legacy ragged output shape; traced sizes use a fixed capacity.
            patch_lengths: Genuine patch count per image, excluding capacity padding.
            patch_grid_sizes: Genuine (height, width) patch counts from the vision
                tower. When omitted, inferred from image_sizes and patch_size.
            patch_attention_mask: Valid patches, shape (1, capacity) or (capacity,).
            return_attention_mask: Also return a validity mask for merged rows.
                Defaults to False, retaining the array-only public return value.

        Returns:
            Projected patches of shape (merged_capacity, hidden_dim), with valid
            complete blocks in a prefix and zeros elsewhere. Concrete image sizes
            set merged_capacity to the exact complete-block count; runtime geometry
            uses capacity // spatial_merge_size**2. When requested, also returns a
            boolean mask of shape (merged_capacity,).
        """
        k = self.spatial_merge_size
        capacity, d = image_features.shape
        # A runtime grid must not become a Python split/reshape dimension. Only
        # statically known input sizes can determine the exact ragged output size.
        concrete_sizes = image_sizes is not None and not any(
            isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(image_sizes)
        )
        if concrete_sizes:
            merged_capacity = sum(
                (int(height) // self.patch_size // k) * (int(width) // self.patch_size // k)
                for height, width in jax.device_get(image_sizes)
            )
        else:
            merged_capacity = capacity // (k * k)
        grids = jnp.asarray(patch_grid_sizes if patch_grid_sizes is not None else image_sizes, dtype=jnp.int32)
        if patch_grid_sizes is None:
            grids = grids // self.patch_size
        lengths = jnp.asarray(patch_lengths, dtype=jnp.int32) if patch_lengths is not None else jnp.prod(grids, -1)
        if grids.ndim != 2 or grids.shape[-1] != 2 or lengths.shape != grids.shape[:1]:
            raise ValueError("Patch geometry must have shapes (num_images, 2) and (num_images,)")
        if grids.shape[0] == 0 or capacity == 0 or merged_capacity == 0:
            merged = jnp.zeros((merged_capacity, d * k * k), dtype=image_features.dtype)
            valid = jnp.zeros((merged_capacity,), dtype=jnp.bool_)
        else:
            merged_lengths = jnp.prod(grids // k, axis=-1)
            merged_ends = jnp.cumsum(merged_lengths)
            rows = jnp.arange(merged_capacity, dtype=jnp.int32)
            image_index = jnp.minimum(jnp.sum(rows[:, None] >= merged_ends[None, :], axis=-1), grids.shape[0] - 1)
            merged_starts = merged_ends - merged_lengths
            patch_starts = jnp.cumsum(lengths) - lengths
            local_rows = rows - merged_starts[image_index]
            merged_width = jnp.maximum(grids[image_index, 1] // k, 1)
            top = (local_rows // merged_width) * k
            left = (local_rows % merged_width) * k
            offsets = jnp.arange(k * k, dtype=jnp.int32)
            indices = (
                patch_starts[image_index, None]
                + (top[:, None] + offsets[None, :] // k) * grids[image_index, 1, None]
                + left[:, None]
                + offsets[None, :] % k
            )
            valid = (rows < merged_ends[-1]) & jnp.all(
                (indices >= patch_starts[image_index, None])
                & (indices < (patch_starts + lengths)[image_index, None])
                & (indices < capacity),
                axis=-1,
            )
            if patch_attention_mask is not None:
                patch_mask = jnp.asarray(patch_attention_mask, dtype=jnp.bool_).reshape(-1)
                if patch_mask.shape != (capacity,):
                    raise ValueError("patch_attention_mask must have one entry per patch capacity row")
                valid &= jnp.all(jnp.take(patch_mask, indices, axis=0, mode="clip"), axis=-1)
            patches = jnp.take(image_features, indices, axis=0, mode="clip")
            # HF unfold is channel-major (channel, block_row, block_column);
            # checkpoint merging_layer weights must not be permuted.
            merged = patches.transpose(0, 2, 1).reshape(merged_capacity, d * k * k)
            merged = jnp.where(valid[:, None], merged, 0)
        image_features = jnp.where(valid[:, None], self.merging_layer(merged), 0)
        return (image_features, valid) if return_attention_mask else image_features


class Mistral3MultiModalProjector(spx.Module):
    """Vision-to-text bridge for Mistral-3 / Pixtral: norm + patch merge + MLP.

    Mistral-3 (and Pixtral, the underlying vision tower) supports
    *aspect-ratio aware* image encoding, so the patch grid is rectangular
    and the number of vision tokens varies per image. The projector first
    pre-normalizes the patch features (``RMSNorm`` at the *text* model's
    epsilon — the vision tower already RMSNorm'd internally, but cross-
    modality fine-tuning behaves better with another normalization here),
    then runs a :class:`Mistral3PatchMerger` that downsamples the patch
    grid by concatenating ``spatial_merge_size**2`` neighbouring patches
    along the channel dimension, and finally applies the standard
    LLaVA-style ``linear -> act -> linear`` MLP to land in the LM hidden
    space. Per-image tokens are produced respecting the actual aspect ratio
    described by ``image_sizes``.

    Attributes:
        norm (RMSNorm): Pre-projection RMSNorm at the text model's epsilon.
        patch_merger (Mistral3PatchMerger): Spatial-to-channel concatenation
            using ``image_sizes`` to handle non-square inputs.
        linear_1 (RowParallelLinear): Vision-to-text expansion projection.
        act (Callable): Activation between the two linears (typically GELU).
        linear_2 (RowParallelLinear): Square ``text_hidden_size`` projection.
    """

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mistral3 multimodal projector.

        Args:
            config (Mistral3Config): Model configuration with vision and text parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.norm = RMSNorm(
            config.vision_config.hidden_size * num_feature_layers,
            eps=config.get_text_config().rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.patch_merger = Mistral3PatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = RowParallelLinear(
            config.vision_config.hidden_size * num_feature_layers,
            config.get_text_config().hidden_size,
            use_bias=config.multimodal_projector_bias,
            kernel_init=jax.nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = RowParallelLinear(
            config.get_text_config().hidden_size,
            config.get_text_config().hidden_size,
            use_bias=config.multimodal_projector_bias,
            kernel_init=jax.nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        image_features: jax.Array,
        image_sizes: jax.Array,
        *,
        patch_lengths: jax.Array | None = None,
        patch_grid_sizes: jax.Array | None = None,
        patch_attention_mask: jax.Array | None = None,
        return_attention_mask: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Project genuine spatial patch blocks into language model space.

        Args:
            image_features: Packed vision features, shape (capacity, vision_hidden_dim).
            image_sizes: Original (height, width) pixel pairs.
            patch_lengths: Genuine patch counts per image from the vision tower.
            patch_grid_sizes: Genuine (height, width) patch counts per image.
            patch_attention_mask: Valid patches, shape (1, capacity) or (capacity,).
            return_attention_mask: Also return merged-row validity. Defaults to False.

        Returns:
            Features of shape (merged_capacity, text_hidden_dim). Concrete sizes
            retain the exact ragged shape; runtime sizes produce a valid prefix
            followed by zeros, including when projection linears have biases.
            When requested, also returns the boolean merged-row validity mask.
        """
        image_features = self.norm(image_features)
        image_features, valid = self.patch_merger(
            image_features,
            image_sizes,
            patch_lengths=patch_lengths,
            patch_grid_sizes=patch_grid_sizes,
            patch_attention_mask=patch_attention_mask,
            return_attention_mask=True,
        )
        hidden_states = checkpoint_name(self.linear_1(image_features), name="projector_linear1")
        hidden_states = self.act(hidden_states)
        hidden_states = checkpoint_name(self.linear_2(hidden_states), name="projector_linear2")
        hidden_states = jnp.where(valid[:, None], hidden_states, 0)
        return (hidden_states, valid) if return_attention_mask else hidden_states


@register_module(TaskType.BASE_MODULE, config=Mistral3Config, model_type="mistral3")
class Mistral3Model(EasyDeLBaseModule):
    """Mistral-3 base trunk: Pixtral vision tower + patch-merging projector + Mistral text decoder.

    Same three-stage architecture as :class:`LlavaModel` but with a
    *Pixtral* aspect-ratio-aware vision tower instead of CLIP-ViT and a
    :class:`Mistral3PatchMerger` inside the projector to compress the
    rectangular patch grid before the LM consumes it. The trunk does not
    own an LM head — that lives on
    :class:`Mistral3ForConditionalGeneration`. Image features are spliced
    into the embedding sequence at every position whose token id equals
    ``config.image_token_index``.

    Attributes:
        vision_tower: Pixtral vision encoder built via
            :class:`AutoEasyDeLVisionModel`.
        multi_modal_projector (Mistral3MultiModalProjector): RMSNorm + patch
            merge + two-layer MLP bridge.
        language_model: Mistral text decoder trunk (no LM head).
    """

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mistral3 base model.

        Args:
            config (Mistral3Config): Model configuration with vision and text parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_tower = AutoEasyDeLVisionModel.from_config(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.multi_modal_projector = Mistral3MultiModalProjector(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = config.get_text_config().vocab_size
        self.language_model = AutoEasyDeLModel.from_config(
            config=config.get_text_config(),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.vision_feature_layer = config.vision_feature_layer

    def get_image_features(
        self, pixel_values: Array, image_sizes: Array, *, return_valid_length: bool = False
    ) -> Array | tuple[Array, Array]:
        """Extract and project image features from pixel values.

        Processes images through the vision tower, selects features from the specified
        layer, and projects them into the language model's embedding space.

        Args:
            pixel_values (Array): Input image pixel values of shape
                (batch, channels, height, width).
            image_sizes (Array): Original image sizes as (height, width) pairs
                for proper patch merging.
            return_valid_length (bool): Also return the number of genuine projected
                rows for merging capacity-padded features. Defaults to False.

        Returns:
            Array: Projected image features of shape (merged_capacity, hidden_dim),
                with genuine rows first and any runtime-capacity padding zeroed.
                Optionally also returns the scalar integer genuine-row count.
        """
        image_features = self.vision_tower(pixel_values, output_hidden_states=True, image_sizes=image_sizes)
        if isinstance(self.vision_feature_layer, int):
            selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        else:
            selected_image_feature = jnp.concatenate(
                [image_features.hidden_states[layer_idx] for layer_idx in self.vision_feature_layer],
                axis=-1,
            )
        projected = self.multi_modal_projector(
            selected_image_feature.squeeze(0),
            image_sizes,
            patch_lengths=getattr(image_features, "patch_lengths", None),
            patch_grid_sizes=getattr(image_features, "patch_grid_sizes", None),
            patch_attention_mask=getattr(image_features, "patch_attention_mask", None),
            return_attention_mask=return_valid_length,
        )
        if return_valid_length:
            features, valid = projected
            return features, jnp.sum(valid, dtype=jnp.int32)
        return projected

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        image_features: Array | None = None,
        image_features_valid_length: Array | int | None = None,
        pixel_values: Array | None = None,
        image_sizes: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute combined text and image embeddings.

        Creates embeddings for text tokens and optionally merges image features at
        positions marked by the image token ID, enabling multimodal input processing.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            image_features (Array | None, optional): Pre-computed image features.
                If None and pixel_values is provided, features are computed.
                Defaults to None.
            image_features_valid_length (Array | int | None, optional): Number of
                genuine rows in capacity-padded precomputed features. None requires
                every row to correspond to an image placeholder. Defaults to None.
            pixel_values (Array | None, optional): Raw image pixel values.
                Used to compute image_features if not provided. Defaults to None.
            image_sizes (Array | None, optional): Original image sizes for patch merging.
                Required when pixel_values is provided. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Array: Combined embeddings with image features merged at image token positions,
                shape (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If input_ids is None or if pixel_values is provided without image_sizes.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        image_token_id = self.config.image_token_index
        if image_token_id >= self.vocab_size:
            llm_input_ids = jnp.where(input_ids == image_token_id, 0, input_ids)
        else:
            llm_input_ids = input_ids

        inputs_embeds = super().compute_embedding(llm_input_ids)

        if image_features is None and pixel_values is not None:
            if image_sizes is None:
                raise ValueError("`image_sizes` must be provided when `pixel_values` is not None.")
            image_features, image_features_valid_length = self.get_image_features(
                pixel_values, image_sizes, return_valid_length=True
            )

        if image_features is not None:
            multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype)
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=image_token_id,
                multimodal_embeddings_valid_length=image_features_valid_length,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_sizes: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ):
        """Forward pass through the Mistral3 base model.

        Processes multimodal inputs by encoding images through the vision tower,
        merging image features with text embeddings, and forwarding through
        the language model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            pixel_values (Array | None, optional): Input image pixel values of shape
                (batch, channels, height, width). Defaults to None.
            image_sizes (Array | None, optional): Original image sizes as (height, width) pairs.
                Required when pixel_values is provided. Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            **lm_kwargs: Additional arguments passed to the language model.

        Returns:
            Mistral3ModelOutput: Contains last_hidden_state, past_key_values, hidden_states,
                attentions, and image_hidden_states.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None,
                or if pixel_values is provided without input_ids.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if pixel_values is not None and input_ids is None:
            raise ValueError("`input_ids` must be provided when `pixel_values` is not None.")

        image_features = None
        image_features_valid_length = None
        if pixel_values is not None:
            if image_sizes is None:
                raise ValueError("`image_sizes` must be provided when `pixel_values` is not None.")
            image_features, image_features_valid_length = self.get_image_features(
                pixel_values, image_sizes, return_valid_length=True
            )

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
                image_features_valid_length=image_features_valid_length,
            )
        outputs = self.language_model(
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            **lm_kwargs,
        )

        return Mistral3ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ):
        """Initialize KV cache for autoregressive generation.

        Creates and returns an empty cache structure for the language model
        to store key-value pairs during generation.

        Args:
            batch_size (int): Number of sequences in the batch.
            max_length (int): Maximum sequence length for the cache.
            starts (int | None, optional): Starting positions for each sequence.
                Defaults to None.
            shardings (dict | None, optional): Sharding configuration for distributed
                cache. Defaults to None.
            pad_token_id (int | None, optional): Token ID used for padding.
                Defaults to None.

        Returns:
            TransformerCache: Initialized empty cache for key-value storage.
        """
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        image_sizes: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ):
        """Prepare model inputs for autoregressive generation.

        Sets up the input dictionary with necessary tensors for generation,
        including optional pixel values for multimodal generation.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            max_length (int): Maximum generation length.
            pad_token_id (int): Token ID used for padding.
            starts (int | None, optional): Starting positions. Defaults to None.
            pixel_values (Array | None, optional): Image pixel values for multimodal
                generation. Defaults to None.
            image_sizes (Array | None, optional): Image size information for each image
                in the batch. Defaults to None.
            attention_mask (Array | None, optional): Attention mask for the inputs.
                Defaults to None.

        Returns:
            dict: Dictionary containing prepared inputs for generation.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs["pixel_values"] = pixel_values
        model_inputs["image_sizes"] = image_sizes
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Update model inputs for the next generation step.

        Updates the input arguments with new values from the model outputs,
        and removes pixel_values after the first iteration since image features
        are only needed once.

        Args:
            model_outputs: Outputs from the previous forward pass.
            model_kwargs (dict): Current model input arguments.

        Returns:
            dict: Updated model input arguments for the next generation step.
        """
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        model_kwargs.pop("image_sizes", None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        The vision tower acts as the encoder in this multi-modal setup.
        """
        return self.vision_tower

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.language_model

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.language_model.embed_tokens


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Mistral3Config, model_type="mistral3")
class Mistral3ForConditionalGeneration(BaseVisionLanguageModule[Mistral3Model, Mistral3Config]):  # type: ignore
    """Mistral-3 / Pixtral image-to-text VLM with LM head.

    Adds the language modelling head on top of :class:`Mistral3Model` and
    plumbs in the VLM-specific configuration consumed by
    :class:`BaseVisionLanguageModule` (vision feature layer index,
    feature-select strategy, image token id). Image-only — does not
    consume video frames — and uses standard RoPE on the text decoder
    rather than M-RoPE.

    The image embedding API is sligthly richer than LLaVA's: this model's
    :meth:`get_image_features` requires ``image_sizes`` so the patch
    merger can handle each image's actual aspect ratio rather than
    assuming a fixed square grid.

    Class flags expose the capability surface that the surrounding
    ``BaseVisionLanguageModule`` machinery uses for routing:

    * ``_supports_video = False`` — image-only.
    * ``_uses_mrope = False`` — text decoder uses standard 1-D RoPE.
    """

    # Class attributes for registration and capabilities
    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "mistral3"
    _config_class = Mistral3Config
    _auto_register = False  # Already registered via decorator
    _supports_video = False
    _uses_mrope = False

    _vision_tower_name = "vision_tower"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mistral3 model for conditional generation.

        Args:
            config (Mistral3Config): Model configuration with vision and text parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Mistral3Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=config.vision_feature_layer,
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "default"),
            image_token_index=config.image_token_index,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        image_sizes: Array | None = None,
        *,
        return_valid_length: bool = False,
        **kwargs,
    ) -> Array | tuple[Array, Array]:
        """Extract and project image features from pixel values.

        Mistral3 uses a patch merger that requires image_sizes to handle
        variable-sized images.

        Args:
            pixel_values: Input image pixel values
            image_sizes: Original sizes of the images (height, width) for patch merging
            return_valid_length: Also return the genuine projected-row count for
                merging capacity-padded features. Defaults to False.
            **kwargs: Additional arguments (unused)

        Returns:
            Projected image features of shape (merged_capacity, hidden_dim),
            optionally paired with the scalar integer genuine-row count.
        """
        return self.base_model.get_image_features(
            pixel_values=pixel_values, image_sizes=image_sizes, return_valid_length=return_valid_length
        )

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute combined text and image embeddings.

        Delegates to the base model's compute_embedding method for multimodal
        embedding computation.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments passed to base model.
            **kwargs: Keyword arguments including image_features, pixel_values, image_sizes.

        Returns:
            Array: Combined embeddings with image features merged at image token positions.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_sizes: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> Mistral3CausalLMOutputWithPast:
        """Forward pass for the Mistral3 model.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            pixel_values: Input pixel values for images
            image_sizes: Original sizes of images for patch merging
            attention_mask: Attention mask
            mask_info: Mask information
            position_ids: Position IDs for text
            mode: Runtime mode
            past_key_values: Cached keys/values for language model
            cache_metadata: Metadata for paged attention
            apply_lm_head: Whether to apply the LM head
            inputs_embeds: Input embeddings (alternative to input_ids)
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            **lm_kwargs: Additional arguments passed to the language model

        Returns:
            VLMCausalLMOutput: Model outputs including logits and optional states
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Forward through base model (handles image_sizes via kwargs)
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            image_sizes=image_sizes,
            cache_metadata=cache_metadata,
            mode=mode,
            **lm_kwargs,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(hidden_states)
            lm_logits = self.apply_logit_cap(lm_logits)

        return Mistral3CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ):
        """Initialize KV cache for autoregressive generation.

        Creates and returns an empty cache structure for storing key-value pairs
        during generation.

        Args:
            batch_size (int): Number of sequences in the batch.
            max_length (int): Maximum sequence length for the cache.
            starts (int | None, optional): Starting positions for each sequence.
                Defaults to None.
            shardings (dict | None, optional): Sharding configuration for distributed
                cache. Defaults to None.
            pad_token_id (int | None, optional): Token ID used for padding.
                Defaults to None.

        Returns:
            TransformerCache: Initialized empty cache for key-value storage.
        """
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def get_vision_tower(self) -> spx.Module:
        """Get the vision tower component.

        Returns:
            spx.Module: The vision encoder used for image feature extraction.
        """
        return self.base_model.vision_tower

    def get_projector(self) -> spx.Module:
        """Get the multimodal projector component.

        Returns:
            spx.Module: The projector that aligns vision features with text embeddings.
        """
        return self.base_model.multi_modal_projector

    def get_language_model(self) -> spx.Module:
        """Get the language model component.

        Returns:
            spx.Module: The underlying language model for text generation.
        """
        return self.base_model.language_model
