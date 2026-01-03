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

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.layers.base_modules import BaseVisionLanguageModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.modules.glm4v.modeling_glm4v import Glm4vModel

from .glm46v_configuration import Glm46VConfig


@register_module(TaskType.VISION_LM, config=Glm46VConfig, model_type="glm46v")
class Glm46VModel(Glm4vModel):
    """GLM-4.6V multimodal model integrating vision encoder and text decoder.

    This model extends GLM-4V with the GLM-4.6V architecture, combining a vision
    transformer encoder for image/video understanding with a text decoder for
    language generation. It supports both image and video inputs through a unified
    multimodal embedding pipeline with spatial merging and M-RoPE position encodings.

    Attributes:
        config (Glm46VConfig): Configuration for the multimodal model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
        visual: Vision transformer encoder for processing image/video inputs.
        language_model: Text decoder model for language generation.
    """

    def __init__(
        self,
        config: Glm46VConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GLM-4.6V multimodal base model.

        Args:
            config (Glm46VConfig): Model configuration containing vision and text configs.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Glm46VConfig, model_type="glm46v")
class Glm46VForConditionalGeneration(BaseVisionLanguageModule[Glm46VModel, Glm46VConfig]):
    """GLM-4.6V model for image/video-to-text conditional generation.

    This model combines the GLM-4.6V multimodal base model with a language modeling
    head for autoregressive text generation conditioned on visual inputs. It supports
    both image and video understanding tasks with multi-turn conversations.

    The model uses M-RoPE (Multi-dimensional Rotary Position Embedding) for handling
    spatial and temporal position information from visual inputs, and includes a
    spatial merge mechanism to efficiently process high-resolution images and videos.

    Attributes:
        config (Glm46VConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
        vocab_size (int): Size of the vocabulary.
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "glm46v"
    _config_class = Glm46VConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Glm46VConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GLM-4.6V model for conditional generation.

        Args:
            config (Glm46VConfig): Model configuration with vision, text, and generation settings.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Glm46VModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            image_token_index=config.image_token_id,
            video_token_index=config.video_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    @property
    def visual(self):
        """Access the vision encoder component.

        Returns:
            Glm4vVisionModel: The vision transformer encoder.
        """
        return self.base_model.visual

    @property
    def language_model(self):
        """Access the language model decoder component.

        Returns:
            Glm4vTextModel: The text decoder model.
        """
        return self.base_model.language_model

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw: Array | None = None, **kwargs):
        """Extract visual features from video inputs.

        Args:
            pixel_values_videos (Array): Video frames tensor with shape
                (num_frames * channels * temporal_patch * height * width).
            video_grid_thw (Array | None, optional): Grid dimensions (T, H, W) for each video.
                Required when pixel_values_videos is provided. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[Array, ...]: Tuple of video embeddings, one per video in the batch.
        """
        return self.base_model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | None = None, **kwargs):
        """Extract visual features from image inputs.

        Args:
            pixel_values (Array): Image tensor with shape matching vision encoder expectations.
            image_grid_thw (Array | None, optional): Grid dimensions (T, H, W) for each image.
                Required when pixel_values is provided. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[Array, ...]: Tuple of image embeddings, one per image in the batch.
        """
        return self.base_model.get_image_features(pixel_values, image_grid_thw)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute combined text and visual embeddings.

        Merges visual embeddings into text embeddings at placeholder token positions.

        Args:
            input_ids: Input token IDs with visual placeholder tokens.
            *args: Additional positional arguments passed to base model.
            **kwargs: Additional keyword arguments including pixel_values, image_grid_thw, etc.

        Returns:
            Array: Combined embeddings with visual features merged at placeholder positions.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
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
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        rope_deltas: Array | None = None,
        cache_position: Array | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for multimodal conditional generation.

        Processes text and visual inputs through the multimodal pipeline and generates
        logits for next token prediction.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Should contain image/video placeholder tokens where visual features will be inserted.
            attention_mask (Array | None, optional): Boolean mask for attention computation,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for M-RoPE, shape
                (3, batch_size, sequence_length) for temporal, height, width dimensions.
                Auto-computed if None. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language modeling head.
                Defaults to True.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
                Defaults to None.
            pixel_values (Array | None, optional): Image pixel values for visual encoding.
                Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixel values for visual encoding.
                Defaults to None.
            image_grid_thw (Array | None, optional): Grid dimensions (T, H, W) for images.
                Required when pixel_values is provided. Defaults to None.
            video_grid_thw (Array | None, optional): Grid dimensions (T, H, W) for videos.
                Required when pixel_values_videos is provided. Defaults to None.
            rope_deltas (Array | None, optional): Position deltas for M-RoPE computation.
                Defaults to None.
            cache_position (Array | None, optional): Cache position indices. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            VLMCausalLMOutput: Contains logits, past_key_values, hidden_states, attentions,
                last_hidden_state, and rope_deltas.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
        )

        hidden_states = apply_logical_sharding(
            outputs.last_hidden_state,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")
            lm_logits = self.apply_logit_cap(lm_logits)

        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )


__all__ = [
    "Glm46VConfig",
    "Glm46VForConditionalGeneration",
    "Glm46VModel",
]
