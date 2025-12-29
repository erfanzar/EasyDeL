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
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import ModelOutput, VLMCausalLMOutput
from easydel.infra.utils import ACT2FN
from easydel.layers.base_modules import BaseVisionLanguageModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.layers.linear import RowParallelLinear
from easydel.modules.auto.auto_modeling import AutoEasyDeLModel, AutoEasyDeLVisionModel

from .aya_vision_configuration import AyaVisionConfig

logger = get_logger(__name__)


@auto_pytree
class AyaVisionCausalLMOutputWithPast(ModelOutput):
    """
    Base class for AyaVision causal language model (or autoregressive) outputs.

    Args:
        loss (`Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(Array))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(Array)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(Array)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            Tuple of `Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`Array`, *optional*):
            An `Array` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    last_hidden_state: Array | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


class AyaVisionMultiModalProjector(nn.Module):
    """
    A multi-modal projector module for AyaVision that processes image features.
    It applies pixel shuffling, layer normalization, and linear transformations.

    Attributes:
        config (AyaVisionConfig): Configuration object.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): JAX precision level.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: AyaVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the AyaVisionMultiModalProjector.

        Args:
            config (AyaVisionConfig): Model configuration containing vision/text settings and projector parameters.
            dtype (jnp.dtype): Computation dtype for activations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Parameter storage dtype for weights. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): JAX matmul precision setting (e.g., jax.lax.Precision.DEFAULT).
                None uses default precision.
            rngs (nn.Rngs): Flax NNX random number generators for parameter initialization.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.downsample_factor = config.downsample_factor
        self.alignment_intermediate_size = getattr(
            config,
            "alignment_intermediate_size",
            config.get_text_config().hidden_size,
        )

        self.layernorm = nn.LayerNorm(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            epsilon=config.adapter_layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.linear_1 = RowParallelLinear(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            self.alignment_intermediate_size,
            use_bias=True,
            kernel_init=nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

        self.act = ACT2FN["silu"]
        self.linear_2 = RowParallelLinear(
            self.alignment_intermediate_size // 2,
            config.get_text_config().hidden_size,
            use_bias=True,
            kernel_init=nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, image_features: jax.Array) -> jax.Array:
        """Forward pass through the projector.

        Applies pixel shuffling for spatial downsampling, layer normalization, and gated
        linear projections to transform vision features to the language model's hidden dimension.

        Args:
            image_features (jax.Array): Input image features from vision encoder.
                Shape: (batch_size, num_patches, vision_hidden_size)
                Example: (1, 576, 1152) for 384x384 image with patch_size=14

        Returns:
            jax.Array: Projected features ready for language model.
                Shape: (batch_size, downsampled_patches, text_hidden_size)
                Example: (1, 144, 4096) after 2x downsampling to Cohere2's hidden size
        """
        image_features = self.pixel_shuffle(image_features)
        image_features = self.layernorm(image_features)
        hidden_states = checkpoint_name(self.linear_1(image_features), name="mlp_gate")
        x, gate = jnp.split(hidden_states, 2, axis=-1)
        hidden_states = self.act(gate) * x

        hidden_states = checkpoint_name(self.linear_2(hidden_states), name="mlp_output")
        return hidden_states

    def pixel_shuffle(self, image_features: jax.Array) -> jax.Array:
        """Performs pixel shuffling on the image features based on the downsample factor.

        Args:
            image_features (jax.Array): Input image features (batch_size, seq_length, hidden_size).

        Returns:
            jax.Array: Image features after pixel shuffling.
        """
        batch_size, seq_length, _ = image_features.shape
        height = width = int(seq_length**0.5)
        image_features = image_features.reshape(
            image_features.shape[0],
            width,
            height,
            -1,
        )
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size,
            width,
            int(height / self.downsample_factor),
            int(channels * self.downsample_factor),
        )
        image_features = jnp.transpose(image_features, (0, 2, 1, 3))
        image_features = image_features.reshape(
            batch_size,
            int(height / self.downsample_factor),
            int(width / self.downsample_factor),
            -1,
        )
        image_features = jnp.transpose(image_features, (0, 2, 1, 3))
        return image_features


@register_module(TaskType.BASE_VISION, config=AyaVisionConfig, model_type="aya_vision")
class AyaVisionModel(EasyDeLBaseModule):
    """
    AyaVision model for conditional text generation based on image inputs.
    Combines a vision tower and a language model with a multi-modal projector.

    Attributes:
        config (AyaVisionConfig): Configuration object.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): JAX precision level.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: AyaVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the AyaVisionModel (base model without LM head).

        Args:
            config (AyaVisionConfig): Model configuration with vision, text, and projector settings.
            dtype (jnp.dtype): Computation dtype for activations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Parameter storage dtype for weights. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): JAX matmul precision. Defaults to None (uses default precision).
            rngs (nn.Rngs): Flax NNX random number generators for initializing all submodules.
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
        self.multi_modal_projector = AyaVisionMultiModalProjector(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = AutoEasyDeLModel.from_config(
            config=config.get_text_config(),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", "default")

    def get_image_features(self, pixel_values: Array) -> Array:
        """Extracts and projects image features from the vision tower.

        Args:
            pixel_values (Array): Input pixel values for the images.

        Returns:
            Array: Processed image features ready for the language model.
        """
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)

        return image_features

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        image_features: Array | None = None,
        pixel_values: Array | None = None,
        **kwargs,
    ) -> Array:
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        vocab_size = self.config.get_text_config().vocab_size
        image_token_id = self.config.image_token_index
        if image_token_id >= vocab_size:
            llm_input_ids = jnp.where(input_ids == image_token_id, 0, input_ids)
        else:
            llm_input_ids = input_ids

        inputs_embeds = super().compute_embedding(llm_input_ids)

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if image_features is not None:
            multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype)
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=image_token_id,
            )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
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
        """Forward pass for the AyaVision base model (no LM head).

        Processes images through the vision tower, merges image features with text embeddings,
        and runs the combined embeddings through the language model.

        Args:
            input_ids (Int[Array, "batch seq_len"]): Input token IDs including image token placeholders.
                Shape: (batch_size, sequence_length). Image tokens should use config.image_token_index (255036).
            pixel_values (Array): Input pixel values for images.
                Shape: (batch_size, num_channels, height, width), e.g., (1, 3, 384, 384)
            attention_mask (Bool[Array, "batch seq_len"] | None): Attention mask for text tokens.
                Shape: (batch_size, sequence_length). True/1 for valid tokens, False/0 for padding.
            mask_info (MaskInfo | None): Structured mask information for efficient attention computation.
            position_ids (Int[Array, "batch seq_len"] | None): Position IDs for positional embeddings.
                Shape: (batch_size, sequence_length)
            mode (common_types.RUNTIME_MODE_TYPES | None): Runtime mode (e.g., "train", "eval", "generate").
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None): Cached attention
                keys/values from previous generation steps for faster autoregressive decoding.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None):
                Metadata for paged attention and cache management.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"] | None): Pre-computed input embeddings.
                If provided, input_ids must be None. Shape: (batch_size, sequence_length, hidden_size)
            output_attentions (bool | None): Whether to return attention weights from all layers.
            output_hidden_states (bool | None): Whether to return hidden states from all layers.
            **lm_kwargs: Additional keyword arguments passed to the language model.

        Returns:
            AyaVisionCausalLMOutputWithPast: Model outputs containing:
                - loss: None (not computed in base model)
                - logits: None (no LM head in base model)
                - past_key_values: Updated KV cache if caching is enabled
                - last_hidden_state: Final hidden states (batch_size, sequence_length, hidden_size)
                - hidden_states: Tuple of hidden states from all layers if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
                - image_hidden_states: Projected image features if pixel_values is provided
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
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
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

        return AyaVisionCausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size,
        max_length,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ):
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ):
        """Prepares inputs for text generation, including pixel values if provided.

        Args:
            input_ids (Array): Initial input token IDs.
            max_length (int): Maximum generation length.
            pixel_values (Optional[Array]): Pixel values for image input.
            attention_mask (Optional[Array]): Attention mask.

        Returns:
            dict: Model inputs ready for generation.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Updates model inputs for the next step of generation, removing pixel values after the first step.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs: Current keyword arguments for the model.

        Returns:
            dict: Updated model keyword arguments.
        """
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)  # only effect first iter
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
        return self.language_model.get_lm_head()

    def get_embedding(self):
        """
        Returns the embedding layer of the language model (decoder).
        """
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=AyaVisionConfig, model_type="aya_vision")
class AyaVisionForConditionalGeneration(BaseVisionLanguageModule[AyaVisionModel, AyaVisionConfig]):
    """AyaVision model for conditional text generation based on image inputs.

    Combines a vision tower and a language model with a multi-modal projector.
    Inherits from BaseVisionLanguageModule to leverage common VLM infrastructure.

    Attributes:
        config (AyaVisionConfig): Configuration object.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): JAX precision level.
        rngs (nn.Rngs): Random number generators.

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "aya_vision" model identifier
        _supports_video: False (AyaVision is image-only)
        _uses_mrope: False (uses standard RoPE)
    """

    # Class attributes for registration and capabilities
    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "aya_vision"
    _config_class = AyaVisionConfig
    _auto_register = False  # Already registered via decorator
    _supports_video = False
    _uses_mrope = False

    # Component name mapping
    _vision_tower_name = "vision_tower"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: AyaVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the AyaVisionForConditionalGeneration model with LM head.

        Args:
            config (AyaVisionConfig): Model configuration with vision, text, and projector settings.
            dtype (jnp.dtype): Computation dtype for activations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Parameter storage dtype for weights. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): JAX matmul precision. Defaults to None (uses default precision).
            rngs (nn.Rngs): Flax NNX random number generators for initializing all submodules.
        """
        super().__init__(
            config=config,
            base_model_class=AyaVisionModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            # VLM-specific configuration
            vision_feature_layer=getattr(config, "vision_feature_layer", -1),
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "default"),
            image_token_index=config.image_token_id,
            # LM head configuration
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract and project image features from pixel values.

        Delegates to the base model's get_image_features implementation which:
        1. Passes pixel_values through the vision tower
        2. Applies pixel shuffling for downsampling
        3. Applies the multimodal projector with gating

        Args:
            pixel_values: Input image pixel values
            **kwargs: Additional arguments (unused for AyaVision)

        Returns:
            Projected image features ready for merging with text embeddings
        """
        return self.base_model.get_image_features(pixel_values)

    def compute_embedding(self, input_ids, *args, **kwargs):
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
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
    ) -> VLMCausalLMOutput:
        """Forward pass for the AyaVision model.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            pixel_values: Input pixel values for images (batch_size, channels, height, width)
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

        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            **lm_kwargs,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), name="lm_head_output")
            lm_logits = self.apply_logit_cap(lm_logits)

        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size,
        max_length,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ):
        """Initialize KV cache for generation."""
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head."""
        return self.lm_head(hidden_states)

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision tower component."""
        return self.base_model.vision_tower

    def get_projector(self) -> nn.Module:
        """Returns the multimodal projector component."""
        return self.base_model.multi_modal_projector

    def get_language_model(self) -> nn.Module:
        """Returns the language model component."""
        return self.base_model.language_model
