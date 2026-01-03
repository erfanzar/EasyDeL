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
from easydel.infra.modeling_outputs import BaseModelOutput, ModelOutput, VLMCausalLMOutput
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
from easydel.layers.norms import RMSNorm
from easydel.modules.auto.auto_modeling import AutoEasyDeLModel, AutoEasyDeLVisionModel

from .mistral3_configuration import Mistral3Config

logger = get_logger(__name__)


@auto_pytree
class Mistral3ModelOutput(BaseModelOutput):
    """Model output carrying text hidden states and optional projected image embeddings."""

    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


@auto_pytree
class Mistral3CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Mistral3 causal language model (or autoregressive) outputs.

    Args:
        loss (`Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(Array))`, *optional*, returned when `use_cache=True` is
            passed or when `config.use_cache=True`):
            Tuple of `tuple(Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(Array)`, *optional*, returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`):
            Tuple of `Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(Array)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`Array`, *optional*):
            A `Array` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


class Mistral3PatchMerger(nn.Module):
    """Spatially merges neighboring vision patches before projecting into text space.

    Reduces the spatial resolution of vision patch tokens by merging adjacent patches
    together, decreasing the sequence length while preserving visual information.
    """

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Mistral3 patch merger.

        Args:
            config (Mistral3Config): Model configuration with vision parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(0.02),
        )

    def forward(self, image_features: jax.Array, image_sizes: jax.Array) -> jax.Array:
        """Merge neighboring patches spatially.

        Takes flattened patch tokens and merges them in spatial neighborhoods defined
        by spatial_merge_size, reducing the total token count while preserving
        semantic information.

        Args:
            image_features (jax.Array): Flattened image patch features of shape
                (num_patches, hidden_dim).
            image_sizes (jax.Array): Original image sizes as (height, width) pairs
                for proper spatial reconstruction.

        Returns:
            jax.Array: Merged patch features with reduced spatial resolution,
                shape (num_merged_patches, hidden_dim).
        """
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]
        k = self.spatial_merge_size
        split_indices = jnp.cumsum(jnp.array(tokens_per_image[:-1]))
        image_features_split = jnp.split(image_features, split_indices, axis=0)
        permuted_tensors = []
        for image_tokens, (h, w) in zip(image_features_split, image_sizes, strict=False):
            image_grid = image_tokens.reshape(h, w, d)
            grid = image_grid.reshape(h // k, k, w // k, k, d)
            grid = grid.transpose(0, 2, 1, 3, 4)
            num_new_tokens = (h // k) * (w // k)
            merged_tokens = grid.reshape(num_new_tokens, k * k * d)
            permuted_tensors.append(merged_tokens)

        image_features = jnp.concatenate(permuted_tensors, axis=0)
        image_features = self.merging_layer(image_features)
        return image_features


class Mistral3MultiModalProjector(nn.Module):
    """Projects vision tower features into the language model embedding space.

    Transforms vision encoder outputs into a representation compatible with
    the language model's hidden dimension through normalization, patch merging,
    and a two-layer MLP projection.
    """

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Mistral3 multimodal projector.

        Args:
            config (Mistral3Config): Model configuration with vision and text parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.norm = RMSNorm(
            config.vision_config.hidden_size,
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
            kernel_init=nn.initializers.normal(0.02),
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
            kernel_init=nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, image_features: jax.Array, image_sizes: jax.Array) -> jax.Array:
        """Project vision features into language model space.

        Applies RMS normalization, patch merging, and a two-layer MLP projection
        to transform vision tower outputs into the language model's embedding space.

        Args:
            image_features (jax.Array): Vision encoder output features of shape
                (num_patches, vision_hidden_dim).
            image_sizes (jax.Array): Original image sizes as (height, width) pairs.

        Returns:
            jax.Array: Projected features of shape (num_merged_patches, text_hidden_dim).
        """
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = checkpoint_name(self.linear_1(image_features), name="projector_linear1")
        hidden_states = self.act(hidden_states)
        hidden_states = checkpoint_name(self.linear_2(hidden_states), name="projector_linear2")
        return hidden_states


@register_module(TaskType.BASE_MODULE, config=Mistral3Config, model_type="mistral3")
class Mistral3Model(EasyDeLBaseModule):
    """Multimodal Mistral3 wrapper combining a vision tower, projector, and language model.

    This model implements the Mistral3 vision-language architecture, integrating a vision
    tower for image encoding, a multimodal projector for feature alignment, and a
    language model for text generation based on both visual and textual inputs.

    Attributes:
        config (Mistral3Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Mistral3 base model.

        Args:
            config (Mistral3Config): Model configuration with vision and text parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
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

    def get_image_features(self, pixel_values: Array, image_sizes: Array) -> Array:
        """Extract and project image features from pixel values.

        Processes images through the vision tower, selects features from the specified
        layer, and projects them into the language model's embedding space.

        Args:
            pixel_values (Array): Input image pixel values of shape
                (batch, channels, height, width).
            image_sizes (Array): Original image sizes as (height, width) pairs
                for proper patch merging.

        Returns:
            Array: Projected image features ready for merging with text embeddings.
        """
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
        return image_features.squeeze(0)

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        image_features: Array | None = None,
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
            image_features = self.get_image_features(pixel_values, image_sizes)

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
        image_sizes: Array = None,
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
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_sizes)

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
class Mistral3ForConditionalGeneration(BaseVisionLanguageModule[Mistral3Model, Mistral3Config]):
    """Mistral3 model for conditional generation with vision-language capabilities.

    Combines a vision tower, patch merger/projector, and language model for
    image-to-text generation. Inherits from BaseVisionLanguageModule.

    Attributes:
        config (Mistral3Config): Configuration object.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): JAX precision level.
        rngs (nn.Rngs): Random number generators.

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "mistral3" model identifier
        _supports_video: False (Mistral3 is image-only)
        _uses_mrope: False (uses standard RoPE)
    """

    # Class attributes for registration and capabilities
    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "mistral3"
    _config_class = Mistral3Config
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
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Mistral3 model for conditional generation.

        Args:
            config (Mistral3Config): Model configuration with vision and text parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Mistral3Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            # VLM-specific configuration
            vision_feature_layer=config.vision_feature_layer,
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "default"),
            image_token_index=config.image_token_index,
            # LM head configuration
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        image_sizes: Array | None = None,
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract and project image features from pixel values.

        Mistral3 uses a patch merger that requires image_sizes to handle
        variable-sized images.

        Args:
            pixel_values: Input image pixel values
            image_sizes: Original sizes of the images (height, width) for patch merging
            **kwargs: Additional arguments (unused)

        Returns:
            Projected image features ready for merging with text embeddings
        """
        return self.base_model.get_image_features(pixel_values=pixel_values, image_sizes=image_sizes)

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

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
        image_sizes: Array = None,
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

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head to hidden states.

        Projects the final hidden states to vocabulary logits for next token prediction.

        Args:
            hidden_states (Array): Hidden states from the model of shape
                (batch_size, sequence_length, hidden_size).

        Returns:
            Array: Logits over vocabulary of shape (batch_size, sequence_length, vocab_size).
        """
        return self.lm_head(hidden_states)

    def get_vision_tower(self) -> nn.Module:
        """Get the vision tower component.

        Returns:
            nn.Module: The vision encoder used for image feature extraction.
        """
        return self.base_model.vision_tower

    def get_projector(self) -> nn.Module:
        """Get the multimodal projector component.

        Returns:
            nn.Module: The projector that aligns vision features with text embeddings.
        """
        return self.base_model.multi_modal_projector

    def get_language_model(self) -> nn.Module:
        """Get the language model component.

        Returns:
            nn.Module: The underlying language model for text generation.
        """
        return self.base_model.language_model
