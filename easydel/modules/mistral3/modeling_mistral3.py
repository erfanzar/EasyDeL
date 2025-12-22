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
    """Spatially merges neighboring vision patches before projecting into text space."""

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
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
    """Projects vision tower features into the language model embedding space."""

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
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
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = checkpoint_name(self.linear_1(image_features), name="projector_linear1")
        hidden_states = self.act(hidden_states)
        hidden_states = checkpoint_name(self.linear_2(hidden_states), name="projector_linear2")
        return hidden_states


@register_module(TaskType.BASE_MODULE, config=Mistral3Config, model_type="mistral3")
class Mistral3Model(EasyDeLBaseModule):
    """Multimodal Mistral3 wrapper combining a vision tower, projector, and language model."""

    def __init__(
        self,
        config: Mistral3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
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
        """Initializes the Mistral3ForConditionalGeneration model."""
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
