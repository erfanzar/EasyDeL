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

"""PaliGemma vision-language model implementation.

Composes a SigLIP vision tower with a Gemma text decoder through a single
linear multimodal projector. Visual features are taken from the tower's last
hidden state, projected into the language-model embedding space, and merged
into the text embedding sequence at positions marked by ``image_token_id``.

PaliGemma-specific behaviors preserved from the HF reference:

- Positions are 1-indexed (RoPE sees ``position + 1``).
- With ``token_type_ids``, all prefix tokens (``token_type_ids == 0``: image
  plus text prompt) attend bidirectionally among themselves; suffix tokens
  (``token_type_ids == 1``) remain strictly causal.
- EasyDeL's ``GemmaModel``/``Gemma2Model`` scale any incoming
  ``inputs_embeds`` by ``sqrt(hidden_size)``, so merged image features are
  pre-divided by that factor here — the net effect matches HF, where the
  embedding-lookup scale applies to text tokens only.

Exports:
    - ``PaliGemmaModelOutputWithPast``: structured output for the base trunk.
    - ``PaliGemmaMultiModalProjector``: single-linear vision-to-text bridge.
    - ``PaliGemmaModel``: base multimodal model returning hidden states.
    - ``PaliGemmaForConditionalGeneration``: full model with LM head.
"""

import typing as tp

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
from easydel.infra.modeling_outputs import ModelOutput, VLMCausalLMOutput
from easydel.layers import RowParallelLinear
from easydel.modules._base import BaseVisionLanguageModule

from ..auto.auto_modeling import AutoEasyDeLModel, AutoEasyDeLVisionModel
from .paligemma_configuration import PaliGemmaConfig

logger = get_logger(__name__)


@auto_pytree
class PaliGemmaModelOutputWithPast(ModelOutput):
    """Base class for PaliGemma trunk outputs.

    Args:
        last_hidden_state (`Array`):
            Final layer hidden states of shape (batch_size, sequence_length, hidden_size).
        past_key_values (`TransformerCache`, *optional*):
            Cached key/value states usable for fast autoregressive decoding.
        hidden_states (`tuple(Array)`, *optional*):
            Hidden states of all layers when `output_hidden_states=True`.
        attentions (`tuple(Array)`, *optional*):
            Attention weights of all layers when `output_attentions=True`.
        image_hidden_states (`Array`, *optional*):
            Projected image features produced by the vision tower + projector.
    """

    last_hidden_state: Array | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


class PaliGemmaMultiModalProjector(spx.Module):
    """Single linear layer that maps SigLIP patch features into the Gemma embedding space.

    HF layout: ``multi_modal_projector.linear`` — a biased
    ``Linear(vision_hidden_size, vision_config.projection_dim)`` where
    ``projection_dim`` equals the text hidden size.

    Attributes:
        linear (RowParallelLinear): The vision-to-text projection.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the PaliGemma multi-modal projector.

        Args:
            config (PaliGemmaConfig): Model configuration with projector parameters.
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

        self.linear = RowParallelLinear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, image_features: jax.Array) -> jax.Array:
        """Project image features into the language model embedding space.

        Args:
            image_features (Array): Visual features from the vision tower,
                shape (batch_size, num_patches, vision_hidden_size).

        Returns:
            Array: Projected features with shape (batch_size, num_patches, projection_dim).
        """
        return checkpoint_name(self.linear(image_features), name="projector_linear")


@register_module(TaskType.BASE_VISION, config=PaliGemmaConfig, model_type="paligemma")
class PaliGemmaModel(EasyDeLBaseModule):
    """PaliGemma base trunk: SigLIP vision tower + linear projector + Gemma language model (no LM head).

    On a forward pass:

    1. ``vision_tower`` (an :class:`AutoEasyDeLVisionModel`, resolved to the
       registered SigLIP vision model) embeds the input image; the tower's
       ``last_hidden_state`` is used directly (SigLIP has no CLS token).
    2. ``multi_modal_projector`` (:class:`PaliGemmaMultiModalProjector`) maps
       the patch features to the text hidden size.
    3. ``language_model`` (an :class:`AutoEasyDeLModel` — Gemma for PaliGemma 1,
       Gemma 2 for PaliGemma 2) consumes the merged embedding sequence formed by
       replacing every ``image_token_id`` position with projected vision tokens.

    Attributes:
        vision_tower: SigLIP image encoder.
        multi_modal_projector (PaliGemmaMultiModalProjector): Vision-to-text linear.
        language_model: Gemma-family causal LM trunk without an LM head.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the PaliGemma base model.

        Args:
            config (PaliGemmaConfig): Model configuration containing vision and text config.
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
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
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

        self.vocab_size = config.get_text_config().vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_image_features(self, pixel_values: Array) -> Array:
        """Extract and project image features from the vision tower.

        Uses the tower's last hidden state (all patch tokens; SigLIP has no
        CLS token) and applies the linear multimodal projector — mirroring HF
        ``PaliGemmaModel.get_image_features``.

        Args:
            pixel_values (Array): Input pixel values of shape
                (batch_size, channels, height, width).

        Returns:
            Array: Projected image features of shape
                (batch_size, num_patches, projection_dim).
        """
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = tp.cast(Array, self.multi_modal_projector(selected_image_feature))
        return image_features

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        image_features: Array | None = None,
        pixel_values: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute input embeddings with merged image and text features.

        Text embeddings are looked up unscaled; image features are pre-divided
        by ``sqrt(text_hidden_size)`` before the merge because EasyDeL's Gemma
        models multiply incoming ``inputs_embeds`` by ``sqrt(hidden_size)``
        unconditionally. The net scaling matches the HF reference, where the
        sqrt-scale is applied at embedding lookup (text tokens only) and image
        features are scattered in raw.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            image_features (Array | None, optional): Pre-extracted projected image
                features. If None and pixel_values is provided, features are
                extracted. Defaults to None.
            pixel_values (Array | None, optional): Raw pixel values for image
                extraction. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Array: Combined embeddings of shape (batch_size, sequence_length, hidden_size)
                with image features merged at image-token positions.

        Raises:
            ValueError: If input_ids is None.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        text_config = self.config.get_text_config()
        image_token_id = self.config.image_token_id
        if image_token_id >= text_config.vocab_size:
            llm_input_ids = jnp.where(input_ids == image_token_id, 0, input_ids)
        else:
            llm_input_ids = input_ids

        inputs_embeds = super().compute_embedding(llm_input_ids)

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if image_features is not None:
            # Counteract the unconditional sqrt(hidden_size) inputs_embeds scale inside
            # the EasyDeL Gemma trunk so image features enter attention unscaled (HF parity).
            image_features = image_features / (text_config.hidden_size**0.5)
            multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype)
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=image_token_id,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        image_features: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> PaliGemmaModelOutputWithPast:
        """Forward pass through the PaliGemma base model.

        Merges projected image features with text embeddings, constructs the
        PaliGemma attention pattern (bidirectional prefix / causal suffix when
        ``token_type_ids`` is given), 1-indexes positions, and runs the Gemma
        language model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape
                (batch_size, sequence_length). Must be provided if inputs_embeds is None.
            pixel_values (Array | None, optional): Input pixel values for images of shape
                (batch_size, num_channels, height, width). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on
                padding tokens, shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention
                operations. Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). PaliGemma positions are 1-indexed; when
                None they are derived from the mask and offset by +1. Defaults to None.
            token_type_ids (Array | None, optional): PaliGemma token types — 0 for
                image + prefix tokens (bidirectional block), nonzero for suffix tokens
                (causal). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for
                optimizations. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None,
                optional): Metadata for cache management. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
                Defaults to None.
            **lm_kwargs: Additional arguments passed to the language model.

        Returns:
            PaliGemmaModelOutputWithPast: Model outputs including past key/values,
                hidden states, attentions, and projected image hidden states.

        Raises:
            ValueError: If both or neither of input_ids and inputs_embeds are provided.
            ValueError: If pixel_values is provided without input_ids.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if pixel_values is not None and input_ids is None:
            raise ValueError("`input_ids` must be provided when `pixel_values` is not None.")

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
            )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids + 1

        # HF PaliGemma ORs a bidirectional block over tokens with token_type_ids == 0
        # (image + text prefix) onto the causal mask; suffix tokens stay causal.
        if token_type_ids is not None:
            token_type_ids = jnp.asarray(token_type_ids, dtype=jnp.int32)
            prefix_groups = (token_type_ids == 0).astype(jnp.int32)
            if attention_mask is not None:
                # Padded positions must not join the bidirectional block (the union
                # would otherwise re-open them past the padding mask).
                prefix_groups = prefix_groups * jnp.asarray(attention_mask, dtype=jnp.int32)
            causal_mask_info = mask_info.apply_causal()
            mask_info = causal_mask_info.apply_token_type_ids(prefix_groups)
            # Causal is baked into the mask; attention kernels must not re-apply it.
            object.__setattr__(mask_info, "_causal_baked", True)

        outputs = self.language_model(
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

        return PaliGemmaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
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
        """Initialize the key-value cache for autoregressive generation.

        Delegates to the underlying language model's cache initialization.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions for cache initialization.
                Defaults to None.
            shardings (Any | None, optional): Sharding specifications for the cache.
                Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache for the language model.
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
        token_type_ids: Array | None = None,
    ):
        """Prepare inputs for autoregressive generation.

        Delegates to the language model's preparation, offsets ``position_ids``
        by +1 (PaliGemma positions are 1-indexed), and attaches vision inputs.

        Args:
            input_ids (Array): Initial input token IDs of shape (batch_size, sequence_length).
            max_length (int): Maximum generation length.
            pad_token_id (int): Token ID used for padding.
            starts (int | None, optional): Starting positions for generation. Defaults to None.
            pixel_values (Array | None, optional): Pixel values for image input of shape
                (batch_size, num_channels, height, width). Defaults to None.
            attention_mask (Array | None, optional): Attention mask of shape
                (batch_size, sequence_length). Defaults to None.
            token_type_ids (Array | None, optional): PaliGemma token types (0 = prefix,
                1 = suffix), consumed on the first forward only. Defaults to None.

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
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] = model_inputs["position_ids"] + 1
        model_inputs["pixel_values"] = pixel_values
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Update model inputs for the next generation step.

        Removes ``pixel_values`` and ``token_type_ids`` after the first step:
        image features are already merged into the cached states, and decode
        steps are strictly causal.

        Args:
            model_outputs (PaliGemmaModelOutputWithPast): Outputs from the previous
                generation step containing past_key_values.
            model_kwargs (dict): Current keyword arguments for the model.

        Returns:
            dict: Updated model keyword arguments for the next step.
        """
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        model_kwargs.pop("token_type_ids", None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """Return the encoder component of the model.

        For PaliGemma, the vision tower serves as the encoder for processing images.

        Returns:
            spx.Module: The vision tower module.
        """
        return self.vision_tower

    def get_decoder(self):
        """Return the decoder component of the model.

        Returns:
            spx.Module: The language model's decoder.
        """
        return self.language_model.get_decoder()

    def get_lm_head(self):
        """Return the language model head.

        Raises:
            NotImplementedError: Base models don't have a language model head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the embedding layer of the model.

        Returns:
            Embed: The text embedding layer from the language model.
        """
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=PaliGemmaConfig, model_type="paligemma")
class PaliGemmaForConditionalGeneration(BaseVisionLanguageModule[PaliGemmaModel, PaliGemmaConfig]):  # type: ignore
    """PaliGemma model for conditional text generation based on image inputs.

    Combines the SigLIP vision tower and Gemma language model with a linear
    multimodal projector, plus an LM head tied to the input embeddings
    (``tie_word_embeddings=True`` by default, as in HF).

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "paligemma" model identifier
        _supports_video: False (PaliGemma is image-only)
        _uses_mrope: False (uses standard RoPE)
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "paligemma"
    _config_class = PaliGemmaConfig
    _auto_register = False  # Already registered via decorator
    _supports_video = False
    _uses_mrope = False

    _vision_tower_name = "vision_tower"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: PaliGemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize PaliGemma for conditional generation.

        Args:
            config (PaliGemmaConfig): Model configuration containing vision, text, and
                projector settings.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=PaliGemmaModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=getattr(config, "vision_feature_layer", -1),
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "full"),
            image_token_index=config.image_token_id,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            lm_head_bias=False,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract and project image features from pixel values.

        Delegates to the base model, which runs the SigLIP tower and the
        linear projector on the tower's last hidden state.

        Args:
            pixel_values: Input image pixel values.
            **kwargs: Additional arguments (unused for PaliGemma).

        Returns:
            Projected image features ready for merging with text embeddings.
        """
        return self.base_model.get_image_features(pixel_values)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute input embeddings with merged image and text features.

        Delegates to the base model's compute_embedding method.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            *args: Additional positional arguments passed to base model.
            **kwargs: Additional keyword arguments including pixel_values and image_features.

        Returns:
            Array: Combined embeddings with image features merged at image token positions.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for image-conditioned text generation.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape
                (batch_size, sequence_length). Must be provided if inputs_embeds is None.
            pixel_values (Array | None, optional): Input pixel values for images of shape
                (batch_size, num_channels, height, width). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on
                padding tokens, shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention
                operations. Defaults to None.
            position_ids (Array | None, optional): Position indices for each token
                (1-indexed for PaliGemma; derived from the mask when None). Defaults to None.
            token_type_ids (Array | None, optional): PaliGemma token types — 0 for
                image + prefix (bidirectional), nonzero for suffix (causal). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode).
                Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None,
                optional): Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language modeling head.
                Defaults to True.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings.
                Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
                Defaults to None.
            **lm_kwargs: Additional arguments passed to the language model.

        Returns:
            VLMCausalLMOutput: Model outputs containing logits (if apply_lm_head is True),
                past_key_values, hidden_states, attentions, and image_hidden_states.

        Raises:
            ValueError: If both or neither of input_ids and inputs_embeds are provided.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
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
        """Initialize the key-value cache for autoregressive generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions for cache initialization.
                Defaults to None.
            shardings (Any | None, optional): Sharding specifications for the cache.
                Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache for the language model.
        """
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def get_vision_tower(self) -> spx.Module:
        """Return the vision tower component.

        Returns:
            spx.Module: The SigLIP vision encoder.
        """
        return self.base_model.vision_tower

    def get_projector(self) -> spx.Module:
        """Return the multimodal projector component.

        Returns:
            spx.Module: The linear projection mapping vision features to text space.
        """
        return self.base_model.multi_modal_projector

    def get_language_model(self) -> spx.Module:
        """Return the language model component.

        Returns:
            spx.Module: The underlying Gemma-family language model.
        """
        return self.base_model.language_model
