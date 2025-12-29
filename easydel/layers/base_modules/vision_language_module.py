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

"""Base class for Vision-Language Models (VLMs).

This module provides BaseVisionLanguageModule as the foundation for all
Vision-Language Models including LLaVA, Qwen2-VL, Qwen3-VL, Gemma3,
AyaVision, Mistral3, and Llama4.

Key features:
- Vision tower management and feature extraction
- Multimodal embedding merge utilities
- Video processing support (optional)
- Multi-dimensional RoPE support for Qwen-style models (optional)
- Unified generation helpers for VLMs
"""

from abc import abstractmethod
from collections.abc import Callable

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)

from ._base_task_module import ConfigT, ModelT
from ._vlm_features import (
    MultiDimensionalRoPEFeature,
    MultiModalMergeFeature,
    VideoProcessingFeature,
    VisionEncoderFeature,
)
from .conditional_generation_module import BaseConditionalGenerationModule


class BaseVisionLanguageModule(BaseConditionalGenerationModule[ModelT, ConfigT]):
    """Base class for all Vision-Language Models.

    Inherits from BaseConditionalGenerationModule and adds:
    - Vision tower management
    - Multimodal embedding merge utilities
    - Video processing support (optional, via _supports_video flag)
    - Multi-dimensional RoPE support (optional, via _uses_mrope flag)
    - Generation helpers for VLMs

    Example:
        ```python
        class LlavaForConditionalGeneration(
            BaseVisionLanguageModule[LlavaModel, LlavaConfig]
        ):
            _task_type = TaskType.IMAGE_TEXT_TO_TEXT
            _model_type = "llava"
            _config_class = LlavaConfig
            _supports_video = False
            _uses_mrope = False

            def get_image_features(self, pixel_values, **kwargs):
                vision_outputs = self.get_vision_tower()(
                    pixel_values, output_hidden_states=True
                )
                selected = self._select_vision_features(vision_outputs.hidden_states)
                return self.get_projector()(selected)
        ```

    Type Parameters:
        ModelT: The base VLM model type
        ConfigT: The VLM configuration type

    Class Attributes:
        _supports_video: Whether this VLM supports video input (default: False)
        _uses_mrope: Whether this VLM uses multi-dimensional RoPE (default: False)
        _vision_tower_name: Attribute name for vision encoder (default: "vision_tower")
        _projector_name: Attribute name for projector (default: "multi_modal_projector")
        _language_model_name: Attribute name for LM (default: "language_model")
    """

    # Class attributes for VLM capabilities
    _supports_video: bool = False
    _uses_mrope: bool = False

    # Component name mapping (override if model uses different names)
    _vision_tower_name: str = "vision_tower"
    _projector_name: str = "multi_modal_projector"
    _language_model_name: str = "language_model"

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        vision_feature_layer: int | list[int] = -1,
        vision_feature_select_strategy: str = "default",
        image_token_index: int | None = None,
        video_token_index: int | None = None,
        temporal_patch_size: int = 2,
        tokens_per_second: float = 1.0,
        spatial_merge_size: int = 2,
        mrope_section: tuple[int, int, int] = (24, 20, 20),
        tie_word_embeddings: bool = False,
        logit_cap: float | None = None,
        router_aux_loss_coef: float | None = None,
        lm_head_name: str = "lm_head",
        create_lm_head: bool = True,
        lm_head_bias: bool = False,
        lm_head_kernel_init: Callable | None = None,
    ):
        """Initialize the Vision-Language module.

        Args:
            config: VLM configuration
            base_model: Pre-instantiated base model (optional)
            base_model_class: Base model class to instantiate (optional)
            base_model_name: Attribute name for the base model
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
            vision_feature_layer: Which vision layer(s) to extract features from
            vision_feature_select_strategy: Feature selection strategy
            image_token_index: Token ID for image placeholders
            video_token_index: Token ID for video placeholders
            temporal_patch_size: Temporal patch size for video
            tokens_per_second: Tokens per second for video temporal positions
            spatial_merge_size: Spatial merge factor for mRoPE
            mrope_section: mRoPE section allocation (T, H, W)
            tie_word_embeddings: Whether to tie embeddings with LM head
            logit_cap: Maximum absolute value for logits
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss
            lm_head_name: Attribute name for the LM head
            create_lm_head: Whether to create a new LM head on this wrapper
            lm_head_bias: Whether to use bias in LM head
            lm_head_kernel_init: Custom kernel initializer for LM head
        """
        super().__init__(
            config=config,
            base_model=base_model,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            tie_word_embeddings=tie_word_embeddings,
            logit_cap=logit_cap,
            lm_head_name=lm_head_name,
            lm_head_bias=lm_head_bias,
            lm_head_kernel_init=lm_head_kernel_init,
            create_lm_head=create_lm_head,
        )

        # Get token IDs from config if not provided
        image_token_index = image_token_index or getattr(config, "image_token_id", None)
        video_token_index = video_token_index or getattr(config, "video_token_id", None)

        # Initialize VLM features
        self._vision_encoder_feature = VisionEncoderFeature(
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

        self._multimodal_merge_feature = MultiModalMergeFeature(
            strategy="placeholder",
            image_token_id=image_token_index,
            video_token_id=video_token_index,
        )

        # Optional video processing feature
        if self._supports_video:
            self._video_feature = VideoProcessingFeature(
                temporal_patch_size=temporal_patch_size,
                tokens_per_second=tokens_per_second,
            )
        else:
            self._video_feature = None

        # Optional mRoPE feature
        if self._uses_mrope:
            self._mrope_feature = MultiDimensionalRoPEFeature(
                spatial_merge_size=spatial_merge_size,
                mrope_section=mrope_section,
            )
        else:
            self._mrope_feature = None

        # Store router aux loss coefficient for MoE models
        self._router_aux_loss_coef = router_aux_loss_coef

    @abstractmethod
    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract and project image features from pixel values.

        This method must be implemented by all VLM subclasses. It should:
        1. Pass pixel_values through the vision tower
        2. Select features from appropriate layer(s)
        3. Apply the multimodal projector

        Args:
            pixel_values: Input image pixel values
            **kwargs: Additional model-specific arguments (e.g., image_grid_thw)

        Returns:
            Projected image features ready for merging with text embeddings
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_image_features()")

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute input embeddings for vision-language models.

        Delegates to the underlying VLM base model's `compute_embedding` when
        available, so task wrappers expose the same embedding behavior.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def compute_embedding_with_info(self, input_ids, *args, **kwargs):
        """Compute embeddings and auxiliary info for vision-language models.

        Delegates to the underlying VLM base model's `compute_embedding_with_info`
        so task wrappers can surface any extra multimodal tensors needed when
        passing `inputs_embeds` directly.
        """
        # When a wrapper overrides `compute_embedding`, we still want to expose
        # any auxiliary embedding info (e.g. position_ids / rope_deltas /
        # deepstack tensors) produced by the underlying base model. Some VLMs
        # require this info when calling the forward with `inputs_embeds`.
        if self.__class__.compute_embedding is BaseVisionLanguageModule.compute_embedding:
            return self.base_model.compute_embedding_with_info(input_ids, *args, **kwargs)

        inputs_embeds = self.compute_embedding(input_ids, *args, **kwargs)
        embed_info = None

        base_fn = getattr(self.base_model, "compute_embedding_with_info", None)
        if callable(base_fn):
            try:
                _, embed_info = base_fn(input_ids, *args, **kwargs)
            except TypeError:
                # Some base models may not accept wrapper-specific kwargs
                # (e.g. pixel_values). Fall back to passing only parameters that
                # exist in the base signature.
                import inspect

                sig = inspect.signature(base_fn)
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    raise
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                _, embed_info = base_fn(input_ids, *args, **filtered_kwargs)

        return inputs_embeds, embed_info

    def get_video_features(
        self,
        pixel_values_videos: Float[Array, "batch temporal channels height width"],
        video_grid_thw: tuple | None = None,
        **kwargs,
    ) -> Float[Array, "batch num_tokens hidden"]:
        """Extract and project video features.

        Override this method for models that support video input.

        Args:
            pixel_values_videos: Input video pixel values
            video_grid_thw: Video grid shape (temporal, height, width)
            **kwargs: Additional model-specific arguments

        Returns:
            Projected video features ready for merging with text embeddings
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support video input. "
            "Override get_video_features() to add video support."
        )

    @staticmethod
    def merge_multimodal_embeddings(
        input_ids: Int[Array, "batch seq_len"],
        inputs_embeds: Float[Array, "batch seq_len hidden"],
        multimodal_embeddings: Float[Array, "num_tokens hidden"],
        placeholder_token_id: int | list[int],
    ) -> Float[Array, "batch seq_len hidden"]:
        """Merge vision embeddings into text embeddings at placeholder positions.

        Uses efficient cumsum-based gathering from Qwen implementation.

        Args:
            input_ids: Input token IDs
            inputs_embeds: Text embeddings
            multimodal_embeddings: Vision embeddings to merge
            placeholder_token_id: Token ID(s) marking placeholder positions

        Returns:
            Merged embeddings with vision features at placeholder positions
        """
        batch_size, seq_len, hidden = inputs_embeds.shape
        if isinstance(placeholder_token_id, list):
            placeholder_token_id = jnp.array(placeholder_token_id)
            is_multimodal = jnp.isin(input_ids, placeholder_token_id)
        else:
            is_multimodal = input_ids == placeholder_token_id

        flat_mask = is_multimodal.reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, hidden)

        dummy_row = jnp.zeros_like(multimodal_embeddings[0:1])
        flattened_padded = jnp.concatenate([dummy_row, multimodal_embeddings], axis=0)
        gather_indices = jnp.cumsum(flat_mask)
        update_values = flattened_padded[gather_indices]

        merged = jnp.where(flat_mask[:, None], update_values, flat_embeds)
        return merged.reshape(batch_size, seq_len, hidden)

    def _select_vision_features(
        self,
        hidden_states: tuple[Array, ...],
        feature_layer: int | list[int] | None = None,
        select_strategy: str | None = None,
    ) -> Array:
        """Select features from vision encoder hidden states.

        Args:
            hidden_states: Tuple of hidden states from vision encoder
            feature_layer: Override for which layer(s) to use
            select_strategy: Override for selection strategy

        Returns:
            Selected vision features
        """
        return self._vision_encoder_feature.extract_features(
            hidden_states,
            feature_layer=feature_layer,
            select_strategy=select_strategy,
        )

    def _get_multimodal_mask(
        self,
        input_ids: Int[Array, "batch seq_len"],
    ) -> Array:
        """Get boolean mask indicating multimodal token positions.

        Args:
            input_ids: Input token IDs

        Returns:
            Boolean mask where True indicates vision placeholder positions
        """
        return self._multimodal_merge_feature.create_multimodal_mask(input_ids)

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision encoder/tower component.

        Returns:
            The vision tower module
        """
        if hasattr(self.base_model, self._vision_tower_name):
            return getattr(self.base_model, self._vision_tower_name)
        # Try common alternative names
        for name in ["visual", "vision_model", "image_encoder"]:
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
        raise AttributeError(
            f"Cannot find vision tower. Tried: {self._vision_tower_name}, visual, vision_model, image_encoder"
        )

    def get_projector(self) -> nn.Module:
        """Returns the multimodal projector component.

        Returns:
            The projector module
        """
        if hasattr(self.base_model, self._projector_name):
            return getattr(self.base_model, self._projector_name)
        # Try common alternative names
        for name in ["projector", "mm_projector", "vision_projector"]:
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
        raise AttributeError(
            f"Cannot find projector. Tried: {self._projector_name}, projector, mm_projector, vision_projector"
        )

    def get_language_model(self) -> nn.Module:
        """Returns the language model component.

        Returns:
            The language model module
        """
        if hasattr(self.base_model, self._language_model_name):
            return getattr(self.base_model, self._language_model_name)
        # Try common alternative names
        for name in ["text_model", "llm", "decoder", "language_model"]:
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
        raise AttributeError(f"Cannot find language model. Tried: {self._language_model_name}, text_model, llm, decoder")

    def get_encoder(self) -> nn.Module:
        """Returns the encoder component.

        For VLMs, the encoder is the vision tower.

        Returns:
            The vision tower module
        """
        return self.get_vision_tower()

    def get_decoder(self) -> nn.Module:
        """Returns the decoder component.

        For VLMs, the decoder is the language model.

        Returns:
            The language model module
        """
        return self.get_language_model()

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        # Vision inputs
        pixel_values: Float[Array, "batch channels height width"] | None = None,
        pixel_values_videos: Float[Array, "..."] | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        # Common arguments
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for Vision-Language models.

        This is a general implementation that can be overridden by specific VLMs
        for custom behavior (e.g., Qwen's mRoPE position computation).

        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Attention mask
            mask_info: Mask information
            position_ids: Position IDs
            pixel_values: Image pixel values
            pixel_values_videos: Video pixel values (for video-capable models)
            image_grid_thw: Image grid shape for mRoPE
            video_grid_thw: Video grid shape for mRoPE
            mode: Runtime mode
            past_key_values: KV cache
            cache_metadata: Cache metadata
            apply_lm_head: Whether to apply LM head
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional model-specific arguments

        Returns:
            VLMCausalLMOutput with logits and model outputs
        """
        # Track image/video hidden states for output
        image_hidden_states = None
        video_hidden_states = None

        # Forward through base model with all inputs
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
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

        # Get optional outputs
        rope_deltas = getattr(outputs, "rope_deltas", None)
        router_logits = getattr(outputs, "router_logits", None)
        image_hidden_states = getattr(outputs, "image_hidden_states", None)

        # Compute aux loss for MoE models
        aux_loss = self.compute_router_aux_loss(outputs)

        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
            video_hidden_states=video_hidden_states,
            rope_deltas=rope_deltas,
            router_logits=router_logits,
            aux_loss=aux_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Array,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        attention_mask: Array | None = None,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation, including vision inputs.

        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            pad_token_id: Padding token ID
            starts: Starting positions
            pixel_values: Image pixel values
            attention_mask: Attention mask
            **kwargs: Additional kwargs

        Returns:
            Dictionary of prepared inputs
        """
        model_inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        # Add vision inputs
        model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def update_inputs_for_generation(
        self,
        model_outputs,
        model_kwargs: dict,
    ) -> dict:
        """Update inputs for next generation step, removing vision inputs.

        Vision inputs are only used on the first generation step, so they
        are removed for subsequent steps.

        Args:
            model_outputs: Outputs from the model
            model_kwargs: Current model kwargs

        Returns:
            Updated model kwargs with vision inputs removed
        """
        model_kwargs = self.base_model.update_inputs_for_generation(
            model_outputs,
            model_kwargs,
        )
        # Vision inputs only needed on first iteration
        model_kwargs.pop("pixel_values", None)
        model_kwargs.pop("pixel_values_videos", None)
        model_kwargs.pop("image_grid_thw", None)
        model_kwargs.pop("video_grid_thw", None)
        return model_kwargs
