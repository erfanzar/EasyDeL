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

"""Base class for Vision-Language Models (VLMs).

This module provides BaseVisionLanguageModule as the foundation for all
Vision-Language Models including LLaVA, Qwen2-VL, Qwen3-VL, Gemma3,
AyaVision, Mistral3, and Llama4.

Vision-Language Models combine visual understanding with language generation,
enabling tasks like image captioning, visual question answering, and
multimodal reasoning.

Key Features:
    - Vision tower management and feature extraction
    - Multimodal embedding merge utilities (placeholder-based merging)
    - Video processing support (optional, via _supports_video flag)
    - Multi-dimensional RoPE support for Qwen-style models (optional)
    - Unified generation helpers for VLMs
    - Support for various vision encoders (ViT, SigLIP, etc.)

Supported Model Architectures:
    - LLaVA: Image-text models with projector
    - Qwen2-VL/Qwen3-VL: Video-capable models with mRoPE
    - Gemma3: Google's multimodal model
    - AyaVision: Multilingual VLM
    - Mistral3: Multimodal Mistral variant
    - Llama4: Meta's vision-language model

Example:
    Creating a vision-language model::

        from easydel.modules._base import BaseVisionLanguageModule

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

See Also:
    - BaseConditionalGenerationModule: Parent class for encoder-decoder models
    - VisionEncoderFeature: Vision feature extraction utilities
    - MultiModalMergeFeature: Embedding merge utilities
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

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.infra.modeling_outputs import VLMCausalLMOutput

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

    Inherits from BaseConditionalGenerationModule and adds comprehensive
    support for multimodal (vision + language) processing:

    - Vision tower management for extracting visual features
    - Multimodal embedding merge utilities for combining vision and text
    - Video processing support (optional, via _supports_video flag)
    - Multi-dimensional RoPE support for spatial-temporal encoding (optional)
    - Generation helpers optimized for VLMs

    The typical VLM pipeline is:
        1. Extract image/video features through vision tower
        2. Project features through multimodal projector
        3. Merge vision embeddings with text embeddings at placeholder positions
        4. Process through language model decoder
        5. Generate text autoregressively

    Example:
        Basic VLM implementation::

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

        Video-capable VLM with mRoPE::

            class Qwen2VLForConditionalGeneration(
                BaseVisionLanguageModule[Qwen2VLModel, Qwen2VLConfig]
            ):
                _supports_video = True
                _uses_mrope = True

                def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
                    # Process images with grid-aware encoding
                    ...

                def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
                    # Process video frames with temporal encoding
                    ...

    Type Parameters:
        ModelT: The base VLM model type containing vision tower, projector,
            and language model components.
        ConfigT: The VLM configuration type with multimodal settings.

    Class Attributes:
        _supports_video (bool): Whether this VLM supports video input.
            When True, enables VideoProcessingFeature. Defaults to False.
        _uses_mrope (bool): Whether this VLM uses multi-dimensional RoPE
            for spatial-temporal position encoding. Defaults to False.
        _vision_tower_name (str): Attribute name for vision encoder.
            Defaults to "vision_tower".
        _projector_name (str): Attribute name for multimodal projector.
            Defaults to "multi_modal_projector".
        _language_model_name (str): Attribute name for the language model.
            Defaults to "language_model".

    Attributes:
        _vision_encoder_feature: Feature helper for vision processing.
        _multimodal_merge_feature: Feature helper for embedding merging.
        _video_feature: Video processing feature (if _supports_video=True).
        _mrope_feature: mRoPE feature (if _uses_mrope=True).
        base_model: The underlying VLM model with all components.

    Note:
        Subclasses must implement get_image_features() to define how
        image features are extracted and projected.
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

        Creates a VLM by wrapping a base multimodal model and setting up
        the necessary features for vision processing, embedding merging,
        and optionally video/mRoPE support.

        Args:
            config: VLM configuration object containing settings for:
                - Vision encoder (hidden sizes, patch sizes, etc.)
                - Language model (vocab size, hidden size, etc.)
                - Multimodal settings (image/video token IDs, etc.)
            base_model: Pre-instantiated base VLM model. If provided,
                base_model_class is ignored.
            base_model_class: VLM class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name for storing the base model.
                Defaults to "model".
            dtype: Data type for computations. Defaults to bfloat16.
            param_dtype: Data type for parameters. Defaults to bfloat16.
            precision: JAX precision setting for matrix multiplications.
            rngs: Flax random number generators for initialization.
            vision_feature_layer: Which vision encoder layer(s) to extract
                features from. Can be:
                - int: Single layer index (e.g., -1 for last, -2 for second-to-last)
                - list[int]: Multiple layers for multi-scale features
            vision_feature_select_strategy: Strategy for feature selection:
                - "default": Take features directly from specified layer(s)
                - "full": Include all spatial tokens
                - "cls": Use only CLS token (if present)
            image_token_index: Token ID used as placeholder for images in
                the input sequence. If None, reads from config.image_token_id.
            video_token_index: Token ID used as placeholder for videos.
                If None, reads from config.video_token_id.
            temporal_patch_size: Number of frames grouped into one temporal
                patch for video processing. Defaults to 2.
            tokens_per_second: Temporal resolution for video position encoding.
                Defaults to 1.0 tokens per second.
            spatial_merge_size: Factor for spatial merging in mRoPE models.
                Affects how spatial positions are encoded. Defaults to 2.
            mrope_section: Allocation of RoPE dimensions for (T, H, W).
                Only used when _uses_mrope=True. Default: (24, 20, 20).
            tie_word_embeddings: Whether to tie input embeddings with LM head.
            logit_cap: Maximum absolute value for output logits.
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss.
            lm_head_name: Attribute name for the LM head.
            create_lm_head: Whether to create a new LM head on this wrapper.
                Set to False if the base model already has an LM head.
            lm_head_bias: Whether to include bias in the LM head.
            lm_head_kernel_init: Custom initializer for LM head weights.

        Example:
            Creating a basic VLM::

                model = BaseVisionLanguageModule(
                    config=llava_config,
                    base_model_class=LlavaModel,
                    rngs=nn.Rngs(0),
                    vision_feature_layer=-2,
                    image_token_index=32000,
                )

            Creating a video-capable VLM with mRoPE::

                class MyVideoVLM(BaseVisionLanguageModule[...]):
                    _supports_video = True
                    _uses_mrope = True

                model = MyVideoVLM(
                    config=config,
                    base_model_class=VideoVLMModel,
                    rngs=nn.Rngs(0),
                    temporal_patch_size=2,
                    mrope_section=(24, 20, 20),
                )
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

        This method must be implemented by all VLM subclasses. It defines
        the vision processing pipeline:
            1. Pass pixel_values through the vision tower
            2. Select features from appropriate layer(s)
            3. Apply the multimodal projector

        Args:
            pixel_values: Input image pixel values with shape
                (batch_size, channels, height, width). Values should be
                normalized according to the vision encoder's requirements.
            **kwargs: Additional model-specific arguments. Common kwargs:
                - image_grid_thw: Grid shape (T, H, W) for mRoPE models
                - output_hidden_states: Whether to output all hidden states

        Returns:
            Float[Array, "batch num_patches hidden"]: Projected image features
                ready for merging with text embeddings. Shape is typically
                (batch_size, num_patches, language_model_hidden_size).

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.

        Example:
            Typical implementation::

                def get_image_features(self, pixel_values, **kwargs):
                    # 1. Extract vision features
                    vision_outputs = self.get_vision_tower()(
                        pixel_values,
                        output_hidden_states=True
                    )

                    # 2. Select features from desired layer
                    hidden_states = vision_outputs.hidden_states
                    selected = self._select_vision_features(hidden_states)

                    # 3. Project to language model dimension
                    return self.get_projector()(selected)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_image_features()")

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute input embeddings for vision-language models.

        Delegates to the underlying VLM base model's compute_embedding method,
        ensuring task wrappers expose the same embedding behavior including
        any multimodal processing.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments passed to base model.
            **kwargs: Keyword arguments passed to base model. May include
                pixel_values, image features, etc.

        Returns:
            Array: Input embeddings, potentially with vision embeddings merged.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def compute_embedding_with_info(self, input_ids, *args, **kwargs):
        """Compute embeddings and auxiliary info for vision-language models.

        Delegates to the underlying VLM base model's compute_embedding_with_info
        to surface any extra multimodal tensors needed when passing inputs_embeds
        directly (e.g., position_ids, rope_deltas, deepstack tensors).

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments passed to base model.
            **kwargs: Keyword arguments passed to base model.

        Returns:
            tuple: (inputs_embeds, embed_info) where:
                - inputs_embeds: The computed embeddings
                - embed_info: Additional information dict (may be None)
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

        Override this method for models that support video input. The default
        implementation raises NotImplementedError.

        Args:
            pixel_values_videos: Input video pixel values with shape
                (batch_size, num_frames, channels, height, width) or similar
                depending on the model's video processing approach.
            video_grid_thw: Video grid shape as (temporal, height, width).
                Used for mRoPE position encoding in models like Qwen2-VL.
            **kwargs: Additional model-specific arguments.

        Returns:
            Float[Array, "batch num_tokens hidden"]: Projected video features
                ready for merging with text embeddings.

        Raises:
            NotImplementedError: If the model doesn't support video input
                (when _supports_video=False or method not overridden).

        Example:
            Video feature extraction::

                def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
                    # Reshape video to process all frames
                    batch, frames, c, h, w = pixel_values_videos.shape
                    flat_frames = pixel_values_videos.reshape(-1, c, h, w)

                    # Extract features
                    vision_outputs = self.get_vision_tower()(flat_frames)

                    # Reshape and project
                    features = vision_outputs.last_hidden_state
                    features = features.reshape(batch, -1, features.shape[-1])
                    return self.get_projector()(features)
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

        Uses an efficient cumsum-based gathering algorithm (from Qwen implementation)
        to replace placeholder tokens with corresponding vision embeddings.

        The algorithm:
            1. Create mask identifying placeholder positions
            2. Use cumulative sum to compute gather indices
            3. Gather vision embeddings at those indices
            4. Replace placeholders with gathered embeddings

        Args:
            input_ids: Input token IDs with shape (batch_size, seq_len).
                Contains placeholder tokens where vision features should go.
            inputs_embeds: Text embeddings with shape (batch_size, seq_len, hidden).
                Initial embeddings before vision merge.
            multimodal_embeddings: Vision embeddings with shape (num_tokens, hidden).
                Flattened vision features from all images/videos in the batch.
            placeholder_token_id: Token ID(s) marking placeholder positions.
                Can be a single int or list of ints for multiple modalities.

        Returns:
            Float[Array, "batch seq_len hidden"]: Merged embeddings with vision
                features inserted at placeholder positions.

        Example:
            Merging image features::

                # input_ids: [CLS, text, <image>, <image>, ..., text, SEP]
                # where <image> tokens will be replaced
                merged = BaseVisionLanguageModule.merge_multimodal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=text_embeds,
                    multimodal_embeddings=image_features.reshape(-1, hidden),
                    placeholder_token_id=IMAGE_TOKEN_ID,
                )

        Note:
            The number of multimodal_embeddings must match the total number
            of placeholder tokens across all sequences in the batch.
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

        Uses the VisionEncoderFeature helper to extract features from
        specified layer(s) using the configured selection strategy.

        Args:
            hidden_states: Tuple of hidden states from all vision encoder layers.
                Each element has shape (batch_size, num_patches, hidden_size).
            feature_layer: Override for which layer(s) to use.
                If None, uses the layer specified during initialization.
            select_strategy: Override for selection strategy.
                If None, uses the strategy specified during initialization.

        Returns:
            Array: Selected vision features. Shape depends on strategy
                and number of layers selected.
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

        Creates a mask identifying where placeholder tokens appear in the
        input sequence. This is used for merging vision embeddings.

        Args:
            input_ids: Input token IDs with shape (batch_size, seq_len).

        Returns:
            Bool[Array, "batch seq_len"]: Boolean mask where True indicates
                positions that are vision/video placeholders.
        """
        return self._multimodal_merge_feature.create_multimodal_mask(input_ids)

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision encoder/tower component.

        Retrieves the vision encoder module from the base model. This is
        typically a ViT, SigLIP, or similar vision transformer.

        Returns:
            nn.Module: The vision tower module responsible for extracting
                visual features from images.

        Raises:
            AttributeError: If no vision tower can be found under any of
                the common attribute names.

        Note:
            Searches for the vision tower using these attribute names in order:
            1. The configured _vision_tower_name
            2. "visual"
            3. "vision_model"
            4. "image_encoder"
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

        Retrieves the projector module that transforms vision features to
        the language model's hidden dimension.

        Returns:
            nn.Module: The projector module (typically an MLP).

        Raises:
            AttributeError: If no projector can be found under any of
                the common attribute names.

        Note:
            Searches for the projector using these attribute names in order:
            1. The configured _projector_name
            2. "projector"
            3. "mm_projector"
            4. "vision_projector"
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

        Retrieves the language model (decoder) that generates text
        conditioned on the multimodal embeddings.

        Returns:
            nn.Module: The language model module (e.g., Llama, Qwen).

        Raises:
            AttributeError: If no language model can be found under any of
                the common attribute names.

        Note:
            Searches for the language model using these attribute names in order:
            1. The configured _language_model_name
            2. "text_model"
            3. "llm"
            4. "decoder"
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

        For VLMs, the encoder is the vision tower that processes images.

        Returns:
            nn.Module: The vision tower module.

        See Also:
            get_vision_tower: Direct method for accessing vision encoder.
        """
        return self.get_vision_tower()

    def get_decoder(self) -> nn.Module:
        """Returns the decoder component.

        For VLMs, the decoder is the language model that generates text.

        Returns:
            nn.Module: The language model module.

        See Also:
            get_language_model: Direct method for accessing language model.
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

        This is the general implementation that handles the full VLM pipeline.
        Specific VLMs may override this for custom behavior (e.g., Qwen's
        mRoPE position computation).

        The pipeline typically:
            1. Process vision inputs (if provided)
            2. Merge vision embeddings with text embeddings
            3. Forward through the language model
            4. Apply LM head for vocabulary predictions

        Args:
            input_ids: Input token IDs with shape (batch_size, seq_len).
                Contains text tokens and placeholder tokens for images/videos.
            inputs_embeds: Pre-computed embeddings (alternative to input_ids).
                Shape: (batch_size, seq_len, hidden_dim).
            attention_mask: Attention mask with shape (batch_size, seq_len).
            mask_info: Structured mask information for optimized kernels.
            position_ids: Position IDs for positional encoding.
            pixel_values: Image pixel values with shape
                (batch_size, channels, height, width). Processed on first
                forward pass only (cached in KV cache for subsequent steps).
            pixel_values_videos: Video pixel values (for video-capable models).
                Shape varies by model.
            image_grid_thw: Image grid shape (T, H, W) for mRoPE models.
            video_grid_thw: Video grid shape (T, H, W) for mRoPE models.
            mode: Runtime mode (train, eval, prefill, decode, etc.).
            past_key_values: Cached key/value states for efficient generation.
            cache_metadata: Cache management metadata.
            apply_lm_head: Whether to compute vocabulary logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            **kwargs: Additional model-specific arguments.

        Returns:
            VLMCausalLMOutput: A dataclass containing:
                - logits: Vocabulary logits (None if apply_lm_head=False)
                - past_key_values: Updated KV cache
                - hidden_states: All layer hidden states (if requested)
                - last_hidden_state: Final layer hidden states
                - attentions: All attention weights (if requested)
                - image_hidden_states: Vision encoder outputs (if available)
                - video_hidden_states: Video encoder outputs (if available)
                - rope_deltas: RoPE position deltas for mRoPE models
                - router_logits: MoE router outputs (for MoE models)
                - aux_loss: Router auxiliary loss (for MoE models)

        Example:
            Image captioning::

                outputs = model(
                    input_ids=input_ids,  # Contains <image> placeholders
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                )
                next_token_logits = outputs.logits[:, -1, :]

            Visual question answering::

                # input_ids: "Question: What is in the image? Answer:"
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )
                # Generate answer autoregressively
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

        Extends the base model's input preparation to include vision inputs
        (pixel_values) which are needed for the first generation step.

        Args:
            input_ids: Input token IDs to start generation from.
            max_length: Maximum sequence length for generation.
            pad_token_id: Token ID to use for padding.
            starts: Starting positions for generation.
            pixel_values: Image pixel values to include in inputs.
            attention_mask: Attention mask for the input.
            **kwargs: Additional arguments passed to base model.

        Returns:
            dict: Prepared inputs including pixel_values for the model.
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

        Vision inputs are only used on the first generation step (when the
        KV cache is empty). On subsequent steps, the vision embeddings are
        already encoded in the cached key/value states, so vision inputs
        should be removed to avoid redundant processing.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs: Current model kwargs dictionary.

        Returns:
            dict: Updated kwargs with vision inputs removed.
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
