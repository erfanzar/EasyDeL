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

"""Protocol definitions for base model interfaces.

This module defines structural types (Protocols) that base models must conform to
in order to be used with the generic task modules. Protocols provide runtime type
checking and better IDE support through Python's typing system.

Protocols in this module define the expected interfaces for different model
architectures:
    - BaseModelProtocol: Standard decoder-only transformers (GPT, LLaMA, etc.)
    - EncoderDecoderProtocol: Encoder-decoder models (T5, BART, etc.)
    - VisionModelProtocol: Vision models (ViT, CLIP vision, etc.)
    - VisionLanguageProtocol: Multimodal VLM models (LLaVA, Qwen2-VL, etc.)

Using Protocols enables:
    - Static type checking with mypy
    - Runtime isinstance() checks with @runtime_checkable
    - Better IDE autocompletion and documentation
    - Clear interface contracts for model implementations

Example:
    Checking if a model conforms to a protocol:

    ```python
    from easydel.modules._base import BaseModelProtocol

    model = LlamaModel(config, rngs=rngs)

    # Runtime check
    if isinstance(model, BaseModelProtocol):
        outputs = model(input_ids=input_ids)

    # Type annotation
    def process_model(model: BaseModelProtocol) -> BaseModelOutput:
        return model(input_ids=ids)
    ```

See Also:
    - typing.Protocol: Python's structural subtyping system
    - typing.runtime_checkable: Decorator for isinstance() support
"""

from typing import Any, Protocol, runtime_checkable

import jax
from eformer import common_types
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.infra.modeling_outputs import BaseModelOutput


@runtime_checkable
class BaseModelProtocol(Protocol):
    """Protocol defining the expected interface for decoder-only base models.

    Any model that implements these methods can be used with the generic task
    modules. This includes decoder-only autoregressive models like GPT, LLaMA,
    Mistral, Qwen, etc.

    This protocol uses @runtime_checkable to enable isinstance() checks at
    runtime, allowing dynamic verification that a model conforms to the
    expected interface.

    Attributes:
        config (Any): Model configuration object containing hyperparameters
            like hidden_size, num_layers, vocab_size, etc.
        dtype (jnp.dtype): Data type for computations (e.g., jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for model parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX matrix
            multiplication operations.

    Example:
        Implementing a model that conforms to this protocol:

        ```python
        class MyModel:
            config: MyConfig
            dtype: jnp.dtype
            param_dtype: jnp.dtype
            precision: jax.lax.PrecisionLike

            def __call__(self, input_ids, ...):
                ...

            def get_embedding(self):
                return self.embed_tokens

            def get_decoder(self):
                return self.decoder
        ```

    Note:
        The __call__ method signature shows all supported parameters, but
        implementations may not use all of them. Required parameters are
        typically input_ids or inputs_embeds.
    """

    config: Any
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    precision: jax.lax.PrecisionLike

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the base model.

        Processes input tokens or embeddings through the transformer layers
        and returns hidden states and optional attention weights.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Mutually exclusive with inputs_embeds.
            inputs_embeds: Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_dim). Mutually exclusive
                with input_ids.
            attention_mask: Mask of shape (batch_size, sequence_length) where
                1 indicates valid positions and 0 indicates padding.
            mask_info: Structured mask information for advanced attention
                patterns (e.g., sliding window, block sparse).
            position_ids: Position indices of shape (batch_size, sequence_length)
                for positional embeddings. If None, computed automatically.
            mode: Runtime mode affecting computation (e.g., train, prefill,
                decode, insert).
            past_key_values: Cached key-value states for efficient generation.
                Can be TransformerCache, RaggedPagesCache, or HybridCache.
            cache_metadata: Metadata for cache operations including sequence
                positions and valid lengths.
            output_attentions: Whether to return attention weights from all
                layers. Defaults to config setting.
            output_hidden_states: Whether to return hidden states from all
                layers. Defaults to config setting.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer hidden states
                - hidden_states: All layer hidden states (if output_hidden_states)
                - attentions: Attention weights (if output_attentions)
                - past_key_values: Updated cache (if using caching)
        """
        ...

    def get_embedding(self) -> nn.Module:
        """Return the input embedding layer of the model.

        The embedding layer converts token IDs to dense vector representations.
        This is typically an nn.Embed module with a matrix of shape
        (vocab_size, hidden_size).

        Returns:
            The embedding module, typically with an `embedding` attribute
            containing the embedding matrix.
        """
        ...

    def get_decoder(self) -> nn.Module:
        """Return the decoder (transformer layers) part of the model.

        For decoder-only models, this returns the stack of transformer layers
        that process the embeddings.

        Returns:
            The decoder module containing transformer layers.
        """
        ...


@runtime_checkable
class EncoderDecoderProtocol(BaseModelProtocol, Protocol):
    """Protocol for encoder-decoder models (e.g., T5, BART).

    Extends BaseModelProtocol with encoder-specific methods for models that
    have separate encoder and decoder components. The encoder processes the
    input sequence, and the decoder generates the output sequence while
    attending to the encoder outputs.

    This protocol inherits all requirements from BaseModelProtocol and adds
    the get_encoder method.

    Example:
        ```python
        class T5Model(EncoderDecoderProtocol):
            def get_encoder(self):
                return self.encoder

            def get_decoder(self):
                return self.decoder

            # ... other BaseModelProtocol methods
        ```

    Note:
        The __call__ method for encoder-decoder models typically accepts
        additional parameters like encoder_outputs, decoder_input_ids, etc.
    """

    def get_encoder(self) -> nn.Module:
        """Return the encoder part of the model.

        The encoder processes the input sequence and produces contextualized
        representations that the decoder attends to.

        Returns:
            The encoder module, typically a stack of transformer encoder layers.
        """
        ...


@runtime_checkable
class VisionModelProtocol(Protocol):
    """Protocol for vision models (e.g., ViT, CLIP vision encoder).

    Defines the interface for models that process image inputs and produce
    visual representations. These models are typically used standalone for
    image classification or as components in vision-language models.

    Attributes:
        config (Any): Model configuration containing vision-specific parameters
            like image_size, patch_size, hidden_size, num_layers, etc.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for model parameters.
        precision (jax.lax.PrecisionLike): Precision setting for operations.

    Example:
        ```python
        class ViTModel(VisionModelProtocol):
            def __call__(self, pixel_values, ...):
                # Process images through patch embedding and transformer
                ...

            def get_embedding(self):
                return self.patch_embed
        ```
    """

    config: Any
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    precision: jax.lax.PrecisionLike

    def __call__(
        self,
        pixel_values: Float[Array, "batch channels height width"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the vision model.

        Processes input images through patch embedding and transformer layers
        to produce visual representations.

        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width).
                Typically normalized to [-1, 1] or [0, 1] range.
            output_attentions: Whether to return attention weights from all
                vision transformer layers.
            output_hidden_states: Whether to return hidden states from all
                layers, useful for feature extraction from intermediate layers.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer hidden states of shape
                    (batch_size, num_patches + 1, hidden_size) where +1 is for
                    the CLS token if present.
                - hidden_states: All layer outputs (if output_hidden_states)
                - attentions: Attention weights (if output_attentions)
        """
        ...

    def get_embedding(self) -> nn.Module:
        """Return the patch embedding/projection layer.

        The embedding layer converts image patches to dense vectors. For ViT,
        this is typically a Conv2D that projects patches to hidden_size.

        Returns:
            The patch embedding module.
        """
        ...


@runtime_checkable
class VisionLanguageProtocol(Protocol):
    """Protocol for vision-language models (e.g., LLaVA, Qwen2-VL).

    Defines the interface for multimodal models that process both image and
    text inputs. These models combine a vision encoder with a language model,
    using a projector to align visual and textual representations.

    VLMs typically:
        1. Encode images with a vision tower (e.g., CLIP, SigLIP)
        2. Project visual features to language model dimension
        3. Merge visual tokens with text tokens
        4. Process combined sequence with the language model

    Attributes:
        config (Any): Model configuration containing both vision and language
            parameters, plus multimodal-specific settings.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for model parameters.
        precision (jax.lax.PrecisionLike): Precision setting for operations.

    Example:
        ```python
        class LlavaModel(VisionLanguageProtocol):
            def __call__(self, input_ids, pixel_values, ...):
                # 1. Get text embeddings
                # 2. Get image features from vision tower
                # 3. Project image features
                # 4. Merge at placeholder positions
                # 5. Forward through language model
                ...

            def get_vision_tower(self):
                return self.vision_tower

            def get_language_model(self):
                return self.language_model
        ```
    """

    config: Any
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    precision: jax.lax.PrecisionLike

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Float[Array, "batch channels height width"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the vision-language model.

        Processes both text and image inputs, merging visual features into
        the text sequence at appropriate positions (typically at placeholder
        tokens like <image>).

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Should contain placeholder tokens where images should be inserted.
            pixel_values: Input images of shape (batch_size, channels, height, width)
                or (batch_size, num_images, channels, height, width) for multi-image.
            attention_mask: Mask of shape (batch_size, sequence_length).
            mask_info: Structured mask information.
            position_ids: Position indices for positional embeddings.
            mode: Runtime mode (train, prefill, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Metadata for cache operations.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            BaseModelOutput with language model hidden states after processing
            the combined vision-language sequence.
        """
        ...

    def get_vision_tower(self) -> nn.Module:
        """Return the vision encoder component.

        The vision tower processes images and produces visual features.
        Common implementations include CLIP, SigLIP, or custom ViT models.

        Returns:
            The vision encoder module.
        """
        ...

    def get_language_model(self) -> nn.Module:
        """Return the language model component.

        The language model processes the combined vision-text sequence and
        generates outputs. This is typically a decoder-only model like LLaMA.

        Returns:
            The language model module.
        """
        ...
