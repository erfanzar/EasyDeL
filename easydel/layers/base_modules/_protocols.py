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

"""Protocol definitions for base model interfaces.

This module defines structural types (Protocols) that base models must conform to
in order to be used with the generic task modules. Protocols provide runtime type
checking and better IDE support.
"""

from typing import Any, Protocol, runtime_checkable

import jax
from eformer import common_types
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)


@runtime_checkable
class BaseModelProtocol(Protocol):
    """Protocol defining the expected interface for decoder-only base models.

    Any model that implements these methods can be used with the generic task modules.
    This includes models like GPT, LLaMA, Mistral, etc.

    Attributes:
        config: Model configuration object
        dtype: Data type for computations
        param_dtype: Data type for parameters
        precision: Precision setting for JAX operations
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

        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Attention mask
            position_ids: Position IDs for positional embeddings
            mode: Runtime mode (train, eval, etc.)
            past_key_values: Cached key-value states
            cache_metadata: Metadata for cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states

        Returns:
            Model outputs including hidden states, attentions, etc.
        """
        ...

    def get_embedding(self) -> nn.Module:
        """Returns the embedding layer of the model."""
        ...

    def get_decoder(self) -> nn.Module:
        """Returns the decoder part of the model."""
        ...


@runtime_checkable
class EncoderDecoderProtocol(BaseModelProtocol, Protocol):
    """Protocol for encoder-decoder models (e.g., T5, BART).

    Extends BaseModelProtocol with encoder-specific methods.
    """

    def get_encoder(self) -> nn.Module:
        """Returns the encoder part of the model."""
        ...


@runtime_checkable
class VisionModelProtocol(Protocol):
    """Protocol for vision models (e.g., ViT, CLIP).

    Defines the interface for models that process image inputs.

    Attributes:
        config: Model configuration object
        dtype: Data type for computations
        param_dtype: Data type for parameters
        precision: Precision setting for JAX operations
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

        Args:
            pixel_values: Input image pixel values
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states

        Returns:
            Model outputs including hidden states, attentions, etc.
        """
        ...

    def get_embedding(self) -> nn.Module:
        """Returns the embedding/patch projection layer."""
        ...


@runtime_checkable
class VisionLanguageProtocol(Protocol):
    """Protocol for vision-language models (e.g., LLaVA, Qwen2-VL).

    Defines the interface for multimodal models.

    Attributes:
        config: Model configuration object
        dtype: Data type for computations
        param_dtype: Data type for parameters
        precision: Precision setting for JAX operations
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
        """Forward pass through the vision-language model."""
        ...

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision encoder."""
        ...

    def get_language_model(self) -> nn.Module:
        """Returns the language model."""
        ...
