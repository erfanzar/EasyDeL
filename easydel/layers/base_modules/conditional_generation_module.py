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

"""Generic base class for Conditional Generation tasks.

This module provides BaseConditionalGenerationModule for encoder-decoder
models like T5, BART, or vision-language models like LLaVA.
"""

from collections.abc import Callable

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.modeling_outputs import Seq2SeqLMOutput
from easydel.infra.utils import auto_remat, get_dot_general_by_bits
from easydel.layers.caching import (
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseConditionalGenerationModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Conditional Generation (Seq2Seq).

    This class supports both encoder-decoder models (T5, BART) and
    vision-language models (LLaVA, Qwen2-VL).

    Example:
        ```python
        class LlavaForConditionalGeneration(
            BaseConditionalGenerationModule[LlavaModel, LlavaConfig]
        ):
            _task_type = TaskType.SEQ_2_SEQ_LM
            _model_type = "llava"
            _config_class = LlavaConfig
        ```

    Type Parameters:
        ModelT: The base model type (encoder-decoder or vision-language)
        ConfigT: The configuration type

    Attributes:
        lm_head: The language modeling head
    """

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
        # Feature flags
        tie_word_embeddings: bool = False,
        logit_cap: float | None = None,
        # LM head configuration
        lm_head_bias: bool = False,
        lm_head_kernel_init: Callable | None = None,
    ):
        """Initialize the Conditional Generation module.

        Args:
            config: Model configuration
            base_model: Pre-instantiated base model (optional)
            base_model_class: Base model class to instantiate (optional)
            base_model_name: Attribute name for the base model
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
            tie_word_embeddings: Whether to tie embeddings with LM head
            logit_cap: Maximum absolute value for logits
            lm_head_bias: Whether to use bias in LM head
            lm_head_kernel_init: Custom kernel initializer
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
            head_bias=lm_head_bias,
            head_kernel_init=lm_head_kernel_init,
        )

        # Create LM head
        lm_head_block = ColumnParallelLinear
        if self._gradient_checkpointing_feature.should_checkpoint():
            lm_head_block = auto_remat(
                lm_head_block,
                **self._gradient_checkpointing_feature.get_config(),
            )

        self.lm_head = lm_head_block(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self._head_bias,
            kernel_init=self._head_kernel_init,
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(
                getattr(config, "bits", None),
                getattr(config, "easy_method", ""),
            ),
        )

        if tie_word_embeddings:
            self._tie_embeddings_feature.setup(self.get_embedding(), self.lm_head)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        # Decoder inputs for encoder-decoder models
        decoder_input_ids: Int[Array, "batch dec_seq_len"] | None = None,
        decoder_attention_mask: Bool[Array, "batch dec_seq_len"] | None = None,
        # Vision inputs for multimodal models
        pixel_values: Float[Array, "batch channels height width"] | None = None,
        # Common arguments
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        """Forward pass for conditional generation.

        Args:
            input_ids: Input token IDs (encoder input or decoder-only input)
            inputs_embeds: Input embeddings
            attention_mask: Attention mask
            position_ids: Position IDs
            decoder_input_ids: Decoder input IDs (for encoder-decoder)
            decoder_attention_mask: Decoder attention mask
            pixel_values: Image pixel values (for vision-language)
            mode: Runtime mode
            past_key_values: KV cache
            cache_metadata: Cache metadata
            apply_lm_head: Whether to apply LM head
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional model-specific arguments

        Returns:
            Seq2SeqLMOutput with logits and model outputs
        """
        # Forward through base model
        # The base model handles encoder-decoder or vision-language logic
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            pixel_values=pixel_values,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Apply logical sharding
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        # Apply LM head if requested
        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")
            lm_logits = self.apply_logit_cap(lm_logits)

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=getattr(outputs, "cross_attentions", None),
            encoder_last_hidden_state=getattr(outputs, "encoder_last_hidden_state", None),
            encoder_hidden_states=getattr(outputs, "encoder_hidden_states", None),
            encoder_attentions=getattr(outputs, "encoder_attentions", None),
        )

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head."""
        return self.lm_head(hidden_states)

    def get_task_head(self):
        """Returns the LM head."""
        return self.lm_head

    def get_lm_head(self):
        """Alias for get_task_head."""
        return self.get_task_head()

    def get_encoder(self):
        """Returns the encoder (if applicable).

        For encoder-decoder models, this returns the encoder.
        For decoder-only models, this raises NotImplementedError.
        """
        if hasattr(self.base_model, "get_encoder"):
            return self.base_model.get_encoder()
        raise NotImplementedError("This model does not have an encoder.")
