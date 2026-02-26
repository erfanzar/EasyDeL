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

"""Generic base class for Conditional Generation tasks.

This module provides BaseConditionalGenerationModule for encoder-decoder
models like T5, BART, as well as vision-language models like LLaVA.

Conditional generation models generate output sequences conditioned on
input sequences, which can include:
    - Text-to-text: Translation, summarization (T5, BART)
    - Image-to-text: Image captioning, VQA (LLaVA, BLIP)
    - Speech-to-text: Transcription (Whisper)

Key Features:
    - Support for encoder-decoder architectures
    - Support for vision-language models
    - Configurable LM head with optional weight tying
    - Logit capping for training stability
    - Automatic registration with EasyDeL factory system

Example:
    Creating a conditional generation model:

    ```python
    class T5ForConditionalGeneration(
        BaseConditionalGenerationModule[T5Model, T5Config]
    ):
        _task_type = TaskType.SEQ_2_SEQ_LM
        _model_type = "t5"
        _config_class = T5Config

        def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
            super().__init__(
                config=config,
                base_model_class=T5Model,
                base_model_name="model",
                dtype=dtype,
                rngs=rngs,
                tie_word_embeddings=True,
            )
    ```

See Also:
    - BaseTaskModule: Parent class with common functionality
    - BaseVisionLanguageModule: Specialized subclass for VLMs
    - easydel.infra.modeling_outputs.Seq2SeqLMOutput: Output dataclass
"""

from collections.abc import Callable

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
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
from easydel.infra.modeling_outputs import Seq2SeqLMOutput
from easydel.infra.utils import auto_remat
from easydel.layers import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseConditionalGenerationModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Conditional Generation (Seq2Seq).

    This class supports both encoder-decoder models (T5, BART) and
    vision-language models (LLaVA, Qwen2-VL). It provides the common
    infrastructure for models that generate output sequences conditioned
    on input sequences.

    The model consists of:
        - Base model: Encoder-decoder or VLM that processes inputs
        - LM head: Projects hidden states to vocabulary logits

    Example:
        Explicit subclassing:

        ```python
        class LlavaForConditionalGeneration(
            BaseConditionalGenerationModule[LlavaModel, LlavaConfig]
        ):
            _task_type = TaskType.SEQ_2_SEQ_LM
            _model_type = "llava"
            _config_class = LlavaConfig

            def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                super().__init__(
                    config=config,
                    base_model_class=LlavaModel,
                    base_model_name="model",
                    dtype=dtype,
                    rngs=rngs,
                    tie_word_embeddings=config.tie_word_embeddings,
                )
        ```

    Type Parameters:
        ModelT: The base model type (encoder-decoder or vision-language).
            Must implement __call__ with appropriate inputs and return
            an object with last_hidden_state.
        ConfigT: The configuration type. Must have vocab_size and either
            hidden_size or get_text_config() method.

    Attributes:
        lm_head (ColumnParallelLinear): The language modeling head that
            projects hidden states to vocabulary logits.
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
        lm_head_name: str = "lm_head",
        lm_head_bias: bool = False,
        lm_head_kernel_init: Callable | None = None,
        lm_head_class: nn.Module = ColumnParallelLinear,
        create_lm_head: bool = True,
    ):
        """Initialize the Conditional Generation module.

        Sets up the base model and language modeling head for sequence-to-sequence
        or conditional generation tasks.

        Args:
            config: Model configuration. Must have:
                - vocab_size: Size of the vocabulary
                - hidden_size or get_text_config().hidden_size: Hidden dimension
                Can also have optional attributes like tie_word_embeddings,
                gradient_checkpointing, etc.
            base_model: Pre-instantiated base model. If provided,
                base_model_class is ignored.
            base_model_class: Model class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name for storing the base model.
                Defaults to "model".
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: JAX precision setting for matrix operations.
            rngs: Flax NNX random number generators.
            tie_word_embeddings: Whether to share weights between input
                embeddings and LM head. Reduces parameters and can improve
                generalization. Defaults to False.
            logit_cap: Maximum absolute value for logits. If set, logits
                are clipped to [-logit_cap, logit_cap]. Useful for training
                stability. Defaults to None (no capping).
            lm_head_name: Attribute name for the LM head. Defaults to "lm_head".
            lm_head_bias: Whether to use bias in the LM head. Defaults to False.
            lm_head_kernel_init: Custom kernel initializer for the LM head.
            lm_head_class: Class to use for the LM head. Defaults to
                ColumnParallelLinear for tensor parallelism support.
            create_lm_head: Whether to create a new LM head. Set to False if
                the base model already has an LM head. Defaults to True.

        Example:
            ```python
            # Encoder-decoder model with weight tying
            model = BaseConditionalGenerationModule(
                config=t5_config,
                base_model_class=T5Model,
                tie_word_embeddings=True,
                rngs=nn.Rngs(0),
            )

            # VLM without creating new LM head (using base model's head)
            model = BaseConditionalGenerationModule(
                config=llava_config,
                base_model_class=LlavaModel,
                create_lm_head=False,
                rngs=nn.Rngs(0),
            )
            ```
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

        self._lm_head_name = lm_head_name

        if create_lm_head:
            head_block = lm_head_class
            if self._gradient_checkpointing_feature.should_checkpoint():
                head_block = auto_remat(
                    head_block,
                    **self._gradient_checkpointing_feature.get_config(),
                )

            lm_head = head_block(
                config.get_text_config().hidden_size,
                config.get_text_config().vocab_size,
                dtype=dtype,
                param_dtype=param_dtype,
                use_bias=self._head_bias,
                kernel_init=self._head_kernel_init,
                precision=precision,
                rngs=rngs,
            )
            setattr(self, lm_head_name, lm_head)

            if tie_word_embeddings:
                self._tie_embeddings_feature.setup(self.get_embedding(), lm_head)

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
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        """Forward pass for conditional generation.

        Processes inputs through the encoder-decoder or VLM architecture and
        produces output logits for next token prediction.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len). For
                encoder-decoder models, these are encoder inputs. For VLMs,
                these may contain image placeholder tokens.
            inputs_embeds: Pre-computed input embeddings of shape
                (batch_size, seq_len, hidden_dim). Alternative to input_ids.
            attention_mask: Mask of shape (batch_size, seq_len) where 1 indicates
                valid positions and 0 indicates padding.
            mask_info: Structured mask information for advanced attention patterns.
            position_ids: Position indices for positional embeddings.
            decoder_input_ids: Decoder input token IDs for encoder-decoder models.
                Shape: (batch_size, decoder_seq_len).
            decoder_attention_mask: Decoder attention mask for encoder-decoder models.
            pixel_values: Image inputs for vision-language models. Shape:
                (batch_size, channels, height, width).
            mode: Runtime mode affecting computation (train, prefill, decode).
            past_key_values: Cached key-value states for efficient generation.
            cache_metadata: Metadata for cache operations.
            apply_lm_head: Whether to apply the LM head and return logits.
                Set to False to get hidden states only. Defaults to True.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            **kwargs: Additional model-specific arguments passed to base model.

        Returns:
            Seq2SeqLMOutput containing:
                - logits: Vocabulary logits of shape (batch, seq_len, vocab_size)
                  if apply_lm_head=True, else None.
                - past_key_values: Updated KV cache for generation.
                - decoder_hidden_states: Decoder hidden states if requested.
                - decoder_attentions: Decoder attention weights if requested.
                - cross_attentions: Cross-attention weights if applicable.
                - encoder_last_hidden_state: Final encoder hidden states.
                - encoder_hidden_states: All encoder hidden states if requested.
                - encoder_attentions: Encoder attention weights if requested.

        Example:
            ```python
            # Training forward pass
            outputs = model(
                input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=encoder_attention_mask,
            )
            loss = cross_entropy(outputs.logits, labels)

            # Generation with caching
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                mode=MODE_DECODE,
            )
            next_token = sample(outputs.logits[:, -1])
            ```
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

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

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
        """Apply the language modeling head to hidden states.

        Projects hidden states to vocabulary logits, optionally using tied
        embedding weights.

        Args:
            hidden_states: Hidden states of shape (batch, seq_len, hidden_dim)
                from the decoder or language model.

        Returns:
            Logits of shape (batch, seq_len, vocab_size) representing
            unnormalized probabilities over the vocabulary.

        Note:
            When weight tying is enabled (via config), this method passes
            the transposed embedding weights to the LM head instead of using
            separate projection weights.
        """
        tie_embeddings = next(
            (
                getattr(self.config, key)
                for key in ["tie_word_embeddings", "use_lm_head", "share_input_output_layers"]
                if hasattr(self.config, key)
            ),
            False,
        )
        w = self.get_embedding().embedding.value.T if tie_embeddings else None
        lm_head = self.get_task_head()
        return lm_head(hidden_states, w=w)

    def get_task_head(self):
        """Return the language modeling head.

        Returns:
            The LM head module that projects hidden states to vocabulary logits.

        Raises:
            AttributeError: If no LM head is found with the configured name
                and the base model doesn't have a get_lm_head method.
        """
        if hasattr(self, self._lm_head_name):
            return getattr(self, self._lm_head_name)
        if hasattr(self.base_model, "get_lm_head"):
            return self.base_model.get_lm_head()

        raise AttributeError(f"{self.__class__.__name__} has no LM head named '{self._lm_head_name}'.")

    def get_lm_head(self):
        """Return the language modeling head.

        Alias for get_task_head() for API compatibility.

        Returns:
            The LM head module.
        """
        return self.get_task_head()

    def get_encoder(self):
        """Return the encoder component if applicable.

        For encoder-decoder models, returns the encoder that processes
        input sequences. For decoder-only models used in conditional
        generation, raises NotImplementedError.

        Returns:
            The encoder module for encoder-decoder architectures.

        Raises:
            NotImplementedError: If the model doesn't have an encoder.
        """
        if hasattr(self.base_model, "get_encoder"):
            return self.base_model.get_encoder()
        raise NotImplementedError("This model does not have an encoder.")
