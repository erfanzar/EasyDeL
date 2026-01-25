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

"""Generic base class for Causal Language Modeling tasks.

This module provides BaseCausalLMModule, a generic, type-safe base class for
creating ForCausalLM model wrappers with minimal boilerplate.

Causal language modeling (also known as autoregressive language modeling) is
the task of predicting the next token given the previous tokens. This is the
foundation for text generation models like GPT, Llama, and Mistral.

Key Features:
    - Generic typing with ModelT and ConfigT type parameters
    - Automatic weight tying between input embeddings and LM head
    - Logit capping for numerical stability
    - MoE router auxiliary loss computation for sparse models
    - Gradient checkpointing support for memory efficiency
    - Full KV caching support for efficient inference
    - Customizable LM head configuration

Example:
    Creating a causal LM model::

        from easydel.layers.base_modules import BaseCausalLMModule

        class LlamaForCausalLM(BaseCausalLMModule[LlamaModel, LlamaConfig]):
            _task_type = TaskType.CAUSAL_LM
            _model_type = "llama"
            _config_class = LlamaConfig

            def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                super().__init__(
                    config=config,
                    base_model_class=LlamaModel,
                    dtype=dtype,
                    rngs=rngs,
                    tie_word_embeddings=config.tie_word_embeddings,
                )

See Also:
    - BaseTaskModule: Parent class with common task functionality
    - BaseConditionalGenerationModule: For encoder-decoder generation
    - BaseSequenceClassificationModule: For sequence classification tasks
"""

import typing
from collections.abc import Callable

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.modeling_outputs import CausalLMOutput, MoeCausalLMOutput
from easydel.infra.utils import auto_remat
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.layers.components import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseCausalLMModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Causal Language Modeling.

    This class provides a fully-featured, type-safe base for creating ForCausalLM
    model wrappers with comprehensive support for modern language model features:

    - Generic typing (ModelT, ConfigT) for type safety and IDE support
    - Automatic model registration via class attributes
    - Modular features including:
        - Logit capping for numerical stability (e.g., Gemma models)
        - Weight tying between input embeddings and LM head
        - MoE router auxiliary loss computation
    - Gradient checkpointing for memory-efficient training
    - Full KV caching support for efficient autoregressive generation
    - Configurable LM head with custom naming and initialization

    The causal language modeling task involves predicting the next token at each
    position, conditioned only on previous tokens (no future information leakage).
    This enables autoregressive text generation.

    Example:
        Basic usage with a custom model::

            class ArcticForCausalLM(BaseCausalLMModule[ArcticModel, ArcticConfig]):
                _task_type = TaskType.CAUSAL_LM
                _model_type = "arctic"
                _config_class = ArcticConfig

                def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                    super().__init__(
                        config=config,
                        base_model_class=ArcticModel,
                        base_model_name="model",
                        dtype=dtype,
                        rngs=rngs,
                        router_aux_loss_coef=0.001,
                    )

        Using the model for text generation::

            model = ArcticForCausalLM(config, rngs=rngs)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]  # Get last position
            next_token = jnp.argmax(next_token_logits, axis=-1)

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol).
            This is the underlying transformer that produces hidden states.
        ConfigT: The configuration type containing model hyperparameters.
            Must have `hidden_size` and `vocab_size` attributes.

    Attributes:
        lm_head: The language modeling head (linear projection from hidden_size
            to vocab_size). The actual attribute name can be customized via
            `lm_head_name` parameter.
        _lm_head_name (str): The name of the LM head attribute.
        base_model: The underlying transformer model.

    Note:
        For MoE (Mixture of Experts) models, this class automatically detects
        router outputs and computes auxiliary load balancing losses when
        `router_aux_loss_coef` is configured.
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
        router_aux_loss_coef: float | None = None,
        # LM head configuration
        lm_head_name: str = "lm_head",
        lm_head_bias: bool = False,
        lm_head_kernel_init: Callable | None = None,
        lm_head_class: nn.Module = ColumnParallelLinear,
    ):
        """Initialize the Causal LM module.

        Creates a causal language model by wrapping a base transformer model
        and adding a language modeling head that projects hidden states to
        vocabulary logits.

        Args:
            config: Model configuration object. Must have the following attributes:
                - hidden_size (int): Dimension of hidden states
                - vocab_size (int): Size of the vocabulary
                - gradient_checkpointing (str, optional): Checkpointing policy
                - tie_word_embeddings (bool, optional): Config-level embedding tying
            base_model: Pre-instantiated base model instance. If provided,
                base_model_class is ignored. Useful for sharing weights.
            base_model_class: Base model class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name for storing the base model.
                Common values: "model", "transformer", "gpt".
                Defaults to "model".
            dtype: Data type for computations (activations). bfloat16 provides
                good balance of speed and precision for most models.
            param_dtype: Data type for parameters (weights). Defaults to bfloat16.
            precision: JAX precision setting for matrix multiplications.
                Can be None, "high", "highest", or a Precision enum value.
            rngs: Flax random number generators for initialization.
            tie_word_embeddings: Whether to tie input embeddings with the LM head.
                When True, the LM head uses the transposed embedding matrix,
                reducing parameters. Common in smaller models.
            logit_cap: Maximum absolute value for output logits. If set, logits
                are capped using tanh scaling: cap * tanh(logits / cap).
                Used in Gemma models for numerical stability.
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss.
                Used for load balancing in Mixture-of-Experts models.
                None disables aux loss computation.
            lm_head_name: Attribute name for the LM head. Defaults to "lm_head".
                Some models use "output", "embed_out", or "lm_head".
            lm_head_bias: Whether to include bias in the LM head.
                Defaults to False for standard transformer LMs.
            lm_head_kernel_init: Custom initializer for LM head weights.
                If None, uses the default Flax initializer.
            lm_head_class: Module class for the LM head.
                Defaults to ColumnParallelLinear for tensor parallelism support.

        Example:
            Creating a model with weight tying::

                model = BaseCausalLMModule(
                    config=my_config,
                    base_model_class=MyTransformerModel,
                    dtype=jnp.bfloat16,
                    rngs=nn.Rngs(0),
                    tie_word_embeddings=True,
                    logit_cap=30.0,  # For numerical stability
                )

            Creating an MoE model with auxiliary loss::

                model = BaseCausalLMModule(
                    config=moe_config,
                    base_model_class=MoETransformerModel,
                    rngs=nn.Rngs(0),
                    router_aux_loss_coef=0.01,
                )
        """
        # Initialize base with features
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
            router_aux_loss_coef=router_aux_loss_coef,
            head_bias=lm_head_bias,
            head_kernel_init=lm_head_kernel_init,
        )

        # Store LM head name for dynamic access
        self._lm_head_name = lm_head_name

        # Create LM head with optional gradient checkpointing

        if self._gradient_checkpointing_feature.should_checkpoint():
            lm_head_class = auto_remat(
                lm_head_class,
                **self._gradient_checkpointing_feature.get_config(),
            )

        # Create LM head with custom attribute name
        lm_head = lm_head_class(
            config.hidden_size,
            config.vocab_size,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass through the Causal LM model.

        Processes input through the base transformer model, applies the language
        modeling head, and returns logits over the vocabulary for each position.

        Args:
            input_ids: Input token IDs with shape (batch_size, sequence_length).
                Each value should be a valid token ID from the vocabulary.
                Either input_ids or inputs_embeds must be provided.
            inputs_embeds: Pre-computed input embeddings with shape
                (batch_size, sequence_length, hidden_dim). Alternative to input_ids
                for passing custom embeddings. Useful for embedding manipulation
                or multimodal inputs.
            attention_mask: Binary mask with shape (batch_size, sequence_length).
                1 for tokens to attend to, 0 for padding tokens. Causal masking
                is applied automatically on top of this mask.
            mask_info: Structured mask information for advanced attention patterns.
                Used by optimized kernels for efficient masking.
            position_ids: Position indices with shape (batch_size, sequence_length).
                If None, positions are inferred from input_ids accounting for
                padding. Important for correct positional encoding.
            mode: Runtime mode controlling model behavior:
                - MODE_TRAIN: Training mode with full forward pass
                - MODE_EVAL: Evaluation mode
                - MODE_PREFILL: Prefill phase for generation
                - MODE_DECODE: Autoregressive decoding phase
            past_key_values: Cached key/value states from previous forward passes.
                Essential for efficient autoregressive generation. Supported types:
                - TransformerCache: Standard KV cache
                - RaggedPagesCache: Paged attention cache
                - HybridCache: Combined cache types
            cache_metadata: Metadata for cache management including sequence
                lengths, cache indices, and operation types.
            apply_lm_head: Whether to apply the LM head and return logits.
                If False, returns None for logits but still computes hidden states.
                Useful for extracting representations without classification.
            output_attentions: If True, include attention weights in the output.
                May increase memory usage significantly for long sequences.
            output_hidden_states: If True, include hidden states from all layers.
                Useful for probing and analysis tasks.

        Returns:
            CausalLMOutput or MoeCausalLMOutput: A dataclass containing:
                - logits: Next-token prediction logits with shape
                    (batch_size, sequence_length, vocab_size). None if apply_lm_head=False.
                - last_hidden_state: Final hidden states before LM head
                - hidden_states: Tuple of hidden states from all layers (if requested)
                - attentions: Tuple of attention weights from all layers (if requested)
                - past_key_values: Updated KV cache for next generation step

            For MoE models, MoeCausalLMOutput additionally contains:
                - aux_loss: Router load balancing loss
                - router_logits: Raw router outputs
                - all_router_losses: Per-layer router losses

        Example:
            Basic text generation step::

                outputs = model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    cache_metadata=metadata,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = jax.random.categorical(key, next_token_logits)

            Training forward pass::

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mode=common_types.MODE_TRAIN,
                )
                loss = cross_entropy(outputs.logits[:, :-1], input_ids[:, 1:])
        """
        # Forward through base model
        # Build kwargs conditionally to support models that don't accept inputs_embeds
        base_model_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mask_info": mask_info,
            "mode": mode,
            "past_key_values": past_key_values,
            "cache_metadata": cache_metadata,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
        }

        # Only pass input_ids or inputs_embeds (not both)
        if inputs_embeds is not None:
            base_model_kwargs["inputs_embeds"] = inputs_embeds
        else:
            base_model_kwargs["input_ids"] = input_ids

        outputs = self.base_model(**base_model_kwargs)

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")

            # Apply logit capping if configured
            lm_logits = self.apply_logit_cap(lm_logits)

        # Compute router auxiliary loss if configured
        aux_loss = self.compute_router_aux_loss(outputs)

        # Return MoeCausalLMOutput for MoE models, CausalLMOutput otherwise
        if aux_loss is not None or hasattr(outputs, "all_router_losses"):
            return MoeCausalLMOutput(
                logits=lm_logits,
                hidden_states=outputs.hidden_states,
                last_hidden_state=outputs.last_hidden_state,
                attentions=outputs.attentions,
                past_key_values=outputs.past_key_values,
                aux_loss=aux_loss,
                router_logits=getattr(outputs, "router_logits", None),
                all_router_losses=getattr(outputs, "all_router_losses", None),
            )
        else:
            return CausalLMOutput(
                logits=lm_logits,
                hidden_states=outputs.hidden_states,
                last_hidden_state=outputs.last_hidden_state,
                attentions=outputs.attentions,
                past_key_values=outputs.past_key_values,
            )

    def forward_moe(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values=None,
        cache_metadata=None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        aux_loss_fn: typing.Callable | None = None,
    ) -> MoeCausalLMOutput:
        """Forward pass for MoE models with router auxiliary loss handling.

        This method provides a standardized interface for Mixture-of-Experts models
        that need specialized handling for:

        1. Router logit output control via output_router_logits
        2. Custom auxiliary loss computation via aux_loss_fn
        3. Proper handling during different inference modes

        Unlike the standard __call__, this method always returns MoeCausalLMOutput
        and provides more control over MoE-specific outputs.

        Args:
            input_ids: Input token IDs with shape (batch_size, sequence_length).
                Either input_ids or inputs_embeds must be provided.
            inputs_embeds: Pre-computed input embeddings with shape
                (batch_size, sequence_length, hidden_dim). Alternative to input_ids.
            attention_mask: Binary mask with shape (batch_size, sequence_length).
                1 for tokens to attend to, 0 for padding tokens.
            mask_info: Structured mask information for optimized kernels.
            position_ids: Position indices for positional embeddings.
            mode: Runtime mode controlling behavior. Auxiliary loss is skipped
                during decode/prefill/insert modes for efficiency.
            past_key_values: Cached key/value states for efficient generation.
            cache_metadata: Metadata for cache management.
            apply_lm_head: Whether to apply the LM head. If False, logits=None.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return raw router logits.
                Defaults to config.output_router_logits if not specified.
            aux_loss_fn: Custom function to compute auxiliary loss.
                Signature: aux_loss_fn(outputs, attention_mask) -> scalar.
                If None, uses the default compute_router_aux_loss method.

        Returns:
            MoeCausalLMOutput: A dataclass containing:
                - logits: Vocabulary logits (None if apply_lm_head=False)
                - last_hidden_state: Final layer hidden states
                - hidden_states: All layer hidden states (if requested)
                - attentions: All attention weights (if requested)
                - past_key_values: Updated KV cache
                - aux_loss: Router auxiliary/load balancing loss
                - router_logits: Raw router outputs (if requested)

        Example:
            Using with custom auxiliary loss::

                def my_aux_loss(outputs, mask):
                    router_logits = outputs.router_logits
                    return compute_load_balancing_loss(router_logits, mask)

                outputs = model.forward_moe(
                    input_ids=input_ids,
                    aux_loss_fn=my_aux_loss,
                    output_router_logits=True,
                )
                total_loss = lm_loss + outputs.aux_loss

        Note:
            Auxiliary loss computation is automatically skipped during inference
            modes (decode, prefill, insert) as it's only needed for training.
        """
        # Set default for output_router_logits
        if output_router_logits is None:
            output_router_logits = getattr(self.config, "output_router_logits", False)

        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )

        logits = None
        if apply_lm_head:
            logits = checkpoint_name(self.apply_lm_head(outputs.last_hidden_state), "lm_head_output")
            logits = self.apply_logit_cap(logits)

        aux_loss = None
        if aux_loss_fn is not None and mode not in [
            common_types.MODE_DECODE,
            common_types.MODE_PREFILL,
            common_types.MODE_INSERT,
        ]:
            aux_loss = aux_loss_fn(outputs, attention_mask)
        else:
            aux_loss = self.compute_router_aux_loss(outputs)

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
            past_key_values=outputs.past_key_values,
        )

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head to hidden states.

        Projects hidden states to vocabulary logits, optionally using tied
        embeddings (transposed input embedding matrix as the projection).

        The method automatically detects weight tying configuration from
        multiple possible config attribute names for compatibility with
        different model implementations.

        Args:
            hidden_states: Hidden state tensor with shape
                (batch_size, sequence_length, hidden_size).

        Returns:
            Array: Vocabulary logits with shape
                (batch_size, sequence_length, vocab_size).

        Note:
            Weight tying is detected from config attributes in this order:
            1. tie_word_embeddings
            2. use_lm_head
            3. share_input_output_layers

            When weight tying is enabled, the transposed embedding matrix
            is passed to the LM head's forward method via the `w` parameter.
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
        lm_head = getattr(self, self._lm_head_name)
        return lm_head(hidden_states, w=w)

    def get_task_head(self):
        """Returns the language modeling head module.

        Retrieves the linear layer that projects hidden states to vocabulary
        logits. This is the task-specific head for causal language modeling.

        Returns:
            nn.Module: The LM head module (typically ColumnParallelLinear).
                Shape of kernel: (hidden_size, vocab_size).

        Example:
            Accessing LM head weights::

                head = model.get_task_head()
                weights = head.kernel.value  # Shape: (hidden_size, vocab_size)
        """
        return getattr(self, self._lm_head_name)

    # Convenience alias
    def get_lm_head(self):
        """Alias for get_task_head for backwards compatibility.

        Returns the language modeling head module. This method exists for
        API compatibility with code expecting a get_lm_head method.

        Returns:
            nn.Module: The LM head module (same as get_task_head).

        See Also:
            get_task_head: The canonical method for accessing the task head.
        """
        return self.get_task_head()
