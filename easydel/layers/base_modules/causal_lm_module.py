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
from easydel.layers.linear import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseCausalLMModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Causal Language Modeling.

    This class provides a fully-featured, type-safe base for creating ForCausalLM
    model wrappers with support for:
    - Generic typing (ModelT, ConfigT)
    - Automatic registration
    - Modular features (logit cap, tie embeddings, router aux loss)
    - Gradient checkpointing
    - KV caching
    - Configurable LM head

    Example:
        ```python
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
        ```

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol)
        ConfigT: The configuration type

    Attributes:
        lm_head: The language modeling head (linear projection to vocab)
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

        Args:
            config: Model configuration
            base_model: Pre-instantiated base model (optional)
            base_model_class: Base model class to instantiate (optional)
            base_model_name: Attribute name for the base model (e.g., "model", "transformer")
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
            tie_word_embeddings: Whether to tie input embeddings with LM head
            logit_cap: Maximum absolute value for logits (None = no capping)
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss
            lm_head_name: Attribute name for the LM head (e.g., "lm_head", "output", "embed_out")
            lm_head_bias: Whether to use bias in the LM head
            lm_head_kernel_init: Custom kernel initializer for LM head
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

        Args:
            input_ids: Input token IDs, shape (batch_size, sequence_length)
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position IDs for positional embeddings
            mode: Runtime mode (train, eval, etc.)
            past_key_values: Cache containing precomputed key/value states
            cache_metadata: Metadata for cache handling
            apply_lm_head: Whether to apply the LM head and return logits
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states of all layers

        Returns:
            CausalLMOutput containing logits, hidden states, attentions, cache, etc.
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

        This method provides a standardized interface for MoE models that need to:
        1. Handle output_router_logits parameter
        2. Compute auxiliary loss from router logits
        3. Return MoeCausalLMOutput

        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position IDs for positional embeddings
            mode: Runtime mode (train, eval, etc.)
            past_key_values: Cache containing precomputed key/value states
            cache_metadata: Metadata for cache handling
            apply_lm_head: Whether to apply the LM head
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            output_router_logits: Whether to return router logits
            aux_loss_fn: Optional custom function to compute aux_loss from outputs

        Returns:
            MoeCausalLMOutput with logits, aux_loss, router_logits, etc.
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
        """Apply the language modeling head (optionally tied to input embeddings)."""
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
        """Returns the language modeling head.

        Returns:
            The LM head module
        """
        return getattr(self, self._lm_head_name)

    # Convenience alias
    def get_lm_head(self):
        """Alias for get_task_head for backwards compatibility."""
        return self.get_task_head()
