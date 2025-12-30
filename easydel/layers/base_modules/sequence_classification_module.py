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

"""Generic base class for Sequence Classification tasks.

This module provides BaseSequenceClassificationModule, a generic, type-safe base
class for creating ForSequenceClassification model wrappers.
"""

from collections.abc import Callable

import jax
from eformer import common_types
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.modeling_outputs import SequenceClassifierOutput
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


class BaseSequenceClassificationModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Sequence Classification.

    This class provides a fully-featured, type-safe base for creating
    ForSequenceClassification model wrappers with support for:
    - Generic typing (ModelT, ConfigT)
    - Automatic registration
    - Modular features (pooling strategies, router aux loss)
    - Gradient checkpointing
    - Configurable classification head

    Example:
        ```python
        class ArcticForSequenceClassification(
            BaseSequenceClassificationModule[ArcticModel, ArcticConfig]
        ):
            _task_type = TaskType.SEQUENCE_CLASSIFICATION
            _model_type = "arctic"
            _config_class = ArcticConfig

            def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                super().__init__(
                    config=config,
                    base_model_class=ArcticModel,
                    base_model_name="model",
                    dtype=dtype,
                    rngs=rngs,
                    pooling_strategy="last",
                    router_aux_loss_coef=0.001,
                )
        ```

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol)
        ConfigT: The configuration type

    Attributes:
        score: The classification head (linear projection to num_labels)
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
        pooling_strategy: str = "last",
        router_aux_loss_coef: float | None = None,
        # Classification head configuration
        score_head_name: str = "score",
        score_head_bias: bool = False,
        score_head_kernel_init: Callable | None = None,
    ):
        """Initialize the Sequence Classification module.

        Args:
            config: Model configuration (must have num_labels attribute)
            base_model: Pre-instantiated base model (optional)
            base_model_class: Base model class to instantiate (optional)
            base_model_name: Attribute name for the base model (e.g., "model", "transformer")
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
            pooling_strategy: Strategy for pooling sequences ("last", "first", "mean", "max")
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss
            score_head_name: Attribute name for the classification head (e.g., "score", "classifier")
            score_head_bias: Whether to use bias in the classification head
            score_head_kernel_init: Custom kernel initializer for classification head

        Raises:
            AssertionError: If config does not have num_labels attribute
        """
        # Validate config has num_labels
        assert hasattr(config, "num_labels"), (
            "in order to use `SequenceClassification` Models in `EasyDeL` "
            "you first need to attach `num_labels` to model `config`"
        )

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
            router_aux_loss_coef=router_aux_loss_coef,
            pooling_strategy=pooling_strategy,
            head_bias=score_head_bias,
            head_kernel_init=score_head_kernel_init,
        )

        # Store score head name for dynamic access
        self._score_head_name = score_head_name

        # Create classification head with optional gradient checkpointing
        score_head_block = ColumnParallelLinear

        if self._gradient_checkpointing_feature.should_checkpoint():
            score_head_block = auto_remat(
                score_head_block,
                **self._gradient_checkpointing_feature.get_config(),
            )

        # Create classification head with custom attribute name
        score_head = score_head_block(
            config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self._head_bias,
            kernel_init=self._head_kernel_init,
            precision=precision,
            rngs=rngs,
        )
        setattr(self, score_head_name, score_head)

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
    ) -> SequenceClassifierOutput:
        """Forward pass through the Sequence Classification model.

        Args:
            input_ids: Input token IDs, shape (batch_size, sequence_length)
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position IDs for positional embeddings
            mode: Runtime mode (train, eval, etc.)
            past_key_values: Cache containing precomputed key/value states
            cache_metadata: Metadata for cache handling
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states of all layers

        Returns:
            SequenceClassifierOutput containing classification logits, hidden states,
            attentions, and auxiliary loss if applicable

        Raises:
            ValueError: If batch size > 1 and no padding token is defined
        """
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
        )

        hidden_states = outputs.last_hidden_state

        # Apply classification head to all positions
        score_head = getattr(self, self._score_head_name)
        logits = score_head(hidden_states)

        # Determine batch size
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # Validate batch size requirements
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        # Pool logits to get sequence-level predictions
        pooled_logits = self.pool_sequence(logits, input_ids)

        # Compute router auxiliary loss if configured
        aux_loss = self.compute_router_aux_loss(outputs)

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss=aux_loss,
        )

    def get_task_head(self):
        """Returns the classification head.

        Returns:
            The score module
        """
        return getattr(self, self._score_head_name)

    def get_lm_head(self):
        """Raises NotImplementedError since this model uses a classification head.

        Raises:
            NotImplementedError: Always raised for sequence classification models
        """
        raise NotImplementedError(
            f"SequenceClassification models use a classification head ({self._score_head_name}), not an lm_head."
        )
