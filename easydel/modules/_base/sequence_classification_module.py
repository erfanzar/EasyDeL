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

"""Generic base class for Sequence Classification tasks.

This module provides BaseSequenceClassificationModule, a generic, type-safe base
class for creating ForSequenceClassification model wrappers with minimal boilerplate.

Sequence classification models predict a single label for an entire input sequence,
making them suitable for tasks like sentiment analysis, topic classification,
natural language inference, and text similarity scoring.

Key Features:
    - Generic typing with ModelT and ConfigT type parameters
    - Configurable pooling strategies (last, first, mean, max)
    - Optional MoE router auxiliary loss computation
    - Gradient checkpointing support for memory efficiency
    - Customizable classification head configuration

Example:
    Creating a sequence classification model::

        from easydel.modules._base import BaseSequenceClassificationModule

        class MyModelForSequenceClassification(
            BaseSequenceClassificationModule[MyModel, MyConfig]
        ):
            _task_type = TaskType.SEQUENCE_CLASSIFICATION
            _model_type = "my_model"
            _config_class = MyConfig

            def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                super().__init__(
                    config=config,
                    base_model_class=MyModel,
                    dtype=dtype,
                    rngs=rngs,
                    pooling_strategy="last",
                )

See Also:
    - BaseTaskModule: Parent class with common task functionality
    - BaseCausalLMModule: For causal language modeling tasks
    - BaseTokenClassificationModule: For token-level classification tasks
"""

from collections.abc import Callable

import jax
from eformer import common_types
from ejkernel.types import MaskInfo
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
from easydel.infra.modeling_outputs import SequenceClassifierOutput
from easydel.infra.utils import auto_remat
from easydel.layers import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseSequenceClassificationModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Sequence Classification.

    This class provides a fully-featured, type-safe base for creating
    ForSequenceClassification model wrappers with support for:

    - Generic typing (ModelT, ConfigT) for type safety
    - Automatic model registration via class attributes
    - Modular features (pooling strategies, router aux loss)
    - Gradient checkpointing for memory-efficient training
    - Configurable classification head with customizable name and parameters

    The sequence classification task involves predicting a single label for
    an entire input sequence. This is achieved by pooling the hidden states
    (using configurable strategies like "last", "first", "mean", or "max")
    and passing the pooled representation through a classification head.

    Example:
        Basic usage with a custom model::

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

        Using the model for inference::

            model = ArcticForSequenceClassification(config, rngs=rngs)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, num_labels)

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol).
            This is the underlying transformer model that produces hidden states.
        ConfigT: The configuration type containing model hyperparameters.
            Must have `num_labels` and `hidden_size` attributes.

    Attributes:
        score: The classification head (linear projection from hidden_size to num_labels).
            The actual attribute name can be customized via `score_head_name` parameter.
        _score_head_name (str): The name of the classification head attribute.
        base_model: The underlying transformer model that produces hidden states.

    Note:
        The configuration must have a `num_labels` attribute specifying the number
        of output classes. This is validated during initialization.
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
        # Backward-compatible aliases used by older model wrappers
        classifier_name: str | None = None,
        classifier_bias: bool | None = None,
        classifier_kernel_init: Callable | None = None,
    ):
        """Initialize the Sequence Classification module.

        Creates a sequence classification model by wrapping a base transformer model
        and adding a classification head. The model uses pooling to reduce the
        sequence of hidden states to a single vector before classification.

        Args:
            config: Model configuration object. Must have the following attributes:
                - num_labels (int): Number of output classes
                - hidden_size (int): Dimension of hidden states
                - pad_token_id (int, optional): ID of padding token
                - gradient_checkpointing (str, optional): Checkpointing policy
            base_model: Pre-instantiated base model instance. If provided,
                base_model_class is ignored. Useful for sharing weights.
            base_model_class: Base model class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name for storing the base model.
                Common values are "model", "transformer", "encoder".
                Defaults to "model".
            dtype: Data type for computations (activations). Defaults to bfloat16.
            param_dtype: Data type for parameters (weights). Defaults to bfloat16.
            precision: JAX precision setting for matrix multiplications.
                Can be None, "high", "highest", or a Precision enum value.
            rngs: Flax random number generators for initialization.
            pooling_strategy: Strategy for pooling sequence representations.
                - "last": Use the last non-padding token (default)
                - "first": Use the first token (CLS token style)
                - "mean": Average all non-padding tokens
                - "max": Max-pool over all non-padding tokens
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss.
                Only used for Mixture-of-Experts models. None disables aux loss.
            score_head_name: Attribute name for the classification head.
                Defaults to "score". Common alternatives: "classifier", "output".
            score_head_bias: Whether to include bias in the classification head.
                Defaults to False for better stability.
            score_head_kernel_init: Custom initializer for classification head weights.
                If None, uses the default Flax initializer.

        Raises:
            AssertionError: If config does not have a `num_labels` attribute.
                This is required for creating the classification head.

        Example:
            Creating a classification model with mean pooling::

                model = BaseSequenceClassificationModule(
                    config=my_config,
                    base_model_class=MyTransformerModel,
                    dtype=jnp.float32,
                    rngs=nn.Rngs(0),
                    pooling_strategy="mean",
                    score_head_bias=True,
                )
        """
        # Validate config has num_labels
        assert hasattr(config, "num_labels"), (
            "in order to use `SequenceClassification` Models in `EasyDeL` "
            "you first need to attach `num_labels` to model `config`"
        )

        if classifier_name is not None:
            score_head_name = classifier_name
        if classifier_bias is not None:
            score_head_bias = classifier_bias
        if classifier_kernel_init is not None:
            score_head_kernel_init = classifier_kernel_init

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

        Processes input through the base transformer model, pools the sequence
        representations according to the configured strategy, and applies the
        classification head to produce logits for each class.

        Args:
            input_ids: Input token IDs with shape (batch_size, sequence_length).
                Each value should be a valid token ID from the vocabulary.
                Either input_ids or inputs_embeds must be provided.
            inputs_embeds: Pre-computed input embeddings with shape
                (batch_size, sequence_length, hidden_dim). Alternative to input_ids
                for passing custom embeddings directly.
            attention_mask: Binary mask with shape (batch_size, sequence_length).
                1 for tokens to attend to, 0 for padding tokens. Used for both
                attention computation and pooling.
            mask_info: Structured mask information for advanced attention patterns.
                Alternative to attention_mask for complex masking scenarios.
            position_ids: Position indices with shape (batch_size, sequence_length).
                If None, positions are inferred from input_ids.
            mode: Runtime mode controlling model behavior. Options include
                training mode, evaluation mode, and various inference modes.
            past_key_values: Cached key/value states from previous forward passes.
                Used for efficient autoregressive generation.
            cache_metadata: Metadata for cache management including sequence
                lengths and cache indices.
            output_attentions: If True, include attention weights in the output.
                Defaults to model config setting if None.
            output_hidden_states: If True, include hidden states from all layers
                in the output. Defaults to model config setting if None.

        Returns:
            SequenceClassifierOutput: A dataclass containing:
                - logits: Classification logits with shape (batch_size, num_labels)
                - past_key_values: Updated KV cache (if caching is enabled)
                - hidden_states: Tuple of hidden states from all layers (if requested)
                - attentions: Tuple of attention weights from all layers (if requested)
                - aux_loss: Router auxiliary loss for MoE models (if applicable)

        Raises:
            ValueError: If batch size > 1 and no padding token is defined in config,
                and no attention_mask is provided. This is required to correctly
                identify the last token position for pooling.

        Example:
            Basic inference::

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                predictions = jnp.argmax(outputs.logits, axis=-1)

            Getting all hidden states::

                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )
                # outputs.hidden_states is a tuple of arrays
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
        if self.config.pad_token_id is None and attention_mask is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        # Pool logits to get sequence-level predictions
        pooled_logits = self.pool_sequence(logits, input_ids=input_ids, attention_mask=attention_mask)

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
        """Returns the classification head module.

        Retrieves the linear layer used for sequence classification. The head
        projects from hidden_size to num_labels dimensions.

        Returns:
            nn.Module: The classification head module (typically ColumnParallelLinear).
                The actual attribute name depends on the score_head_name parameter
                used during initialization.

        Example:
            Accessing the classification head weights::

                head = model.get_task_head()
                weights = head.kernel.value  # Shape: (hidden_size, num_labels)
        """
        return getattr(self, self._score_head_name)

    def get_lm_head(self):
        """Raises NotImplementedError since this model uses a classification head.

        Sequence classification models do not have a language modeling head.
        They use a classification head instead, which can be accessed via
        get_task_head().

        Raises:
            NotImplementedError: Always raised for sequence classification models
                with a message indicating the correct method to use.

        See Also:
            get_task_head: Use this method to access the classification head.
        """
        raise NotImplementedError(
            f"SequenceClassification models use a classification head ({self._score_head_name}), not an lm_head."
        )
