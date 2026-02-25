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

"""Generic base class for Token Classification tasks.

This module provides BaseTokenClassificationModule for tasks like Named Entity
Recognition (NER), Part-of-Speech (POS) tagging, and other token-level
classification tasks where each token is classified independently.

Unlike sequence classification which produces a single label for the entire
sequence, token classification produces a label for each token in the input,
making it suitable for structured prediction tasks.

Key Features:
    - Generic typing with ModelT and ConfigT type parameters
    - Optional dropout before the classification head
    - Gradient checkpointing support for memory efficiency
    - Customizable classifier configuration

Example:
    Creating a token classification model::

        from easydel.modules._base import BaseTokenClassificationModule

        class MyModelForTokenClassification(
            BaseTokenClassificationModule[MyModel, MyConfig]
        ):
            _task_type = TaskType.TOKEN_CLASSIFICATION
            _model_type = "my_model"
            _config_class = MyConfig

See Also:
    - BaseTaskModule: Parent class with common task functionality
    - BaseSequenceClassificationModule: For sequence-level classification
    - BaseQuestionAnsweringModule: For span extraction tasks
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
from easydel.infra.modeling_outputs import TokenClassifierOutput
from easydel.infra.utils import auto_remat
from easydel.layers import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseTokenClassificationModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Token Classification.

    This class provides token-level classification capabilities suitable for
    tasks like Named Entity Recognition (NER), Part-of-Speech (POS) tagging,
    chunking, and other sequence labeling tasks.

    Each token in the input sequence receives its own classification, allowing
    the model to identify and label spans of text with different categories.

    Key capabilities:
        - Generic typing (ModelT, ConfigT) for type safety
        - Optional dropout regularization before classification
        - Gradient checkpointing for memory-efficient training
        - Support for various cache types (TransformerCache, RaggedPagesCache, etc.)

    Example:
        Basic usage with a custom model::

            class RobertaForTokenClassification(
                BaseTokenClassificationModule[RobertaModel, RobertaConfig]
            ):
                _task_type = TaskType.TOKEN_CLASSIFICATION
                _model_type = "roberta"
                _config_class = RobertaConfig

        Using the model for NER::

            # config.num_labels = 9  # e.g., B-PER, I-PER, B-ORG, I-ORG, etc.
            model = RobertaForTokenClassification(config, rngs=rngs)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = jnp.argmax(outputs.logits, axis=-1)
            # predictions shape: (batch_size, sequence_length)

    Type Parameters:
        ModelT: The base model type that produces hidden state representations.
            Must implement the BaseModelProtocol interface.
        ConfigT: The configuration type containing model hyperparameters.
            Must have `num_labels` and `hidden_size` attributes.

    Attributes:
        classifier: The token classification head that projects hidden states
            to label logits. Shape: (hidden_size, num_labels).
        dropout: Optional dropout layer applied before the classifier.
            Only present if classifier_dropout is specified during init.
        base_model: The underlying transformer model.

    Note:
        Unlike sequence classification, token classification does not use
        pooling since each token needs its own prediction.
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
        classifier_dropout: float | None = None,
        classifier_bias: bool = True,
        classifier_kernel_init: Callable | None = None,
    ):
        """Initialize the Token Classification module.

        Creates a token classification model by wrapping a base transformer model
        and adding a classification head that predicts labels for each token.

        Args:
            config: Model configuration object. Must have the following attributes:
                - num_labels (int): Number of token classes (e.g., NER tags)
                - hidden_size (int): Dimension of hidden states
                - gradient_checkpointing (str, optional): Checkpointing policy
            base_model: Pre-instantiated base model instance. If provided,
                base_model_class is ignored. Useful for sharing weights across
                multiple task heads.
            base_model_class: Base model class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name for storing the base model.
                Common values are "model", "transformer", "encoder".
                Defaults to "model".
            dtype: Data type for computations (activations). Using bfloat16
                provides good balance of speed and precision.
            param_dtype: Data type for parameters (weights). Defaults to bfloat16.
            precision: JAX precision setting for matrix multiplications.
                Higher precision may improve numerical stability at the cost
                of speed.
            rngs: Flax random number generators for parameter initialization
                and dropout.
            classifier_dropout: Dropout probability applied to hidden states
                before the classifier. If None or 0, no dropout is applied.
                Common values: 0.1 for regularization during training.
            classifier_bias: Whether to include bias in the classifier linear
                layer. Defaults to True for standard classification behavior.
            classifier_kernel_init: Custom initializer for classifier weights.
                If None, uses the default Flax initializer.

        Raises:
            AssertionError: If config does not have a `num_labels` attribute.
                This is required for creating the classification head.

        Example:
            Creating a NER model with dropout::

                config.num_labels = 9  # BIO tags for 4 entity types + O
                model = BaseTokenClassificationModule(
                    config=config,
                    base_model_class=BertModel,
                    dtype=jnp.float32,
                    rngs=nn.Rngs(0),
                    classifier_dropout=0.1,
                )
        """
        assert hasattr(config, "num_labels"), "config must have num_labels attribute"

        super().__init__(
            config=config,
            base_model=base_model,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            head_bias=classifier_bias,
            head_kernel_init=classifier_kernel_init,
        )

        # Optional dropout before classifier
        self.dropout = nn.Dropout(rate=classifier_dropout, rngs=rngs) if classifier_dropout is not None else None

        # Create classifier head
        classifier_block = ColumnParallelLinear
        if self._gradient_checkpointing_feature.should_checkpoint():
            classifier_block = auto_remat(
                classifier_block,
                **self._gradient_checkpointing_feature.get_config(),
            )

        self.classifier = classifier_block(
            config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self._head_bias,
            kernel_init=self._head_kernel_init,
            precision=precision,
            rngs=rngs,
        )

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
    ) -> TokenClassifierOutput:
        """Forward pass for token classification.

        Processes input through the base transformer model and applies the
        classification head to each token position independently.

        Args:
            input_ids: Input token IDs with shape (batch_size, sequence_length).
                Each value should be a valid token ID from the vocabulary.
                Either input_ids or inputs_embeds must be provided.
            inputs_embeds: Pre-computed input embeddings with shape
                (batch_size, sequence_length, hidden_dim). Alternative to input_ids
                for passing custom embeddings directly.
            attention_mask: Binary mask with shape (batch_size, sequence_length).
                1 for tokens to attend to, 0 for padding tokens. Note that
                predictions are still made for padding tokens; they should be
                masked during loss computation.
            mask_info: Structured mask information for advanced attention patterns.
                Alternative to attention_mask for complex masking scenarios.
            position_ids: Position indices with shape (batch_size, sequence_length).
                If None, positions are inferred sequentially from input_ids.
            mode: Runtime mode controlling model behavior (training, evaluation,
                or various inference modes). Affects dropout behavior.
            past_key_values: Cached key/value states from previous forward passes.
                Primarily used for efficient autoregressive generation, though
                less common for token classification tasks.
            cache_metadata: Metadata for cache management including sequence
                lengths and cache indices.
            output_attentions: If True, include attention weights in the output.
                Useful for interpretability and debugging.
            output_hidden_states: If True, include hidden states from all layers
                in the output. Defaults to model config setting if None.

        Returns:
            TokenClassifierOutput: A dataclass containing:
                - logits: Per-token classification logits with shape
                    (batch_size, sequence_length, num_labels)
                - hidden_states: Tuple of hidden states from all layers (if requested)
                - attentions: Tuple of attention weights from all layers (if requested)

        Example:
            Basic NER inference::

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Get predicted labels for each token
                predictions = jnp.argmax(outputs.logits, axis=-1)
                # Shape: (batch_size, sequence_length)

            Computing loss (excluding padding)::

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Only compute loss on non-padding tokens
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = outputs.logits.reshape(-1, num_labels)[active_loss]
                active_labels = labels.reshape(-1)[active_loss]
                loss = cross_entropy(active_logits, active_labels)
        """
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

        # Apply dropout if configured
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        # Apply classifier to each token
        logits = self.classifier(hidden_states)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_task_head(self):
        """Returns the token classification head module.

        Retrieves the linear layer used for per-token classification. The head
        projects from hidden_size to num_labels dimensions for each token.

        Returns:
            nn.Module: The classifier module (typically ColumnParallelLinear).
                Shape of kernel: (hidden_size, num_labels).

        Example:
            Accessing classifier weights::

                head = model.get_task_head()
                weights = head.kernel.value  # Shape: (hidden_size, num_labels)
        """
        return self.classifier

    def get_lm_head(self):
        """Raises NotImplementedError as token classification has no LM head.

        Token classification models use a classification head for per-token
        predictions, not a language modeling head for vocabulary predictions.

        Raises:
            NotImplementedError: Always raised with a message indicating that
                token classification models use get_task_head() instead.

        See Also:
            get_task_head: Use this method to access the classifier.
        """
        raise NotImplementedError("Token classification models don't have an lm_head.")
