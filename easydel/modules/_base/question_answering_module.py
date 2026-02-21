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

"""Generic base class for Question Answering tasks.

This module provides BaseQuestionAnsweringModule for extractive Question Answering
(QA) tasks where the model predicts start and end positions of answer spans
within a given context.

Extractive QA is the task of finding the answer to a question within a given
passage of text. The model learns to identify the start and end positions of
the answer span, making it suitable for tasks like SQuAD, Natural Questions,
and similar reading comprehension benchmarks.

Key Features:
    - Generic typing with ModelT and ConfigT type parameters
    - Span prediction via start and end position logits
    - Gradient checkpointing support for memory efficiency
    - Customizable QA head configuration

Example:
    Creating a question answering model::

        from easydel.layers.base_modules import BaseQuestionAnsweringModule

        class BertForQuestionAnswering(
            BaseQuestionAnsweringModule[BertModel, BertConfig]
        ):
            _task_type = TaskType.QUESTION_ANSWERING
            _model_type = "bert"
            _config_class = BertConfig

See Also:
    - BaseTaskModule: Parent class with common task functionality
    - BaseTokenClassificationModule: For token-level classification (NER, POS)
    - BaseSequenceClassificationModule: For sequence-level classification
"""

from collections.abc import Callable

import jax
from eformer import common_types
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.modeling_outputs import QuestionAnsweringModelOutput
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


class BaseQuestionAnsweringModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Question Answering (extractive QA).

    This class provides span-based question answering capabilities where the
    model predicts the start and end positions of answer spans within a context.

    The model outputs two sets of logits:
        - start_logits: Probability of each token being the start of the answer
        - end_logits: Probability of each token being the end of the answer

    The answer span is typically extracted by finding positions (i, j) where
    i <= j that maximize start_logits[i] + end_logits[j].

    Example:
        Basic usage with a custom model::

            class RobertaForQuestionAnswering(
                BaseQuestionAnsweringModule[RobertaModel, RobertaConfig]
            ):
                _task_type = TaskType.QUESTION_ANSWERING
                _model_type = "roberta"
                _config_class = RobertaConfig

        Using the model for extractive QA::

            # Input: [CLS] question [SEP] context [SEP]
            model = RobertaForQuestionAnswering(config, rngs=rngs)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get most likely answer span
            start_idx = jnp.argmax(outputs.start_logits, axis=-1)
            end_idx = jnp.argmax(outputs.end_logits, axis=-1)
            answer_tokens = input_ids[0, start_idx:end_idx+1]

    Type Parameters:
        ModelT: The base model type that produces hidden state representations.
            Must implement the BaseModelProtocol interface.
        ConfigT: The configuration type containing model hyperparameters.
            Must have `hidden_size` attribute.

    Attributes:
        qa_outputs: Linear layer that outputs start and end logits.
            Projects from hidden_size to 2 (start logit, end logit).
        base_model: The underlying transformer model.

    Note:
        Unlike classification tasks, QA outputs are per-token predictions
        (like token classification) but with exactly 2 outputs per token
        representing start and end probabilities.
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
        qa_head_bias: bool = True,
        qa_head_kernel_init: Callable | None = None,
    ):
        """Initialize the Question Answering module.

        Creates an extractive QA model by wrapping a base transformer model
        and adding a QA head that predicts start and end positions.

        Args:
            config: Model configuration object. Must have the following attributes:
                - hidden_size (int): Dimension of hidden states
                - gradient_checkpointing (str, optional): Checkpointing policy
            base_model: Pre-instantiated base model instance. If provided,
                base_model_class is ignored. Useful for sharing weights.
            base_model_class: Base model class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name for storing the base model.
                Common values: "model", "transformer", "encoder".
                Defaults to "model".
            dtype: Data type for computations (activations). Defaults to bfloat16.
            param_dtype: Data type for parameters (weights). Defaults to bfloat16.
            precision: JAX precision setting for matrix multiplications.
                Higher precision may improve numerical stability.
            rngs: Flax random number generators for initialization.
            qa_head_bias: Whether to include bias in the QA output layer.
                Defaults to True for standard QA behavior.
            qa_head_kernel_init: Custom initializer for QA head weights.
                If None, uses the default Flax initializer.

        Example:
            Creating a QA model::

                config = BertConfig.from_pretrained("bert-base-uncased")
                model = BaseQuestionAnsweringModule(
                    config=config,
                    base_model_class=BertModel,
                    dtype=jnp.float32,
                    rngs=nn.Rngs(0),
                    qa_head_bias=True,
                )

        Note:
            The QA head outputs 2 values per token (start and end logits),
            so the output shape is (batch_size, sequence_length, 2).
            These are then split into separate start_logits and end_logits
            tensors of shape (batch_size, sequence_length).
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
            head_bias=qa_head_bias,
            head_kernel_init=qa_head_kernel_init,
        )

        # Create QA outputs head (outputs 2 logits per token: start and end)
        qa_outputs_block = ColumnParallelLinear
        if self._gradient_checkpointing_feature.should_checkpoint():
            qa_outputs_block = auto_remat(
                qa_outputs_block,
                **self._gradient_checkpointing_feature.get_config(),
            )

        self.qa_outputs = qa_outputs_block(
            config.hidden_size,
            2,  # start and end logits
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
    ) -> QuestionAnsweringModelOutput:
        """Forward pass for question answering.

        Processes input through the base transformer model and applies the
        QA head to predict start and end positions for answer spans.

        The input is typically formatted as: [CLS] question [SEP] context [SEP]
        where the model learns to identify the answer span within the context.

        Args:
            input_ids: Input token IDs with shape (batch_size, sequence_length).
                Typically contains question and context separated by special tokens.
                Either input_ids or inputs_embeds must be provided.
            inputs_embeds: Pre-computed input embeddings with shape
                (batch_size, sequence_length, hidden_dim). Alternative to input_ids.
            attention_mask: Binary mask with shape (batch_size, sequence_length).
                1 for tokens to attend to, 0 for padding. Important for masking
                out padding tokens when computing answer spans.
            mask_info: Structured mask information for advanced attention patterns.
            position_ids: Position indices for positional embeddings.
                If None, positions are inferred from input_ids.
            mode: Runtime mode controlling model behavior (training, evaluation,
                or various inference modes).
            past_key_values: Cached key/value states. Less commonly used for QA
                since it's typically not autoregressive.
            cache_metadata: Metadata for cache management.
            output_attentions: If True, include attention weights in output.
                Useful for visualizing which tokens the model focuses on.
            output_hidden_states: If True, include hidden states from all layers.

        Returns:
            QuestionAnsweringModelOutput: A dataclass containing:
                - start_logits: Start position logits with shape
                    (batch_size, sequence_length). Higher values indicate
                    higher probability of being the answer start.
                - end_logits: End position logits with shape
                    (batch_size, sequence_length). Higher values indicate
                    higher probability of being the answer end.
                - hidden_states: Tuple of hidden states from all layers (if requested)
                - attentions: Tuple of attention weights from all layers (if requested)

        Example:
            Basic extractive QA::

                # Prepare input: [CLS] Who is the president? [SEP] Joe Biden is... [SEP]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Simple answer extraction
                start_idx = jnp.argmax(outputs.start_logits[0])
                end_idx = jnp.argmax(outputs.end_logits[0])

                # Better: find best (start, end) pair where start <= end
                # and both are within the context (not question)

            Computing QA loss::

                outputs = model(input_ids=input_ids)
                start_loss = cross_entropy(outputs.start_logits, start_positions)
                end_loss = cross_entropy(outputs.end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

        Note:
            For multi-span or no-answer scenarios, additional post-processing
            is typically needed. Some models add a null answer score by
            using the [CLS] token's logits.
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

        # Get start and end logits
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = jnp.split(logits, 2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return QuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_task_head(self):
        """Returns the QA output head module.

        Retrieves the linear layer that produces start and end logits.
        The head projects from hidden_size to 2 dimensions.

        Returns:
            nn.Module: The QA outputs module (typically ColumnParallelLinear).
                Shape of kernel: (hidden_size, 2).

        Example:
            Accessing QA head weights::

                head = model.get_task_head()
                weights = head.kernel.value  # Shape: (hidden_size, 2)
        """
        return self.qa_outputs

    def get_lm_head(self):
        """Raises NotImplementedError as QA models have no LM head.

        Question answering models use a span prediction head, not a language
        modeling head that predicts vocabulary tokens.

        Raises:
            NotImplementedError: Always raised with a message indicating that
                QA models use get_task_head() to access the qa_outputs head.

        See Also:
            get_task_head: Use this method to access the QA head.
        """
        raise NotImplementedError("Question answering models don't have an lm_head.")
