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

This module provides BaseQuestionAnsweringModule for extractive QA tasks
where the model predicts start and end positions of answer spans.
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
from easydel.layers.linear import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseQuestionAnsweringModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Question Answering (extractive QA).

    This class provides span-based question answering with start and end
    position predictions.

    Example:
        ```python
        class RobertaForQuestionAnswering(
            BaseQuestionAnsweringModule[RobertaModel, RobertaConfig]
        ):
            _task_type = TaskType.QUESTION_ANSWERING
            _model_type = "roberta"
            _config_class = RobertaConfig
        ```

    Type Parameters:
        ModelT: The base model type
        ConfigT: The configuration type

    Attributes:
        qa_outputs: Linear layer that outputs start and end logits
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

        Args:
            config: Model configuration
            base_model: Pre-instantiated base model (optional)
            base_model_class: Base model class to instantiate (optional)
            base_model_name: Attribute name for the base model
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
            qa_head_bias: Whether to use bias in QA head
            qa_head_kernel_init: Custom kernel initializer
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

        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Attention mask
            position_ids: Position IDs
            mode: Runtime mode
            past_key_values: KV cache
            cache_metadata: Cache metadata
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states

        Returns:
            QuestionAnsweringModelOutput with start and end logits
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
        """Returns the QA head."""
        return self.qa_outputs

    def get_lm_head(self):
        """Raises NotImplementedError."""
        raise NotImplementedError("Question answering models don't have an lm_head.")
