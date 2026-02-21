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

"""Abstract base class for all task-specific modules.

This module provides the foundation for all task-specific model wrappers,
including auto-registration, feature management, and common interfaces.

The BaseTaskModule class serves as the abstract base for creating specialized
model wrappers for different NLP and ML tasks such as:
    - Causal Language Modeling (BaseCausalLMModule)
    - Sequence Classification (BaseSequenceClassificationModule)
    - Token Classification (BaseTokenClassificationModule)
    - Question Answering (BaseQuestionAnsweringModule)
    - Conditional Generation (BaseConditionalGenerationModule)
    - Image Classification (BaseImageClassificationModule)
    - Vision-Language Modeling (BaseVisionLanguageModule)

Key Features:
    - Generic typing support with ModelT and ConfigT type parameters
    - Automatic registration with EasyDeL's factory system
    - Modular feature system for logit capping, embedding tying, etc.
    - Configurable task heads with gradient checkpointing support
    - Consistent interface across all task-specific modules

Example:
    Creating a custom task module:

    ```python
    class MyModelForCausalLM(BaseTaskModule[MyModel, MyConfig]):
        _task_type = TaskType.CAUSAL_LM
        _model_type = "my_model"
        _config_class = MyConfig

        def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
            super().__init__(
                config=config,
                base_model_class=MyModel,
                dtype=dtype,
                rngs=rngs,
            )

        def __call__(self, input_ids, **kwargs):
            # Implementation
            pass

        def get_task_head(self):
            return self.lm_head
    ```

See Also:
    - easydel.infra.base_module.EasyDeLBaseModule: Parent class
    - easydel.infra.factory: Registration and TaskType definitions
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import jax
from flax import nnx as nn
from jax import numpy as jnp

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module

from ._features import (
    GradientCheckpointingFeature,
    LogitCapFeature,
    RouterAuxLossFeature,
    SequenceLengthPoolingFeature,
    TieEmbeddingsFeature,
)
from ._protocols import BaseModelProtocol

# Type variables for generic base models and configs
ModelT = TypeVar("ModelT", bound=BaseModelProtocol)
"""Type variable for base model types that implement BaseModelProtocol."""

ConfigT = TypeVar("ConfigT", bound=EasyDeLBaseConfig)
"""Type variable for configuration types that extend EasyDeLBaseConfig."""


class BaseTaskModule(EasyDeLBaseModule, Generic[ModelT, ConfigT], ABC):
    """Abstract base class for all task-specific modules.

    This class provides the foundation for creating task-specific model wrappers
    (e.g., ForCausalLM, ForSequenceClassification) with automatic registration,
    feature management, and consistent interfaces.

    The class uses Python's Generic typing system to provide type safety when
    working with different base model and configuration types. Subclasses should
    specify their concrete types:

    ```python
    class MyForCausalLM(BaseTaskModule[MyModel, MyConfig]):
        ...
    ```

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol).
            This is the underlying transformer model that will be wrapped.
        ConfigT: The configuration type (must extend EasyDeLBaseConfig).
            This contains all hyperparameters and settings for the model.

    Attributes:
        config (ConfigT): Model configuration containing hyperparameters.
        dtype (jnp.dtype): Data type for computations (e.g., jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for model parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.

    Class Attributes:
        _task_type (TaskType | None): The TaskType this module implements.
            Set in subclasses to enable auto-registration (e.g., TaskType.CAUSAL_LM).
        _auto_register (bool): Whether to auto-register this module with the
            EasyDeL factory system. Defaults to True.
        _model_type (str | None): Model type string for registration
            (e.g., "mistral", "arctic", "llama"). Must be set for registration.
        _config_class (type | None): The configuration class for this model.
            Must be set for registration.

    Note:
        Subclasses must implement:
            - __call__: Forward pass through the model
            - get_task_head: Return the task-specific head (e.g., lm_head)
    """

    # Class variables for registration (to be set by subclasses)
    _task_type: TaskType | None = None
    _auto_register: bool = True
    _model_type: str | None = None
    _config_class: type | None = None

    def __init_subclass__(cls, **kwargs):
        """Handle automatic registration when subclassed.

        This method is called automatically by Python when a class inherits
        from BaseTaskModule. It registers the module with the EasyDeL factory
        system if auto-registration is enabled and all required class attributes
        are set.

        The registration allows models to be loaded via AutoModel classes:
        ```python
        model = AutoModelForCausalLM.from_pretrained("model_name")
        ```

        Args:
            **kwargs: Additional keyword arguments passed to parent __init_subclass__.

        Note:
            Registration only occurs when all of:
                - _auto_register is True
                - _task_type is set
                - _model_type is set
                - _config_class is set
        """
        super().__init_subclass__(**kwargs)

        # Auto-register if enabled and task type is specified
        if cls._auto_register and cls._task_type is not None:
            # Only register if this is a concrete subclass with a model type
            if cls._model_type is not None and cls._config_class is not None:
                # Apply registration decorator
                register_module(
                    task_type=cls._task_type,
                    config=cls._config_class,
                    model_type=cls._model_type,
                )(cls)

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
        pooling_strategy: str = "last",
        # Head configuration
        head_bias: bool = False,
        head_kernel_init: Callable | None = None,
    ):
        """Initialize the base task module.

        This constructor sets up the base model, configures features like logit
        capping and embedding tying, and prepares the module for task-specific
        head creation in subclasses.

        Args:
            config: Model configuration containing hyperparameters such as
                hidden_size, vocab_size, num_layers, etc.
            base_model: Pre-instantiated base model instance. If provided,
                base_model_class is ignored. Useful when sharing a base model
                across multiple task heads.
            base_model_class: Base model class to instantiate. Required if
                base_model is not provided. The class will be instantiated
                with the same dtype, param_dtype, precision, and rngs.
            base_model_name: Attribute name under which to store the base model.
                Common values include "model", "transformer", "bert", etc.
                Defaults to "model".
            dtype: Data type for computations. Affects activations and
                intermediate values. Defaults to jnp.bfloat16.
            param_dtype: Data type for model parameters (weights). Can differ
                from dtype for mixed-precision training. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX matrix operations. Can be
                None (default), jax.lax.Precision.DEFAULT, HIGH, or HIGHEST.
            rngs: Flax NNX random number generators for initialization and
                dropout. Must include at least 'params' key.
            tie_word_embeddings: Whether to share weights between the input
                embedding layer and the output projection (LM head). Reduces
                parameters and can improve performance. Defaults to False.
            logit_cap: Maximum absolute value for output logits. If set,
                logits are clipped to [-logit_cap, logit_cap]. Useful for
                stabilizing training. Defaults to None (no capping).
            router_aux_loss_coef: Coefficient for Mixture-of-Experts router
                auxiliary loss. Encourages balanced expert utilization.
                Defaults to None (no aux loss). Common values: 0.001-0.1.
            pooling_strategy: Strategy for reducing sequence to single vector.
                Options: "last" (last non-pad token), "first" (first token/CLS),
                "mean" (average all tokens), "max" (max pooling).
                Defaults to "last".
            head_bias: Whether to include bias in the task-specific head
                (e.g., lm_head, classifier). Defaults to False.
            head_kernel_init: Custom kernel initializer for the task head.
                If None, uses normal distribution with stddev from
                config.initializer_range (default 0.02).

        Raises:
            ValueError: If neither base_model nor base_model_class is provided.

        Example:
            ```python
            class MyForCausalLM(BaseTaskModule[MyModel, MyConfig]):
                def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                    super().__init__(
                        config=config,
                        base_model_class=MyModel,
                        base_model_name="model",
                        dtype=dtype,
                        rngs=rngs,
                        tie_word_embeddings=config.tie_word_embeddings,
                        logit_cap=30.0,  # Cap logits at +/- 30
                    )
                    # Create task head...
            ```
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Create or store base model
        if base_model is None:
            if base_model_class is None:
                raise ValueError("Either base_model or base_model_class must be provided")
            base_model = base_model_class(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

        # Store base model with custom attribute name
        setattr(self, base_model_name, base_model)
        self._base_model_name = base_model_name

        # Store head configuration
        self._head_bias = head_bias
        self._head_kernel_init = head_kernel_init or jax.nn.initializers.normal(
            stddev=getattr(config, "initializer_range", 0.02)
        )

        self._logit_cap_feature = LogitCapFeature(logit_cap) if logit_cap is not None else None
        self._tie_embeddings_feature = TieEmbeddingsFeature(tie_word_embeddings)
        self._router_aux_loss_feature = (
            RouterAuxLossFeature(router_aux_loss_coef) if router_aux_loss_coef is not None else None
        )
        pad_token_id = getattr(config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = -1
        self._pooling_feature = SequenceLengthPoolingFeature(strategy=pooling_strategy, pad_token_id=pad_token_id)
        self._gradient_checkpointing_feature = GradientCheckpointingFeature(
            policy=getattr(config, "gradient_checkpointing", None),
            save_names=getattr(config, "gradient_checkpointing_targets", None),
            exclude_names=getattr(config, "gradient_checkpointing_targets", None),
        )

    @property
    def base_model(self) -> ModelT:
        """Access the base model via the configured attribute name.

        This property provides a consistent way to access the underlying
        transformer model regardless of what attribute name was used
        during initialization (e.g., "model", "transformer", "bert").

        Returns:
            ModelT: The base model instance that was either passed to
                __init__ or instantiated from base_model_class.

        Example:
            ```python
            # Access base model for direct operations
            hidden_states = module.base_model(input_ids)

            # Get embedding layer
            embeddings = module.base_model.get_embedding()
            ```
        """
        return getattr(self, self._base_model_name)

    def apply_logit_cap(self, logits):
        """Apply logit capping if configured.

        Logit capping clips output logits to a specified range to prevent
        extreme values that can cause numerical instability during training
        or lead to overconfident predictions.

        Args:
            logits: Input logits tensor of any shape, typically
                (batch_size, sequence_length, vocab_size) for language models.

        Returns:
            Logits tensor with values clipped to [-cap, cap] if logit capping
            is enabled, otherwise returns the input unchanged.

        Example:
            ```python
            # In forward pass
            raw_logits = self.lm_head(hidden_states)
            capped_logits = self.apply_logit_cap(raw_logits)
            # capped_logits are now in range [-logit_cap, logit_cap]
            ```

        Note:
            This method is a no-op if logit_cap was not set during initialization.
        """
        if self._logit_cap_feature is not None:
            return self._logit_cap_feature.apply(logits)
        return logits

    def compute_router_aux_loss(self, outputs) -> Any:
        """Compute router auxiliary loss for MoE models if configured.

        For Mixture-of-Experts (MoE) models, this method computes an auxiliary
        loss that encourages balanced utilization across experts. This prevents
        the router from collapsing to always select the same few experts.

        Args:
            outputs: Model outputs object that may contain `all_router_losses`
                attribute with per-layer router loss values.

        Returns:
            The weighted sum of router losses multiplied by router_aux_loss_coef,
            or None if MoE auxiliary loss is not configured or no router losses
            are available in the outputs.

        Example:
            ```python
            outputs = self.base_model(input_ids)
            aux_loss = self.compute_router_aux_loss(outputs)
            if aux_loss is not None:
                total_loss = lm_loss + aux_loss
            ```

        Note:
            This method is a no-op if router_aux_loss_coef was not set
            during initialization or if the model is not an MoE model.
        """
        if self._router_aux_loss_feature is None:
            return None

        router_losses = getattr(outputs, "all_router_losses", None)
        if router_losses is None:
            return None

        return self._router_aux_loss_feature.compute_loss(router_losses)

    def pool_sequence(self, hidden_states, input_ids=None, attention_mask=None):
        """Pool sequence of hidden states to a single vector.

        Reduces a sequence of hidden state vectors to a single vector
        representation, which is required for sequence-level tasks like
        classification.

        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_dim)
                containing the hidden states for each token position.
            input_ids: Optional tensor of shape (batch_size, sequence_length)
                containing input token IDs. Required for "last" pooling strategy
                when attention_mask is not provided, to find non-padding positions.
            attention_mask: Optional tensor of shape (batch_size, sequence_length)
                indicating valid (1) vs padding (0) positions. Preferred over
                input_ids for finding sequence lengths.

        Returns:
            Tensor of shape (batch_size, hidden_dim) containing the pooled
            representation for each sequence in the batch.

        Raises:
            ValueError: If using "last" pooling strategy without input_ids
                or attention_mask.

        Example:
            ```python
            # Pool for classification
            hidden_states = outputs.last_hidden_state
            pooled = self.pool_sequence(hidden_states, input_ids=input_ids)
            logits = self.classifier(pooled)
            ```

        Note:
            The pooling strategy is configured during initialization via
            the pooling_strategy parameter.
        """
        return self._pooling_feature.pool(hidden_states, input_ids, attention_mask=attention_mask)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Forward pass through the module.

        This method must be implemented by subclasses to define the specific
        task logic (e.g., language modeling, classification, etc.).

        The signature varies by task type but typically includes:
            - input_ids: Input token IDs
            - attention_mask: Attention mask
            - position_ids: Position IDs
            - past_key_values: KV cache for generation
            - output_attentions: Whether to return attention weights
            - output_hidden_states: Whether to return all hidden states

        Returns:
            A task-specific output dataclass (e.g., CausalLMOutput,
            SequenceClassifierOutput, etc.).

        Raises:
            NotImplementedError: Always raised in base class. Subclasses
                must provide implementation.
        """
        raise NotImplementedError

    def get_encoder(self):
        """Return the encoder part of the model's architecture.

        For encoder-decoder models (like T5, BART), this returns the encoder
        component. For decoder-only models (like GPT, LLaMA), this raises
        NotImplementedError.

        Returns:
            The encoder module for encoder-decoder architectures.

        Raises:
            NotImplementedError: For decoder-only models that don't have
                a separate encoder component.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Return the decoder part of the model's architecture.

        For both encoder-decoder and decoder-only models, this returns the
        decoder component that generates output sequences.

        Returns:
            The decoder module from the base model.
        """
        return self.base_model.get_decoder()

    @abstractmethod
    def get_task_head(self):
        """Return the task-specific output head.

        Each task type has a specific head that transforms hidden states
        into task-appropriate outputs:
            - CausalLM: lm_head (projects to vocabulary)
            - Classification: classifier/score (projects to num_labels)
            - QA: qa_outputs (projects to start/end logits)

        Returns:
            The task-specific head module (typically a Linear layer).

        Raises:
            NotImplementedError: Always raised in base class. Subclasses
                must provide implementation.

        Example:
            ```python
            # In CausalLM
            def get_task_head(self):
                return self.lm_head

            # In SequenceClassification
            def get_task_head(self):
                return self.score
            ```
        """
        raise NotImplementedError

    def get_embedding(self):
        """Return the input embedding layer of the model.

        The embedding layer converts token IDs to dense vectors. This is
        needed for operations like weight tying (sharing embeddings with
        the output projection).

        Returns:
            The embedding module from the base model, typically containing
            an embedding matrix of shape (vocab_size, hidden_size).

        Example:
            ```python
            embedding = module.get_embedding()
            if tie_embeddings:
                lm_head.kernel = embedding.embedding.T
            ```
        """
        return self.base_model.get_embedding()
