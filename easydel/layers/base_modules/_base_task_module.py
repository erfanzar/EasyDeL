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

"""Abstract base class for all task-specific modules.

This module provides the foundation for all task-specific model wrappers,
including auto-registration, feature management, and common interfaces.
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
ConfigT = TypeVar("ConfigT", bound=EasyDeLBaseConfig)


class BaseTaskModule(EasyDeLBaseModule, Generic[ModelT, ConfigT], ABC):
    """Abstract base class for all task-specific modules.

    This class provides the foundation for creating task-specific model wrappers
    (e.g., ForCausalLM, ForSequenceClassification) with automatic registration,
    feature management, and consistent interfaces.

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol)
        ConfigT: The configuration type

    Attributes:
        base_model: The underlying model instance
        config: Model configuration
        dtype: Data type for computations
        param_dtype: Data type for parameters
        precision: Precision setting for JAX operations

    Class Attributes:
        _task_type: The TaskType this module implements (set in subclasses)
        _auto_register: Whether to auto-register this module (default: True)
        _model_type: Model type string for registration (e.g., "mistral", "arctic")
    """

    # Class variables for registration (to be set by subclasses)
    _task_type: TaskType | None = None
    _auto_register: bool = True
    _model_type: str | None = None
    _config_class: type | None = None

    def __init_subclass__(cls, **kwargs):
        """Automatic registration when subclassed.

        This method is called when a class inherits from BaseTaskModule.
        It automatically registers the module with the EasyDeL factory system
        if _auto_register is True and _task_type is set.
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

        Args:
            config: Model configuration
            base_model: Pre-instantiated base model (optional)
            base_model_class: Base model class to instantiate (optional, used if base_model is None)
            base_model_name: Attribute name for the base model (e.g., "model", "transformer")
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
            tie_word_embeddings: Whether to tie input embeddings with output head
            logit_cap: Maximum absolute value for logits (None = no capping)
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss (None = no aux loss)
            pooling_strategy: Strategy for pooling sequences ("last", "first", "mean", "max")
            head_bias: Whether to use bias in the task head
            head_kernel_init: Custom kernel initializer for the task head
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
        self._pooling_feature = SequenceLengthPoolingFeature(
            strategy=pooling_strategy, pad_token_id=getattr(config, "pad_token_id", None) or -1
        )
        self._gradient_checkpointing_feature = GradientCheckpointingFeature(
            policy=getattr(config, "gradient_checkpointing", None),
            save_names=getattr(config, "gradient_checkpointing_targets", None),
            exclude_names=getattr(config, "gradient_checkpointing_targets", None),
        )

    @property
    def base_model(self) -> ModelT:
        """Access the base model via the configured attribute name.

        Returns:
            The base model instance
        """
        return getattr(self, self._base_model_name)

    def apply_logit_cap(self, logits):
        """Apply logit capping if configured.

        Args:
            logits: Input logits tensor

        Returns:
            Logits tensor (capped if feature is enabled)
        """
        if self._logit_cap_feature is not None:
            return self._logit_cap_feature.apply(logits)
        return logits

    def compute_router_aux_loss(self, outputs) -> Any:
        """Compute router auxiliary loss if configured.

        Args:
            outputs: Model outputs that may contain router_logits

        Returns:
            Auxiliary loss value, or None if not configured
        """
        if self._router_aux_loss_feature is None:
            return None

        router_losses = getattr(outputs, "all_router_losses", None)
        if router_losses is None:
            return None

        return self._router_aux_loss_feature.compute_loss(router_losses)

    def pool_sequence(self, hidden_states, input_ids=None):
        """Pool sequence of hidden states to single vector.

        Args:
            hidden_states: Sequence of hidden states
            input_ids: Input token IDs (required for some pooling strategies)

        Returns:
            Pooled representation
        """
        return self._pooling_feature.pool(hidden_states, input_ids)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Forward pass through the module.

        This method must be implemented by subclasses to define the specific
        task logic (e.g., language modeling, classification, etc.).
        """
        raise NotImplementedError

    def get_encoder(self):
        """Returns the encoder part of the model's graph definition.

        For decoder-only models, this raises NotImplementedError.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Returns the decoder part of the model's graph definition."""
        return self.base_model.get_decoder()

    @abstractmethod
    def get_task_head(self):
        """Returns the task-specific head (e.g., lm_head, score).

        This method must be implemented by subclasses to return their
        specific task head.
        """
        raise NotImplementedError

    def get_embedding(self):
        """Returns the embedding layer of the module."""
        return self.base_model.get_embedding()
