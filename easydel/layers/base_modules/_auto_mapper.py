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

"""Auto-mapper for creating task-specific model classes dynamically.

This module provides factory functions that automatically generate ForCausalLM,
ForSequenceClassification, and other task-specific model classes from base models.
"""

from typing import Any, TypeVar

from easydel.infra.factory import TaskType

from .causal_lm_module import BaseCausalLMModule
from .conditional_generation_module import BaseConditionalGenerationModule
from .image_classification_module import BaseImageClassificationModule
from .question_answering_module import BaseQuestionAnsweringModule
from .sequence_classification_module import BaseSequenceClassificationModule
from .token_classification_module import BaseTokenClassificationModule

ModelT = TypeVar("ModelT")
ConfigT = TypeVar("ConfigT")


def create_causal_lm_class(
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseCausalLMModule[ModelT, ConfigT]]:
    """Automatically create a ForCausalLM class.

    This factory function creates a new ForCausalLM class that inherits from
    BaseCausalLMModule with the specified base model and configuration.

    Args:
        model_name: Name prefix for the class (e.g., "Arctic" -> "ArcticForCausalLM")
        base_model_class: The base model class to wrap
        config_class: The configuration class
        model_type: Model type string for registration (e.g., "arctic")
        base_model_name: Attribute name for the base model (default: "model")
        **default_feature_kwargs: Default feature flags (e.g., router_aux_loss_coef=0.001)

    Returns:
        A new ForCausalLM class

    Example:
        ```python
        ArcticForCausalLM = create_causal_lm_class(
            "Arctic",
            ArcticModel,
            ArcticConfig,
            model_type="arctic",
            base_model_name="model",
            router_aux_loss_coef=0.001,
        )
        ```
    """
    class_name = f"{model_name}ForCausalLM"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        # Merge default kwargs with instance kwargs (instance overrides defaults)
        merged_kwargs = {**default_feature_kwargs, **kwargs}

        # Set default dtype if not provided
        import jax.numpy as jnp

        dtype = dtype or jnp.bfloat16
        param_dtype = param_dtype or jnp.bfloat16

        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **merged_kwargs,
        )

    # Create class dynamically with proper metadata
    cls = type(
        class_name,
        (BaseCausalLMModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.CAUSAL_LM,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )

    return cls


def create_sequence_classification_class(
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseSequenceClassificationModule[ModelT, ConfigT]]:
    """Automatically create a ForSequenceClassification class.

    Args:
        model_name: Name prefix for the class
        base_model_class: The base model class to wrap
        config_class: The configuration class
        model_type: Model type string for registration
        base_model_name: Attribute name for the base model
        **default_feature_kwargs: Default feature flags

    Returns:
        A new ForSequenceClassification class
    """
    class_name = f"{model_name}ForSequenceClassification"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        merged_kwargs = {**default_feature_kwargs, **kwargs}

        import jax.numpy as jnp

        dtype = dtype or jnp.bfloat16
        param_dtype = param_dtype or jnp.bfloat16

        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **merged_kwargs,
        )

    cls = type(
        class_name,
        (BaseSequenceClassificationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.SEQUENCE_CLASSIFICATION,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )

    return cls


def create_token_classification_class(
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseTokenClassificationModule[ModelT, ConfigT]]:
    """Automatically create a ForTokenClassification class.

    Args:
        model_name: Name prefix for the class
        base_model_class: The base model class to wrap
        config_class: The configuration class
        model_type: Model type string for registration
        base_model_name: Attribute name for the base model
        **default_feature_kwargs: Default feature flags

    Returns:
        A new ForTokenClassification class
    """
    class_name = f"{model_name}ForTokenClassification"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        merged_kwargs = {**default_feature_kwargs, **kwargs}

        import jax.numpy as jnp

        dtype = dtype or jnp.bfloat16
        param_dtype = param_dtype or jnp.bfloat16

        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **merged_kwargs,
        )

    cls = type(
        class_name,
        (BaseTokenClassificationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.TOKEN_CLASSIFICATION,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )

    return cls


def create_question_answering_class(
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseQuestionAnsweringModule[ModelT, ConfigT]]:
    """Automatically create a ForQuestionAnswering class.

    Args:
        model_name: Name prefix for the class
        base_model_class: The base model class to wrap
        config_class: The configuration class
        model_type: Model type string for registration
        base_model_name: Attribute name for the base model
        **default_feature_kwargs: Default feature flags

    Returns:
        A new ForQuestionAnswering class
    """
    class_name = f"{model_name}ForQuestionAnswering"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        merged_kwargs = {**default_feature_kwargs, **kwargs}

        import jax.numpy as jnp

        dtype = dtype or jnp.bfloat16
        param_dtype = param_dtype or jnp.bfloat16

        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **merged_kwargs,
        )

    cls = type(
        class_name,
        (BaseQuestionAnsweringModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.QUESTION_ANSWERING,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )

    return cls


def create_conditional_generation_class(
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseConditionalGenerationModule[ModelT, ConfigT]]:
    """Automatically create a ForConditionalGeneration class.

    Args:
        model_name: Name prefix for the class
        base_model_class: The base model class to wrap
        config_class: The configuration class
        model_type: Model type string for registration
        base_model_name: Attribute name for the base model
        **default_feature_kwargs: Default feature flags

    Returns:
        A new ForConditionalGeneration class
    """
    class_name = f"{model_name}ForConditionalGeneration"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        merged_kwargs = {**default_feature_kwargs, **kwargs}

        import jax.numpy as jnp

        dtype = dtype or jnp.bfloat16
        param_dtype = param_dtype or jnp.bfloat16

        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **merged_kwargs,
        )

    cls = type(
        class_name,
        (BaseConditionalGenerationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.SEQ_2_SEQ_LM,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )

    return cls


def create_image_classification_class(
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "vision_model",
    **default_feature_kwargs: Any,
) -> type[BaseImageClassificationModule[ModelT, ConfigT]]:
    """Automatically create a ForImageClassification class.

    Args:
        model_name: Name prefix for the class
        base_model_class: The base vision model class to wrap
        config_class: The configuration class
        model_type: Model type string for registration
        base_model_name: Attribute name for the base model (default: "vision_model")
        **default_feature_kwargs: Default feature flags

    Returns:
        A new ForImageClassification class
    """
    class_name = f"{model_name}ForImageClassification"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        merged_kwargs = {**default_feature_kwargs, **kwargs}

        import jax.numpy as jnp

        dtype = dtype or jnp.bfloat16
        param_dtype = param_dtype or jnp.bfloat16

        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **merged_kwargs,
        )

    cls = type(
        class_name,
        (BaseImageClassificationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.IMAGE_CLASSIFICATION,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )

    return cls


# Registry mapping task types to factory functions
AUTO_MODEL_FACTORY_REGISTRY = {
    TaskType.CAUSAL_LM: create_causal_lm_class,
    TaskType.SEQUENCE_CLASSIFICATION: create_sequence_classification_class,
    # TaskType.TOKEN_CLASSIFICATION: create_token_classification_class,  # Not yet in TaskType enum
    # TaskType.QUESTION_ANSWERING: create_question_answering_class,  # Not yet in TaskType enum
    TaskType.SEQUENCE_TO_SEQUENCE: create_conditional_generation_class,
    TaskType.IMAGE_CLASSIFICATION: create_image_classification_class,
}


def create_task_model_class(
    task_type: TaskType,
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
):
    """Create a task-specific model class using the appropriate factory.

    Args:
        task_type: The task type (e.g., TaskType.CAUSAL_LM)
        model_name: Name prefix for the class
        base_model_class: The base model class to wrap
        config_class: The configuration class
        model_type: Model type string for registration
        base_model_name: Attribute name for the base model
        **default_feature_kwargs: Default feature flags

    Returns:
        A new task-specific model class

    Raises:
        ValueError: If task_type is not supported
    """
    if task_type not in AUTO_MODEL_FACTORY_REGISTRY:
        raise ValueError(f"Unsupported task type: {task_type}. Supported: {list(AUTO_MODEL_FACTORY_REGISTRY.keys())}")

    factory_fn = AUTO_MODEL_FACTORY_REGISTRY[task_type]
    return factory_fn(
        model_name=model_name,
        base_model_class=base_model_class,
        config_class=config_class,
        model_type=model_type,
        base_model_name=base_model_name,
        **default_feature_kwargs,
    )
