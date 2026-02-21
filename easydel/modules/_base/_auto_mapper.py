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

"""Auto-mapper for creating task-specific model classes dynamically.

This module provides factory functions that automatically generate ForCausalLM,
ForSequenceClassification, and other task-specific model classes from base models.

The factory functions provide an alternative to explicit subclassing for creating
task-specific wrappers. They are useful when:
    - Creating wrappers for many similar models
    - Generating classes programmatically
    - Reducing boilerplate for simple wrappers

Factory Functions:
    - create_causal_lm_class: Create ForCausalLM wrapper
    - create_sequence_classification_class: Create ForSequenceClassification wrapper
    - create_token_classification_class: Create ForTokenClassification wrapper
    - create_question_answering_class: Create ForQuestionAnswering wrapper
    - create_conditional_generation_class: Create ForConditionalGeneration wrapper
    - create_image_classification_class: Create ForImageClassification wrapper
    - create_task_model_class: Generic factory using TaskType

Example:
    Using factory functions:

    ```python
    from easydel.modules._base import create_causal_lm_class
    from easydel.modules.llama import LlamaModel, LlamaConfig

    # Create a ForCausalLM class dynamically
    LlamaForCausalLM = create_causal_lm_class(
        model_name="Llama",
        base_model_class=LlamaModel,
        config_class=LlamaConfig,
        model_type="llama",
        base_model_name="model",
    )

    # Use it like a normal class
    config = LlamaConfig(...)
    model = LlamaForCausalLM(config, dtype=jnp.bfloat16, rngs=rngs)
    ```

See Also:
    - BaseCausalLMModule: Base class for causal LM wrappers
    - BaseSequenceClassificationModule: Base class for classification
    - easydel.infra.factory.TaskType: Task type enumeration
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
"""Type variable for base model types."""

ConfigT = TypeVar("ConfigT")
"""Type variable for configuration types."""


def create_causal_lm_class(
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseCausalLMModule[ModelT, ConfigT]]:
    """Create a ForCausalLM class dynamically.

    This factory function creates a new ForCausalLM class that inherits from
    BaseCausalLMModule with the specified base model and configuration. The
    generated class is automatically registered with EasyDeL's factory system.

    Args:
        model_name: Name prefix for the generated class. The final class name
            will be "{model_name}ForCausalLM" (e.g., "Arctic" -> "ArcticForCausalLM").
        base_model_class: The base model class to wrap. Must implement
            BaseModelProtocol (have __call__, get_embedding, get_decoder methods).
        config_class: The configuration class for this model. Used for
            registration with the EasyDeL factory system.
        model_type: Model type string for registration. This is used by
            AutoModelForCausalLM to look up the correct class (e.g., "arctic",
            "llama", "mistral").
        base_model_name: Attribute name for storing the base model instance.
            Defaults to "model". Some models use different names like
            "transformer" or "bert".
        **default_feature_kwargs: Default feature flags that will be applied
            to all instances. These can be overridden at instantiation time.
            Common kwargs include:
                - tie_word_embeddings (bool): Share input/output embeddings
                - logit_cap (float): Cap logit values
                - router_aux_loss_coef (float): MoE auxiliary loss weight

    Returns:
        A new class that inherits from BaseCausalLMModule and is configured
        for the specified base model. The class is registered with EasyDeL's
        factory system.

    Example:
        ```python
        # Create wrapper class
        ArcticForCausalLM = create_causal_lm_class(
            "Arctic",
            ArcticModel,
            ArcticConfig,
            model_type="arctic",
            base_model_name="model",
            router_aux_loss_coef=0.001,  # Default for MoE
        )

        # Instantiate the model
        model = ArcticForCausalLM(
            config=config,
            dtype=jnp.bfloat16,
            rngs=rngs,
            # Can override defaults:
            router_aux_loss_coef=0.01,
        )
        ```

    Note:
        The generated class uses Python's dynamic type() function. The
        class will have proper __module__ and __qualname__ attributes
        for debugging and introspection.
    """
    class_name = f"{model_name}ForCausalLM"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        """Initialize the dynamically created ForCausalLM model.

        Args:
            config: Model configuration.
            dtype: Computation dtype. Defaults to jnp.bfloat16.
            param_dtype: Parameter dtype. Defaults to jnp.bfloat16.
            precision: JAX precision setting.
            rngs: Flax NNX random number generators.
            **kwargs: Additional kwargs merged with default_feature_kwargs.
        """
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
    """Create a ForSequenceClassification class dynamically.

    This factory function creates a new ForSequenceClassification class that
    inherits from BaseSequenceClassificationModule with the specified base
    model and configuration.

    Args:
        model_name: Name prefix for the generated class. The final class name
            will be "{model_name}ForSequenceClassification".
        base_model_class: The base model class to wrap.
        config_class: The configuration class for this model. Must have
            num_labels attribute for classification.
        model_type: Model type string for registration.
        base_model_name: Attribute name for storing the base model instance.
            Defaults to "model".
        **default_feature_kwargs: Default feature flags. Common kwargs include:
            - pooling_strategy (str): "last", "first", "mean", or "max"
            - router_aux_loss_coef (float): MoE auxiliary loss weight

    Returns:
        A new class that inherits from BaseSequenceClassificationModule and is
        configured for the specified base model.

    Example:
        ```python
        LlamaForSequenceClassification = create_sequence_classification_class(
            "Llama",
            LlamaModel,
            LlamaConfig,
            model_type="llama",
            pooling_strategy="last",
        )
        ```

    Note:
        The config_class must have a num_labels attribute set before
        instantiation, as it's required for creating the classification head.
    """
    class_name = f"{model_name}ForSequenceClassification"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        """Initialize the dynamically created ForSequenceClassification model.

        Args:
            config: Model configuration with num_labels attribute.
            dtype: Computation dtype. Defaults to jnp.bfloat16.
            param_dtype: Parameter dtype. Defaults to jnp.bfloat16.
            precision: JAX precision setting.
            rngs: Flax NNX random number generators.
            **kwargs: Additional kwargs merged with default_feature_kwargs.
        """
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
    """Create a ForTokenClassification class dynamically.

    This factory function creates a new ForTokenClassification class for
    token-level classification tasks like Named Entity Recognition (NER)
    or Part-of-Speech (POS) tagging.

    Args:
        model_name: Name prefix for the generated class. The final class name
            will be "{model_name}ForTokenClassification".
        base_model_class: The base model class to wrap.
        config_class: The configuration class for this model. Must have
            num_labels attribute.
        model_type: Model type string for registration.
        base_model_name: Attribute name for storing the base model instance.
            Defaults to "model".
        **default_feature_kwargs: Default feature flags. Common kwargs include:
            - classifier_dropout (float): Dropout rate before classifier
            - classifier_bias (bool): Whether to use bias in classifier

    Returns:
        A new class that inherits from BaseTokenClassificationModule.

    Example:
        ```python
        BertForTokenClassification = create_token_classification_class(
            "Bert",
            BertModel,
            BertConfig,
            model_type="bert",
            classifier_dropout=0.1,
        )
        ```
    """
    class_name = f"{model_name}ForTokenClassification"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        """Initialize the dynamically created ForTokenClassification model.

        Args:
            config: Model configuration with num_labels attribute.
            dtype: Computation dtype. Defaults to jnp.bfloat16.
            param_dtype: Parameter dtype. Defaults to jnp.bfloat16.
            precision: JAX precision setting.
            rngs: Flax NNX random number generators.
            **kwargs: Additional kwargs merged with default_feature_kwargs.
        """
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
    """Create a ForQuestionAnswering class dynamically.

    This factory function creates a new ForQuestionAnswering class for
    extractive question answering tasks where the model predicts start
    and end positions of answer spans within the input context.

    Args:
        model_name: Name prefix for the generated class. The final class name
            will be "{model_name}ForQuestionAnswering".
        base_model_class: The base model class to wrap.
        config_class: The configuration class for this model.
        model_type: Model type string for registration.
        base_model_name: Attribute name for storing the base model instance.
            Defaults to "model".
        **default_feature_kwargs: Default feature flags. Common kwargs include:
            - qa_head_bias (bool): Whether to use bias in QA head

    Returns:
        A new class that inherits from BaseQuestionAnsweringModule.

    Example:
        ```python
        BertForQuestionAnswering = create_question_answering_class(
            "Bert",
            BertModel,
            BertConfig,
            model_type="bert",
        )
        ```

    Note:
        The QA head outputs 2 values per token: start logits and end logits.
    """
    class_name = f"{model_name}ForQuestionAnswering"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        """Initialize the dynamically created ForQuestionAnswering model.

        Args:
            config: Model configuration.
            dtype: Computation dtype. Defaults to jnp.bfloat16.
            param_dtype: Parameter dtype. Defaults to jnp.bfloat16.
            precision: JAX precision setting.
            rngs: Flax NNX random number generators.
            **kwargs: Additional kwargs merged with default_feature_kwargs.
        """
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
    """Create a ForConditionalGeneration class dynamically.

    This factory function creates a new ForConditionalGeneration class for
    sequence-to-sequence tasks. Supports both encoder-decoder models (T5, BART)
    and vision-language models (LLaVA).

    Args:
        model_name: Name prefix for the generated class. The final class name
            will be "{model_name}ForConditionalGeneration".
        base_model_class: The base model class to wrap. Should be an
            encoder-decoder model or VLM.
        config_class: The configuration class for this model.
        model_type: Model type string for registration.
        base_model_name: Attribute name for storing the base model instance.
            Defaults to "model".
        **default_feature_kwargs: Default feature flags. Common kwargs include:
            - tie_word_embeddings (bool): Share input/output embeddings
            - logit_cap (float): Cap logit values

    Returns:
        A new class that inherits from BaseConditionalGenerationModule.

    Example:
        ```python
        T5ForConditionalGeneration = create_conditional_generation_class(
            "T5",
            T5Model,
            T5Config,
            model_type="t5",
            tie_word_embeddings=True,
        )
        ```
    """
    class_name = f"{model_name}ForConditionalGeneration"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        """Initialize the dynamically created ForConditionalGeneration model.

        Args:
            config: Model configuration.
            dtype: Computation dtype. Defaults to jnp.bfloat16.
            param_dtype: Parameter dtype. Defaults to jnp.bfloat16.
            precision: JAX precision setting.
            rngs: Flax NNX random number generators.
            **kwargs: Additional kwargs merged with default_feature_kwargs.
        """
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
    """Create a ForImageClassification class dynamically.

    This factory function creates a new ForImageClassification class for
    image classification tasks using vision models like ViT or CLIP.

    Args:
        model_name: Name prefix for the generated class. The final class name
            will be "{model_name}ForImageClassification".
        base_model_class: The base vision model class to wrap. Must accept
            pixel_values and return hidden states.
        config_class: The configuration class for this model. Must have
            num_labels attribute.
        model_type: Model type string for registration.
        base_model_name: Attribute name for storing the base model instance.
            Defaults to "vision_model" for vision-specific models.
        **default_feature_kwargs: Default feature flags. Common kwargs include:
            - pooling_strategy (str): "first" for CLS token, "mean" for average
            - classifier_bias (bool): Whether to use bias in classifier

    Returns:
        A new class that inherits from BaseImageClassificationModule.

    Example:
        ```python
        ViTForImageClassification = create_image_classification_class(
            "ViT",
            ViTModel,
            ViTConfig,
            model_type="vit",
            pooling_strategy="first",  # Use CLS token
        )
        ```

    Note:
        The default base_model_name is "vision_model" unlike text models
        which use "model".
    """
    class_name = f"{model_name}ForImageClassification"

    def __init__(self, config, dtype=None, param_dtype=None, precision=None, *, rngs, **kwargs):
        """Initialize the dynamically created ForImageClassification model.

        Args:
            config: Model configuration with num_labels attribute.
            dtype: Computation dtype. Defaults to jnp.bfloat16.
            param_dtype: Parameter dtype. Defaults to jnp.bfloat16.
            precision: JAX precision setting.
            rngs: Flax NNX random number generators.
            **kwargs: Additional kwargs merged with default_feature_kwargs.
        """
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
AUTO_MODEL_FACTORY_REGISTRY: dict[TaskType, callable] = {
    TaskType.CAUSAL_LM: create_causal_lm_class,
    TaskType.SEQUENCE_CLASSIFICATION: create_sequence_classification_class,
    # TaskType.TOKEN_CLASSIFICATION: create_token_classification_class,  # Not yet in TaskType enum
    # TaskType.QUESTION_ANSWERING: create_question_answering_class,  # Not yet in TaskType enum
    TaskType.SEQUENCE_TO_SEQUENCE: create_conditional_generation_class,
    TaskType.IMAGE_CLASSIFICATION: create_image_classification_class,
}
"""Registry mapping TaskType enum values to their corresponding factory functions.

This registry enables the generic create_task_model_class function to dispatch
to the appropriate specialized factory based on task type.

Currently supported task types:
    - TaskType.CAUSAL_LM -> create_causal_lm_class
    - TaskType.SEQUENCE_CLASSIFICATION -> create_sequence_classification_class
    - TaskType.SEQUENCE_TO_SEQUENCE -> create_conditional_generation_class
    - TaskType.IMAGE_CLASSIFICATION -> create_image_classification_class

Note:
    TOKEN_CLASSIFICATION and QUESTION_ANSWERING are not yet in the TaskType
    enum but their factory functions are available for direct use.
"""


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

    This is a generic factory function that dispatches to the appropriate
    specialized factory based on the task_type parameter. Useful when the
    task type is determined at runtime or when building generic tooling.

    Args:
        task_type: The task type from TaskType enum. Determines which
            specialized factory function to use.
        model_name: Name prefix for the generated class.
        base_model_class: The base model class to wrap.
        config_class: The configuration class for this model.
        model_type: Model type string for registration.
        base_model_name: Attribute name for storing the base model instance.
            Defaults to "model".
        **default_feature_kwargs: Default feature flags passed to the
            specialized factory function.

    Returns:
        A new task-specific model class appropriate for the given task_type.

    Raises:
        ValueError: If task_type is not supported (not in AUTO_MODEL_FACTORY_REGISTRY).

    Example:
        ```python
        # Dynamically create class based on task type
        for task_type in [TaskType.CAUSAL_LM, TaskType.SEQUENCE_CLASSIFICATION]:
            cls = create_task_model_class(
                task_type=task_type,
                model_name="MyModel",
                base_model_class=MyModel,
                config_class=MyConfig,
                model_type="my_model",
            )
            print(f"Created {cls.__name__}")
        ```

    See Also:
        - AUTO_MODEL_FACTORY_REGISTRY: Dictionary of supported task types
        - Individual factory functions for task-specific documentation
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
