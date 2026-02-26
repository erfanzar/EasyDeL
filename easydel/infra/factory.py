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

"""Factory and registry system for EasyDeL modules and configurations.

This module provides a centralized registration system for managing EasyDeL model
configurations and module implementations. It enables dynamic discovery and instantiation
of models based on task types and model identifiers.

The registry system supports:
    - Configuration class registration with type identifiers
    - Module registration with task-specific categorization
    - Metadata tracking for special layers (embeddings, layer norms)
    - Type-safe retrieval of registered components

The module exposes a global ``registry`` singleton instance and convenience functions
``register_config`` and ``register_module`` that delegate to the registry methods.

Example:
    Registering a new model configuration and module::

        from easydel.infra.factory import registry, register_config, register_module
        from easydel.infra.factory import TaskType, ConfigType

        # Register a configuration class
        @register_config("my_model")
        class MyModelConfig(EasyDeLBaseConfig):
            hidden_size: int = 768
            num_layers: int = 12

        # Register a module for causal language modeling
        @register_module(
            task_type=TaskType.CAUSAL_LM,
            config=MyModelConfig,
            model_type="my_model",
            embedding_layer_names=["embed_tokens"],
            layernorm_names=["ln_1", "ln_2"]
        )
        class MyModelForCausalLM(EasyDeLBaseModule):
            pass

        # Retrieve registered components
        config_cls = registry.get_config("my_model")
        module_reg = registry.get_module_registration(TaskType.CAUSAL_LM, "my_model")

The registry pattern enables:
    - Clean separation of model definitions from registration logic
    - Easy extension with new model types without modifying core code
    - Consistent interface for model discovery and instantiation
    - Support for multiple task-specific implementations of the same model

Attributes:
    registry (Registry): The global singleton registry instance used throughout EasyDeL.
    register_config (Callable): Module-level function for registering configuration classes.
    register_module (Callable): Module-level function for registering module classes.

See Also:
    - :class:`EasyDeLBaseConfig`: Base class for all configuration classes.
    - :class:`EasyDeLBaseModule`: Base class for all module implementations.
"""

import inspect
import typing as tp
from enum import StrEnum

from eformer.pytree import auto_pytree

from .base_config import EasyDeLBaseConfig
from .base_module import EasyDeLBaseModule

T = tp.TypeVar("T")


class ConfigType(StrEnum):
    """Enumeration of configuration types that can be registered in the registry.

    This enum defines the categories under which configuration classes can be
    organized within the registry system. Currently supports module configurations,
    but the design allows for future extension to other configuration types.

    Attributes:
        MODULE_CONFIG: Represents standard module configuration classes that define
            model architecture parameters such as hidden sizes, number of layers,
            attention heads, and other hyperparameters.

    Example:
        Using ConfigType when registering a configuration::

            from easydel.infra.factory import register_config, ConfigType

            @register_config("custom_model", config_field=ConfigType.MODULE_CONFIG)
            class CustomModelConfig(EasyDeLBaseConfig):
                hidden_size: int = 512
    """

    MODULE_CONFIG = "module-config"


class TaskType(StrEnum):
    """Enumeration of supported model task types in the EasyDeL registry.

    This enum categorizes modules by their intended use case, enabling the
    registry to organize and retrieve the appropriate module implementation
    for a given task. Each task type represents a distinct model architecture
    pattern or downstream application.

    Attributes:
        CAUSAL_LM: Causal Language Modeling for autoregressive text generation.
            Examples include GPT-style models where each token can only attend
            to previous tokens.
        VISION_LM: Vision Language Modeling for multimodal tasks combining
            image understanding with language generation.
        DIFFUSION_LM: Diffusion-based Language Modeling for iterative denoising
            approaches to text generation.
        IMAGE_TEXT_TO_TEXT: Models that process both image and text inputs to
            generate text outputs, commonly used in visual question answering
            and image captioning.
        BASE_MODULE: Base or backbone modules that provide core functionality
            without task-specific heads. Useful for transfer learning or as
            building blocks for custom architectures.
        BASE_VISION: Base vision encoder modules that process visual inputs
            without task-specific classification or generation heads.
        SEQUENCE_TO_SEQUENCE: Encoder-decoder models for tasks like translation,
            summarization, and other sequence transformation tasks.
        SPEECH_SEQUENCE_TO_SEQUENCE: Models for speech-to-text transcription
            and other audio sequence transformation tasks.
        ZERO_SHOT_IMAGE_CLASSIFICATION: Vision models capable of classifying
            images into categories not seen during training, typically using
            text-based class descriptions.
        SEQUENCE_CLASSIFICATION: Models that classify entire input sequences
            into predefined categories, such as sentiment analysis or topic
            classification.
        AUDIO_CLASSIFICATION: Models specialized for classifying audio inputs
            into categories like speech emotion recognition or sound event
            detection.
        IMAGE_CLASSIFICATION: Standard image classification models that assign
            category labels to input images.
        ANY_TO_ANY: Flexible multimodal models capable of handling various
            input and output modalities.
        AUTO_BIND: Special marker indicating automatic task type inference
            based on model architecture or configuration.

    Example:
        Registering modules for different tasks::

            from easydel.infra.factory import register_module, TaskType

            @register_module(
                task_type=TaskType.CAUSAL_LM,
                config=LlamaConfig,
                model_type="llama"
            )
            class LlamaForCausalLM(EasyDeLBaseModule):
                pass

            @register_module(
                task_type=TaskType.SEQUENCE_CLASSIFICATION,
                config=LlamaConfig,
                model_type="llama"
            )
            class LlamaForSequenceClassification(EasyDeLBaseModule):
                pass
    """

    CAUSAL_LM = "causal-language-model"
    VISION_LM = "vision-language-model"
    DIFFUSION_LM = "diffusion-language-model"
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"
    BASE_MODULE = "base-module"
    BASE_VISION = "vision-module"
    SEQUENCE_TO_SEQUENCE = "sequence-to-sequence"
    SPEECH_SEQUENCE_TO_SEQUENCE = "speech-sequence-to-sequence"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    AUDIO_CLASSIFICATION = "audio-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    ANY_TO_ANY = "any-to-any"
    AUTO_BIND = "auto-bind"


@auto_pytree
class ModuleRegistration:
    """Container for registered EasyDeL module metadata.

    This class stores comprehensive information about a registered module,
    including the module class itself, its associated configuration class,
    and optional metadata about special layer names. This metadata is useful
    for parameter transformations, weight loading, and model analysis.

    The class is decorated with ``@auto_pytree`` to enable JAX pytree
    compatibility for potential use in functional transformations.

    Attributes:
        module (type[EasyDeLBaseModule]): The registered EasyDeL module class.
            This is the actual class that can be instantiated to create
            model instances.
        config (type[EasyDeLBaseConfig]): The configuration class associated
            with the module. Used to construct configuration objects that
            parameterize the module architecture.
        embedding_layer_names (list[str] | None): Names of embedding layers
            within the module's parameter structure. These names help identify
            embedding weights for special handling during initialization,
            quantization, or transfer learning. Defaults to None.
        layernorm_names (list[str] | None): Names of Layer Normalization layers
            within the module's parameter structure. Useful for identifying
            normalization layers that may require special treatment during
            optimization or quantization. Defaults to None.

    Example:
        Creating a ModuleRegistration manually (typically done via decorator)::

            from easydel.infra.factory import ModuleRegistration

            registration = ModuleRegistration(
                module=MyModelForCausalLM,
                config=MyModelConfig,
                embedding_layer_names=["model.embed_tokens"],
                layernorm_names=["model.norm", "model.layers.*.input_layernorm"]
            )

            # Access registration information
            model_class = registration.module
            config_class = registration.config

    Note:
        The layer names support pattern matching with wildcards (``*``) to match
        multiple layers in repeated blocks, such as transformer layers.
    """

    module: type[EasyDeLBaseModule]
    config: type[EasyDeLBaseConfig]
    embedding_layer_names: list[str] | None = None
    layernorm_names: list[str] | None = None


class Registry:
    """Central registry for managing EasyDeL configurations and modules.

    This class provides a unified interface for registering and retrieving
    model configurations and module implementations. It organizes registrations
    by configuration type and task type, enabling efficient lookup and discovery
    of available models.

    The registry maintains two internal dictionaries:
        - ``_config_registry``: Maps ConfigType to dictionaries of config classes
        - ``_task_registry``: Maps TaskType to dictionaries of ModuleRegistration

    The class provides decorator factories (``register_config``, ``register_module``)
    for convenient registration during class definition.

    Attributes:
        task_registry (dict[TaskType, dict[str, ModuleRegistration]]): Read-only
            access to the task-based module registry.
        config_registry (dict[ConfigType, dict]): Read-only access to the
            configuration registry.

    Example:
        Using the registry directly::

            from easydel.infra.factory import Registry, TaskType, ConfigType

            # Create a new registry (or use the global `registry` instance)
            my_registry = Registry()

            # Register a configuration
            @my_registry.register_config("custom_model")
            class CustomConfig(EasyDeLBaseConfig):
                hidden_size: int = 256

            # Register a module
            @my_registry.register_module(
                task_type=TaskType.CAUSAL_LM,
                config=CustomConfig,
                model_type="custom_model"
            )
            class CustomModelForCausalLM(EasyDeLBaseModule):
                pass

            # Retrieve registered components
            config_cls = my_registry.get_config("custom_model")
            registration = my_registry.get_module_registration(
                TaskType.CAUSAL_LM, "custom_model"
            )

    Note:
        In most cases, you should use the global ``registry`` instance and the
        module-level ``register_config`` and ``register_module`` functions rather
        than creating your own Registry instance.
    """

    def __init__(self):
        """Initialize the registry with empty configuration and task registries.

        Creates the internal storage dictionaries with pre-initialized keys for
        all defined ConfigType and TaskType enum values. This ensures that
        lookups always find a valid dictionary, even if empty.
        """
        self._config_registry: dict[ConfigType, dict] = {ConfigType.MODULE_CONFIG: {}}

        self._task_registry: dict[TaskType, dict[str, ModuleRegistration]] = {task_type: {} for task_type in TaskType}

    def register_config(
        self,
        config_type: str,
        config_field: ConfigType = ConfigType.MODULE_CONFIG,
    ) -> tp.Callable[[T], T]:
        """Create a decorator for registering a configuration class.

        This method returns a decorator that registers a configuration class
        under the specified type identifier. The decorator also enhances the
        class with improved ``__str__`` and ``__repr__`` methods for better
        debugging and logging output.

        Args:
            config_type: A unique string identifier for the configuration class.
                This identifier is used to retrieve the configuration class later.
                Common examples include model family names like "llama", "mistral",
                "gpt2", etc.
            config_field: The category under which to register the configuration.
                Determines which sub-registry stores the configuration class.
                Defaults to ``ConfigType.MODULE_CONFIG``.

        Returns:
            A decorator function that takes a configuration class, registers it
            in the appropriate registry, enhances its string representation, and
            returns the original class unchanged (aside from the enhanced methods).

        Example:
            Basic configuration registration::

                from easydel.infra.factory import register_config

                @register_config("my_transformer")
                class MyTransformerConfig(EasyDeLBaseConfig):
                    hidden_size: int = 768
                    num_attention_heads: int = 12
                    num_hidden_layers: int = 6

                # The config is now registered and can be retrieved
                from easydel.infra.factory import registry
                config_cls = registry.get_config("my_transformer")
                config = config_cls(hidden_size=1024)
                print(config)  # Pretty-printed output

        Note:
            The enhanced ``__str__`` method introspects the ``__init__`` signature
            to display all configuration parameters and their values in a readable
            nested format.
        """

        def wrapper(obj: T) -> T:
            """Inner decorator that performs the actual registration.

            Args:
                obj: The configuration class to register.

            Returns:
                The same configuration class with enhanced string methods.
            """

            def _str(self):
                """Generate a pretty-printed string representation.

                Returns:
                    A formatted string showing the class name and all
                    configuration parameters with their values.
                """
                _stre = f"{obj.__name__}(\n"
                for key in list(inspect.signature(obj.__init__).parameters.keys()):
                    attrb = getattr(self, key, "EMT_ATTR_EPLkey")
                    if attrb != "EMT_ATTR_EPLkey":
                        if hasattr(attrb, "__str__") and not isinstance(
                            attrb,
                            str | int | float | bool | list | dict | tuple,
                        ):
                            nested_str = str(attrb).replace("\n", "\n  ")
                            _stre += f"  {key}={nested_str},\n"
                        else:
                            _stre += f"  {key}={attrb!r},\n"
                return _stre + ")"

            obj.__str__ = _str
            obj.__repr__ = lambda self: repr(_str(self))
            self._config_registry[config_field][config_type] = obj
            return obj

        return wrapper

    def register_module(
        self,
        task_type: TaskType,
        config: type[EasyDeLBaseConfig],
        model_type: str,
        embedding_layer_names: list[str] | None = None,
        layernorm_names: list[str] | None = None,
    ) -> tp.Callable[[T], T]:
        """Create a decorator for registering an EasyDeL module class.

        This method returns a decorator that registers a module class for a
        specific task type. The registration includes the module class, its
        associated configuration, and optional metadata about layer names.

        The decorator also sets internal class attributes (``_model_task`` and
        ``_model_type``) and enhances the ``__str__`` and ``__repr__`` methods
        using the ``printify_nnx`` utility.

        Args:
            task_type: The task category for this module. Determines which
                task registry stores the module registration. Use values from
                the ``TaskType`` enum (e.g., ``TaskType.CAUSAL_LM``).
            config: The configuration class associated with this module.
                This should be a subclass of ``EasyDeLBaseConfig`` that defines
                the architecture parameters for the module.
            model_type: A unique string identifier for this model implementation.
                This identifier, combined with task_type, uniquely identifies
                the module registration. Common examples: "llama", "mistral".
            embedding_layer_names: Names of embedding layers in the module's
                parameter tree. Used for identifying embedding weights during
                operations like quantization or weight tying. Supports glob-style
                patterns with ``*`` for matching layer indices. Defaults to None.
            layernorm_names: Names of LayerNorm layers in the module's parameter
                tree. Used for identifying normalization layers that may need
                special handling during optimization. Supports glob-style patterns.
                Defaults to None.

        Returns:
            A decorator function that takes a module class, creates a
            ``ModuleRegistration`` entry, stores it in the task registry, and
            returns the original class with enhanced attributes and methods.

        Example:
            Registering a causal language model::

                from easydel.infra.factory import register_module, TaskType

                @register_module(
                    task_type=TaskType.CAUSAL_LM,
                    config=LlamaConfig,
                    model_type="llama",
                    embedding_layer_names=["model.embed_tokens"],
                    layernorm_names=[
                        "model.norm",
                        "model.layers.*.input_layernorm",
                        "model.layers.*.post_attention_layernorm"
                    ]
                )
                class LlamaForCausalLM(EasyDeLBaseModule):
                    def __call__(self, input_ids, attention_mask=None):
                        # Model implementation
                        pass

        Note:
            The same model_type can be registered multiple times under different
            task_types. For example, "llama" can have registrations for both
            ``CAUSAL_LM`` and ``SEQUENCE_CLASSIFICATION``.
        """

        def wrapper(module: T) -> T:
            """Inner decorator that performs the actual registration.

            Args:
                module: The module class to register.

            Returns:
                The same module class with enhanced attributes and methods.
            """
            from .mixins.protocol import printify_nnx

            module.__str__ = printify_nnx
            module.__repr__ = printify_nnx
            module._model_task = task_type
            module._model_type = model_type
            self._task_registry[task_type][model_type] = ModuleRegistration(
                module=module,
                config=config,
                embedding_layer_names=embedding_layer_names,
                layernorm_names=layernorm_names,
            )
            return module

        return wrapper

    def get_config(
        self,
        config_type: str,
        config_field: ConfigType = ConfigType.MODULE_CONFIG,
    ) -> type[EasyDeLBaseConfig]:
        """Retrieve a registered configuration class by its type identifier.

        Looks up and returns a previously registered configuration class from
        the appropriate category registry.

        Args:
            config_type: The identifier of the configuration class to retrieve.
                This should match the identifier used during registration.
                Examples: "llama", "mistral", "gpt2".
            config_field: The category of the configuration to look up.
                Determines which sub-registry to search. Defaults to
                ``ConfigType.MODULE_CONFIG``.

        Returns:
            The registered configuration class that can be instantiated to
            create configuration objects.

        Raises:
            KeyError: If the ``config_type`` is not found in the specified
                ``config_field`` registry. This indicates either the configuration
                was never registered or it was registered under a different name.

        Example:
            Retrieving and using a registered configuration::

                from easydel.infra.factory import registry

                # Get the LLaMA configuration class
                LlamaConfig = registry.get_config("llama")

                # Create a configuration instance
                config = LlamaConfig(
                    hidden_size=4096,
                    num_attention_heads=32,
                    num_hidden_layers=32
                )

            Handling missing configurations::

                try:
                    config_cls = registry.get_config("nonexistent_model")
                except KeyError:
                    print("Configuration not registered")
        """
        return self._config_registry[config_field][config_type]

    def get_module_registration(
        self,
        task_type: TaskType
        | tp.Literal[
            "causal-language-model",
            "sequence-classification",
            "vision-language-model",
            "audio-classification",
            "base-module",
            "sequence-to-sequence",
        ],
        model_type: str,
    ) -> ModuleRegistration:
        """Retrieve the registration information for a specific module.

        Looks up and returns the complete ``ModuleRegistration`` for a module
        identified by its task type and model type. The registration contains
        the module class, configuration class, and layer name metadata.

        Args:
            task_type: The task category to search within. Can be a ``TaskType``
                enum value or its corresponding string literal (e.g.,
                ``TaskType.CAUSAL_LM`` or ``"causal-language-model"``).
            model_type: The identifier of the model to retrieve. This should
                match the identifier used during registration. Examples:
                "llama", "mistral", "gpt2".

        Returns:
            A ``ModuleRegistration`` object containing:
                - ``module``: The registered module class
                - ``config``: The associated configuration class
                - ``embedding_layer_names``: List of embedding layer names (or None)
                - ``layernorm_names``: List of LayerNorm layer names (or None)

        Raises:
            AssertionError: If the ``task_type`` is not a valid task category
                in the registry, or if the ``model_type`` is not registered
                under the specified task type.

        Example:
            Retrieving and using a module registration::

                from easydel.infra.factory import registry, TaskType

                # Get the registration for LLaMA causal LM
                registration = registry.get_module_registration(
                    TaskType.CAUSAL_LM,
                    "llama"
                )

                # Access the module class
                LlamaForCausalLM = registration.module

                # Access the configuration class
                LlamaConfig = registration.config

                # Access layer metadata
                embedding_names = registration.embedding_layer_names
                layernorm_names = registration.layernorm_names

                # Instantiate the model
                config = LlamaConfig(hidden_size=4096)
                model = LlamaForCausalLM(config=config, ...)

            Using string literal task type::

                registration = registry.get_module_registration(
                    "causal-language-model",  # String literal alternative
                    "llama"
                )
        """
        task_in = self._task_registry.get(task_type, None)
        assert task_in is not None, f"task type {task_type} is not defined."
        type_in = task_in.get(model_type, None)
        assert type_in is not None, f"model type {model_type} is not defined. (upper task {task_type})"

        return type_in

    @property
    def task_registry(self) -> dict[TaskType, dict[str, ModuleRegistration]]:
        """Provide read-only access to the task-based module registry.

        Returns the internal dictionary mapping task types to their registered
        modules. This can be used for introspection, listing available models,
        or implementing custom lookup logic.

        Returns:
            A dictionary where keys are ``TaskType`` enum values and values are
            dictionaries mapping model type strings to ``ModuleRegistration``
            objects.

        Example:
            Listing all registered causal LM models::

                from easydel.infra.factory import registry, TaskType

                causal_lm_models = registry.task_registry[TaskType.CAUSAL_LM]
                for model_type, registration in causal_lm_models.items():
                    print(f"{model_type}: {registration.module.__name__}")
        """
        return self._task_registry

    @property
    def config_registry(self) -> dict[ConfigType, dict]:
        """Provide read-only access to the configuration registry.

        Returns the internal dictionary mapping configuration types to their
        registered configuration classes. Useful for introspection and
        listing available configurations.

        Returns:
            A dictionary where keys are ``ConfigType`` enum values and values
            are dictionaries mapping config type strings to configuration
            classes.

        Example:
            Listing all registered module configurations::

                from easydel.infra.factory import registry, ConfigType

                module_configs = registry.config_registry[ConfigType.MODULE_CONFIG]
                for config_type, config_cls in module_configs.items():
                    print(f"{config_type}: {config_cls.__name__}")
        """
        return self._config_registry


# Global registry instance used throughout EasyDeL for managing model registrations.
# This singleton pattern ensures all modules register to the same central registry.
registry = Registry()

# Module-level convenience functions that delegate to the global registry instance.
# These allow direct decoration without importing the registry object.
register_config = registry.register_config
register_module = registry.register_module
