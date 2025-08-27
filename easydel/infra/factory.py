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

import inspect
import typing as tp
from enum import Enum

from eformer.pytree import auto_pytree

from .base_module import EasyDeLBaseConfig, EasyDeLBaseModule

T = tp.TypeVar("T")


class ConfigType(str, Enum):
    """
    Enumeration defining types of configurations that can be registered.

    Attributes:
        MODULE_CONFIG: Represents standard module configuration classes.
    """

    MODULE_CONFIG = "module-config"


class TaskType(str, Enum):
    """
    Enumeration defining different model task types supported by the registry.

    Attributes:
        CAUSAL_LM: Causal Language Modeling (e.g., GPT-style models).
        VISION_LM: Vision Language Modeling (models combining vision and text).
        DIFFUSION_LM: Diffusion Language Modeling
        IMAGE_TEXT_TO_TEXT: Models that take image and text input to produce text output.
        BASE_MODULE: Basic, potentially abstract, modules.
        BASE_VISION: Basic vision modules.
        SEQUENCE_TO_SEQUENCE: Sequence-to-sequence tasks (e.g., translation, summarization).
        SPEECH_SEQUENCE_TO_SEQUENCE: Speech-to-text or other speech sequence tasks.
        ZERO_SHOT_IMAGE_CLASSIFICATION: Image classification without task-specific training.
        SEQUENCE_CLASSIFICATION: Classifying entire sequences (e.g., sentiment analysis).
        AUDIO_CLASSIFICATION: Classifying audio data.
        IMAGE_CLASSIFICATION: Classifying images.
                    AUTO_BIND: Whenever to automatically decide what todo.
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
    AUTO_BIND = "auto-bind"


@auto_pytree
class ModuleRegistration:
    """
    A container class holding information about a registered EasyDeL module.

    This class stores the module class itself, its corresponding configuration class,
    and optional metadata like the names of embedding and LayerNorm layers, which
    can be useful for parameter transformation or analysis.

    Attributes:
        module (type[EasyDeLBaseModule]): The class of the registered EasyDeL module.
        config (type[EasyDeLBaseConfig]): The configuration class associated with the module.
        embedding_layer_names (tp.Optional[tp.List[str]]): A list of names identifying embedding layers
            within the module structure. Defaults to None.
        layernorm_names (tp.Optional[tp.List[str]]): A list of names identifying Layer Normalization layers
            within the module structure. Defaults to None.
    """

    module: type[EasyDeLBaseModule]
    config: type[EasyDeLBaseConfig]
    embedding_layer_names: list[str] | None = None
    layernorm_names: list[str] | None = None


class Registry:
    """
    A central registry for managing EasyDeL configurations and modules.

    This class provides decorators (`register_config`, `register_module`) to easily
    add new configurations and module implementations. It organizes registrations
    by configuration type and task type, allowing for retrieval based on identifiers.
    """

    def __init__(self):
        """Initializes the registry dictionaries."""
        self._config_registry: dict[ConfigType, dict] = {ConfigType.MODULE_CONFIG: {}}

        self._task_registry: dict[TaskType, dict[str, ModuleRegistration]] = {task_type: {} for task_type in TaskType}

    def register_config(
        self,
        config_type: str,
        config_field: ConfigType = ConfigType.MODULE_CONFIG,
    ) -> callable:
        """
        Decorator factory to register a configuration class.

        Args:
            config_type (str): A unique string identifier for this configuration class (e.g., "llama").
            config_field (ConfigType): The category under which to register the config.
                Defaults to `ConfigType.MODULE_CONFIG`.

        Returns:
            callable: A decorator that takes the configuration class, registers it,
                and enhances its string representation.
        """

        def wrapper(obj: T) -> T:
            # Enhance the __str__ and __repr__ for better readability
            def _str(self):
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
        config: EasyDeLBaseConfig,
        model_type: str,
        embedding_layer_names: list[str] | None = None,
        layernorm_names: list[str] | None = None,
    ) -> callable:
        """
        Decorator factory to register an EasyDeL module class for a specific task.

        Args:
            task_type (TaskType): The task the module is designed for (e.g., `TaskType.CAUSAL_LM`).
            config (EasyDeLBaseConfig): The configuration class associated with this module.
            model_type (str): A unique string identifier for this model implementation (e.g., "llama").
            embedding_layer_names (tp.Optional[tp.List[str]]): Optional list of embedding layer names.
                Defaults to None.
            layernorm_names (tp.Optional[tp.List[str]]): Optional list of LayerNorm layer names.
                Defaults to None.

        Returns:
            callable: A decorator that takes the module class, registers it with its metadata,
                and sets internal `_model_task` and `_model_type` attributes on the class.
        """

        def wrapper(module: T) -> T:
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
    ) -> type:
        """
        Retrieves a registered configuration class by its type identifier.

        Args:
            config_type (str): The identifier of the configuration class (e.g., "llama").
            config_field (ConfigType): The category of the configuration.
                Defaults to `ConfigType.MODULE_CONFIG`.

        Returns:
            tp.Type: The registered configuration class.

        Raises:
            KeyError: If the `config_type` is not found in the specified `config_field` registry.
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
        """
        Retrieves the `ModuleRegistration` information for a given task and model type.

        Args:
            task_type (TaskType | Literal): The task type (enum or string literal).
            model_type (str): The identifier of the model type (e.g., "llama").

        Returns:
            ModuleRegistration: The registration information containing the module class, config class,
                and optional metadata.

        Raises:
            AssertionError: If the `task_type` or `model_type` is not found in the registry.
        """
        task_in = self._task_registry.get(task_type, None)
        assert task_in is not None, f"task type {task_type} is not defined."
        type_in = task_in.get(model_type, None)
        assert type_in is not None, f"model type {model_type} is not defined. (upper task {task_type})"

        return type_in

    @property
    def task_registry(self):
        """Provides access to the underlying task registry dictionary."""
        return self._task_registry

    @property
    def config_registry(self):
        """Provides access to the underlying configuration registry dictionary."""
        return self._config_registry


# Global registry instance
registry = Registry()

# Expose registration methods as module-level functions
register_config = registry.register_config
register_module = registry.register_module
