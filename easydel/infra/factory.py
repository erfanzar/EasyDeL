# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from easydel.utils import traversals as etr

from .base_module import (
	EasyDeLBaseConfig,
	EasyDeLBaseModule,
)

T = tp.TypeVar("T")


class ConfigType(str, Enum):
	MODULE_CONFIG = "module-config"


class TaskType(str, Enum):
	CAUSAL_LM = "causal-language-model"
	VISION_LM = "vision-language-model"
	IMAGE_TEXT_TO_TEXT = "image-text-to-text"
	BASE_MODULE = "base-module"
	BASE_VISION = "vision-module"
	SEQUENCE_TO_SEQUENCE = "sequence-to-sequence"
	SPEECH_SEQUENCE_TO_SEQUENCE = "speech-sequence-to-sequence"
	ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
	SEQUENCE_CLASSIFICATION = "sequence-classification"
	AUDIO_CLASSIFICATION = "audio-classification"
	IMAGE_CLASSIFICATION = "image-classification"


@etr.auto_pytree
class ModuleRegistration:
	module: type[EasyDeLBaseModule]
	config: type[EasyDeLBaseConfig]
	embedding_layer_names: tp.Optional[tp.List[str]] = None
	layernorm_names: tp.Optional[tp.List[str]] = None


class Registry:
	def __init__(self):
		self._config_registry: tp.Dict[ConfigType, tp.Dict] = {ConfigType.MODULE_CONFIG: {}}

		self._task_registry: tp.Dict[TaskType, tp.Dict[str, ModuleRegistration]] = {
			task_type: {} for task_type in TaskType
		}

	def register_config(
		self,
		config_type: str,
		config_field: ConfigType = ConfigType.MODULE_CONFIG,
	) -> callable:
		"""
		Register a configuration class.

		Args:
		    config_type: Identifier for the configuration
		    config_field: Type of configuration registry

		Returns:
		    Decorator function
		"""

		def wrapper(obj: T) -> T:
			def _str(self):
				_stre = f"{obj.__name__}(\n"
				for key in list(inspect.signature(obj.__init__).parameters.keys()):
					attrb = getattr(self, key, "EMT_ATTR_EPLkey")
					if attrb != "EMT_ATTR_EPLkey":
						if hasattr(attrb, "__str__") and not isinstance(
							attrb,
							(str, int, float, bool, list, dict, tuple),
						):
							nested_str = str(attrb).replace("\n", "\n  ")
							_stre += f"  {key}={nested_str},\n"
						else:
							_stre += f"  {key}={repr(attrb)},\n"
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
		embedding_layer_names: tp.Optional[tp.List[str]] = None,
		layernorm_names: tp.Optional[tp.List[str]] = None,
	) -> callable:
		"""
		Register a module for a specific task.

		Args:
		    task_type: Type of task
		    config: Configuration for the module
		    model_type: Identifier for the model
		    embedding_layer_names: Names of embedding layers
		    layernorm_names: Names of layer normalization layers

		Returns:
		    Decorator function
		"""

		def wrapper(module: T) -> T:
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
	) -> tp.Type:
		"""Get registered configuration class."""
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
		task_in = self._task_registry.get(task_type, None)
		assert task_in is not None, f"task type {task_type} is not defined."
		type_in = task_in.get(model_type, None)
		assert type_in is not None, (
			f"model type {model_type} is not defined. (upper task {task_type})"
		)

		return type_in

	@property
	def task_registry(self):
		return self._task_registry

	@property
	def config_registry(self):
		return self._config_registry


# Global registry instance
registry = Registry()

# Expose registration methods as module-level functions
register_config = registry.register_config
register_module = registry.register_module
