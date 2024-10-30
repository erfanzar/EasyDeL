from enum import Enum
from typing import Dict, List, Optional, Type, TypeVar
from dataclasses import dataclass

from easydel.modules.modeling_utils import EDPretrainedConfig, EDPretrainedModel

T = TypeVar("T")


class ConfigType(str, Enum):
	MODULE_CONFIG = "module-config"


class TaskType(str, Enum):
	CAUSAL_LM = "causal-language-model"
	SEQ_CLASS = "sequence-classification"
	VISION_LM = "vision-language-model"
	COND_GEN = "conditional-generation"
	AUDIO_CLASS = "audio-classification"
	BASE_MODULE = "base-module"
	SEQ_TO_SEQ = "seq-to-seq"


@dataclass
class ModuleRegistration:
	module: EDPretrainedModel
	config: EDPretrainedConfig
	embedding_layer_names: Optional[List[str]] = None
	layernorm_names: Optional[List[str]] = None
	rnn_based_or_rwkv: bool = False


class Registry:
	def __init__(self):
		self._config_registry: Dict[ConfigType, Dict] = {ConfigType.MODULE_CONFIG: {}}

		self._task_registry: Dict[TaskType, Dict[str, ModuleRegistration]] = {
			task_type: {} for task_type in TaskType
		}

	def register_config(
		self, config_type: str, config_field: ConfigType = ConfigType.MODULE_CONFIG
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
			self._config_registry[config_field][config_type] = obj
			return obj

		return wrapper

	def register_module(
		self,
		task_type: TaskType,
		config: EDPretrainedConfig,
		model_type: str,
		embedding_layer_names: Optional[List[str]] = None,
		layernorm_names: Optional[List[str]] = None,
		rnn_based_or_rwkv: bool = False,
	) -> callable:
		"""
		Register a module for a specific task.

		Args:
		    task_type: Type of task
		    config: Configuration for the module
		    model_type: Identifier for the model
		    embedding_layer_names: Names of embedding layers
		    layernorm_names: Names of layer normalization layers
		    rnn_based_or_rwkv: Whether the model is RNN-based or RWKV

		Returns:
		    Decorator function
		"""

		def wrapper(module: T) -> T:
			self._task_registry[task_type][model_type] = ModuleRegistration(
				module=module,
				config=config,
				embedding_layer_names=embedding_layer_names,
				layernorm_names=layernorm_names,
				rnn_based_or_rwkv=rnn_based_or_rwkv,
			)
			return module

		return wrapper

	def get_config(
		self, config_type: str, config_field: ConfigType = ConfigType.MODULE_CONFIG
	) -> Type:
		"""Get registered configuration class."""
		return self._config_registry[config_field][config_type]

	def get_module_registration(
		self, task_type: TaskType, model_type: str
	) -> ModuleRegistration:
		"""Get registered module information."""
		return self._task_registry[task_type][model_type]

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
