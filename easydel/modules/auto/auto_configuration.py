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
import functools
import typing as tp

import flax.nnx
from fjformer import make_shard_and_gather_fns, match_partition_rules
from jax.sharding import PartitionSpec

from easydel.etils.etils import (
	EasyDeLBackends,
	EasyDeLPlatforms,
	get_logger,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.infra.base_module import (
	EasyDeLBaseConfig,
	EasyDeLBaseModule,
)
from easydel.infra.factory import TaskType, registry
from easydel.utils.parameters_transformation import torch_dict_to_easydel_params
from easydel.utils.traversals import flatten_dict, unflatten_dict

logger = get_logger(name=__name__)


def get_modules_by_type(
	model_type: str,
	task_type: TaskType,
) -> tp.Tuple[
	tp.Type[EasyDeLBaseConfig],
	tp.Type[EasyDeLBaseModule] | tp.Any,
	functools.partial | tp.Any,
]:
	"""
	The get_modules_by_type function is a helper function that returns the following:
	    1. The config class for the model type specified (e.g., LlamaConfig, FalconConfig)
	    2. The Flax Model class for the model type specified (e.g., FlaxLlamaForCausalLM, FalconForCausalLM)
	    3. A function to convert a HuggingFace pretrained checkpoint into an easydel checkpoint
	"""
	registred_module = registry.get_module_registration(
		task_type=task_type, model_type=model_type
	)
	return (
		registred_module.config,
		registred_module.module,
		functools.partial(
			torch_dict_to_easydel_params,
			embedding_layer_names=registred_module.embedding_layer_names,
			layernorm_names=registred_module.layernorm_names,
			rnn_based_or_rwkv=registred_module.rnn_based_or_rwkv,
		),
	)


def is_flatten(pytree: dict):
	"""The is_flatten function checks if the pytree is flattened.
	    If it is, then the first key in the dictionary will be a tuple of (mpl, mpl_id).
	    Otherwise, it will be an integer representing mpl_id.

	Args:
	    pytree: dict: Pass the pytree to the function

	Returns:
	    True if the pytree is a flattened tree, and false otherwise
	"""
	mpl = [k for k in pytree.keys()][0]
	return True if isinstance(mpl, tuple) else False


class AutoEasyDeLConfig:
	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: tp.Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		backend: tp.Optional[EasyDeLBackends] = None,
		platform: tp.Optional[EasyDeLPlatforms] = None,
		model_task: TaskType = TaskType.CAUSAL_LM,
		from_torch: bool = False,
		**kwargs,
	) -> EasyDeLBaseConfig:
		"""The from_pretrained function is a helper function that allows you to instantiate a model from the pretrained
		model repository. It takes as input the name of the model (e.g., 'bert-base-uncased') and returns an instance of
		the class corresponding to your model, with all weights loaded from disk.

		Args:
		    cls: Create an instance of the class that called this function.
		    pretrained_model_name_or_path: str: Identify the model in the huggingface model hub.
		    sharding_axis_dims: tp.Sequence[int]: Specify the dimension of each axis in the sharded model_tasking arrays in easydel.
		    shard_attention_computation: bool: whenever to use shard_map for attention.
		    backend: tp.Optional[EasyDeLBackends] : backend to use for model.
				model_task (TaskType): Task type of model load and find.
		    from_torch: should config be loaded from torch models or not.
		    **kwargs: Pass additional arguments to the model and config classes.
		generation process

		Returns:
		    A Model Config
		"""
		if partition_axis is None:
			partition_axis = PartitionAxis()
		from transformers import AutoConfig

		cls_main = AutoConfig if from_torch else EasyDeLBaseConfig
		config = cls_main.from_pretrained(pretrained_model_name_or_path)
		model_type: str = config.model_type

		config_class, module, transform_function = get_modules_by_type(
			model_type,
			model_task,
		)
		config = config_class.from_pretrained(pretrained_model_name_or_path)
		if hasattr(config, "add_jax_args"):
			config.add_jax_args()
		config.add_basic_configurations(
			axis_dims=sharding_axis_dims,
			axis_names=sharding_axis_names,
			partition_axis=partition_axis,
			backend=backend,
			platform=platform,
			shard_attention_computation=shard_attention_computation,
		)
		for k, v in kwargs.items():
			setattr(config, k, v)

		return config


class AutoShardAndGatherFunctions:
	"""
	A class to automatically generate shard and gather functions for a given model configuration.

	This class provides two methods to generate shard and gather functions:

	- `from_config`: Generates functions based on a provided `EasyDeLBaseConfig` object.
	- `from_pretrained`: Generates functions based on a pretrained model name or path.

	Attributes:
	    None

	Methods:
	    from_config: Generates shard and gather functions based on a provided `EasyDeLBaseConfig` object.
	    from_pretrained: Generates functions based on a pretrained model name or path.
	"""

	@classmethod
	def from_config(
		cls,
		config: EasyDeLBaseConfig,
		partition_rules: tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec]]] = None,
		flatten: bool = True,
		model_task: TaskType = TaskType.CAUSAL_LM,
		depth_target: tp.Optional[tp.List[str]] = None,
	):
		"""
		Generates shard and gather functions based on a provided `EasyDeLBaseConfig` object.

		Args:
		    config: An `EasyDeLBaseConfig` object containing the model configuration.
		    partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
		      If None, uses the default partition rules from the `config`.
		    flatten: Whether to flatten the shard and gather functions. Defaults to True.
				model_task (TaskType): Task type of model load and find.
		    depth_target: Pad the sharding to depth, for example make {params:tensor} with depth_target = ["row"] to {row:{params:tensor}}. Defaults to None.

		Returns:
		    A tuple containing the shard and gather functions.
		"""
		if partition_rules is None:
			partition_rules = config.get_partition_rules(True)
		_, module, _ = get_modules_by_type(config.model_type, model_task)
		model = module.lazy_init(config=config, rngs=flax.nnx.Rngs(0))

		partition_specs = match_partition_rules(
			partition_rules,
			model.graphtree_params_shape,
		)

		shard_fns, gather_fns = make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=config.mesh,
		)
		if flatten and not is_flatten(shard_fns):
			gather_fns = flatten_dict(gather_fns)
			shard_fns = flatten_dict(shard_fns)
		elif not flatten and is_flatten(shard_fns):
			gather_fns = unflatten_dict(gather_fns)
			shard_fns = unflatten_dict(shard_fns)

		return shard_fns, gather_fns

	@staticmethod
	def from_params(params, partition_rules, mesh):
		partition_specs = match_partition_rules(partition_rules, params)
		return make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=mesh,
		)

	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: tp.Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		backend: tp.Optional[EasyDeLBackends] = None,
		platform: tp.Optional[EasyDeLPlatforms] = None,
		partition_rules: tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec]]] = None,
		flatten: bool = True,
		config_kwargs: tp.Optional[tp.Mapping[str, tp.Any]] = None,
		model_task: TaskType = TaskType.CAUSAL_LM,
		from_torch: bool = False,
		trust_remote_code: bool = False,
	) -> tp.Tuple[tp.Mapping[str, tp.Callable], tp.Mapping[str, tp.Callable]]:
		"""
		Generates shard and gather functions based on a pretrained model name or path.

		Args:
		    pretrained_model_name_or_path: The name or path of the pretrained model.
		    sharding_axis_dims: The dimensions of the sharding axes. Defaults to (1, -1, 1, 1).
		    sharding_axis_names: The names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation: Whether to shard the attention computation. Defaults to True.
		    backend: The backend to use for custom kernels. Defaults to None.
		    partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
		        If None, uses the default partition rules from the `config`.
		    flatten: Whether to flatten the shard and gather functions. Defaults to True.
		    config_kwargs: Additional keyword arguments to pass to the `AutoEasyDeLConfig` constructor. Defaults to None.
				model_task (TaskType): Task type of model load and find.
				from_torch: should config be loaded from torch models or not.
		    trust_remote_code (bool): whenever to trust remote code loaded from HF.
		Returns:
		    A tuple containing the shard and gather functions.
		"""
		if partition_axis is None:
			partition_axis = PartitionAxis()
		config = AutoEasyDeLConfig.from_pretrained(
			pretrained_model_name_or_path,
			sharding_axis_dims=sharding_axis_dims,
			sharding_axis_names=sharding_axis_names,
			partition_axis=partition_axis,
			shard_attention_computation=shard_attention_computation,
			backend=backend,
			platform=platform,
			from_torch=from_torch,
			trust_remote_code=trust_remote_code,
			model_task=model_task,
		)
		if config_kwargs is not None:
			for k, v in config_kwargs.items():
				setattr(config, k, v)
		return cls.from_config(
			config=config,
			partition_rules=partition_rules,
			flatten=flatten,
			model_task=model_task,
		)
