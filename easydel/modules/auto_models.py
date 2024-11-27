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
import gc
import os
import re
import warnings
from typing import (
	Any,
	Callable,
	List,
	Mapping,
	Optional,
	Sequence,
	Tuple,
	Type,
	Union,
)

import flax.traverse_util
import jax.numpy
from fjformer import make_shard_and_gather_fns, match_partition_rules
from flax.traverse_util import unflatten_dict
from jax.sharding import PartitionSpec

from easydel.etils.easystate import EasyDeLState
from easydel.etils.etils import (
	EasyDeLBackends,
	EasyDeLPlatforms,
	EasyDeLQuantizationMethods,
	get_logger,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.modules.factory import registry
from easydel.modules.modeling_utils import (
	EasyDeLBaseConfig,
	EasyDeLBaseConfigDict,
	EasyDeLBaseModule,
)
from easydel.transform.parameters_transformation import torch_dict_to_easydel_params
from easydel.utils.quantizers import DEFAULT_QUANTIZATION_PATTERN

logger = get_logger(name=__name__)


def get_modules_by_type(
	model_type: str,
	task_type: str = "causal-language-model",
) -> Tuple[
	Type[EasyDeLBaseConfig],
	Type[EasyDeLBaseModule] | Any,
	functools.partial | Any,
]:
	"""
	The get_modules_by_type function is a helper function that returns the following:
	    1. The config class for the model type specified (e.g., LlamaConfig, FalconConfig)
	    2. The Flax Model class for the model type specified (e.g., FlaxLlamaForCausalLM, FlaxFalconForCausalLM)
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


class AutoEasyDeLModelForCausalLM:
	"""This class provides a convenient way to load and shard pretrained causal language models from the Hugging Face Hub
	and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed training and inference
	with JAX.

	This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
	parameter sharding, and interaction with the EasyDeL framework.

	Attributes:
	    None

	Examples:
	    ```python
	    import jax
	    from easydel import AutoEasyDeLModelForCausalLM

	    # Load a GPT-2 model on a single CPU
	    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
	      "gpt2", device=jax.devices("cpu")[0]
	    )

	    # Load a GPT-2 model sharded across 8 GPUs with data parallelism (DP) and fully sharded data parallelism (FSDP)
	    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
	      "gpt2",
	      sharding_axis_dims=(1, 8, 1, 1),
	      sharding_axis_names=("dp", "fsdp", "tp", "sp"),
	      device=jax.devices("cpu")[0],  # offload to CPU [OPTIONAL]
	      from_torch=True,
	    )
	    ```
	"""

	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		device: Optional[jax.Device] = None,
		dtype: jax.numpy.dtype = jax.numpy.float32,
		param_dtype: jax.numpy.dtype = jax.numpy.float32,
		precision: Optional[jax.lax.Precision] = None,
		sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		input_shape: Tuple[int, int] = (1, 1),
		shard_fns: Optional[Mapping[tuple, Callable] | dict] = None,
		backend: Optional[EasyDeLBackends] = None,
		platform: Optional[EasyDeLPlatforms] = None,
		config_kwargs: Optional[EasyDeLBaseConfigDict] = None,
		auto_shard_params: bool = False,
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]] = None,
		quantization_method: Optional[EasyDeLQuantizationMethods] = None,
		quantization_platform: Optional[EasyDeLPlatforms] = EasyDeLPlatforms.JAX,
		quantization_block_size: int = 128,
		bit_targeted_params: Optional[List[str]] = None,
		verbose_params: bool = False,
		safe: bool = True,
		from_torch: Optional[bool] = None,
		**kwargs,
	) -> Tuple[EasyDeLBaseModule, dict]:
		"""Loads and shards a pretrained causal language model from the Hugging Face Hub and converts it into an
		EasyDeL compatible model.

		Args:
		    pretrained_model_name_or_path (str): Path or name of the pretrained model in the Hugging Face Hub.
		    device (jax.Device, optional): Device to load the model on. Defaults to the first CPU.
		    dtype (jax.numpy.dtype, optional): Data type of the model. Defaults to jax.numpy.float32.
		    param_dtype (jax.numpy.dtype, optional): Data type of the model parameters. Defaults to jax.numpy.float32.
		    precision (jax.lax.Precision, optional): Precision for computations. Defaults to jax.lax.Precision("fastest").
		    sharding_axis_dims (Sequence[int], optional): Dimensions of each sharding axis. Defaults to (1, -1, 1, 1).
		    sharding_axis_names (Sequence[str], optional): Names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
		    input_shape (Tuple[int, int], optional): Shape of the input to the model. Defaults to (1, 1).
		    shard_fns (Optional[Mapping[tuple, Callable] | dict], optional): Sharding functions to use for the model. If None, auto-sharding is used if auto_shard_params is True. Defaults to None.
		    platform (Optional[EasyDeLPlatforms], optional): platform to use for the model. Defaults to None.
				backend (Optional[EasyDeLBackends], optional): backend to use for the model. Defaults to None.
		    config_kwargs (Optional[Mapping[str, Any] | EasyDeLBaseConfigDict], optional): Configuration keyword arguments to pass to the model config. Defaults to None.
		    auto_shard_params (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
		    partition_rules (Optional[Tuple[Tuple[str, PartitionSpec]]], optional): Custom partition rules for parameter sharding. If not None, shard_fns should also be provided. Defaults to None.
		    quantization_method (EasyDeLQuantizationMethods, optional): quantization_method to be used to quantize model weights. Defaults to None.
		    quantization_platform (Optional[EasyDeLPlatforms], optional): Platform to use for the weight quants. Defaults to None.
				quantization_block_size (int): block size to be used for quantizing arrays (only for NF4).
		    bit_targeted_params (Optional[List[str]], optional): List of parameter names to convert to 8-bit precision. If  None and 8bit is True, all kernels and embeddings are converted to 8-bit. Defaults to None.
		    verbose_params (bool): whenever to log number of parameters in converting state.
		    safe (bool): whenever to use safetensors to load engine or parameters (requires engine or parameters to be saved with safe=True while saving them)
		    from_torch (bool): whenever to load the model from transformers-pytorch.
		    **kwargs: Additional keyword arguments to pass to the model and config classes.

		Returns:
		    Tuple[EasyDeLBaseModule, dict]: A tuple containing the EasyDeL model and the loaded and sharded
		        model parameters.
		"""
		if device is None:
			device = jax.devices("cpu")[0]
		if precision is None:
			precision = jax.lax.Precision("fastest")
		if partition_axis is None:
			partition_axis = PartitionAxis()
		if from_torch is None:
			from_torch = not cls._is_easydel(
				pretrained_model_name_or_path=pretrained_model_name_or_path,
			)

		if from_torch:
			return cls._from_torch(
				pretrained_model_name_or_path=pretrained_model_name_or_path,
				param_dtype=param_dtype,
				dtype=dtype,
				shard_fns=shard_fns,
				auto_shard_params=auto_shard_params,
				precision=precision,
				backend=backend,
				platform=platform,
				verbose_params=verbose_params,
				partition_axis=partition_axis,
				quantization_method=quantization_method,
				quantization_platform=quantization_platform,
				quantization_block_size=quantization_block_size,
				partition_rules=partition_rules,
				bit_targeted_params=bit_targeted_params,
				sharding_axis_names=sharding_axis_names,
				sharding_axis_dims=sharding_axis_dims,
				input_shape=input_shape,
				config_kwargs=config_kwargs,
				device=device,
				shard_attention_computation=shard_attention_computation,
				**kwargs,
			)
		with jax.default_device(device):
			return cls._from_easydel_params(
				auto_shard_params=auto_shard_params,
				input_shape=input_shape,
				partition_axis=partition_axis,
				sharding_axis_dims=sharding_axis_dims,
				sharding_axis_names=sharding_axis_names,
				shard_fns=shard_fns,
				param_dtype=param_dtype,
				config_kwargs=config_kwargs,
				partition_rules=partition_rules,
				precision=precision,
				dtype=dtype,
				backend=backend,
				platform=platform,
				pretrained_model_name_or_path=pretrained_model_name_or_path,
				quantization_method=quantization_method,
				quantization_platform=quantization_platform,
				quantization_block_size=quantization_block_size,
				bit_targeted_params=bit_targeted_params,
				safe=safe,
				**kwargs,
			)

	@staticmethod
	def _from_torch(
		pretrained_model_name_or_path,
		device,
		dtype: jax.numpy.dtype,
		param_dtype: jax.numpy.dtype,
		precision: Optional[jax.lax.Precision],
		sharding_axis_dims: Sequence[int],
		sharding_axis_names: Sequence[str],
		partition_axis: PartitionAxis,
		shard_attention_computation: bool,
		input_shape: Tuple[int, int],
		shard_fns: Optional[Mapping[tuple, Callable] | dict],
		backend: Optional[EasyDeLBackends],
		platform: Optional[EasyDeLPlatforms],
		config_kwargs: Optional[Mapping[str, Any]],
		auto_shard_params: bool,
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]],
		quantization_method: Optional[EasyDeLQuantizationMethods],
		quantization_platform: Optional[EasyDeLPlatforms],
		quantization_block_size: int,
		bit_targeted_params: Optional[List[str]],
		verbose_params: bool,
		**kwargs,
	):
		from transformers import AutoConfig, AutoModelForCausalLM

		try:
			import torch

			if torch.cuda.is_available():

				def _clear():
					gc.collect()
					torch.cuda.empty_cache()

			else:

				class torch:
					bfloat16 = None

				def _clear():
					gc.collect()

		except ModuleNotFoundError as er:
			raise ModuleNotFoundError(
				"in order to load model from torch you should install torch first "
				"run `pip install torch`"
			) from er

		logger.debug(f"Downloading model config from {pretrained_model_name_or_path}")
		trust_remote_code = kwargs.get("trust_remote_code", False)
		config = AutoConfig.from_pretrained(
			pretrained_model_name_or_path,
			trust_remote_code=trust_remote_code,
		)
		model_type: str = config.model_type

		config_class, module, transform_function = get_modules_by_type(model_type)

		logger.debug(f"Downloading model weights from {pretrained_model_name_or_path}")
		model = AutoModelForCausalLM.from_pretrained(
			pretrained_model_name_or_path,
			**kwargs,
		)
		generation_config = getattr(model, "generation_config", None)
		if verbose_params:
			print(
				f"PyTorch - HF Model contains {sum(p.numel() for p in model.parameters()) / 1e9} Billion Parameters"
			)
		config_class = config_class.from_pretrained(pretrained_model_name_or_path)
		state_dict = model.state_dict()

		# Clear and collect memory after deleting the model
		del model
		_clear()

		logger.debug("adding model basic EasyDeL configurations.")
		if hasattr(config_class, "add_jax_args"):
			config_class.add_jax_args()
		config_class.add_basic_configurations(
			axis_dims=sharding_axis_dims,
			axis_names=sharding_axis_names,
			partition_axis=partition_axis,
			backend=backend,
			platform=platform,
			shard_attention_computation=shard_attention_computation,
		)
		if config_kwargs is not None:
			for k, v in config_kwargs.items():
				setattr(config_class, k, v)
		logger.debug("creating easydel model")
		ed_model = module(
			config=config_class,
			_do_init=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			input_shape=input_shape,
		)
		ed_model.generation_config = generation_config
		needs = [
			s.replace(".kernel", ".weight")
			.replace(".scale", ".weight")
			.replace(".embedding", ".weight")
			for s in list(
				flax.traverse_util.flatten_dict(ed_model.params_shape_tree, sep=".").keys()
			)
		]
		for k in list(state_dict.keys()):
			if k not in needs:
				tensor = state_dict.pop(k)
				del tensor
				_clear()
				logger.debug(f"removing {k} from weights as it was not needed by flax model")

		_clear()

		if shard_fns is not None:
			if auto_shard_params:
				warnings.warn(
					"`auto_shard_params` will be ignored since you are passing custom sharding functions",
					stacklevel=1,
				)
			logger.debug("sharding model parameters based on the given shard_fns.")
			if not is_flatten(shard_fns):
				shard_fns = flax.traverse_util.flatten_dict(shard_fns)
		elif auto_shard_params:
			shard_fns, _ = AutoShardAndGatherFunctions.from_pretrained(
				pretrained_model_name_or_path=pretrained_model_name_or_path,
				partition_rules=partition_rules,
				sharding_axis_dims=sharding_axis_dims,
				sharding_axis_names=sharding_axis_names,
				partition_axis=partition_axis,
				shard_attention_computation=shard_attention_computation,
				backend=backend,
				platform=platform,
				input_shape=input_shape,  # type:ignore
				config_kwargs=config_kwargs,
				trust_remote_code=trust_remote_code,
			)
		logger.debug("converting huggingface-model to easydel-model.")
		params_pattern_selection = None
		if bit_targeted_params is None:
			params_pattern_selection = re.compile(DEFAULT_QUANTIZATION_PATTERN)

		leg_load_8bit_detected = kwargs.get("load_8bit", None)
		if leg_load_8bit_detected is not None:
			warnings.warn(
				"load_8bit=True Detected, "
				"please use `quantization_method=='8bit'` (automatically setting quantization_method to 8bit)",
				stacklevel=1,
			)
		uses_tie_word_embedding = getattr(config, "tie_word_embeddings", False)

		params = transform_function(
			state_dict,
			config=config,
			device=device,
			shard_fns=shard_fns,
			quantization_method=quantization_method,
			quantization_platform=quantization_platform,
			params_pattern_selection=params_pattern_selection,
			remove_state_dict=True,
			uses_tie_word_embedding=uses_tie_word_embedding,
			dtype=param_dtype,
			block_size=quantization_block_size,
		)

		# Clear and collect memory after converting the model
		del state_dict
		_clear()

		if is_flatten(params):
			logger.info("converted parameters are flatten making them unflatten ")
			params = unflatten_dict(params)

		if verbose_params:
			print(
				f"JAX - EasyDeL Model contains "
				f"{sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(params))[0]) / 1e9}"
				f" Billion Parameters"
			)
		return ed_model, params

	@staticmethod
	def _from_easydel_params(
		pretrained_model_name_or_path,
		dtype: jax.numpy.dtype,
		param_dtype: jax.numpy.dtype,
		precision: Optional[jax.lax.Precision],
		sharding_axis_dims: Sequence[int],
		sharding_axis_names: Sequence[str],
		partition_axis: PartitionAxis,
		input_shape: Tuple[int, int],
		shard_fns: Optional[Mapping[tuple, Callable] | dict],
		quantization_method: Optional[EasyDeLQuantizationMethods],
		quantization_platform: Optional[EasyDeLPlatforms],
		backend: Optional[EasyDeLBackends],
		platform: Optional[EasyDeLPlatforms],
		bit_targeted_params: Optional[List[str]],
		quantization_block_size: int,
		config_kwargs: Optional[Mapping[str, Any]],
		auto_shard_params: bool,
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]],
		safe: bool,
		**kwargs,
	):
		from easydel.modules.modeling_utils import EasyDeLBaseModule

		return EasyDeLBaseModule.from_pretrained(
			pretrained_model_name_or_path=pretrained_model_name_or_path,
			input_shape=input_shape,
			dtype=dtype,
			precision=precision,
			param_dtype=param_dtype,
			partition_axis=partition_axis,
			auto_shard_params=auto_shard_params,
			shard_fns=shard_fns,
			sharding_axis_dims=sharding_axis_dims,
			sharding_axis_names=sharding_axis_names,
			backend=backend,
			platform=platform,
			config_kwargs=config_kwargs,
			partition_rules=partition_rules,
			quantization_method=quantization_method,
			quantization_platform=quantization_platform,
			bit_targeted_params=bit_targeted_params,
			quantization_block_size=quantization_block_size,
			safe=safe,
			**kwargs,
		)

	@classmethod
	def _is_easydel(
		cls,
		pretrained_model_name_or_path,
		FLAX_WEIGHTS_NAME="easydel-model.parameters",
		cache_dir: Optional[Union[str, os.PathLike]] = None,
		force_download: bool = False,
		local_files_only: bool = False,
		token: Optional[Union[str, bool]] = None,
		revision: str = "main",
	):
		from transformers.utils import cached_file as _cached_file
		from transformers.utils import download_url as _download_url
		from transformers.utils import is_remote_url as _is_remote_url

		proxies = None
		subfolder = ""
		commit_hash = None
		pretrained_model_name_or_path = str(pretrained_model_name_or_path)
		if os.path.isdir(pretrained_model_name_or_path):
			if os.path.isfile(
				os.path.join(
					pretrained_model_name_or_path,
					subfolder,
					FLAX_WEIGHTS_NAME,
				)
			):
				archive_file = os.path.join(  # noqa
					pretrained_model_name_or_path,
					subfolder,
					FLAX_WEIGHTS_NAME,
				)
			else:
				raise EnvironmentError(
					f"Error no file named {FLAX_WEIGHTS_NAME} found in"
					f" directory {pretrained_model_name_or_path}"
				)
		elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
			...
		elif _is_remote_url(pretrained_model_name_or_path):
			filename = pretrained_model_name_or_path
			resolved_archive_file = _download_url(pretrained_model_name_or_path)
		else:
			filename = FLAX_WEIGHTS_NAME
			try:
				cached_file_kwargs = {
					"cache_dir": cache_dir,
					"force_download": force_download,
					"proxies": proxies,
					"local_files_only": local_files_only,
					"token": token,
					"user_agent": {
						"file_type": "model",
						"framework": "flax",
						"from_auto_class": False,
					},
					"revision": revision,
					"subfolder": subfolder,
					"_raise_exceptions_for_gated_repo": False,
					"_raise_exceptions_for_missing_entries": False,
					"_commit_hash": commit_hash,
				}
				resolved_archive_file = _cached_file(
					pretrained_model_name_or_path,
					filename,
					**cached_file_kwargs,
				)

				if resolved_archive_file is None:
					return False
			except EnvironmentError:
				raise
			except Exception:
				return False
		return True


class AutoEasyDeLConfig:
	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		backend: Optional[EasyDeLBackends] = None,
		platform: Optional[EasyDeLPlatforms] = None,
		from_torch: bool = False,
		**kwargs,
	) -> EasyDeLBaseConfig:
		"""The from_pretrained function is a helper function that allows you to instantiate a model from the pretrained
		model repository. It takes as input the name of the model (e.g., 'bert-base-uncased') and returns an instance of
		the class corresponding to your model, with all weights loaded from disk.

		Args:
		    cls: Create an instance of the class that called this function
		    pretrained_model_name_or_path: str: Identify the model in the huggingface model hub
		    sharding_axis_dims: Sequence[int]: Specify the dimension of each axis in the sharded model
		    sharding_axis_names: Sequence[str]: Specify the order of sharding
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation: bool: whenever to use shard_map for attention
		    backend: Optional[EasyDeLBackends] : backend to use for model
		    from_torch: should config be loaded from torch models or not.
		    **kwargs: Pass additional arguments to the model and config classes
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

		config_class, module, transform_function = get_modules_by_type(model_type)
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
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec]]] = None,
		flatten: bool = True,
		input_shape: Tuple[int, int] = (1, 1),
		depth_target: Optional[List[str]] = None,
	):
		"""
		Generates shard and gather functions based on a provided `EasyDeLBaseConfig` object.

		Args:
		    config: An `EasyDeLBaseConfig` object containing the model configuration.
		    partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
		        If None, uses the default partition rules from the `config`.
		    flatten: Whether to flatten the shard and gather functions. Defaults to True.
		    input_shape: The input shape of the model. Defaults to (1, 1).
		    depth_target: Pad the sharding to depth, for example make {params:tensor} with depth_target = ["row"] to {row:{params:tensor}}. Defaults to None.

		Returns:
		    A tuple containing the shard and gather functions.
		"""
		if partition_rules is None:
			partition_rules = config.get_partition_rules(True)
		_, module, _ = get_modules_by_type(config.model_type)
		model = module(config=config, _do_init=False, input_shape=input_shape)

		partition_specs = match_partition_rules(partition_rules, model.params_shape_tree)
		shard_fns, gather_fns = make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=config.mesh,
		)
		if depth_target is not None:
			for dp in depth_target[::-1]:
				gather_fns = {dp: gather_fns}
				shard_fns = {dp: shard_fns}
		if flatten and not is_flatten(shard_fns):
			gather_fns = flax.traverse_util.flatten_dict(gather_fns)
			shard_fns = flax.traverse_util.flatten_dict(shard_fns)
		elif not flatten and is_flatten(shard_fns):
			gather_fns = flax.traverse_util.unflatten_dict(gather_fns)
			shard_fns = flax.traverse_util.unflatten_dict(shard_fns)

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
		input_shape: Tuple[int, int] = (1, 1),
		sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		backend: Optional[EasyDeLBackends] = None,
		platform: Optional[EasyDeLPlatforms] = None,
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec]]] = None,
		flatten: bool = True,
		config_kwargs: Optional[Mapping[str, Any]] = None,
		depth_target: Optional[List[str]] = None,
		from_torch: bool = False,
		trust_remote_code: bool = False,
	) -> Tuple[Mapping[str, Callable], Mapping[str, Callable]]:
		"""
		Generates shard and gather functions based on a pretrained model name or path.

		Args:
		    pretrained_model_name_or_path: The name or path of the pretrained model.
		    input_shape: The input shape of the model. Defaults to (1, 1).
		    sharding_axis_dims: The dimensions of the sharding axes. Defaults to (1, -1, 1, 1).
		    sharding_axis_names: The names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation: Whether to shard the attention computation. Defaults to True.
		    backend: The backend to use for custom kernels. Defaults to None.
		    partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
		        If None, uses the default partition rules from the `config`.
		    flatten: Whether to flatten the shard and gather functions. Defaults to True.
		    config_kwargs: Additional keyword arguments to pass to the `AutoEasyDeLConfig` constructor. Defaults to None.
		    depth_target: Pad the sharding to depth, for example make {params:tensor} with depth_target = ["row"] to {row:{params:tensor}}. Defaults to None.
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
		)
		if config_kwargs is not None:
			for k, v in config_kwargs.items():
				setattr(config, k, v)
		return cls.from_config(
			config=config,
			partition_rules=partition_rules,
			flatten=flatten,
			input_shape=input_shape,
			depth_target=depth_target,
		)


class AutoStateForCausalLM:
	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		device: Optional[jax.Device] = None,
		dtype: jax.numpy.dtype = jax.numpy.float32,
		param_dtype: jax.numpy.dtype = jax.numpy.float32,
		precision: Optional[jax.lax.Precision] = None,
		sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		input_shape: Tuple[int, int] = (1, 1),
		shard_fns: Optional[Mapping[tuple, Callable] | dict] = None,
		backend: Optional[str] = None,
		config_kwargs: Optional[Mapping[str, Any]] = None,
		auto_shard_params: bool = False,
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]] = None,
		load_in_8bit: bool = False,
		bit_targeted_params: Optional[List[str]] = None,
		verbose_params: bool = False,
		safe: bool = True,
		from_torch: bool = True,
		**kwargs,
	) -> EasyDeLState:
		"""
		Loads and shards a pretrained causal language model from the Hugging Face Hub and converts it into an
		EasyDeL compatible state.

		Args:
		    pretrained_model_name_or_path (str): Path or name of the pretrained model in the Hugging Face Hub.
		    device (jax.Device, optional): Device to load the model on. Defaults to the first CPU.
		    dtype (jax.numpy.dtype, optional): Data type of the model. Defaults to jax.numpy.float32.
		    param_dtype (jax.numpy.dtype, optional): Data type of the model parameters. Defaults to jax.numpy.float32.
		    precision (jax.lax.Precision, optional): Precision for computations. Defaults to jax.lax.Precision("fastest").
		    sharding_axis_dims (Sequence[int], optional): Dimensions of each sharding axis. Defaults to (1, -1, 1, 1).
		    sharding_axis_names (Sequence[str], optional): Names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
		    input_shape (Tuple[int, int], optional): Shape of the input to the model. Defaults to (1, 1).
		    shard_fns (Optional[Mapping[tuple, Callable] | dict], optional): Sharding functions to use for the model. If None, auto-sharding is used if auto_shard_params is True. Defaults to None.
		    backend (Optional[str], optional): Backend to use for the model. Defaults to None.
		    config_kwargs (Optional[Mapping[str, Any]], optional): Configuration keyword arguments to pass to the model config. Defaults to None.
		    auto_shard_params (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
		    partition_rules (Optional[Tuple[Tuple[str, PartitionSpec]]], optional): Custom partition rules for parameter sharding. If not None, shard_fns should also be provided. Defaults to None.
		    quantization_method (EasyDeLQuantizationMethods, optional): quantization_method to be used to quantize model weights. Defaults to None.
		    bit_targeted_params (Optional[List[str]], optional): List of parameter names to convert to 8-bit precision. If  None and 8bit is True, all kernels and embeddings are converted to 8-bit. Defaults to None.
		    verbose_params (bool): whenever to log number of parameters in converting state.
		    safe (bool): whenever to use safetensors to load engine or parameters (requires engine or parameters to be saved with safe=True while saving them)
		    from_torch (bool): whenever to load the model from transformers-pytorch.
		    **kwargs: Additional keyword arguments to pass to the model and config classes.

		Returns:
		    EasyDeLState: containing the EasyDeL state and the loaded and sharded model parameters.
		"""
		model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
			pretrained_model_name_or_path=pretrained_model_name_or_path,
			device=device,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			sharding_axis_dims=sharding_axis_dims,
			sharding_axis_names=sharding_axis_names,
			partition_axis=partition_axis,
			shard_attention_computation=shard_attention_computation,
			input_shape=input_shape,
			shard_fns=shard_fns,
			backend=backend,
			config_kwargs=config_kwargs,
			auto_shard_params=auto_shard_params,
			partition_rules=partition_rules,
			load_in_8bit=load_in_8bit,
			bit_targeted_params=bit_targeted_params,
			verbose_params=verbose_params,
			safe=safe,
			from_torch=from_torch,
			**kwargs,
		)
		return EasyDeLState.create(
			apply_fn=model.__call__,
			params=params,
			module=model,
			module_config=model.config,
			tx=None,
			tx_init=None,
		)
