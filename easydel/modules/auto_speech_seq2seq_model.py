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

import gc
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
)

import flax.traverse_util
import jax.numpy
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
from easydel.modules.auto_configuration import (
	AutoShardAndGatherFunctions,
	get_modules_by_type,
	is_flatten,
)
from easydel.modules.auto_modeling import BaseAutoEasyModel
from easydel.modules.factory import TaskType
from easydel.modules.modeling_utils import (
	EasyDeLBaseConfigDict,
	EasyDeLBaseModule,
)
from easydel.utils.quantizers import DEFAULT_QUANTIZATION_PATTERN

logger = get_logger(name=__name__)


class AutoEasyDeLModelForSpeechSeq2Seq(BaseAutoEasyModel):
	"""This class provides a convenient way to load and shard pretrained causal language models from the Hugging Face Hub
	and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed training and inference
	with JAX.

	This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
	parameter sharding, and interaction with the EasyDeL framework.

	Attributes:
	    None

	Examples:

	    >>> import jax
	    >>> from easydel import AutoEasyDeLModelForSpeechSeq2Seq

	    >>> # Load a openai/whisper-large-v3-turbo sharded
	    >>> model, params = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
	    ...  "openai/whisper-large-v3-turbo",
	    ...  auto_shard_params=True,
	    >>> )

	    >>> # Load a openai/whisper-large-v3-turbo model sharded across 8 GPUs with data parallelism (DP) and fully sharded data parallelism (FSDP)
	    >>> model, params = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
	    ...  "openai/whisper-large-v3-turbo",
	    ...  sharding_axis_dims=(1, 8, 1, 1),
	    ...  sharding_axis_names=("dp", "fsdp", "tp", "sp"),
	    ...  device=jax.devices("cpu")[0],  # offload to CPU [OPTIONAL]
	    ...  from_torch=True,
	    >>> )
	    ```
	"""

	model_task: TaskType = TaskType.CAUSAL_LM  # Static

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
		input_shape: Optional[Tuple[int, int, int]] = None,
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
		    input_shape (Tuple[int, int, int], optional): Shape of the input to the model. Defaults to (1, 1).
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
		from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

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

		config_class, module, transform_function = get_modules_by_type(
			model_type,
			task_type=TaskType.SEQ_TO_SEQ,
		)

		logger.debug(f"Downloading model weights from {pretrained_model_name_or_path}")
		model = AutoModelForSpeechSeq2Seq.from_pretrained(
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
				model_task=TaskType.SEQ_TO_SEQ,
				input_shape=input_shape,  # type:ignore
				config_kwargs=config_kwargs,
				trust_remote_code=trust_remote_code,
			)
		logger.debug("converting huggingface-model to easydel-model.")
		params_pattern_selection = None
		if bit_targeted_params is None:
			params_pattern_selection = re.compile(DEFAULT_QUANTIZATION_PATTERN)
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


class AutoStateForSpeechSeq2Seq:
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
		model, params = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
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
