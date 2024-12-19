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

import typing as tp
import jax.numpy
from jax.sharding import PartitionSpec

from easydel.etils.easystate import EasyDeLState
from easydel.etils.etils import (
	EasyDeLBackends,
	EasyDeLPlatforms,
	EasyDeLQuantizationMethods,
	get_logger,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.infra.base_config import EasyDeLBaseConfigDict
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType
from easydel.modules.auto_modeling import BaseAutoEasyModel

logger = get_logger(name=__name__)


class AutoEasyDeLModelForCausalLM(BaseAutoEasyModel):
	"""This class provides a convenient way to load and shard pretrained causal language models from the Hugging Face Hub
	and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed training and inference
	with JAX.

	This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
	parameter sharding, and interaction with the EasyDeL framework.

	Attributes:
	    None

	Examples:

	    >>> import jax
	    >>> from easydel import AutoEasyDeLModelForCausalLM

	    >>> # Load a GPT-2 model on a single CPU
	    >>> model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
	    >>>   "gpt2", device=jax.devices("cpu")[0]
	    >>> )

	    >>> # Load a GPT-2 model sharded across 8 GPUs with data parallelism (DP) and fully sharded data parallelism (FSDP)
	    >>> model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
	    ...  "gpt2",
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
		device: tp.Optional[jax.Device] = None,
		dtype: jax.numpy.dtype = jax.numpy.float32,
		param_dtype: jax.numpy.dtype = jax.numpy.float32,
		precision: tp.Optional[jax.lax.Precision] = None,
		sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: tp.Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		shard_fns: tp.Optional[tp.Mapping[tuple, tp.Callable] | dict] = None,
		backend: tp.Optional[EasyDeLBackends] = None,
		platform: tp.Optional[EasyDeLPlatforms] = None,
		config_kwargs: tp.Optional[EasyDeLBaseConfigDict] = None,
		auto_shard_model: bool = False,
		partition_rules: tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec], ...]] = None,
		quantization_method: tp.Optional[EasyDeLQuantizationMethods] = None,
		quantization_block_size: int = 128,
		from_torch: tp.Optional[bool] = None,
		**kwargs,
	) -> EasyDeLBaseModule:
		"""Loads and shards a pretrained causal language model from the Hugging Face Hub and converts it into an
		EasyDeL compatible model.

		Args:
		    pretrained_model_name_or_path (str): Path or name of the pretrained model in the Hugging Face Hub.
		    device (jax.Device, optional): Device to load the model on. Defaults to the first CPU.
		    dtype (jax.numpy.dtype, optional): Data type of the model. Defaults to jax.numpy.float32.
		    param_dtype (jax.numpy.dtype, optional): Data type of the model parameters. Defaults to jax.numpy.float32.
		    precision (jax.lax.Precision, optional): Precision for computations. Defaults to jax.lax.Precision("fastest").
		    sharding_axis_dims (tp.Sequence[int], optional): Dimensions of each sharding axis. Defaults to (1, -1, 1, 1).
		    sharding_axis_names (tp.Sequence[str], optional): Names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
		    shard_fns (tp.Optional[tp.Mapping[tuple, tp.Callable] | dict], optional): Sharding functions to use for the model. If None, auto-sharding is used if auto_shard_model is True. Defaults to None.
		    platform (tp.Optional[EasyDeLPlatforms], optional): platform to use for the model. Defaults to None.
				backend (tp.Optional[EasyDeLBackends], optional): backend to use for the model. Defaults to None.
		    config_kwargs (tp.Optional[tp.Mapping[str, Any] | EasyDeLBaseConfigDict], optional): Configuration keyword arguments to pass to the model config. Defaults to None.
		    auto_shard_model (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
		    partition_rules (tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec]]], optional): Custom partition rules for parameter sharding. If not None, shard_fns should also be provided. Defaults to None.
		    quantization_method (EasyDeLQuantizationMethods, optional): quantization_method to be used to quantize model weights. Defaults to None.
		    quantization_block_size (int): block size to be used for quantizing arrays (only for NF4).
		    bit_targeted_params (tp.Optional[List[str]], optional): List of parameter names to convert to 8-bit precision. If  None and 8bit is True, all kernels and embeddings are converted to 8-bit. Defaults to None.
		    from_torch (bool): whenever to load the model from transformers-pytorch.
		    **kwargs: Additional keyword arguments to pass to the model and config classes.

		Returns:
		    tp.Tuple[EasyDeLBaseModule, dict]: A tuple containing the EasyDeL model and the loaded and sharded
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
			return cls._from_torch_pretrained(
				pretrained_model_name_or_path=pretrained_model_name_or_path,
				param_dtype=param_dtype,
				dtype=dtype,
				shard_fns=shard_fns,
				auto_shard_model=auto_shard_model,
				precision=precision,
				backend=backend,
				platform=platform,
				partition_axis=partition_axis,
				quantization_method=quantization_method,
				quantization_block_size=quantization_block_size,
				partition_rules=partition_rules,
				sharding_axis_names=sharding_axis_names,
				sharding_axis_dims=sharding_axis_dims,
				config_kwargs=config_kwargs,
				device=device,
				shard_attention_computation=shard_attention_computation,
				**kwargs,
			)
		with jax.default_device(device):
			return cls._from_easydel_params(
				auto_shard_model=auto_shard_model,
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
				quantization_block_size=quantization_block_size,
				**kwargs,
			)


class AutoStateForCausalLM:
	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		device: tp.Optional[jax.Device] = None,
		dtype: jax.numpy.dtype = jax.numpy.float32,
		param_dtype: jax.numpy.dtype = jax.numpy.float32,
		precision: tp.Optional[jax.lax.Precision] = None,
		sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: tp.Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		shard_fns: tp.Optional[tp.Mapping[tuple, tp.Callable] | dict] = None,
		backend: tp.Optional[EasyDeLBackends] = None,
		platform: tp.Optional[EasyDeLPlatforms] = None,
		config_kwargs: tp.Optional[EasyDeLBaseConfigDict] = None,
		auto_shard_model: bool = False,
		partition_rules: tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec], ...]] = None,
		quantization_method: tp.Optional[EasyDeLQuantizationMethods] = None,
		quantization_block_size: int = 128,
		from_torch: tp.Optional[bool] = None,
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
		    sharding_axis_dims (tp.Sequence[int], optional): Dimensions of each sharding axis. Defaults to (1, -1, 1, 1).
		    sharding_axis_names (tp.Sequence[str], optional): Names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
		    shard_fns (tp.Optional[tp.Mapping[tuple, tp.Callable] | dict], optional): Sharding functions to use for the model. If None, auto-sharding is used if auto_shard_model is True. Defaults to None.
		    backend (tp.Optional[str], optional): Backend to use for the model. Defaults to None.
		    config_kwargs (tp.Optional[tp.Mapping[str, Any]], optional): Configuration keyword arguments to pass to the model config. Defaults to None.
		    auto_shard_model (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
		    partition_rules (tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec]]], optional): Custom partition rules for parameter sharding. If not None, shard_fns should also be provided. Defaults to None.
		    quantization_method (EasyDeLQuantizationMethods, optional): quantization_method to be used to quantize model weights. Defaults to None.
		    bit_targeted_params (tp.Optional[List[str]], optional): List of parameter names to convert to 8-bit precision. If  None and 8bit is True, all kernels and embeddings are converted to 8-bit. Defaults to None.
		    verbose_params (bool): whenever to log number of parameters in converting state.
		    safe (bool): whenever to use safetensors to load engine or parameters (requires engine or parameters to be saved with safe=True while saving them)
		    from_torch (bool): whenever to load the model from transformers-pytorch.
		    **kwargs: Additional keyword arguments to pass to the model and config classes.

		Returns:
		    EasyDeLState: containing the EasyDeL state and the loaded and sharded model parameters.
		"""
		model = AutoEasyDeLModelForCausalLM.from_pretrained(
			pretrained_model_name_or_path=pretrained_model_name_or_path,
			param_dtype=param_dtype,
			dtype=dtype,
			shard_fns=shard_fns,
			auto_shard_model=auto_shard_model,
			precision=precision,
			backend=backend,
			platform=platform,
			partition_axis=partition_axis,
			quantization_method=quantization_method,
			quantization_block_size=quantization_block_size,
			partition_rules=partition_rules,
			sharding_axis_names=sharding_axis_names,
			sharding_axis_dims=sharding_axis_dims,
			config_kwargs=config_kwargs,
			device=device,
			shard_attention_computation=shard_attention_computation,
			from_torch=from_torch,
			**kwargs,
		)
		return EasyDeLState.create(
			model=model,
			tx=None,
			init_opt_state=False,
			step=0,
		)
