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
import os
import typing as tp
from contextlib import contextmanager

import jax
from fjformer.checkpoint import get_dtype
from jax import dlpack
from jax import numpy as jnp
from tqdm.autonotebook import tqdm

from easydel.etils.etils import get_logger

from .analyze_memory import SMPMemoryMonitor
from .traversals import flatten_dict, unflatten_dict

if tp.TYPE_CHECKING:
	from transformers import PreTrainedModel

	from easydel.infra.base_config import EasyDeLBaseConfig
	from easydel.infra.base_module import EasyDeLBaseModule

else:
	PreTrainedModel = tp.Any
	EasyDeLBaseModule = tp.Any
	EasyDeLBaseConfig = tp.Any

mem_ops = SMPMemoryMonitor(5)
logger = get_logger(__name__)


def float_tensor_to_dtype(tensor, dtype):
	"""
	The float_tensor_to_dtype function is used to convert a tensor's dtype to the specified dtype.

	"""
	if dtype is None or dtype == "":
		return tensor
	if isinstance(dtype, str):
		dtype = get_dtype(dtype)
	float_dtypes = (
		jax.numpy.bfloat16,
		jax.numpy.float16,
		jax.numpy.float32,
		jax.numpy.float64,
	)
	if getattr(tensor, "dtype", None) in float_dtypes:
		tensor = tensor.astype(dtype)
	return tensor


def convert_pytorch_tensor_to_jax(tensor, dtype):
	if "bfloat16" in str(tensor.dtype):
		tensor = tensor.float()
	return jnp.asarray(tensor.cpu().detach().numpy(), dtype=dtype)


def match_keywords(string, ts, ns):
	"""The match_keywords function takes a string, and two lists of strings.
	The first list is the &quot;must-have&quot; keywords, and the second list is the &quot;not-allowed&quot; keywords.
	It returns True if all the must-have keywords are in string, but none of not allowed are in it.

	Args:
	    string: Pass in the text that is being searched
	    ts: Specify the required keywords and ns is used to specify the
	        non-required keywords
	    ns: Specify a list of negative keywords

	Returns:
	    True if all the keywords in ts are present and none of the
	"""
	for t in ts:
		if t not in string:
			return False
	for n in ns:
		if n in string:
			return False
	return True


@contextmanager
def _dummy_context_manager():
	yield


def process_tensor(
	key: str, tensor: tp.Any, config: tp.Dict[str, tp.Any]
) -> tp.Optional[tp.Tuple[tuple, jax.numpy.ndarray]]:
	"""
	Process a single tensor and return its processed key and value.

	Args:
	    key: The parameter key
	    tensor: The tensor to process
	    config: Dictionary containing processing configuration

	Returns:
	    tp.Tuple of processed key tuple and JAX array, or None if tensor should be skipped
	"""
	new_key = key
	# Handle embedding layers
	if any(layer_name in key for layer_name in config["embedding_layer_names"]):
		new_key = f"{key[:-len('.weight')]}.embedding"

	# Handle RNN/RWKV specific tensors
	elif config["rnn_based_or_rwkv"] and any(x in key for x in ["time_mix_", "time_"]):
		tensor = tensor.reshape(-1)

	# Handle layer normalization

	elif any(layer_norm in key for layer_norm in config["layernorm_names"]):
		new_key = key.replace(".weight", ".scale")

	# Handle regular weights
	elif "weight" in key:
		ndim = len(tensor.shape)
		match ndim:
			case 2:
				# linear layers
				tensor = tensor.transpose(0, 1)
			case 3:
				# 1d conv layers
				tensor = tensor.transpose(0, 2)
			case 4:
				# 2d conv layers
				tensor = tensor.permute(2, 3, 1, 0)
			case 5:
				# 3d conv layers
				tensor = tensor.permute(2, 3, 4, 1, 0)
			case 6:
				# 4d conv layers
				tensor = tensor.permute(4, 5, 3, 2, 1, 0)
			case _:
				...
		new_key = key.replace(".weight", ".kernel")

	# Convert key string to tuple
	key_tuple = tuple(int(n) if n.isdigit() else n for n in new_key.split("."))

	# Skip if using tied embeddings and this is the language model head
	if config["uses_tie_word_embedding"] and config["lm_head_name"]:
		if key_tuple[0] == config["lm_head_name"]:
			return None

	# Convert tensor to JAX array
	array = convert_pytorch_tensor_to_jax(tensor, config["dtype"])

	return key_tuple, array


def torch_dict_to_easydel_params(
	state_dict: tp.Dict[str, tp.Any],
	*,
	device: tp.Optional[jax.Device] = None,
	embedding_layer_names: tp.Optional[tp.List[str]] = None,
	layernorm_names: tp.Optional[tp.List[str]] = None,
	rnn_based_or_rwkv: bool = False,
	shard_fns: tp.Optional[tp.Mapping[tuple, tp.Callable]] = None,
	dtype: jax.numpy.dtype = jnp.float16,
	verbose: bool = True,
	remove_state_dict: bool = False,
	lm_head_name: tp.Optional[str] = None,
	uses_tie_word_embedding: bool = False,
	**kwargs,
) -> tp.Dict[str, tp.Any]:
	"""
	Convert PyTorch state dict to EasyDel parameter format.

	Args:
	    state_dict: PyTorch state dictionary
	    device: JAX device to use
	    embedding_layer_names: Names of embedding layers
	    layernorm_names: Names of layer normalization layers
	    rnn_based_or_rwkv: Whether model is RNN-based or RWKV
	    shard_fns: tp.Mapping of parameter names to sharding functions
	    block_size: Size of processing blocks
	    params_pattern_selection: Regex pattern for parameter selection
	    dtype: Target dtype for parameters
	    verbose: Whether to show progress bar
	    remove_state_dict: Whether to delete state_dict after conversion
	    lm_head_name: Name of language model head
	    uses_tie_word_embedding: Whether model uses tied embeddings
	    **kwargs: Additional arguments

	Returns:
	    Dictionary of converted parameters in EasyDel format
	"""
	try:
		import torch

		_clear = torch.cuda.empty_cache if torch.cuda.is_available() else gc.collect
	except ModuleNotFoundError:
		_clear = gc.collect

	# Configuration dictionary
	config = {
		"embedding_layer_names": set(embedding_layer_names or []),
		"layernorm_names": set(layernorm_names or []),
		"rnn_based_or_rwkv": rnn_based_or_rwkv,
		"lm_head_name": lm_head_name,
		"uses_tie_word_embedding": uses_tie_word_embedding,
		"dtype": dtype,
	}

	device = device or jax.devices()[0]
	ctx_m = jax.default_device(device) if shard_fns is None else _dummy_context_manager()

	with ctx_m:
		flax_dict = {}
		with tqdm(
			total=len(state_dict), disable=not verbose, desc="Converting Model"
		) as pbar:
			for key, tensor in state_dict.items():
				try:
					result = process_tensor(key, tensor, config)
					if result is not None:
						key_tuple, jax_array = result
						if shard_fns and key_tuple in shard_fns:
							jax_array = shard_fns[key_tuple](jax_array)
						flax_dict[key_tuple] = jax_array
				except Exception as e:
					raise e
					print(f"Error processing key {key}: {str(e)}")
				pbar.update(1)

		if remove_state_dict:
			del state_dict
			_clear()

		return unflatten_dict(flax_dict)


def module_to_torch(
	module: EasyDeLBaseModule,
	dtype: jnp.dtype = jnp.float16,
	transpose_needed: tp.Optional[tp.Dict] = None,
	transpose_not_needed: tp.Optional[tp.Dict] = None,
	rnn_based_or_rwkv: bool = False,
):
	if dtype is None:
		dtype = module.param_dtype
	if transpose_needed is None:
		transpose_needed = ["kernel"]
	if transpose_not_needed is None:
		transpose_not_needed = ["none"]

	def match_keywords(string, do_transpose, dont_transpose):
		for dtr in do_transpose:
			if dtr not in string:
				return False
		for ntr in dont_transpose:
			if ntr in string:
				return False
		return True

	graphtree = unflatten_dict(module.parameters)
	model_parameters = flatten_dict(graphtree, sep=".")
	torch_state_dict = {}
	pbar = tqdm(
		model_parameters.items(), desc="Converting EasyDeLBaseModule to torch state_dict"
	)
	for key, tensor in pbar:
		if tensor is None:
			continue
		if match_keywords(key, transpose_needed, transpose_not_needed):
			ndim = tensor.ndim
			match ndim:
				case 2:
					# linear layers
					tensor = jnp.swapaxes(tensor, 0, 1)  # Same as PT->JAX since it's symmetric
				case 3:
					# 1d conv layers
					tensor = jnp.swapaxes(tensor, 0, 2)  # Same as PT->JAX since it's symmetric
				case 4:
					# 2d conv layers
					tensor = jnp.transpose(tensor, (3, 2, 0, 1))  # Reverse of (2,3,1,0)
				case 5:
					# 3d conv layers
					tensor = jnp.transpose(tensor, (4, 3, 0, 1, 2))  # Reverse of (2,3,4,1,0)
				case 6:
					# 4d conv layers
					tensor = jnp.transpose(tensor, (5, 4, 3, 2, 0, 1))  # Reverse of (4,5,3,2,1,0)
				case _:
					...
		elif rnn_based_or_rwkv and ("time_mix_" in key or "time_" in key):
			tensor = tensor.reshape(1, 1, -1)
		if tensor.dtype != get_dtype(dtype):
			tensor = tensor.astype(get_dtype(dtype))  # ignores double allocation on GPUs
		key = (
			key.replace(".kernel", ".weight")
			.replace(".embedding", ".weight")
			.replace(".scale", ".weight")
		)

		torch_state_dict[key] = jax2pt(tensor)
	return torch_state_dict


def module_to_huggingface_model(
	module: EasyDeLBaseModule,
	config: EasyDeLBaseConfig,
	base_huggingface_module: PreTrainedModel,
	base_huggingface_module_kwarguments: tp.Optional[tp.Dict] = None,
	dtype=jnp.float16,
	transpose_needed: tp.Optional[tp.List] = None,
	transpose_not_needed=None,
	rnn_based_or_rwkv: bool = False,
	auto_correct: bool = True,
	use_meta_torch: bool = True,
):
	if not rnn_based_or_rwkv and auto_correct:
		import transformers

		if isinstance(base_huggingface_module, transformers.RwkvForCausalLM) or isinstance(
			base_huggingface_module, transformers.RwkvModel
		):
			logger.warning(
				"Rnn Based Model detected 'setting `rnn_based_or_rwkv = True`' for correct weight handling"
			)
			rnn_based_or_rwkv = True
	if base_huggingface_module_kwarguments is None:
		base_huggingface_module_kwarguments = {}

	if auto_correct:
		for k, v in config.__dict__.items():
			if not hasattr(config, k):
				setattr(config, k, v)

	state_dict = module_to_torch(
		module=module,
		dtype=dtype,
		transpose_needed=transpose_needed,
		transpose_not_needed=transpose_not_needed,
		rnn_based_or_rwkv=rnn_based_or_rwkv,
	)
	import torch

	if use_meta_torch:
		with torch.device("meta"):
			model = base_huggingface_module(
				config=config,
				**base_huggingface_module_kwarguments,
			)
		model.load_state_dict(
			state_dict,
			assign=True,
			strict=True,
		)
	else:
		model = base_huggingface_module(
			config=config,
			**base_huggingface_module_kwarguments,
		)
		model.load_state_dict(
			state_dict,
			assign=True,
			strict=True,
		)
	return model


def jax2pt(x: jax.Array):
	from torch import cuda
	from torch.utils import dlpack as dlpack_pt

	_jax_device = list(x.devices())[0].platform
	cpu_force = not cuda.is_available()
	if (
		_jax_device in ["cpu", "gpu"]
		and not cpu_force
		and not bool(os.environ.get("EASYDEL_FORCE_TORCH_USE_CPU", "false"))
	):
		dl_pack_jax = dlpack.to_dlpack(
			x,
			stream=True if (_jax_device == "gpu" and not cpu_force) else None,
			src_device=list(x.devices())[0],
		)
	else:
		device = os.environ.get("EASYDEL_PERFRED_HOST_COPY", "cpu")
		if device.lower() == "none":
			device = None  # Auto JAX Select
		perfred_host = jax.devices(device)[
			int(os.environ.get("EASYDEL_PERFRED_HOST_COPY_IDEX", "0"))
		]
		x = jax.device_get(x)  # make sure it's local
		x = jax.device_put(x, perfred_host)
		dl_pack_jax = dlpack.to_dlpack(
			x,
			stream=None,
		)
	return dlpack_pt.from_dlpack(dl_pack_jax)


def pt2jax(x, transpose_raw: tp.Optional[tuple] = None):
	return jax.numpy.asarray(x.detach().cpu().numpy())
