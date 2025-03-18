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

import contextlib
import functools
import gc
import os
import typing as tp
import warnings

import jax
import jax.extend
import numpy as np
from jax import dlpack
from jax import numpy as jnp
from tqdm.autonotebook import tqdm

from easydel.utils.helpers import check_bool_flag, get_logger

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

EASYDEL_PERFRED_HOST_COPY_INDEX = int(os.getenv("EASYDEL_PERFRED_HOST_COPY_INDEX", "0"))
EASYDEL_PERFRED_HOST_COPY = str(os.getenv("EASYDEL_PERFRED_HOST_COPY", "cpu")).lower()

EASYDEL_PERFRED_HOST_COPY = (
	None if EASYDEL_PERFRED_HOST_COPY == "none" else EASYDEL_PERFRED_HOST_COPY
)


def get_dtype(dtype):
	if isinstance(dtype, str):
		dtype = {
			"bf16": jnp.bfloat16,
			"bfloat16": jnp.bfloat16,
			"fp16": jnp.float16,
			"float16": jnp.float16,
			"fp32": jnp.float32,
			"float32": jnp.float32,
			"fp64": jnp.float64,
			"float64": jnp.float64,
			"fp8": jnp.float8_e5m2,
			"fp8_e4m3fn": jnp.float8_e4m3fn,
			"fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
			"fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
			"fp8_e5m2": jnp.float8_e5m2,
			"fp8_e5m2fnuz": jnp.float8_e5m2fnuz,
			"float8_e4m3fn": jnp.float8_e4m3fn,
			"float8_e4m3fnuz": jnp.float8_e4m3fnuz,
			"float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
			"float8_e5m2": jnp.float8_e5m2,
			"float8_e5m2fnuz": jnp.float8_e5m2fnuz,
		}[dtype]
	return dtype


def float_tensor_to_dtype(tensor, dtype):
	if dtype is None or dtype == "":
		return tensor
	if isinstance(dtype, str):
		dtype = get_dtype(dtype)
	float_dtypes = (
		jnp.bfloat16,
		jnp.float16,
		jnp.float32,
		jnp.float64,
		jnp.float8_e4m3fn,
		jnp.float8_e4m3fnuz,
		jnp.float8_e4m3b11fnuz,
		jnp.float8_e5m2,
		jnp.float8_e5m2fnuz,
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


def process_tensor(
	key: str, tensor: tp.Any, config: tp.Dict[str, tp.Any]
) -> tp.Optional[tp.Tuple[tuple, jnp.ndarray]]:
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
		new_key = f"{key[: -len('.weight')]}.embedding"

	# Handle layer normalization

	elif any(layer_norm in key for layer_norm in config["layernorm_names"]):
		new_key = key.replace(".weight", ".scale")

	# Handle regular weights
	elif "weight" in key:
		ndim = len(tensor.shape)
		match ndim:
			case 2:
				# linear layers
				tensor = tensor.permute(1, 0)
			case 3:
				# 1d conv layers
				tensor = tensor.permute(2, 1, 0)
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
	shard_fns: tp.Optional[tp.Mapping[tuple, tp.Callable]] = None,
	dtype: jnp.dtype = jnp.float16,
	verbose: bool = True,
	callback: tp.Optional[tp.Callable[[jax.Array, tuple], jax.Array]] = None,
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
	    shard_fns: tp.Mapping of parameter names to sharding functions
	    block_size: Size of processing blocks
	    params_pattern_selection: Regex pattern for parameter selection
	    dtype: Target dtype for parameters
	    verbose: Whether to show progress bar
			callback: callback for tensors after they are converted to a jax array.
	    remove_state_dict: Whether to delete state_dict after conversion
	    lm_head_name: Name of language model head
	    uses_tie_word_embedding: Whether model uses tied embeddings
	    **kwargs: Additional arguments

	Returns:
	    Dictionary of converted parameters in EasyDel format
	"""
	try:
		import torch  # type:ignore #noqa

		_clear = torch.cuda.empty_cache if torch.cuda.is_available() else gc.collect
	except ModuleNotFoundError:
		_clear = gc.collect

	# Configuration dictionary
	config = {
		"embedding_layer_names": set(embedding_layer_names or []),
		"layernorm_names": set(layernorm_names or []),
		"lm_head_name": lm_head_name,
		"uses_tie_word_embedding": uses_tie_word_embedding,
		"dtype": dtype,
	}
	cmg = (
		jax.default_device(device)
		if device is not None and shard_fns is None
		else contextlib.nullcontext()
	)

	with cmg:
		flax_dict = {}
		with tqdm(
			total=len(state_dict),
			disable=not verbose,
			desc="Converting Model",
		) as pbar:
			for key, tensor in state_dict.items():
				try:
					result = process_tensor(key, tensor, config)
					if result is not None:
						key_tuple, jax_array = result
						if shard_fns and key_tuple in shard_fns:
							jax_array = shard_fns[key_tuple](jax_array)
						if callback is not None:
							jax_array = callback(jax_array, key_tuple)
						flax_dict[key_tuple] = jax_array
				except Exception as e:
					print(f"Error processing key {key}: {str(e)}")
				pbar.update(1)

		if remove_state_dict:
			del state_dict
			_clear()

		return unflatten_dict(flax_dict)


def module_to_torch(module: EasyDeLBaseModule, dtype: jnp.dtype = jnp.float16):
	if dtype is None:
		dtype = module.param_dtype

	graphtree = unflatten_dict(module.parameters)
	model_parameters = flatten_dict(graphtree, sep=".")
	torch_state_dict = {}
	pbar = tqdm(
		model_parameters.items(),
		desc="Converting EasyDeLBaseModule to torch state_dict",
	)
	for key, tensor in pbar:
		if tensor is None:
			continue

		if tensor.dtype != get_dtype(dtype):
			if hasattr(tensor, "materialize"):
				tensor = tensor.materialize()
			if hasattr(tensor, "value") and hasattr(tensor.value, "materialize"):
				tensor = tensor.value.materialize()
			tensor = tensor.astype(get_dtype(dtype))

		tensor = jax2pt(jax.block_until_ready(tensor))

		if key.endswith(".kernel"):
			match tensor.ndim:
				case 2:
					tensor = tensor.permute(1, 0)
				case 3:
					tensor = tensor.permute(2, 1, 0)
				case 4:
					tensor = tensor.permute(3, 2, 0, 1)
				case 5:
					tensor = tensor.permute(4, 3, 0, 1, 2)
				case 6:
					tensor = tensor.permute(5, 4, 3, 2, 0, 1)
				case _:
					...

		key = (
			key.replace(".kernel", ".weight")
			.replace(".embedding", ".weight")
			.replace(".scale", ".weight")
		)

		torch_state_dict[key] = tensor
	return torch_state_dict


def module_to_huggingface_model(
	module: EasyDeLBaseModule,
	config: EasyDeLBaseConfig,
	base_huggingface_module: PreTrainedModel,
	base_huggingface_module_kwarguments: tp.Optional[tp.Dict] = None,
	dtype: jnp.dtype = jnp.float16,
	use_meta_torch: bool = True,
	**kw,
):
	if base_huggingface_module_kwarguments is None:
		base_huggingface_module_kwarguments = {}

	state_dict = module_to_torch(module=module, dtype=dtype)
	import torch  # type:ignore

	base_config = base_huggingface_module.config_class.from_dict(config.to_dict())
	ctxm = torch.device("meta") if use_meta_torch else contextlib.nullcontext()
	with ctxm:
		model: torch.nn.Module = base_huggingface_module(
			config=base_config,
			**base_huggingface_module_kwarguments,
		)
		key_shape_checks = {
			k: v.shape for k, v in model.state_dict().items() if hasattr(v, "shape")
		}
		if len(list(key_shape_checks.keys())) != len(list(state_dict.keys())):
			warnings.warn(
				"There might be an issue with converted `state_dict`.", stacklevel=1
			)
		for key, shape in key_shape_checks.items():
			if state_dict[key].shape != shape:
				warnings.warn(f"Shape conflict at {key}.", stacklevel=1)
		model.load_state_dict(state_dict, assign=True, strict=True)
	return model


@functools.lru_cache
def get_torch():
	import torch  # type:ignore

	return torch


def jax2pt(x: jax.Array):
	if check_bool_flag("EASY_SAFE_TRANSFER", True):
		x = jax.device_get(x)
		return get_torch().from_numpy(np.array(x.tolist(), dtype=x.dtype))
	else:
		# This one causes a lot of funny bugs, where weights in state_dict are same (both in cpp and python)
		#  but estimated correct elements are ~85%
		from torch import cuda  # type:ignore
		from torch.utils import dlpack as dlpack_pt  # type:ignore

		platform = jax.extend.backend.get_backend()
		cpu_force = not cuda.is_available()
		if (
			platform in ["cpu", "gpu"]
			and not cpu_force
			and not check_bool_flag("EASYDEL_FORCE_TORCH_USE_CPU", False)
		):
			dl_pack_jax = dlpack.to_dlpack(
				x,
				stream=True if (platform == "gpu" and not cpu_force) else None,
				src_device=list(x.devices())[0],
			)
		else:
			dl_pack_jax = dlpack.to_dlpack(
				jax.device_put(
					jax.device_get(x),
					jax.devices(EASYDEL_PERFRED_HOST_COPY)[EASYDEL_PERFRED_HOST_COPY_INDEX],
				),
				stream=None,
			)
		return dlpack_pt.from_dlpack(dl_pack_jax)


def pt2jax(x):
	return jnp.asarray(x.detach().cpu().numpy())
