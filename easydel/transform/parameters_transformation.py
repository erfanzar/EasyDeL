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
from typing import Callable, List, Mapping, Optional, Tuple

import jax
import transformers
from fjformer.checkpoint import get_dtype
from flax import traverse_util
from flax.traverse_util import flatten_dict
from jax import numpy as jnp
from tqdm.autonotebook import tqdm

from easydel.etils.etils import EasyDeLPlatforms, EasyDeLQuantizationMethods, get_logger
from easydel.transform.utils import jax2pt
from easydel.utils.quantizers import EasyQuantizer

logger = get_logger(__name__)


def float_tensor_to_dtype(tensor, dtype):
	"""
	The float_tensor_to_dtype function is used to convert a tensor's dtype to the specified dtype.

	:param tensor: Convert the tensor to a float dtype
	:param dtype: Convert the tensor to a specific dtype
	:return: A tensor with the specified dtype

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


class _DummyContextManager:
	def __enter__(self):
		pass

	def __exit__(self, exc_type, exc_value, traceback):
		pass


def process_tensor(
	key: str,
	tensor,
	embedding_layer_names: set,
	layernorm_names: set,
	rnn_based_or_rwkv: bool,
	lm_head_name: Optional[str],
	uses_tie_word_embedding: bool,
	dtype: jax.numpy.dtype,
) -> Optional[Tuple[tuple, jax.numpy.ndarray]]:
	new_key = key

	if any(layer_name in key for layer_name in embedding_layer_names):
		new_key = key[: -len(".weight")] + ".embedding"
	elif rnn_based_or_rwkv and ("time_mix_" in key or "time_" in key):
		tensor = tensor.reshape(-1)
	elif any(layer_norm in key for layer_norm in layernorm_names):
		new_key = key.replace(".weight", ".scale")
	elif "weight" in key:
		if len(tensor.shape) == 2:
			tensor = tensor.transpose(0, 1)
		elif len(tensor.shape) == 3:
			tensor = tensor.transpose(0, 2)
		new_key = key.replace(".weight", ".kernel")

	key_tuple = tuple(new_key.split("."))

	if uses_tie_word_embedding and lm_head_name:
		if key_tuple[0] == lm_head_name:
			return None
	# Convert tensor to JAX array
	if hasattr(tensor, "cpu"):  # Check if it's a PyTorch tensor
		if str(tensor.dtype) == "torch.bfloat16":
			tensor = tensor.float()
		array = jnp.asarray(tensor.cpu().detach().numpy()).astype(dtype)
	else:  # Assume it's already a numpy array
		array = jnp.array(tensor).astype(dtype)

	return key_tuple, array


def torch_dict_to_easydel_params(
	state_dict: dict,
	*,
	device: Optional[jax.Device] = None,
	embedding_layer_names: Optional[List[str]] = None,
	layernorm_names: Optional[List[str]] = None,
	rnn_based_or_rwkv: bool = False,
	shard_fns: Optional[Mapping[tuple, Callable]] = None,
	quantization_method: Optional[EasyDeLQuantizationMethods] = None,
	quantization_platform: Optional[EasyDeLPlatforms] = EasyDeLPlatforms.JAX,
	block_size: int = 256,
	params_pattern_selection: Optional[re.Pattern] = None,
	dtype: jax.numpy.dtype = jax.numpy.float16,
	verbose: bool = True,
	remove_state_dict: bool = False,
	lm_head_name: Optional[str] = None,
	uses_tie_word_embedding: bool = False,
	**kwargs,
) -> dict:
	"""
	The torch_dict_to_easydel_params function takes a torch model's state_dict and converts it to an easydel
	model's params. The function is designed to be used in conjunction with the load_huggingface function, which
	loads a huggingface model from disk. The embedding layer name must be specified as well as the device on which
	the conversion will take place.

	Args:
	    state_dict: Load the weights from a huggingface model
	    embedding_layer_names: List[str]: Identify the embedding layer in the huggingface model
	    device: Determine which device the model will be loaded on
	    layernorm_names: Replaces weight or kernel with (scale)
	    shard_fns: Optional[Mapping[tuple, Callable]]: Sharding Function to be used to shard model
	    quantization_method (EasyDeLQuantizationMethods, optional): quantization_method to be used to quantize model weights. Defaults to None.
	    quantization_platform (Optional[EasyDeLQuantizationMethods], optional): Platform to use for the weight quants. Defaults to None.
			block_size (int): blocksize for nf4 quantization.
	    params_pattern_selection: Optional[re.Pattern]: patter to use to find the parameters of the model which will
	    dtype: jax.numpy.dtype: Specify the data type of the tensors
	    rnn_based_or_rwkv: bool: rnn_based_or_rwkv is a conditioner  which decide whenever it finds a value in tree
	    verbose: bool: whenever to log sharding or converting process
	    remove_state_dict: bool : whether to remove state dict during  the transforming process
	be converted to 8bit format.
	that start with time_mix_ it will automatically reshape that for easydel use case

	Returns:
	    A dictionary of the weights and biases in a format that can be
	    used by flax (it's an UnFlattenDict)
	"""
	try:
		import torch

		_clear = torch.cuda.empty_cache if torch.cuda.is_available() else gc.collect
	except ModuleNotFoundError:
		_clear = gc.collect

	embedding_layer_names = set(embedding_layer_names or [])
	layernorm_names = set(layernorm_names or [])
	quantizer = None
	if quantization_method is not None:
		if params_pattern_selection is None:
			raise ValueError(
				"In case of quantizing parameters you should pass "
				"`params_pattern_selection` too, to tell the quantizer"
				" which parameters should be quantized."
			)
		quantizer = EasyQuantizer(
			quantization_method=quantization_method,
			block_size=block_size,
			quantization_platform=quantization_platform,
		)
	device = device or jax.devices()[0]

	ctx_m = jax.default_device(device) if shard_fns is None else _DummyContextManager()
	with ctx_m:
		total_items = len(state_dict)
		pbar = tqdm(total=total_items, disable=not verbose, desc="Converting Model")

		flax_dict = {}
		for key, tensor in state_dict.items():
			result = process_tensor(
				key,
				tensor,
				embedding_layer_names,
				layernorm_names,
				rnn_based_or_rwkv,
				lm_head_name,
				uses_tie_word_embedding,
				dtype,
			)
			if result is not None:
				key_tuple, jax_array = result
				if shard_fns and key_tuple in shard_fns:
					jax_array = shard_fns[key_tuple](jax_array)
				if (
					quantizer is not None
					and key_tuple[-1] != "embedding"
					and params_pattern_selection.search("/".join(key_tuple))
				):
					jax_array = quantizer(array=jax_array)
				flax_dict[key_tuple] = jax_array
			pbar.update(1)

		pbar.close()

		if remove_state_dict:
			del state_dict
			_clear()

		return traverse_util.unflatten_dict(flax_dict)


def easystate_to_torch(
	state,
	dtype=jnp.float16,
	transpose_needed=None,
	transpose_not_needed=None,
	select_params_field: bool = True,
	rnn_based_or_rwkv: bool = False,
):
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

	model_parameters = (
		flatten_dict(state.params["params"], sep=".")
		if select_params_field
		else flatten_dict(state.params, sep=".")
	)
	torch_state_dict = {}
	pbar = tqdm(
		model_parameters.items(), desc="Converting EasyDeLState to torch state_dict"
	)
	for key, tensor in pbar:
		if match_keywords(key, transpose_needed, transpose_not_needed):
			tensor = tensor.T
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


def easystate_to_huggingface_model(
	state,
	config,
	base_huggingface_module: transformers.PreTrainedModel,
	base_huggingface_module_kwarguments=None,
	dtype=jnp.float16,
	transpose_needed=None,
	transpose_not_needed=None,
	select_params_field: bool = True,
	rnn_based_or_rwkv: bool = False,
	auto_correct: bool = True,
	use_meta_torch: bool = True,
):
	if not rnn_based_or_rwkv and auto_correct:
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
		for k, v in state.unsafe_dict(config.__dict__).items():
			if not hasattr(config, k):
				setattr(config, k, v)

	state_dict = easystate_to_torch(
		state=state,
		dtype=dtype,
		transpose_needed=transpose_needed,
		transpose_not_needed=transpose_not_needed,
		select_params_field=select_params_field,
		rnn_based_or_rwkv=rnn_based_or_rwkv,
	)
	import torch

	if use_meta_torch:
		with torch.device("meta"):
			model = base_huggingface_module(
				config=config,
				**base_huggingface_module_kwarguments,
			)
		model.load_state_dict(state_dict, assign=True, strict=True)
	else:
		model = base_huggingface_module(
			config=config,
			**base_huggingface_module_kwarguments,
		)
		model.load_state_dict(state_dict, assign=True, strict=True)
	return model
