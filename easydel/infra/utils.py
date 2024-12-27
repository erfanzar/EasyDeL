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
import inspect
import re
import types
import typing as tp
import warnings
from functools import partial

import fjformer
import flax
import flax.core
import jax
import jax.experimental
import jax.tree_util
from aqt.jax.v2 import config as q_config
from aqt.jax.v2.flax import aqt_flax as q_flax
from einops import rearrange
from flax import nnx as nn
from tqdm.auto import tqdm

from easydel.etils.errors import EasyDeLBlockWiseFFNError
from easydel.etils.etils import (
	AVAILABLE_SPARSE_MODULE_TYPES,
	EasyDeLGradientCheckPointers,
	EasyDeLQuantizationMethods,
	get_logger,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.utils.traversals import flatten_dict, unflatten_dict

from .base_config import EasyMethod

warnings.filterwarnings(
	"ignore",
	message="Primitive dynamic_update_slice was not handled by class",
)
logger = get_logger(__name__)


def quick_gelu(x):
	return x * jax.nn.sigmoid(1.702 * x)


ACT2FN = {
	"gelu": partial(nn.gelu, approximate=False),
	"relu": nn.relu,
	"silu": nn.swish,
	"swish": nn.swish,
	"gelu_new": partial(nn.gelu, approximate=True),
	"gelu_pytorch_tanh": partial(nn.gelu, approximate=True),
	"tanh": nn.tanh,
	"sigmoid": nn.sigmoid,
	"leaky_relu": partial(nn.leaky_relu, negative_slope=0.01),
	"glu": nn.glu,
	"elu": nn.elu,
	"softmax": nn.softmax,
	"quick_gelu": quick_gelu,
}

ROPE_TYPES = tp.Optional[
	tp.Literal[
		"none",
		"linear",
		"dynamic",
		"yarn",
		"su",
		"llama3",
		"longrope",
	]
]


with_sharding_constraint = fjformer.with_sharding_constraint


def canonicalize_dtype(
	*args,
	dtype: tp.Optional[jax.numpy.dtype] = None,
	inexact: bool = True,
) -> jax.numpy.dtype:
	"""Canonicalize an optional dtype to the definitive dtype.

	If the ``dtype`` is None this function will infer the dtype. If it is not
	None it will be returned unmodified or an exceptions is raised if the dtype
	is invalid.
	from the input arguments using ``jnp.result_type``.

	Args:
	  *args: JAX array compatible values. None values
	    are ignored.
	  dtype: tp.Optional dtype override. If specified the arguments are cast to
	    the specified dtype instead and dtype inference is disabled.
	  inexact: When True, the output dtype must be a subdtype
	  of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
	  is useful when you want to apply operations that don'position_ids work directly on
	  integers like taking a mean for example.
	Returns:
	  The dtype that *args should be cast to.
	"""
	if dtype is None:
		args_filtered = [jax.numpy.asarray(x) for x in args if x is not None]
		dtype = jax.numpy.result_type(*args_filtered)
		if inexact and not jax.numpy.issubdtype(dtype, jax.numpy.inexact):
			dtype = jax.numpy.promote_types(jax.numpy.float32, dtype)
	if inexact and not jax.numpy.issubdtype(dtype, jax.numpy.inexact):
		raise ValueError(f"Dtype must be inexact: {dtype}")
	return dtype


def get_gradient_checkpoint_policy(name):
	"""
	The get_gradient_checkpoint_policy function is a helper function that returns the gradient checkpoint policy
	specified by the name parameter.
	"""
	gradients = dict(
		everything_saveable=jax.checkpoint_policies.everything_saveable,
		nothing_saveable=jax.checkpoint_policies.nothing_saveable,
		dots_saveable=jax.checkpoint_policies.dots_saveable,
		checkpoint_dots=jax.checkpoint_policies.checkpoint_dots,
		dots_with_no_batch_dims_saveable=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
		checkpoint_dots_with_no_batch_dims=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
		save_anything_except_these_names=jax.checkpoint_policies.save_anything_except_these_names,
		save_any_names_but_these=jax.checkpoint_policies.save_any_names_but_these,
		save_only_these_names=jax.checkpoint_policies.save_only_these_names,
		save_from_both_policies=jax.checkpoint_policies.save_from_both_policies,
	)
	return gradients[name]


def add_start_docstrings(*docstr):
	"""The add_start_docstrings function is a decorator that adds the docstrings to the beginning of a function.
	The add_start_docstrings function takes in an arbitrary number of strings and returns a decorator.
	The returned decorator takes in one argument, fn, which is assumed to be a function. The docstring for fn is set equal to
	the concatenation of all the strings passed into add_start_docstrings plus (if it exists) the original docstring for fn.

	Args:
	    *docstr: Pass in a variable number of arguments to the function

	Returns:
	    A decorator that adds the docstrings to the function
	"""

	def docstring_decorator(fn):
		fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
		return fn

	return docstring_decorator


def get_dot_general_by_bits(
	bits: tp.Optional[int] = None,
	mode: tp.Literal["train", "serve", "convert"] = EasyMethod.TRAIN,
) -> dict:
	"""The get_general_dot function is a helper function that returns a q_flax.QDotGeneral object
	with the specified number of bits for forward and backward passes. If no bits are specified,
	the function returns None.

	Args:
	    bits: tp.Optional[int]: Specify the number of bits for quantization
	    mode: EasyMethod: Specify the use of model to init the QDot
	        Method for (e.q TRAIN,SERVE,...)

	Returns:
	    A dict that contain dot_general_cls
	"""
	if mode == EasyMethod.TRAIN:
		rhs_quant_mode = q_flax.QuantMode.TRAIN
	elif mode == EasyMethod.EVAL or mode == EasyMethod.SERVE:
		rhs_quant_mode = q_flax.QuantMode.SERVE
	elif mode == EasyMethod.CONVERT:
		rhs_quant_mode = q_flax.QuantMode.CONVERT
	else:
		raise ValueError("Unknown Quant Method for EasyMethod")
	if bits is not None:
		return {
			"dot_general_cls": functools.partial(
				q_flax.AqtDotGeneral,
				cfg=q_config.fully_quantized(fwd_bits=bits, bwd_bits=bits),
				rhs_quant_mode=rhs_quant_mode,
			)
		}
	return {}  # empty just in case of not getting any error


def block_wise_ffn(remat_ffn, inputs, chunk_size: int, deterministic: bool):
	generating = inputs.shape[1] == 1
	try:
		if generating:
			return remat_ffn(inputs)
		else:
			inputs = rearrange(inputs, "b (c n) d -> b c n d", c=chunk_size)

			def scan_ffn(remat_ffn_, carry, hidden_states):
				outputs = remat_ffn_(hidden_states)
				return carry, outputs

			scan_axis = inputs.ndim - 2
			_, output = nn.scan(
				scan_ffn,
				variable_broadcast="params",
				split_rngs={"params": False, "dropout": True},
				in_axes=scan_axis,
				out_axes=scan_axis,
			)(remat_ffn, None, inputs)
			output = rearrange(output, "b c n d -> b (c n) d")
			return output
	except Exception as e:
		raise EasyDeLBlockWiseFFNError(
			"You Are using BlockWise FFN from near-infinite-context length paper and you might be passing "
			"input arguments in wrong way in case that you don'position_ids want to use this just pass `use_scan_mlp=False` in "
			"model config or in config_kwargs in AutoEasyDeLModelForCausalLM or change `scan_mlp_chunk_size` "
			f"in configs for more information read Docs.\nOriginal Error\n{e}"
		) from e


def control_mlp_sharding(x: jax.Array, partition_axis: PartitionAxis):
	"""
	this functions is disabled for now, it will cause breakdown and incorrect computation on gpu with CU lower than 7.5
	"""
	return x


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


def quantize_linear_layers(
	model: nn.Module,
	/,
	*,
	method: tp.Optional[EasyDeLQuantizationMethods] = None,
	block_size: int = 256,
	quantization_pattern: tp.Optional[str] = None,
	verbose: bool = True,
) -> nn.Module:
	"""
	Quantize parameters to requested precision, excluding specified layers.

	Args:
	    model: The model to quantize.
	    method (EasyDeLQuantizationMethods): quantization method for params.
	    quantization_pattern (str): re pattern for layers to be quantized.
	    verbose (bool): whenever to use tqdm for logging stuff.

	Returns:
	    Quantized parameters in the same structure as the input.
	"""
	if method == EasyDeLQuantizationMethods.NONE or method is None:
		return model

	from easydel.layers.quantization import Linear8bit, LinearNF4
	from easydel.utils.graph_utils import (
		get_module_from_path,
		iter_module_search,
		set_module_from_path,
	)

	quantizer: Linear8bit = {
		EasyDeLQuantizationMethods.NF4: LinearNF4,
		EasyDeLQuantizationMethods.A8BIT: Linear8bit,
		EasyDeLQuantizationMethods.A4Q: LinearNF4,
		EasyDeLQuantizationMethods.A8Q: Linear8bit,
	}.get(method, None)
	if quantizer is None:
		raise NotImplementedError("Requested Quantizer is not Supported")
	if quantization_pattern is None:
		quantization_pattern = ".*"

	if hasattr(model, "config"):
		model.config.quantization_method = method
		model.config.quantization_block_size = block_size
		model.config.quantization_pattern = quantization_pattern

	pattern = re.compile(quantization_pattern)

	with tqdm(
		total=len([p[0] for p in iter_module_search(model, nn.Linear)]),
		desc=f"Quantizing to {method}",
		disable=not verbose,
	) as pbar:
		for path, _ in iter_module_search(model, nn.Linear):
			if pattern.search(".".join([str(p) for p in path])):
				set_module_from_path(
					model=model,
					path=path,
					new_value=quantizer.from_linear(
						linear=get_module_from_path(model=model, path=path),
						rngs=None,
						block_size=block_size,
					),
				)
			pbar.update(1)

	return model


def apply_lora_to_layers(
	model: nn.Module,
	/,
	*,
	lora_rank: int,
	lora_pattern: tp.Optional[str] = None,
	verbose: bool = True,
	rngs: tp.Optional[nn.Rngs] = None,
) -> nn.Module:
	"""
	Applies LoRA (Low-Rank Adaptation) to specified linear layers within a model.

	Args:
	    model: The EasyDeL model to modify.
	    lora_rank: The rank of the LoRA adapters.
	    lora_pattern: A regular expression pattern to match the names of
	                  modules to which LoRA should be applied. Defaults to ".*" (all linear layers).
	    verbose: Whether to display a progress bar.
	    rngs:  A `flax.nnx.Rngs` instance for random number generation. If None, initializes with a seed of 0.

	Returns:
	    The modified model with LoRA applied to the specified layers.
	"""
	from easydel.utils.graph_utils import (
		get_module_from_path,
		iter_module_search,
		set_module_from_path,
	)

	if lora_pattern is None:
		lora_pattern = ".*"
	if rngs is None:
		rngs = nn.Rngs(0)
	pattern = re.compile(lora_pattern)

	with tqdm(
		total=len([p[0] for p in iter_module_search(model, nn.Linear)]),
		desc="Applying LoRA",
		disable=not verbose,
	) as pbar:
		for path, _ in iter_module_search(model, nn.Linear):
			if pattern.search(".".join([str(p) for p in path])):
				base_module: nn.Linear = get_module_from_path(model=model, path=path)
				set_module_from_path(
					model=model,
					path=path,
					new_value=nn.LoRA(
						base_module=base_module,
						rngs=rngs,
						dtype=base_module.dtype,
						param_dtype=base_module.param_dtype,
						in_features=base_module.in_features,
						lora_rank=lora_rank,
						out_features=base_module.out_features,
					),
				)
			pbar.update(1)

	return model


def unwrap_lora_to_layers(
	model: nn.Module,
	/,
	*,
	verbose: bool = True,
) -> nn.Module:
	"""
	UnWrap LoRA (Low-Rank Adaptation) from specified linear layers within a model.
	"""
	from easydel.utils.graph_utils import (
		get_module_from_path,
		iter_module_search,
		set_module_from_path,
	)

	with tqdm(
		total=len([p[0] for p in iter_module_search(model, nn.Linear)]),
		desc="Unwarping LoRA Layers",
		disable=not verbose,
	) as pbar:
		for path, _ in iter_module_search(model, nn.LoRA):
			base_module: nn.LoRA = get_module_from_path(model=model, path=path)
			with jax.default_matmul_precision("float32"):
				base_module.base_module.kernel.value = (
					base_module.base_module.kernel.value
					+ base_module.lora_a.value @ base_module.lora_b.value
				)
			del base_module.lora_a, base_module.lora_b
			set_module_from_path(
				model=model,
				path=path,
				new_value=base_module.base_module,
			)
		pbar.update(1)

	return model


def print_pytree(pytree):
	jax.tree_util.tree_map_with_path(
		lambda p, v: print(
			f"{fjformer.tree_path_to_string(p,'.')}: dtype:{v.dtype}, shape:{v.shape}"
		),
		pytree,
	)


def apply_sparsity_to_params(
	params: tp.Union[tp.Dict[str, tp.Any], tp.Any],
	sparsify_module: AVAILABLE_SPARSE_MODULE_TYPES = "bcoo",
	verbose: bool = True,
) -> tp.Union[tp.Dict[str, tp.Any], tp.Any]:
	its_frozen = isinstance(params, flax.core.FrozenDict)
	flatten = is_flatten(params)
	if not flatten:
		params = flatten_dict(params)
	from jax.experimental import sparse

	sparser = {
		"bcoo": sparse.BCOO,
		"bcsr": sparse.BCSR,
		"coo": sparse.COO,
		"csr": sparse.CSR,
	}.get(sparsify_module, None)
	assert sparser is not None, f"unkown type of sparser {sparsify_module}"

	def filter_params(path, array):
		layer_name = ".".join(path[0].key)
		if layer_name.endswith("kernel") and 4 > array.ndim > 1:
			array = sparser.fromdense(array)
		return array

	total_params = len(jax.tree_util.tree_leaves(params))
	with tqdm(
		total=total_params,
		desc=f"{sparsify_module.capitalize()}",
		disable=not verbose,
	) as pbar:

		def _with_progress(path, array):
			pbar.set_postfix_str(".".join(path[0].key))
			result = filter_params(path, array)
			pbar.update(1)
			return result

		params = jax.tree_util.tree_map_with_path(_with_progress, params)

	if not flatten:
		params = unflatten_dict(params)
	if its_frozen:
		return flax.core.FrozenDict(params)
	return params


def extract_static_parameters(module):
	"""
	Extract static_argnums for specified parameters across functions in a module.

	Args:
	    module (types.ModuleType): The module to inspect

	Returns:
	    dict: A dictionary mapping function names to their static parameter indices
	"""

	# Predefined list of parameters to check for static status
	target_params = [
		"causal_mask",
		"frequencies",
		"output_attentions",
		"output_hidden_states",
		"output_router_logits",
	]
	obj = getattr(module, "__call__", None)  # noqa
	if isinstance(obj, (types.FunctionType, types.MethodType)):
		static_args = ()
		signature = inspect.signature(obj)
		for idx, (param_name, param) in enumerate(signature.parameters.items()):
			if param_name in target_params:
				static_args += (idx,)
		return static_args
	return None


def auto_remat(
	*modules,
	policy: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
	prevent_cse: bool = True,
):
	if policy == EasyDeLGradientCheckPointers.NONE:
		return modules
	if isinstance(policy, str):
		policy = get_gradient_checkpoint_policy(policy)
	outs = ()
	for module in modules:
		assert issubclass(module, nn.Module)
		static_argnums = extract_static_parameters(module=module)
		if static_argnums is None:
			static_argnums = ()

		module.__call__ = nn.remat(
			f=module.__call__,
			prevent_cse=prevent_cse,
			static_argnums=static_argnums,
			policy=policy,
		)
		outs += (module,)
	return outs


if tp.TYPE_CHECKING:
	from transformers import (
		BaseImageProcessor,
		FeatureExtractionMixin,
		PreTrainedTokenizerBase,
		ProcessorMixin,
	)

	ProcessingClassType = tp.Optional[
		tp.Union[
			PreTrainedTokenizerBase,
			BaseImageProcessor,
			FeatureExtractionMixin,
			ProcessorMixin,
		]
	]
else:
	ProcessingClassType = tp.Any
