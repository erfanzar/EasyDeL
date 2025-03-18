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
from contextlib import contextmanager
from functools import lru_cache, partial
from typing import List, Set

import flax
import flax.core
import jax
import jax.experimental
import jax.tree_util
import numpy as np
from eformer.escale import PartitionAxis, with_sharding_constraint
from einops import rearrange
from flax import nnx as nn
from jax.sharding import PartitionSpec
from tqdm.auto import tqdm

from easydel.utils.helpers import get_logger
from easydel.utils.traversals import flatten_dict, unflatten_dict

from .base_config import EasyMethod
from .errors import EasyDeLBlockWiseFFNError
from .etils import (
	AVAILABLE_SPARSE_MODULE_TYPES,
	EasyDeLGradientCheckPointers,
	EasyDeLQuantizationMethods,
)

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


with_sharding_constraint = with_sharding_constraint


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
	if bits is not None:
		try:
			from aqt.jax.v2 import config as q_config  # type: ignore
			from aqt.jax.v2.flax import aqt_flax as q_flax  # type: ignore
		except ModuleNotFoundError as e:
			raise ModuleNotFoundError(
				"No module named `aqt` has been found, please "
				"install aqt before using bits option in EasyDeL"
			) from e
		if mode == EasyMethod.TRAIN:
			rhs_quant_mode = q_flax.QuantMode.TRAIN
		elif mode == EasyMethod.EVAL or mode == EasyMethod.SERVE:
			rhs_quant_mode = q_flax.QuantMode.SERVE
		elif mode == EasyMethod.CONVERT:
			rhs_quant_mode = q_flax.QuantMode.CONVERT
		else:
			raise ValueError("Unknown Quant Method for EasyMethod")

			return {
				"dot_general_cls": functools.partial(
					q_flax.AqtDotGeneral,
					cfg=q_config.fully_quantized(fwd_bits=bits, bwd_bits=bits),
					rhs_quant_mode=rhs_quant_mode,
				)
			}
	return {}  # empty just in case of not getting any error


def block_wise_ffn(remat_ffn, inputs, chunk_size: int):
	generating = inputs.shape[1] == 1
	try:
		if generating:
			return remat_ffn(inputs)
		else:
			return rearrange(
				jax.lax.scan(
					f=lambda carry, idx: (carry.at[:, idx].set(remat_ffn(carry[:, idx])), None),
					init=rearrange(inputs, "b (c n) d -> b c n d", c=chunk_size),
					xs=jax.numpy.arange(chunk_size),
					length=chunk_size,
					unroll=True,
				)[0],
				"b c n d -> b (c n) d",
			)
	except Exception as e:
		raise EasyDeLBlockWiseFFNError(
			"You Are using BlockWise FFN from near-infinite-context length paper and you might be passing "
			"input arguments in wrong way in case that you don'position_ids want to use this just pass `use_scan_mlp=False` in "
			"model config or in config_kwargs in AutoEasyDeLModelFor... or change `scan_mlp_chunk_size` "
			f"in configs for more information read Docs.\nOriginal Error\n{e}"
		) from e


def control_mlp_sharding(x: jax.Array, partition_axis: PartitionAxis):
	"""
	handles MLP Shardings
	"""
	sqax = (
		partition_axis.sequence_axis
		if x.shape[1] != 1
		else partition_axis.generation_query_sequence_axis
	)
	x = with_sharding_constraint(
		x,
		sharding=PartitionSpec(
			partition_axis.batch_axis,
			sqax,
			partition_axis.hidden_state_axis,
		),
	)
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

	if not (lora_rank > 0):
		raise ValueError("lora_rank should be a positive value and higher than `0`.")
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


def split_lora_params(model: nn.Module) -> nn.Module:
	"""
	get LoRA (Low-Rank Adaptation) from layers within a model.

	Args:
	    model: The EasyDeL model.
	Returns:
	    LoRA Layer Weights.
	"""
	from easydel.utils.graph_utils import get_module_from_path, iter_module_search

	od = {}
	with tqdm(
		total=len([p[0] for p in iter_module_search(model, nn.LoRA)]),
		desc="Split LoRA Params",
	) as pbar:
		for path, _ in iter_module_search(model, nn.LoRA):
			base_module: nn.LoRA = get_module_from_path(model=model, path=path)
			od.update({path: {"lora_a": base_module.lora_a, "lora_b": base_module.lora_b}})
			pbar.update(1)
	return unflatten_dict(od)


def merge_lora_params(model: nn.Module, lora_tree: tp.Dict) -> nn.Module:
	"""
	get LoRA (Low-Rank Adaptation) from layers within a model.

	Args:
	    model: The EasyDeL model.
	Returns:
	    LoRA Layer Weights.
	"""
	from easydel.utils.graph_utils import get_module_from_path, iter_module_search

	if not is_flatten(lora_tree):
		lora_tree = flatten_dict(lora_tree)
	with tqdm(
		total=len([p[0] for p in iter_module_search(model, nn.LoRA)]),
		desc="Merge LoRA Params",
	) as pbar:
		for path, _ in iter_module_search(model, nn.LoRA):
			base_module: nn.LoRA = get_module_from_path(model=model, path=path)
			base_module.lora_b = lora_tree[path + ("lora_b",)]
			base_module.lora_a = lora_tree[path + ("lora_a",)]
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


M = tp.TypeVar("M")


def auto_remat(
	*modules: tp.Type[M],
	policy: tp.Union[
		EasyDeLGradientCheckPointers, str
	] = EasyDeLGradientCheckPointers.NONE,
	prevent_cse: bool = True,
) -> tp.Tuple[tp.Type[M], ...]:
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


# Main FLOP counting function
def count_flop_jaxpr(jaxpr) -> int:
	"""Count flops in a Jaxpr."""

	def get_shape_size(shape) -> int:
		"""Calculate total size of an array shape."""
		return int(np.prod(shape)) if shape else 1

	def compute_binary_op_flops(eqn) -> int:
		"""Generic FLOP counter for binary operations with broadcasting."""
		shape0 = eqn.invars[0].aval.shape
		shape1 = eqn.invars[1].aval.shape
		output_shape = np.broadcast_shapes(shape0, shape1)
		return get_shape_size(output_shape)

	def compute_unary_op_flops(eqn) -> int:
		"""FLOP counter for unary operations."""
		shape = eqn.invars[0].aval.shape
		return get_shape_size(shape)

	def compute_dot_general_flops(eqn) -> int:
		"""Compute FLOPs for dot_general operation."""
		shapes = [var.aval.shape for var in eqn.invars]
		if len(shapes) != 2:
			return 0

		dimension_numbers = eqn.params.get("dimension_numbers", None)
		if not dimension_numbers:
			return 0

		(lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

		# Calculate sizes for contracting dimensions
		contracting_size = np.prod([shapes[0][d] for d in lhs_contract])

		# Calculate output shape size
		batch_size = np.prod([shapes[0][d] for d in lhs_batch])
		lhs_remaining = [
			d for i, d in enumerate(shapes[0]) if i not in lhs_contract and i not in lhs_batch
		]
		rhs_remaining = [
			d for i, d in enumerate(shapes[1]) if i not in rhs_contract and i not in rhs_batch
		]
		out_size = batch_size * np.prod(lhs_remaining) * np.prod(rhs_remaining)

		# Each output element requires 2*contracting_size - 1 operations
		return out_size * (2 * contracting_size - 1)

	def compute_conv_flops(eqn) -> int:
		"""Compute FLOPs for convolution operation."""
		lhs_shape = eqn.invars[0].aval.shape
		rhs_shape = eqn.invars[1].aval.shape

		dimension_numbers = eqn.params.get("dimension_numbers", None)
		if not dimension_numbers:
			return 0

		lhs_spec, rhs_spec, out_spec = dimension_numbers

		batch_size = lhs_shape[lhs_spec.index("N")]
		in_channels = lhs_shape[lhs_spec.index("C")]
		out_channels = rhs_shape[rhs_spec.index("O")]

		spatial_size = 1
		kernel_size = 1
		for d in range(len(lhs_spec) - 2):
			spatial_size *= lhs_shape[lhs_spec.index(str(d))]
			kernel_size *= rhs_shape[rhs_spec.index(str(d))]

		ops_per_point = 2 * kernel_size * in_channels - 1
		total_points = batch_size * spatial_size * out_channels

		return ops_per_point * total_points

	def compute_reduce_flops(eqn) -> int:
		"""Compute FLOPs for reduction operations."""
		shape = eqn.invars[0].aval.shape
		reduced_axes = eqn.params.get("axes", ())

		if not reduced_axes:
			return 0

		reduced_size = np.prod([shape[ax] for ax in reduced_axes])
		remaining_shape = [s for i, s in enumerate(shape) if i not in reduced_axes]
		remaining_size = np.prod(remaining_shape) if remaining_shape else 1

		return remaining_size * (reduced_size - 1)

	def compute_attention_flops(eqn) -> int:
		"""Compute FLOPs for attention operation."""
		q_shape = eqn.invars[0].aval.shape
		k_shape = eqn.invars[1].aval.shape

		batch, q_len, num_heads, head_dim = q_shape
		_, kv_len, _, _ = k_shape

		qk_flops = batch * num_heads * q_len * kv_len * (2 * head_dim - 1)
		softmax_flops = batch * num_heads * q_len * (kv_len + (kv_len - 1) + 1)
		av_flops = batch * num_heads * q_len * head_dim * (2 * kv_len - 1)

		return qk_flops + softmax_flops + av_flops

	def count_scan_flops(eqn) -> int:
		"""Count FLOPs in a scan operation."""
		scan_jaxpr = eqn.params.get("jaxpr", None)
		if scan_jaxpr:
			body_flops = count_flop_jaxpr(scan_jaxpr)
			length = eqn.invars[0].aval.shape[0]
			return body_flops * length
		return 0

	def count_cond_flops(eqn) -> int:
		"""Count FLOPs in a conditional operation."""
		true_jaxpr = eqn.params.get("true_jaxpr", None)
		false_jaxpr = eqn.params.get("false_jaxpr", None)

		total_flops = 0
		if true_jaxpr:
			total_flops += count_flop_jaxpr(true_jaxpr)
		if false_jaxpr:
			total_flops += count_flop_jaxpr(false_jaxpr)
		return total_flops // 2

	def get_scatter_flops(eqn) -> int:
		"""Count FLOPs in a scatter operation."""
		updates_shape = eqn.invars[2].aval.shape
		return get_shape_size(updates_shape)

	def compute_select_n_flops(eqn) -> int:
		"""Compute FLOPs for select_n operation."""
		pred_shape = eqn.invars[0].aval.shape
		return get_shape_size(pred_shape)

	def compute_cumsum_flops(eqn) -> int:
		"""Compute FLOPs for cumulative sum."""
		shape = eqn.invars[0].aval.shape
		axis = eqn.params.get("axis", 0)
		# Each element adds to the previous sum
		return get_shape_size(shape) - shape[axis]

	def compute_max_flops(eqn) -> int:
		"""Compute FLOPs for max operation."""
		if len(eqn.invars) == 2:
			# Binary max
			return compute_binary_op_flops(eqn)
		# Unary max
		return compute_unary_op_flops(eqn)

	def compute_pow_flops(eqn) -> int:
		"""Compute FLOPs for power operation."""
		if len(eqn.invars) == 2:
			shape0 = eqn.invars[0].aval.shape
			shape1 = eqn.invars[1].aval.shape
			output_shape = np.broadcast_shapes(shape0, shape1)
			return 8 * get_shape_size(output_shape)  # Power is expensive
		return 8 * get_shape_size(eqn.invars[0].aval.shape)

	def compute_integer_pow_flops(eqn) -> int:
		"""Compute FLOPs for integer power."""
		shape = eqn.invars[0].aval.shape
		power = eqn.params.get("y", 2)
		return (power - 1) * get_shape_size(shape)

	def compute_and_flops(eqn) -> int:
		"""Compute FLOPs for logical and operation."""
		return compute_binary_op_flops(eqn)

	def count_custom_vjp_flops(eqn) -> int:
		"""Count FLOPs in custom VJP operation."""
		fwd_jaxpr = eqn.params.get("fun_jaxpr", None)
		if fwd_jaxpr:
			return count_flop_jaxpr(fwd_jaxpr)
		return 0

	def compute_sqrt_flops(eqn) -> int:
		"""Compute FLOPs for square root operation."""
		# Square root is typically more expensive than basic operations
		return 4 * compute_unary_op_flops(eqn)

	def compute_argmax_flops(eqn) -> int:
		"""Compute FLOPs for argmax operation."""
		shape = eqn.invars[0].aval.shape
		axis = eqn.params.get("axes", (0,))[0]
		# For each output element, we need to compare n-1 elements where n is the size of the reduction axis
		remaining_size = get_shape_size(shape) // shape[axis]
		return remaining_size * (shape[axis] - 1)

	def compute_min_flops(eqn) -> int:
		"""Compute FLOPs for min operation."""
		if len(eqn.invars) == 2:
			# Binary min
			return compute_binary_op_flops(eqn)
		# Unary min
		return compute_unary_op_flops(eqn)

	def compute_rem_flops(eqn) -> int:
		"""Compute FLOPs for remainder operation."""
		# Remainder typically involves division and multiplication
		return 2 * compute_binary_op_flops(eqn)

	def compute_square_flops(eqn) -> int:
		"""Compute FLOPs for square operation (x * x)."""
		# Square is a single multiplication of a number by itself
		shape = eqn.invars[0].aval.shape
		return get_shape_size(shape)

	def compute_triangular_solve_flops(eqn) -> int:
		"""Compute FLOPs for triangular solve operation."""
		# For a triangular solve with a matrix of size n x n,
		# each row/column requires n^2/2 multiply-adds
		matrix_shape = eqn.invars[0].aval.shape
		n = matrix_shape[-1]  # Size of the last dimension
		batch_dims = matrix_shape[:-2]
		batch_size = np.prod(batch_dims) if batch_dims else 1
		return batch_size * n * (n + 1) * (2 * n + 1) // 6

	def compute_erf_inv_flops(eqn) -> int:
		"""Compute FLOPs for inverse error function."""
		# erf_inv is computationally expensive, typically implemented
		# as a series expansion or numerical approximation
		return 15 * compute_unary_op_flops(eqn)

	def compute_or_flops(eqn) -> int:
		"""Compute FLOPs for logical or operation."""
		return compute_binary_op_flops(eqn)

	def compute_shift_right_logical_flops(eqn) -> int:
		"""Compute FLOPs for logical right shift."""
		return compute_binary_op_flops(eqn)

	# Dictionary mapping primitives to their FLOP counting functions
	primitive_flops: tp.Dict[str, tp.Callable] = {
		# Binary operations
		"mul": compute_binary_op_flops,
		"add": compute_binary_op_flops,
		"sub": compute_binary_op_flops,
		"div": compute_binary_op_flops,
		"gt": compute_binary_op_flops,
		"lt": compute_binary_op_flops,
		"ge": compute_binary_op_flops,
		"le": compute_binary_op_flops,
		"ne": compute_binary_op_flops,
		"eq": compute_binary_op_flops,
		# Unary operations
		"neg": compute_unary_op_flops,
		"sin": lambda eqn: 5 * compute_unary_op_flops(eqn),
		"cos": lambda eqn: 5 * compute_unary_op_flops(eqn),
		"exp": lambda eqn: 4 * compute_unary_op_flops(eqn),
		"log": lambda eqn: 6 * compute_unary_op_flops(eqn),
		"log1p": lambda eqn: 6 * compute_unary_op_flops(eqn),
		"tanh": lambda eqn: 7 * compute_unary_op_flops(eqn),
		"rsqrt": lambda eqn: 6 * compute_unary_op_flops(eqn),
		# Linear algebra
		"dot_general": compute_dot_general_flops,
		"conv_general_dilated": compute_conv_flops,
		# Reduction operations
		"reduce_sum": compute_reduce_flops,
		"reduce_max": compute_reduce_flops,
		"reduce_min": compute_reduce_flops,
		# Special operations
		"scatter-add": get_scatter_flops,
		"scan": count_scan_flops,
		"cond": count_cond_flops,
		# Memory operations (0 FLOPs)
		"broadcast_in_dim": lambda eqn: 0,
		"reshape": lambda eqn: 0,
		"transpose": lambda eqn: 0,
		"slice": lambda eqn: 0,
		"gather": lambda eqn: 0,
		"concatenate": lambda eqn: 0,
		"convert_element_type": lambda eqn: 0,
		"dynamic_slice": lambda eqn: 0,
		"pad": lambda eqn: 0,
		# Parallel/Sharding operations (0 FLOPs)
		"pjit": lambda eqn: 0,
		"shard_map": lambda eqn: 0,
		"sharding_constraint": lambda eqn: 0,
		# Other operations
		"dot_product_attention_fwd_wrapper": compute_attention_flops,
		"select_n": compute_select_n_flops,
		"cumsum": compute_cumsum_flops,
		"max": compute_max_flops,
		"iota": lambda eqn: 0,  # Memory operation, no FLOPs
		"pow": compute_pow_flops,
		"integer_pow": compute_integer_pow_flops,
		"and": compute_and_flops,
		"random_fold_in": lambda eqn: 0,  # Random number generation, no FLOPs
		"custom_vjp_call_jaxpr": count_custom_vjp_flops,
		"logistic": lambda eqn: 4 * compute_unary_op_flops(eqn),  # sigmoid function
		# No-op operations (0 FLOPs)
		"stop_gradient": lambda eqn: 0,  # Just passes through the value
		"squeeze": lambda eqn: 0,  # Reshapes data, no computation
		"copy": lambda eqn: 0,  # Memory operation only
		"split": lambda eqn: 0,
		"remat2": lambda eqn: 0,
		"random_seed": lambda eqn: 0,
		"random_unwrap": lambda eqn: 0,
		"random_wrap": lambda eqn: 0,
		"random_split": lambda eqn: 0,
		"random_bits": lambda eqn: 0,
		# Bitwise and type conversion operations
		"shift_right_logical": compute_shift_right_logical_flops,
		"or": compute_or_flops,
		"bitcast_convert_type": lambda eqn: 0,  # Type conversion, no computation
		# Mathematical operations
		"abs": compute_unary_op_flops,  # Single comparison/selection per element
		"erf_inv": compute_erf_inv_flops,  # Inverse error function
		"triangular_solve": compute_triangular_solve_flops,
		# Computation operations
		"square": compute_square_flops,
		"sqrt": compute_sqrt_flops,
		"argmax": compute_argmax_flops,
		"add_any": compute_binary_op_flops,  # Similar to regular add
		"min": compute_min_flops,
		"rem": compute_rem_flops,
	}

	flops = 0

	def visit_jaxpr(jaxpr):
		nonlocal flops
		for eqn in jaxpr.eqns:
			primitive_name = eqn.primitive.name
			if primitive_name in primitive_flops:
				flops += primitive_flops[primitive_name](eqn)
			else:
				warnings.warn(f"Unhandled primitive {primitive_name}", stacklevel=1)

			# Recursively visit subjaxprs
			for subjaxpr in jax.core.jaxprs_in_params(eqn.params):
				visit_jaxpr(subjaxpr)

	visit_jaxpr(jaxpr)
	return flops


class TraceResult:
	def __init__(self, executable):
		self._executable = executable
		self._cached_cost = None

	@property
	@lru_cache(maxsize=1)  # noqa
	def cost_analysis(self):
		return self._executable.cost_analysis()

	@property
	def flops(self):
		return self.cost_analysis["flops"]


class FunctionTracer:
	def __init__(self):
		self.new_executables: List[TraceResult] = []
		self._before: Set = set()

	def __getitem__(self, idx):
		return self.new_executables[idx]


class CompilationTracker:
	def __init__(self):
		self.first_time = True
		self.cached_flops = 0
		self.functions = None

	@property
	def online_flops(self):
		if self.functions is None:
			return 0
		cached_flops = 0
		for cm in self.functions:
			try:
				cached_flops += cm.cost_analysis()["flops"]
			except Exception:
				...
		return cached_flops

	@contextmanager
	def trace_compilation(self):
		if self.first_time:
			before = set(jax.lib.xla_bridge.get_backend().live_executables())
			yield
			after = set(jax.lib.xla_bridge.get_backend().live_executables())
			new = after - before
			if new:
				cmpf = list(new)
				self.functions = cmpf
				for cm in cmpf:
					try:
						self.cached_flops += cm.cost_analysis()["flops"]
					except Exception:
						...
			self.first_time = False
		else:
			yield


@contextmanager
def trace_functions():
	tracer = FunctionTracer()
	tracer._before = set(jax.lib.xla_bridge.get_backend().live_executables())

	try:
		yield tracer
	finally:
		after = set(jax.lib.xla_bridge.get_backend().live_executables())
		new = after - tracer._before
		tracer.new_executables = [TraceResult(exe) for exe in new]


class ModuleCaches(nn.Cache): ...


class OverWriteWithGradient(nn.Param): ...


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
