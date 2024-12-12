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
import re
import warnings
from functools import partial
from typing import Any, Dict, Literal, Optional, Sequence, Union

import chex
import fjformer
import flax
import flax.core
import jax
import jax.experimental
import jax.tree_util
from aqt.jax.v2 import config as q_config
from aqt.jax.v2.flax import aqt_flax as q_flax
from einops import rearrange
from flax import nnx
from flax import nnx as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental.mesh_utils import create_device_mesh
from jax.interpreters import pxla
from tqdm.auto import tqdm

from easydel.etils.errors import EasyDeLBlockWiseFFNError
from easydel.etils.etils import (
	AVAILABLE_SPARSE_MODULE_TYPES,
	EasyDeLQuantizationMethods,
	get_logger,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.modules._base.base_config import EasyMethod

warnings.filterwarnings(
	"ignore",
	message="Primitive dynamic_update_slice was not handled by class",
)
logger = get_logger(__name__)
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
}

ROPE_TYPES = Optional[
	Literal[
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


class MaskVariable(nnx.Variable): ...


class FrequenciesVariable(nnx.Variable): ...


def canonicalize_dtype(
	*args,
	dtype: Optional[chex.ArrayDType] = None,  # type:ignore
	inexact: bool = True,
) -> chex.ArrayDType:  # type:ignore
	"""Canonicalize an optional dtype to the definitive dtype.

	If the ``dtype`` is None this function will infer the dtype. If it is not
	None it will be returned unmodified or an exceptions is raised if the dtype
	is invalid.
	from the input arguments using ``jnp.result_type``.

	Args:
	  *args: JAX array compatible values. None values
	    are ignored.
	  dtype: Optional dtype override. If specified the arguments are cast to
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


def get_names_from_partition_spec(partition_specs):
	"""The get_names_from_partition_spec function takes a partition_specs argument, which is either a dictionary or list.
	If it's a dictionary, the function converts it to a list of values. Then for each item in the partition_specs list:
	    If the item is None, continue (do nothing) and move on to next iteration of loop.
	    If the item is an instance of str (i.e., if it's just one string), add that string to names set and move
	    on to next iteration of loop.
	    Otherwise, (if not None or str), call get_names_from_partition_spec recurs

	Args:
	    partition_specs: Define the partitioning of a table

	Returns:
	    A list of the names of all partitions
	"""
	names = set()
	if isinstance(partition_specs, dict):
		partition_specs = partition_specs.values()
	for item in partition_specs:
		if item is None:
			continue
		elif isinstance(item, str):
			names.add(item)
		else:
			names.update(get_names_from_partition_spec(item))

	return list(names)


def names_in_mesh(*names):
	"""The names_in_mesh function is a decorator that can be used to check whether
	the names of the axes passed into a function are valid.  It will raise an
	exception if any of the axis names are not in the physical mesh.  For example,
	if you have a function that takes two axes as arguments, and you want to make sure they're both in your mesh:

	Args:
	    *names: Collect all the names passed to the function into a
	        tuple

	Returns:
	    A boolean indicating whether all the given
	"""
	return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def get_gradient_checkpoint_policy(name):
	"""
	The get_gradient_checkpoint_policy function is a helper function that returns the gradient checkpoint policy
	    specified by the name parameter.

	:param name: Select the checkpoint policy from the dictionary
	:return: A function that is used in the jax

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


def get_ranks_and_size(mesh):
	"""The get_ranks_and_size function is used to determine the number of MPI processes
	(``mp_node_size``) and the number of devices per process (``dp_node_size``).
	The ``mesh.shape[mp]`` determines how many MPI processes are needed,
	and then we divide that by the local device count to get ``mp_node_size = max( 1, mp / jax.local )`.
	This means that if there are more than enough devices for all MPI ranks on a node, each rank will only use one device; otherwise it will use

	Args:
	    mesh: Get the shape of the mesh

	Returns:
	    A dictionary with the following keys:
	"""
	out = dict(mesh=mesh)
	total_process_size = mesh.shape["tp"] * mesh.shape["sp"]
	mp_node_size = max(1, total_process_size // jax.local_device_count())
	dp_node_size = jax.process_count() // mp_node_size
	out.update(mp_node_size=mp_node_size, dp_node_size=dp_node_size)

	dp_node_rank = jax.process_index() // mp_node_size
	mp_node_rank = jax.process_index() % mp_node_size
	out.update(dp_node_rank=dp_node_rank, mp_node_rank=mp_node_rank)
	return out


def create_mesh(
	axis_dims: Sequence[int] = (1, -1, 1, 1),
	axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
	backend="",
):
	"""The create_mesh function creates a mesh object that can be used to shard arrays.

	Args:
	    axis_dims: Sequence[int]: Specify the dimensions of the mesh
	    axis_names: Sequence[str]: Name the axes of the mesh
	    backend: Specify the backend to use

	Returns:
	    A mesh object
	"""
	array_devices = jax.numpy.ones(
		(len(jax.devices() if backend == "" else jax.devices(backend)), 1)
	)
	resh = array_devices.reshape(axis_dims).shape

	return jax.sharding.Mesh(create_device_mesh(resh), axis_names)


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
	bits: Optional[int] = None,
	mode: Literal["train", "serve", "convert"] = EasyMethod.TRAIN,
) -> dict:
	"""The get_general_dot function is a helper function that returns a q_flax.QDotGeneral object
	with the specified number of bits for forward and backward passes. If no bits are specified,
	the function returns None.

	Args:
	    bits: Optional[int]: Specify the number of bits for quantization
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


def read_depth(params: dict, path: str | None = None, state: dict | None = None):
	if state is None:
		state = {}
	for key, value in params.items():
		if isinstance(value, dict):
			accureated_path = path + "/" + key if path is not None else key
			state = read_depth(
				params[key], path=key if path is None else accureated_path, state=state
			)
		else:
			value_string = type(value).__name__ + f"(shape={value.shape})"
			state[path] = value_string
	return state


def get_maximum_depths(dictionary: dict):
	maximums = {}
	minimums = {}
	for k, _ in dictionary.items():
		splits = k.split("/")
		for index, split in enumerate(splits):
			try:
				split = int(split)
				if str(index) in maximums.keys():
					current = maximums[str(index)]
					if current < split:
						maximums[str(index)] = split
				else:
					maximums[str(index)] = split
				if str(index) in minimums.keys():
					split = int(split)
					if str(index) in minimums.keys():
						current = minimums[str(index)]
						if current > split:
							minimums[str(index)] = split
				else:
					minimums[str(index)] = split
			except ValueError:
				...
	return maximums, minimums


def control_mlp_sharding(x: jax.Array, partition_axis: PartitionAxis):
	"""
	this functions is disabled for now, it will cause breakdown and incorrect computation on gpu with CU lower than 7.5
	"""
	# batch_size, sequence_length, hidden_size = x.shape
	# is_gen = sequence_length == 1
	# mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
	# if not mesh.empty:
	#     partition_spec = PartitionSpec(
	#         partition_axis.batch_axis,
	#         None if is_gen else partition_axis.sequence_axis,
	#         (
	#             partition_axis.hidden_state_axis
	#             if (
	#                     mesh.shape[partition_axis.hidden_state_axis] / hidden_size
	#             ).is_integer()
	#             else None
	#         ),
	#     )
	#     x = with_sharding_constraint(x, partition_spec)
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
	method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.A8BIT,
	block_size: int = 256,
	quantization_pattern: Optional[str] = None,
	verbose: bool = True,
) -> nn.Module:
	"""
	Quantize parameters to 8-bit or nf4 precision, excluding specified layers.

	Args:
	    model: The model to quantize.
			method (EasyDeLQuantizationMethods): quantization method for params.
	    quantization_pattern (str): re pattern for layers to be quantized.
	    verbose (bool): whenever to use tqdm for logging stuff.

	Returns:
	    Quantized parameters in the same structure as the input.
	"""
	if method == EasyDeLQuantizationMethods.NONE:
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
	}.get(method, None)
	if quantizer is None:
		raise NotImplementedError("Requested Quantizer is not Supported")
	if quantization_pattern is None:
		quantization_pattern = ".*"
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


def print_pytree(pytree):
	jax.tree_util.tree_map_with_path(
		lambda p, v: print(
			f"{fjformer.tree_path_to_string(p,'.')}: dtype:{v.dtype}, shape:{v.shape}"
		),
		pytree,
	)


def apply_sparsity_to_params(
	params: Union[Dict[str, Any], Any],
	sparsify_module: AVAILABLE_SPARSE_MODULE_TYPES = "bcoo",
	verbose: bool = True,
) -> Union[Dict[str, Any], Any]:
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
		# print(layer_name)
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
