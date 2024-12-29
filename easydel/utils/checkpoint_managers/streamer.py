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

import os
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import msgpack
import safetensors
from flax.serialization import to_bytes, to_state_dict
from flax.struct import PyTreeNode
from tqdm import tqdm

from easydel.etils.etils import get_logger

from ..traversals import flatten_dict, is_flatten, unflatten_dict

logger = get_logger(__name__)

ALLOWED_DATA_TYPES = [
	jnp.int4,
	jnp.int8,
	jnp.int16,
	jnp.int32,
	jnp.int64,
	jnp.uint4,
	jnp.uint8,
	jnp.uint16,
	jnp.uint32,
	jnp.uint64,
	jnp.float16,
	jnp.float32,
	jnp.float64,
	jnp.bfloat16,
	jnp.float_,
]


def get_dtype(
	array: jax.Array,
	dtype: tp.Optional[tp.Union[str, jnp.dtype]],
) -> jax.Array:
	"""
	Get the tensor with the specified data type.

	Args:
		array: The input tensor.
		dtype: The desired data type.

	Returns:
		The tensor with the specified data type.
	"""
	if not dtype:
		return array

	if isinstance(dtype, str):
		dtype_map = {
			"bf16": jnp.bfloat16,
			"bfloat16": jnp.bfloat16,
			"fp16": jnp.float16,
			"float16": jnp.float16,
			"fp32": jnp.float32,
			"float32": jnp.float32,
			"fp64": jnp.float64,
			"float64": jnp.float64,
		}
		try:
			dtype = dtype_map[dtype]
		except KeyError as e:
			raise ValueError(f"Unsupported dtype string: {dtype}") from e

	if array.dtype in (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64):
		return array.astype(dtype)
	return array  # Return original array if it's not a float


def _read_process_array(
	key,
	shard_fns,
	mismatch_allowed,
	manager,
	callback: tp.Optional[tp.Callable[[jax.Array, str], jax.Array]] = None,
):
	"""Helper function to process a single tensor from a checkpoint."""
	tensor = manager.get_tensor(key)
	mismatch = 0
	if shard_fns:
		try:
			callable_func = shard_fns.get(key)
			if callable_func is None:
				if not mismatch_allowed:
					raise KeyError(
						f"Shard Function {key} is None and NoneType OBJ is not callable."
					)
				mismatch = 1
			else:
				tensor = callable_func(tensor)
		except KeyError as k_err:
			if not mismatch_allowed:
				raise KeyError(k_err) from None
			mismatch = 1

	if callback:
		tensor = callback(tensor, key)
	return key, tensor, mismatch


class CheckpointManager:
	"""
	A class to manage saving and loading checkpoints.

	Args:
		checkpoint_dir: The directory to save checkpoints to.
		enable: Whether to enable saving and loading checkpoints.
		float_dtype: The floating-point data type to use for saving checkpoints.
		save_optimizer_state: Whether to save the optimizer state in the checkpoint.
		verbose: Whether to print verbose output.
	"""

	def __init__(
		self,
		checkpoint_dir: tp.Union[str, os.PathLike],
		enable: bool = True,
		float_dtype: jnp.dtype = jnp.bfloat16,
		save_optimizer_state: bool = True,
		verbose: bool = False,
	):
		self.float_dtype = float_dtype
		self.save_optimizer_state = save_optimizer_state
		self.checkpoint_dir = checkpoint_dir
		self.enable = enable
		self.verbose = verbose

	@staticmethod
	def load_checkpoint(
		path: tp.Union[str, os.PathLike],
		shard_fns: tp.Optional[dict[tp.Callable]] = None,
		verbose: bool = False,
		mismatch_allowed: bool = True,
		callback: tp.Optional[tp.Callable[[jax.Array, str], jax.Array]] = None,
	) -> tp.Tuple[tp.Union[PyTreeNode, dict], dict]:
		"""
		Load a checkpoint from the given path.

		Args:
			path: The path to the checkpoint file.
			target: The target PyTree to load the checkpoint into.
			shard_fns: A dictionary of functions to shard the state after loading.
			verbose: Whether to print verbose output.
			mismatch_allowed: Whether to allow mismatches between the state dictionary and shard functions.
			callback: Optional callback applied to each loaded tensor
		Returns:
			A tuple containing the loaded state dictionary and metadata.
		"""
		with safetensors.safe_open(path, framework="flax") as f:
			metadata = f.metadata()
			keys = list(f.keys())

			if shard_fns and not is_flatten(shard_fns):
				shard_fns = flatten_dict(shard_fns, sep=".")

			process_func = partial(
				_read_process_array,
				shard_fns=shard_fns,
				mismatch_allowed=mismatch_allowed,
				manager=f,
				callback=callback,
			)
			results = [
				process_func(key)
				for key in tqdm(
					keys,
					desc="Loading",
					total=len(keys),
					disable=not verbose,
				)
			]

		state = {key: tensor for key, tensor, _ in results}
		mismatch_count = sum(mismatch for _, _, mismatch in results)

		if verbose and mismatch_count:
			logger.info(f"Sharding mismatch: {mismatch_count}")

		state = unflatten_dict(state, sep=".")
		return state, metadata

	@staticmethod
	def save_checkpoint(
		state: PyTreeNode,
		path: tp.Union[str, os.PathLike],
		gather_fns: tp.Optional[dict[tp.Callable]] = None,
		float_dtype: tp.Optional[tp.Union[str, jnp.dtype]] = None,
		verbose: bool = True,
		mismatch_allowed: bool = True,
		metadata: tp.Optional[dict[str, str]] = None,
	):
		"""
		Save a checkpoint to the given path using SafeTensors.

		Args:
			state: The state dictionary to save.
			path: The path to the checkpoint file.
			gather_fns: A dictionary of functions to gather the state before saving.
			float_dtype: The floating-point data type to use for saving the checkpoint.
			verbose: Whether to print verbose output.
			mismatch_allowed: Whether to allow mismatches between the state dictionary and gather functions.
			metadata: Additional metadata to store in the checkpoint.
		"""
		state = to_state_dict(state)
		gather_mismatch_count = 0

		if not is_flatten(state):
			state = flatten_dict(state, sep=".")

		if gather_fns:
			if not is_flatten(gather_fns):
				gather_fns = flatten_dict(gather_fns)

			pbar_gather = tqdm(
				list(state.keys()),
				desc="Gathering State",
				disable=not verbose,
			)
			for key in pbar_gather:
				try:
					callable_func = gather_fns.get(key)
					if callable_func is None:
						if not mismatch_allowed:
							raise KeyError(
								f"Gather Function {key} is None and NoneType OBJ is not callable."
							)
						gather_mismatch_count += 1
					else:
						state[key] = callable_func(state[key])

				except KeyError as e:
					if not mismatch_allowed:
						raise KeyError(e) from None
					gather_mismatch_count += 1
				pbar_gather.set_postfix(gather_mismatch=gather_mismatch_count)
				pbar_gather.update(1)

		state = {
			key: get_dtype(
				jnp.array(value) if not isinstance(value, jax.Array) else value,
				float_dtype,
			)
			for key, value in state.items()
			if value is not None
		}

		safetensors.flax.save_file(tensors=state, filename=path, metadata=metadata)

	@staticmethod
	def save_state_to_file(
		state: PyTreeNode,
		path: tp.Union[str, os.PathLike],
		gather_fns: tp.Optional[dict[tp.Callable]] = None,
		float_dtype: tp.Optional[tp.Union[str, jnp.dtype]] = None,
		verbose: bool = False,
		mismatch_allowed: bool = True,
	):
		"""
		Save the state dictionary to a file.

		Args:
			state: The state dictionary to save.
			path: The path to the file to save the state dictionary to.
			gather_fns: A dictionary of functions to gather the state before saving.
			float_dtype: The floating-point data type to use for saving the state dictionary.
			verbose: Whether to print verbose output.
			mismatch_allowed: Whether to allow mismatches between the state dictionary and gather functions.
		"""
		state = to_state_dict(state)
		packer = msgpack.Packer()
		flatten_state = flatten_dict(state)
		if gather_fns:
			gather_fns = flatten_dict(gather_fns)

		pbar = tqdm(
			flatten_state.items(),
			disable=not verbose,
			desc="Saving State to File",
		)

		gather_mismatch_count = 0
		with open(path, "wb") as stream:
			for key, value in pbar:
				if gather_fns:
					try:
						callable_func = gather_fns.get(key)
						if callable_func is None:
							if not mismatch_allowed:
								raise KeyError(
									f"Gather Function {key} is None and NoneType OBJ is not callable."
								)
							gather_mismatch_count += 1

						else:
							value = callable_func(value)
					except KeyError as k_err:
						if not mismatch_allowed:
							raise KeyError(k_err) from None
						gather_mismatch_count += 1
				pbar.set_postfix(gather_mismatch=gather_mismatch_count)
				value = get_dtype(value, float_dtype)
				stream.write(packer.pack((key, to_bytes(value))))

	def save_pickle(self, obj: object, filename: tp.Union[str, os.PathLike]):
		"""
		Save an object to a pickle file.

		Args:
			obj: The object to save.
			filename: The filename to save the object to.
		"""
		import pickle

		if self.enable:
			path = os.path.join(self.checkpoint_dir, filename)
		else:
			path = "/dev/null"

		with open(path, "wb") as stream:
			pickle.dump(obj, stream)
