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

from __future__ import annotations

import dataclasses
import typing as tp
from abc import abstractmethod
from enum import Enum

import einops
import jax
from eformer.escale import PartitionAxis
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
from easydel.utils import traversals as etr
from easydel.utils.helpers import get_logger

from ..ops import BaseOperation

logger = get_logger("EasyDeL-AttentionOperator")


@etr.auto_pytree
class AttentionOutput:
	"""
	Container for the outputs of an attention operation.

	Attributes:
	    attention_weights: The attention probabilities, typically of shape
	        (batch, num_heads, query_seq_len, key_value_seq_len). Optional.
	    attention_outputs: The final weighted sum of values, typically of shape
	        (batch, query_seq_len, num_heads, head_dim) or (batch, num_heads, query_seq_len, head_dim).
	        Optional.
	"""

	attention_weights: tp.Optional[Array] = None
	attention_outputs: tp.Optional[Array] = None


class RuntimeType(Enum):
	"""
	Enumerates the possible runtime modes for attention operations.

	Attributes:
	    normal: Standard training or evaluation mode.
	    generation: Autoregressive generation mode, often involving KV caching and
	        single token decoding.
	"""

	normal = "normal"
	generation = "generation"


@etr.auto_pytree
class AttentionMetadata:
	"""
	Holds configuration, context, and metadata for attention operations.

	This class centralizes various parameters needed by different attention
	implementations, facilitating consistent behavior and configuration. It handles
	default values and can be initialized from an `EasyDeLBaseConfig`.

	Attributes:
	    runtime_dtype: The primary JAX dtype for computations (e.g., q, k, v).
	    runtime_softmax_dtype: Optional JAX dtype for the softmax computation,
	        allowing for higher precision if needed (e.g., float32).
	    sequence_axis_name: The name used for the sequence axis in JAX parallelism
	        (axis_names for pjit).
	    mesh: The JAX device mesh for distributed computation. Must be provided
	        or inferred from context.
	    platform: The target hardware platform (e.g., TPU, GPU).
	    backend: The specific JAX backend being used (e.g., TPU, CUDA, ROCM).
	    partition_axis: Configuration for partitioning axes in distributed settings.
	        (Likely from `eformer.escale`).
	    base_config: An optional reference to the base model configuration object
	        for sourcing default values.
	    scan_ring_attention: Boolean flag indicating whether to use ring attention
	        via `jax.lax.scan`.
	    softmax_scale: The scaling factor applied before the softmax operation.
	        Often `1 / sqrt(head_dim)`.
	    dropout_prob: The dropout probability applied to attention weights.
	    blocksize_q: Block size for the query sequence dimension in blockwise attention.
	    blocksize_k: Block size for the key/value sequence dimension in blockwise attention.
	    blocksize_b: Block size for the batch dimension in blockwise attention (often 1).
	"""

	runtime_dtype: jax.typing.DTypeLike
	runtime_softmax_dtype: tp.Optional[jax.typing.DTypeLike] = None
	sequence_axis_name: str = ...
	mesh: tp.Optional[jax.sharding.Mesh] = ...
	platform: EasyDeLPlatforms = ...
	backend: EasyDeLBackends = ...
	partition_axis: PartitionAxis = ...
	base_config: tp.Optional[EasyDeLBaseConfig] = None
	scan_ring_attention: bool = ...
	softmax_scale: float = ...
	dropout_prob: float = ...
	blocksize_q: int = ...
	blocksize_k: int = ...
	blocksize_b: int = ...

	def __post_init__(self):
		"""
		Initializes default values and performs safety checks after dataclass creation.

		Sets reasonable defaults for various parameters if they are not provided
		(or marked as Ellipsis). It attempts to source defaults from the `base_config`
		if available. It also infers the JAX mesh and backend if not explicitly given.
		Finally, it performs a safety check to ensure no essential attributes remain
		uninitialized (as Ellipsis).
		"""
		# fmt:off
		self.set_attrs_carefully("dropout_prob", 0.0, use_base_config=False)
		self.set_attrs_carefully("blocksize_q", 512)
		self.set_attrs_carefully("blocksize_k", 1024)
		self.set_attrs_carefully("blocksize_b", 1)
		self.set_attrs_carefully("runtime_dtype",  jnp.float32, "attn_dtype")
		self.set_attrs_carefully("runtime_softmax_dtype", jnp.float32, "attn_softmax_dtype")
		self.set_attrs_carefully("softmax_scale", None, "softmax_scale")
		self.set_attrs_carefully("scan_ring_attention", True)
		self.set_attrs_carefully("partition_axis", PartitionAxis())
		self.set_attrs_carefully("sequence_axis_name", "sp", "sequence_axis_name", use_base_config=False)  # DON'T READ FROM CONFIG
		self.set_attrs_carefully("backend", jax.default_backend(), "backend")
		self.set_attrs_carefully("platform", ..., "platform") 
		self.set_attrs_carefully("mesh", ..., "mesh") 
		# fmt:on
		if self.mesh == Ellipsis and self.base_config is None:
			mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
			assert not mesh.empty, (
				"You should pass 'mesh' to `AttentionMetadata` or "
				"at least create that under mesh context manager"
			)
			self.mesh = mesh
		self._safety_check()
		if self.backend is None:
			current_backend = jax.default_backend()
			self.backend = getattr(
				EasyDeLBackends,
				current_backend,
				getattr(EasyDeLBackends, current_backend.upper()),
			)

	def _safety_check(self):
		"""Ensures no essential attributes are left uninitialized (as Ellipsis)."""
		for field in dataclasses.fields(self):
			val = getattr(self, field.name)
			if val == Ellipsis:
				raise ValueError(f"`{field.name}` shouldn't be ellipsis")

	@classmethod
	def from_config(
		cls,
		config: EasyDeLBaseConfig,
		softmax_scale: float,
		dropout_prob: float = 0.0,
	) -> AttentionMetadata:
		"""
		Factory method to create AttentionMetadata from an EasyDeLBaseConfig.

		Args:
		    config: The base configuration object (e.g., model config).
		    softmax_scale: The attention softmax scaling factor. Usually calculated
		        based on head dimension.
		    dropout_prob: The attention dropout probability. Defaults to 0.0.

		Returns:
		    An initialized AttentionMetadata instance.
		"""
		return cls(
			runtime_dtype=config.attn_dtype,
			runtime_softmax_dtype=config.attn_softmax_dtype,
			sequence_axis_name=config.sequence_axis_name,
			mesh=config.mesh,
			platform=config.platform,
			backend=config.backend,
			partition_axis=config.partition_axis,
			base_config=config,
			scan_ring_attention=config.scan_attention_layers,
			softmax_scale=softmax_scale,
			dropout_prob=dropout_prob,
			blocksize_q=config.blocksize_q,
			blocksize_k=config.blocksize_k,
			blocksize_b=config.blocksize_b,
		)

	def get_partition_specs(self, mode: RuntimeType, BTHD: bool = True):
		"""
		Generates JAX PartitionSpecs for attention tensors based on runtime mode.

		Args:
		    mode: The current runtime mode (normal or generation).
		    BTHD: Boolean indicating tensor layout. True for (Batch, Time, Head, Dim),
		        False for (Batch, Head, Time, Dim).

		Returns:
		    A tuple containing PartitionSpecs for:
		    (query, key, value, bias, mask, attention_output)
		"""
		assert mode in RuntimeType, f"mode should be in `RuntimeType` but we got {mode}"

		standmode = mode == RuntimeType.generation
		batch_axis = self.partition_axis.batch_axis
		head_axis = (
			self.partition_axis.generation_head_axis
			if standmode
			else self.partition_axis.head_axis
		)
		query_seq_axis = (
			self.partition_axis.generation_query_sequence_axis
			if standmode
			else self.partition_axis.query_sequence_axis
		)
		key_seq_axis = (
			self.partition_axis.generation_key_sequence_axis
			if standmode
			else self.partition_axis.key_sequence_axis
		)
		attn_dim_axis = (
			self.partition_axis.generation_attention_dim_axis
			if standmode
			else self.partition_axis.attention_dim_axis
		)
		bias_head_seq_axis = self.partition_axis.bias_head_sequence_axis
		bias_key_seq_axis = self.partition_axis.bias_key_sequence_axis

		if BTHD:
			# BTHD ordering
			query_partition_spec = Ps(batch_axis, query_seq_axis, head_axis, attn_dim_axis)
			key_partition_spec = Ps(batch_axis, key_seq_axis, head_axis, attn_dim_axis)
			value_partition_spec = Ps(batch_axis, key_seq_axis, head_axis, attn_dim_axis)
		else:
			# BHTD ordering
			query_partition_spec = Ps(batch_axis, head_axis, query_seq_axis, attn_dim_axis)
			key_partition_spec = Ps(batch_axis, head_axis, key_seq_axis, attn_dim_axis)
			value_partition_spec = Ps(batch_axis, head_axis, key_seq_axis, attn_dim_axis)

		qk_extern = (query_seq_axis, bias_key_seq_axis)

		bias_partition_spec = Ps(batch_axis, bias_head_seq_axis, *qk_extern)
		mask_partition_spec = Ps(batch_axis, None, *qk_extern)

		attention_partition_spec = query_partition_spec

		return (
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			mask_partition_spec,
			attention_partition_spec,
		)

	def set_attrs_carefully(
		self,
		attr_name: str,
		default: tp.Optional[tp.Any],
		pickup_name: tp.Optional[str] = None,
		use_base_config: bool = True,
	):
		"""
		Internal helper to set an attribute if it's not already set (or is Ellipsis).

		Optionally retrieves the value from `self.base_config` using `pickup_name`
		(or `attr_name` if `pickup_name` is None).

		Args:
		    attr_name: The name of the attribute to set on `self`.
		    default: The default value to use if not found in `base_config` or
		        if `use_base_config` is False.
		    pickup_name: The name of the attribute to look for in `base_config`.
		        Defaults to `attr_name`.
		    use_base_config: Whether to attempt retrieving the value from `base_config`.
		"""
		if not hasattr(self, attr_name) or getattr(self, attr_name, ...) == Ellipsis:
			pn = attr_name if pickup_name is None else pickup_name
			new_value = (
				default
				if (self.base_config is None or not use_base_config)
				else getattr(self.base_config, pn, default)
			)
			setattr(self, attr_name, new_value)


class AttentionImpl(BaseOperation):
	"""
	Abstract Base Class for specific attention implementations.

	Inherits from `BaseOperation` to leverage backend-specific dispatching.
	Subclasses must implement the core attention logic (`forward_native`) and
	potentially provide optimized versions for TPU (`forward_tpu`), GPU (`forward_gpu`),
	etc. They also need to declare their name and associated metadata.

	Provides common helper methods for attention processing like mask manipulation,
	head repeating (for GQA/MQA), and determining runtime mode.
	"""

	def __init__(self, metadata: AttentionMetadata) -> None:
		"""
		Initializes the attention implementation with its metadata.

		Args:
		    metadata: An `AttentionMetadata` instance containing configuration
		        and context for this attention operation.
		"""
		self.metadata = metadata

	@classmethod
	@abstractmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str, ...]]:
		"""
		Returns the unique name(s) identifying this attention implementation.

		Used by the `AttentionRegistry`. Can return a single string or a tuple/list
		of strings if the implementation has multiple aliases.

		Returns:
		    A string or tuple/list of strings representing the implementation name(s).
		"""

	@abstractmethod
	def get_impl_metadata(self) -> AttentionMetadata:
		"""
		Returns the `AttentionMetadata` associated with this implementation instance.

		Returns:
		    The `AttentionMetadata` instance passed during initialization.
		"""

	def get_runtime_type(self, q: jax.Array, BTHD: bool = True) -> RuntimeType:
		"""
		Determines the runtime mode (normal or generation) based on query shape.

		Assumes generation mode if the query sequence length dimension is 1.

		Args:
		    q: The query tensor.
		    BTHD: Boolean indicating tensor layout (True for B, T, H, D; False for B, H, T, D).

		Returns:
		    RuntimeType.generation if query sequence length is 1, else RuntimeType.normal.
		"""
		ingeneration = q.shape[1] == 1 if BTHD else q.shape[2] == 1
		return RuntimeType.generation if ingeneration else RuntimeType.normal

	def current_backend(self) -> tp.Literal["tpu", "gpu", "cpu"]:
		"""
		Returns the current JAX default backend as a lowercase string literal.

		Returns:
		    "tpu", "gpu", or "cpu".
		"""
		return jax.default_backend()

	@staticmethod
	def _split_attention_mask(attn_mask: Array) -> tp.Tuple[Array, Array]:
		"""
		Splits a combined attention mask into separate query and key-value masks.

		Assumes the input mask `attn_mask` might be 4D (batch, head, q_seq, kv_seq)
		or 3D (batch, q_seq, kv_seq). It derives the query mask by checking which
		query positions can attend to *any* key position, and the key-value mask
		by checking which key positions *can be attended to* by any query position.

		Args:
		    attn_mask: The combined attention mask (3D or 4D). If 4D, the last head dim
		        is used. Shape (..., q_seq, kv_seq).

		Returns:
		    A tuple `(q_mask, kv_mask)`:
		        - `q_mask`: Boolean array of shape (..., q_seq). True for valid query tokens.
		        - `kv_mask`: Boolean array of shape (..., kv_seq). True for valid key/value tokens.
		"""
		if attn_mask.ndim == 4:
			attn_mask = attn_mask[:, -1, :, :]
		return (
			jnp.any(attn_mask, axis=-1),
			jnp.any(attn_mask, axis=-2),
		)

	@staticmethod
	def _combine_query_kv_masks(q_mask: Array, kv_mask: Array) -> Array:
		"""
		Combines separate query and key-value masks into a standard attention mask.

		Creates a broadcastable mask where `mask[b, i, j]` is True if both
		`q_mask[b, i]` and `kv_mask[b, j]` are True.

		Args:
		    q_mask: Boolean array of shape (..., q_seq). True for valid query tokens.
		    kv_mask: Boolean array of shape (..., kv_seq). True for valid key/value tokens.

		Returns:
		    A boolean attention mask of shape (..., q_seq, kv_seq).
		"""
		if kv_mask.ndim == 2:
			kv_mask = kv_mask[:, None, :]
		if q_mask.ndim == 2:
			q_mask = q_mask[:, :, None]
		return q_mask * kv_mask

	@staticmethod
	def _create_causal_mask(qseq: int) -> Array:
		"""
		Creates a causal attention mask (lower triangular).

		Args:
		    qseq: The sequence length .

		Returns:
		    A boolean array of shape (qseq, qseq) where `mask[i, j]` is
		    True if `j <= i`, representing causal visibility.
		"""
		return jnp.tril(jnp.ones((qseq, qseq), dtype="b1"))

	@staticmethod
	def repeat_kv_heads(
		k: Array,
		v: Array,
		num_reps: int,
	) -> tp.Tuple[Array, Array]:
		"""
		Repeats Key and Value heads for Grouped Query Attention (GQA) or Multi-Query Attention (MQA).

		Expands the head dimension of K and V tensors to match the number of query heads.

		Args:
		    k: Key tensor, assumes shape (batch, seq_len, num_kv_heads, head_dim).
		    v: Value tensor, assumes shape (batch, seq_len, num_kv_heads, head_dim).
		    num_reps: The number of times to repeat each KV head (num_q_heads // num_kv_heads).

		Returns:
		    A tuple `(k_repeated, v_repeated)` with shapes
		    (batch, seq_len, num_q_heads, head_dim).
		"""
		return (
			einops.repeat(k, "b s h d -> b s (h r) d", r=num_reps),
			einops.repeat(v, "b s h d -> b s (h r) d", r=num_reps),
		)

	def _handle_kvhead(
		self,
		array: tp.Optional[Array],
		num_q_heads: int,
		num_kv_heads: int,
	) -> tp.Optional[Array]:
		"""
		Processes an attention bias or similar array based on head configuration (GQA/MQA).

		If the array's head dimension matches `num_kv_heads`, it repeats the heads
		to match `num_q_heads`. If it matches `num_q_heads` or is 1 (broadcastable),
		it's returned as is.

		Args:
		    array: The input array, typically an attention bias. Assumes head dimension
		        is at index 1. Shape (batch, num_heads, q_seq, kv_seq) or similar.
		        Can be None.
		    num_q_heads: The number of query heads.
		    num_kv_heads: The number of key/value heads.

		Returns:
		    The processed array with head dimension matching `num_q_heads`, or None
		    if the input was None.

		Raises:
		    ValueError: If the array's head dimension is incompatible.
		"""
		if array is None:
			return None

		if array.shape[1] == num_q_heads or array.shape[1] == 1:
			return array

		elif array.shape[1] == num_kv_heads:
			return einops.repeat(
				array,
				"b h q k -> b (h r) q k",
				r=num_q_heads // array.shape[1],
			)
		else:
			raise ValueError(
				f"Incompatible array shape. Got {array.shape[1]} heads, "
				f"expected {num_q_heads}, {num_kv_heads}, or 1"
			)

	def create_stable_sharding(
		self,
		state_ps: tp.Optional[Ps] = None,
		preserved_indices: tp.List[int] = None,
		clone_ps: tp.Optional[Ps] = None,
		dep: tp.Optional[tp.Union[Ps, bool]] = True,
	) -> tp.Optional[Ps]:
		"""
		Helper to create a PartitionSpec, potentially preserving only certain axes.

		This might be used for ensuring intermediate tensors or states have compatible
		sharding, possibly replicating across axes not specified in `preserved_indices`.

		Args:
		    state_ps: The base PartitionSpec to modify.
		    preserved_indices: A list of dimension indices whose partitioning should be
		        kept from `state_ps` (or `clone_ps` if provided). Other dimensions
		        will be set to None (replicated). If None, `state_ps` is returned.
		    clone_ps: An optional PartitionSpec to copy axis names from for the
		        preserved indices, instead of using `state_ps`.
		    dep: A dependency flag or PartitionSpec. If None, returns None. Defaults to True.
		        (The exact purpose might be context-specific, potentially for control flow).

		Returns:
		    A new PartitionSpec with only specified axes partitioned, or None based on `dep`.
		    Returns `state_ps` directly if `preserved_indices` is None.
		"""
		if dep is None:
			return None

		if state_ps is None:
			return None

		if preserved_indices is None:
			return state_ps

		new_spec = [None] * len(state_ps)
		for idx in preserved_indices:
			new_spec[idx] = state_ps[idx] if clone_ps is None else clone_ps[idx]

		return Ps(*new_spec)

	def __call__(self, *args, **kwargs) -> AttentionOutput:
		"""
		Executes the appropriate forward method based on the backend in metadata.

		Overrides `BaseOperation.__call__` to dispatch based on `self.metadata.backend`
		instead of the global `jax.default_backend()`. This allows forcing a specific
		path (e.g., GPU path even if JAX defaults to CPU) based on the configuration
		stored in `AttentionMetadata`.

		Args:
		    *args: Positional arguments to pass to the forward method.
		    **kwargs: Keyword arguments to pass to the forward method.

		Returns:
		    An `AttentionOutput` object containing the results.

		Raises:
		    RuntimeError: If the backend specified in `self.metadata` is unknown.
		"""
		match self.metadata.backend:
			case EasyDeLBackends.TPU:
				logger.debug("Calling into TPU exec")
				return self.forward_tpu(*args, **kwargs)
			case EasyDeLBackends.GPU:
				logger.debug("Calling into GPU exec")
				return self.forward_gpu(*args, **kwargs)
			case EasyDeLBackends.CPU:
				logger.debug("Calling into CPU exec")
				return self.forward_native(*args, **kwargs)
			case _:
				raise RuntimeError(f"unknown backend at AttentionImpl! {self.metadata.backend}")


class AttentionRegistry:
	"""
	Registry for discovering and managing different `AttentionImpl` classes.

	Allows registering implementations using a decorator and retrieving or
	instantiating them by name.
	"""

	_registry: tp.Dict[str, tp.Type[AttentionImpl]] = {}

	@classmethod
	def register(cls, impl_cls: tp.Type[AttentionImpl]) -> tp.Type[AttentionImpl]:
		"""
		Class method decorator to register an `AttentionImpl` subclass.

		The implementation is registered under the name(s) returned by its
		`get_impl_name()` class method.

		Example:
		```python
		@AttentionRegistry.register
		class FlashAttentionImpl(AttentionImpl):
		  @classmethod
		  def get_impl_name(cls) -> str:
		    return "flash"

		  # ... implementation ...
		```

		Args:
		    impl_cls: The `AttentionImpl` subclass to register.

		Returns:
		    The registered class itself.
		"""

		impl_names = impl_cls.get_impl_name()
		if not isinstance(impl_names, (list, tuple)):
			impl_names = [impl_names]

		for impl_name in impl_names:
			if impl_name in cls._registry:
				logger.warning(
					f"Attention implementation '{impl_name}' already registered. Overwriting."
				)
			cls._registry[impl_name] = impl_cls
			logger.debug(f"Registered attention implementation: {impl_name}")
		return impl_cls

	@classmethod
	def get(cls, impl_name: str) -> tp.Type[AttentionImpl]:
		"""
		Retrieves an attention implementation class by its registered name.

		Args:
		    impl_name: The name of the implementation to retrieve.

		Returns:
		    The `AttentionImpl` subclass registered under the given name.

		Raises:
		    ValueError: If no implementation is registered with that name.
		"""
		if impl_name not in cls._registry:
			raise ValueError(
				f"Attention implementation '{impl_name}' not found. Available implementations: {list(cls._registry.keys())}"
			)
		return cls._registry[impl_name]

	@classmethod
	def create(cls, impl_name: str, metadata: AttentionMetadata) -> AttentionImpl:
		"""
		Creates an instance of an attention implementation by name.

		Retrieves the class associated with `impl_name` and initializes it
		with the provided `metadata`.

		Args:
		    impl_name: The name of the implementation to instantiate.
		    metadata: The `AttentionMetadata` to pass to the implementation's constructor.

		Returns:
		    An initialized instance of the requested `AttentionImpl` subclass.

		Raises:
		    ValueError: If no implementation is registered with `impl_name`.
		"""
		impl_cls = cls.get(impl_name)
		return impl_cls(metadata)

	@classmethod
	def list_implementations(cls) -> tp.List[str]:
		"""
		Returns a list of names of all registered attention implementations.

		Returns:
		    A list of strings, where each string is a registered implementation name.
		"""
		return list(cls._registry.keys())
