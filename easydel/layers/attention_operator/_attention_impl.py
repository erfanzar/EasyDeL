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

import dataclasses
import typing as tp
from abc import ABC, abstractmethod
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

logger = get_logger("EasyDeL-AttentionOperator")


@etr.auto_pytree
class AttentionOutput:
	attention_weights: tp.Optional[Array] = None
	attention_outputs: tp.Optional[Array] = None


class RuntimeType(Enum):
	normal = "normal"
	generation = "generation"


@etr.auto_pytree
class AttentionMetadata:
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
	):
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
		if not hasattr(self, attr_name) or getattr(self, attr_name, ...) == Ellipsis:
			pn = attr_name if pickup_name is None else pickup_name
			new_value = (
				default
				if (self.base_config is None or not use_base_config)
				else getattr(self.base_config, pn, default)
			)
			setattr(self, attr_name, new_value)


class AttentionImpl(ABC):
	def __init__(self, metadata: AttentionMetadata) -> None:
		self.metadata = metadata

	@abstractmethod
	def forward_native(self, *args, **kwargs) -> AttentionOutput: ...

	@abstractmethod
	def forward_tpu(self, *args, **kwargs) -> AttentionOutput: ...
	@abstractmethod
	def forward_cpu(self, *args, **kwargs) -> AttentionOutput: ...
	@abstractmethod
	def forward_gpu(self, *args, **kwargs) -> AttentionOutput: ...

	@abstractmethod
	def forward_rocm(self, *args, **kwargs) -> AttentionOutput: ...
	@abstractmethod
	def forward_cuda(self, *args, **kwargs) -> AttentionOutput: ...

	@classmethod
	@abstractmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str]]: ...
	@abstractmethod
	def get_impl_metadata(self) -> AttentionMetadata: ...

	def get_runtime_type(self, q: jax.Array, BTHD: bool = True) -> RuntimeType:
		ingeneration = q.shape[1] == 1 if BTHD else q.shape[2] == 1
		return RuntimeType.generation if ingeneration else RuntimeType.normal

	def current_backend(self) -> tp.Literal["tpu", "gpu", "cpu"]:
		return jax.default_backend()

	@staticmethod
	def _split_attention_mask(attn_mask: Array) -> tp.Tuple[Array, Array]:
		"""
		Takes an attention mask and splits it into query mask and key-value mask.
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
		Takes separate query and key-value masks and combines them into an attention mask.
		"""
		if kv_mask.ndim == 2:
			kv_mask = kv_mask[:, None, :]
		if q_mask.ndim == 2:
			q_mask = q_mask[:, :, None]
		return q_mask * kv_mask

	@staticmethod
	def _create_causal_mask(qseq) -> Array:
		return jnp.tril(jnp.ones((qseq, qseq), dtype="b1"))

	@staticmethod
	def repeat_kv_heads(
		k: Array,
		v: Array,
		num_reps: int,
	) -> tp.Tuple[Array, Array]:
		"""Repeats k and v heads to match q heads."""
		return (
			einops.repeat(k, "b s h d -> b s (h r) d", r=num_reps),
			einops.repeat(v, "b s h d -> b s (h r) d", r=num_reps),
		)

	def _handle_kvhead(
		self,
		array: Array,
		num_q_heads: int,
		num_kv_heads: int,
	) -> tp.Optional[Array]:
		"""Processes attention bias based on head configuration."""
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
	"""Registry for attention implementations."""

	_registry: tp.Dict[str, tp.Type[AttentionImpl]] = {}

	@classmethod
	def register(cls, impl_cls: tp.Type[AttentionImpl]) -> tp.Type[AttentionImpl]:
		"""
		Decorator to register an attention implementation.

		Example usage:

		@AttentionRegistry.register
		class CustomAttention(AttentionImpl):
		    ...
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
		"""Get an attention implementation by name."""
		if impl_name not in cls._registry:
			raise ValueError(
				f"Attention implementation '{impl_name}' not found. Available implementations: {list(cls._registry.keys())}"
			)
		return cls._registry[impl_name]

	@classmethod
	def create(cls, impl_name: str, metadata: AttentionMetadata) -> AttentionImpl:
		"""Create an instance of an attention implementation by name."""
		impl_cls = cls.get(impl_name)
		return impl_cls(metadata)

	@classmethod
	def list_implementations(cls) -> tp.List[str]:
		"""List all registered attention implementations."""
		return list(cls._registry.keys())
