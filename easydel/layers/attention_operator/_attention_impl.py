import dataclasses
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import jax
from eformer.escale import PartitionAxis
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
from easydel.utils.helpers import get_logger

logger = get_logger("EasyDeL-AttentionOperator", "DEBUG")


@dataclass
class AttentionOutput:
	attention_weights: tp.Optional[Array] = None
	attention_outputs: tp.Optional[Array] = None


class RuntimeType(Enum):
	normal = "normal"
	generation = "generation"


@dataclass
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

	def _safety_check(self):
		for field in dataclasses.fields(self):
			val = getattr(self, field.name)
			if val == Ellipsis:
				raise ValueError(f"`{field.name}` shouldn't be ellipsis")

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

	@abstractmethod
	def get_impl_name(self) -> str: ...
	@abstractmethod
	def get_impl_metadata(self) -> AttentionMetadata: ...

	def get_runtime_type(self, q: jax.Array, BTHD: bool = True) -> RuntimeType:
		ingeneration = q.shape[1] == 1 if BTHD else q.shape[2] == 1
		return RuntimeType.generation if ingeneration else RuntimeType.normal

	def current_backend(self) -> tp.Literal["tpu", "gpu", "cpu"]:
		return jax.default_backend()

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
