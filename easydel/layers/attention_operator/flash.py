import functools
import typing as tp

import einops
import jax
from eformer.escale import with_sharding_constraint
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.shard_map import shard_map
from easydel.kernels.gpu_ops import triton_flash_attention
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes
from jax.experimental.pallas.ops.tpu.flash_attention import (
	flash_attention as pallas_flash_attention,
)
from ._attention_impl import (
	AttentionImpl,
	AttentionMetadata,
	AttentionOutput,
)


def get_device_memory_usage(device: jax.Device) -> float:
	"""
	Get the memory usage for a specific JAX device using local_devices stats.

	Args:
	    device: JAX device to check
	Returns:
	    float: Memory usage in bytes
	"""
	try:
		memory_stats = device.memory_stats()
		return memory_stats["bytes_in_use"] if memory_stats else float("inf")
	except:  # noqa
		return float("inf")


def free_gpu_in_process() -> int:
	"""
	Returns the index of the GPU with the most available memory using JAX local_devices.

	Returns:
	    int: Index of the GPU with most free memory
	"""
	devices = jax.local_devices()
	gpu_devices = [d for d in devices if d.platform == "gpu"]

	if not gpu_devices:
		return 0

	memory_usage = [get_device_memory_usage(device) for device in gpu_devices]
	return memory_usage.index(min(memory_usage))


class FlashAttn(AttentionImpl):
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

	def get_impl_name(self) -> str:
		return "flash_attn2"

	def get_impl_metadata(self) -> AttentionMetadata:
		return self.metadata

	def forward_native(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		causal: bool = False,
	) -> AttentionOutput:
		raise NotImplementedError("we wont call cpu impl of flash attention")

	def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_cuda(*args, **kwargs)

	def forward_tpu(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		causal: bool = False,
	) -> AttentionOutput:
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		dtype = self.metadata.runtime_dtype
		runtime_type = self.get_runtime_type(q=q, BTHD=False)

		(
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			mask_partition_spec,
			attention_partition_spec,
		) = self.metadata.get_partition_specs(runtime_type, BTHD=False)

		if mask is None and bias is None and init_bias is not None:
			bias = init_bias()

		if bias is None and mask is not None:
			bias = jnp.where(mask, 0, jnp.finfo(q.dtype).min)
		k, v = self.repeat_kv_heads(k, v, q.shape[2] // k.shape[2])
		query_lenght = q.shape[1]
		value_lenght = v.shape[1]
		if bias is not None:
			if bias.shape[1] != v.shape[2]:
				bias = jnp.repeat(bias, v.shape[2] // bias.shape[1], 1)

		block_sizes = BlockSizes(
			block_q=min(self.metadata.blocksize_q, query_lenght),
			block_k_major=min(self.metadata.blocksize_k, value_lenght),
			block_k=min(self.metadata.blocksize_k, value_lenght),
			block_b=1,
			block_q_major_dkv=min(self.metadata.blocksize_q, query_lenght),
			block_k_major_dkv=min(self.metadata.blocksize_k, value_lenght),
			block_k_dkv=min(self.metadata.blocksize_k, value_lenght),
			block_q_dkv=min(self.metadata.blocksize_q, query_lenght),
			block_k_major_dq=min(self.metadata.blocksize_k, value_lenght),
			block_k_dq=min(self.metadata.blocksize_k, value_lenght),
			block_q_dq=min(self.metadata.blocksize_q, query_lenght),
		)
		attn = shard_map(
			functools.partial(
				pallas_flash_attention,
				sm_scale=sm_scale,
				block_sizes=block_sizes,
				causal=causal,
			),
			mesh=self.metadata.mesh,
			in_specs=(
				self.create_stable_sharding(query_partition_spec, [0, 3], dep=q),
				self.create_stable_sharding(key_partition_spec, [0, 3], dep=k),
				self.create_stable_sharding(value_partition_spec, [0, 3], dep=v),
				self.create_stable_sharding(bias_partition_spec, dep=bias),
			),
			out_specs=self.create_stable_sharding(attention_partition_spec, [0, 3]),
			check_rep=False,
		)(
			q.transpose(0, 2, 1, 3).astype(dtype),
			k.transpose(0, 2, 1, 3).astype(dtype),
			v.transpose(0, 2, 1, 3).astype(dtype),
			bias.astype(dtype) if bias is not None else bias,
		).transpose(0, 2, 1, 3)

		return AttentionOutput(
			attention_weights=None,
			attention_outputs=with_sharding_constraint(
				arr=attn,
				sharding=attention_partition_spec,
			),
		)

	def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)

	def forward_cuda(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		causal: bool = False,
	) -> AttentionOutput:
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		dtype = self.metadata.runtime_dtype
		runtime_type = self.get_runtime_type(q=q, BTHD=True)

		(
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			mask_partition_spec,
			attention_partition_spec,
		) = self.metadata.get_partition_specs(runtime_type, BTHD=True)
		if mask is None and bias is None and init_bias is not None:
			bias = init_bias()
		func = functools.partial(
			triton_flash_attention,
			dropout_prob=self.metadata.dropout_prob,
			dropout_seed=None,
			softmax_scale=self.metadata.softmax_scale,
			causal=causal,
		)
		attn = shard_map(
			func,
			mesh=self.metadata.mesh,
			in_specs=(
				self.create_stable_sharding(query_partition_spec, [0, 2], dep=q),
				self.create_stable_sharding(key_partition_spec, [0, 2], dep=k),
				self.create_stable_sharding(value_partition_spec, [0, 2], dep=v),
				self.create_stable_sharding(bias_partition_spec, dep=bias),
				self.create_stable_sharding(mask_partition_spec, dep=mask),
			),
			out_specs=self.create_stable_sharding(attention_partition_spec, [0, 2]),
			check_rep=False,
		)(
			q.astype(dtype),
			k.astype(dtype),
			v.astype(dtype),
			mask.astype(dtype) if mask is not None else bias,
			bias.astype("b1") if bias is not None else bias,
		)
		return AttentionOutput(
			attention_weights=None,
			attention_outputs=with_sharding_constraint(
				arr=attn,
				sharding=attention_partition_spec,
			),
		)

	def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)


if __name__ == "__main__":
	from easydel.infra import EasyDeLBaseConfig

	# Test cace when qkv might refer to mla
	b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 8, 128, 128
	q = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
	k = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
	v = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
	a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")

	gpu_attn = FlashAttn(
		AttentionMetadata(
			runtime_dtype=jnp.float16,
			base_config=EasyDeLBaseConfig(),
			backend="gpu",
		)
	)
	cpu_attn = FlashAttn(
		AttentionMetadata(
			runtime_dtype=jnp.float16,
			base_config=EasyDeLBaseConfig(),
			backend="cpu",
		)
	)
	tpu_attn = FlashAttn(
		AttentionMetadata(
			runtime_dtype=jnp.bfloat16,
			base_config=EasyDeLBaseConfig(),
			backend="tpu",
		)
	)

	try:
		cout = cpu_attn(q=q, k=k, v=v, mask=a).attention_outputs
	except NotImplementedError as e:
		print(e)

	try:
		gout = gpu_attn(q=q, k=k, v=v, mask=a).attention_outputs
	except NotImplementedError as e:
		print(e)

	try:
		tout = tpu_attn(q=q, k=k, v=v, mask=a).attention_outputs
	except NotImplementedError as e:
		print(e)
