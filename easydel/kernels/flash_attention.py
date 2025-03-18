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
import warnings
from enum import Enum
from functools import partial

import chex
import einops
import flax
import flax.nnx
import jax
import jax.numpy as jnp
from jax import random as jrnd
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes as TPUBlockSizes
from jax.experimental.pallas.ops.tpu.flash_attention import (
	flash_attention as pallas_flash_attention_tpu,
)
from jax.extend.backend import get_backend

from easydel.utils import traversals as etr

from .cpu_ops import jax_flash_attention
from .gpu_ops import triton_flash_attention

AVAILABLE_FLASH_ATTENTION2_PLATFORMS = tp.Literal["triton", "pallas", "jax"]
AVAILABLE_BACKENDS = tp.Literal["gpu", "tpu", "cpu"]


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


class Backend(str, Enum):
	"""Supported compute backends."""

	GPU = "gpu"
	TPU = "tpu"
	CPU = "cpu"


class Platform(str, Enum):
	"""Supported Flash Attention platforms."""

	TRITON = "triton"
	PALLAS = "pallas"
	JAX = "jax"


@etr.auto_pytree
class AttentionConfig:
	"""Configuration for Flash Attention computation."""

	blocksize_q: int = 128
	blocksize_k: int = 128
	softmax_scale: tp.Optional[float] = None
	backend: tp.Optional[Backend] = None
	platform: tp.Optional[Platform] = None

	def __post_init__(self):
		if self.backend is None:
			self.backend = Backend(get_backend().platform)

		if self.platform is None:
			self.platform = self._default_platform()

	def _default_platform(self) -> Platform:
		"""Determines the default platform based on the backend."""
		platform_map = {
			Backend.GPU: Platform.TRITON,
			Backend.CPU: Platform.JAX,
			Backend.TPU: Platform.PALLAS,
		}
		return platform_map.get(self.backend)


class FlashAttention:
	"""Flash Attention implementation with multiple backend support."""

	def __init__(self, config: tp.Optional[AttentionConfig] = None):
		self.config = config or AttentionConfig()
		self._validate_config()

	def _validate_config(self):
		"""Validates the configuration settings."""
		valid_combinations = {
			(Backend.GPU, Platform.TRITON),
			(Backend.GPU, Platform.PALLAS),
			(Backend.GPU, Platform.JAX),
			(Backend.CPU, Platform.JAX),
			(Backend.TPU, Platform.JAX),
			(Backend.TPU, Platform.PALLAS),
		}

		if (self.config.backend, self.config.platform) not in valid_combinations:
			raise ValueError(
				f"Invalid backend-platform combination: "
				f"{self.config.backend}-{self.config.platform}"
			)

	@staticmethod
	def repeat_kv_heads(
		key: chex.Array, value: chex.Array, num_reps: int
	) -> tp.Tuple[chex.Array, chex.Array]:
		"""Repeats key and value heads to match query heads."""
		return (
			einops.repeat(key, "b s h d -> b s (h r) d", r=num_reps),
			einops.repeat(value, "b s h d -> b s (h r) d", r=num_reps),
		)

	def _handle_kvhead(
		self,
		array: chex.Array,
		num_q_heads: int,
		num_kv_heads: int,
	) -> tp.Optional[chex.Array]:
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

	def __call__(
		self,
		query: chex.Array,
		key: chex.Array,
		value: chex.Array,
		bias: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		dropout_prob: float = 0.0,
		causal: bool = False,
		dropout_seed: tp.Optional[int] = None,
		adjust_sharindgs: bool = False,
	) -> chex.Array:
		"""
		Computes flash attention using the configured backend and platform.
		"""
		num_q_heads = query.shape[2]
		num_kv_heads = key.shape[2]

		if num_q_heads % num_kv_heads != 0:
			raise ValueError(
				f"Query heads ({num_q_heads}) must be divisible by "
				f"key/value heads ({num_kv_heads})"
			)

		if bias is not None:
			bias = self._handle_kvhead(bias, num_q_heads, num_kv_heads)

		kw = dict(
			query=query,
			key=key,
			value=value,
			bias=bias,
			adjust_sharindgs=adjust_sharindgs,
			attention_mask=attention_mask,
			causal=causal,
			dropout_prob=dropout_prob,
			dropout_seed=dropout_seed,
		)
		if self.config.platform == Platform.TRITON:
			return self._compute_triton(**kw)
		elif self.config.platform == Platform.PALLAS:
			return self._compute_pallas(**kw)
		else:  # Platform.JAX
			return self._compute_jax(**kw)

	def _compute_triton(
		self,
		query: chex.Array,
		key: chex.Array,
		value: chex.Array,
		bias: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		dropout_prob: float = 0.0,
		causal: bool = False,
		dropout_seed: tp.Optional[int] = None,
		adjust_sharindgs: bool = False,
	) -> chex.Array:
		"""Computes attention using Triton backend."""
		if adjust_sharindgs:
			query_sharding = query.sharding if hasattr(query, "sharding") else None
			target_gpu_idx = int(os.getenv("GPU_IDX_FLASH_ATTN", free_gpu_in_process()))
			devices = jax.local_devices(process_index=jax.process_index(), backend="gpu")
			target_device = devices[target_gpu_idx]
			query = jax.device_put(query, target_device)
			key = jax.device_put(key, target_device)
			value = jax.device_put(value, target_device)
			if bias is not None:
				bias = jax.device_put(bias, target_device)
			if attention_mask is not None:
				attention_mask = jax.device_put(attention_mask, target_device)

		attn = triton_flash_attention(
			q=query,
			k=key,
			v=value,
			bias=bias,
			attention_mask=attention_mask,
			dropout_prob=dropout_prob,
			causal=causal,
			dropout_seed=dropout_seed,
			softmax_scale=self.config.softmax_scale,
		)

		if adjust_sharindgs and query_sharding is not None:
			attn = jax.device_put(attn, query_sharding)
		return attn

	def _compute_pallas(
		self,
		query: chex.Array,
		key: chex.Array,
		value: chex.Array,
		bias: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		dropout_prob: float = 0.0,
		causal: bool = False,
		dropout_seed: tp.Optional[int] = None,
		adjust_sharindgs: bool = False,
	) -> chex.Array:
		"""Computes attention using Pallas backend."""

		if self.config.backend == Backend.GPU:
			warnings.warn(
				"Pallas-FlashAttention has been deprecated on GPUs (triton backend will be used)",
				stacklevel=1,
			)
			return self._compute_triton(
				query=query,
				key=key,
				value=value,
				bias=bias,
				adjust_sharindgs=adjust_sharindgs,
				attention_mask=attention_mask,
				dropout_prob=dropout_prob,
				causal=causal,
				dropout_seed=dropout_seed,
			)

		key, value = self.repeat_kv_heads(key, value, query.shape[2] // key.shape[2])
		query_lenght = query.shape[1]
		value_lenght = value.shape[1]
		if bias is not None:
			if bias.shape[1] != value.shape[2]:
				bias = jnp.repeat(bias, value.shape[2] // bias.shape[1], 1)
		# TPU implementation
		block_sizes = TPUBlockSizes(
			block_q=min(self.config.blocksize_q, query_lenght),
			block_k_major=min(self.config.blocksize_k, value_lenght),
			block_k=min(self.config.blocksize_k, value_lenght),
			block_b=1,
			block_q_major_dkv=min(self.config.blocksize_q, query_lenght),
			block_k_major_dkv=min(self.config.blocksize_k, value_lenght),
			block_k_dkv=min(self.config.blocksize_k, value_lenght),
			block_q_dkv=min(self.config.blocksize_q, query_lenght),
			block_k_major_dq=min(self.config.blocksize_k, value_lenght),
			block_k_dq=min(self.config.blocksize_k, value_lenght),
			block_q_dq=min(self.config.blocksize_q, query_lenght),
		)
		if bias is None and attention_mask is not None:
			bias = jnp.where(attention_mask, 0, jnp.finfo(query.dtype).min)
		return partial(
			pallas_flash_attention_tpu,
			sm_scale=self.config.softmax_scale,
			block_sizes=block_sizes,
			causal=causal,
		)(
			query.transpose(0, 2, 1, 3),
			key.transpose(0, 2, 1, 3),
			value.transpose(0, 2, 1, 3),
			bias,
		).transpose(0, 2, 1, 3)

	def _compute_jax(
		self,
		query: chex.Array,
		key: chex.Array,
		value: chex.Array,
		bias: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		dropout_prob: float = 0.0,
		causal: bool = False,
		dropout_seed: tp.Optional[int] = None,
		adjust_sharindgs: bool = False,
	) -> chex.Array:
		"""Computes attention using JAX backend."""
		key, value = self.repeat_kv_heads(key, value, query.shape[2] // key.shape[2])

		if bias is None and attention_mask is not None:
			bias = jnp.where(attention_mask, 0, jnp.finfo(query.dtype).min)

		return jax_flash_attention(
			query_state=query,
			key_state=key,
			value_state=value,
			mask=None,
			bias=bias,
			blocksize_q=self.config.blocksize_q,
			blocksize_k=self.config.blocksize_k,
			dtype=query.dtype,
			softmax_scale=self.config.softmax_scale,
			dropout=dropout_prob,
		)


def create_flash_attention(
	backend: tp.Optional[tp.Union[Backend, str]] = None,
	platform: tp.Optional[tp.Union[Platform, str]] = None,
	**kwargs,
) -> FlashAttention:
	"""
	Factory function to create a FlashAttention instance with the specified configuration.

	Args:
	    backend: Compute backend to use (GPU, TPU, or CPU)
	    platform: Platform to use (Triton, Pallas, or JAX)
	    **kwargs: Additional configuration parameters for AttentionConfig

	Returns:
	    Configured FlashAttention instance
	"""
	if isinstance(backend, str):
		backend = Backend(backend)
	if isinstance(platform, str):
		platform = Platform(platform)

	config = AttentionConfig(backend=backend, platform=platform, **kwargs)
	return FlashAttention(config)


def _attn_reference(query_states, key_states, value_states, bias):
	b, qs, num_q_heads, d = query_states.shape
	num_kv_heads = value_states.shape[2]
	ks = value_states.shape[1]
	query_states = jnp.reshape(
		query_states,
		(b, qs, num_kv_heads, num_q_heads // num_kv_heads, d),
	)

	query_states = query_states * (d**-0.5)
	attention_weight = jnp.einsum(
		"bskhd,bmkd->bkhsm",
		query_states,
		key_states,
	)

	if bias is not None:
		if bias.shape[1] == num_q_heads:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, num_kv_heads, num_q_heads // num_kv_heads, qs, ks),
			)
		elif bias.shape[1] == num_kv_heads:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, num_kv_heads, 1, qs, ks),
			)
		elif bias.shape[1] == 1:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, 1, 1, qs, ks),
			)
		else:
			raise NotImplementedError("bias heads wont match!")

	attention_weight = jax.nn.softmax(attention_weight)

	return jnp.einsum(
		"bkhsm,bmkd->bskhd",
		attention_weight,
		value_states,
	).reshape(b, qs, num_q_heads, d)


def _test_backward():
	"""Tests the backward pass of the attention mechanism."""
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, QH, KH, QS, KS, D = 1, 32, 32, 2048, 2048, 128
	use_bias = True
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, QH, D), dtype="f2")
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, KH, D), dtype="f2")
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, KH, D), dtype="f2")
	a = jnp.asarray(jrnd.randint(v_key, (B, 1, QS, KS), 0, 4) > 2, "b1")
	b = jnp.where(a, 0, jnp.finfo(q.dtype).min) if use_bias else None

	attention = create_flash_attention()
	try:
		co = jax.grad(lambda *x: attention(*x).sum())(q, k, v, None, a)
		print("Custom op backward pass gradients:")
		print(co[0, 0, 0, :5])  # Print last 5 elements of last head of last batch
	except Exception as er:
		print(f"Custom op backward pass failed: {er}")
		co = None

	try:
		fo = jax.grad(lambda *x: flax.nnx.dot_product_attention(*x).sum())(q, k, v, b)

		print(fo[0, 0, 0, :5])  # Print last 5 elements of last head of last batch
	except Exception as e:
		print(f"Flax backward pass failed : {e}")
		fo = None
		exit()

	if fo is not None and co is not None:
		if jnp.allclose(co, fo, atol=0.125):
			print("Backward pass results are close.")
		else:
			print("Backward pass results differ significantly!")


def _test_forward():
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, QH, KH, QS, KS, D = 1, 32, 8, 2048, 2048, 128
	use_bias = True

	q = jax.nn.initializers.normal(2)(q_key, (B, QS, QH, D), dtype="f2")
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, KH, D), dtype="f2")
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, KH, D), dtype="f2")
	a = jnp.asarray(jrnd.randint(v_key, (B, 1, QS, KS), 0, 4) > 2, "b1")
	b = jnp.where(a, 0, jnp.finfo(q.dtype).min) if use_bias else None
	attention = create_flash_attention()
	print("QKV Allocated")
	try:
		co = attention(q, k, v, None, a)
		print(co[0, 0, 0, :5])
	except Exception as er:
		print("Flash OOM", er)
		co = None
	try:
		fo = _attn_reference(q, k, v, b)
		print(fo[0, 0, 0, :5])
	except Exception as er:
		print("Flax OOM", er)
		fo = None
	if fo is not None and co is not None:
		print(jnp.allclose(co, fo, 0, 0.125))


if __name__ == "__main__":
	_test_forward()
	_test_backward()
