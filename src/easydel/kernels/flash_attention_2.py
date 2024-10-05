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

from typing import Literal, Optional

import chex
from jax.experimental.pallas.ops.tpu.flash_attention import (
	BlockSizes as TPUBlockSizes,
)
from jax.experimental.pallas.ops.tpu.flash_attention import (
	flash_attention as pallas_flash_attention_tpu,
)
from jax.extend.backend import get_backend

from easydel.kernels.cpu_ops import jax_flash_attn_2_mu
from easydel.kernels.gpu_ops import pallas_flash_attn_2_gpu, triton_flash_attn_2_gpu

AVAILABLE_FLASH_ATTENTION2_BACKENDS = Literal["triton", "pallas", "jax"]
PLATFORM = get_backend().platform


def flash_attention2(
	query: Optional[chex.Array],
	key: Optional[chex.Array],
	value: Optional[chex.Array],
	bias: Optional[chex.Array] = None,
	softmax_scale: Optional[float] = None,
	blocksize_q: int = 128,
	blocksize_k: int = 128,
	backend: AVAILABLE_FLASH_ATTENTION2_BACKENDS = ...,
	platform=...,
):
	if isinstance(platform, Ellipsis):
		platform = PLATFORM
	if isinstance(backend, Ellipsis):
		match platform:
			case "gpu":
				backend = "triton"
			case "cpu":
				backend = "jax"
			case "tpu":
				backend = "pallas"
			case _:
				backend = ...

	if isinstance(backend, Ellipsis):
		raise NotImplementedError(
			f"there's no available backend for platform {platform}"
		)
	match platform:
		case "gpu":
			match backend:
				case "triton":
					return triton_flash_attn_2_gpu(
						query=query,
						key=key,
						value=value,
						bias=bias,
						softmax_scale=softmax_scale,
						blocksize_k=blocksize_k,
						blocksize_q=blocksize_q,
					)
				case "pallas":
					return pallas_flash_attn_2_gpu(
						q=query,
						k=key,
						v=value,
						b=bias,
						qblock=blocksize_q,
						kblock=blocksize_k,
						softmax_scale=softmax_scale,
					)
				case "jax":
					return jax_flash_attn_2_mu(
						query_state=query,
						key_state=key,
						value_state=value,
						bias=bias,
						q_block=blocksize_q,
						k_block=blocksize_k,
					)
		case "cpu":
			match backend:
				case "jax":
					return jax_flash_attn_2_mu(
						query_state=query,
						key_state=key,
						value_state=value,
						bias=bias,
						q_block=blocksize_q,
						k_block=blocksize_k,
					)
		case "tpu":
			match backend:
				case "jax":
					return jax_flash_attn_2_mu(
						query_state=query,
						key_state=key,
						value_state=value,
						bias=bias,
						q_block=blocksize_q,
						k_block=blocksize_k,
					)
				case "pallas":
					return pallas_flash_attention_tpu(
						q=query.transpose(0, 2, 1, 3),
						k=key.transpose(0, 2, 1, 3),
						v=value.transpose(0, 2, 1, 3),
						ab=bias,
						sm_scale=softmax_scale,
						block_sizes=TPUBlockSizes(
							block_q=blocksize_q,
							block_k_major=blocksize_k,
							block_k=blocksize_k,
							block_b=1,
							block_q_major_dkv=blocksize_q,
							block_k_major_dkv=blocksize_k,
							block_k_dkv=blocksize_k,
							block_q_dkv=blocksize_q,
							block_k_major_dq=blocksize_k,
							block_k_dq=blocksize_k,
							block_q_dq=blocksize_q,
						),
					).transpose(0, 2, 1, 3)
	raise NotImplementedError(f"NotImplemented {platform}-{backend}")


def gpu_fwd_test():
	import flax
	import flax.linen
	import flax.linen.attention
	import jax
	from jax import numpy as jnp
	from jax import random as jrnd

	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, QS, KS, D = 1, 32, 1, 128_000, 128
	blocksize_k = 64
	blocksize_q = 128
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, H, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, H, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, H, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, H, QS, KS), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if False
		else None
	)
	print("QKV Allocated")
	try:
		co = flash_attention2(
			q, k, v, b, None, blocksize_k, blocksize_q
		)  # passes 256K on 24G GPU 3090
		print(co[-1, -1, -1, :5])
	except Exception as er:
		print("Flash OOM", er)
		co = None
	try:
		fo = flax.linen.attention.dot_product_attention(q, k, v, b)
		print(fo[-1, -1, -1, :5])
	except Exception as er:
		print("Flax OOM", er)
		fo = None
	if fo is not None and co is not None:
		print(jnp.allclose(co, fo, 0, 0.125))
