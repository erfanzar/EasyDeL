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

import math
from functools import partial
from typing import Optional

import jax
from fjformer import GenerateRNG
from flax.linen.attention import dot_product_attention as _dot_product_attention
from jax import custom_jvp, numpy as jnp
from jax import random as jrand
from jax.experimental import pallas as pl
from jax.lib import xla_bridge

PLATFORM = xla_bridge.get_backend().platform
INTERPRET = PLATFORM == "cpu"

rng = GenerateRNG()


def _gpu_fwd_flash_attn_kernel(
	q_ref,
	k_ref,
	v_ref,
	b_ref,
	o_ref,
	lse_ref,
	*,
	dtype,
	qblock,
	kblock,
	seqlen_q_rounded,
	softmax_scale,
	nheads,
	block_headdim,
): ...


def _call_gpu_fwd_flash_attn(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	b: Optional[jax.Array],
	dtype: jnp.dtype = jnp.float32,
	qblock: int = 128,
	kblock: int = 128,
	softmax_scale: float = None,
):
	batch_size, seqlen_q, nheads, dim = q.shape

	softmax_scale = softmax_scale or 1.0 / math.sqrt(dim)

	seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

	lse_shape = (batch_size, nheads, seqlen_q_rounded)
	in_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(k.shape, lambda *_: (0,) * k.ndim),
		pl.BlockSpec(v.shape, lambda *_: (0,) * v.ndim),
	]
	in_specs.append(
		None if b is None else pl.BlockSpec(b.shape, lambda *_: (0,) * b.ndim)
	)

	out_shape = [
		jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
		jax.ShapeDtypeStruct(shape=lse_shape, dtype=q.dtype),
	]

	out_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(lse_shape, lambda *_: (0,) * len(lse_shape)),
	]
	grid = (pl.cdiv(seqlen_q, qblock), batch_size, nheads)

	method = pl.pallas_call(
		f=partial(
			_gpu_fwd_flash_attn_kernel,
			dtype=dtype,
			qblock=qblock,
			kblock=kblock,
			seqlen_q_rounded=seqlen_q_rounded,
			softmax_scale=softmax_scale,
			nheads=nheads,
			block_headdim=dim,
		),
		grid=grid,
		out_specs=out_specs,
		in_specs=in_specs,
		out_shape=out_shape,
		compiler_params=dict(triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)),
		interpret=True,
		debug=False,
	)

	o, lse = method(q, k, v, b)
	return o, (o, lse)


def _call_gpu_bwd_flash_attn(
	dtype: jnp.dtype,
	qblock: int,
	kblock: int,
	softmax_scale: float,
	res,
	dO,
): ...


@partial(custom_jvp, nondiff_argnums=(4, 5, 6, 7))
def _gpu_flash_attn(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	b: Optional[jax.Array],
	dtype: jnp.dtype = jnp.float32,
	qblock: int = 128,
	kblock: int = 128,
	softmax_scale: float = None,
):
	return _call_gpu_fwd_flash_attn(
		q,
		k,
		v,
		b,
		dtype,
		qblock,
		kblock,
		softmax_scale,
	)[0]


_gpu_flash_attn.defvjp(_call_gpu_fwd_flash_attn, _call_gpu_bwd_flash_attn)


def main():
	b, h, qs, s, d = 1, 8, 64, 64, 128
	dtype = jnp.float32

	def transposehs(x):
		return x.transpose(0, 2, 1, 3)

	q = jrand.normal(rng.rng, shape=(b, h, qs, d), dtype=dtype)
	k = jrand.normal(rng.rng, shape=(b, h, s, d), dtype=dtype)
	v = jrand.normal(rng.rng, shape=(b, h, s, d), dtype=dtype)
	b = jnp.where(
		jrand.randint(rng.rng, shape=(b, h, qs, s), minval=0, maxval=3) > 1,
		0,
		jnp.finfo(dtype).min,
	)

	excepted_result = _dot_product_attention(query=q, key=k, value=v, bias=b)
	result = _call_gpu_fwd_flash_attn(
		q,
		k,
		v,
		b,
		dtype=dtype,
		qblock=32,
		kblock=32,
	)[0]
	print(jnp.allclose(excepted_result, result, 0.125, 0))


if __name__ == "__main__":
	main()
