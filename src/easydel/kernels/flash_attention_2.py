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
from jax import custom_vjp, numpy as jnp
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
	seqlen_q,
	seqlen_k,
	seqlen_q_rounded,
	softmax_scale,
	nheads,
	block_headdim,
):
	# grid pl.cdiv(seqlen_q, qblock), batch_size, nheads
	qg, bg, nhg = pl.program_id(0), pl.program_id(1), pl.program_id(2)
	qs_mask = qg * qblock + jnp.arange(0, qblock) < seqlen_q
	headdim_mask = jnp.arange(0, block_headdim) < q_ref.shape[3]
	qmask = qs_mask[:, None] & headdim_mask[None, :]

	q = pl.load(
		q_ref,
		idx=(bg, pl.dslice(qg * qblock, qblock), nhg, pl.dslice(None)),
		mask=qmask,
		other=0.0,
	)

	def body(carry):
		idx, acc_o, max_i, lse_i = carry
		kv_idx = (bg, pl.dslice(idx * kblock, kblock), nhg, pl.dslice(None))
		ks_mask = idx * kblock + jnp.arange(0, kblock) < seqlen_k
		kv_mask = ks_mask[:, None] & headdim_mask[None, :]
		k = pl.load(k_ref, idx=kv_idx, mask=kv_mask, other=0.0)
		qk = jnp.dot(q, k.transpose(1, 0))
		if b_ref is not None:
			bslice = (
				bg,
				nhg,
				pl.dslice(qg * qblock, qblock),
				pl.dslice(idx * kblock, kblock),
			)
			bmask = qs_mask[:, None] & ks_mask[None, :]
			b = pl.load(b_ref, idx=bslice, mask=bmask, other=0.0)
			qk = qk * softmax_scale + b
		else:
			qk = qk * softmax_scale
		max_ij = jnp.maximum(jnp.max(qk, -1), lse_i)
		p = jnp.exp(qk - max_ij[:, None])
		lse_ij = jnp.sum(p, -1)
		acc_o_scale = jnp.exp(max_i - max_ij)
		acc_o = acc_o * acc_o_scale[:, None]
		v = pl.load(v_ref, idx=kv_idx, mask=kv_mask, other=0.0)
		acc_o += jnp.dot(p, v)
		lin = jnp.exp(lse_i - max_ij) + lse_ij
		return (idx + 1, acc_o, max_ij, max_ij + jnp.log(lin))

	end_krange = pl.cdiv(seqlen_k, kblock)
	init_block = (
		0,
		jnp.zeros((qblock, block_headdim), dtype=jnp.float32),
		jnp.full((qblock,), -float("inf"), dtype=jnp.float32),
		jnp.full((qblock,), -float("inf"), dtype=jnp.float32),
	)

	i, o, max_i, lse_i = jax.lax.while_loop(
		lambda state: state[0] < end_krange, body, init_block
	)
	o_scale = jnp.exp(max_i - lse_i)
	o = o * o_scale[:, None]
	pl.store(
		o_ref,
		val=o,
		idx=(bg, pl.dslice(qg * qblock, qblock), nhg, pl.dslice(None)),
		mask=qmask,
	)


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
	seqlen_k = k.shape[1]
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
			seqlen_q=seqlen_q,
			seqlen_k=seqlen_k,
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


@partial(custom_vjp, nondiff_argnums=(4, 5, 6, 7))
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
	b, h, qs, s, d = 1, 2, 2, 2, 4
	dtype = jnp.float32

	def transposehs(x):
		return x.transpose(0, 2, 1, 3)

	q = jrand.normal(rng.rng, shape=(b, qs, h, d), dtype=dtype)
	k = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	v = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
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
	print(f"PRED : \n {result} \n")
	print(f"ORGN : \n {excepted_result} \n")
	
	print(jnp.allclose(excepted_result, result, 0.125, 0))


if __name__ == "__main__":
	main()
