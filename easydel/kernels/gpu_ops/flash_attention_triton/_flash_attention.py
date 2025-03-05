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
import typing as tp

import chex
import jax

from ._backward_triton import _bwd_attention_kernel_call
from ._forward_triton import _fwd_attention_kernel_call

DEV_MODE = True


def _jax_fwd_attention_call(
	q: tp.Optional[chex.Array],
	k: tp.Optional[chex.Array],
	v: tp.Optional[chex.Array],
	attention_mask: tp.Optional[chex.Array] = None,
	bias: tp.Optional[chex.Array] = None,
	softmax_scale: tp.Optional[float] = None,
	dropout_prob: float = 0.0,
	causal: bool = False,
	dropout_seed: tp.Optional[int] = None,
	varlen_mode: bool = True,
):
	out, lse = _fwd_attention_kernel_call(
		q=q,
		k=k,
		v=v,
		attention_mask=attention_mask,
		bias=bias,
		softmax_scale=softmax_scale,
		dropout_prob=dropout_prob,
		causal=causal,
		dropout_seed=dropout_seed,
		varlen_mode=varlen_mode,
	)
	residual = (
		q,
		k,
		v,
		bias,
		attention_mask,
		out,
		lse,
		dropout_seed,
	)
	return out, residual


def _jax_bwd_attention_call(
	softmax_scale: tp.Optional[float],
	dropout_prob: float,
	causal: bool,
	varlen_mode: bool,
	residual: tp.Tuple[chex.Array],
	dO: chex.Array,
):
	q, k, v, bias, attention_mask, out, lse, dropout_seed = residual
	dq, dk, dv = _bwd_attention_kernel_call(
		dO=dO,
		q=q,
		k=k,
		v=v,
		bias=bias,
		attention_mask=attention_mask,
		o=out,
		M=lse,
		dropout_prob=dropout_prob,
		causal=causal,
		dropout_seed=dropout_seed,
		softmax_scale=softmax_scale,
		varlen_mode=varlen_mode,
	)
	return dq, dk, dv, None, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 9))
@functools.partial(jax.jit, static_argnums=(5, 6, 7, 9))
def flash_attention_call(
	q: tp.Optional[chex.Array],
	k: tp.Optional[chex.Array],
	v: tp.Optional[chex.Array],
	attention_mask: tp.Optional[chex.Array] = None,
	bias: tp.Optional[chex.Array] = None,
	softmax_scale: tp.Optional[float] = None,
	dropout_prob: float = 0.0,
	causal: bool = False,
	dropout_seed: tp.Optional[int] = None,
	varlen_mode: bool = True,
) -> chex.Array:
	return _fwd_attention_kernel_call(
		q=q,
		k=k,
		v=v,
		attention_mask=attention_mask,
		bias=bias,
		softmax_scale=softmax_scale,
		dropout_prob=dropout_prob,
		causal=causal,
		dropout_seed=dropout_seed,
		varlen_mode=varlen_mode,
	)[0]


flash_attention_call.defvjp(
	_jax_fwd_attention_call,
	_jax_bwd_attention_call,
)


def flash_attention(
	q: tp.Optional[chex.Array],
	k: tp.Optional[chex.Array],
	v: tp.Optional[chex.Array],
	attention_mask: tp.Optional[chex.Array] = None,
	bias: tp.Optional[chex.Array] = None,
	softmax_scale: tp.Optional[float] = None,
	dropout_prob: float = 0.0,
	causal: bool = False,
	dropout_seed: tp.Optional[int] = None,
	varlen_mode: bool = True,
) -> chex.Array:
	del varlen_mode  # TODO: Debug varlen mode
	return flash_attention_call(
		q=q,
		k=k,
		v=v,
		attention_mask=attention_mask,
		bias=bias,
		softmax_scale=softmax_scale,
		dropout_prob=dropout_prob,
		causal=causal,
		dropout_seed=dropout_seed,
		varlen_mode=False,
	)
