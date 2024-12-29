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

import math
import typing as tp

# from functools import partial
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx as nn


def _yarn_find_correction_dim(
	num_rotations: int,
	dim: int,
	base: float = 10000,
	max_position_embeddings: int = 2048,
) -> float:
	return (
		dim
		* jnp.log(
			max_position_embeddings / (num_rotations * 2 * jnp.pi),
		)
	) / (2 * jnp.log(base))


def _yarn_find_correction_range(
	low_rot: int,
	high_rot: int,
	dim: int,
	base: float = 10000,
	max_position_embeddings: int = 2048,
) -> tp.Tuple[int, int]:
	hr = jnp.ceil(
		_yarn_find_correction_dim(
			high_rot,
			dim,
			base,
			max_position_embeddings,
		)
	)
	lr = jnp.floor(
		_yarn_find_correction_dim(
			low_rot,
			dim,
			base,
			max_position_embeddings,
		)
	)
	return jax.lax.max(lr, 0.0), jax.lax.min(hr, jnp.array(dim - 1, dtype=jnp.float32))


def _yarn_linear_ramp_mask(
	low: float,
	high: float,
	dim: int,
	dtype: jnp.dtype,
) -> jnp.ndarray:
	high = jax.lax.cond(low == high, lambda x: x + 0.001, lambda x: x, high)
	linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
	ramp_func = jnp.clip(linear_func, 0, 1)
	return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
	if scale <= 1:
		return 1.0
	return 0.1 * jnp.log(scale) + 1.0


def _rotate_neox(x: jnp.ndarray) -> jnp.ndarray:
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return jnp.concatenate((-x2, x1), axis=-1)


def _rotate_gptj(x: jnp.ndarray) -> jnp.ndarray:
	x1 = x[..., ::2]
	x2 = x[..., 1::2]
	x = jnp.stack((-x2, x1), axis=-1)
	return x.reshape(x.shape[:-2] + (-1,))


def _apply_rotary_emb(
	x: jnp.ndarray,
	cos: jnp.ndarray,
	sin: jnp.ndarray,
	is_neox_style: bool,
) -> jnp.ndarray:
	"""
	Args:
	    x: [num_tokens, num_heads, head_size]
	    cos: [num_tokens, head_size // 2]
	    sin: [num_tokens, head_size // 2]
	    is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
	        positional embeddings.
	"""
	cos = cos[:, :, None].astype(x.dtype)
	sin = sin[:, :, None].astype(x.dtype)
	assert sin.ndim == x.ndim
	if is_neox_style:
		x1, x2 = jnp.split(x, 2, axis=-1)
	else:
		x1 = x[..., ::2]
		x2 = x[..., 1::2]

	o1 = x1 * cos - x2 * sin
	o2 = x2 * cos + x1 * sin

	if is_neox_style:
		return jnp.concatenate((o1, o2), axis=-1)
	else:
		return jnp.stack((o1, o2), axis=-1).reshape(x.shape)


AVAILABLE_ROPE_TYPES = {}


def rope_wraper(type):
	def w(rope: RotaryEmbedding):
		properties = {k: v for k, v in rope.__dict__.items()}
		AVAILABLE_ROPE_TYPES[type] = properties
		rope.__str__ = lambda cls: str(cls.__class__.__name__)
		rope.__repr__ = lambda cls: repr(cls.__class__.__name__)
		rope._type = type
		return rope

	return w


def compute_basic_inv_frequencies(base: int, rotary_dim: int):
	return 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))


def compute_yarn_inv_frequencies(
	base: float,
	rotary_dim: int,
	beta_fast: float,
	beta_slow: float,
	max_position_embeddings: int,
	scaling_factor: float,
	extrapolation_factor: float,
) -> jnp.ndarray:
	pos_freqs = base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
	inv_freq_extrapolation = 1.0 / pos_freqs
	inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
	low, high = _yarn_find_correction_range(
		low_rot=beta_fast,
		high_rot=beta_slow,
		dim=rotary_dim,
		base=base,
		max_position_embeddings=max_position_embeddings,
	)
	inv_frequencies_mask = (
		1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=jnp.float32)
	) * extrapolation_factor
	inv_frequencies = (
		inv_freq_interpolation * (1 - inv_frequencies_mask)
		+ inv_freq_extrapolation * inv_frequencies_mask
	)
	return inv_frequencies


def compute_llama3_inv_frequencies(
	base,
	rotary_dim,
	low_freq_factor,
	high_freq_factor,
	orig_max_position,
	scaling_factor,
):
	inv_freqs = compute_basic_inv_frequencies(base, rotary_dim)
	low_freq_wavelen = orig_max_position / low_freq_factor
	high_freq_wavelen = orig_max_position / high_freq_factor

	wave_len = 2 * jnp.pi / inv_freqs
	if low_freq_factor != high_freq_factor:
		smooth = (orig_max_position / wave_len - low_freq_factor) / (
			high_freq_factor - low_freq_factor
		)
	else:
		smooth = 0
	new_freqs = jnp.where(
		wave_len < high_freq_wavelen,
		inv_freqs,
		jnp.where(
			wave_len > low_freq_wavelen,
			inv_freqs / scaling_factor,
			(1 - smooth) * inv_freqs / scaling_factor + smooth * inv_freqs,
		),
	)
	return new_freqs


# @partial(jax.jit, static_argnames=["base", "rotary_dim", "max_position_embeddings"])
def compute_basic_frequencies(
	base: int,
	rotary_dim: int,
	max_position_embeddings: int,
):
	inv = compute_basic_inv_frequencies(base, rotary_dim)
	freqs = jnp.einsum(
		"i,j -> ij",
		jnp.arange(max_position_embeddings, dtype=jnp.float32),
		inv,
	)
	freqs = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
	return freqs


def compute_linear_frequencies(
	base: int,
	rotary_dim: int,
	max_position_embeddings: int,
	scaling_factors: tp.List[float],
):
	inv_freq = compute_basic_inv_frequencies(
		base=base,
		rotary_dim=rotary_dim,
	)
	cache_list: tp.List[jnp.ndarray] = []
	offsets: tp.List[int] = []

	for scaling_factor in scaling_factors:
		max_len = max_position_embeddings * scaling_factor
		t = jnp.arange(max_len, dtype=jnp.float32)
		t = t / scaling_factor

		freqs = jnp.einsum("i,j -> ij", t, inv_freq)
		cache = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
		if not cache_list:
			offset = 0
		else:
			last_offset = offsets[-1]
			next_max_len = cache_list[-1].shape[0]
			offset = last_offset + next_max_len
		offsets.append(offset)
		cache_list.append(cache)

	assert len(scaling_factors) == len(offsets)
	return jnp.concatenate(cache_list, axis=0)


def compute_dynamic_frequencies(
	base: int,
	rotary_dim: int,
	max_position_embeddings: int,
	scaling_factor: float,
):
	max_length = max_position_embeddings * scaling_factor
	base = base * (
		(scaling_factor * max_length / max_position_embeddings) - (scaling_factor - 1)
	) ** (rotary_dim / (rotary_dim - 2))
	inv_frequencies = compute_basic_inv_frequencies(base=base, rotary_dim=rotary_dim)
	times = jnp.arange(max_length, dtype=jnp.float32)
	frequencies = jnp.einsum("i,j -> ij", times, inv_frequencies)
	return jnp.concatenate([jnp.cos(frequencies), jnp.sin(frequencies)], -1)


def compute_yarn_frequencies(
	base: float,
	rotary_dim: int,
	beta_fast: float,
	beta_slow: float,
	max_position_embeddings: int,
	scaling_factor: float,
	extrapolation_factor: float,
	attn_factor: float,
) -> jnp.ndarray:
	inv_freq = compute_yarn_inv_frequencies(
		base=base,
		rotary_dim=rotary_dim,
		beta_fast=beta_fast,
		beta_slow=beta_slow,
		max_position_embeddings=max_position_embeddings,
		scaling_factor=scaling_factor,
		extrapolation_factor=extrapolation_factor,
	)
	t = jnp.arange(max_position_embeddings * scaling_factor, dtype=jnp.float32)
	freqs = jnp.einsum("i,j -> ij", t, inv_freq)
	mscale = _yarn_get_mscale(scaling_factor) * attn_factor
	cos = jnp.cos(freqs) * mscale
	sin = jnp.sin(freqs) * mscale
	return jnp.concatenate([cos, sin], axis=-1)


def compute_phi3_frequencies(
	base,
	head_size,
	rotary_dim,
	max_position_embeddings,
	original_max_position_embeddings,
	short_factor,
	long_factor,
):
	if rotary_dim != head_size:
		raise ValueError(f"rotary_dim != head_size ({rotary_dim}!={head_size})")
	if max_position_embeddings > original_max_position_embeddings:
		ext_factors = jnp.array(long_factor, dtype=jnp.float32)
	else:
		ext_factors = jnp.array(short_factor, dtype=jnp.float32)

	inv_freq_shape = (
		jnp.arange(0, head_size, 2, dtype=jnp.int64).astype(jnp.float32) / head_size
	)
	inv_freq = 1.0 / (ext_factors * (base**inv_freq_shape))

	inv_freq_expanded = jnp.expand_dims(inv_freq, (0, 2)).astype(jnp.float32)
	position_ids = jnp.arange(max_position_embeddings, dtype=jnp.int32).reshape(1, -1)
	position_ids_expanded = jnp.expand_dims(position_ids, 1).astype(jnp.float32)

	freqs = (inv_freq_expanded @ position_ids_expanded).swapaxes(1, 2)
	emb = jnp.concatenate((freqs, freqs), axis=-1)
	scale = max_position_embeddings / original_max_position_embeddings
	if scale <= 1.0:
		scaling_factor = 1.0
	else:
		scaling_factor = math.sqrt(
			1 + math.log(scale) / math.log(original_max_position_embeddings)
		)

	cos = jnp.cos(emb) * scaling_factor
	sin = jnp.sin(emb) * scaling_factor
	return jnp.concatenate([cos, sin], axis=-1)


def compute_llama3_frequencies(
	base,
	rotary_dim,
	low_freq_factor,
	high_freq_factor,
	scaling_factor,
	max_position_embeddings: int,
):
	inv = compute_llama3_inv_frequencies(
		base,
		rotary_dim,
		low_freq_factor,
		high_freq_factor,
		max_position_embeddings,
		scaling_factor,
	)
	freqs = jnp.einsum(
		"i,j -> ij",
		jnp.arange(max_position_embeddings, dtype=jnp.float32),
		inv,
	)
	freqs = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
	return freqs


def compute_deepseek_frequencies(
	base,
	rotary_dim,
	scaling_factor,
	extrapolation_factor,
	beta_fast,
	beta_slow,
	max_position_embeddings,
	mscale,
	mscale_all_dim,
	attn_factor,
) -> jnp.ndarray:
	pos_freqs = base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
	inv_freq_extrapolation = 1.0 / pos_freqs
	inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
	low, high = _yarn_find_correction_range(
		beta_fast,
		beta_slow,
		rotary_dim,
		base,
		max_position_embeddings,
	)
	inv_freq_mask = (
		1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=jnp.float32)
	) * extrapolation_factor
	inv_freq = (
		inv_freq_interpolation * (1 - inv_freq_mask)
		+ inv_freq_extrapolation * inv_freq_mask
	)

	t = jnp.arange(
		max_position_embeddings * scaling_factor,
		dtype=jnp.float32,
	)
	freqs = jnp.einsum("i,j -> ij", t, inv_freq)
	mscale = (
		yarn_get_mscale(scaling_factor, mscale)
		/ yarn_get_mscale(scaling_factor, mscale_all_dim)
		* attn_factor
	)

	return jnp.concatenate([jnp.cos(freqs) * mscale, jnp.sin(freqs) * mscale], axis=-1)


# @partial(jax.jit, static_argnames=["rotary_dim", "is_neox_style", "dtype"])
def apply_basic_rope(
	query: jax.Array,
	key: jax.Array,
	positions: jax.Array,
	frequencies: jax.Array,
	rotary_dim: int,
	is_neox_style: bool,
	offsets: jax.Array = None,
	dtype: jnp.dtype = jnp.float32,
):
	if offsets is not None:
		positions = positions + offsets
	cos, sin = jnp.split(frequencies[positions], 2, -1)
	if rotary_dim != query.shape[-1]:
		query_rot = _apply_rotary_emb(query[..., :rotary_dim], cos, sin, is_neox_style)
		query = jnp.concatenate((query_rot, query[..., rotary_dim:]), axis=-1)
		key_rot = _apply_rotary_emb(key[..., :rotary_dim], cos, sin, is_neox_style)
		key = jnp.concatenate((key_rot, key[..., rotary_dim:]), axis=-1)
		return query.astype(dtype), key.astype(dtype)
	else:
		query = _apply_rotary_emb(query, cos, sin, is_neox_style)
		key = _apply_rotary_emb(key, cos, sin, is_neox_style)
		return query.astype(dtype), key.astype(dtype)


# @partial(
# 	jax.jit,
# 	static_argnames=["original_max_position_embeddings", "dtype"],
# )
def apply_phi3_rope(
	query,
	key,
	positions,
	frequencies,
	offsets: jax.Array = None,
	dtype: jnp.dtype = jnp.float32,
):
	positions = positions
	if offsets is not None:
		positions = positions + offsets
	emb = frequencies[0, positions]
	cos, sin = jnp.split(emb, 2, axis=-1)
	cos = jnp.expand_dims(cos, 2)
	sin = jnp.expand_dims(sin, 2)

	with jax.default_matmul_precision("float32"):
		query_rot = query * cos + _rotate_neox(query) * sin
		key_rot = key * cos + _rotate_neox(key) * sin

	return query_rot.astype(dtype), key_rot.astype(dtype)


@rope_wraper("default")
class RotaryEmbedding(nn.Module):
	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
	):
		self.head_size = head_size
		self.rotary_dim = rotary_dim
		self.max_position_embeddings = max_position_embeddings
		self.base = base
		self.is_neox_style = is_neox_style
		self.dtype = dtype

	@jax.named_scope("easydel-rope-embedding")
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: tp.Optional[jnp.ndarray] = None,
		frequencies: tp.Optional[jnp.ndarray] = None,
	) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		with jax.ensure_compile_time_eval():
			if frequencies is None:
				frequencies = compute_basic_frequencies(
					base=self.base,
					rotary_dim=self.rotary_dim,
					max_position_embeddings=self.max_position_embeddings,
				)
			return apply_basic_rope(
				query=query,
				key=key,
				positions=positions,
				frequencies=frequencies,
				rotary_dim=self.rotary_dim,
				is_neox_style=self.is_neox_style,
				offsets=offsets,
				dtype=self.dtype,
			)


@rope_wraper("linear")
class LinearScalingRotaryEmbedding(RotaryEmbedding):
	def __init__(
		self,
		scaling_factors: tp.Union[tp.List[float], float],
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
	):
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)
		self.scaling_factors = scaling_factors

	@jax.named_scope("easydel-rope-linear-scaling")
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: tp.Optional[jnp.ndarray] = None,
		frequencies: tp.Optional[jnp.ndarray] = None,
	) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		with jax.ensure_compile_time_eval():
			if frequencies is None:
				frequencies = compute_linear_frequencies(
					base=self.base,
					rotary_dim=self.rotary_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factors=self.scaling_factors,
				)
			return apply_basic_rope(
				query=query,
				key=key,
				positions=positions,
				frequencies=frequencies,
				rotary_dim=self.rotary_dim,
				is_neox_style=self.is_neox_style,
				offsets=offsets,
				dtype=self.dtype,
			)


@rope_wraper("dynamic")
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
	"""RotaryEmbedding extended with Dynamic NTK scaling."""

	def __init__(
		self,
		scaling_factor: tp.Union[tp.List[float], float],
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
	):
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)
		self.scaling_factor = scaling_factor

	@jax.named_scope("easydel-rope-dynamic-ntk-scaling")
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: tp.Optional[jnp.ndarray] = None,
		frequencies: tp.Optional[jnp.ndarray] = None,
	) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		with jax.ensure_compile_time_eval():
			if frequencies is None:
				frequencies = compute_dynamic_frequencies(
					base=self.base,
					rotary_dim=self.rotary_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factor=self.scaling_factor,
				)
			return apply_basic_rope(
				query=query,
				key=key,
				positions=positions,
				frequencies=frequencies,
				rotary_dim=self.rotary_dim,
				is_neox_style=self.is_neox_style,
				offsets=offsets,
				dtype=self.dtype,
			)


@rope_wraper("yarn")
class YaRNScalingRotaryEmbedding(RotaryEmbedding):
	"""RotaryEmbedding extended with YaRN method.

	Credits to Peng et al. github.com/jquesnelle/yarn
	"""

	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
		scaling_factor: tp.Union[float, int] = 1.0,
		extrapolation_factor: float = 1.0,
		attn_factor: float = 1.0,
		beta_fast: int = 32,
		beta_slow: int = 1,
	):
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)

		self.scaling_factor = scaling_factor
		self.extrapolation_factor = extrapolation_factor
		self.attn_factor = attn_factor
		self.beta_fast = beta_fast
		self.beta_slow = beta_slow

	@jax.named_scope("easydel-rope-yarn-scaling")
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: tp.Optional[jnp.ndarray] = None,
		frequencies: tp.Optional[jnp.ndarray] = None,
	) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		with jax.ensure_compile_time_eval():
			if frequencies is None:
				frequencies = compute_yarn_frequencies(
					base=self.base,
					rotary_dim=self.rotary_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factor=self.scaling_factor,
					beta_fast=self.beta_fast,
					beta_slow=self.beta_slow,
					extrapolation_factor=self.extrapolation_factor,
					attn_factor=self.attn_factor,
				)
			return apply_basic_rope(
				query=query,
				key=key,
				positions=positions,
				frequencies=frequencies,
				rotary_dim=self.rotary_dim,
				is_neox_style=self.is_neox_style,
				offsets=offsets,
				dtype=self.dtype,
			)


@rope_wraper("longrope")
class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		original_max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
		short_factor: tp.List[float],
		long_factor: tp.List[float],
	):
		super().__init__()

		self.head_size = head_size
		self.rotary_dim = rotary_dim
		self.max_position_embeddings = max_position_embeddings
		self.original_max_position_embeddings = original_max_position_embeddings
		self.base = base
		self.is_neox_style = is_neox_style
		self.dtype = dtype
		self.short_factor = short_factor
		self.long_factor = long_factor

	@jax.named_scope("easydel-rope-phi3-long")
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: tp.Optional[jnp.ndarray] = None,
		frequencies: tp.Optional[jnp.ndarray] = None,
	) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		with jax.ensure_compile_time_eval():
			if frequencies is None:
				frequencies = compute_phi3_frequencies(
					base=self.base,
					head_size=self.head_size,
					rotary_dim=self.rotary_dim,
					max_position_embeddings=self.max_position_embeddings,
					original_max_position_embeddings=self.original_max_position_embeddings,
					short_factor=self.short_factor,
					long_factor=self.long_factor,
				)
			return apply_phi3_rope(
				query=query,
				key=key,
				positions=positions,
				frequencies=frequencies,
				offsets=offsets,
				dtype=self.dtype,
			)


@rope_wraper("llama3")
class Llama3RotaryEmbedding(RotaryEmbedding):
	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
		scaling_factor: float,
		low_freq_factor: float,
		high_freq_factor: float,
		orig_max_position: int,
	):
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)

		self.scaling_factor = scaling_factor
		self.low_freq_factor = low_freq_factor
		self.high_freq_factor = high_freq_factor
		self.orig_max_position = orig_max_position

	@jax.named_scope("easydel-rope-llama3")
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: tp.Optional[jnp.ndarray] = None,
		frequencies: tp.Optional[jnp.ndarray] = None,
	) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		with jax.ensure_compile_time_eval():
			if frequencies is None:
				frequencies = compute_llama3_frequencies(
					base=self.base,
					rotary_dim=self.rotary_dim,
					low_freq_factor=self.low_freq_factor,
					high_freq_factor=self.high_freq_factor,
					scaling_factor=self.scaling_factor,
					max_position_embeddings=self.orig_max_position,
				)

			return apply_basic_rope(
				query=query,
				key=key,
				positions=positions,
				frequencies=frequencies,
				rotary_dim=self.rotary_dim,
				is_neox_style=self.is_neox_style,
				offsets=offsets,
				dtype=self.dtype,
			)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
	if scale <= 1:
		return 1.0
	return 0.1 * mscale * jnp.log(scale) + 1.0


@rope_wraper("deepseek_yarn")
class DeepseekScalingRotaryEmbedding(nn.Module):
	"""RotaryEmbedding extended with YaRN method."""

	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
		scaling_factor: float,
		extrapolation_factor: float = 1,
		attn_factor: float = 1,
		beta_fast: int = 32,
		beta_slow: int = 1,
		mscale: float = 1,
		mscale_all_dim: float = 0,
	):
		self.head_size = head_size
		self.rotary_dim = rotary_dim
		self.max_position_embeddings = max_position_embeddings
		self.base = base
		self.is_neox_style = is_neox_style
		self.dtype = dtype
		self.scaling_factor = scaling_factor
		self.extrapolation_factor = extrapolation_factor
		self.attn_factor = attn_factor
		self.beta_fast = beta_fast
		self.beta_slow = beta_slow
		self.mscale = mscale
		self.mscale_all_dim = mscale_all_dim

	@jax.named_scope("easydel-rope-deepseek")
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: tp.Optional[jnp.ndarray] = None,
		frequencies: tp.Optional[jnp.ndarray] = None,
	) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
		if frequencies is None:
			frequencies = compute_deepseek_frequencies(
				self.base,
				self.rotary_dim,
				self.scaling_factor,
				self.extrapolation_factor,
				self.beta_fast,
				self.beta_slow,
				self.max_position_embeddings,
				self.mscale,
				self.mscale_all_dim,
				self.attn_factor,
			)
		cos, sin = jnp.split(frequencies[positions], 2, -1)
		if offsets is not None:
			positions += offsets
		query_rot = query[..., : self.rotary_dim]
		key_rot = key[..., : self.rotary_dim]

		if self.rotary_dim < self.head_size:
			query_pass = query[..., self.rotary_dim :]
			key_pass = key[..., self.rotary_dim :]

		target_sc_shape = (query.shape[0], -1, 1, self.rotary_dim)
		if self.is_neox_style:
			cos = cos.repeat(2, axis=1).reshape(target_sc_shape)
			sin = sin.repeat(2, axis=1).reshape(target_sc_shape)
		else:
			cos = cos.repeat_interleave(2, axis=1).reshape(target_sc_shape)
			sin = sin.repeat_interleave(2, axis=1).reshape(target_sc_shape)
		rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
		query_rot = query_rot * cos + rotate_fn(query_rot) * sin
		key_rot = key_rot * cos + rotate_fn(key_rot) * sin

		if self.rotary_dim < self.head_size:
			query = jnp.concatenate((query_rot, query_pass), axis=-1)
			key = jnp.concatenate((key_rot, key_pass), axis=-1)
		else:
			query = query_rot
			key = key_rot
		return query, key


def get_rope(
	head_size: int,
	rotary_dim: int,
	max_position: int,
	base: int,
	is_neox_style: bool = True,
	rope_scaling: tp.Optional[tp.Dict[str, tp.Any]] = None,
	dtype: tp.Optional[jnp.dtype] = None,
	partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
	if dtype is None:
		dtype = jnp.float32  # Default JAX dtype

	if partial_rotary_factor < 1.0:
		rotary_dim = int(rotary_dim * partial_rotary_factor)

	if rope_scaling is None:
		rotary_emb = RotaryEmbedding(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)
	else:
		scaling_type = rope_scaling["rope_type"]
		if scaling_type == "llama3":
			scaling_factor = rope_scaling["factor"]
			low_freq_factor = rope_scaling["low_freq_factor"]
			high_freq_factor = rope_scaling["high_freq_factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			rotary_emb = Llama3RotaryEmbedding(
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				base=base,
				is_neox_style=is_neox_style,
				dtype=dtype,
				scaling_factor=scaling_factor,
				low_freq_factor=low_freq_factor,
				high_freq_factor=high_freq_factor,
				orig_max_position=original_max_position,
			)
		elif scaling_type == "default":
			rotary_emb = RotaryEmbedding(
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				base=base,
				is_neox_style=is_neox_style,
				dtype=dtype,
			)
		elif scaling_type == "linear":
			scaling_factor = rope_scaling["factor"]
			rotary_emb = LinearScalingRotaryEmbedding(
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				base=base,
				is_neox_style=is_neox_style,
				scaling_factor=scaling_factor,
				dtype=dtype,
			)
		elif scaling_type == "dynamic":
			scaling_factor = rope_scaling["factor"]
			rotary_emb = DynamicNTKScalingRotaryEmbedding(
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				base=base,
				is_neox_style=is_neox_style,
				scaling_factor=scaling_factor,
				dtype=dtype,
			)
		elif scaling_type == "yarn":
			scaling_factor = rope_scaling["factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			extra_kwargs = {
				k: v
				for k, v in rope_scaling.items()
				if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
			}
			rotary_emb = YaRNScalingRotaryEmbedding(
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=original_max_position,
				base=base,
				is_neox_style=is_neox_style,
				scaling_factor=scaling_factor,
				dtype=dtype,
				**extra_kwargs,
			)
		elif scaling_type == "deepseek_yarn":
			scaling_factor = rope_scaling["factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			extra_kwargs = {
				k: v
				for k, v in rope_scaling.items()
				if k
				in (
					"extrapolation_factor",
					"attn_factor",
					"beta_fast",
					"beta_slow",
					"mscale",
					"mscale_all_dim",
				)
			}
			rotary_emb = DeepseekScalingRotaryEmbedding(
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=original_max_position,
				base=base,
				is_neox_style=is_neox_style,
				scaling_factor=scaling_factor,
				dtype=dtype,
				**extra_kwargs,
			)
		elif scaling_type == "longrope":
			short_factor = rope_scaling["short_factor"]
			long_factor = rope_scaling["long_factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]

			rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				original_max_position_embeddings=original_max_position,
				base=base,
				is_neox_style=is_neox_style,
				dtype=dtype,
				short_factor=short_factor,
				long_factor=long_factor,
			)
		else:
			raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

	return rotary_emb


@partial(
	jax.jit,
	static_argnames=[
		"head_size",
		"rotary_dim",
		"max_position",
		"base",
		"rope_scaling",
		"partial_rotary_factor",
	],
)
def get_frequencies(
	head_size: int,
	rotary_dim: int,
	max_position: int,
	base: int,
	rope_scaling: tp.Optional[tp.Dict[str, tp.Any]] = None,
	partial_rotary_factor: float = 1.0,
) -> jax.Array:
	if partial_rotary_factor < 1.0:
		rotary_dim = int(rotary_dim * partial_rotary_factor)

	if rope_scaling is None:
		frequencies = compute_basic_frequencies(
			base=base,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position,
		)
	else:
		scaling_type = rope_scaling["rope_type"]

		if scaling_type == "llama3":
			scaling_factor = rope_scaling["factor"]
			low_freq_factor = rope_scaling["low_freq_factor"]
			high_freq_factor = rope_scaling["high_freq_factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			frequencies = compute_llama3_frequencies(
				base=base,
				rotary_dim=rotary_dim,
				low_freq_factor=low_freq_factor,
				high_freq_factor=high_freq_factor,
				scaling_factor=scaling_factor,
				max_position_embeddings=original_max_position,
			)

		elif scaling_type == "default":
			frequencies = compute_basic_frequencies(
				base=base,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
			)
		elif scaling_type == "linear":
			scaling_factors = rope_scaling["factor"]
			frequencies = compute_linear_frequencies(
				base=base,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				scaling_factors=scaling_factors,
			)
		elif scaling_type == "dynamic":
			scaling_factor = rope_scaling["factor"]
			frequencies = compute_dynamic_frequencies(
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				base=base,
				scaling_factor=scaling_factor,
			)
		elif scaling_type == "yarn":
			scaling_factor = rope_scaling["factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			extra_kwargs = {
				k: v
				for k, v in rope_scaling.items()
				if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
			}

			frequencies = compute_yarn_frequencies(
				base=base,
				rotary_dim=rotary_dim,
				beta_fast=extra_kwargs["beta_fast"],
				beta_slow=extra_kwargs["beta_slow"],
				max_position_embeddings=original_max_position,
				scaling_factor=scaling_factor,
				extrapolation_factor=extra_kwargs["extrapolation_factor"],
				attn_factor=extra_kwargs["attn_factor"],
			)
		elif scaling_type == "deepseek_yarn":
			scaling_factor = rope_scaling["factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			extra_kwargs = {
				k: v
				for k, v in rope_scaling.items()
				if k
				in (
					"extrapolation_factor",
					"attn_factor",
					"beta_fast",
					"beta_slow",
					"mscale",
					"mscale_all_dim",
				)
			}
			frequencies = compute_deepseek_frequencies(
				base,
				rotary_dim,
				scaling_factor,
				extra_kwargs["extrapolation_factor"],
				extra_kwargs["beta_fast"],
				extra_kwargs["beta_slow"],
				original_max_position,
				extra_kwargs["mscale"],
				extra_kwargs["mscale_all_dim"],
				extra_kwargs["attn_factor"],
			)
		elif scaling_type == "longrope":
			short_factor = rope_scaling["short_factor"]
			long_factor = rope_scaling["long_factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			extra_kwargs = {
				k: v for k, v in rope_scaling.items() if k in ("short_mscale", "long_mscale")
			}

			frequencies = compute_phi3_frequencies(
				base=base,
				head_size=head_size,
				rotary_dim=rotary_dim,
				max_position_embeddings=max_position,
				original_max_position_embeddings=original_max_position,
				short_factor=short_factor,
				long_factor=long_factor,
			)
		else:
			raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

	return frequencies


# Example usage
if __name__ == "__main__":
	head_size = 64
	rotary_dim = 64
	max_position = 2048
	base = 10000
	is_neox_style = True
	dtype = jnp.float32

	rope_scaling = {
		"rope_type": "yarn",
		"factor": 2.0,
		"original_max_position_embeddings": 1024,
		"extrapolation_factor": 1.0,
		"attn_factor": 1.0,
		"beta_fast": 32,
		"beta_slow": 1,
	}

	rope = get_rope(
		head_size,
		rotary_dim,
		max_position,
		base,
		is_neox_style,
		rope_scaling,
		dtype,
	)
	freq = get_frequencies(
		head_size,
		rotary_dim,
		max_position,
		base,
		rope_scaling,
	)
