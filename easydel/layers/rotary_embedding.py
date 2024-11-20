from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap


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


@partial(jit, static_argnums=(4, 5, 6, 7))
def _rotary_embedding_call(
	positions: jnp.ndarray,
	query: jnp.ndarray,
	key: jnp.ndarray,
	base: float,
	max_position_embeddings: int,
	rotary_dim: int,
	is_neox_style: bool,
	offsets: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
	if offsets is not None:
		positions = positions + offsets
	inv = 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
	freqs = jnp.einsum(
		"i,j -> ij",
		jnp.arange(max_position_embeddings, dtype=jnp.float32),
		inv,
	)[positions]
	cos = jnp.cos(freqs)
	sin = jnp.sin(freqs)
	query_rot = _apply_rotary_emb(query[..., :rotary_dim], cos, sin, is_neox_style)
	query = jnp.concatenate((query_rot, query[..., rotary_dim:]), axis=-1)
	key_rot = _apply_rotary_emb(key[..., :rotary_dim], cos, sin, is_neox_style)
	key = jnp.concatenate((key_rot, key[..., rotary_dim:]), axis=-1)
	return query, key


@partial(
	vmap,
	in_axes=(0, 0, 0, None, None, None, 0),
	out_axes=(0, 0),
)
@partial(jit, static_argnums=(4, 5))
def _phi3_rope_call(
	positions: jnp.ndarray,
	query: jnp.ndarray,
	key: jnp.ndarray,
	long_short_cos_sin_cache: jnp.ndarray,
	original_max_position_embeddings: int,
	head_size: int,
	offsets: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
	iniq_shape = query.shape
	inik_shape = key.shape
	query = query.reshape(key.shape[0], -1, head_size)
	key = key.reshape(key.shape[0], -1, head_size)

	k = original_max_position_embeddings
	long_prompt_offset = (
		jnp.any(positions > k).astype(jnp.float32) * jnp.full_like(positions, k)
	).astype(jnp.int32)
	idx = positions + long_prompt_offset if long_prompt_offset is not None else positions
	idx = idx + offsets if offsets is not None else idx
	cos_sin = long_short_cos_sin_cache[idx]
	cos, sin = jnp.split(cos_sin, 2, axis=-1)
	cos = cos.repeat(2, axis=-1).reshape(-1, 1, cos.shape[-1] * 2)
	sin = sin.repeat(2, axis=-1).reshape(-1, 1, sin.shape[-1] * 2)
	query = query * cos + _rotate_neox(query) * sin
	key = key * cos + _rotate_neox(key) * sin

	return query.reshape(iniq_shape), key.reshape(inik_shape)


@partial(
	vmap,
	in_axes=(0, 0, 0, None, None, None, None, 0),
	out_axes=(0, 0),
)
@partial(jit, static_argnums=(4, 5, 6))
def _deepseek_rope_call(
	positions: jnp.ndarray,
	query: jnp.ndarray,
	key: jnp.ndarray,
	cos_sin_cache: jnp.ndarray,
	rotary_dim: int,
	head_size: int,
	is_neox_style: bool,
	offsets: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
	query_rot = query[..., :rotary_dim]
	key_rot = key[..., :rotary_dim]

	if rotary_dim < head_size:
		query_pass = query[..., rotary_dim:]
		key_pass = key[..., rotary_dim:]

	cos_sin = cos_sin_cache[
		jnp.add(positions, offsets) if offsets is not None else positions
	]
	cos, sin = jnp.split(cos_sin, 2, axis=-1)
	if is_neox_style:
		cos = cos.repeat(2, axis=1).reshape(-1, 1, cos.shape[-1] * 2)
		sin = sin.repeat(2, axis=1).reshape(-1, 1, sin.shape[-1] * 2)
	else:
		cos = cos.repeat_interleave(2, axis=1).reshape(-1, 1, cos.shape[-1] * 2)
		sin = sin.repeat_interleave(2, axis=1).reshape(-1, 1, sin.shape[-1] * 2)
	rotate_fn = _rotate_neox if is_neox_style else _rotate_gptj
	query_rot = query_rot * cos + rotate_fn(query_rot) * sin
	key_rot = key_rot * cos + rotate_fn(key_rot) * sin

	if rotary_dim < head_size:
		query = jnp.concatenate((query_rot, query_pass), axis=-1)
		key = jnp.concatenate((key_rot, key_pass), axis=-1)
	else:
		query = query_rot
		key = key_rot
	return query, key


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


@partial(jit, static_argnums=(0, 1, 2))
def _default_compute_cos_sin_cache(
	base,
	rotary_dim,
	max_position_embeddings,
) -> jnp.ndarray:
	"""Compute the cos and sin cache."""
	inv_freq = 1.0 / (
		base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
	)
	t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
	freqs = jnp.einsum("i,j -> ij", t, inv_freq)
	cache = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
	return cache


@rope_wraper("default")
class RotaryEmbedding:
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

	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: Optional[jnp.ndarray] = None,
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		query, key = _rotary_embedding_call(
			positions,
			query,
			key,
			self.base,
			self.max_position_embeddings,
			self.rotary_dim,
			self.is_neox_style,
			offsets,
		)
		return query.astype(self.dtype), key.astype(self.dtype)


@rope_wraper("linear")
class LinearScalingRotaryEmbedding(RotaryEmbedding):
	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		scaling_factors: Union[List[float], float],
		dtype: jnp.dtype,
	):
		if isinstance(scaling_factors, float):
			scaling_factors = [scaling_factors]
		self.scaling_factors: List[float] = scaling_factors
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)
		self._scaling_factor_to_offset: Dict[float, int] = {}

	def _compute_cos_sin_cache(self) -> jnp.ndarray:
		inv_freq = self._compute_inv_freq(self.base)
		cache_list: List[jnp.ndarray] = []
		offsets: List[int] = []
		for scaling_factor in self.scaling_factors:
			max_len = self.max_position_embeddings * scaling_factor
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
		self._scaling_factor_to_offset = {
			float(scaling_factor): offsets[i]
			for i, scaling_factor in enumerate(self.scaling_factors)
		}
		assert len(self.scaling_factors) == len(offsets)
		return jnp.concatenate(cache_list, axis=0)

	@property
	def scaling_factor_to_offset(self) -> Dict[float, int]:
		return self._scaling_factor_to_offset


@rope_wraper("dynamic")
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
	"""RotaryEmbedding extended with Dynamic NTK scaling."""

	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		scaling_factor: float,
		dtype: jnp.dtype,
	):
		self.scaling_factor = scaling_factor
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)

	def _compute_cos_sin_cache(self) -> jnp.ndarray:
		max_len = self.max_position_embeddings * self.scaling_factor
		base = self.base * (
			(self.scaling_factor * max_len / self.max_position_embeddings)
			- (self.scaling_factor - 1)
		) ** (self.rotary_dim / (self.rotary_dim - 2))
		inv_freq = self._compute_inv_freq(base)
		t = jnp.arange(max_len, dtype=jnp.float32)

		freqs = jnp.einsum("i,j -> ij", t, inv_freq)
		cos = jnp.cos(freqs)
		sin = jnp.sin(freqs)
		cache = jnp.concatenate([cos, sin], axis=-1)
		return cache


def _yarn_find_correction_dim(
	num_rotations: int,
	dim: int,
	base: float = 10000,
	max_position_embeddings: int = 2048,
) -> float:
	return (dim * jnp.log(max_position_embeddings / (num_rotations * 2 * jnp.pi))) / (
		2 * jnp.log(base)
	)


def _yarn_find_correction_range(
	low_rot: int,
	high_rot: int,
	dim: int,
	base: float = 10000,
	max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
	low = jnp.floor(
		_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
	)
	high = jnp.ceil(
		_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
	)
	return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
	low: float,
	high: float,
	dim: int,
	dtype: jnp.dtype,
) -> jnp.ndarray:
	if low == high:
		high += 0.001  # Prevent singularity

	linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
	ramp_func = jnp.clip(linear_func, 0, 1)
	return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
	if scale <= 1:
		return 1.0
	return 0.1 * jnp.log(scale) + 1.0


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
		scaling_factor: float,
		dtype: jnp.dtype,
		*,
		extrapolation_factor: float = 1,
		attn_factor: float = 1,
		beta_fast: int = 32,
		beta_slow: int = 1,
	):
		self.scaling_factor = scaling_factor
		self.extrapolation_factor = extrapolation_factor
		self.attn_factor = attn_factor
		self.beta_fast = beta_fast
		self.beta_slow = beta_slow
		self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)

	def _compute_inv_freq(self, scaling_factor: float) -> jnp.ndarray:
		pos_freqs = self.base ** (
			jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim
		)
		inv_freq_extrapolation = 1.0 / pos_freqs
		inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

		low, high = _yarn_find_correction_range(
			self.beta_fast,
			self.beta_slow,
			self.rotary_dim,
			self.base,
			self.max_position_embeddings,
		)
		inv_freq_mask = (
			1 - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=jnp.float32)
		) * self.extrapolation_factor
		inv_freq = (
			inv_freq_interpolation * (1 - inv_freq_mask)
			+ inv_freq_extrapolation * inv_freq_mask
		)
		return inv_freq

	def _compute_cos_sin_cache(self) -> jnp.ndarray:
		inv_freq = self._compute_inv_freq(self.scaling_factor)
		t = jnp.arange(
			self.max_position_embeddings * self.scaling_factor, dtype=jnp.float32
		)
		freqs = jnp.einsum("i,j -> ij", t, inv_freq)
		cos = jnp.cos(freqs) * self.mscale
		sin = jnp.sin(freqs) * self.mscale
		cache = jnp.concatenate([cos, sin], axis=-1)
		return cache


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
	if scale <= 1:
		return 1.0
	return 0.1 * mscale * jnp.log(scale) + 1.0


@rope_wraper("longrope")
class Phi3LongRoPEScaledRotaryEmbedding:
	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		original_max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		dtype: jnp.dtype,
		short_factor: List[float],
		long_factor: List[float],
		short_mscale: Optional[float] = None,
		long_mscale: Optional[float] = None,
	):
		if rotary_dim != head_size:
			raise ValueError(
				f"`Phi3LongRoPEScaledRotaryEmbedding` does not support "
				f"rotary_dim != head_size ({rotary_dim}!={head_size})."
			)
		if not is_neox_style:
			raise ValueError("`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style.")

		self.head_size = head_size
		self.max_position_embeddings = max_position_embeddings
		self.original_max_position_embeddings = original_max_position_embeddings
		self.base = base
		self.short_factor = short_factor
		self.long_factor = long_factor

		scale = max_position_embeddings / original_max_position_embeddings
		if scale <= 1.0:
			scaling_factor = 1.0
		else:
			scaling_factor = jnp.sqrt(
				1 + jnp.log(scale) / jnp.log(original_max_position_embeddings)
			)

		if short_mscale is None:
			short_mscale = scaling_factor
		if long_mscale is None:
			long_mscale = scaling_factor

		self.short_mscale = short_mscale
		self.long_mscale = long_mscale

		self.short_cos_sin_cache = self._compute_cos_sin_cache(
			original_max_position_embeddings, short_factor, short_mscale
		).astype(dtype)
		self.long_cos_sin_cache = self._compute_cos_sin_cache(
			max_position_embeddings, long_factor, long_mscale
		).astype(dtype)

		self.long_short_cos_sin_cache = jnp.concatenate(
			[self.short_cos_sin_cache, self.long_cos_sin_cache], axis=0
		)

	def _compute_inv_freq(self, rescale_factors: List[float]) -> jnp.ndarray:
		rescale_factors = jnp.array(rescale_factors, dtype=jnp.float32)
		inv_freq = 1.0 / (
			rescale_factors
			* (
				self.base
				** (jnp.arange(0, self.head_size, 2, dtype=jnp.float32) / self.head_size)
			)
		)
		return inv_freq

	def _compute_cos_sin_cache(
		self,
		max_position_embeddings: int,
		rescale_factors: List[float],
		mscale: float,
	) -> jnp.ndarray:
		inv_freq = self._compute_inv_freq(rescale_factors)
		t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
		freqs = jnp.einsum("i,j -> ij", t, inv_freq)
		cos = jnp.cos(freqs) * mscale
		sin = jnp.sin(freqs) * mscale
		cache = jnp.concatenate([cos, sin], axis=-1)
		return cache

	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: Optional[jnp.ndarray] = None,
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		return _phi3_rope_call(
			positions,
			query,
			key,
			self.long_short_cos_sin_cache,
			self.original_max_position_embeddings,
			self.head_size,
			offsets,
		)


def _yarn_find_correction_dim(
	num_rotations: int, dim: int, base: float = 10000, max_position_embeddings: int = 2048
) -> float:
	return (dim * jnp.log(max_position_embeddings / (num_rotations * 2 * jnp.pi))) / (
		2 * jnp.log(base)
	)


def _yarn_find_correction_range(
	low_rot: int,
	high_rot: int,
	dim: int,
	base: float = 10000,
	max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
	low = jnp.floor(
		_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
	)
	high = jnp.ceil(
		_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
	)
	return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
	low: float, high: float, dim: int, dtype: jnp.dtype
) -> jnp.ndarray:
	if low == high:
		high += 0.001  # Prevent singularity

	linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
	ramp_func = jnp.clip(linear_func, 0, 1)
	return ramp_func


@rope_wraper("deepseek_yarn")
class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
	"""RotaryEmbedding extended with YaRN method."""

	def __init__(
		self,
		head_size: int,
		rotary_dim: int,
		max_position_embeddings: int,
		base: int,
		is_neox_style: bool,
		scaling_factor: float,
		dtype: jnp.dtype,
		*,
		extrapolation_factor: float = 1,
		attn_factor: float = 1,
		beta_fast: int = 32,
		beta_slow: int = 1,
		mscale: float = 1,
		mscale_all_dim: float = 0,
	):
		self.scaling_factor = scaling_factor
		self.extrapolation_factor = extrapolation_factor
		self.attn_factor = attn_factor
		self.beta_fast = beta_fast
		self.beta_slow = beta_slow
		self.mscale = float(
			yarn_get_mscale(self.scaling_factor, float(mscale))
			/ yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
			* attn_factor
		)
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)

	def _compute_inv_freq(self, scaling_factor: float) -> jnp.ndarray:
		pos_freqs = self.base ** (
			jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim
		)
		inv_freq_extrapolation = 1.0 / pos_freqs
		inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

		low, high = _yarn_find_correction_range(
			self.beta_fast,
			self.beta_slow,
			self.rotary_dim,
			self.base,
			self.max_position_embeddings,
		)
		inv_freq_mask = (
			1 - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=jnp.float32)
		) * self.extrapolation_factor
		inv_freq = (
			inv_freq_interpolation * (1 - inv_freq_mask)
			+ inv_freq_extrapolation * inv_freq_mask
		)
		return inv_freq

	def _compute_cos_sin_cache(self) -> jnp.ndarray:
		inv_freq = self._compute_inv_freq(self.scaling_factor)
		t = jnp.arange(
			self.max_position_embeddings * self.scaling_factor, dtype=jnp.float32
		)
		freqs = jnp.einsum("i,j -> ij", t, inv_freq)
		cos = jnp.cos(freqs) * self.mscale
		sin = jnp.sin(freqs) * self.mscale
		cache = jnp.concatenate([cos, sin], axis=-1)
		return cache

	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: Optional[jnp.ndarray] = None,
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		return _deepseek_rope_call(
			positions,
			query,
			key,
			self.cos_sin_cache,
			self.rotary_dim,
			self.head_size,
			self.is_neox_style,
			offsets,
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
		self.scaling_factor = scaling_factor
		self.low_freq_factor = low_freq_factor
		self.high_freq_factor = high_freq_factor
		self.orig_max_position = orig_max_position
		super().__init__(
			head_size=head_size,
			rotary_dim=rotary_dim,
			max_position_embeddings=max_position_embeddings,
			base=base,
			is_neox_style=is_neox_style,
			dtype=dtype,
		)

	def _compute_inv_freq(self, base: Union[int, float]) -> jnp.ndarray:
		inv_freqs = super()._compute_inv_freq(base)
		low_freq_wavelen = self.orig_max_position / self.low_freq_factor
		high_freq_wavelen = self.orig_max_position / self.high_freq_factor

		wave_len = 2 * jnp.pi / inv_freqs
		if self.low_freq_factor != self.high_freq_factor:
			smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
				self.high_freq_factor - self.low_freq_factor
			)
		else:
			smooth = 0
		new_freqs = jnp.where(
			wave_len < high_freq_wavelen,
			inv_freqs,
			jnp.where(
				wave_len > low_freq_wavelen,
				inv_freqs / self.scaling_factor,
				(1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,
			),
		)
		return new_freqs


_ROPE_CACHE: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
	head_size: int,
	rotary_dim: int,
	max_position: int,
	base: int,
	is_neox_style: bool = True,
	rope_scaling: Optional[Dict[str, Any]] = None,
	dtype: Optional[jnp.dtype] = None,
	partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
	if dtype is None:
		dtype = jnp.float32  # Default JAX dtype

	if rope_scaling is not None:
		# Transforms every value that is a list into a tuple for caching calls
		rope_scaling_tuple = {
			k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
		}
		rope_scaling_args = tuple(rope_scaling_tuple.items())
	else:
		rope_scaling_args = None

	if partial_rotary_factor < 1.0:
		rotary_dim = int(rotary_dim * partial_rotary_factor)

	key = (
		head_size,
		rotary_dim,
		max_position,
		base,
		is_neox_style,
		rope_scaling_args,
		dtype,
	)
	if key in _ROPE_CACHE:
		return _ROPE_CACHE[key]

	if rope_scaling is None:
		rotary_emb = RotaryEmbedding(
			head_size, rotary_dim, max_position, base, is_neox_style, dtype
		)
	else:
		scaling_type = rope_scaling["rope_type"]

		if scaling_type == "llama3":
			scaling_factor = rope_scaling["factor"]
			low_freq_factor = rope_scaling["low_freq_factor"]
			high_freq_factor = rope_scaling["high_freq_factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			rotary_emb = Llama3RotaryEmbedding(
				head_size,
				rotary_dim,
				max_position,
				base,
				is_neox_style,
				dtype,
				scaling_factor,
				low_freq_factor,
				high_freq_factor,
				original_max_position,
			)
		elif scaling_type == "default":
			rotary_emb = RotaryEmbedding(
				head_size, rotary_dim, max_position, base, is_neox_style, dtype
			)
		elif scaling_type == "linear":
			scaling_factor = rope_scaling["factor"]
			rotary_emb = LinearScalingRotaryEmbedding(
				head_size, rotary_dim, max_position, base, is_neox_style, scaling_factor, dtype
			)
		elif scaling_type == "dynamic":
			scaling_factor = rope_scaling["factor"]
			rotary_emb = DynamicNTKScalingRotaryEmbedding(
				head_size, rotary_dim, max_position, base, is_neox_style, scaling_factor, dtype
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
				head_size,
				rotary_dim,
				original_max_position,
				base,
				is_neox_style,
				scaling_factor,
				dtype,
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
				head_size,
				rotary_dim,
				original_max_position,
				base,
				is_neox_style,
				scaling_factor,
				dtype,
				**extra_kwargs,
			)
		elif scaling_type == "longrope":
			short_factor = rope_scaling["short_factor"]
			long_factor = rope_scaling["long_factor"]
			original_max_position = rope_scaling["original_max_position_embeddings"]
			extra_kwargs = {
				k: v for k, v in rope_scaling.items() if k in ("short_mscale", "long_mscale")
			}
			rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
				head_size,
				rotary_dim,
				max_position,
				original_max_position,
				base,
				is_neox_style,
				dtype,
				short_factor,
				long_factor,
				**extra_kwargs,
			)
		else:
			raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

	_ROPE_CACHE[key] = rotary_emb
	return rotary_emb


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
	print(rope)
