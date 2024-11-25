from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import flax
import flax.linen
import jax
import jax.numpy as jnp


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
class RotaryEmbedding(flax.linen.Module):
	head_size: int
	rotary_dim: int
	max_position_embeddings: int
	base: int
	is_neox_style: bool
	dtype: jnp.dtype

	def _compute_inv_freq(self, base):
		return 1.0 / (
			base ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
		)

	def _compute_cos_sin_cache(self):
		inv = self._compute_inv_freq(self.base)
		freqs = jnp.einsum(
			"i,j -> ij",
			jnp.arange(self.max_position_embeddings, dtype=jnp.float32),
			inv,
		)
		return jnp.cos(freqs), jnp.sin(freqs)

	@flax.linen.jit
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: Optional[jnp.ndarray] = None,
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""__call__ pass for the rotary embedding."""
		with jax.ensure_compile_time_eval():
			if offsets is not None:
				positions = positions + offsets
			cos, sin = self._compute_cos_sin_cache()
			cos = cos[positions]
			sin = sin[positions]
			query_rot = _apply_rotary_emb(
				query[..., : self.rotary_dim],
				cos,
				sin,
				self.is_neox_style,
			)
			query = jnp.concatenate(
				(query_rot, query[..., self.rotary_dim :]),
				axis=-1,
			)
			key_rot = _apply_rotary_emb(
				key[..., : self.rotary_dim],
				cos,
				sin,
				self.is_neox_style,
			)
			key = jnp.concatenate(
				(key_rot, key[..., self.rotary_dim :]),
				axis=-1,
			)

			return query.astype(self.dtype), key.astype(self.dtype)


@rope_wraper("linear")
class LinearScalingRotaryEmbedding(RotaryEmbedding):
	scaling_factors: Union[List[float], float]

	def _compute_cos_sin_cache(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

		assert len(self.scaling_factors) == len(offsets)
		return jnp.split(jnp.concatenate(cache_list, axis=0), 2, -1)


@rope_wraper("dynamic")
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
	"""RotaryEmbedding extended with Dynamic NTK scaling."""

	scaling_factor: Union[float, int]

	def _compute_cos_sin_cache(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
		scaling_factor = self.scaling_factor
		max_len = self.max_position_embeddings * scaling_factor
		base = self.base * (
			(scaling_factor * max_len / self.max_position_embeddings) - (scaling_factor - 1)
		) ** (self.rotary_dim / (self.rotary_dim - 2))
		inv_freq = self._compute_inv_freq(base)
		t = jnp.arange(max_len, dtype=jnp.float32)
		freqs = jnp.einsum("i,j -> ij", t, inv_freq)
		return jnp.cos(freqs), jnp.sin(freqs)


@partial(
	jax.jit,
	static_argnames=[
		"num_rotations",
		"dim",
		"base",
		"max_position_embeddings",
	],
)
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
) -> Tuple[int, int]:
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


@partial(
	jax.jit,
	static_argnames=[
		"base",
		"rotary_dim",
		"scaling_factor",
		"extrapolation_factor",
		"beta_fast",
		"beta_slow",
		"max_position_embeddings",
		"mscale",
		"mscale_all_dim",
		"attn_factor",
	],
)
def deepseek_yarn_sincos_compute(
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

	return jnp.cos(freqs) * mscale, jnp.sin(freqs) * mscale


@rope_wraper("yarn")
class YaRNScalingRotaryEmbedding(RotaryEmbedding):
	"""RotaryEmbedding extended with YaRN method.

	Credits to Peng et al. github.com/jquesnelle/yarn
	"""

	scaling_factor: Union[float, int] = 1.0
	extrapolation_factor: float = 1.0
	attn_factor: float = 1.0
	beta_fast: int = 32
	beta_slow: int = 1

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
		mscale = _yarn_get_mscale(self.scaling_factor) * self.attn_factor
		cos = jnp.cos(freqs) * mscale
		sin = jnp.sin(freqs) * mscale 
		return cos, sin


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
	if scale <= 1:
		return 1.0
	return 0.1 * mscale * jnp.log(scale) + 1.0


@rope_wraper("longrope")
class Phi3LongRoPEScaledRotaryEmbedding(flax.linen.Module):
	head_size: int
	rotary_dim: int
	max_position_embeddings: int
	original_max_position_embeddings: int
	base: int
	is_neox_style: bool
	dtype: jnp.dtype
	short_factor: List[float]
	long_factor: List[float]
	short_mscale: Optional[float] = None
	long_mscale: Optional[float] = None

	def setup(self):
		if self.rotary_dim != self.head_size:
			raise ValueError(
				f"`Phi3LongRoPEScaledRotaryEmbedding` does not support "
				f"rotary_dim != head_size ({self.rotary_dim}!={self.head_size})."
			)
		if not self.is_neox_style:
			raise ValueError("`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style.")

		scale = self.max_position_embeddings / self.original_max_position_embeddings
		if scale <= 1.0:
			scaling_factor = 1.0
		else:
			scaling_factor = jnp.sqrt(
				1 + jnp.log(scale) / jnp.log(self.original_max_position_embeddings)
			)

		if self.short_mscale is None:
			self.short_mscale = scaling_factor
		if self.long_mscale is None:
			self.long_mscale = scaling_factor

	def _compute_cos_sin_cache(
		self,
		position_embeddings: int,
		rescale_factors: List[float],
		mscale: float,
	) -> jnp.ndarray:
		rescale_factors = jnp.array(rescale_factors, dtype=jnp.float32)
		inv_freq = 1.0 / (
			rescale_factors
			* (
				self.base
				** (jnp.arange(0, self.head_size, 2, dtype=jnp.float32) / self.head_size)
			)
		)

		t = jnp.arange(position_embeddings, dtype=jnp.float32)
		freqs = jnp.einsum("i,j -> ij", t, inv_freq)
		cos = jnp.cos(freqs) * mscale
		sin = jnp.sin(freqs) * mscale
		cache = jnp.concatenate([cos, sin], axis=-1)
		return cache

	@flax.linen.jit
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: Optional[jnp.ndarray] = None,
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		short_cache = self._compute_cos_sin_cache(
			position_embeddings=self.original_max_position_embeddings,
			rescale_factors=self.short_factor,
			mscale=self.short_mscale,
		)
		long_cache = self._compute_cos_sin_cache(
			position_embeddings=self.max_position_embeddings,
			rescale_factors=self.long_factor,
			mscale=self.long_mscale,
		)
		long_prompt_offset = (
			jnp.any(positions > self.original_max_position_embeddings).astype(jnp.float32)
			* jnp.full_like(positions, self.original_max_position_embeddings)
		).astype(jnp.int32)
		idx = (
			positions + long_prompt_offset if long_prompt_offset is not None else positions
		)
		idx = idx + offsets if offsets is not None else idx
		cos_sin = jnp.concatenate([short_cache, long_cache], axis=0)[idx]
		cos, sin = jnp.split(cos_sin, 2, axis=-1)
		cos = cos.repeat(2, axis=-1).reshape(cos.shape[0], -1, 1, cos.shape[-1] * 2)
		sin = sin.repeat(2, axis=-1).reshape(cos.shape[0], -1, 1, sin.shape[-1] * 2)
		query = query * cos + _rotate_neox(query) * sin
		key = key * cos + _rotate_neox(key) * sin
		return query, key


@rope_wraper("deepseek_yarn")
class DeepseekScalingRotaryEmbedding(flax.linen.Module):
	"""RotaryEmbedding extended with YaRN method."""

	head_size: int
	rotary_dim: int
	max_position_embeddings: int
	base: int
	is_neox_style: bool
	dtype: jnp.dtype
	scaling_factor: float
	extrapolation_factor: float = 1
	attn_factor: float = 1
	beta_fast: int = 32
	beta_slow: int = 1
	mscale: float = 1
	mscale_all_dim: float = 0

	@flax.linen.jit
	def __call__(
		self,
		positions: jnp.ndarray,
		query: jnp.ndarray,
		key: jnp.ndarray,
		offsets: Optional[jnp.ndarray] = None,
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		if offsets is not None:
			positions += offsets
		query_rot = query[..., : self.rotary_dim]
		key_rot = key[..., : self.rotary_dim]

		if self.rotary_dim < self.head_size:
			query_pass = query[..., self.rotary_dim :]
			key_pass = key[..., self.rotary_dim :]

		cos, sin = deepseek_yarn_sincos_compute(
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

		target_sc_shape = (query.shape[0], -1, 1, self.rotary_dim)
		if self.is_neox_style:
			cos = cos[positions].repeat(2, axis=1).reshape(target_sc_shape)
			sin = sin[positions].repeat(2, axis=1).reshape(target_sc_shape)
		else:
			cos = cos[positions].repeat_interleave(2, axis=1).reshape(target_sc_shape)
			sin = sin[positions].repeat_interleave(2, axis=1).reshape(target_sc_shape)
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


@rope_wraper("llama3")
class Llama3RotaryEmbedding(RotaryEmbedding):
	scaling_factor: float
	low_freq_factor: float
	high_freq_factor: float
	orig_max_position: int

	def _compute_inv_freq(self, base) -> jnp.ndarray:
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
			extra_kwargs = {
				k: v for k, v in rope_scaling.items() if k in ("short_mscale", "long_mscale")
			}
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
				**extra_kwargs,
			)
		else:
			raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

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
