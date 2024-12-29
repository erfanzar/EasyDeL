"""
Efficient Ring Attention Implementation for Single-Device Execution

This module provides an optimized implementation of ring attention,
originally inspired by the work of Liu et al. (2023)
([https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)).
It incorporates the following enhancements:

- Single-Device Focus: Adapted for efficient execution on a single device,
  removing the need for parallel communication primitives.
- Enhanced JIT Compatibility: Streamlined for smoother integration with
  JAX's Just-In-Time (JIT) compilation.
- Performance Optimizations:  Includes code optimizations for improved speed
  and memory usage.

Note: While based on existing implementations, this version offers significant
modifications to enhance its usability and performance in single-device and multi-host
settings.
- also adding softmax scale option to support custom scales
"""

import dataclasses
import functools
import typing as tp
from functools import partial

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange
from jax import extend
from jax.experimental import pallas as pl  # type: ignore[import]
from jax.experimental.pallas import tpu as pltpu  # type: ignore[import]

_PLATFORM = extend.backend.get_backend().platform
_INTERPRET = _PLATFORM != "tpu"
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8


class SegmentIds(tp.NamedTuple):
	"""SegmentIds for Q and KV sequences.

	SegmentIds are used to generate segment mask, which prevents attention between
	different segments in the input sequence. Each array is a list of ids
	(integers).
	Only the token with the same id can attend to each other.

	Attributes:
	  query: segment ids along the Q sequence.
	  kv: segment ids along the KV sequence.
	"""

	query: jax.Array  # [q_seq_len]
	kv: jax.Array  # [kv_seq_len]


def _ring_flash_attention_fwd_tpu(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: tp.Optional[chex.Array] = None,
	segment_ids: tp.Optional[SegmentIds] = None,
	cache_idx: tp.Optional[int] = None,
	axis_name: tp.Optional[str] = None,
	float32_logits: bool = True,
	softmax_scale: tp.Optional[float] = None,
	blocksize_q: int = 256,
	blocksize_k: int = 256,
	blocksize_c: tp.Optional[int] = None,
) -> tp.Tuple[chex.Array, tp.Tuple]:
	"""Forward pass for ring attention on TPU.

	Args:
			query: Query array of shape (batch, query_len, num_heads, dim_per_head).
			key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
			value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
			bias: tp.Optional bias array. Its shape depends on the attention mechanism.
			segment_ids: tp.Optional segment ids for Q and KV sequences.
			cache_idx: tp.Optional cache index for use with caching.
			axis_name: tp.Optional name of the axis to ppermute over (for multi-host support).
			float32_logits: Whether to compute logits in float32.
			softmax_scale: tp.Optional scaling factor for the softmax function.
			blocksize_q: Block size for the query sequence.
			blocksize_k: Block size for the key/value sequence.
			blocksize_c: tp.Optional block size for causal masking.

	Returns:
			A tuple containing the output array and a tuple of intermediate values for use in the backward pass.
	"""
	if float32_logits:
		query, key = query.astype(jnp.float32), key.astype(jnp.float32)
	query, key, value = map(
		lambda x: rearrange(x, "b query h d -> b h query d"), [query, key, value]
	)
	batch, num_heads, q_len, dim_per_head = query.shape
	batch, num_heads, kv_len, dim_per_head = key.shape
	if bias is not None:
		bias = bias[:, 0, 0]  # (batch, k_len)

	o = jnp.zeros((batch, num_heads, q_len, dim_per_head)).astype(query.dtype)
	l = jnp.zeros((batch, num_heads, q_len)).astype(query.dtype)
	m = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(query.dtype)

	axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
	q_blocksize, kv_blocksize = (q_len, kv_len)
	axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
	if segment_ids is not None:
		if cache_idx is None:
			q_offset = axis_idx * q_len
		else:
			q_offset = cache_idx
		q_segment_ids = lax.dynamic_slice_in_dim(segment_ids, q_offset, q_len, axis=-1)

	blocksizes = BlockSizes(
		blocksize_q=blocksize_q,
		blocksize_k_major=blocksize_k,
		blocksize_k=blocksize_k,
		blocksize_b=1,
		blocksize_q_major_dkv=blocksize_q,
		blocksizek_major_dkv=blocksize_k,
		blocksizek_dkv=blocksize_k,
		blocksizeq_dkv=blocksize_q,
		blocksizek_major_dq=blocksize_k,
		blocksizek_dq=blocksize_k,
		blocksizeq_dq=blocksize_q,
	)

	if softmax_scale is None:
		softmax_scale = query.shape[-1] ** -0.5

	def scan_kv_block(carry, idx):
		o, l, m, key, value = carry
		axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
		if bias is not None:
			attn_bias_slice = lax.dynamic_slice_in_dim(
				bias,
				(axis_idx - idx) % axis_size * kv_len,
				kv_len,
				axis=-1,
			)
		else:
			attn_bias_slice = None
		if segment_ids is not None:
			kv_segment_ids = lax.dynamic_slice_in_dim(
				segment_ids,
				(axis_idx - idx) % axis_size * kv_len,
				kv_len,
				axis=-1,
			)
			segment_ids_slice = SegmentIds(query=q_segment_ids, kv=kv_segment_ids)
		else:
			segment_ids_slice = None
		if cache_idx is None:
			q_blocksizeidx = axis_idx
			q_chunk_idx_start = q_blocksizeidx * (q_blocksize // blocksize_q)
		else:
			q_chunk_idx_start = cache_idx // blocksize_q
		k_blocksizeidx = (axis_idx - idx) % axis_size
		k_chunk_idx_start = k_blocksizeidx * (kv_blocksize // blocksize_k)
		o, l, m = _flash_attention_fwd(
			query,
			key,
			value,
			carry=(o, l, m),
			q_chunk_idx_start=q_chunk_idx_start,
			k_chunk_idx_start=k_chunk_idx_start,
			ab=attn_bias_slice,
			segment_ids=segment_ids_slice,
			save_residuals=False,
			blocksize_c=blocksize_c,
			softmax_scale=softmax_scale,
			blocksizes=blocksizes,
			debug=False,
		)
		key, value = map(
			lambda x: lax.ppermute(
				x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]
			)
			if axis_name is not None
			else x,
			(key, value),
		)
		return (o, l, m, key, value), None

	(o, l, m, _, _), _ = lax.scan(
		scan_kv_block,
		init=(o, l, m, key, value),
		xs=jnp.arange(0, axis_size),
	)

	output = rearrange(o.astype(value.dtype), "b h query d -> b query h d")
	return output, (o, query, key, value, bias, segment_ids, cache_idx, l, m)


def _ring_flash_attention_bwd_tpu(
	axis_name: tp.Optional[str],
	float32_logits: bool,
	softmax_scale: tp.Optional[float],
	blocksize_q: int,
	blocksize_k: int,
	blocksize_c: tp.Optional[int],
	res: tp.Tuple,
	g: chex.Array,
) -> tp.Tuple[chex.Array, chex.Array, chex.Array, None, None, None]:
	"""Backward pass for ring attention on TPU.

	Args:
	    axis_name: tp.Optional name of the axis to ppermute over.
	    float32_logits: Whether logits were computed in float32.
	    softmax_scale: Softmax scaling factor used in the forward pass.
	    blocksize_q: Block size for the query sequence.
	    blocksize_k: Block size for the key/value sequence.
	    blocksize_c: Block size for causal masking.
	    res: Residuals from the forward pass.
	    g: Gradient of the output.

	Returns:
	    tp.Tuple containing the gradients of query, key, value, bias, segment_ids, and cache_idx.
	"""
	del float32_logits
	o, query, key, value, bias, segment_ids, cache_idx, l, m = res
	_, _, kv_len, _ = key.shape
	axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
	dq = jnp.zeros_like(query, dtype=jnp.float32)
	dk = jnp.zeros_like(key, dtype=jnp.float32)
	dv = jnp.zeros_like(value, dtype=jnp.float32)
	q_blocksize, kv_blocksize = (query.shape[2], key.shape[2])
	if softmax_scale is None:
		softmax_scale = query.shape[-1] ** -0.5
	axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
	if segment_ids is not None:
		if cache_idx is None:
			q_offset = axis_idx * q_blocksize
		else:
			q_offset = cache_idx
		q_segment_ids = lax.dynamic_slice_in_dim(
			segment_ids, q_offset, q_blocksize, axis=-1
		)
	g = rearrange(g, "b query h d -> b h query d")

	blocksizes = BlockSizes(
		blocksize_q=blocksize_q,
		blocksize_k_major=blocksize_k,
		blocksize_k=blocksize_k,
		blocksize_b=1,
		blocksize_q_major_dkv=blocksize_q,
		blocksizek_major_dkv=blocksize_k,
		blocksizek_dkv=blocksize_k,
		blocksizeq_dkv=blocksize_q,
		blocksizek_major_dq=blocksize_k,
		blocksizek_dq=blocksize_k,
		blocksizeq_dq=blocksize_q,
	)

	def scan_kv_block(carry, idx):
		dq, dk, dv, key, value = carry
		axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
		if bias is not None:
			attn_bias_slice = lax.dynamic_slice_in_dim(
				bias,
				(axis_idx - idx) % axis_size * kv_len,
				kv_len,
				axis=-1,
			)
		else:
			attn_bias_slice = None
		if segment_ids is not None:
			kv_segment_ids = lax.dynamic_slice_in_dim(
				segment_ids,
				(axis_idx - idx) % axis_size * kv_len,
				kv_len,
				axis=-1,
			)
			segment_ids_slice = SegmentIds(query=q_segment_ids, kv=kv_segment_ids)
		else:
			segment_ids_slice = None
		if cache_idx is None:
			q_blocksizeidx = axis_idx
			q_chunk_idx_start = q_blocksizeidx * (q_blocksize // blocksize_q)
		else:
			q_chunk_idx_start = cache_idx // blocksize_q
		k_blocksizeidx = (axis_idx - idx) % axis_size
		k_chunk_idx_start = k_blocksizeidx * (kv_blocksize // blocksize_k)
		(
			dq_i,
			dk_i,
			dv_i,
		) = _flash_attention_bwd(
			save_residuals=False,
			blocksize_c=blocksize_c,
			softmax_scale=softmax_scale,
			blocksizes=blocksizes,
			debug=False,
			q_chunk_idx_start=q_chunk_idx_start,
			k_chunk_idx_start=k_chunk_idx_start,
			residuals=(query, key, value, attn_bias_slice, segment_ids_slice, o, l, m),
			do=g,
		)
		dq += dq_i
		dk += dk_i
		dv += dv_i
		key, value, dk, dv = map(
			lambda x: lax.ppermute(
				x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]
			)
			if axis_name is not None
			else x,
			(key, value, dk, dv),
		)
		return (dq, dk, dv, key, value), None

	(dq, dk, dv, key, value), _ = lax.scan(
		scan_kv_block, init=(dq, dk, dv, key, value), xs=jnp.arange(0, axis_size)
	)
	dq, dk, dv = dq.astype(query.dtype), dk.astype(key.dtype), dv.astype(value.dtype)
	dq, dk, dv = map(lambda x: rearrange(x, "b h query d -> b query h d"), (dq, dk, dv))
	return dq, dk, dv, None, None, None


@partial(
	jax.custom_vjp,
	nondiff_argnums=[6, 7, 8, 9, 10, 11],
)
def ring_attention_tpu(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: tp.Optional[chex.Array] = None,
	segment_ids: tp.Optional[SegmentIds] = None,
	cache_idx: tp.Optional[int] = None,
	axis_name: tp.Optional[str] = None,
	float32_logits: bool = True,
	softmax_scale: tp.Optional[float] = None,
	blocksize_q: int = 256,
	blocksize_k: int = 256,
	blocksize_c: tp.Optional[int] = None,
) -> chex.Array:
	"""Computes ring attention using FlashAttention on TPU.

	Args:
	    query: Query array of shape (batch, query_len, num_heads, dim_per_head).
	    key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
	    value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
	    bias: tp.Optional bias array.  Its shape depends on the attention mechanism.
	    segment_ids: tp.Optional segment ids for Q and KV sequences.
	    cache_idx: tp.Optional cache index for use with caching.
	    axis_name: tp.Optional name of the axis to ppermute over (for multi-host support).
	    float32_logits: Whether to compute logits in float32.
	    softmax_scale: tp.Optional scaling factor for the softmax function.
	    blocksize_q: Block size for the query sequence.
	    blocksize_k: Block size for the key/value sequence.
	    blocksize_c: tp.Optional block size for causal masking.


	Returns:
	    Output array of shape (batch, query_len, num_heads, dim_per_head).
	"""
	y, _ = _ring_flash_attention_fwd_tpu(
		query,
		key,
		value,
		bias,
		segment_ids,
		cache_idx,
		axis_name,
		float32_logits,
		softmax_scale,
		blocksize_q,
		blocksize_k,
		blocksize_c,
	)
	return y


ring_attention_tpu.defvjp(
	_ring_flash_attention_fwd_tpu,
	_ring_flash_attention_bwd_tpu,
)


@dataclasses.dataclass(frozen=True)
class BlockSizes:
	blocksize_q: int
	blocksize_k_major: int
	blocksize_k: int
	blocksize_b: int

	blocksize_q_major_dkv: int | None = None
	blocksizek_major_dkv: int | None = None
	blocksizek_dkv: int | None = None
	blocksizeq_dkv: int | None = None

	blocksizek_major_dq: int | None = None
	blocksizek_dq: int | None = None
	blocksizeq_dq: int | None = None

	def __post_init__(self):
		def verify_major_minor(prefix, suffix, major, minor):
			if minor > major:
				raise ValueError(
					f"{prefix}{suffix}={minor} should be smaller than"
					f" {prefix}_major{suffix}={major}"
				)
			if major % minor != 0:
				raise ValueError(
					f"{prefix}{suffix}={minor} should divide" f" {prefix}_major{suffix}={major}"
				)

		verify_major_minor("blocksize_k", "", self.blocksize_k_major, self.blocksize_k)
		if self.blocksize_q_major_dkv is not None and self.blocksizeq_dkv is not None:
			verify_major_minor(
				"blocksize_q", "_dkv", self.blocksize_q_major_dkv, self.blocksizeq_dkv
			)
		if self.blocksizek_major_dkv is not None and self.blocksizek_dkv is not None:
			verify_major_minor(
				"blocksize_k", "_dkv", self.blocksizek_major_dkv, self.blocksizek_dkv
			)
		if self.blocksizek_major_dq is not None and self.blocksizek_dq is not None:
			verify_major_minor(
				"blocksize_k", "_dq", self.blocksizek_major_dq, self.blocksizek_dq
			)

	@property
	def has_backward_blocks(self) -> bool:
		backward_blocks = (
			self.blocksize_q_major_dkv,
			self.blocksizek_major_dkv,
			self.blocksizeq_dkv,
			self.blocksizek_dkv,
			self.blocksizek_major_dq,
			self.blocksizek_dq,
			self.blocksizeq_dq,
		)
		return all(b is not None for b in backward_blocks)

	@classmethod
	def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
		del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
		return BlockSizes(
			blocksize_q=128,
			blocksize_k_major=128,
			blocksize_k=128,
			blocksize_b=1,
			blocksize_q_major_dkv=128,
			blocksizek_major_dkv=128,
			blocksizek_dkv=128,
			blocksizeq_dkv=128,
			blocksizek_major_dq=128,
			blocksizek_dq=128,
			blocksizeq_dq=128,
		)


def _flash_attention(
	query,
	key,
	value,
	carry,
	q_chunk_idx_start,
	k_chunk_idx_start,
	ab,
	segment_ids,
	save_residuals,
	blocksize_c,
	softmax_scale,
	blocksizes,
	debug,
):
	return _flash_attention_impl(
		query,
		key,
		value,
		carry,
		q_chunk_idx_start,
		k_chunk_idx_start,
		ab,
		segment_ids,
		save_residuals,
		blocksize_c,
		softmax_scale,
		blocksizes.blocksize_b,
		blocksizes.blocksize_q,
		blocksizes.blocksize_k_major,
		blocksizes.blocksize_k,
		debug,
	)


def _flash_attention_fwd(
	query,
	key,
	value,
	carry,
	q_chunk_idx_start,
	k_chunk_idx_start,
	ab,
	segment_ids,
	save_residuals,
	blocksize_c,
	softmax_scale,
	blocksizes,
	debug,
):
	if save_residuals:
		raise NotImplementedError("Higher-order AD not supported")
	o, l, m = _flash_attention(
		query,
		key,
		value,
		carry,
		q_chunk_idx_start,
		k_chunk_idx_start,
		ab,
		segment_ids,
		True,
		blocksize_c,
		softmax_scale,
		blocksizes,
		debug,
	)
	return o, l, m


def _flash_attention_bwd(
	save_residuals: bool,
	blocksize_c: tp.Optional[int],
	softmax_scale: float,
	blocksizes: BlockSizes,
	debug: bool,
	q_chunk_idx_start,
	k_chunk_idx_start,
	residuals,
	do,
):
	"""VJP rule for FlashAttention."""
	if save_residuals:
		raise NotImplementedError("Higher-order AD not supported")
	(query, key, value, ab, segment_ids, o, l, m) = residuals
	if not blocksizes.has_backward_blocks:
		raise ValueError(
			"Program is being differentiated, but not all backward blocks are" " specified"
		)

	di = jnp.sum(
		o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1
	)  # [batch_size, num_heads, q_seq_len]

	dk, dv = _flash_attention_bwd_dkv(
		q_chunk_idx_start,
		k_chunk_idx_start,
		query,
		key,
		value,
		ab,
		segment_ids,
		l,
		m,
		do,
		di,
		blocksizeq_major=blocksizes.blocksize_q_major_dkv,
		blocksize_k_major=blocksizes.blocksizek_major_dkv,
		blocksize_k=blocksizes.blocksizek_dkv,
		blocksize_q=blocksizes.blocksizeq_dkv,
		softmax_scale=softmax_scale,
		blocksize_c=blocksize_c,
		mask_value=DEFAULT_MASK_VALUE,
		debug=debug,
	)

	dq, ds = _flash_attention_bwd_dq(
		q_chunk_idx_start,
		k_chunk_idx_start,
		query,
		key,
		value,
		ab,
		segment_ids,
		l,
		m,
		do,
		di,
		blocksizeq_major=blocksizes.blocksizeq_dq,
		blocksize_k_major=blocksizes.blocksizek_major_dq,
		blocksize_k=blocksizes.blocksizek_dq,
		softmax_scale=softmax_scale,
		blocksize_c=blocksize_c,
		mask_value=DEFAULT_MASK_VALUE,
		debug=debug,
	)
	return dq, dk, dv


MIN_blocksize = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def _flash_attention_kernel(
	q_idx_chunk_start,
	k_idx_chunk_start,
	q_tile_ref,
	*args,
	**kwargs,
):
	blocksize_b = q_tile_ref.shape[0]
	# If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
	if kwargs["blocksize_k"] == kwargs["kv_seq_len"]:
		raise NotImplementedError()
	else:
		kernel = _flash_attention_kernel_single_batch
	for batch_idx in range(blocksize_b):
		kernel(
			(batch_idx, 0),
			q_idx_chunk_start,
			k_idx_chunk_start,
			q_tile_ref,
			*args,
			**kwargs,
		)


def _flash_attention_kernel_single_batch(
	batch_idx: tuple[int, ...],
	q_chunk_idx_start_ref,
	k_chunk_idx_start_ref,
	q_tile_ref,
	k_tile_ref,
	v_tile_ref,
	acc_tile_ref,
	l_tile_ref,
	m_tile_ref,
	ab_tile_ref,
	q_segment_ids_tile_ref,
	kv_segment_ids_tile_ref,  # Input arrays
	o_tile_ref,  # Output arrays
	m_scratch_ref,
	l_scratch_ref,
	acc_scratch_ref,
	l_ref: tp.Any | None = None,
	m_ref: tp.Any | None = None,
	*,
	blocksize_c,
	softmax_scale,
	blocksize_k,
	kv_seq_len,
	mask_value,
	blocksize_q,
):
	blocksize_k_major = k_tile_ref.shape[2]
	blocksize_q = q_tile_ref.shape[2]
	head_dim = q_tile_ref.shape[-1]

	kv_seq_idx = pl.program_id(3)

	@pl.when(kv_seq_idx == 0)
	def start_new_sequence():
		m_scratch_ref[batch_idx] = m_tile_ref[batch_idx]
		l_scratch_ref[batch_idx] = l_tile_ref[batch_idx]
		acc_scratch_ref[batch_idx] = acc_tile_ref[batch_idx]

	try:
		q_chunk_idx_start = q_chunk_idx_start_ref[0]
		k_chunk_idx_start = k_chunk_idx_start_ref[0]
	except ValueError:
		q_chunk_idx_start = q_chunk_idx_start_ref
		k_chunk_idx_start = k_chunk_idx_start_ref

	q_seq_idx = pl.program_id(2)
	if blocksize_c is not None:
		should_run = below_or_on_diag(
			q_seq_idx + q_chunk_idx_start,
			blocksize_q,
			kv_seq_idx + k_chunk_idx_start,
			blocksize_k_major,
			blocksize_c,
		)
	else:
		should_run = True

	@pl.when(should_run)
	def run():
		@functools.partial(
			lax.fori_loop,
			0,
			blocksize_k_major // blocksize_k,
			init_val=None,
			unroll=True,
		)
		def body(i, _):
			m_prev = m_scratch_ref[batch_idx]
			l_prev = l_scratch_ref[batch_idx]
			query = q_tile_ref[batch_idx]  # [blocksize_q, head_dim]
			start_k = i * blocksize_k
			key = pl.load(
				k_tile_ref, (*batch_idx, pl.dslice(start_k, blocksize_k), slice(None))
			)  # [blocksize_k, head_dim]

			s = jax.lax.dot_general(
				query, key, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
			)  # [blocksize_q, blocksize_k]

			# Add attention bias if needed.
			if ab_tile_ref is not None:
				ab = pl.load(
					ab_tile_ref,
					(
						batch_idx[0],
						pl.dslice(0, blocksize_q),
						pl.dslice(start_k, blocksize_k),
					),
				).astype(jnp.float32)
				s += ab

			if softmax_scale != 1.0:
				s *= softmax_scale

			mask = None
			if q_segment_ids_tile_ref is not None:
				repeats, rem = divmod(blocksize_k, NUM_LANES)
				if rem:
					raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
				q_segment_ids = pltpu.repeat(
					q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1
				)  # [blocksize_q, blocksize_k].
				kv_segment_ids = pl.load(
					kv_segment_ids_tile_ref,
					(batch_idx[0], pl.dslice(1), pl.dslice(start_k, blocksize_k)),
				)  # [1, blocksize_k].
				mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

			if blocksize_c is not None:
				mask_shape = (blocksize_q, blocksize_k)
				row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
				row_ids += (q_seq_idx + q_chunk_idx_start) * blocksize_q
				row_ids = jax.lax.div(row_ids, blocksize_c)
				col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
				col_ids += (kv_seq_idx + k_chunk_idx_start) * blocksize_k_major + start_k
				col_ids = jax.lax.div(col_ids, blocksize_c)
				causal_mask = col_ids <= row_ids
				mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

			s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

			m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [blocksize_q, 1].
			m_next = jnp.maximum(m_prev, m_curr)  # Shape [blocksize_q, 128].

			blocksizek_repeats, rem = divmod(blocksize_k, MIN_blocksize)
			if rem:
				raise NotImplementedError(
					f"{blocksize_k=} should be a multiple of {MIN_blocksize}"
				)
			p = jnp.exp(s - pltpu.repeat(m_next, blocksizek_repeats, 1))

			alpha = jnp.exp(m_prev - m_next)  # Shape [blocksize_q, 128].

			l_corr = alpha * l_prev

			l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [blocksize_q, 128]

			head_dim_repeats, rem = divmod(head_dim, MIN_blocksize)
			l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
			if rem:
				if head_dim_repeats == 0:
					l_broadcast = lambda l: l[:, :head_dim]
				else:
					raise NotImplementedError(
						f"{head_dim=} should be a multiple of {MIN_blocksize} if larger"
					)
			l_scratch_ref[batch_idx] = l_next
			m_scratch_ref[batch_idx] = m_next

			l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
			acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
			value = pl.load(
				v_tile_ref, (*batch_idx, pl.dslice(start_k, blocksize_k), slice(None))
			)
			o_curr = jax.lax.dot(
				p.astype(value.dtype), value, preferred_element_type=jnp.float32
			)
			acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

	@pl.when(kv_seq_idx == (kv_seq_len // blocksize_k_major) - 1)
	def store_output():
		o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
		if l_ref is not None:
			l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
		if m_ref is not None:
			m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_impl(
	query,
	key,
	value,
	carry,
	q_chunk_idx_start,
	k_chunk_idx_start,
	ab,
	segment_ids,
	save_residuals,
	blocksize_c,
	softmax_scale,
	blocksize_b,
	blocksize_q,
	blocksize_k_major,
	blocksize_k,
	debug,
):
	if blocksize_c is not None:
		assert blocksize_c % blocksize_q == 0 or blocksize_q % blocksize_c == 0
		assert blocksize_c % blocksize_k == 0 or blocksize_k % blocksize_c == 0
	assert blocksize_k_major == blocksize_k, (blocksize_k_major, blocksize_k)

	batch_size, num_heads, q_seq_len, head_dim = query.shape
	_, _, kv_seq_len, _ = key.shape

	acc, l_prev, m_prev = carry
	l_prev, m_prev = map(
		lambda x: jnp.broadcast_to(x[..., None], (*x.shape, MIN_blocksize)),
		(l_prev, m_prev),
	)
	try:
		q_chunk_idx_start, k_chunk_idx_start = (
			q_chunk_idx_start[None],
			k_chunk_idx_start[None],
		)
	except TypeError:
		...
	_verify_block("blocksize_q", "q_seq_len", blocksize_q, q_seq_len, should_divide=False)
	_verify_block("blocksize_k_major", "kv_seq_len", blocksize_k_major, kv_seq_len)
	_verify_block("blocksize_k", "kv_seq_len", blocksize_k, kv_seq_len)
	_verify_block("blocksize_b", "batch", blocksize_b, batch_size, should_divide=False)

	grid = (
		pl.cdiv(batch_size, blocksize_b),
		num_heads,
		pl.cdiv(q_seq_len, blocksize_q),
		kv_seq_len // blocksize_k_major,
	)

	def q_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, q_seq_index, 0)

	def kv_index_map(
		batch_index,
		head_index,
		q_seq_index,
		kv_seq_index,
		q_idx_ref,
		k_idx_ref,
	):
		if blocksize_c is not None:
			next_kv_index = lax.select(
				below_or_on_diag(
					q_seq_index + q_idx_ref[0],
					blocksize_q,
					kv_seq_index + k_idx_ref[0],
					blocksize_k_major,
					blocksize_c,
				),
				kv_seq_index,
				0,
			)
		else:
			next_kv_index = kv_seq_index
		return (batch_index, head_index, next_kv_index, 0)

	def ab_index_map(
		batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
	):
		if blocksize_c is not None:
			should_run = below_or_on_diag(
				q_seq_index + q_idx_ref[0],
				blocksize_q,
				kv_seq_index + k_idx_ref[0],
				blocksize_k_major,
				blocksize_c,
			)
			next_kv_index = lax.select(should_run, kv_seq_index, 0)
		else:
			next_kv_index = kv_seq_index

		return (batch_index, 0, next_kv_index)

	def o_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, q_seq_index, 0)

	def lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, q_seq_index, 0)

	kernel = functools.partial(
		_flash_attention_kernel,
		mask_value=DEFAULT_MASK_VALUE,
		softmax_scale=softmax_scale,
		kv_seq_len=kv_seq_len,
		blocksize_q=blocksize_q,
		blocksize_k=blocksize_k,
		blocksize_c=blocksize_c,
	)
	out_shape = [jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype)]
	out_specs = [pl.BlockSpec(o_index_map, (blocksize_b, 1, blocksize_q, head_dim))]

	if blocksize_k != kv_seq_len:
		scratch_shape = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
		m_scratch = scratch_shape((blocksize_b, 1, blocksize_q, MIN_blocksize))
		l_scratch = scratch_shape((blocksize_b, 1, blocksize_q, MIN_blocksize))
		acc_scratch = scratch_shape((blocksize_b, 1, blocksize_q, head_dim))
		out_shape += [m_scratch, l_scratch, acc_scratch]
		out_specs += [
			pl.BlockSpec(lambda *_: (0, 0, 0, 0), m_scratch.shape),
			pl.BlockSpec(lambda *_: (0, 0, 0, 0), l_scratch.shape),
			pl.BlockSpec(lambda *_: (0, 0, 0, 0), acc_scratch.shape),
		]
	else:
		raise NotImplementedError("blocksize_k != kv_seq_len not supported at the moment")

	if save_residuals:
		out_specs = [
			*out_specs,
			pl.BlockSpec(lm_index_map, (blocksize_b, 1, blocksize_q, MIN_blocksize)),
			pl.BlockSpec(lm_index_map, (blocksize_b, 1, blocksize_q, MIN_blocksize)),
		]
		l = jax.ShapeDtypeStruct(
			(batch_size, num_heads, q_seq_len, MIN_blocksize), dtype=jnp.float32
		)
		m = jax.ShapeDtypeStruct(
			(batch_size, num_heads, q_seq_len, MIN_blocksize), dtype=jnp.float32
		)
		out_shape = (*out_shape, l, m)

	ab_blocksizespec = (
		pl.BlockSpec(ab_index_map, (blocksize_b, blocksize_q, blocksize_k_major))
		if ab is not None
		else None
	)

	if ab is not None:
		ab = ab[:, None].repeat(blocksize_q, axis=1)

	q_segment_ids_spec = kv_segment_ids_spec = None
	q_segment_ids = kv_segment_ids = None
	if segment_ids is not None:

		def q_segment_ids_index_map(
			batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref
		):
			del head_index
			return (batch_index, q_seq_index, 0)

		def kv_segment_ids_index_map(
			batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
		):
			del head_index
			if blocksize_c is not None:
				next_kv_index = lax.select(
					below_or_on_diag(
						q_seq_index + q_idx_ref[0],
						blocksize_q,
						kv_seq_index + k_idx_ref[0],
						blocksize_k_major,
						blocksize_c,
					),
					kv_seq_index,
					0,
				)
			else:
				next_kv_index = kv_seq_index
			return (batch_index, 0, next_kv_index)

		q_segment_ids_spec = pl.BlockSpec(
			q_segment_ids_index_map, (blocksize_b, blocksize_q, NUM_LANES)
		)
		kv_segment_ids_spec = pl.BlockSpec(
			kv_segment_ids_index_map, (blocksize_b, NUM_SUBLANES, blocksize_k_major)
		)

		q_segment_ids = jax.lax.broadcast_in_dim(
			segment_ids.query,
			(batch_size, q_seq_len, NUM_LANES),
			(
				0,
				1,
			),
		)
		kv_segment_ids = jax.lax.broadcast_in_dim(
			segment_ids.kv,
			(batch_size, NUM_SUBLANES, kv_seq_len),
			(
				0,
				2,
			),
		)

	in_specs = [
		pl.BlockSpec(q_index_map, (blocksize_b, 1, blocksize_q, head_dim)),
		pl.BlockSpec(kv_index_map, (blocksize_b, 1, blocksize_k_major, head_dim)),
		pl.BlockSpec(kv_index_map, (blocksize_b, 1, blocksize_k_major, head_dim)),
		pl.BlockSpec(q_index_map, (blocksize_b, 1, blocksize_q, head_dim)),
		pl.BlockSpec(lm_index_map, (blocksize_b, 1, blocksize_q, MIN_blocksize)),
		pl.BlockSpec(lm_index_map, (blocksize_b, 1, blocksize_q, MIN_blocksize)),
		ab_blocksizespec,
		q_segment_ids_spec,
		kv_segment_ids_spec,
	]
	o, *aux = pl.pallas_call(
		kernel,
		out_shape=out_shape,
		grid_spec=pltpu.PrefetchScalarGridSpec(
			num_scalar_prefetch=2, in_specs=in_specs, out_specs=out_specs, grid=grid
		),
		debug=debug,
		compiler_params=dict(
			mosaic=dict(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"))
		),
		interpret=_INTERPRET,
	)(
		q_chunk_idx_start,
		k_chunk_idx_start,
		query,
		key,
		value,
		acc,
		l_prev,
		m_prev,
		ab,
		q_segment_ids,
		kv_segment_ids,
	)

	if save_residuals:
		l, m = (value[..., 0] for value in aux[-2:])
		return (o, l, m)
	else:
		return o


def _flash_attention_dkv_kernel(
	q_chunk_idx_start_ref,
	k_chunk_idx_start_ref,
	q_tile_ref,
	k_tile_ref,
	v_tile_ref,
	ab_tile_ref,
	q_segment_ids_tile_ref,
	kv_segment_ids_tile_ref,
	l_tile_ref,
	m_tile_ref,
	do_tile_ref,
	di_tile_ref,
	dk_tile_ref,
	dv_tile_ref,
	dk_scratch_ref,
	dv_scratch_ref,
	*,
	softmax_scale: float,
	blocksize_c: tp.Optional[int],
	mask_value: float,
	q_seq_len: int,
	blocksize_q: int,
	blocksize_k: int,
):
	_, _, blocksizeq_major, _ = q_tile_ref.shape
	_, _, blocksize_k_major, _ = k_tile_ref.shape

	q_seq_index = pl.program_id(axis=3)
	kv_seq_index = pl.program_id(axis=2)

	q_chunk_idx_start = q_chunk_idx_start_ref[0]
	k_chunk_idx_start = k_chunk_idx_start_ref[0]

	@pl.when(q_seq_index == 0)
	def start_new_sequence():
		dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
		dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

	def q_body(j, _):
		start_q = j * blocksize_q

		def k_body(i, _):
			start_k = i * blocksize_k
			key = pl.load(k_tile_ref, (0, 0, pl.ds(start_k, blocksize_k), slice(None)))
			value = pl.load(v_tile_ref, (0, 0, pl.ds(start_k, blocksize_k), slice(None)))
			query = pl.load(
				q_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None))
			)  # [blocksize_q, head_dim]
			l = pl.load(
				l_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None))
			)  # [blocksize_q, 128]
			m = pl.load(
				m_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None))
			)  # [blocksize_q, 128]
			do = pl.load(
				do_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None))
			)  # [blocksize_q, 128]
			di = pl.load(
				di_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None))
			).astype(jnp.float32)  # [blocksize_q, 128]

			capped_logits = lax.dot_general(
				query, key, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
			)  # [blocksizeq_major, blocksize_k]

			if ab_tile_ref is not None:
				ab = pl.load(
					ab_tile_ref,
					(
						0,
						pl.dslice(0, blocksize_q),
						pl.dslice(i * blocksize_k, blocksize_k),
					),
				).astype(jnp.float32)
				capped_logits += ab

			if softmax_scale != 1.0:
				capped_logits *= softmax_scale

			mask = None
			if q_segment_ids_tile_ref is not None:
				repeats, rem = divmod(blocksize_k, NUM_LANES)
				if rem:
					raise NotImplementedError()
				q_segment_ids = pl.load(
					q_segment_ids_tile_ref,
					(0, pl.ds(start_q, blocksize_q), slice(None)),
				)  # [blocksize_q, NUM_LANES].
				q_segment_ids = pltpu.repeat(
					q_segment_ids, repeats, axis=1
				)  # [blocksize_q, blocksize_k].
				kv_segment_ids = pl.load(
					kv_segment_ids_tile_ref,
					(slice(None), 0, pl.ds(start_k, blocksize_k)),
				)  # [1, blocksize_k].
				mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

			if blocksize_c is not None:
				mask_shape = (blocksize_q, blocksize_k)
				row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
				row_ids += (q_seq_index + q_chunk_idx_start) * blocksizeq_major + start_q
				row_ids = jax.lax.div(row_ids, blocksize_c)
				col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
				col_ids += (kv_seq_index + k_chunk_idx_start) * blocksize_k_major + start_k
				col_ids = jax.lax.div(col_ids, blocksize_c)
				causal_mask = col_ids <= row_ids
				mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

			capped_logits = (
				capped_logits
				if mask is None
				else capped_logits + jnp.where(mask, 0.0, mask_value)
			)

			p = jnp.exp(capped_logits - pltpu.repeat(m, blocksize_k // MIN_blocksize, axis=1))
			p = p * pltpu.repeat(
				1 / l, blocksize_k // MIN_blocksize, axis=1
			)  # [blocksizeq_major, blocksize_k_major]
			dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
			pl.store(
				dv_scratch_ref,
				(pl.ds(start_k, blocksize_k), slice(None)),
				pl.load(dv_scratch_ref, (pl.ds(start_k, blocksize_k), slice(None)))
				+ dv.astype(dv_scratch_ref.dtype),
			)

			# di: [blocksize_q, 128]
			# do: [blocksize_q, head_dim]
			# value: [blocksize_k_major, head_dim]
			dp = lax.dot_general(
				do, value, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
			)
			ds = (dp - pltpu.repeat(di, blocksize_k // MIN_blocksize, axis=1)) * p

			if softmax_scale != 1.0:
				ds = ds * softmax_scale

			# ds: [blocksizeq_major, blocksize_k_major]
			# query: [blocksizeq_major, head_dim]
			dk = lax.dot(ds.T.astype(do.dtype), query, preferred_element_type=jnp.float32)
			pl.store(
				dk_scratch_ref,
				(pl.ds(start_k, blocksize_k), slice(None)),
				pl.load(dk_scratch_ref, (pl.ds(start_k, blocksize_k), slice(None)))
				+ dk.astype(dk_scratch_ref.dtype),
			)

		lax.fori_loop(0, blocksize_k_major // blocksize_k, k_body, None, unroll=True)

	if blocksize_c is not None:
		should_run = below_or_on_diag(
			q_seq_index + q_chunk_idx_start,
			blocksizeq_major,
			kv_seq_index + k_chunk_idx_start,
			blocksize_k_major,
			blocksize_c,
		)
	else:
		should_run = True

	@pl.when(should_run)
	def run():
		lax.fori_loop(0, blocksizeq_major // blocksize_q, q_body, None, unroll=True)

	@pl.when(q_seq_index == q_seq_len // blocksizeq_major - 1)
	def end_of_q_sequence():
		dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref)
		dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref)


def _flash_attention_bwd_dkv(
	q_chunk_idx_start,
	k_chunk_idx_start,
	query,
	key,
	value,
	ab,
	segment_ids,
	l,
	m,
	do,
	di,
	*,
	blocksizeq_major: int | None,
	blocksize_q: int | None,
	blocksize_k_major: int | None,
	blocksize_k: int | None,
	softmax_scale: float,
	blocksize_c: tp.Optional[int] = None,
	mask_value: float = DEFAULT_MASK_VALUE,
	debug: bool = False,
):
	batch_size, num_heads, q_seq_len, head_dim = query.shape
	_, _, kv_seq_len, _ = key.shape
	q_chunk_idx_start, k_chunk_idx_start = (
		q_chunk_idx_start[None],
		k_chunk_idx_start[None],
	)
	_verify_block("blocksize_q_major_dkv", "q_seq_len", blocksizeq_major, q_seq_len)
	_verify_block("blocksizeq_dkv", "q_seq_len", blocksize_q, q_seq_len)
	_verify_block("blocksizek_major_dkv", "kv_seq_len", blocksize_k_major, kv_seq_len)
	_verify_block("blocksizek_dkv", "kv_seq_len", blocksize_k, kv_seq_len)

	# Broadcast out scalar values
	m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_blocksize))
	l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_blocksize))
	# Preprocess contraction for bwd pass
	di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_blocksize))

	# kv index needs to be before query index since query index is the contractng
	# dimension.
	grid = (
		batch_size,
		num_heads,
		kv_seq_len // blocksize_k_major,
		q_seq_len // blocksizeq_major,
	)

	def qo_index_map(
		batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref
	):
		if blocksize_c is not None:
			# If the query block is skipped, stay at the 0th query block.
			next_q_index = lax.select(
				below_or_on_diag(
					q_seq_index + q_idx_ref[0],
					blocksizeq_major,
					kv_seq_index + k_idx_ref[0],
					blocksize_k_major,
					blocksize_c,
				),
				q_seq_index,
				0,
			)
		else:
			next_q_index = q_seq_index

		return (batch_index, head_index, next_q_index, 0)

	qo_spec = pl.BlockSpec(qo_index_map, (1, 1, blocksizeq_major, head_dim))
	assert qo_spec.block_shape is not None
	assert query.ndim == len(qo_spec.block_shape)
	do_spec = qo_spec
	assert do.ndim == len(qo_spec.block_shape)

	def kv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, kv_seq_index, 0)

	kv_spec = pl.BlockSpec(kv_index_map, (1, 1, blocksize_k_major, head_dim))
	assert kv_spec.block_shape is not None
	assert key.ndim == len(kv_spec.block_shape)
	assert value.ndim == len(kv_spec.block_shape)

	def lm_index_map(batch_index, head_index, _, q_seq_index, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, q_seq_index, 0)

	lm_spec = pl.BlockSpec(lm_index_map, (1, 1, blocksizeq_major, MIN_blocksize))
	assert lm_spec.block_shape is not None
	assert l.ndim == len(lm_spec.block_shape)
	assert m.ndim == len(lm_spec.block_shape)

	di_spec = pl.BlockSpec(qo_index_map, (1, 1, blocksizeq_major, MIN_blocksize))
	assert di_spec.block_shape is not None
	assert di.ndim == len(di_spec.block_shape)

	def ab_index_map(
		batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref
	):
		return (batch_index, 0, kv_seq_index)

	if ab is not None:
		ab = ab[:, None].repeat(blocksizeq_major, axis=1)

	dab_spec = (
		pl.BlockSpec(ab_index_map, (1, blocksizeq_major, blocksize_k_major))
		if ab is not None
		else None
	)

	q_segment_ids_spec = kv_segment_ids_spec = None
	q_segment_ids = kv_segment_ids = None
	if segment_ids is not None:

		def q_segment_ids_index_map(
			batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref
		):
			del head_index
			if blocksize_c is not None:
				next_q_index = lax.select(
					below_or_on_diag(
						q_seq_index + q_idx_ref[0],
						blocksizeq_major,
						kv_seq_index + k_idx_ref[0],
						blocksize_k_major,
						blocksize_c,
					),
					q_seq_index,
					0,
				)
			else:
				next_q_index = q_seq_index
			return (batch_index, next_q_index, 0)

		def kv_segment_ids_index_map(
			batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref
		):
			del head_index
			return (batch_index, 0, kv_seq_index)

		q_segment_ids_spec = pl.BlockSpec(
			q_segment_ids_index_map, (1, blocksizeq_major, NUM_LANES)
		)
		kv_segment_ids_spec = pl.BlockSpec(
			kv_segment_ids_index_map, (1, NUM_SUBLANES, blocksize_k_major)
		)

		q_segment_ids = jax.lax.broadcast_in_dim(
			segment_ids.query,
			(batch_size, q_seq_len, NUM_LANES),
			(
				0,
				1,
			),
		)
		kv_segment_ids = jax.lax.broadcast_in_dim(
			segment_ids.kv,
			(batch_size, NUM_SUBLANES, kv_seq_len),
			(
				0,
				2,
			),
		)

	in_specs = [
		qo_spec,
		kv_spec,
		kv_spec,
		dab_spec,
		q_segment_ids_spec,
		kv_segment_ids_spec,
		lm_spec,
		lm_spec,
		do_spec,
		di_spec,
	]

	out_shapes = [
		jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), key.dtype),
		jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), value.dtype),
		jax.ShapeDtypeStruct((blocksize_k_major, head_dim), jnp.float32),
		jax.ShapeDtypeStruct((blocksize_k_major, head_dim), jnp.float32),
	]

	def dkv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, kv_seq_index, 0)

	dkv_spec = pl.BlockSpec(dkv_index_map, (1, 1, blocksize_k_major, head_dim))
	out_specs = [
		dkv_spec,
		dkv_spec,
		pl.BlockSpec(lambda *_: (0, 0), (blocksize_k_major, head_dim)),
		pl.BlockSpec(lambda *_: (0, 0), (blocksize_k_major, head_dim)),
	]

	kernel = functools.partial(
		_flash_attention_dkv_kernel,
		blocksize_q=blocksize_q,
		blocksize_k=blocksize_k,
		softmax_scale=softmax_scale,
		blocksize_c=blocksize_c,
		mask_value=mask_value,
		q_seq_len=q_seq_len,
	)
	name_scope = f"flash_mha_bwd_dkv_{blocksizeq_major=}_{blocksize_q=}_{blocksize_k_major=}_{blocksize_k=}"
	with jax.named_scope(name_scope):
		dk, dv, _, _ = pl.pallas_call(
			kernel,
			out_shape=out_shapes,
			grid_spec=pltpu.PrefetchScalarGridSpec(
				num_scalar_prefetch=2, in_specs=in_specs, out_specs=out_specs, grid=grid
			),
			debug=debug,
			compiler_params=dict(
				mosaic=dict(
					dimension_semantics=(
						"parallel",
						"parallel",
						"parallel",
						"arbitrary",
					)
				)
			),
			interpret=_INTERPRET,
		)(
			q_chunk_idx_start,
			k_chunk_idx_start,
			query,
			key,
			value,
			ab,
			q_segment_ids,
			kv_segment_ids,
			l,
			m,
			do,
			di,
		)
		assert dk.shape == key.shape
		assert dv.shape == value.shape
	return dk, dv


def _flash_attention_dq_kernel(
	q_chunk_idx_start_ref,
	k_chunk_idx_start_ref,
	q_tile_ref,
	k_tile_ref,
	v_tile_ref,
	ab_tile_ref,
	q_segment_ids_tile_ref,
	kv_segment_ids_tile_ref,
	l_tile_ref,
	m_tile_ref,
	do_tile_ref,
	di_tile_ref,
	dq_tile_ref,
	dq_scratch_ref,
	ds_tile_ref,
	*,
	softmax_scale: float,
	blocksize_c: tp.Optional[int],
	mask_value: float,
	kv_seq_len: int,
	blocksize_k: int,
):
	_, _, blocksize_k_major, _ = k_tile_ref.shape
	_, _, blocksizeq_major, _ = q_tile_ref.shape

	kv_seq_index = pl.program_id(axis=3)
	q_seq_index = pl.program_id(axis=2)

	q_chunk_idx_start = q_chunk_idx_start_ref[0]
	k_chunk_idx_start = k_chunk_idx_start_ref[0]

	@pl.when(kv_seq_index == 0)
	def start_new_sequence():
		dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

	def body(i, _):
		k_slice = pl.ds(i * blocksize_k, blocksize_k)
		query = q_tile_ref[0, 0, :, :]
		key = pl.load(
			k_tile_ref,
			(0, 0, k_slice, slice(None)),
		)  # [blocksize_k, head_dim]
		value = pl.load(
			v_tile_ref,
			(0, 0, k_slice, slice(None)),
		)  # [blocksize_k, head_dim]
		l = l_tile_ref[0, 0, :, :]  # [blocksizeq_major, 128]
		m = m_tile_ref[0, 0, :, :]  # [blocksizeq_major, 128]
		do = do_tile_ref[0, 0, :, :]  # [blocksizeq_major, head_dim]
		di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [blocksizeq_major, 128]

		capped_logits = jax.lax.dot_general(
			query, key, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
		)

		if ab_tile_ref is not None:
			ab = pl.load(
				ab_tile_ref,
				(
					0,
					pl.dslice(0, blocksizeq_major),
					pl.dslice(i * blocksize_k, blocksize_k),
				),
			).astype(jnp.float32)
			capped_logits += ab

		if softmax_scale != 1.0:
			capped_logits *= softmax_scale

		mask = None
		if q_segment_ids_tile_ref is not None:
			repeats, rem = divmod(blocksize_k, NUM_LANES)
			if rem:
				raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
			q_segment_ids = pltpu.repeat(
				q_segment_ids_tile_ref[0], repeats, axis=1
			)  # [blocksize_q, blocksize_k].
			kv_segment_ids = pl.load(
				kv_segment_ids_tile_ref, (slice(None), 0, k_slice)
			)  # [1, blocksize_k].
			mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

		if blocksize_c is not None:
			mask_shape = (blocksizeq_major, blocksize_k)
			row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
			row_ids += (q_seq_index + q_chunk_idx_start) * blocksizeq_major
			row_ids = jax.lax.div(row_ids, blocksize_c)
			col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
			col_ids += (
				kv_seq_index + k_chunk_idx_start
			) * blocksize_k_major + i * blocksize_k
			col_ids = jax.lax.div(col_ids, blocksize_c)
			causal_mask = col_ids <= row_ids
			mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
		capped_logits = (
			capped_logits
			if mask is None
			else capped_logits + jnp.where(mask, 0.0, mask_value)
		)

		p = jnp.exp(capped_logits - pltpu.repeat(m, blocksize_k // MIN_blocksize, axis=1))
		p = p * pltpu.repeat(
			1 / l, blocksize_k // MIN_blocksize, axis=1
		)  # [blocksizeq_major, blocksize_k]

		# di: [blocksizeq_major, 128]
		# do: [blocksizeq_major, head_dim]
		# value: [blocksize_k_major, head_dim]
		dp = jax.lax.dot_general(
			do,
			value,
			TRANS_B_DIM_NUMBERS,
			preferred_element_type=jnp.float32,
		)
		ds = (dp - pltpu.repeat(di, blocksize_k // MIN_blocksize, axis=1)) * p

		if softmax_scale != 1.0:
			ds = ds * softmax_scale

		if ds_tile_ref is not None:
			pl.store(
				ds_tile_ref,
				(0, pl.dslice(None), pl.dslice(i * blocksize_k, blocksize_k)),
				ds.astype(ds_tile_ref.dtype),
			)

		# dp: [blocksizeq_major, blocksize_k]
		# key: [blocksize_k, head_dim]
		dq_scratch_ref[:, :] += lax.dot(
			ds.astype(key.dtype),
			key,
			preferred_element_type=jnp.float32,
		).astype(dq_scratch_ref.dtype)

	if blocksize_c is not None:
		should_run = below_or_on_diag(
			q_seq_index + q_chunk_idx_start,
			blocksizeq_major,
			kv_seq_index + k_chunk_idx_start,
			blocksize_k_major,
			blocksize_c,
		)
		should_not_run = lax.select(should_run, False, True)
	else:
		should_run = True
		should_not_run = False  # type: ignore

	@pl.when(should_run)
	def run():
		lax.fori_loop(0, blocksize_k_major // blocksize_k, body, None, unroll=True)

	@pl.when(should_not_run)
	def zero_out_ds():
		if ds_tile_ref is not None:
			ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

	@pl.when(kv_seq_index == kv_seq_len // blocksize_k_major - 1)
	def end_of_kv_sequence():
		dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref)
		dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
	q_chunk_idx_start,
	k_chunk_idx_start,
	query,
	key,
	value,
	ab,
	segment_ids,
	l,
	m,
	do,
	di,
	*,
	blocksizeq_major: int | None,
	blocksize_k_major: int | None,
	blocksize_k: int | None,
	softmax_scale: float,
	blocksize_c: tp.Optional[int],
	mask_value: float,
	debug: bool,
):
	batch_size, num_heads, q_seq_len, head_dim = query.shape
	_, _, kv_seq_len, _ = key.shape
	q_chunk_idx_start, k_chunk_idx_start = (
		q_chunk_idx_start[None],
		k_chunk_idx_start[None],
	)
	_verify_block("blocksizeq_dq", "q_seq_len", blocksizeq_major, q_seq_len)
	_verify_block("blocksizek_major_dq", "kv_seq_len", blocksize_k_major, kv_seq_len)
	_verify_block("blocksizek_dq", "blocksize_k", blocksize_k, kv_seq_len)

	# Broadcast out scalar values
	m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_blocksize))
	l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_blocksize))
	# Preprocess contraction for bwd pass
	di = jnp.broadcast_to(di[..., None], (*di.shape, blocksize_k_major))

	grid = (
		batch_size,
		num_heads,
		q_seq_len // blocksizeq_major,
		kv_seq_len // blocksize_k_major,
	)

	def qo_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, q_seq_index, 0)

	qo_spec = pl.BlockSpec(qo_index_map, (1, 1, blocksizeq_major, head_dim))
	do_spec = qo_spec

	def kv_index_map(
		batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
	):
		if blocksize_c is not None:
			# If the kv block is skipped, prefetch the next valid kv block, i.e. the
			# 0th one to be used for the next blocksize_q rows.
			next_kv_index = lax.select(
				below_or_on_diag(
					q_seq_index + q_idx_ref[0],
					blocksizeq_major,
					kv_seq_index + k_idx_ref[0],
					blocksize_k_major,
					blocksize_c,
				),
				kv_seq_index,
				0,
			)
		else:
			next_kv_index = kv_seq_index
		return (batch_index, head_index, next_kv_index, 0)

	kv_spec = pl.BlockSpec(kv_index_map, (1, 1, blocksize_k_major, head_dim))
	assert kv_spec.block_shape is not None
	assert key.ndim == len(kv_spec.block_shape)
	assert value.ndim == len(kv_spec.block_shape)

	def lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
		return (batch_index, head_index, q_seq_index, 0)

	lm_spec = pl.BlockSpec(lm_index_map, (1, 1, blocksizeq_major, MIN_blocksize))
	assert lm_spec.block_shape is not None
	assert l.ndim == len(lm_spec.block_shape)
	assert m.ndim == len(lm_spec.block_shape)

	di_spec = pl.BlockSpec(qo_index_map, (1, 1, blocksizeq_major, MIN_blocksize))
	assert di_spec.block_shape is not None
	assert di.ndim == len(di_spec.block_shape)

	def ab_index_map(
		batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
	):
		return (batch_index, 0, kv_seq_index)

	if ab is not None:
		ab = ab[:, None].repeat(blocksizeq_major, axis=1)

	dab_spec = (
		pl.BlockSpec(ab_index_map, (1, blocksizeq_major, blocksize_k_major))
		if ab is not None
		else None
	)

	q_segment_ids_spec = kv_segment_ids_spec = None
	q_segment_ids = kv_segment_ids = None
	if segment_ids is not None:

		def q_segment_ids_index_map(
			batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref
		):
			del head_index
			return (batch_index, q_seq_index, 0)

		def kv_segment_ids_index_map(
			batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
		):
			del head_index
			if blocksize_c is not None:
				# If the kv block is skipped, prefetch the next valid kv block, i.e. the
				# 0th one to be used for the next blocksize_q rows.
				next_kv_index = lax.select(
					below_or_on_diag(
						q_seq_index + q_idx_ref[0],
						blocksizeq_major,
						kv_seq_index + k_idx_ref[0],
						blocksize_k_major,
						blocksize_c,
					),
					kv_seq_index,
					0,
				)
			else:
				next_kv_index = kv_seq_index
			return (batch_index, 0, next_kv_index)

		q_segment_ids_spec = pl.BlockSpec(
			q_segment_ids_index_map, (1, blocksizeq_major, NUM_LANES)
		)
		kv_segment_ids_spec = pl.BlockSpec(
			kv_segment_ids_index_map, (1, NUM_SUBLANES, blocksize_k_major)
		)

		q_segment_ids = jax.lax.broadcast_in_dim(
			segment_ids.query,
			(batch_size, q_seq_len, NUM_LANES),
			(
				0,
				1,
			),
		)
		kv_segment_ids = jax.lax.broadcast_in_dim(
			segment_ids.kv,
			(batch_size, NUM_SUBLANES, kv_seq_len),
			(
				0,
				2,
			),
		)

	in_specs = [
		qo_spec,
		kv_spec,
		kv_spec,
		dab_spec,
		q_segment_ids_spec,
		kv_segment_ids_spec,
		lm_spec,
		lm_spec,
		do_spec,
		di_spec,
	]

	out_shapes = [
		jax.ShapeDtypeStruct(query.shape, query.dtype),
		jax.ShapeDtypeStruct((blocksizeq_major, head_dim), jnp.float32),
		jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
	]
	dq_spec = pl.BlockSpec(qo_index_map, (1, 1, blocksizeq_major, head_dim))
	out_specs = [
		dq_spec,
		pl.BlockSpec(lambda *_: (0, 0), (blocksizeq_major, head_dim)),
		dab_spec,
	]

	kernel = functools.partial(
		_flash_attention_dq_kernel,
		softmax_scale=softmax_scale,
		blocksize_c=blocksize_c,
		mask_value=mask_value,
		blocksize_k=blocksize_k,
		kv_seq_len=kv_seq_len,
	)
	name_scope = (
		f"flash_mha_bwd_dq_{blocksizeq_major=}_{blocksize_k_major=}_{blocksize_k=}"
	)
	with jax.named_scope(name_scope):
		dq, _, ds = pl.pallas_call(
			kernel,
			out_shape=out_shapes,
			grid_spec=pltpu.PrefetchScalarGridSpec(
				num_scalar_prefetch=2,
				in_specs=in_specs,
				out_specs=out_specs,
				grid=grid,
			),
			debug=debug,
			compiler_params=dict(
				mosaic=dict(
					dimension_semantics=(
						"parallel",
						"parallel",
						"parallel",
						"arbitrary",
					)
				)
			),
			interpret=_INTERPRET,
		)(
			q_chunk_idx_start,
			k_chunk_idx_start,
			query,
			key,
			value,
			ab,
			q_segment_ids,
			kv_segment_ids,
			l,
			m,
			do,
			di,
		)

	return dq, ds


def _verify_block(blocksizename, dim_name, block, dim, should_divide=True):
	if block > dim:
		raise ValueError(
			f"{blocksizename}={block} should be smaller or equal to {dim_name}={dim}"
		)
	if should_divide and dim % block != 0:
		raise ValueError(f"{dim_name}={dim} should be divisible by {blocksizename}={block}")


def below_or_on_diag(
	r: int, r_blk_size: int, c: int, c_blk_size: int, blocksize_c: int
):
	"""Checks if the element at (r, c) is below or on the diagonal.

	Args:
		r: Row index.
		r_blk_size: Block size of the row.
		c: Column index.
		c_blk_size: Block size of the column.
		blocksize_c: Size of causal blocks.

	Returns:
		True if the element is below or on the diagonal, False otherwise.
	"""
	causal_blocksize_q = max(blocksize_c, r_blk_size)
	causal_blocksize_k = max(blocksize_c, c_blk_size)
	r = jax.lax.div(r, causal_blocksize_q // r_blk_size)
	c = jax.lax.div(c, causal_blocksize_k // c_blk_size)
	return ((r + 1) * causal_blocksize_q - 1) > (c * causal_blocksize_k)


pallas_ring_attention_tpu = ring_attention_tpu
__all__ = ["pallas_ring_attention_tpu"]
