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
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange
from jax.experimental import pallas as pl  # type: ignore[import]
from jax.experimental.pallas import tpu as pltpu  # type: ignore[import]

from ._utils import (
	DEFAULT_MASK_VALUE,
	INTERPRET,
	NUM_LANES,
	NUM_SUBLANES,
	TRANS_B_DIM_NUMBERS,
	BlockSizes,
	MIN_blocksize,
	SegmentIds,
	_verify_block,
	below_or_on_diag,
)


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

			def l_broadcast(la):
				return pltpu.repeat(la, head_dim_repeats, 1)

			if rem:
				if head_dim_repeats == 0:

					def l_broadcast(la):
						return la[:, :head_dim]
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
		lse = jax.ShapeDtypeStruct(
			(batch_size, num_heads, q_seq_len, MIN_blocksize), dtype=jnp.float32
		)
		m = jax.ShapeDtypeStruct(
			(batch_size, num_heads, q_seq_len, MIN_blocksize), dtype=jnp.float32
		)
		out_shape = (*out_shape, lse, m)

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
		interpret=INTERPRET,
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
		lse, m = (value[..., 0] for value in aux[-2:])
		return (o, lse, m)
	else:
		return o


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
	o, lse, m = _flash_attention(
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
	return o, lse, m


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
	lse = jnp.zeros((batch, num_heads, q_len)).astype(query.dtype)
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
		o, lse, m, key, value = carry
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
		o, lse, m = _flash_attention_fwd(
			query,
			key,
			value,
			carry=(o, lse, m),
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
		return (o, lse, m, key, value), None

	(o, lse, m, _, _), _ = lax.scan(
		scan_kv_block,
		init=(o, lse, m, key, value),
		xs=jnp.arange(0, axis_size),
	)

	output = rearrange(o.astype(value.dtype), "b h query d -> b query h d")
	return output, (o, query, key, value, bias, segment_ids, cache_idx, lse, m)
