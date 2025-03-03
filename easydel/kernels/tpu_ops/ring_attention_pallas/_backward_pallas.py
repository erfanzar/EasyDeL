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
			query = pl.load(q_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None)))
			lse = pl.load(l_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None)))
			m = pl.load(m_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None)))
			do = pl.load(do_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None)))
			di = pl.load(
				di_tile_ref, (0, 0, pl.ds(start_q, blocksize_q), slice(None))
			).astype(jnp.float32)

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
				1 / lse, blocksize_k // MIN_blocksize, axis=1
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
	lse,
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
	lse = jnp.broadcast_to(lse[..., None], (*lse.shape, MIN_blocksize))
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
	assert lse.ndim == len(lm_spec.block_shape)
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
			interpret=INTERPRET,
		)(
			q_chunk_idx_start,
			k_chunk_idx_start,
			query,
			key,
			value,
			ab,
			q_segment_ids,
			kv_segment_ids,
			lse,
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
		lse = l_tile_ref[0, 0, :, :]  # [blocksizeq_major, 128]
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
			1 / lse, blocksize_k // MIN_blocksize, axis=1
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
	lse,
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
	lse = jnp.broadcast_to(lse[..., None], (*lse.shape, MIN_blocksize))
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
	assert lse.ndim == len(lm_spec.block_shape)
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
			interpret=INTERPRET,
		)(
			q_chunk_idx_start,
			k_chunk_idx_start,
			query,
			key,
			value,
			ab,
			q_segment_ids,
			kv_segment_ids,
			lse,
			m,
			do,
			di,
		)

	return dq, ds


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
	(query, key, value, ab, segment_ids, o, lse, m) = residuals
	if not blocksizes.has_backward_blocks:
		raise ValueError(
			"Program is being differentiated, but not all backward blocks are specified"
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
		lse,
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
		lse,
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
	o, query, key, value, bias, segment_ids, cache_idx, lse, m = res
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
			residuals=(query, key, value, attn_bias_slice, segment_ids_slice, o, lse, m),
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
