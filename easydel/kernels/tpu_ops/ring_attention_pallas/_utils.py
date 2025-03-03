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


import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp

INTERPRET = False
MIN_blocksize = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))
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


@dataclass(frozen=True)
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
					f"{prefix}{suffix}={minor} should divide {prefix}_major{suffix}={major}"
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
