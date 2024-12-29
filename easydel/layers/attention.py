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
import math
import os
import time
import typing as tp
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache, partial

import einops
import fjformer
import flax.nnx as nn
import jax
import jax.experimental
import jax.extend
import jax.lib
import jax.tree_util
import numpy
from chex import Array
from fjformer import with_sharding_constraint
from flax.nnx.nn.dtypes import promote_dtype
from jax import lax, random
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
	BlockSizes as BlockSizesSplashAttn,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
	CausalMask,
	MultiHeadMask,
	SegmentIds,
	make_splash_mha,
)
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from easydel.etils.etils import (
	_AVAILABLE_ATTENTION_MECHANISMS,
	AVAILABLE_ATTENTION_MECHANISMS,
	EasyDeLBackends,
	EasyDeLPlatforms,
	get_logger,
)
from easydel.etils.partition_module import PartitionAxis
from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.kernels.flash_attention_2 import create_flash_attention
from easydel.kernels.ring_attention import ring_attention
from easydel.layers._blockwise_attention import blockwise_attn
from easydel.layers.caching import TransformerCacheView
from easydel.utils.quantizers import EasyQuantizer

try:
	from flash_attn_jax import flash_mha as cuda_flash_attn2_mha  # noqa #type:ignore
	from flash_attn_jax.flash_hlo import (  # noqa #type:ignore
		dtypes,
		ShapedArray,
		_flash_mha_fwd_hlo_p,
		_flash_mha_bwd_hlo_p,
	)
	from flash_attn_jax.flash import _flash_mha_fwd_p, _flash_mha_bwd_p  # noqa #type:ignore

	def _flash_mha_fwd_abstract(
		q,
		k,
		v,
		softmax_scale=None,
		is_causal=None,
		window_size=None,
	):
		q_dtype = dtypes.canonicalize_dtype(q.dtype)
		k_dtype = dtypes.canonicalize_dtype(k.dtype)
		v_dtype = dtypes.canonicalize_dtype(v.dtype)
		[n, s, h, d] = q.shape
		assert q_dtype == k_dtype and q_dtype == v_dtype
		assert q_dtype in [jnp.bfloat16, jnp.float16]
		return (
			ShapedArray(q.shape, q_dtype, sharding=getattr(q, "sharding", None)),
			ShapedArray([n, h, s], jnp.float32),
		)

	def _flash_mha_bwd_abstract(
		dout, q, k, v, out, lse, softmax_scale=None, is_causal=None, window_size=None
	):
		dout_dtype = dtypes.canonicalize_dtype(dout.dtype)
		q_dtype = dtypes.canonicalize_dtype(q.dtype)
		k_dtype = dtypes.canonicalize_dtype(k.dtype)
		v_dtype = dtypes.canonicalize_dtype(v.dtype)
		out_dtype = dtypes.canonicalize_dtype(out.dtype)
		[n, lq, hq, d] = q.shape
		assert len(set([dout_dtype, q_dtype, k_dtype, v_dtype, out_dtype])) == 1
		assert q_dtype in [jnp.bfloat16, jnp.float16]
		return (
			ShapedArray(q.shape, q_dtype, sharding=getattr(q, "sharding", None)),
			ShapedArray(k.shape, k_dtype, sharding=getattr(k, "sharding", None)),
			ShapedArray(v.shape, v_dtype, sharding=getattr(v, "sharding", None)),
		)

	_flash_mha_bwd_hlo_p.def_abstract_eval(_flash_mha_bwd_abstract)
	_flash_mha_fwd_hlo_p.def_abstract_eval(_flash_mha_fwd_abstract)
	_flash_mha_fwd_p.def_abstract_eval(_flash_mha_fwd_abstract)
	_flash_mha_bwd_p.def_abstract_eval(_flash_mha_bwd_abstract)

except:  # noqa
	cuda_flash_attn2_mha = None
logger = get_logger(__name__)

DEFAULT_K_BLOCK = 128
DEFAULT_Q_BLOCK = 64

PRINT_COMMON = False


@lru_cache
def get_cached_flash_attention(
	backend,
	platform,
	blocksize_q,
	blocksize_k,
	softmax_scale,
):
	return create_flash_attention(
		backend=backend,
		platform=platform,
		blocksize_q=blocksize_q,
		blocksize_k=blocksize_k,
		softmax_scale=softmax_scale,
	)


def create_target_only_spec(original_spec, target_dict):
	"""
	Creates a new partition spec that only includes Single sharding.
	"""
	if original_spec is None:
		return None

	new_spec = []
	for _ in original_spec:
		new_spec.append(None)
	for k, v in target_dict.items():
		new_spec[k] = v
	return PartitionSpec(*new_spec)


def _get_jax_dtype_from_string(dtype_string):
	dtype_mapping = {
		"<class 'jax.numpy.float32'>": jnp.float32,
		"<class 'jax.numpy.float64'>": jnp.float64,
		"<class 'jax.numpy.int32'>": jnp.int32,
		"<class 'jax.numpy.int64'>": jnp.int64,
		"<class 'jax.numpy.bool_'>": jnp.bool_,
		"<class 'jax.numpy.complex64'>": jnp.complex64,
		"<class 'jax.numpy.complex128'>": jnp.complex128,
	}
	return dtype_mapping.get(dtype_string, None)


@dataclass
class AttentionOutput:
	attention_weights: tp.Optional[Array] = None
	attention_outputs: tp.Optional[Array] = None


class AttentionMechanisms(str, Enum):
	FLASH_ATTN2 = "flash_attn2"
	RING = "ring"
	VANILLA = "vanilla"
	SPLASH = "splash"
	CUDNN = "cudnn"
	BLOCKWISE = "blockwise"
	SDPA = "sdpa"
	CUDA_FLASH_ATTN2 = "cuda_flash_attn2"


def combine_flash_masks(causal_mask, segment_ids):
	causal_mask = causal_mask.astype(jnp.bool_)
	if causal_mask.ndim == 2:
		query_sequence_length, key_sequence_length = causal_mask.shape
		causal_mask = causal_mask.reshape(1, 1, query_sequence_length, key_sequence_length)
	elif causal_mask.ndim == 4:
		*_, query_sequence_length, key_sequence_length = causal_mask.shape
	else:
		raise ValueError("unexpected shape for `causal_mask`")
	if segment_ids.ndim == 2:
		b, seq_query_sequence_length = segment_ids.shape
		seq_key_sequence_length = seq_query_sequence_length
	elif segment_ids.ndim == 4:
		b, _, _, seq_key_sequence_length = segment_ids.shape
		seq_query_sequence_length = seq_key_sequence_length
		segment_ids = segment_ids[:, 0, -1]  # taking final mask
	else:
		raise ValueError("unexpected shape for `segment_ids`")

	assert (
		seq_query_sequence_length == query_sequence_length
	), "`segment_ids` and `causal_mask` don't have same query axis length"
	assert (
		seq_key_sequence_length == key_sequence_length
	), "`segment_ids` and `causal_mask` don't have same key/value axis length"
	assert (
		segment_ids.ndim == 2
	), f"`segment_ids` don't have excepted shape {segment_ids.shape}"
	segment_ids = jnp.expand_dims(
		~jnp.equal(
			jnp.expand_dims(segment_ids, axis=2), jnp.expand_dims(segment_ids, axis=1)
		).astype(jax.numpy.bool_),
		1,
	)
	return jnp.logical_or(~causal_mask, segment_ids)


DEFAULT_ATTENTION_MECHANISM = (
	"sdpa" if jax.extend.backend.get_backend().platform == "gpu" else "vanilla"
)


def set_attrs_smartly_with_prp(
	self,
	attr_name: str,
	default: tp.Any,
	new_attr: tp.Any,
	prp: EasyDeLBaseConfig = None,
	pickup_name=None,
):
	if not hasattr(self, attr_name) or getattr(self, attr_name, ...) == Ellipsis:
		setattr(
			self,
			attr_name,
			(
				default
				if prp is None
				else getattr(prp, (attr_name if pickup_name is None else pickup_name))
			),
		)
	if not new_attr == Ellipsis:
		setattr(self, attr_name, new_attr)


class FlexibleAttentionModule(nn.Module):
	"""
	Manages different attention mechanisms for efficient computation in EasyDeL models.

	This class serves as a central hub for handling various attention mechanisms, including
	optimized implementations like FlashAttention, SplashAttention, RingAttention, and more traditional
	approaches like vanilla (dot-product) attention. It provides a unified interface to
	select and execute the appropriate attention mechanism based on the model's configuration and
	hardware platform.

	Key Features:

	* **Attention Mechanism Selection:** Supports a wide range of attention mechanisms,
	  allowing users to choose the most suitable option based on performance and hardware constraints.
	* **Sharding and Partitioning:** Integrates with JAX's sharding capabilities, enabling efficient
	  distribution of computations and data across multiple devices.
	* **Block-wise Computation:** Implements block-wise attention computations for optimized memory
	  usage and speed, particularly beneficial for large models.
	* **Performance Optimization:** Includes support for highly optimized implementations like
	  FlashAttention, SplashAttention, and RingAttention for TPU and GPU acceleration.
	* **Flexibility and Customization:** Offers fine-grained control over attention parameters,
	  sharding specifications, and block sizes, providing flexibility for different use cases.
	* **Testing and Evaluation:** Includes a `run_attention_benchmarks` method to systematically evaluate
	  different attention mechanisms and help users identify the best-performing option.

	Example Usage:

	>>> # Initialize an FlexibleAttentionModule instance
	>>> attention_module = FlexibleAttentionModule(
	...    mesh=mesh,
	...    attn_mechanism="ring",  # Select the desired attention mechanism
	...    sm_scale=1.0 / math.sqrt(head_dim),
	...    num_attention_heads=num_heads,
	...    head_dims=head_dim,
	...    # ... other configuration parameters ...
	>>> )

	>>> # Compute attention outputs
	>>> attention_output = attention_module(
	...    query_states=query_states,
	...    key_states=key_states,
	...    value_states=value_states,
	...     # ... other attention inputs ...
	>>> )

	>>> # Access attention outputs
	>>> attention_outputs = attention_output.attention_outputs
	>>> attention_weights = attention_output.attention_weights

	The FlexibleAttentionModule class is a crucial component within EasyDeL, responsible for managing and optimizing attention
	computations. It provides a user-friendly way to select and execute different attention mechanisms,
	leveraging JAX's sharding capabilities and offering performance enhancements through specialized implementations
	 like FlashAttention and SplashAttention. Its ability to handle block-wise computations and customization options
	  makes it adaptable to a variety of model architectures and hardware configurations.
	"""

	def __init__(
		self,
		mesh: Mesh,
		sm_scale: float,
		num_kv_heads: int,
		num_q_heads: int,
		head_dims: int,
		attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = DEFAULT_ATTENTION_MECHANISM,
		blocksize_k: int = ...,
		blocksize_q: int = ...,
		blocksize_b: int = ...,
		partition_axis: PartitionAxis = ...,
		scan_ring_attention: bool = ...,
		scan_attention_layers: bool = ...,
		attention_dropout: float = 0.0,
		dtype: jnp.dtype = ...,
		precision: lax.Precision = ...,
		force_float32_tpu: bool = ...,
		shard_attention_computation: bool = ...,
		use_sharding_constraint: tp.Optional[bool] = ...,
		axis_name: str = ...,
		platform: EasyDeLPlatforms = ...,
		backend: tp.Optional[EasyDeLBackends] = ...,
		backward_pass_impl: tp.Literal["triton", "xla"] = "triton",
		base_config: tp.Optional[EasyDeLBaseConfig] = None,
		_do_check: bool = True,
	):
		self.blocksize_k: int = ...
		self.blocksize_q: int = ...
		self.blocksize_b: int = ...
		self.partition_axis: PartitionAxis = ...
		self.scan_ring_attention: bool = ...
		self.precision: lax.Precision = ...
		self.force_float32_tpu: bool = ...
		self.shard_attention_computation: bool = ...
		self.use_sharding_constraint: tp.Optional[bool] = ...
		self.axis_name: str = ...
		self.backend: str = ...
		self.platform: str = ...

		# fmt:off
		set_attrs_smartly_with_prp(self, "use_sharding_constraint", False, use_sharding_constraint, base_config)
		set_attrs_smartly_with_prp(self, "blocksize_q", DEFAULT_Q_BLOCK, blocksize_q, base_config)
		set_attrs_smartly_with_prp(self, "blocksize_k", DEFAULT_K_BLOCK, blocksize_k, base_config)
		set_attrs_smartly_with_prp(self, "blocksize_b", 1, blocksize_b, base_config)
		set_attrs_smartly_with_prp(self, "dtype", jnp.float32, dtype, base_config, "attn_dtype")
		set_attrs_smartly_with_prp(self, "shard_attention_computation", True, shard_attention_computation, base_config)
		set_attrs_smartly_with_prp(self, "scan_ring_attention", True, scan_ring_attention, base_config)
		set_attrs_smartly_with_prp(self, "partition_axis", PartitionAxis(), partition_axis, base_config)
		set_attrs_smartly_with_prp(self, "precision", lax.Precision("fastest"), precision)  # DON'T READ FROM CONFIG
		set_attrs_smartly_with_prp(self, "force_float32_tpu", True, force_float32_tpu)  # DON'T READ FROM CONFIG
		set_attrs_smartly_with_prp(self, "axis_name", "sp", axis_name, base_config, "attention_axis_name")  # DON'T READ FROM CONFIG
		set_attrs_smartly_with_prp(self, "backend", jax.default_backend(), backend, base_config, "backend") 
		set_attrs_smartly_with_prp(self, "platform", ..., platform, base_config, "platform") 
		# fmt:on

		self.mesh = mesh
		self.attn_mechanism = attn_mechanism
		self.sm_scale = sm_scale
		self.head_dims = head_dims
		self.scan_attention_layers = scan_attention_layers
		self.attention_dropout = attention_dropout
		self.backward_pass_impl = backward_pass_impl
		self._do_check = _do_check
		self.num_kv_heads = num_kv_heads or num_q_heads
		self.num_q_heads = num_q_heads
		assert num_q_heads % num_kv_heads == 0

		if attn_mechanism == "splash" and jax.default_backend() != "tpu":
			raise OSError("splash attention is only supported on TPU.")
		if attn_mechanism == "cudnn" and jax.default_backend() != "gpu":
			raise OSError("flash attention is only supported on GPU.")
		if isinstance(self.dtype, str):
			self.dtype = _get_jax_dtype_from_string(self.dtype)
			assert self.dtype is not None, "Please consider passing attn_dtype to config."

	def get_block_size_splash_attn(self, q_seq, k_seq):
		return BlockSizesSplashAttn(
			block_q=min(self.blocksize_q, q_seq),
			block_kv_compute=min(self.blocksize_k, k_seq),
			block_kv=min(self.blocksize_k, k_seq),
			block_q_dkv=min(self.blocksize_q, q_seq),
			block_kv_dkv=min(self.blocksize_k, k_seq),
			block_kv_dkv_compute=min(self.blocksize_k, k_seq),
			block_q_dq=min(self.blocksize_q, q_seq),
			block_kv_dq=min(self.blocksize_k, k_seq),
		)

	def get_bshd_partition_specs(
		self,
		query_sequence_length,
		bias_dim_eql=False,
	) -> tp.Tuple[
		PartitionSpec,
		PartitionSpec,
		PartitionSpec,
		PartitionSpec,
		PartitionSpec,
		bool,
	]:
		in_generating_processerating = query_sequence_length == 1
		if not in_generating_processerating:
			query_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.query_sequence_axis,
				self.partition_axis.head_axis,
				self.partition_axis.attention_dim_axis,
			)
			key_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.key_sequence_axis,
				self.partition_axis.head_axis,
				self.partition_axis.attention_dim_axis,
			)
			value_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.key_sequence_axis,
				self.partition_axis.head_axis,
				self.partition_axis.attention_dim_axis,
			)
			bias_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.bias_head_sequence_axis
				if not bias_dim_eql
				else self.partition_axis.head_axis,
				self.partition_axis.query_sequence_axis,
				self.partition_axis.bias_key_sequence_axis,
			)
			attention_partition_spec = query_partition_spec
		else:
			query_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.generation_query_sequence_axis,
				self.partition_axis.generation_head_axis,
				self.partition_axis.generation_attention_dim_axis,
			)
			key_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.generation_key_sequence_axis,
				self.partition_axis.generation_head_axis,
				self.partition_axis.generation_attention_dim_axis,
			)
			value_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.generation_key_sequence_axis,
				self.partition_axis.generation_head_axis,
				self.partition_axis.generation_attention_dim_axis,
			)
			bias_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.bias_head_sequence_axis
				if not bias_dim_eql
				else self.partition_axis.head_axis,
				self.partition_axis.generation_query_sequence_axis,
				self.partition_axis.bias_key_sequence_axis,
			)
			attention_partition_spec = query_partition_spec
		return (
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			attention_partition_spec,
			in_generating_processerating,
		)

	def get_bhsd_partition_specs(
		self,
		query_sequence_length,
		bias_dim_eql=False,
	) -> tp.Tuple[
		PartitionSpec, PartitionSpec, PartitionSpec, PartitionSpec, PartitionSpec, bool
	]:
		in_generating_processerating = query_sequence_length == 1
		if not in_generating_processerating:
			query_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.head_axis,
				self.partition_axis.query_sequence_axis,
				self.partition_axis.attention_dim_axis,
			)
			key_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.head_axis,
				self.partition_axis.key_sequence_axis,
				self.partition_axis.attention_dim_axis,
			)
			value_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.head_axis,
				self.partition_axis.key_sequence_axis,
				self.partition_axis.attention_dim_axis,
			)
			bias_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.bias_head_sequence_axis,
				self.partition_axis.query_sequence_axis,
				self.partition_axis.bias_key_sequence_axis,
			)
			attention_partition_spec = query_partition_spec
		else:
			query_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.generation_head_axis,
				self.partition_axis.generation_query_sequence_axis,
				self.partition_axis.generation_attention_dim_axis,
			)
			key_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.generation_head_axis,
				self.partition_axis.generation_key_sequence_axis,
				self.partition_axis.generation_attention_dim_axis,
			)
			value_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.generation_head_axis,
				self.partition_axis.generation_key_sequence_axis,
				self.partition_axis.generation_attention_dim_axis,
			)
			bias_partition_spec = PartitionSpec(
				self.partition_axis.batch_axis,
				self.partition_axis.bias_head_sequence_axis
				if not bias_dim_eql
				else self.partition_axis.head_axis,
				self.partition_axis.generation_query_sequence_axis,
				self.partition_axis.bias_key_sequence_axis,
			)
			attention_partition_spec = query_partition_spec
		return (
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			attention_partition_spec,
			in_generating_processerating,
		)

	def _check_states(
		self,
		query_states: Array,
		key_states: Array,
		value_states: Array,
		query_sequence_length: int,
		key_value_sequence_length: int,
	):
		batch_size = query_states.shape[0]
		assert (
			batch_size == key_states.shape[0] == value_states.shape[0]
		), "Batch Size for q,k,v wont match"
		k_v_req_shape = (
			batch_size,
			key_value_sequence_length,
			self.num_kv_heads,
			self.head_dims,
		)
		q_shape = (
			batch_size,
			query_sequence_length,
			self.num_q_heads,
			self.head_dims,
		)

		assertion_mkv_err = f"""
        query_states, key_states, value_states and bias shapes must be like
        query_states Shape : [batch_size, q_seq_len , {self.num_q_heads=}, {self.head_dims=}]
        key_states   Shape : [batch_size, kv_seq_len, {self.num_kv_heads=}, {self.head_dims=}]
        value_states Shape : [batch_size, kv_seq_len, {self.num_kv_heads=}, {self.head_dims=}]
        bias         Shape : [batch_size, {self.num_q_heads=}, q_seq_len , kv_seq_len]
            """

		assert query_states.shape == q_shape, assertion_mkv_err + (
			f"\nMiss Match {query_states.shape} and " f"required Shape {q_shape}"
		)
		assert key_states.shape == k_v_req_shape, assertion_mkv_err + (
			f"\nMiss Match {key_states.shape} and " f"required Shape {k_v_req_shape}"
		)
		assert value_states.shape == k_v_req_shape, assertion_mkv_err + (
			f"\nMiss Match {value_states.shape} and " f"required Shape {k_v_req_shape}"
		)

	@jax.named_scope("easydel-flexible-attention")
	def __call__(
		self,
		query_states: Array,
		key_states: Array,
		value_states: Array,
		query_sequence_length: tp.Optional[int] = None,
		key_value_sequence_length: tp.Optional[int] = None,
		bias: tp.Optional[Array] = None,
		attention_mask: tp.Optional[Array] = None,
		segment_ids: tp.Optional[Array] = None,
		causal: bool = True,
		deterministic: bool = True,
		dropout_rng: tp.Optional[random.PRNGKey] = None,
		uses_cache: bool = False,
		causal_mask: tp.Optional[Array] = None,
	):
		global PRINT_COMMON
		if query_sequence_length is None:
			query_sequence_length = query_states.shape[1]
		if key_value_sequence_length is None:
			key_value_sequence_length = key_states.shape[1]
		with self.mesh:
			# if self._do_check:
			# 	self._check_states(
			# 		query_states=query_states,
			# 		key_states=key_states,
			# 		value_states=value_states,
			# 		query_sequence_length=query_sequence_length,
			# 		key_value_sequence_length=key_value_sequence_length,
			# 	)
			match self.attn_mechanism:
				case AttentionMechanisms.FLASH_ATTN2:
					return self.flash_attn2(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						bias=bias,
					)
				case AttentionMechanisms.SDPA:
					return self.sdpa(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						bias=bias,
						causal=causal,
					)
				case AttentionMechanisms.VANILLA:
					return self.vanilla_attention(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						bias=bias,
						dropout_rng=dropout_rng,
						deterministic=deterministic,
						query_sequence_length=query_sequence_length,
						key_value_sequence_length=key_value_sequence_length,
					)
				case AttentionMechanisms.RING:
					return self.ring_attention(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						bias=bias,
						dropout_rng=dropout_rng,
						deterministic=deterministic,
						segment_ids=segment_ids,
						attention_mask=attention_mask,
						query_sequence_length=query_sequence_length,
						key_value_sequence_length=key_value_sequence_length,
					)
				case AttentionMechanisms.SPLASH:
					if PRINT_COMMON:
						if segment_ids is not None:
							warnings.warn(
								"Splash attention don't support `segment_ids` this argument will be ignored",
								UserWarning,
								stacklevel=1,
							)
						if self.attention_dropout != 0.0:
							warnings.warn(
								"Splash attention don't support `attention_dropout` this argument will be ignored",
								UserWarning,
								stacklevel=1,
							)
						if bias is not None:
							warnings.warn(
								"Splash attention don't support `bias` this argument will be ignored",
								UserWarning,
								stacklevel=1,
							)
						PRINT_COMMON = False
					return self.splash_attention(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						query_sequence_length=query_sequence_length,
						key_value_sequence_length=key_value_sequence_length,
						attention_mask=attention_mask,
					)
				case AttentionMechanisms.BLOCKWISE:
					if segment_ids is not None and PRINT_COMMON:
						warnings.warn(
							"BlockWise Attention don't support `segment_ids` this argument will be ignored",
							UserWarning,
							stacklevel=1,
						)
						PRINT_COMMON = False
					return self.blockwise_attention(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						bias=bias,
						deterministic=deterministic,
						dropout_rng=dropout_rng,
						query_sequence_length=query_sequence_length,
						key_value_sequence_length=key_value_sequence_length,
					)
				case AttentionMechanisms.CUDNN:
					return self.cuddn_flash_attention(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						bias=bias,
						causal=causal,
						deterministic=deterministic,
						query_sequence_length=query_sequence_length,
						key_value_sequence_length=key_value_sequence_length,
					)
				case AttentionMechanisms.CUDA_FLASH_ATTN2:
					if bias is not None and PRINT_COMMON:
						warnings.warn(
							"`CUDA_FLASH_ATTN2` doesn't support bias and attention mask and "
							f"causal will only be used which is passed as {causal}, please check outputs to make sure this is what you want.",
							stacklevel=1,
						)
						PRINT_COMMON = False
					return self.cuda_flash_attn2(
						query_states=query_states,
						key_states=key_states,
						value_states=value_states,
						causal=causal,
						attention_mask=attention_mask,
					)

		raise ValueError(f"Unknown Attention mechanism of {self.attn_mechanism}")

	def sdpa(
		self,
		*,
		query_states: Array,
		key_states: Array,
		value_states: Array,
		bias: tp.Optional[Array] = None,
		causal: bool = False,
	):
		(
			query_partitionspec,
			key_partitionspec,
			value_partitionspec,
			bias_partitionspec,
			attention_partitionspec,
			in_generating_process,
		) = self.get_bshd_partition_specs(query_states.shape[1])
		with self.mesh:
			func = functools.partial(
				jax.nn.dot_product_attention,
				implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
				scale=self.sm_scale,
				is_causal=(causal if not in_generating_process else False)
				if jax.default_backend() == "gpu"
				else (causal if bias is None else False),
			)
			dtype = self.dtype
			if jax.default_backend() == "gpu" and dtype == jnp.float32:
				dtype = jnp.float16
			attention_output = shard_map(
				func,
				mesh=self.mesh,
				in_specs=(
					create_target_only_spec(
						query_partitionspec,
						{0: query_partitionspec[0], 2: query_partitionspec[2]},
					),
					create_target_only_spec(
						key_partitionspec,
						{0: key_partitionspec[0], 2: key_partitionspec[2]},
					),
					create_target_only_spec(
						value_partitionspec,
						{0: value_partitionspec[0], 2: value_partitionspec[2]},
					),
					(
						create_target_only_spec(
							bias_partitionspec,
							{0: bias_partitionspec[0], 1: bias_partitionspec[1]},
						)
						if bias_partitionspec is not None
						else None
					),
				),
				out_specs=create_target_only_spec(
					attention_partitionspec,
					{0: attention_partitionspec[0], 2: attention_partitionspec[2]},
				),
				check_rep=False,
			)(
				query_states.astype(dtype),
				key_states.astype(dtype),
				value_states.astype(dtype),
				bias.astype(dtype) if bias is not None else None,
			)
			return AttentionOutput(
				attention_weights=None,
				attention_outputs=with_sharding_constraint(
					attention_output, attention_partitionspec
				),
			)

	def flash_attn2(
		self,
		*,  # it's Kwarg Only
		query_states: Array,
		key_states: Array,
		value_states: Array,
		bias: tp.Optional[Array] = None,
	):
		(
			query_partitionspec,
			key_partitionspec,
			value_partitionspec,
			bias_partitionspec,
			attention_partitionspec,
			in_generating_process,
		) = self.get_bshd_partition_specs(query_states.shape[1])

		blocksize_q = self.blocksize_q
		if in_generating_process:
			blocksize_q = int(os.environ.get("GENERATION_BLOCKSIZE_Q", self.blocksize_q))
		if bias is not None:
			assert bias.ndim == 4
		with self.mesh:
			# Helper function to get axis size
			def get_axis_size(axis_name):
				if isinstance(axis_name, tuple):
					return numpy.prod([self.mesh.shape[name] for name in axis_name])
				return self.mesh.shape[axis_name]

			attention = get_cached_flash_attention(
				backend=self.backend,
				platform=self.platform,
				blocksize_q=blocksize_q,
				blocksize_k=self.blocksize_k,
				softmax_scale=self.sm_scale,
			)

			with self.mesh:
				attention_outputs = shard_map(
					attention,
					mesh=self.mesh,
					in_specs=(
						create_target_only_spec(
							query_partitionspec,
							{0: query_partitionspec[0], 2: query_partitionspec[2]},
						),
						create_target_only_spec(
							key_partitionspec,
							{0: key_partitionspec[0], 2: key_partitionspec[2]},
						),
						create_target_only_spec(
							value_partitionspec,
							{0: value_partitionspec[0], 2: value_partitionspec[2]},
						),
						(
							create_target_only_spec(
								bias_partitionspec,
								{0: bias_partitionspec[0], 1: bias_partitionspec[1]},
							)
							if bias_partitionspec is not None
							else None
						),
					),
					out_specs=create_target_only_spec(
						attention_partitionspec,
						{0: attention_partitionspec[0], 2: attention_partitionspec[2]},
					),
					check_rep=False,
				)(
					query_states.astype(self.dtype),
					key_states.astype(self.dtype),
					value_states.astype(self.dtype),
					bias.astype(self.dtype) if bias is not None else None,
				)
			return AttentionOutput(
				attention_weights=None,
				attention_outputs=attention_outputs,
			)

	def cuda_flash_attn2(
		self,
		*,  # it's Kwarg Only
		query_states: Array,
		key_states: Array,
		value_states: Array,
		causal: bool = True,
		attention_mask=None,
	):
		if cuda_flash_attn2_mha is not None:
			key_states, value_states = self.repeat_kv_heads(
				key_states,
				value_states,
				self.num_q_heads // self.num_kv_heads,
			)
			(
				query_partitionspec,
				key_partitionspec,
				value_partitionspec,
				_,
				attention_partitionspec,
				in_generating_process,
			) = self.get_bshd_partition_specs(query_states.shape[1], True)

			output = cuda_flash_attn2_mha(
				q=with_sharding_constraint(
					query_states.astype(self.dtype), query_partitionspec
				),
				k=with_sharding_constraint(key_states.astype(self.dtype), key_partitionspec),
				v=with_sharding_constraint(
					value_states.astype(self.dtype), value_partitionspec
				),
				softmax_scale=self.sm_scale,
				is_causal=False if in_generating_process else causal,
				window_size=(-1, -1),
			)
			return AttentionOutput(
				attention_weights=None,
				attention_outputs=with_sharding_constraint(
					output,
					attention_partitionspec,
				),
			)
		else:
			raise ModuleNotFoundError("please install flash_attn_jax==0.2.2")

	def ring_attention(
		self,
		*,  # it's Kwarg Only
		query_states: Array,
		key_states: Array,
		value_states: Array,
		query_sequence_length: int,
		key_value_sequence_length: int,
		bias: tp.Optional[Array] = None,
		attention_mask: tp.Optional[Array] = None,
		deterministic: bool = False,
		dropout_rng: tp.Optional[random.PRNGKey] = None,
		segment_ids: tp.Optional[Array] = None,
	):
		key_states, value_states = self.repeat_kv_heads(
			key_states,
			value_states,
			self.num_q_heads // self.num_kv_heads,
		)
		(
			query_partitionspec,
			key_partitionspec,
			value_partitionspec,
			bias_partitionspec,
			attention_partitionspec,
			gen,
		) = self.get_bshd_partition_specs(query_states.shape[1], True)
		attn_output = shard_map(
			partial(
				ring_attention,
				axis_name=self.axis_name,
				float32_logits=False
				if jax.extend.backend.get_backend().platform == "gpu"
				else True,
				platform=self.platform,
				backend=self.backend,
				autocheck=True,
				blocksize_c=None,
				blocksize_k=self.blocksize_k,
				blocksize_q=self.blocksize_q,
				dtype=self.dtype,
				softmax_scale=self.sm_scale,
				deterministic=deterministic,
				dropout_rng=dropout_rng,
			),
			in_specs=(
				query_partitionspec,
				key_partitionspec,
				value_partitionspec,
				bias_partitionspec,
			),
			out_specs=attention_partitionspec,
			mesh=self.mesh,
			check_rep=False,
		)(
			query_states.astype(self.dtype),
			key_states.astype(self.dtype),
			value_states.astype(self.dtype),
			bias.astype(self.dtype),
		)

		return AttentionOutput(attention_weights=None, attention_outputs=attn_output)

	def vanilla_attention(
		self,
		*,  # it's Kwarg Only
		query_states: Array,
		key_states: Array,
		value_states: Array,
		bias: tp.Optional[Array] = None,
		deterministic: bool = False,
		dropout_rng: tp.Optional[random.PRNGKey] = None,
		query_sequence_length: int,
		key_value_sequence_length: int,
	) -> AttentionOutput:
		with self.mesh:
			(
				query_partitionspec,
				key_partitionspec,
				value_partitionspec,
				bias_partitionspec,
				attention_partitionspec,
				_,
			) = self.get_bshd_partition_specs(query_sequence_length)
		b, qs, qh, d = query_states.shape
		b, ks, kh, d = key_states.shape

		*_, vd = value_states.shape
		with self.mesh:
			query_states = fjformer.with_sharding_constraint(
				query_states, query_partitionspec
			)
			key_states = fjformer.with_sharding_constraint(key_states, key_partitionspec)
			value_states = fjformer.with_sharding_constraint(
				value_states, value_partitionspec
			)
			query_states = jnp.reshape(
				query_states,
				(b, qs, self.num_kv_heads, qh // self.num_kv_heads, d),
			)
			query_states, key_states, value_states = promote_dtype(
				(query_states, key_states, value_states),
				dtype=self.dtype,
			)

			query_states = query_states * self.sm_scale
			attention_weight = jnp.einsum(
				"bskhd,bmkd->bkhsm",
				query_states,
				key_states,
				precision=self.precision,
			)

		if bias is not None:
			if bias.shape[1] == self.num_q_heads:
				bias = bias.reshape(
					b, self.num_kv_heads, self.num_q_heads // self.num_kv_heads, qs, ks
				)
			elif bias.shape[1] == self.num_kv_heads:
				bias = bias.reshape(b, self.num_kv_heads, 1, qs, ks)
			elif bias.shape[1] == 1:
				bias = bias.reshape(b, 1, 1, qs, ks)
			else:
				raise NotImplementedError("bias heads wont match!")
			attention_weight = jnp.add(attention_weight, bias.astype(attention_weight))
		attention_weight = jax.nn.softmax(attention_weight).astype(self.dtype)

		if not deterministic and self.attention_dropout > 0.0:
			keep_prob = 1.0 - self.attention_dropout
			dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
			keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore

			multiplier = keep.astype(self.dtype) / jnp.asarray(keep_prob, dtype=self.dtype)
			attention_weight = attention_weight * multiplier

		attention = jnp.einsum(
			"bkhsm,bmkd->bskhd",
			attention_weight,
			value_states,
			precision=self.precision,
		).reshape(b, qs, qh, vd)
		attention = fjformer.with_sharding_constraint(attention, attention_partitionspec)
		return AttentionOutput(
			attention_weights=attention_weight, attention_outputs=attention
		)

	def blockwise_attention(
		self,
		*,  # it's Kwarg Only
		query_states: Array,
		key_states: Array,
		value_states: Array,
		bias: tp.Optional[Array] = None,
		deterministic: bool = False,
		dropout_rng: tp.Optional[random.PRNGKey] = None,
		query_sequence_length: int,
		key_value_sequence_length: int,
	) -> AttentionOutput:
		key_states, value_states = self.repeat_kv_heads(
			key_states,
			value_states,
			self.num_q_heads // self.num_kv_heads,
		)
		(
			query_partitionspec,
			key_partitionspec,
			value_partitionspec,
			bias_partitionspec,
			attention_partitionspec,
			_,
		) = self.get_bshd_partition_specs(query_sequence_length)

		with self.mesh:
			query_states = with_sharding_constraint(query_states, query_partitionspec)
			key_states = with_sharding_constraint(key_states, key_partitionspec)
			value_states = with_sharding_constraint(value_states, value_partitionspec)
			bias = with_sharding_constraint(bias, bias_partitionspec)
			o = blockwise_attn(
				query=query_states,
				key=key_states,
				value=value_states,
				bias=bias,
				deterministic=deterministic,
				dtype=self.dtype,
				dropout_rng=dropout_rng,
				precision=self.precision,
				attn_pdrop=self.attention_dropout,
				key_chunk_size=min(self.blocksize_k, key_value_sequence_length),
				query_chunk_size=min(self.blocksize_q, query_sequence_length),
				prevent_cse=not self.scan_attention_layers,
				causal=True,
				float32_logits=True,
			)

			o = with_sharding_constraint(o, attention_partitionspec)
			return AttentionOutput(attention_weights=None, attention_outputs=o)

	def splash_attention(
		self,
		query_states: Array,
		key_states: Array,
		value_states: Array,
		query_sequence_length: int,
		key_value_sequence_length: int,
		attention_mask: Array,
	) -> AttentionOutput:
		key_states, value_states = self.repeat_kv_heads(
			key_states,
			value_states,
			self.num_q_heads // self.num_kv_heads,
		)
		(
			query_partitionspec,
			key_partitionspec,
			value_partitionspec,
			_,
			_,
			_,
		) = self.get_bhsd_partition_specs(query_sequence_length)

		query_states, key_states, value_states = map(
			lambda s: s.astype(jnp.float32).transpose(0, 2, 1, 3),
			(query_states, key_states, value_states),
		)
		if attention_mask is not None:
			if attention_mask.ndim == 4:
				attention_mask = attention_mask[:, 0, -1]
			attention_mask = SegmentIds(attention_mask, attention_mask)
		else:
			warnings.warn(
				"`attention_mask` is not passed to SplashAttention. (except miss computation problem)",
				stacklevel=1,
			)

		@partial(
			shard_map,
			in_specs=(
				query_partitionspec,
				key_partitionspec,
				value_partitionspec,
				PartitionSpec(query_partitionspec[0], query_partitionspec[2]),
			),
			out_specs=query_partitionspec,
			mesh=self.mesh,
			check_rep=False,
		)
		def splash_attention_call(query, key, value, mask):
			block_size = self.get_block_size_splash_attn(
				query_sequence_length,
				key_value_sequence_length,
			)
			masks = CausalMask(shape=(query.shape[2], query.shape[2]))
			multi_head_mask = MultiHeadMask(masks=(masks,) * query.shape[1])
			splash_kernel = make_splash_mha(
				mask=multi_head_mask,
				head_shards=1,
				q_seq_shards=1,
				block_sizes=block_size,
			)

			return jax.vmap(splash_kernel)(query, key, value, segment_ids=mask)

		output = splash_attention_call(
			query_states,
			key_states,
			value_states,
			attention_mask,
		).transpose(0, 2, 1, 3)
		return AttentionOutput(attention_outputs=output, attention_weights=None)

	def cuddn_flash_attention(
		self,
		*,  # it's Kwarg Only
		query_states: Array,
		key_states: Array,
		value_states: Array,
		bias: tp.Optional[Array] = None,
		causal: bool = False,
		deterministic: bool = True,
		query_sequence_length: int,
		key_value_sequence_length: int,
	) -> AttentionOutput:
		"""CUDNN Flash Attention with Transformer Engine."""
		key_states, value_states = self.repeat_kv_heads(
			key_states,
			value_states,
			self.num_q_heads // self.num_kv_heads,
		)
		try:
			import transformer_engine.jax.attention as attention  # noqa #type:ignore
			from transformer_engine.jax.attention import (  # noqa #type:ignore
				AttnBiasType,
				AttnMaskType,
				QKVLayout,
			)
			from transformer_engine.jax.attention import (  # noqa #type:ignore
				is_fused_attn_kernel_available,
			)
		except (ModuleNotFoundError, ImportError) as err:
			raise RuntimeError(
				"Please install transformer_engine first. you can install that by running "
				f"`pip install git+https://github.com/NVIDIA/TransformerEngine`"
				f"\nhere's extra information on error\n{err}",
			) from err
		batch, query_sequence_length, num_attention_heads, head_dim = query_states.shape

		qkv_layout = QKVLayout.BS3HD
		attn_mask_type = AttnMaskType.CAUSAL_MASK
		attn_bias_type = AttnBiasType.NO_BIAS

		if self.sm_scale is None:
			self.sm_scale = 1 / math.sqrt(head_dim)
		has_fused_attn_kernel = is_fused_attn_kernel_available(
			q_dtype=self.dtype,
			kv_dtype=self.dtype,
			qkv_layout=qkv_layout,
			attn_bias_type=attn_bias_type,
			attn_mask_type=attn_mask_type,
			dropout_probability=self.attention_dropout,
			q_num_heads=self.num_q_heads,
			kv_num_heads=self.num_q_heads,
			q_max_seqlen=query_sequence_length,
			kv_max_seqlen=key_value_sequence_length,
			head_dim=head_dim,
		)

		if not has_fused_attn_kernel:
			raise ValueError(
				"Flash attention kernel is not supported for current requested arrays"
				" for details check this repo https://github.com/NVIDIA/TransformerEngine/"
			)

		return AttentionOutput(
			attention_weights=None,
			attention_outputs=attention.fused_attn(
				qkv=jnp.concatenate(
					(
						jnp.reshape(
							query_states,
							(*query_states.shape[:2], 1, *query_states.shape[-2:]),
						),
						jnp.reshape(
							key_states,
							(*query_states.shape[:2], 1, *query_states.shape[-2:]),
						),
						jnp.reshape(
							value_states,
							(*query_states.shape[:2], 1, *query_states.shape[-2:]),
						),
					),
					axis=2,
				),
				bias=bias,
				# mask=(
				# 	jnp.zeros((batch, 1, query_sequence_length, key_value_sequence_length))
				# 	if causal
				# 	else None
				# ),
				seed=None,
				attn_bias_type=attn_bias_type,
				attn_mask_type=attn_mask_type,
				scaling_factor=self.sm_scale,
				dropout_probability=self.attention_dropout,
				is_training=deterministic,
			),
		)

	@staticmethod
	def repeat_kv_heads(key, value, num_reps: int):
		return (
			einops.repeat(key, "b s h d -> b s (h r) d", r=num_reps),
			einops.repeat(value, "b s h d -> b s (h r) d", r=num_reps),
		)

	@staticmethod
	def run_attention_benchmarks(
		batch_sizes=[1, 2, 4, 8],  # noqa
		sequence_lengths=[512, 1024, 2048],  # noqa
		attention_types=None,
	) -> tp.Dict[str, tp.Union[tp.Dict, "pd.DataFrame"]]:
		"""Run comprehensive benchmarks across different configurations."""
		results = {}

		for batch_size in batch_sizes:
			for seq_len in sequence_lengths:
				config = BenchmarkConfig(
					batch_size=batch_size,
					sequence_length=seq_len,
					num_warmup_runs=2,
					num_benchmark_runs=5,
					blocksize_k=64,
					blocksize_q=32,
				)

				benchmarker = AttentionBenchmarker(
					config=config,
					run_attention_benchmarks=attention_types,
					calculate_gradients=True,
				)

				results[f"b{batch_size}_s{seq_len}"] = benchmarker.run_benchmarks()

		return results

	def __repr__(self):
		string = f"{self.__class__.__name__}(\n"
		for k, v in self.__dict__.items():
			if not k.startswith("_"):
				try:
					repr_src = f"  {k} : " + v.__str__().replace("\n", "\n  ") + "\n"
					string += (
						repr_src
						if len(repr_src) < 500
						else f"  {k} : " + f"{v.__class__.__name__}(...)" + "\n"
					)
				except TypeError:
					pass
		return string + ")"

	def __str__(self):
		return self.__repr__()


class BenchmarkMetrics(str, Enum):
	LATENCY = "latency"
	THROUGHPUT = "throughput"
	MEMORY = "memory"
	ACCURACY = "accuracy"
	GRADIENT_DIFF = "gradient_diff"


@dataclass
class BenchmarkConfig:
	batch_size: int = 8
	sequence_length: int = 128 * 8
	num_attention_heads: int = 32
	num_key_value_heads: int = 32
	blocksize_q: int = 64
	blocksize_k: int = 64
	axis_dims: tp.Tuple[int, int, int, int] = (1, -1, 1, 1)
	head_dim: int = 128
	dtype: jnp.dtype = jnp.float16
	num_warmup_runs: int = 3
	num_benchmark_runs: int = 10
	metrics: tp.List[BenchmarkMetrics] = None


class AttentionBenchmarker:
	_printed_errors = set()

	def __init__(
		self,
		config: BenchmarkConfig,
		run_attention_benchmarks: tp.List[str] = None,
		calculate_gradients: bool = True,
	):
		from fjformer import GenerateRNG

		from easydel import MistralConfig

		self.config = config
		self.run_attention_benchmarks = (
			run_attention_benchmarks or _AVAILABLE_ATTENTION_MECHANISMS
		)
		self.calculate_gradients = calculate_gradients
		self.rng = GenerateRNG()

		try:
			import pandas as pd

			self.pd = pd
		except (ModuleNotFoundError, ImportError):
			warnings.warn(
				"Couldn't import pandas. Results will be returned as dict.", stacklevel=1
			)
			self.pd = None

		self.model_config = MistralConfig(
			axis_dims=config.axis_dims,
			blocksize_q=config.blocksize_q,
			blocksize_k=config.blocksize_k,
			attn_dtype=config.dtype,
		)

	def _create_attention_inputs(self):
		"""Create random inputs for attention testing."""
		query = jax.nn.initializers.normal(2.0)(
			self.rng.rng,
			(
				self.config.batch_size,
				self.config.sequence_length,
				self.config.num_attention_heads,
				self.config.head_dim,
			),
			dtype=self.config.dtype,
		)
		key = jax.nn.initializers.normal(2.0)(
			self.rng.rng,
			(
				self.config.batch_size,
				self.config.sequence_length,
				self.config.num_key_value_heads,
				self.config.head_dim,
			),
			dtype=self.config.dtype,
		)
		value = jax.nn.initializers.normal(2.0)(
			self.rng.rng,
			(
				self.config.batch_size,
				self.config.sequence_length,
				self.config.num_key_value_heads,
				self.config.head_dim,
			),
			dtype=self.config.dtype,
		)

		# Create attention masks
		causal_mask = nn.make_causal_mask(
			jnp.ones((self.config.batch_size, self.config.sequence_length))
		)
		attention_mask = jnp.ones((self.config.batch_size, self.config.sequence_length))
		attention_mask = attention_mask.at[:, self.config.sequence_length // 2 :].set(0)

		bias = jnp.where(
			nn.combine_masks(
				jnp.expand_dims(jnp.expand_dims(attention_mask, 1), 1),
				causal_mask,
			),
			0,
			jnp.finfo(query.dtype).min,
		)

		return query, key, value, bias, attention_mask

	def _measure_memory_usage(self, fn, *args, **kwargs):
		"""Measure peak memory usage of a function."""
		try:
			import psutil

			process = psutil.Process()
			memory_before = process.memory_info().rss
			result = fn(*args, **kwargs)
			memory_after = process.memory_info().rss
			return result, memory_after - memory_before
		except ImportError:
			warnings.warn(
				"psutil not available. Memory usage won't be measured.", stacklevel=1
			)
			return fn(*args, **kwargs), None

	def _create_attention_fn(self, attention_type: str):
		"""Create attention function based on type."""
		if attention_type == "dot_product":
			return lambda q, k, v, b, _: nn.dot_product_attention(q, k, v, b)

		return lambda q, k, v, b, mask: FlexibleAttentionModule(
			attn_mechanism=attention_type,
			axis_name="sp",
			dtype=self.config.dtype,
			mesh=self.model_config.mesh,
			head_dims=q.shape[-1],
			sm_scale=q.shape[-1] ** -0.5,
			num_q_heads=q.shape[-2],
			num_kv_heads=q.shape[-2],
			blocksize_q=self.config.blocksize_q,
			blocksize_k=self.config.blocksize_k,
			base_config=self.model_config,
		)(
			query_states=q,
			key_states=k,
			value_states=v,
			bias=b,
			attention_mask=mask,
		).attention_outputs

	def _value_and_grad_wrapper(self, fn):
		"""Wrapper to compute both value and gradient."""

		def inner(*args, **kwargs):
			return jnp.sum(fn(*args, **kwargs))

		return jax.value_and_grad(inner) if self.calculate_gradients else inner

	def _compute_diff(self, t1, t2):
		"""Compute maximum absolute difference between tensors."""
		return jnp.max(jnp.abs(t1 - t2))

	def _benchmark_single_attention(self, attention_fn, inputs):
		"""Benchmark a single attention mechanism with gradient computation."""
		query, key, value, bias, attention_mask = inputs
		metrics = {}

		# Wrap function to compute gradients
		wrapped_fn = self._value_and_grad_wrapper(attention_fn)

		# Warmup runs
		for _ in range(self.config.num_warmup_runs):
			if self.calculate_gradients:
				_, _ = wrapped_fn(query, key, value, bias, attention_mask)
			else:
				_ = wrapped_fn(query, key, value, bias, attention_mask)

		# Benchmark runs
		latencies = []
		throughputs = []
		memory_usage = []
		outputs = []
		gradients = []

		for _ in range(self.config.num_benchmark_runs):
			start_time = time.perf_counter()
			if self.calculate_gradients:
				(result, grad), mem = self._measure_memory_usage(
					wrapped_fn, query, key, value, bias, attention_mask
				)
				gradients.append(grad)
			else:
				result, mem = self._measure_memory_usage(
					wrapped_fn, query, key, value, bias, attention_mask
				)

			end_time = time.perf_counter()

			outputs.append(result)
			latency = end_time - start_time
			throughput = (self.config.batch_size * self.config.sequence_length) / latency

			latencies.append(latency)
			throughputs.append(throughput)
			if mem is not None:
				memory_usage.append(mem)

		metrics.update(
			{
				"mean_latency": jnp.mean(jnp.array(latencies)),
				"std_latency": jnp.std(jnp.array(latencies)),
				"mean_throughput": jnp.mean(jnp.array(throughputs)),
				"peak_memory": max(memory_usage) if memory_usage else None,
			}
		)

		# Average results across runs
		avg_output = jnp.mean(jnp.stack(outputs), axis=0)
		if self.calculate_gradients:
			avg_gradient = jax.tree_util.tree_map(
				lambda *x: jnp.mean(jnp.stack(x), axis=0), *gradients
			)
			return metrics, (avg_output, avg_gradient)

		return metrics, avg_output

	def run_benchmarks(self) -> tp.Union[tp.Dict, "pd.DataFrame"]:
		"""Run benchmarks with gradient comparison."""
		inputs = self._create_attention_inputs()

		# Get baseline results
		baseline_fn = self._create_attention_fn("dot_product")
		baseline_metrics, baseline_results = self._benchmark_single_attention(
			baseline_fn, inputs
		)
		if self.calculate_gradients:
			baseline_output, baseline_grads = baseline_results
		else:
			baseline_output = baseline_results

		results = {}
		for attn_name in self.run_attention_benchmarks:
			try:
				attention_fn = self._create_attention_fn(attn_name)
				metrics, attn_results = self._benchmark_single_attention(attention_fn, inputs)

				if self.calculate_gradients:
					output, grads = attn_results
					# Compare gradients
					grad_diffs = jax.tree_util.tree_map(self._compute_diff, baseline_grads, grads)
					grad_diff_sum = sum(
						jnp.sum(diff) for diff in jax.tree_util.tree_leaves(grad_diffs)
					)

					metrics.update(
						{
							"grad_diff_sum": float(grad_diff_sum),
							"grad_check_passed": float(grad_diff_sum) < 1.0,
						}
					)
				else:
					output = attn_results

				# Compare outputs
				output_diff = self._compute_diff(baseline_output, output)

				results[attn_name] = {
					**metrics,
					"output_diff": float(output_diff),
					"output_check_passed": jnp.allclose(
						baseline_output,
						output,
						atol=0.045,
						rtol=0,
					),
				}

			except Exception as e:
				error_key = f"{attn_name}:{str(e)}"
				if error_key not in self._printed_errors:
					print(f"Benchmark failed for {attn_name}: {str(e)}")
					self._printed_errors.add(error_key)

				results[attn_name] = {"error": str(e), "status": "failed"}

		if self.pd is not None:
			df = self.pd.DataFrame.from_dict(results, orient="index")
			# Reorder columns for better readability
			cols = (["status", "error"] if "status" in df.columns else []) + [
				"mean_latency",
				"std_latency",
				"mean_throughput",
				"peak_memory",
				"output_diff",
				"output_check_passed",
			]
			if self.calculate_gradients:
				cols.extend(["grad_diff_sum", "grad_check_passed"])

			return df[cols]

		return results


class FlaxAttentionModule(nn.Module):
	def __init__(
		self,
		config: "EasyDeLBaseConfig",  # type:ignore  # noqa
	):
		super().__init__()
		self.config = config

		self.cached_key: nn.Cache[Array] | None = None
		self.cached_value: nn.Cache[Array] | None = None
		self.cache_index: nn.Cache[Array] | None = None

	@cached_property
	def quantizer(self):
		return EasyQuantizer(
			quantization_method=self.config.kv_cache_quantization_method,
			block_size=self.config.kv_cache_quantization_blocksize,
		)

	@staticmethod
	def _transpose_sequence_head(*args):
		"""The _transpose_sequence_head function transposes the query, key and value matrices.

		Args:
		    *args: arrays to transpose

		Returns:
		    The transpose of the query, key and value matrices
		"""
		return map(
			lambda x: jnp.transpose(x, (0, 2, 1, 3)),
			args,
		)

	def _concatenate_to_cache(
		self,
		query: Array,
		key: Array,
		value: Array,
		cache_view: TransformerCacheView,
		attention_mask: Array,
		causal_mask: tp.Optional[Array] = None,
	) -> tp.Tuple[Array, Array, Array]:
		num_updated_cache_vectors = query.shape[1]
		end_index = cache_view.index[0]

		key_value_specs = PartitionSpec(
			self.config.partition_axis.batch_axis,
			self.config.partition_axis.key_sequence_axis,
			self.config.partition_axis.head_axis,
			self.config.partition_axis.attention_dim_axis,
		)

		*batch_dims, max_length, num_heads, depth_per_head = cache_view.value.shape

		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

		if causal_mask is not None:
			causal_mask = lax.dynamic_slice(
				causal_mask,
				(0, 0, end_index, 0),
				(1, 1, num_updated_cache_vectors, max_length),
			)
			causal_mask = jnp.broadcast_to(
				causal_mask,
				(query.shape[0],) + causal_mask.shape[1:],
			)
			attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
			attention_mask = jnp.logical_and(attention_mask, causal_mask)

		slice_indices = (0, end_index % cache_view.value.shape[1], 0, 0)
		value_cache = cache_view.value
		key_cache = cache_view.key
		try:
			key_cache = key_cache.materialize()
			value_cache = value_cache.materialize()
		except Exception:
			...
		value_cache = lax.dynamic_update_slice(value_cache, value, slice_indices)
		key_cache = lax.dynamic_update_slice(key_cache, key, slice_indices)
		pad_mask = jnp.broadcast_to(
			jnp.arange(max_length) < end_index + num_updated_cache_vectors,
			tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
		)
		attention_mask = jnp.logical_and(pad_mask, attention_mask)

		cache_view.key = self.quantizer(
			with_sharding_constraint(key_cache, key_value_specs)
		)
		cache_view.value = self.quantizer(
			with_sharding_constraint(value_cache, key_value_specs)
		)
		cache_view.index = cache_view.index + num_updated_cache_vectors

		return key_cache, value_cache, attention_mask

	def concatenate(
		self,
		*,
		query: Array,
		key: Array,
		value: Array,
		attention_mask: Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		causal_mask: tp.Optional[Array] = None,
		fcm_mask: tp.Optional[Array] = None,
		sliding_windows: tp.Optional[int] = None,
	) -> tp.Tuple[Array, Array, Array, Array]:
		if cache_view is None:
			query_length = query.shape[1]
			key_length = key.shape[1]
			if causal_mask is not None:
				causal_mask = causal_mask[:, :, :query_length, :key_length]
				causal_mask = jnp.broadcast_to(
					causal_mask, (query.shape[0],) + causal_mask.shape[1:]
				)
				if attention_mask.ndim == 2:
					attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

				attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
				attention_mask = nn.combine_masks(attention_mask, causal_mask, fcm_mask)

			else:
				attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
				attention_mask = jnp.repeat(attention_mask, query.shape[1], -2)
		else:
			key, value, attention_mask = self._concatenate_to_cache(
				query=query,
				key=key,
				value=value,
				cache_view=cache_view,
				attention_mask=attention_mask,
				causal_mask=causal_mask,
			)
		if sliding_windows is not None:
			sliding_window_mask = jnp.tril(
				jnp.ones_like(attention_mask, dtype=jnp.bool),
				k=-sliding_windows,
			)
			window_mask = jnp.where(sliding_window_mask, 0, 1)
			attention_mask = jnp.logical_and(window_mask, attention_mask)
			if attention_mask.shape[-1] <= 1:
				attention_mask = attention_mask[:, :, :, -sliding_windows:]

		attention_bias = lax.select(
			attention_mask > 0,
			jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
			jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
		)
		return key, value, attention_mask, attention_bias

	def shard_attention_prod(self, attn_output: jax.Array) -> jax.Array:
		"""
		shards attention output before passing that to output_proj

		Args:
		    attn_output (jax.Array): merged output of dot product attention with 3 dims, (batch, seqlen, hidden_size).

		Returns:
		    jax.Array: sharded version of `attn_output`
		"""
		return with_sharding_constraint(
			attn_output,
			PartitionSpec(
				self.config.partition_axis.batch_axis,
				(
					self.config.partition_axis.sequence_axis
					if attn_output.shape[1] != 1
					else None
				),
				self.config.partition_axis.hidden_state_axis,
			),
		)

	def _merge_heads(self, hidden_states: jax.Array) -> jax.Array:
		"""
		Merges the attention heads into a single hidden state tensor.

		Args:
		    hidden_states (jax.Array): The hidden states with separate head dimensions.

		Returns:
		    jax.Array: The hidden states with merged head dimensions.
		"""
		return hidden_states.reshape(hidden_states.shape[:2] + (-1,))

	@staticmethod
	def repeat_key_value(key, value, num_reps: int):
		key = einops.repeat(
			key,
			"b s h d -> b s (h r) d",
			r=num_reps,
		)
		value = einops.repeat(
			value,
			"b s h d -> b s (h r) d",
			r=num_reps,
		)
		return key, value


if __name__ == "__main__":
	import pandas as pd

	FlexibleAttentionModule.run_attention_benchmarks(
		batch_sizes=[1],
	)["b1_s2048"].to_csv("res.csv")
