import math
from typing import Optional, Tuple, Union, List

import chex
import fjformer.linen.linen
from fjformer import linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax, Array
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput

from .chatglm_configuration import ChatGLMConfig
from fjformer.linen import Dense
from ..attention_module import AttentionModule
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel
import flax.linen
# easydel.modules
from ..flax_modelling_utils import (
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    precompute_freq_cis,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
    block_wise_ffn,
    control_mlp_sharding
)

from ..common import RMSNorm


def flatten_axes(a: Array, start: int = 0, end: int = -1) -> Array:
    return a.reshape(a.shape[:start] + (-1,) + a.shape[end:][1:])


def split_tensor_along_last_dim(
        tensor: jax.Array,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> tuple[Array, ...] | list[Array]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.ndim - 1
    last_dim_size = tensor.shape[last_dim] // num_partitions
    # Split.
    tensor_list = jnp.split(tensor, last_dim_size, axis=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(jax.lax.stop_gradient(chunk) for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    rope_ratio: float
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2, dtype=self.dtype) / self.dim))

    def forward(self, seq_len: int, n_elem: int, base: int = 10000):
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (jnp.arange(0, n_elem, 2, dtype=jnp.float32) / n_elem))
        seq_idx = jnp.arange(seq_len, dtype=jnp.float32)
        idx_theta = jnp.outer(seq_idx, theta).astype(jnp.float32)

        cache = jnp.stack([jnp.cos(idx_theta), jnp.sin(idx_theta)], axis=-1)

        if self.dtype in (jnp.float16, jnp.bfloat16, jnp.int8):
            cache = cache.astype(jnp.bfloat16) if self.dtype == jnp.bfloat16 else cache.astype(jnp.float16)
        return cache


def apply_rotary_pos_emb(x: jax.Array, rope_cache: jax.Array) -> jax.Array:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.reshape(-1, 1, sq, xshaped.shape[3], 2)
    x_out2 = jnp.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = flatten_axes(x_out2, 3)
    return jnp.concatenate((x_out2, x_pass), axis=-1)


class CoreAttention(nn.Module):
    config: ChatGLMConfig
    layer_number: int

    def setup(self) -> None:
        layer_number = self.layer_number
        config = self.config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.attention_performer = AttentionModule(
            attention_dropout=self.config.attention_dropout,
            num_attention_heads=self.config.num_attention_heads,
            head_dims=self.head_dim,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            mesh=self.config.get_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
            axis_name=self.config.attention_axis_name,
            base_module_class=self.config
        )

    def __call__(
            self,
            query_layer: jax.Array,
            key_layer: jax.Array,
            value_layer: jax.Array,
            attention_mask: jax.Array,
            causal_mask: jax.Array
    ):
        batch_size = query_layer.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask,
            (batch_size,) + causal_mask.shape[1:]
        )
        mask = causal_mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = jnp.expand_dims(attention_mask, (-3, -2))
            mask = jnp.logical_and(causal_mask, attention_mask)
        bias = lax.select(
            mask,
            jnp.full(mask.shape, 0, dtype=query_layer.dtype),
            jnp.full(mask.shape, jnp.finfo(query_layer.dtype).min, dtype=query_layer.dtype),

        )
        context_layer = self.attention_performer(
            query_layer,
            key_layer,
            value_layer,
            bias=bias,
            attention_mask=attention_mask,
            causal_mask=causal_mask
        ).attention_outputs
        new_context_layer_shape = context_layer.reshape[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer
