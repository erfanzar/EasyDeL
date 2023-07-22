# coding=utf-8
# Copyright 2021 T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This model is copied from the Transformers and this script will apply pjit on them
""" Flax T5 model."""

import copy
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.interpreters import pxla
from jax.random import PRNGKey

from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
)

from transformers import PretrainedConfig
from jax.experimental.pjit import with_sharding_constraint as wsc
from jax.sharding import PartitionSpec


def get_names_from_parition_spec(partition_specs):
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)


def names_in_mesh(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def with_sharding_constraint(x, partition_specs):
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


class T5Config(PretrainedConfig):
    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
            self,
            vocab_size=32128,
            d_model=512,
            d_kv=64,
            d_ff=2048,
            num_layers=6,
            num_decoder_layers=None,
            num_heads=8,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
            gradient_checkpointing: str = 'nothing_saveable',
            use_pjit_attention_force: bool = False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

    def get_partition_rules(self, fully_fsdp: bool = True):
        return (
            ("wi_0/kernel", PartitionSpec('fsdp')),
            ("wi_1/kernel", PartitionSpec('fsdp')),
            ("wi/kernel", PartitionSpec('fsdp', 'mp')),
            ("wo/kernel", PartitionSpec('fsdp', 'mp')),
            ("SelfAttention/(q|k|v|o)/kernel", PartitionSpec('fsdp')),
            ("EncDecAttention/(q|k|v|o)/kernel", PartitionSpec('fsdp')),
            ('.*', PartitionSpec(None))
        ) if not fully_fsdp else (
            ('.*', PartitionSpec('fsdp'))
        )


remat = nn_partitioning.remat


def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = jnp.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


class FlaxT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16
    eps: float = 1e-6
    weight_init: Callable[..., np.ndarray] = jax.nn.initializers.ones

    def setup(self):
        self.weight = self.param("weight", self.weight_init, (self.hidden_size,))

    def __call__(self, hidden_states):
        variance = jnp.power(hidden_states.astype("f4"), 2).mean(axis=-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)

        return self.weight * hidden_states


class FlaxT5DenseActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        wi_init_std = self.config.initializer_factor * (self.config.d_model ** -0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff ** -0.5)

        self.wi = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.act = ACT2FN[self.config.dense_act_fn]

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FlaxT5DenseGatedActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation

    def setup(self):
        wi_init_std = self.config.initializer_factor * (self.config.d_model ** -0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff ** -0.5)

        self.wi_0 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        self.wi_1 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.act = ACT2FN[self.config.dense_act_fn]

    def __call__(self, hidden_states, deterministic):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FlaxT5LayerFF(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation

    def setup(self):
        if self.config.is_gated_act:
            self.DenseReluDense = FlaxT5DenseGatedActDense(self.config, dtype=self.dtype)
        else:
            self.DenseReluDense = FlaxT5DenseActDense(self.config, dtype=self.dtype)

        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, hidden_states, deterministic=True):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states, deterministic=deterministic)
        hidden_states = hidden_states + self.dropout(forwarded_states, deterministic=deterministic)
        return hidden_states


class FlaxT5Attention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    causal: bool = False
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation

    def setup(self):
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim ** -0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim ** -0.5)

        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
                dtype=self.dtype,
            )

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):

        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
                jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]
        return values

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):

        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions
            # that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def _create_position_bias(
            self, key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
    ):
        cache_is_filled = self.causal and self.has_variable("cache", "cached_key") and (not init_cache)
        key_length = key_states.shape[1]
        query_length = key_length if cache_is_filled else query_states.shape[1]

        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(query_length, key_length)
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        else:
            position_bias = jnp.zeros((1, self.n_heads, query_length, key_length), dtype=self.dtype)

        # if key and values are already calculated, only the last query position bias should be taken
        if cache_is_filled:
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            position_bias = jax.lax.dynamic_slice(
                position_bias,
                (0, 0, causal_attention_mask_shift, 0),
                (1, self.n_heads, seq_length, max_decoder_length),
            )
        return position_bias

    def __call__(
            self,
            hidden_states,
            attention_mask=None,
            key_value_states=None,
            position_bias=None,
            use_cache=False,
            output_attentions=False,
            deterministic=True,
            init_cache=False,
    ):

        batch_size, seq_length = hidden_states.shape[:2]

        # q, k, v projections
        query_states = self.q(hidden_states)  # (batch_size, n_heads, seq_length, dim_per_head)
        key_states = self.k(hidden_states) if key_value_states is None else self.k(key_value_states)
        value_states = self.v(hidden_states) if key_value_states is None else self.v(key_value_states)
        if self.config.use_pjit_attention_force:
            query_states = with_sharding_constraint(query_states, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            key_states = with_sharding_constraint(key_states, PartitionSpec(('dp', 'fsdp'), None, 'mp'))
            value_states = with_sharding_constraint(value_states, PartitionSpec(('dp', 'fsdp'), None, 'mp'))

        # reshape to (batch_size, seq_length, n_heads, head_dim)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # counter-act scaling in dot_product_attention_weights function
        query_states *= jnp.sqrt(query_states.shape[-1])

        # for fast decoding causal attention mask should be shifted
        causal_attention_mask_shift = (
            self.variables["cache"]["cache_index"] if (self.has_variable("cache", "cached_key") and self.causal) else 0
        )
        # create causal attention_mask; attention_mask has to be defined when model is causal
        if self.causal:
            causal_attention_mask = make_causal_mask(attention_mask, dtype="bool")

            # fast decoding for generate requires special attention_mask
            if self.has_variable("cache", "cached_key"):
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_attention_mask = jax.lax.dynamic_slice(
                    causal_attention_mask,
                    (0, 0, causal_attention_mask_shift, 0),
                    (1, 1, seq_length, max_decoder_length),
                )

            # broadcast causal attention mask & attention mask to fit for merge
            causal_attention_mask = jnp.broadcast_to(
                causal_attention_mask, (batch_size,) + causal_attention_mask.shape[1:]
            )
            attention_mask = jnp.broadcast_to(
                jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_attention_mask.shape
            )
            attention_mask = combine_masks(attention_mask, causal_attention_mask)
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key_states, value_states, attention_attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # replace masked positions with -10_000
        if attention_mask is not None:
            mask_value = jnp.finfo(self.dtype).min
            attention_mask = jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, mask_value).astype(self.dtype),
            )

        if position_bias is None:
            # compute position bias (only for first layer)
            position_bias = self._create_position_bias(
                key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
            )

            if attention_mask is not None:
                position_bias = position_bias + attention_mask

        # create dropout rng
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Softmax(QK^T)
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=position_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
        )

        if self.config.use_pjit_attention_force:
            attn_weights = with_sharding_constraint(attn_weights, PartitionSpec(
                ('dp', 'fsdp'), 'mp', None, None
            ))

        # multiply with value states
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        # bring back to (batch_size, seq_length, d_model)
        attn_output = self._merge_heads(attn_output)

        # apply output matrix
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


class FlaxT5LayerSelfAttention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation

    def setup(self):
        self.SelfAttention = FlaxT5Attention(
            self.config,
            has_relative_attention_bias=self.has_relative_attention_bias,
            causal=self.config.causal,
            dtype=self.dtype,
        )
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            output_attentions=False,
            deterministic=True,
            init_cache=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class FlaxT5LayerCrossAttention(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation

    def setup(self):
        self.EncDecAttention = FlaxT5Attention(
            self.config, has_relative_attention_bias=False, causal=False, dtype=self.dtype
        )
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
            self,
            hidden_states,
            key_value_states,
            attention_mask=None,
            position_bias=None,
            output_attentions=False,
            deterministic=True,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class FlaxT5Block(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation

    def setup(self):
        self.causal = self.config.causal
        self.layer = (
            FlaxT5LayerSelfAttention(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                name=str(0),
                dtype=self.dtype,
            ),
        )
        feed_forward_index = 1
        if self.causal:
            self.layer += (FlaxT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),)
            feed_forward_index += 1

        self.layer += (FlaxT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),)

    def __call__(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            output_attentions=False,
            return_dict=True,
            deterministic=True,
            init_cache=False,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        do_cross_attention = self.causal and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            hidden_states = cross_attention_outputs[0]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        outputs = outputs + attention_outputs

        # returns hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        return outputs


class FlaxT5LayerCollection(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation

    def setup(self):
        block = FlaxT5Block
        if self.config.gradient_checkpointing != '':
            block = remat(
                block, static_argnums=(5, 6, 7, 8, 9),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )
        self.layer = block(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )

    def __call__(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            output_attentions=False,
            deterministic=True,
            init_cache=False,
    ):
        return self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


class FlaxT5BlockCollection(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.causal = self.config.causal
        block = FlaxT5LayerCollection
        if self.config.gradient_checkpointing != '':
            block = remat(block,
                          policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                          static_argnums=(6, 7, 8))
        self.blocks = [
            block(
                self.config,
                has_relative_attention_bias=(i == 0),
                dtype=self.dtype,
                name=str(i),
            )
            for i in range(self.config.num_layers)
        ]

    def __call__(
            self,
            hidden_states=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            deterministic: bool = True,
            init_cache: bool = False,
    ):
        # Prepare head mask if needed
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.causal) else None
        position_bias = None
        encoder_decoder_position_bias = None

        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                output_attentions,
                deterministic,
                init_cache,
            )

            hidden_states = layer_outputs[0]

            position_bias = layer_outputs[1]

            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class FlaxT5Stack(nn.Module):
    config: T5Config
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.causal = self.config.causal

        self.block = FlaxT5BlockCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.final_layer_norm = FlaxT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            deterministic: bool = True,
            init_cache: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        hidden_states = outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # Add last layer
        all_hidden_states = None

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxT5PreTrainedModel(FlaxPreTrainedModel):
    config_class = T5Config
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
            self,
            config: T5Config,
            input_shape: Tuple[int] = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.bfloat16,
            _do_init: bool = True,
            gradient_checkpointing: bool = False,
            **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")

        attention_mask = jnp.ones_like(input_ids)
        args = [input_ids, attention_mask]
        if self.module_class not in [FlaxT5EncoderModule]:
            decoder_input_ids = jnp.ones_like(input_ids)
            decoder_attention_mask = jnp.ones_like(input_ids)
            args.extend([decoder_input_ids, decoder_attention_mask])

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            *args,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
            self,
            input_ids: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            decoder_input_ids: jnp.ndarray = None,
            decoder_attention_mask: Optional[jnp.ndarray] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            train: bool = False,
            params: dict = None,
            dropout_rng: PRNGKey = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if decoder_input_ids is None:
            raise ValueError(
                "Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed"
                " here."
            )

        # prepare encoder inputs
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # prepare decoder inputs
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )

    def init_cache(self, batch_size, max_length, encoder_outputs):

        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # we only need to call the decoder to init the cache
        )
        return unfreeze(init_variables["cache"])

    def encode(
            self,
            input_ids: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            train: bool = False,
            params: dict = None,
            dropout_rng: PRNGKey = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    def decode(
            self,
            decoder_input_ids,
            encoder_outputs,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            decoder_attention_mask: Optional[jnp.ndarray] = None,
            past_key_values: dict = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            train: bool = False,
            params: dict = None,
            dropout_rng: PRNGKey = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs


class FlaxT5Module(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxT5Stack(
            decoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            deterministic: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxT5Model(FlaxT5PreTrainedModel):
    module_class = FlaxT5Module


class FlaxT5EncoderModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
            self,
            input_ids=None,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict: bool = True,
            deterministic: bool = True,
    ):
        # Encode if needed (training, first prediction pass)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        return encoder_outputs


class FlaxT5EncoderModel(FlaxT5PreTrainedModel):
    module_class = FlaxT5EncoderModule

    def __call__(
            self,
            input_ids: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            train: bool = False,
            params: dict = None,
            dropout_rng: PRNGKey = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # prepare encoder inputs
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


class FlaxT5ForConditionalGenerationModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.bfloat16  # the dtype of the computation
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def setup(self):
        self.model_dim = self.config.d_model

        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FlaxT5Stack(
            encoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxT5Stack(
            decoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

    def __call__(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            deterministic: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if self.config.tie_word_embeddings:
            shared_embedding = self.shared.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxT5ForConditionalGeneration(FlaxT5PreTrainedModel):
    module_class = FlaxT5ForConditionalGenerationModule

    def decode(
            self,
            decoder_input_ids,
            encoder_outputs,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            decoder_attention_mask: Optional[jnp.ndarray] = None,
            past_key_values: dict = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            train: bool = False,
            params: dict = None,
            dropout_rng: PRNGKey = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            decoder_outputs = decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

            sequence_output = decoder_outputs[0]

            if self.config.tie_word_embeddings:
                sequence_output = sequence_output * (self.config.d_model ** -0.5)

            if self.config.tie_word_embeddings:
                shared_embedding = module.shared.variables["params"]["embedding"]
                lm_logits = module.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
            else:
                lm_logits = module.lm_head(sequence_output)

            return lm_logits, decoder_outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            max_length,
            attention_mask: Optional[jnp.DeviceArray] = None,
            decoder_attention_mask: Optional[jnp.DeviceArray] = None,
            encoder_outputs=None,
            **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs
