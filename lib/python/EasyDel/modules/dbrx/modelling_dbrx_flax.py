import math
import random
from typing import Optional, Tuple, Union

import chex
from fjformer import linen as nn
import jax
import jax.numpy as jnp
import flax.linen
from flax.linen import combine_masks
from fjformer.linen import Linear
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput

from .dbrx_configuration import DbrxConfig, DbrxAttentionConfig, DbrxFFNConfig
from ..attention_module import AttentionModule
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel
# EasyDel.modules
from ..flax_modelling_utils import (
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    get_dot_general_by_bits, BaseJAXAttentionModule, block_wise_ffn, ACT2FN
)


class FlaxDbrxEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, query, key, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query = apply_rotary_pos_emb(query, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


def repeat_kv(x: chex.Array, n_rep: int) -> chex.Array:
    bs, s, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jnp.newaxis, :, :]
    x = jnp.repeat(x, n_rep, axis=2)

    return x.reshape(bs, s,
                     n_kv_heads * n_rep,
                     head_dim)


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: chex.Array) -> chex.Array:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: chex.Array) -> chex.Array:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = nn.linen.control_quantization(self.weight, self.dtype)
        return output * weight


class FlaxDbrxAttention(BaseJAXAttentionModule):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.Wqkv = Linear(
            self.hidden_size + 2 * self.config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

        self.out_proj = Linear(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

        self.rotary = FlaxDbrxEmbedding(self.dtype)
        self.attention_performer = AttentionModule(
            use_sharding_constraint=self.config.use_sharding_constraint,
            block_k_major=self.config.block_k_major,
            block_b=self.config.block_b,
            block_q=self.config.block_q,
            block_k=self.config.block_k,
            block_q_major_dkv=self.config.block_q_major_dkv,
            block_k_major_dkv=self.config.block_k_major_dkv,
            block_k_major_dq=self.config.block_k_major_dq,
            block_k_dkv=self.config.block_k_dkv,
            block_q_dkv=self.config.block_q_dkv,
            block_q_dq=self.config.block_q_dq,
            block_k_dq=self.config.block_k_dq,
            num_attention_heads=self.config.num_attention_heads,
            attention_dropout=self.config.attention_dropout,
            head_dims=self.head_dim,
            attention_partition_spec=self.config.attention_partition_spec,
            shard_attention_computation=self.config.shard_attention_computation,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            bias_partition_spec=self.config.bias_partition_spec,
            key_partition_spec=self.config.key_partition_spec,
            query_partition_spec=self.config.query_partition_spec,
            generation_query_partition_spec=self.config.generation_query_partition_spec,
            generation_bias_partition_spec=self.config.generation_bias_partition_spec,
            generation_attention_partition_spec=self.config.generation_attention_partition_spec,
            value_partition_spec=self.config.value_partition_spec,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.jax_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
        )
        self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop)

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @staticmethod
    def _transpose_sequence_head(query, key, value):
        """
        The _transpose_sequence_head function transposes the query, key and value matrices.

        :param query: Get the attention weights for each of the heads
        :param key: Determine the number of heads
        :param value: Store the values of the input
        :return: The transpose of the query, key and value matrices

        """
        return jnp.transpose(query, (0, 2, 1, 3)), jnp.transpose(key, (0, 2, 1, 3)), jnp.transpose(value, (0, 2, 1, 3))

    def apply_rotary(self, batch_size, sequence_length, query, key, value, freq_cis, position_ids):
        """
        The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freq_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        :param self: Access variables that belong to the class
        :param batch_size: Reshape the query, key and value tensors
        :param sequence_length: Reshape the query, key and value tensors
        :param query: Calculate the attention weights
        :param key: Calculate the attention
        :param value: Compute the attention weights
        :param freq_cis: Calculate the frequency of each word in the vocabulary
        :param position_ids: Identify the position of each token in the sequence
        :return: A tuple of 3 tensors: query, key and value

        """
        query = query.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim
        )
        key = key.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim
        )
        value = value.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim
        )

        query, key, value = self._transpose_sequence_head(query, key, value)
        query, key = self.rotary(
            position_ids=position_ids, query=query, key=key, freq_cis=freq_cis
        )
        key = repeat_kv_bnsh(key, self.num_key_value_groups)
        value = repeat_kv_bnsh(value, self.num_key_value_groups)
        return self._transpose_sequence_head(query, key, value)

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask=None,
    ):
        """
        The __call__ function is the main function of a JAX module. It defines how the module behaves when called
        with inputs. The __call__ function can be thought of as a &quot;forward pass&quot; through the model,
        and it should return all outputs that are needed for training or inference.

        :param self: Access variables that belong to the class
        :param hidden_states: chex.Array: Pass the hidden states of the previous layer
        :param freq_cis: chex.Array: Pass in the frequency coefficients for each position
        :param attention_mask: chex.Array: Mask out certain tokens in the input sequence
        :param position_ids: chex.Array: Determine the position of each token in a sequence
        :param causal_mask: chex.Array: Mask out the future tokens in the decoder
        :param deterministic: bool: Determine whether to use dropout or not
        :param init_cache: bool: Initialize the cache
        :param output_attentions: bool: Determine whether to return the attention weights or not
        :param fcm_mask: Mask out the attention weights between the input and output tokens
        :param : Determine if the attention is causal or not
        :return: A tuple of two arrays

        """
        batch_size, sequence_length = hidden_states.shape[:2]
        qkv_states = self.Wqkv(hidden_states)
        if self.config.attn_config.clip_qkv is not None:
            qkv_states = qkv_states.clip(min=-self.self.config.attn_config.clip_qkv, max=self.clip_qkv)

        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=2,
        )

        query_states, key_states, value_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            value=value_states,
            position_ids=position_ids,
            freq_cis=freq_cis,
            batch_size=batch_size,
            sequence_length=sequence_length
        )

        # if self.config.use_sharding_constraint:
        #     query_states = with_sharding_constraint(
        #         query_states, PartitionSpec(("dp", "fsdp"), "sp" if query_states.shape != [1] else None, "tp", None)
        #     )
        #     key_states = with_sharding_constraint(key_states, PartitionSpec(("dp", "fsdp"), "sp", "tp", None))
        #     value_states = with_sharding_constraint(value_states, PartitionSpec(("dp", "fsdp"), "sp", "tp", None))

        assert_msg = (
            "num_attention_heads repeat wont work likely\n"
            f"INFO :\n\trepeat_kv_bnsh Used with num_key_value_groups = {self.num_key_value_groups}\n\t"
            f"NH : {self.config.num_attention_heads} KVH : {self.config.num_attention_heads}"
        )

        assert query_states.shape[-2] == self.config.num_attention_heads, assert_msg
        assert key_states.shape[-2] == self.config.num_attention_heads, assert_msg
        assert value_states.shape[-2] == self.config.num_attention_heads, assert_msg

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(
            attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        dropout_rng = None

        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states,
                value_states,
                query_states,
                attention_mask
            )

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(
                self.dtype).min).astype(self.dtype),
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_performer.__call__(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=False,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
            segment_ids=segment_ids
        )
        attentions.attention_outputs = attentions.attention_outputs

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation and self.config.attn_mechanism == "vanilla":
            attn_output = with_sharding_constraint(
                attn_output, PartitionSpec(
                    ("dp", "fsdp"),
                    "sp" if attn_output.shape[1] != 1 else None,
                    "tp"
                )
            )
        attn_output = self.out_proj(attn_output)

        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attentions.attention_weights) if output_attentions else (attn_output,)
        return outputs


class DbrxNormAttentionNorm(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.norm_1 = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False
        )
        self.attn = FlaxDbrxAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm_2 = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False
        )

        self.dropout = flax.linen.Dropout(
            self.config.resid_pdrop
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask=None,
    ):
        residual_states = hidden_states
        hidden_states = self.norm_1(hidden_states)

        hidden_states, attn_weights, past_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            freq_cis=freq_cis,
            causal_mask=causal_mask,
            segment_ids=segment_ids,
            init_cache=init_cache,
            deterministic=deterministic,
            fcm_mask=fcm_mask
        )

        hidden_states = self.dropout(
            hidden_states,
            deterministic=deterministic
        )
        hidden_states = hidden_states + residual_states

        residual_states = hidden_states
        hidden_states = self.norm_2(hidden_states)

        return residual_states, hidden_states, attn_weights, past_key_value


class DbrxExpertGLU(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        shape = (
            self.config.ffn_config.moe_num_experts * self.config.ffn_config.ffn_hidden_size,
            self.config.d_model
        )
        init_fn = nn.initializers.normal(
            dtype=self.dtype
        )
        self.w1 = self.param("w1", init_fn, shape, self.param_dtype)
        self.v1 = self.param("w1", init_fn, shape, self.param_dtype)
        self.w2 = self.param("w2", init_fn, shape, self.param_dtype)
        self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn]

    def __call__(self, x: chex.Array, expert_idx: int) -> chex.Array:
        expert_shape = (
            self.config.ffn_config.moe_num_experts,
            self.config.ffn_config.ffn_hidden_size,
            self.config.d_model
        )
        expert_w1 = self.w1.reshape(expert_shape)[expert_idx]
        expert_v1 = self.v1.reshape(expert_shape)[expert_idx]
        expert_w2 = self.w2.reshape(expert_shape)[expert_idx]

        x1 = jax.lax.batch_matmul(
            x,
            expert_w1.T,
            precision=self.precision
        )
        x2 = jax.lax.batch_matmul(
            x,
            expert_v1.T,
            precision=self.precision
        )
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = jax.lax.batch_matmul(
            x1,
            expert_w2,
            precision=self.precision
        )
        return x1


class DbrxExperts(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.mlp = DbrxExpertGLU(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            x: chex.Array,
            weights: chex.Array,
            top_weights: chex.Array,
            top_experts: chex.Array
    ):
        final_hidden_state = jnp.zeros_like(x)
        for index in range(self.config.ffn_config.moe_num_experts):
            output_moe_layer = self.mlp(
                x, index
            )
            final_hidden_state += jnp.sum(
                jnp.multiply(
                    index == top_experts, top_weights
                ), axis=-1
            )[:, :, None] * output_moe_layer
        return final_hidden_state


class DbrxRouter(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.hidden_size = self.config.d_model
        self.moe_num_experts = self.config.ffn_config.moe_num_experts
        self.moe_top_k = self.config.ffn_config.moe_top_k
        self.moe_jitter_eps = self.config.ffn_config.moe_jitter_eps
        self.moe_normalize_expert_weights = self.config.ffn_config.moe_normalize_expert_weights
        self.uniform_expert_assignment = self.config.ffn_config.uniform_expert_assignment

        self.layer = Linear(
            self.moe_num_experts,
            use_bias=False
        )

    def jitter(self, x: chex.Array) -> chex.Array:
        if self.moe_jitter_eps is None:
            raise RuntimeError('The router does not have moe_jitter_eps set.')
        low = 1.0 - self.moe_jitter_eps
        high = 1.0 + self.moe_jitter_eps
        noise = jax.random.normal(
            self.make_rng("params"),
            x.shape,
            dtype=x.dtype
        )
        return low + noise * (high - low)

    def __call__(
            self,
            x: chex.Array,
            deterministic: bool = True
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        if not deterministic and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        weights = self.layer(
            x.astype(
                jnp.promote_types(self.dtype, jnp.float32)
            )
        )
        weights = jax.nn.softmax(
            weights.astype(
                jnp.promote_types(self.dtype, jnp.float32)
            )
        )
        top_weights, top_experts = jax.lax.top_k(
            weights,
            self.moe_top_k
        )

        if self.moe_normalize_expert_weights:
            top_weights = top_weights / jnp.linalg.norm(
                top_weights,
                ord=int(self.moe_normalize_expert_weights),
                axis=-1,
                keepdims=True
            )

        if self.uniform_expert_assignment:
            top_experts = jax.lax.stop_gradient(
                (
                        jnp.arange(
                            0,
                            jnp.prod(jnp.asarray(top_experts.shape, dtype=jnp.int32), dtype=jnp.int32),
                            dtype=top_experts.dtype
                        ) % self.moe_num_experts
                ).reshape(top_experts.shape)
            )

        weights = weights.astype(x.dtype)
        top_weights = top_weights.astype(x.dtype)
        return weights, top_weights, top_experts


class DbrxFFN(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.router = DbrxRouter(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.experts = DbrxExperts(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            x: chex.Array,
            deterministic: bool = False
    ) -> Tuple[chex.Array, chex.Array]:
        weights, top_weights, top_experts = self.router(x, deterministic=deterministic)
        out = self.experts(x, weights, top_weights, top_experts)
        return out, weights
