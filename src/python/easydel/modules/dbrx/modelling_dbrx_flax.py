import math
from typing import Optional, Tuple, Union

import chex
from fjformer import linen as nn, auxiliary_load_balancing_loss_func
import jax
import jax.numpy as jnp
import flax.linen
from flax.core import FrozenDict, unfreeze, freeze
from flax.linen import combine_masks
from fjformer.linen import Linear
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import (
    FlaxMaskedLMOutput
)

from .dbrx_configuration import DbrxConfig
from ..attention_module import AttentionModule
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel
# easydel.modules
from ..flax_modelling_utils import (
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
    ACT2FN
)
import flax.struct


@flax.struct.dataclass
class MoeModelOutput:
    last_hidden_state: chex.Array = None
    hidden_states: Optional[Tuple[chex.Array]] = None
    attentions: Optional[Tuple[chex.Array]] = None
    router_logits: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MoeCausalLMOutput(FlaxMaskedLMOutput):
    aux_loss: Optional[chex.Array] = None
    router_logits: Optional[Tuple[chex.Array]] = None


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

    return x.reshape(
        bs,
        s,
        n_kv_heads * n_rep,
        head_dim
    )


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
        self.num_attention_heads = self.config.n_heads
        self.num_key_value_heads = self.config.attn_config.kv_n_heads
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.d_model // self.config.n_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.num_attention_heads == self.config.attn_config.kv_n_heads
        self.Wqkv = Linear(
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
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
            num_attention_heads=self.num_attention_heads,
            attention_dropout=self.config.attn_config.attn_pdrop,
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
            axis_name=self.config.attention_axis_name,
            backward_pass_impl=self.config.flash_attention_backward_pass_impl
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
            self.num_attention_heads,
            self.head_dim
        )
        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim
        )
        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
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
            freq_cis: Tuple[chex.Array, chex.Array],
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
        :param freq_cis: Tuple[chex.Array, chex.Array],: Pass in the frequency coefficients for each position
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
            qkv_states = qkv_states.clip(
                min=-self.config.attn_config.clip_qkv,
                max=self.config.attn_config.clip_qkv
            )

        query_size = self.hidden_size
        key_size = self.num_key_value_heads * self.head_dim

        query_states, key_value_states = jnp.split(qkv_states, [query_size], axis=2)
        key_states, value_states = jnp.split(key_value_states, [key_size], axis=2)

        query_states, key_states, value_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            value=value_states,
            position_ids=position_ids,
            freq_cis=freq_cis,
            batch_size=batch_size,
            sequence_length=sequence_length
        )

        assert_msg = (
            "num_attention_heads repeat wont work likely\n"
            f"INFO :\n\trepeat_kv_bnsh Used with num_key_value_groups = {self.num_key_value_groups}\n\t"
            f"NH : {self.num_attention_heads} KVH : {self.num_attention_heads}"
        )

        assert query_states.shape[-2] == self.num_attention_heads, assert_msg
        assert key_states.shape[-2] == self.num_attention_heads, assert_msg
        assert value_states.shape[-2] == self.num_attention_heads, assert_msg

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

        if not deterministic and self.config.attn_config.attn_pdrop > 0.0:
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
            causal=True,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
            segment_ids=segment_ids,
            causal_mask=causal_mask
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output, PartitionSpec(
                    ("dp", "fsdp"),
                    "sp" if attn_output.shape[1] != 1 else None,
                    "tp"
                )
            )
        attn_output = self.out_proj(attn_output)

        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        return attn_output, attentions.attention_weights


class FlaxDbrxNormAttentionNorm(nn.Module):
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
            freq_cis: Tuple[chex.Array, chex.Array],
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

        hidden_states, attn_weights = self.attn(
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

        return residual_states, hidden_states, attn_weights


class FlaxDbrxExpertGLU(nn.Module):
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
        self.v1 = self.param("v1", init_fn, shape, self.param_dtype)
        self.w2 = self.param("w2", init_fn, shape, self.param_dtype)
        self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn["name"]]

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
            jnp.expand_dims(expert_w1.T, 0),
            precision=self.precision
        )
        x2 = jax.lax.batch_matmul(
            x,
            jnp.expand_dims(expert_v1.T, 0),
            precision=self.precision
        )
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = jax.lax.batch_matmul(
            x1,
            jnp.expand_dims(expert_w2, 0),
            precision=self.precision
        )
        return x1


class FlaxDbrxExperts(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.mlp = FlaxDbrxExpertGLU(
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


class FlaxDbrxRouter(nn.Module):
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
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
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


class FlaxDbrxFFN(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.router = FlaxDbrxRouter(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.experts = FlaxDbrxExperts(
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


class FlaxDbrxBlock(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.hidden_size = self.config.d_model
        self.resid_pdrop = self.config.resid_pdrop
        self.norm_attn_norm = FlaxDbrxNormAttentionNorm(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ffn = FlaxDbrxFFN(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_router_logits: bool = False,
            fcm_mask=None,
    ):
        resid_states, hidden_states, self_attn_weights = self.norm_attn_norm(
            hidden_states=hidden_states,
            freq_cis=freq_cis,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            init_cache=init_cache,
        )

        hidden_states, router_logits = self.ffn(
            hidden_states,
            deterministic=deterministic
        )
        hidden_states = resid_states + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class FlaxDbrxBlockCollection(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.blocks = [
            FlaxDbrxBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=f"{i}"
            )
            for i in range(self.config.n_layers)
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array,
            segment_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_router_logits: bool = False,
            output_hidden_states: bool = False,
            fcm_mask=None,
    ):
        all_hidden_states = ()
        all_router_logits = ()
        all_attentions = ()
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = block(
                hidden_states=hidden_states,
                freq_cis=freq_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                segment_ids=segment_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                fcm_mask=fcm_mask,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions += (outputs[1],)
            if output_router_logits:
                all_router_logits += (outputs[-1],)
        return hidden_states, all_attentions, all_hidden_states, all_router_logits,


class DbrxPreTrainedModel(EasyDeLFlaxPretrainedModel):
    config_class: DbrxConfig = DbrxConfig
    module_class: nn.Module = None
    base_model_prefix = "model"

    def __init__(
            self,
            config: DbrxConfig,
            dtype: jnp.dtype = jnp.bfloat16,
            param_dtype: jnp.dtype = jnp.bfloat16,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            input_shape: Tuple[int, int] = (1, 1),
            seed: int = 0,
            _do_init: bool = False,
            **kwargs
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **kwargs
        )

        super().__init__(
            dtype=dtype, _do_init=_do_init,
            module=module, config=config, input_shape=input_shape,
            seed=seed,
        )

    def init_weights(
            self,
            rng: jax.random.PRNGKey,
            input_shape: Tuple,
            params: FrozenDict = None
    ) -> FrozenDict:
        """
        The init_weights function is used to initialize the weights of a model.
        It takes in a rng, which is a random number generator key that can be used to generate random numbers.
        The input_shape parameter specifies the shape of the inputs that will be fed into this model.
        The params parameter allows you to pass in pre-trained weights for your model, if you have them available.

        :param self: Access variables that belong to the class
        :param rng: jax.random.PRNGKey: Initialize the weights of the model
        :param input_shape: Tuple: Initialize the input_ids, attention_mask and position_ids
        :param params: flax.core.FrozenDict: Pass in the parameters of a pre-trained model
        :return: A frozendict of parameters
        """

        self.config.initialization_of_moe = True
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1], dtype="i4"),
            input_shape,
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(
                input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rngs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=False
            )
        random_params = module_init_outputs["params"]

        self.config.initialization_of_moe = False
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(
            jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            add_params_field: bool = False,
            **kwargs
    ):
        """
        The __call__ function is the main function of a JAX module.
        It takes as input:
        - The parameters of the model (self.params)
        - The inputs to the model (input_ids, attention_mask, position_ids)
        - Whether we are training (train=True/False) and whether we want to return all hidden states and
        attentions weights at each layer in addition to just the last layer output (output_hidden_states=True/False).

        :param self: Represent the instance of the class
        :param input_ids: Pass the input sequence to the model
        :param attention_mask: Mask out the padding tokens
        :param position_ids: Specify the position of each token in the sequence
        :param params: dict: Pass in the parameters of the model
        :param past_key_values: dict: Pass the past key values to the model
        :param dropout_rng: jax.random.PRNGKey: Pass in a random number generator key to the model
        :param train: bool: Determine whether to use dropout or not
        :param output_attentions: Optional[bool]: Determine whether to return the attention weights
        :param output_hidden_states: Optional[bool]: Determine whether to return the hidden states of all layers
        :param return_dict: Optional[bool]: Return a dictionary of the outputs
        :param add_params_field: bool: Add a params field to the inputs dictionary
        :return: A tuple of (last_hidden_state, past_key_values)

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[
                                            None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rng_s = {}
        if dropout_rng is not None:
            rng_s["dropout"] = dropout_rng

        inputs = {
            "params": params or self.params} if add_params_field else params or self.params

        if self.config.bits is not None:
            rng_s['params'] = jax.random.key(0)
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),  # input_ids: chex.Array
            # attention_mask: Optional[chex.Array] = None
            jnp.array(attention_mask, dtype="i4"),
            # position_ids: Optional[chex.Array] = None
            jnp.array(position_ids, dtype="i4"),
            None,  # inputs_embeds: Optional[chex.Array] = None
            output_attentions,  # output_attentions: Optional[bool] = None
            # output_hidden_states: Optional[bool] = None
            output_hidden_states,
            # output_router_logits: Optional[bool] = None
            output_router_logits,
            False,  # init_cache: bool = False
            not train,  # deterministic: bool = True
            return_dict,  # return_dict: bool = True
            rngs=rng_s,
            mutable=mutable,
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxDbrxModule(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.emb_pdrop = self.config.emb_pdrop

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.blocks = FlaxDbrxBlockCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm_f = nn.LayerNorm(
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

        initial_rope_kwargs = dict(
            rope_type="none"
        )
        if getattr(self.config, "rope_scaling", None) is not None:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            initial_rope_kwargs = dict(
                scaling_factor=scaling_factor,
                rope_type=scaling_type
            )
        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=(
                getattr(self.config, "freq_max_position_embeddings", self.config.max_seq_len)
            ),
            dim=self.config.d_model // self.config.n_heads,
            base=self.config.attn_config.rope_theta,
            **initial_rope_kwargs
        )
        self.causal_mask = flax.linen.make_causal_mask(
            jnp.ones(
                (1, getattr(self.config, "c_max_position_embeddings", self.config.max_seq_len)),
                dtype="bool"
            ), dtype="bool"
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            inputs_embeds: Optional[chex.Array] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            return_dict: bool = True,
    ) -> Union[Tuple, MoeModelOutput]:
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.wte(input_ids.astype("i4"))
        else:
            raise ValueError(
                "you should specify inputs_embeds or input_ids one of them")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        collection_outputs = self.blocks(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            causal_mask=self.causal_mask,
            freq_cis=self.freq_cis,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            init_cache=init_cache,
            deterministic=deterministic,
        )
        all_self_attns = None
        all_hidden_states = None
        all_router_logits = None
        hidden_states = collection_outputs[0]
        if output_attentions:
            all_self_attns = collection_outputs[1]
        if output_hidden_states:
            all_hidden_states = collection_outputs[2 if output_attentions else 1]
        if output_router_logits:
            all_router_logits = collection_outputs[-1]
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class FlaxDbrxModel(DbrxPreTrainedModel):
    module_class = FlaxDbrxModule


class FlaxDbrxForCausalLMModule(nn.Module):
    config: DbrxConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.transformer = FlaxDbrxModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.lm_head = Linear(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            inputs_embeds: Optional[chex.Array] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            init_cache: bool = False,
            deterministic: bool = True,
            return_dict: bool = True,
    ) -> MoeCausalLMOutput | Tuple:

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            init_cache=init_cache,
            deterministic=deterministic,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        batch_size, seq_length, hd = logits.shape
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=tuple(  # type:ignore
                    [logit.reshape(batch_size * seq_length, -1) for logit in outputs.router_logits]  # type:ignore
                ),
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask
            )
            aux_loss = aux_loss * self.config.router_aux_loss_coef
        if not return_dict:
            outputs = (logits,) + tuple(
                v
                for v in [
                    aux_loss,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits
                ]
                if v is not None
            )
            return outputs

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class FlaxDbrxForCausalLM(DbrxPreTrainedModel):
    module_class = FlaxDbrxForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        """
        The prepare_inputs_for_generation function is used to prepare the inputs for a generation task.

        :param self: Access variables that belong to the class
        :param input_ids: Pass in the input tokens
        :param max_length: Set the length of the sequence to be generated
        :param attention_mask: Optional[chex.Array]: Mask the attention weights
        :return: A dictionary of the past_key_values, attention_mask and position ids

        """
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones(
            (batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[
                                            None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
