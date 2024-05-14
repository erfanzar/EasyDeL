import functools
import math
import typing

import fjformer
import flax.core
from jax import numpy as jnp, Array, lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
import jax
from flax.traverse_util import unflatten_dict, flatten_dict
from flax.core import freeze, unfreeze
from typing import Union, Optional, Tuple

from ..attention_module import AttentionModule
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel
from flax.linen import partitioning as nn_partitioning, combine_masks
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput

from fjformer import linen as nn
from ..flax_modelling_utils import (
    ACT2FN,
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
    block_wise_ffn
)
import chex
from .openelm_configuration import OpenELMConfig, make_divisible

re_mat = nn_partitioning.remat


class OpenELMRMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)

        weight = fjformer.linen.linen.control_quantization(self.weight, self.dtype)
        return output * weight


class FlaxOpenELMRotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, key, query, freq_cis, position_ids):
        sin, cos = freq_cis

        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]
        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(
            key,
            sin[..., :key_len, :],
            cos[..., :key_len, :]
        )
        query = apply_rotary_pos_emb(
            query,
            sin[..., key_len - query_len: key_len, :],
            cos[..., key_len - query_len: key_len, :]
        )

        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxOpenELMMLP(nn.Module):
    config: OpenELMConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")


class FlaxOpenELMMultiHeadCausalAttention(BaseJAXAttentionModule):
    config: OpenELMConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        layer_idx = self.layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            (q_heads + k_heads + v_heads) * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                dim=config.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )
            self.k_norm = OpenELMRMSNorm(
                dim=config.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            config.model_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.head_dim = head_dim
        self.rotary = FlaxOpenELMRotaryEmbedding(self.dtype)
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
            num_attention_heads=q_heads,
            attention_dropout=0.0,
            head_dims=head_dim,
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

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_q_heads * self.head_dim,))

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
            self.num_q_heads,
            self.head_dim
        )
        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_k_heads,
            self.head_dim
        )
        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_dim
        )

        query, key, value = self._transpose_sequence_head(query, key, value)
        query, key = self.rotary(position_ids=position_ids, query=query, key=key, freq_cis=freq_cis)
        key = repeat_kv_bnsh(key, self.num_groups)
        value = repeat_kv_bnsh(value, self.num_groups)
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
        output_attentions = False

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose(0, 2, 1, 3)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        query_states = qkv[:, :self.num_q_heads, :, :]
        key_states = qkv[:, self.num_q_heads:self.num_k_heads + self.num_q_heads, :, :]
        value_states = qkv[:, self.num_k_heads + self.num_q_heads:, :, :]
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        query_states, key_states, value_states = map(
            lambda x: x.transpose(0, 2, 1, 3),
            [query_states, key_states, value_states]
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

        assert_msg = (
            "num_attention_heads repeat wont work likely\n"
            f"INFO :\n\trepeat_kv_bnsh Used with num_key_value_groups = {self.num_groups}\n\t"
            f"NH : {self.num_q_heads} KVH : {self.num_k_heads}"
        )

        assert query_states.shape[-2] == self.num_q_heads, assert_msg
        assert key_states.shape[-2] == self.num_q_heads, assert_msg
        assert value_states.shape[-2] == self.num_q_heads, assert_msg

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask, (0, 0, mask_shift, 0), (1, 1,
                                                     query_length, max_decoder_length)
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:])
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = jnp.broadcast_to(
            attention_mask, causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

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

        outputs = (
            attn_output, attentions.attention_weights
        ) if output_attentions else (
            attn_output, None
        )
        return outputs


class FlaxOpenELMFeedForwardNetwork(nn.Module):
    config: OpenELMConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config
        layer_idx = self.layer_idx
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,  # type:ignore
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nn.Linear(
                2 * intermediate_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
            )
            self.proj_2 = nn.Linear(
                config.model_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
            )
            self.ffn_with_glu = True
        else:
            self.proj_1 = nn.Linear(
                intermediate_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
            )
            self.proj_2 = nn.Linear(
                config.model_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def __call__(self, x: chex.Array, e: bool = False) -> chex.Array:
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = jnp.split(y_12, 2, axis=-1)
            return self.proj_2(self.act(y_1) * y_2)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class FlaxOpenELMDecoderLayer(nn.Module):
    config: OpenELMConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:

        attn_block = FlaxOpenELMMultiHeadCausalAttention
        mlp_block = FlaxOpenELMFeedForwardNetwork
        if self.config.gradient_checkpointing != "":
            # hidden_states: chex.Array,
            # freq_cis: Tuple[chex.Array, chex.Array],
            # attention_mask: chex.Array,
            # position_ids: chex.Array,
            # causal_mask: chex.Array,
            # segment_ids: Optional[chex.Array] = None,
            # deterministic: bool = True,
            # init_cache: bool = False,
            # output_attentions: bool = False,
            # fcm_mask = None,
            attn_block = re_mat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1, 3, 4, 6, 7, 8)
            )
            mlp_block = re_mat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )

        self.attn = attn_block(
            config=self.config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ffn = mlp_block(
            config=self.config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ffn_norm = OpenELMRMSNorm(
            self.config.model_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.attn_norm = OpenELMRMSNorm(
            self.config.model_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            causal_mask: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            segment_ids: Optional[chex.Array] = None,
            output_attentions: Optional[bool] = False,
            init_cache: Optional[bool] = False,
            deterministic: bool = True
    ) -> Tuple[chex.Array, Optional[chex.Array]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention

        # hidden_states: chex.Array,
        # freq_cis: Tuple[chex.Array, chex.Array],
        # attention_mask: chex.Array,
        # position_ids: chex.Array,
        # causal_mask: chex.Array,
        # segment_ids: Optional[chex.Array] = None,
        # deterministic: bool = True,
        # init_cache: bool = False,
        # output_attentions: bool = False,
        # fcm_mask = None,
        hidden_states, self_attn_weights = self.attn(
            hidden_states,
            freq_cis,
            attention_mask,
            position_ids,
            causal_mask,
            segment_ids,
            deterministic,
            init_cache,
            output_attentions,
            None
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.ffn,
                hidden_states,
                self.config.scan_mlp_chunk_size,
                deterministic,
            )
        else:
            feed_forward_hidden_states = self.ffn(
                hidden_states,
                deterministic,
            )
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs  # type:ignore


class FlaxOpenELMDecoderLayerCollection(nn.Module):
    config: OpenELMConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.layers = [
            FlaxOpenELMDecoderLayer(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                layer_idx=i,
                name=str(i)
            ) for i in range(self.config.num_transformer_layers)
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            attention_mask: chex.Array,
            causal_mask: chex.Array,
            position_ids: chex.Array,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            output = layer(
                hidden_states=hidden_states,
                freq_cis=freq_cis,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                output_attentions=output_attentions,
                init_cache=init_cache,
                segment_ids=None,
                deterministic=deterministic,
                position_ids=position_ids,
            )
            hidden_states = output[0]

            if output_attentions:
                output_attentions += (output[1],)

        return hidden_states, all_hidden_states, all_attentions


class FlaxOpenELMModule(nn.Module):
    config: OpenELMConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        config = self.config
        self.token_embeddings = nn.Embed(
            config.vocab_size,
            config.model_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

        self.layers = FlaxOpenELMDecoderLayerCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm = OpenELMRMSNorm(
            config.model_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Linear(
                config.vocab_size,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
            )
        self.num_transformer_layers = config.num_transformer_layers

        initial_rope_kwargs = dict(
            rope_type="none"
        )
        if self.config.rope_scaling is not None:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            initial_rope_kwargs = dict(
                scaling_factor=scaling_factor,
                rope_type=scaling_type
            )
        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=(
                getattr(self.config, "freq_max_position_embeddings", self.config.rope_max_length)
            ),
            dim=self.config.head_dim,
            base=self.config.rope_freq_constant,
            **initial_rope_kwargs
        )
        self.causal_mask = flax.linen.make_causal_mask(
            jnp.ones(
                (1, getattr(self.config, "c_max_position_embeddings", self.config.max_context_length)),
                dtype="bool"
            ), dtype="bool"
        )

    def __call__(
            self,
            input_ids: Optional[chex.Array] = None,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            deterministic: bool = True,
            inputs_embeds: chex.Array = None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ) -> typing.Union[Tuple[Array, ...], FlaxBaseModelOutput]:
        """
        The __call__ function is the main function of a Flax model.
        It takes in input_ids, attention_mask, and position_ids as inputs to the model.
        The output is a tuple containing: last hidden state (hidden states), all hidden states (if output_hidden_states=True), attentions (if output attentions=True).


        :param self: Represent the instance of the class
        :param input_ids: chex.Array: Pass in the input ids
        :param attention_mask: chex.Array: Mask out the attention weights for certain tokens
        :param position_ids: chex.Array: Determine the position of each token in a sequence
        :param deterministic: bool: Determine whether to use dropout or not
        :param inputs_embeds: chex.Array: Pass in the embedding of the input_ids
        :param init_cache: bool: Initialize the cache for the decoder
        :param output_attentions: bool: Determine whether to return the attention weights or not
        :param output_hidden_states: bool: Return all hidden states or just the last one
        :param return_dict: bool: Return a dictionary of the outputs or not
        :param : Determine whether the model is in training mode or not
        :return: A tuple of the hidden states, all hidden states, and attentions

        """
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids.astype("i4"))
        if attention_mask.ndim == 2:
            b, s = attention_mask.shape
            attention_mask = attention_mask.reshape(b, 1, 1, s)

        outputs = self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            freq_cis=self.freq_cis,
            init_cache=init_cache,
            output_attentions=output_attentions,
            deterministic=deterministic,
            causal_mask=self.causal_mask,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(value for value in outputs if value is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxOpenELMPretrainedModel(EasyDeLFlaxPretrainedModel):
    config_class = OpenELMConfig
    base_model_prefix = "openelm"
    module_class: nn.Module = None

    def __init__(self,
                 config: OpenELMConfig,
                 input_shape: Tuple = (1, 1),
                 seed: int = 0,
                 dtype: jnp.dtype = jnp.bfloat16,
                 param_dtype: jnp.dtype = jnp.bfloat16,
                 _do_init: bool = True,
                 **kwargs
                 ):
        super().__init__(
            config,
            self.module_class(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                **kwargs
            ),
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init
        )

    def init_weights(
            self,
            rng: jax.random.PRNGKey,
            input_shape: Tuple,
            params: flax.core.FrozenDict = None
    ) -> flax.core.FrozenDict:
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
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
            input_shape
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rng_s = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(
                input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rng_s,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rng_s, input_ids, attention_mask, position_ids, return_dict=False
            )

        random_params = module_init_outputs["params"]

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
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True
        )
        return init_variables["cache"]

    def __call__(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
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
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            None,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
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


class FlaxOpenELMModel(FlaxOpenELMPretrainedModel):
    config_class = OpenELMConfig
    module_class = FlaxOpenELMModule


class FlaxOpenELMForCausalLMModule(nn.Module):
    config: OpenELMConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.transformer: FlaxOpenELMModule = FlaxOpenELMModule(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.lm_head = nn.Linear(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            deterministic: bool = True,
            inputs_embeds: chex.Array = None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        """
            The __call__ function is the main function of a Flax module. It defines how the model will be called,
            and what it returns. In this case, we are calling our Transformer model with input_ids and attention_mask
            as inputs (these are defined in __init__). We also have some optional arguments that can be passed to
            the call function: deterministic (whether to use dropout), inputs_embeds (if you want to pass your own embeddings),
            output_attentions and output_hidden states which return additional outputs from the transformer layers if set True. Finally,

            :param self: Refer to the object itself
            :param input_ids: chex.Array: Pass in the input tokens
            :param attention_mask: chex.Array: Mask out the padding tokens
            :param position_ids: chex.Array: Specify the position of each token in the sequence
            :param deterministic: bool: Determine whether to use dropout in the model
            :param inputs_embeds: chex.Array: Pass in the embeddings of the input tokens
            :param init_cache: bool: Initialize the cache for the decoder
            :param output_attentions: bool: Return the attention weights
            :param output_hidden_states: bool: Return the hidden states of all layers
            :param return_dict: bool: Return a dictionary of the outputs or just the logits
            :param : Determine whether to return the logits or not
            :return: A tuple of (lm_logits, hidden_states, attentions)

        """
        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            inputs_embeds=inputs_embeds,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        if self.config.share_input_output_layers:
            shared_kernel = self.transformer.variables["params"]["token_embeddings"]["embedding"]
            shared_kernel = fjformer.linen.linen.control_quantization(shared_kernel, self.param_dtype).T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits = lm_logits[:, : self.config.vocab_size]
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxOpenELMForCausalLM(FlaxOpenELMPretrainedModel):
    module_class = FlaxOpenELMForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones(
            (batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[
                                            None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):  # noqa:E722 # type:ignore
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
