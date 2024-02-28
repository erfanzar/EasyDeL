import math
from typing import Optional, Tuple, Union, Any

import chex
import flax.linen as nn
import flax.linen.partitioning
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput, \
    FlaxMaskedLMOutput

from .stablelm_configuration import StableLmConfig
from ..easy_attention import EasyAttention
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

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class FlaxStableLmMLP(nn.Module):
    config: StableLmConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config

        self.gate_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.down_proj = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.up_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        The __call__ function is the main function of a class.
        It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
        The __call__ method enables instances of a class to be called like standard Python functions.

        :param self: Represent the instance of the class
        :param x: jnp.ndarray: Pass in the input to the layer
        :param deterministic: bool: Determine whether to use dropout # Ignored
        :return: A tensor that is the result of function to x

        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FlaxStableLmAttention(BaseJAXAttentionModule):
    config: StableLmConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config: StableLmConfig = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.use_qkv_bias,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.k_proj = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.use_qkv_bias,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.v_proj = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.use_qkv_bias,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.o_proj = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

        self.rotary_emb_dim = int(self.config.partial_rotary_factor * self.head_dim)
        self.attention_performer = EasyAttention(
            attn_type="normal",
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
            use_shard_map=self.config.use_shard_map,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.dtype,
            bias_partition_spec=self.config.bias_partition_spec,
            key_partition_spec=self.config.key_partition_spec,
            query_partition_spec=self.config.query_partition_spec,
            value_partition_spec=self.config.value_partition_spec,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.jax_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
        )

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
        :param batch_size: Reshape the query_states, key and value tensors
        :param sequence_length: Reshape the query_states, key and value tensors
        :param query: Calculate the attention weights
        :param key: Calculate the attention
        :param value: Compute the attention weights
        :param freq_cis: Calculate the frequency of each word in the vocabulary
        :param position_ids: Identify the position of each token in the sequence
        :return: A tuple of 3 tensors: query_states, key and value

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

        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        query_rot, query_pass = (
            query[..., : self.rotary_emb_dim],
            query[..., self.rotary_emb_dim:],
        )
        key_rot, key_pass = (
            key[..., : self.rotary_emb_dim],
            key[..., self.rotary_emb_dim:],
        )

        key_rot = apply_rotary_pos_emb(key_rot, sin, cos)
        query_rot = apply_rotary_pos_emb(query_rot, sin, cos)

        query = jnp.concatenate((query_rot, query_pass), axis=-1)
        key = jnp.concatenate((key_rot, key_pass), axis=-1)

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
        query_state, key_state, value_state = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(
            hidden_states)

        if self.config.use_pjit_attention_force:
            query_state = with_sharding_constraint(
                query_state, PartitionSpec(("dp", "fsdp"), "sp", "tp"))
            key_state = with_sharding_constraint(
                key_state, PartitionSpec(("dp", "fsdp"), "sp", "tp"))
            value_state = with_sharding_constraint(
                value_state, PartitionSpec(("dp", "fsdp"), "sp", "tp"))

        query_state = query_state.reshape(
            batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key_state = key_state.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        value_state = value_state.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)

        query_state, key_state, value_state = self.apply_rotary(
            query=query_state,
            key=key_state,
            value=value_state,
            position_ids=position_ids,
            freq_cis=freq_cis,
            batch_size=batch_size,
            sequence_length=sequence_length
        )

        assert_msg = (
            "num_attention_heads repeat wont work likely\n"
            f"INFO :\n\trepeat_kv_bnsh Used with num_key_value_groups = {self.num_key_value_groups}\n\t"
            f"NH : {self.config.num_attention_heads} KVH : {self.config.num_attention_heads}"
        )

        assert query_state.shape[-2] == self.config.num_attention_heads, assert_msg
        assert key_state.shape[-2] == self.config.num_attention_heads, assert_msg
        assert value_state.shape[-2] == self.config.num_attention_heads, assert_msg

        query_length, key_length = query_state.shape[1], key_state.shape[1]

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
            key_state, value_state, attention_mask = self._concatenate_to_cache(
                key_state,
                value_state,
                query_state,
                attention_mask
            )

        use_qkv_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(
                self.dtype).min).astype(self.dtype),
        )

        query_length, key_length = query_state.shape[1], key_state.shape[1]

        attentions = self.attention_performer.__call__(
            query_states=query_state,
            key_states=key_state,
            value_states=value_state,
            bias=use_qkv_bias,
            causal=False,
            use_pjit_attention_force=self.config.use_pjit_attention_force,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
        )
        attentions.attention_outputs = attentions.attention_outputs

        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.o_proj(attn_output)
        outputs = (attn_output, attentions.attention_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxStableLmDecoderLayer(nn.Module):
    config: StableLmConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        attn_block = FlaxStableLmAttention
        mlp_block = FlaxStableLmMLP

        if self.config.gradient_checkpointing != "":
            mlp_block = flax.linen.partitioning.remat(
                mlp_block,
                static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )
            # hidden_states: chex.Array,
            # freq_cis: chex.Array,
            # attention_mask: chex.Array,
            # position_ids: chex.Array,
            # causal_mask: chex.Array,
            # deterministic: bool = True,
            # init_cache: bool = False,
            # output_attentions: bool = False,
            # fcm_mask=None,
            attn_block = flax.linen.partitioning.remat(
                attn_block,
                static_argnums=(2, 5, 6, 7, 8),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )
        self.self_attn = attn_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.mlp = mlp_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.input_layernorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout)

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: chex.Array,
            attention_mask: Optional[chex.Array],
            position_ids: Optional[chex.Array],
            causal_mask: Optional[chex.Array],
            deterministic: bool = True,
            output_attentions: bool = False,
            init_cache: bool = False,
    ):
        residual = hidden_states
        attn_out = self.self_attn(
            self.input_layernorm(hidden_states),
            freq_cis,
            attention_mask,
            position_ids,
            causal_mask,
            deterministic,
            init_cache,
            output_attentions,
            None
        )
        hidden_states, self_attn_weights = (attn_out[0], attn_out[1]) if len(attn_out) == 2 else (attn_out[0], None)

        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.config.use_scan_mlp:
            hidden_states = block_wise_ffn(
                self.mlp,
                self.post_attention_layernorm(hidden_states),
                self.config.scan_mlp_chunk_size,
                deterministic
            )
        else:
            hidden_states = self.mlp(
                self.post_attention_layernorm(hidden_states),
                deterministic,
            )
        hidden_states = self.dropout(
            hidden_states, deterministic=deterministic
        )
        hidden_states = hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class FlaxStableLmDecoderLayerCollection(nn.Module):
    config: StableLmConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.layers = [
            FlaxStableLmDecoderLayer(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(idx),
            )
            for idx in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            freq_cis: chex.Array,
            attention_mask: Optional[chex.Array],
            position_ids: Optional[chex.Array],
            causal_mask: Optional[chex.Array],
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            init_cache: bool = False,
            return_dict: bool = True,
    ) -> tuple[tuple, ...] | FlaxBaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # hidden_states: chex.Array,
            # freq_cis: chex.Array,
            # attention_mask: Optional[chex.Array],
            # position_ids: Optional[chex.Array],
            # causal_mask: Optional[chex.Array],
            # deterministic: bool = True,
            # output_attentions: bool = False,
            # init_cache: bool = False,

            layer_outputs = decoder_layer(
                hidden_states,
                freq_cis,
                attention_mask,
                position_ids,
                causal_mask,
                deterministic,
                output_attentions,
                init_cache,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class FlaxStableLmModule(nn.Module):
    config: StableLmConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        config = self.config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.layers = FlaxStableLmDecoderLayerCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm = nn.LayerNorm(
            epsilon=config.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.causal_mask = nn.make_causal_mask(
            jnp.ones(
                (1, getattr(self.config, "c_max_position_embeddings", self.config.max_position_embeddings)),
                dtype="bool"
            ), dtype="bool"
        )

        initial_rope_kwargs = dict(
            rope_type="none"
        )
        if hasattr(config, "rope_scaling"):
            if config.rope_scaling is not None:
                scaling_type = config.rope_scaling["type"]
                scaling_factor = config.rope_scaling["factor"]
                initial_rope_kwargs = dict(
                    scaling_factor=scaling_factor,
                    rope_type=scaling_type
                )
        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=getattr(
                self.config,
                "freq_max_position_embeddings",
                self.config.max_position_embeddings
            ),
            dim=int(config.partial_rotary_factor * (config.hidden_size // config.num_attention_heads)),
            # dim=config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
            **initial_rope_kwargs
        )

    def __call__(
            self,
            input_ids: Optional[chex.Array] = None,
            inputs_embeds: Optional[chex.Array] = None,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            extra_embedding: Optional[chex.Array] = None,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            init_cache: bool = False,
            return_dict: bool = True,
    ) -> tuple[tuple[Any, ...], ...] | FlaxBaseModelOutput:
        if input_ids is None and inputs_embeds is None:
            raise RuntimeError("Both `input_ids` and `inputs_embeds` can not be None !")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        batch_size, sequence_length, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), dtype="i4")
        if position_ids is None:
            position_ids = (jnp.cumsum(attention_mask) - 1).reshape(batch_size, sequence_length).astype("i4")
        assert sequence_length <= self.config.max_position_embeddings, "Maximum Position Embedding Reached !"
        inputs_embeds = inputs_embeds + extra_embedding if extra_embedding is not None else inputs_embeds

        outputs = self.layers(
            hidden_states=inputs_embeds,
            freq_cis=self.freq_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            causal_mask=self.causal_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1] if output_hidden_states else None,
            attentions=outputs[-1] if output_attentions else None,
        )


class FlaxStableLmForCausalLMModule(nn.Module):
    config: StableLmConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        self.model = FlaxStableLmModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids: Optional[chex.Array] = None,
            inputs_embeds: Optional[chex.Array] = None,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            extra_embedding: Optional[chex.Array] = None,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            init_cache: bool = False,
            return_dict: bool = True,
    ) -> tuple[Any, ...] | FlaxMaskedLMOutput:
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            extra_embedding=extra_embedding,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        outputs = (res.last_hidden_state, res.hidden_states, res.attentions)
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"].T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, res.last_hidden_state)
        else:
            lm_logits = self.lm_head(res.last_hidden_state)

        lm_logits = lm_logits.astype(jnp.float32)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=res.hidden_states, attentions=res.attentions)


class FlaxStableLmPreTrainedModel(EasyDelFlaxPretrainedModel):
    """StableLm pre-trained model."""
    module_class = None
    config_class = StableLmConfig
    base_model_prefix = "model"

    def __init__(
            self,
            config: StableLmConfig,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            input_shape=(1, 1),
            seed: int = 42,
            _do_init: bool = False
    ) -> None:
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision
        )
        super().__init__(
            config=config,
            module=module,
            input_shape=input_shape,
            _do_init=_do_init,
            seed=seed
        )

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(
            jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, input_ids, attention_mask)

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

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array = None,
            position_ids: chex.Array = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
            add_params_field: bool = False,
            **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        assert sequence_length <= self.config.max_position_embeddings, "Maximum Position Embedding Reached !"

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if self.config.bits is not None:
            rngs['params'] = jax.random.key(0)

        inputs = {"params": params or self.params} if add_params_field else params or self.params

        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_ids=input_ids,
            inputs_embeds=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            extra_embedding=extra_embedding,
            deterministic=not train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            init_cache=False,
            return_dict=return_dict,
            rngs=rngs,
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


class FlaxStableLmModel(FlaxStableLmPreTrainedModel):
    module_class = FlaxStableLmModule


class FlaxStableLmForCausalLM(FlaxStableLmPreTrainedModel):
    module_class = FlaxStableLmForCausalLMModule
