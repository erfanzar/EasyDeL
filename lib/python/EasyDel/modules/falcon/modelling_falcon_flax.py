import math

import flax.linen.partitioning
from flax import linen as nn
from flax.core import FrozenDict, unfreeze, freeze
from typing import Optional, Dict, Union, Tuple

from flax.linen import combine_masks
from flax.traverse_util import unflatten_dict, flatten_dict
from jax import numpy as jnp, lax
import jax

from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutput
from ..flax_modelling_utils import (
    get_gradient_checkpoint_policy,
    with_sharding_constraint,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
    block_wise_ffn
)
import chex
from fjformer.func import transpose
from .falcon_configuration import FalconConfig
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel


def built_bloom_alibi(attention_mask, num_attention_heads):
    """
    The built_bloom_alibi function is used to create a bloom alibi for the attention mask.
    The bloom alibi is used in the Bloom Attention layer to ensure that each token has a unique
    attention vector, even if it's masked out. This ensures that all tokens have an equal chance of being selected as
    the most important token in the sequence, which helps with training stability and performance.

    :param attention_mask: Mask out the padding tokens in the input sequence
    :param num_attention_heads: Determine the number of attention heads in the model
    :return: A tensor of shape (batch_size, num_attention_heads, 1, sequence_length)
    
    """
    batch_size, sequence_length = attention_mask.shape
    cp2 = 2 ** math.floor(math.log2(num_attention_heads))
    base = jnp.asarray(
        2 ** (- (2 ** -(math.log2(cp2) - 3))), dtype=jnp.float32
    )
    powers = jnp.arange(1, 1 + cp2, dtype=jnp.float32)
    slops = jnp.power(base, powers)
    if cp2 != num_attention_heads:
        extra_base = jnp.asarray(
            2 ** (-(2 ** -(math.log2(2 * cp2) - 3))), dtype=jnp.float32
        )
        num_rem_heads = min(cp2, num_attention_heads - cp2)
        extra_power = jnp.arange(1, 1 + 2 * num_rem_heads, 2, dtype=jnp.dtype)
        slops = jnp.concatenate([slops, jnp.power(extra_base, extra_power)], axis=0)
    arange_tensor = (((jnp.cumsum(attention_mask, axis=-1)) - 1) * attention_mask)[:, jnp.newaxis, :]
    alibi = slops[..., jnp.newaxis].astype(jnp.bfloat16) * arange_tensor
    return alibi.reshape(batch_size, num_attention_heads, 1, sequence_length)


def precompute_falcon_freq_cis(max_position_embedding: int, head_dim: int, theta: float = 10000):
    """
    The precompute_falcon_freq_cis function is used to precompute the sinusoidal frequencies for the FALCON model.
    The function takes in three arguments: max_position_embedding, head_dim, and theta. The first two are self-explanatory;
    the third is a hyperparameter that controls how quickly the frequency increases with position (i.e., how many times
    higher it will be at position i than at position 0). The default value of 10000 was chosen because it worked well on
    the tasks we tested.

    :param max_position_embedding: int: Set the maximum length of the sequence
    :param head_dim: int: Determine the size of the positional embedding
    :param theta: float: Adjust the frequency of the sinusoid
    :return: A tuple of two arrays
    
    """
    inv_freq_cis = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freq = jnp.einsum("i , j -> i j", jnp.arange(max_position_embedding), inv_freq_cis).astype("float32")

    embed = jnp.concatenate((freq, freq), axis=-1)
    return jnp.sin(embed)[:, :], jnp.cos(embed)[:, :]


def _rotate_half(x):
    """
    The _rotate_half function takes a 1D array and rotates it by half its length.
    For example, if the input is [0, 1, 2, 3], then the output will be [-2,-3,-0,-4].
    This function is used to rotate the Fourier transform of an image so that its zero-frequency component
    is in the center of the spectrum.

    :param x: Specify the input array
    :return: The negative of the second half of x concatenated with the first half
    
    """
    return jnp.concatenate((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), axis=-1)


def apply_rotary_pos_embedding(tensor, sin_, cos_):
    """
    The apply_rotary_pos_embedding function applies a rotary positional embedding to the input tensor.

    :param tensor: Pass in the tensor that we want to apply the positional embedding to
    :param sin_: Rotate the tensor by half of its length
    :param cos_: Multiply the tensor and cosine of the angle
    :return: A tensor with the same shape as its input,
    
    """
    return (tensor * cos_) + (_rotate_half(tensor) * sin_)


def dropout_add(linen_drop: nn.Dropout, x: chex.Array, residual: chex.Array, deterministic: bool) -> chex.Array:
    """
    The dropout_add function is a helper function that adds the residual to the output of
    the dropout layer. This is necessary because we want to use deterministic=True when
    we are evaluating our model, but we still need to add in the residual. The reason for this
    is that during training, we have two paths through our network: one with dropout and one without.
    The path without dropout (residual) allows us to backpropagate gradients through both paths at once.

    :param linen_drop: nn.Dropout: Specify the dropout layer
    :param x: chex.Array: Pass in the input to the dropout layer
    :param residual: chex.Array: Add the residual to the output of dropout_add
    :param deterministic: bool: Determine whether the dropout layer is active or not
    :return: A tensor that is the sum of the residual and a dropout layer
    
    """
    out = linen_drop(inputs=x, deterministic=deterministic)
    out = residual + out
    return out


class FlaxFalconRotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, query, key, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, :]
        cos = cos[position_ids][:, :]

        query = apply_rotary_pos_embedding(query, sin, cos)
        key = apply_rotary_pos_embedding(key, sin, cos)
        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxFalconAttention(BaseJAXAttentionModule):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.query_key_value = nn.Dense(
            features=3 * self.config.hidden_size if not self.config.multi_query else (
                    self.config.hidden_size + 2 * head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.inv_norm_factor = 1 / math.sqrt(head_dim)
        self.dense = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.head_dim = head_dim
        self.maybe_rotary = FlaxFalconRotaryEmbedding(self.dtype) if not self.config.alibi else lambda q, k, a, s: (
            q, k)
        assert self.head_dim * self.config.num_attention_heads == self.config.hidden_size
        self.num_kv_heads = self.config.num_kv_heads if (
                self.config.new_decoder_architecture or not self.config.multi_query
        ) else 1
        self.num_heads = self.config.num_attention_heads

    @staticmethod
    def _transpose_sequence_head(query, key, value):
        return jnp.transpose(query, (0, 2, 1, 3)), jnp.transpose(key, (0, 2, 1, 3)), jnp.transpose(value, (0, 2, 1, 3))

    def split_head(self, qkv: chex.Array):
        batch_size, sequence_length, _ = qkv.shape

        if self.config.new_decoder_architecture:
            batch, sequence_length, _ = qkv.shape
            qkv = qkv.reshape(batch, sequence_length, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query_state = qkv[:, :, :, :-2]
            key_state = qkv[:, :, :, [-2]]
            value_state = qkv[:, :, :, [-1]]
            key_state = jnp.broadcast_to(key_state, query_state.shape)
            value_state = jnp.broadcast_to(value_state, query_state.shape)

            query_state, key_state, value_state = [x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],)) for x in
                                                   (query_state, key_state, value_state)]
            if self.config.use_pjit_attention_force:
                query_state = with_sharding_constraint(query_state, PartitionSpec(("dp", "fsdp"), None, "sp"))
                key_state = with_sharding_constraint(key_state, PartitionSpec(("dp", "fsdp"), None, "sp"))
                value_state = with_sharding_constraint(value_state, PartitionSpec(("dp", "fsdp"), None, "sp"))
            return query_state, key_state, value_state
        if self.config.multi_query:
            qkv = qkv.reshape(
                batch_size, sequence_length, self.config.num_attention_heads + 2, -1
            )
            query_state, key_state, value_state = qkv[..., :-2, :], qkv[..., [-2], :], qkv[..., [-1], :]

        else:
            query_state, key_state, value_state = jnp.split(qkv, 3, -1)

        if self.config.use_pjit_attention_force:
            query_state = with_sharding_constraint(query_state, PartitionSpec(("dp", "fsdp"), None, "sp"))
            key_state = with_sharding_constraint(key_state, PartitionSpec(("dp", "fsdp"), None, "sp"))
            value_state = with_sharding_constraint(value_state, PartitionSpec(("dp", "fsdp"), None, "sp"))
        return query_state, key_state, value_state

    def _merge_heads(self, x: chex.Array) -> chex.Array:

        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.reshape(batch_size, self.config.num_attention_heads, seq_length, self.head_dim)

        # x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.config.num_attention_heads * self.head_dim)

    def __call__(
            self,
            hidden_states: chex.Array,
            attention_mask: chex.Array,
            position_ids: chex.Array,
            causal_mask: chex.Array = None,
            alibi: chex.Array = None,
            freq_cis: Tuple[chex.Array, chex.Array] = None,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        # FLAX
        batch_size, sequence_length, _ = hidden_states.shape
        num_kv_heads = self.num_heads if self.config.new_decoder_architecture else self.num_kv_heads
        qkv = self.query_key_value(hidden_states)

        query_layer, key_layer, value_layer = self.split_head(qkv)
        query_length, key_length = query_layer.shape[1], key_layer.shape[1]
        query_layer = transpose(
            query_layer, 1, 2
        ).reshape(
            batch_size,
            self.config.num_attention_heads,
            query_length,
            self.head_dim
        )
        key_layer = transpose(
            key_layer, 1, 2
        ).reshape(
            batch_size,
            num_kv_heads,
            key_length,
            self.head_dim,
        )
        value_layer = transpose(
            value_layer, 1, 2
        ).reshape(
            batch_size,
            num_kv_heads,
            key_length,
            self.head_dim
        )
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask, (
                    0,
                    0,
                    mask_shift,
                    0
                ), (
                    1,
                    1,
                    query_length,
                    max_decoder_length
                )
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = combine_masks(attention_mask, causal_mask)

        if not self.config.alibi:
            query_layer, key_layer = self.maybe_rotary(
                position_ids=position_ids,
                query=query_layer,
                key=key_layer,
                freq_cis=freq_cis
            )
        if self.has_variable("cache", "cached_key") or init_cache:
            key_layer, value_layer, attention_mask = self._concatenate_to_cache(
                key_layer,
                value_layer,
                query_layer,
                attention_mask
            )
        float_min = jnp.finfo(query_layer.dtype).min
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, float_min).astype(self.dtype),
        )
        dtype = jnp.promote_types(key_layer.dtype, jnp.float32)

        (
            query_layer,
            key_layer,
            value_layer,
            attention_bias
        ) = map(
            lambda x: x.astype(dtype=dtype), (
                query_layer,
                key_layer,
                value_layer,
                attention_bias
            )
        )

        attention_scores = jnp.einsum(
            "...qhd,...khd->...hqk",
            query_layer.transpose(0, 2, 1, 3),
            key_layer.transpose(0, 2, 1, 3),
            precision=self.precision
        )
        if alibi is None:

            attention_scores /= math.sqrt(self.head_dim)

            attention_scores = jax.nn.softmax(
                attention_scores + attention_bias, axis=-1
            )
            attn_output = jnp.einsum(
                "...hqk,...khd->...qhd",
                attention_scores,
                value_layer.transpose(0, 2, 1, 3),
                precision=self.precision
            )
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads, self.head_dim)
            # attn_output = transpose(attn_output, 2, 1)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, attention_scores
            else:
                return output_tensor,
        else:

            attention_scores = attention_scores.reshape(batch_size, self.num_heads, query_length, key_length)
            attention_scores = attention_scores + alibi.reshape(batch_size, self.num_heads, 1, -1)
            attention_scores *= self.inv_norm_factor
            attention_scores = jax.nn.softmax(
                attention_scores + attention_bias, axis=-1
            )
            attention_scores = attention_scores.reshape(batch_size, self.num_heads, query_length, key_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]

            attn_output = jax.lax.batch_matmul(attention_scores, value_layer)
            attn_output = attn_output.reshape((attn_output.shape[1] * attn_output.shape[0],) + attn_output.shape[2:])
            attn_output = self._merge_heads(attn_output)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, attention_scores
            else:
                return output_tensor,


class FlaxFalconMlp(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.dense_h_to_4h = nn.Dense(
            features=self.config.hidden_size * 4,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.dense_4h_to_h = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.bias,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(self, x: chex.Array, deterministic: bool = True):
        return self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x)))


class FlaxFalconBlock(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config
        self.input_layernorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                            dtype=self.dtype)
        if not config.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                                         dtype=self.dtype)
        if config.new_decoder_architecture:
            self.ln_attn = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                        dtype=self.dtype)
            self.ln_mlp = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                                       dtype=self.dtype)

        # hidden_states: chex.Array
        # attention_mask: chex.Array
        # position_ids: chex.Array
        # causal_mask: chex.Array = None
        # alibi: chex.Array = None
        # freq_cis: Tuple[chex.Array, chex.Array] = None
        # init_cache: bool = False
        # output_attentions: bool = False

        attn_block = FlaxFalconAttention
        mlp_block = FlaxFalconMlp
        if self.config.gradient_checkpointing != "":
            attn_block = flax.linen.partitioning.remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(3, 5, 6, 7)
            )

            mlp_block = flax.linen.partitioning.remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )

        self.mlp = mlp_block(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.self_attention = attn_block(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.dropout = nn.Dropout(self.config.attention_dropout)
        self.dropout_mlp = nn.Dropout(self.config.hidden_dropout)

    def __call__(
            self,
            hidden_states: chex.Array,
            alibi: chex.Array,
            attention_mask: chex.Array,
            freq_cis: Tuple[chex.Array, chex.Array],
            position_ids: chex.Array,
            causal_mask: chex.Array,
            init_cache: bool = False,
            output_attentions: bool = False,
            deterministic: bool = True
    ):
        residual = hidden_states
        mlp_layernorm_out = None
        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.

        # hidden_states: chex.Array
        # attention_mask: chex.Array
        # position_ids: chex.Array
        # causal_mask: chex.Array = None
        # alibi: chex.Array = None
        # freq_cis: Tuple[chex.Array, chex.Array] = None
        # init_cache: bool = False
        # output_attentions: bool = False

        attn_outputs = self.self_attention(
            attention_layernorm_out,
            attention_mask,
            position_ids,
            causal_mask,
            alibi,
            freq_cis,
            init_cache,
            output_attentions,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    linen_drop=self.dropout,
                    x=attention_output,
                    residual=residual,
                    deterministic=deterministic
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]
        if self.config.use_scan_mlp:
            mlp_output = block_wise_ffn(
                self.mlp,
                mlp_layernorm_out,
                self.config.scan_mlp_chunk_size,
                deterministic,
            )
        else:
            mlp_output = self.mlp(
                mlp_layernorm_out,
                deterministic,
            )

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(
            linen_drop=self.dropout_mlp,
            x=mlp_output,
            residual=residual,
            deterministic=deterministic

        )

        return output, outputs


class FlaxFalconCollection(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        block = FlaxFalconBlock
        self.layers = [
            block(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            )
            for i in range(
                self.config.num_hidden_layers
            )
        ]

    def __call__(self,
                 hidden_states: chex.Array,
                 attention_mask: chex.Array,
                 alibi: chex.Array,
                 freq_cis: Tuple[chex.Array, chex.Array],
                 position_ids: chex.Array,
                 causal_mask: chex.Array,
                 output_attentions: bool = False,
                 init_cache: bool = False,
                 deterministic: bool = True
                 ):
        for layer in self.layers:
            # hidden_states: chex.Array
            # alibi: chex.Array
            # attention_mask: chex.Array
            # freq_cis: Tuple[chex.Array, chex.Array]
            # position_ids: chex.Array
            # causal_mask: chex.Array
            # init_cache: bool = False
            # output_attentions: bool = False
            # deterministic: bool = True

            out = layer(
                hidden_states,
                alibi,
                attention_mask,
                freq_cis,
                position_ids,
                causal_mask,
                init_cache,
                output_attentions if self.config.gradient_checkpointing == "" else False,
                deterministic,
            )
            hidden_states = out[0]
        return hidden_states


class FlaxFalconModule(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.word_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.h = FlaxFalconCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ln_f = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, epsilon=self.config.layer_norm_epsilon)
        self.causal_mask = nn.make_causal_mask(
            jnp.ones(
                (1, getattr(self.config, "c_max_position_embeddings", self.config.max_position_embeddings)),
                dtype="bool"
            ), dtype="bool"
        )
        if not self.config.alibi:
            self.freq_cis: Tuple[chex.Array, chex.Array] = precompute_falcon_freq_cis(
                max_position_embedding=self.config.max_position_embeddings,
                head_dim=self.config.hidden_size // self.config.num_attention_heads
            )
        else:
            self.freq_cis = None

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            head_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            output_attentions: bool = False,
            deterministic: bool = True,
            init_cache: bool = False,
            return_dict: Optional[bool] = True
    ):
        batch, sequence_length = input_ids.shape

        hidden_states = self.word_embeddings(
            inputs=input_ids.astype(jnp.int32)
        )
        alibi = None
        if self.config.alibi:
            alibi = built_bloom_alibi(attention_mask, self.config.num_attention_heads).astype(hidden_states.dtype)

        else:
            alibi = None
            if position_ids is None:
                position_ids = jnp.arange(0, sequence_length).reshape(1, -1)
        if attention_mask is None:
            attention_mask = jnp.ones((batch, sequence_length), dtype="i4")
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, jnp.newaxis, jnp.newaxis, :]

        out_layers = self.h(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            freq_cis=self.freq_cis,
            causal_mask=self.causal_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache
        )

        out = out_layers[0]
        outputs = out_layers[1:]
        output = self.ln_f(out)

        if return_dict:
            if output_attentions:
                return FlaxBaseModelOutput(
                    last_hidden_state=output,
                    attentions=out_layers[1]
                )
            else:
                return FlaxBaseModelOutput(
                    last_hidden_state=output,
                )
        else:
            return output, outputs


class FlaxFalconPretrainedModel(EasyDelFlaxPretrainedModel):
    module_class: nn.Module = None
    config_class = FalconConfig

    def __init__(self, config,
                 _do_init=False,
                 dtype: jnp.dtype = jnp.float32,
                 param_dtype: jnp.dtype = jnp.float32,
                 input_shape: Tuple = (1, 1),
                 precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")
                 ):
        module = self.module_class(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision)
        super().__init__(_do_init=_do_init, module=module, config=config, dtype=dtype, input_shape=input_shape)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        """
        The init_weights function is used to initialize the weights of a model.

        :param self: Access variables that belong to the class
        :param rng: jax.random.PRNGKey: Initialize the weights of the model
        :param input_shape: Tuple: Specify the shape of the input tensor
        :param params: FrozenDict: Pass in the parameters of a pre-trained model
        :return: A frozendict of parameters

        """
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(
            jnp.atleast_2d(input_ids).shape[-1]), input_shape)
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
                input_ids,
                attention_mask,
                position_ids,
                return_dict=False
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

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: Optional[chex.Array] = None,
            position_ids: Optional[chex.Array] = None,
            past_key_values: Optional[nn.Module] = None,
            output_attentions: bool = False,
            train: bool = True,
            return_dict: Optional[bool] = True,
            params: FrozenDict = None,
            add_params_field: bool = False,
            **kwargs
    ):
        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        inputs = {'params': params or self.params} if add_params_field else params or self.params
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(input_ids.shape[1])[None, :],
                                            (input_ids.shape[0], input_ids.shape[1]))
        rngs = {}
        if self.config.bits is not None:
            rngs['params'] = jax.random.key(0)
        if attention_mask is None:
            attention_mask = jnp.ones((input_ids.shape[0], input_ids.shape[1]))

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            output_attentions,
            not train,
            False,
            return_dict,
            mutable=mutable,
            rngs=rngs
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]
        return outputs

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    @staticmethod
    def update_inputs_for_generation(model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


class FlaxFalconModel(FlaxFalconPretrainedModel):
    module_class = FlaxFalconModule

    def get_input_embeddings(self):
        return self.module.word_embeddings

    def set_input_embeddings(self, value):
        self.module.word_embeddings = value


class FlaxFalconForCausalLMModule(nn.Module):
    config: FalconConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.transformer = FlaxFalconModule(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(self,
                 input_ids: chex.Array,
                 attention_mask: Optional[chex.Array] = None,
                 position_ids: Optional[chex.Array] = None,
                 output_attentions: bool = False,
                 deterministic: bool = True,
                 init_cache: Optional[bool] = None,
                 return_dict: Optional[bool] = False
                 ):
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            init_cache=init_cache,
            return_dict=return_dict
        )
        if return_dict:
            hidden_state = transformer_output.last_hidden_state
        else:
            hidden_state = transformer_output[0]
        output = self.lm_head(hidden_state)
        if return_dict:
            if output_attentions:
                return FlaxCausalLMOutput(
                    logits=output,
                    attentions=transformer_output.attentions
                )
            else:
                return FlaxCausalLMOutput(
                    logits=output,
                )
        else:
            return (output, transformer_output[1]) if output_attentions else (output,)


class FlaxFalconForCausalLM(FlaxFalconPretrainedModel):
    module_class = FlaxFalconForCausalLMModule

    def get_decoder(self):
        return self.module.transformer

    def get_output_embeddings(self):
        return self.module.lm_head

    def get_input_embeddings(self):
        return self.module.transformer.word_embeddings

    def set_input_embeddings(self, value):
        self.module.transformer.word_embeddings = value

    def set_decoder(self, decoder):
        self.module.transformer = decoder

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings
