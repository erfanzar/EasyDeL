import functools
import math
from typing import List, Optional, Tuple, Union

import chex
import jax
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.falcon.falcon_configuration import FalconConfig as FalconConfig
from easydel.models.flax_modelling_utils import (
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    precompute_freqs_cis,
    with_sharding_constraint,
)
from easydel.models.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
    FlaxSequenceClassifierOutput,
)
from easydel.models.modelling_utils import BaseNNXModule


def built_bloom_alibi(attention_mask, num_attention_heads):
    """The built_bloom_alibi function is used to create a bloom alibi for the attention mask.
    The bloom alibi is used in the Bloom Attention layer to ensure that each token has a unique
    attention vector, even if it's masked out. This ensures that all tokens have an equal chance of being selected as
    the most important token in the sequence, which helps with training stability and performance.

    Args:
        attention_mask: Mask out the padding tokens in the input sequence
        num_attention_heads: Determine the number of attention heads in the model

    Returns:
        A tensor of shape (batch_size, num_attention_heads, 1, sequence_length)
    """
    batch_size, sequence_length = attention_mask.shape
    cp2 = 2 ** math.floor(math.log2(num_attention_heads))
    slops = jnp.power(
        jnp.asarray(2 ** (-(2 ** -(math.log2(cp2) - 3))), dtype=jnp.float32),
        jnp.arange(1, 1 + cp2, dtype=jnp.float32),
    )
    if cp2 != num_attention_heads:
        extra_base = jnp.asarray(
            2 ** (-(2 ** -(math.log2(2 * cp2) - 3))), dtype=jnp.float32
        )
        num_rem_heads = min(cp2, num_attention_heads - cp2)
        extra_power = jnp.arange(1, 1 + 2 * num_rem_heads, 2, dtype=jnp.dtype)
        slops = jnp.concatenate([slops, jnp.power(extra_base, extra_power)], axis=0)
    arange_tensor = (((jnp.cumsum(attention_mask, axis=-1)) - 1) * attention_mask)[
        :, jnp.newaxis, :
    ]
    alibi = slops[..., jnp.newaxis].astype(jnp.bfloat16) * arange_tensor
    return alibi.reshape(batch_size, num_attention_heads, 1, sequence_length)


def dropout_add(
    linen_drop: nnx.Dropout,
    x: chex.Array,
    residual: chex.Array,
) -> chex.Array:
    """The dropout_add function is a helper function that adds the residual to the output of
    the dropout layer. This is necessary because we want to use deterministic=True when
    we are evaluating our model, but we still need to add in the residual. The reason for this
    is that during training, we have two paths through our network: one with dropout and one without.
    The path without dropout (residual) allows us to backpropagate gradients through both paths at once.

    Args:
        linen_drop: flax.linen.Dropout: Specify the dropout layer
        x: chex.Array: Pass in the input to the dropout layer
        residual: chex.Array: Add the residual to the output of dropout_add

    Returns:
        A tensor that is the sum of the residual and a dropout layer
    """

    out = linen_drop(x)
    return residual + out


def apply_rope(query, key, freqs_cis, position_ids, dtype: jnp.dtype = jnp.float32):
    sin, cos = freqs_cis

    sin = sin[position_ids][:, None, :, :]
    cos = cos[position_ids][:, None, :, :]

    key = apply_rotary_pos_emb(key, sin, cos)
    query = apply_rotary_pos_emb(query, sin, cos)

    return query.astype(dtype), key.astype(dtype)


class FalconAttention(BaseAttentionModule):
    def __init__(
        self,
        config: FalconConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: FalconConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.precision = precision
        self.rngs = rngs
        config = self.config
        head_dim = config.hidden_size // config.num_attention_heads
        if config.new_decoder_architecture:
            qkv_out_dim = (
                config.num_kv_heads * 2 + config.num_attention_heads
            ) * head_dim
        elif config.multi_query:
            qkv_out_dim = config.hidden_size + 2 * head_dim
        else:
            qkv_out_dim = 3 * config.hidden_size

        self.head_dim = head_dim
        assert self.head_dim * config.num_attention_heads == config.hidden_size
        self.num_kv_heads = (
            config.num_kv_heads
            if (config.new_decoder_architecture or not config.multi_query)
            else 1
        )
        self.new_decoder_architecture = config.new_decoder_architecture
        self.num_heads = config.num_attention_heads
        self.query_key_value = nnx.Linear(
            qkv_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=precision,
            use_bias=config.bias,
            rngs=rngs,
        )
        self.inv_norm_factor = 1 / math.sqrt(head_dim)
        self.dense = nnx.Linear(
            head_dim * config.num_attention_heads,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_bias=config.bias,
            rngs=rngs,
        )
        self.rotary = functools.partial(apply_rope, dtype=dtype)
        self.attention_module: FlexibleAttentionModule = FlexibleAttentionModule(
            mesh=config.mesh,
            attn_mechanism=config.attn_mechanism,
            sm_scale=1 / math.sqrt(self.head_dim),
            num_attention_heads=config.num_attention_heads,
            head_dims=self.head_dim,
            precision=precision,
            base_config=config,
            _do_check=False,
        )
        self.resid_dropout = nnx.Dropout(
            rate=config.resid_pdrop,
            rngs=rngs,
        )

    def _split_heads(
        self,
        qkv: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        batch_size, sequence_length, _ = qkv.shape

        if self.config.new_decoder_architecture:
            qkv = qkv.reshape(
                batch_size,
                sequence_length,
                -1,
                self.num_heads // self.num_kv_heads + 2,
                self.head_dim,
            )
            query_layer = qkv[:, :, :, :-2]
            key_layer = qkv[:, :, :, [-2]]
            value_layer = qkv[:, :, :, [-1]]
            key_layer = jnp.broadcast_to(key_layer, query_layer.shape)
            value_layer = jnp.broadcast_to(value_layer, query_layer.shape)

            query_layer, key_layer, value_layer = [
                x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
                for x in (query_layer, key_layer, value_layer)
            ]

            return query_layer, key_layer, value_layer
        if self.config.multi_query:
            qkv = qkv.reshape(
                batch_size, sequence_length, self.config.num_attention_heads + 2, -1
            )
            query_layer, key_layer, value_layer = (
                qkv[..., :-2, :],
                qkv[..., [-2], :],
                qkv[..., [-1], :],
            )

        else:
            query_layer, key_layer, value_layer = jnp.split(qkv, 3, -1)
        return query_layer, key_layer, value_layer

    def _merge_heads(self, x: chex.Array) -> chex.Array:
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.reshape(
            batch_size,
            self.config.num_attention_heads,
            seq_length,
            self.head_dim,
        )
        return x.reshape(
            batch_size,
            seq_length,
            self.config.num_attention_heads * self.head_dim,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        alibi: chex.Array = None,
        freqs_cis: Tuple[chex.Array, chex.Array] = None,
        past_key_values: Optional[KVCache] = None,
    ):
        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = (
            self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        )
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, query_length, _, _ = query_layer.shape
        key_length = query_length
        query_layer = query_layer.reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        key_layer = key_layer.reshape(
            batch_size,
            query_length,
            num_kv_heads,
            self.head_dim,
        )
        value_layer = value_layer.reshape(
            batch_size,
            query_length,
            num_kv_heads,
            self.head_dim,
        )

        if alibi is None:
            query_layer, key_layer = map(
                lambda x: x.transpose(0, 2, 1, 3), [query_layer, key_layer]
            )  # noqa
            query_layer, key_layer = self.rotary(
                query_layer, key_layer, freqs_cis, position_ids
            )
            query_layer, key_layer = map(
                lambda x: x.transpose(0, 2, 1, 3), [query_layer, key_layer]
            )  # noqa

        if past_key_values is not None:
            past_key_values.update(key_states=key_layer, value_states=value_layer)
            key_length, value_layer, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        query_length, key_length = query_layer.shape[1], key_layer.shape[1]

        attention_bias = None
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, :key_length]
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(
                    attention_mask.shape,
                    jnp.finfo(self.dtype).min,
                ).astype(self.dtype),
            )

        query_length, key_length = query_layer.shape[1], key_layer.shape[1]
        dtype = jnp.promote_types(key_layer.dtype, jnp.float32)

        query_layer, key_layer, value_layer, attention_bias = map(
            lambda x: x.astype(dtype=dtype),
            (query_layer, key_layer, value_layer, attention_bias),
        )

        if alibi is None:
            attention = self.attention_module.__call__(
                query_states=query_layer,
                key_states=key_layer,
                value_states=value_layer,
                attention_mask=attention_mask,
                segment_ids=None,
                query_sequence_length=query_length,
                key_value_sequence_length=key_length,
                bias=attention_bias,
                causal=False,
            )
            attention_outputs = attention.attention_outputs
            attention_outputs = attention_outputs.reshape(
                batch_size, query_length, self.num_heads * self.head_dim
            )
            output_tensor = self.dense(attention_outputs)
            return output_tensor, attention.attention_weights

        else:
            attention_scores = jnp.einsum(
                "...qhd,...khd->...hqk",
                query_layer,
                key_layer,
                precision=self.precision,
            )
            attention_scores = attention_scores.reshape(
                batch_size, self.num_heads, query_length, key_length
            )
            attention_scores = attention_scores + alibi.reshape(
                batch_size, self.num_heads, 1, -1
            )
            attention_scores *= self.inv_norm_factor
            attention_scores = jax.nn.softmax(
                attention_scores + attention_bias, axis=-1
            )
            attention_scores = attention_scores.reshape(
                batch_size, self.num_heads, query_length, key_length
            )
            # matmul: [batch_size * num_heads, q_length, head_dim]
            attn_output = jax.lax.batch_matmul(
                attention_scores, value_layer.transpose(0, 2, 1, 3)
            )  # noqa
            attn_output = attn_output.reshape(
                (attn_output.shape[1] * attn_output.shape[0],) + attn_output.shape[2:]
            )
            attn_output = self._merge_heads(attn_output)
            if self.config.shard_attention_computation:
                attn_output = with_sharding_constraint(
                    attn_output,
                    PartitionSpec(
                        ("dp", "fsdp"),
                        "sp" if attn_output.shape[1] != 1 else None,
                        "tp",
                    ),
                )

            output_tensor = self.dense(attn_output)

            return output_tensor, attention_scores


class FalconMlp(nnx.Module):
    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: FalconConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.dense_h_to_4h = nnx.Linear(
            self.config.hidden_size,
            self.config.ff_factor * self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.bias,
            precision=precision,
            rngs=rngs,
        )
        self.dense_4h_to_h = nnx.Linear(
            self.config.ff_factor * self.config.hidden_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.bias,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: chex.Array):
        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        return self.dense_4h_to_h(
            nnx.gelu(
                self.dense_h_to_4h(hidden_states),
                approximate=False,
            )
        )


class FalconBlock(nnx.Module):
    def __init__(
        self,
        config: FalconConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config: FalconConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.precision = precision
        self.rngs = rngs
        config = self.config

        if config.new_decoder_architecture and config.num_ln_in_parallel_attn == 2:
            self.ln_attn = nnx.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.ln_mlp = nnx.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.input_layernorm = nnx.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if not config.parallel_attn:
                self.post_attention_layernorm = nnx.LayerNorm(
                    config.hidden_size,
                    epsilon=config.layer_norm_epsilon,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
        attn_block = FalconAttention
        mlp_block = FalconMlp
        # if self.config.gradient_checkpointing != "":
        #     attn_block = flax.linen.partitioning.remat(
        #         attn_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(3, 5, 6, 7, 8),
        #     )

        #     mlp_block = flax.linen.partitioning.remat(
        #         mlp_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(1,),
        #     )

        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.self_attention = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            layer_idx=layer_idx,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(self.config.attention_dropout)
        self.dropout_mlp = nnx.Dropout(self.config.hidden_dropout)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        alibi: chex.Array = None,
        freqs_cis: Tuple[chex.Array, chex.Array] = None,
        past_key_values: Optional[KVCache] = None,
    ):
        residual = hidden_states

        if self.config.num_ln_in_parallel_attn == 2:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        attention_output, attn_score = self.self_attention(
            attention_layernorm_out,
            attention_mask,
            position_ids,
            alibi,
            freqs_cis,
            past_key_values,
        )

        if self.config.num_ln_in_parallel_attn == 1:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(self.dropout, attention_output, residual)
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if self.config.use_scan_mlp:
            mlp_output = block_wise_ffn(
                self.mlp, mlp_layernorm_out, self.config.scan_mlp_chunk_size
            )
        else:
            mlp_output = self.mlp(
                mlp_layernorm_out,
            )

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(self.dropout_mlp, mlp_output, residual)
        return output, attn_score


class FalconModel(BaseNNXModule):
    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config: FalconConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.word_embeddings = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.h = [
            FalconBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=i,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.ln_f = nnx.LayerNorm(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            epsilon=config.layer_norm_epsilon,
            rngs=rngs,
        )

        self._causal_mask = None
        self._freqs_cis = None

    @property
    def freqs_cis(self):
        if self._freqs_cis is None and not self.config.alibi:
            config = self.config
            initial_rope_kwargs = dict(rope_type="none")
            if config.rope_scaling is not None:
                scaling_type = config.rope_scaling["type"]
                scaling_factor = config.rope_scaling["factor"]
                initial_rope_kwargs = dict(
                    scaling_factor=scaling_factor, rope_type=scaling_type
                )
            self._freqs_cis = precompute_freqs_cis(
                max_position_embeddings=getattr(
                    self.config,
                    "freq_max_position_embeddings",
                    self.config.max_position_embeddings,
                ),
                dim=config.hidden_size // config.num_attention_heads,
                base=config.rope_theta,
                **initial_rope_kwargs,
            )
        return self._freqs_cis

    @property
    def causal_mask(self):
        if self._causal_mask is None:
            self._causal_mask = nnx.make_causal_mask(
                jnp.ones(
                    (
                        1,
                        getattr(
                            self.config,
                            "causal_mask_max_position_embeddings",
                            self.config.max_position_embeddings,
                        ),
                    ),
                    dtype=jnp.bool,
                ),
                dtype=jnp.bool,
            )
        return self._causal_mask

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jnp.ndarray] = None,
    ):
        """
        The __call__ function is the main function of a Flax model. It takes in input_ids, attention_mask, and position_ids
        and returns the output of the model. These optional arguments are passed as keyword arguments when calling a Flax model.

        Args:
            self: Represent the instance of the class
            input_ids: chex.Array: Pass in the input token ids
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Indicate the position of each token in a sequence
            input_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Determine whether to return the attentions or not
            output_hidden_states: bool: Determine whether to return hidden states
            return_dict: bool: Return a dictionary of the output or not
            extra_embedding: Optional[Union[jnp.ndarray]]: Pass in the extra embedding

        Returns:
            A tuple of: predictions
        """

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
            )
        if input_embeds is None and input_ids is not None:
            input_embeds = self.embed_tokens(input_ids.astype("i4"))
        else:
            raise ValueError("you should specify input_embeds or input_ids one of them")

        batch_size, sequence_length, _ = input_embeds.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, sequence_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :sequence_length, :]
            )

        alibi = None

        if self.config.alibi:
            alibi = built_bloom_alibi(
                attention_mask, self.config.num_attention_heads
            ).astype(input_embeds.dtype)

        assert (
            sequence_length <= self.config.max_position_embeddings
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

        hidden_states = (
            input_embeds + extra_embedding
            if extra_embedding is not None
            else input_embeds
        )

        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers

        for idx, layer in enumerate(self.h):
            hidden_states, attention_weights = layer(
                alibi=alibi,
                attention_mask=attention_mask,
                freqs_cis=self.freqs_cis,
                hidden_states=hidden_states,
                past_key_values=past_key_values[idx],
                position_ids=position_ids,
            )
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if output_attentions:
                all_attentions += (attention_weights,)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_dict:
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_states,
                attentions=all_attentions,
                hidden_states=all_hidden_states,
            )

        return tuple(
            [
                s
                for s in [hidden_states, all_attentions, all_attentions]
                if s is not None
            ]
        )


class FalconForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config: FalconConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.transformer = FalconModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jnp.ndarray] = None,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            extra_embedding=extra_embedding,
            input_embeds=input_embeds,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.word_embeddings.embedding.value.T
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits = lm_logits.astype(jnp.float32)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

    @property
    def can_generate(self):
        return True


class FalconForSequenceClassification(BaseNNXModule):
    def __init__(
        self,
        num_classes: int,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.transformer = FalconModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.classifier = nnx.Linear(
            config.hidden_size,
            num_classes,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array = None,
        position_ids: chex.Array = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jnp.ndarray] = None,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_embedding=extra_embedding,
            past_key_values=None,
        )

        hidden_states = outputs[0]
        prediction = self.classifier(hidden_states)
        if return_dict:
            return FlaxSequenceClassifierOutput(
                logits=prediction,
                hidden_states=hidden_states,
            )
        else:
            return (prediction,)

    @property
    def can_generate(self):
        return True
