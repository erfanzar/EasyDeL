import functools
from functools import partial
from typing import List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from fjformer.sharding import with_sharding_constraint
from flax import nnx
from jax import lax
from jax.sharding import PartitionSpec

from easydel.etils.etils import get_logger
from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    apply_rotary_pos_emb,
    block_wise_ffn,
    control_mlp_sharding,
    precompute_freqs_cis,
)
from easydel.models.gemma2.gemma2_configuration import Gemma2Config as Gemma2Config
from easydel.models.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from easydel.models.modelling_utils import BaseNNXModule

logger = get_logger(__name__)


def add_positional_embedding(
    input_embedding: jax.Array,
    position: int,
    theta: int = 10_000,
) -> jax.Array:
    """Adds positional embeddings to input embeddings. From DeepMind Gemma"""
    embed_dim = input_embedding.shape[-1]
    num_timescales = embed_dim // 2
    log_timescale_increment = jnp.log(float(theta)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    inv_timescales = jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)])
    signal = jnp.pad(signal, [[0, jnp.mod(embed_dim, 2)]])
    position_embedding = signal.astype(jnp.float32)

    return input_embedding + position_embedding


def apply_rope(query, key, freqs_cis, position_ids, dtype: jnp.dtype = jnp.float32):
    sin, cos = freqs_cis

    sin = sin[position_ids][:, None, :, :]
    cos = cos[position_ids][:, None, :, :]

    key = apply_rotary_pos_emb(key, sin, cos)
    query = apply_rotary_pos_emb(query, sin, cos)

    return query.astype(dtype), key.astype(dtype)


class Gemma2RMSNorm(nnx.Module):

    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.config = config
        self.dtype = dtype
        self.epsilon = self.config.rms_norm_eps
        self.kernel = nnx.Param(jnp.ones(self.config.hidden_size, dtype=param_dtype))

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return (1 + self.kernel) * jnp.asarray(hidden_states, dtype=self.dtype)


class Gemma2Attention(BaseAttentionModule):
    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.layer_idx = layer_idx
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        kernel = jax.nn.initializers.normal(self.config.initializer_range)

        dense_class = partial(
            nnx.Linear,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel,
            rngs=rngs,
        )
        self.q_proj = dense_class(self.num_heads * self.head_dim)
        self.k_proj = dense_class(self.num_key_value_heads * self.head_dim)
        self.v_proj = dense_class(self.num_key_value_heads * self.head_dim)
        self.o_proj = dense_class(self.embed_dim)
        self.sliding_window = (
            config.sliding_window if (self.layer_idx % 2 == 0) else None
        )
        self.attention_module = FlexibleAttentionModule(
            num_attention_heads=self.config.num_attention_heads,
            head_dims=self.head_dim,
            attn_mechanism=self.config.attn_mechanism,
            mesh=self.config.get_mesh(),
            sm_scale=self.config.query_pre_attn_scalar**-0.5,
            axis_name=self.config.attention_axis_name,
            base_config=config,
        )

        self.rotary_emb = functools.partial(apply_rope, dtype=dtype)

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads * self.head_dim,)
        )

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def apply_rotary(self, query, key, freqs_cis, position_ids):
        """The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freqs_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            query: Calculate the attention weights
            key: Calculate the attention
            freqs_cis: Calculate the frequency of each word in the vocabulary
            position_ids: Identify the position of each token in the sequence

        Returns:
            A tuple of 3 tensors: query, key and value
        """
        query, key = self._transpose_sequence_head(query, key)
        query, key = self.rotary_emb(
            position_ids=position_ids,
            query_states=query,
            key_states=key,
            freqs_cis=freqs_cis,
        )
        return self._transpose_sequence_head(query, key)

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        """
        The __call__ function is the main function of a JAX module. It defines how the module behaves when called
        with inputs. The __call__ function can be thought of as a &quot;forward pass&quot; through the model,
        and it should return all outputs that are needed for training or inference.

        Args:
            self: Access variables that belong to the class
            hidden_states: (chex.Array): Pass the hidden states of the previous layer
            freqs_cis: (Tuple[chex.Array, chex.Array]),: Pass in the frequency coefficients for each position
            attention_mask: (chex.Array): Mask out certain tokens in the input sequence
            past_key_values: (Optional(KVCache)): Past key and values used for generation
            position_ids: (Optional(chex.Array)): Determine the position of each token in a sequence

        Returns:
            A tuple of two arrays HiddenState and attentionWeight
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        query_states, key_states = self.apply_rotary(
            query_states,
            key_states,
            freqs_cis,
            position_ids,
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if past_key_values is not None:
            past_key_values.update(key_states=key_states, value_states=value_states)
            key_length, value_states, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        key_states, value_states = self.repeat_key_value(
            key_states,
            value_states,
            self.num_key_value_groups,
        )
        attention_bias = None

        if attention_mask is not None:
            if bool((self.layer_idx % 2) == 0):
                sliding_window_mask = jnp.tril(
                    jnp.ones_like(attention_mask, dtype=jnp.bool),
                    k=-self.sliding_window,
                )
                window_mask = jnp.where(sliding_window_mask, 0, 1)
                attention_mask = jnp.logical_and(window_mask, attention_mask)

                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                    self.dtype
                ),
            )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_module(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=True,
            deterministic=self.resid_dropout.deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            segment_ids=segment_ids,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
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
        attn_output = self.resid_dropout(self.o_proj(attn_output))
        return attn_output, attentions.attention_weights


class Gemma2MLP(nnx.Module):
    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        embed_dim = self.config.hidden_size
        inner_dim = (
            self.config.intermediate_size
            if self.config.intermediate_size is not None
            else 4 * embed_dim
        )

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        hidden_activation = self.config.hidden_activation
        self.act = ACT2FN[hidden_activation]

        self.gate_proj = nnx.Linear(
            embed_dim,
            inner_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            inner_dim,
            embed_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            embed_dim,
            inner_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, hidden_states: chex.Array):

        hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


class Gemma2DecoderLayer(nnx.Module):

    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        mlp_block = Gemma2MLP
        attn_block = Gemma2Attention

        # if self.config.gradient_checkpointing != "":
        #     mlp_block = flax.linen.partitioning.remat(
        #         mlp_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(1,),
        #     )
        #     attn_block = flax.linen.partitioning.remat(
        #         attn_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(3, 4, 6, 7, 8),
        #     )
        self.is_sliding = bool(self.layer_idx % 2)
        self.self_attn = attn_block(
            config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = mlp_block(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = Gemma2RMSNorm(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.post_attention_layernorm = Gemma2RMSNorm(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weight = self.self_attn(
            hidden_states,
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weight


class Gemma2Model(BaseNNXModule):
    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Gemma2DecoderLayer(
                config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.norm = Gemma2RMSNorm(config, dtype=dtype, param_dtype=param_dtype)

        self._causal_mask = None
        self._freqs_cis = None

    @property
    def freqs_cis(self):
        if self._freqs_cis is None:
            config = self.config

            self._freqs_cis = precompute_freqs_cis(
                max_position_embeddings=config.max_position_embeddings,
                dim=config.head_dim,
                base=config.rope_theta,
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
        inputs_embeds: Optional[chex.Array] = None,
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
            inputs_embeds: (Optional(chex.Array)): Pass in the embeddings of the input tokens
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

        batch_size, seq_length = input_ids.shape

        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length),
            ).astype(jnp.int32)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
            attention_mask = jnp.logical_and(
                attention_mask, self.causal_mask[:, :, :seq_length, :]
            )

        if inputs_embeds is None:
            input_embeds = self.embed_tokens(input_ids.astype("i4"))
        if extra_embedding is not None:
            input_embeds += extra_embedding
        hidden_states = input_embeds * jnp.asarray(
            self.config.hidden_size**0.5,
            dtype=input_embeds.dtype,
        )

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weights = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                freqs_cis=self.freqs_cis,
                past_key_values=past_key_values[idx],
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions += (attn_weights,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class Gemma2ForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.model = Gemma2Model(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        extra_embedding: Optional[jax.Array] = None,
    ):
        """The __call__ function is the main function of a Flax module. It takes in inputs and returns outputs.

        Args:
            self: Refer to the object itself
            input_ids: chex.Array: Pass the input token ids to the model
            attention_mask: (Optional(chex.Array)): Mask out the padding tokens
            position_ids: (Optional(chex.Array)): Specify the position of each token in the input sequence
            past_key_values: (Optional(List[KVCache])): Past key and values used for generation
            output_attentions: bool: Return the attention weights
            output_hidden_states: bool: Determine whether to return the hidden states
            return_dict: bool: Return a dictionary of the outputs or not
            extra_embedding: (Optional(chex.Array)): Pass in the embedding of the word that we want to predict

        Returns:
            The logits and the hidden states
        """
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            extra_embedding=extra_embedding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.embed_tokens.embedding.value.T.astype(
                self.param_dtype
            )
            self.lm_head.kernel.value = shared_kernel
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            lm_logits = lm_logits / self.config.final_logit_softcapping
            lm_logits = jax.nn.tanh(lm_logits)
            lm_logits = lm_logits * self.config.final_logit_softcapping
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def can_generate(self):
        return True
