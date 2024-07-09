from typing import Any, List, Optional, Union

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen.attention import dot_product_attention_weights
from jax import lax
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)

from easydel.models.caching_utils import KVCache
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    block_wise_ffn,
    with_sharding_constraint,
)
from easydel.models.gpt2.gpt2_configuration import GPT2Config as GPT2Config
from easydel.models.modelling_utils import BaseNNXModule


class Conv1D(nnx.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        dot_general: Optional[None] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.kernel = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(
                rngs.params(), (out_features, in_features)
            ),
        )

        self.bias = nnx.Param(
            nnx.initializers.zeros(
                rngs.params(),
                (self.features,),
            )
            if use_bias
            else None
        )

        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dot_general = dot_general

    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        bias = self.bias.value
        kernel = self.kernel.value.transpose().astype(self.dtype)
        if self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = lax.dot_general

        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y = y + bias.astype(self.dtype)
        return y


class GPT2Attention(BaseAttentionModule):
    def __init__(
        self,
        config: GPT2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.precision = precision
        self.dtype = dtype
        self.rngs = rngs
        self.is_cross_attention = is_cross_attention
        self.causal = causal
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.is_cross_attention:
            self.c_attn = Conv1D(
                self.embed_dim,
                2 * self.embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.q_attn = Conv1D(
                self.embed_dim,
                self.embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.c_attn = Conv1D(
                self.embed_dim,
                3 * self.embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.c_proj = Conv1D(
            self.embed_dim,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.resid_dropout = nnx.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        key_value_states: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        past_key_values: Optional[KVCache] = None,
    ):
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        if not is_cross_attention:
            qkv_out = self.c_attn(hidden_states)
            query, key, value = jnp.split(qkv_out, 3, axis=2)
        else:
            q_out = self.q_attn(hidden_states)
            (query,) = jnp.split(q_out, 1, axis=2)
            kv_out = self.c_attn(key_value_states)
            key, value = jnp.split(kv_out, 2, axis=2)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if past_key_values is not None:
            past_key_values.update(key_states=key, value_states=value)
            key, value, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        key_length = key.shape[1]
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
            attention_bias = None

        # usual dot product attention
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rate=self.config.attn_pdrop,
            dtype=self.dtype,
            precision=self.precision,
        )
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output,
                PartitionSpec(
                    ("dp", "fsdp"), "sp" if attn_output.shape[1] != 1 else None, "tp"
                ),
            )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class GPT2MLP(nnx.Module):

    def __init__(
        self,
        config: GPT2Config,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.precision = precision
        self.dtype = dtype
        self.rngs = rngs
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(
            embed_dim,
            intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.c_proj = Conv1D(
            intermediate_size,
            embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.activation_function]
        self.dropout = nnx.Dropout(
            rate=config.resid_pdrop,
            rngs=rngs,
        )

    def __call__(self, hidden_states):
        return self.dropout(self.c_proj(self.act(self.c_fc(hidden_states))))


class GPT2Block(nnx.Module):

    def __init__(
        self,
        config: GPT2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        *,
        rngs: nnx.Rngs,
    ):

        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        hidden_size = self.config.hidden_size
        inner_dim = (
            self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
        )

        self.ln_1 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        attn_block = GPT2Attention
        mlp_block = GPT2MLP
        # if self.config.gradient_checkpointing != "":
        #     attn_block = flax.linen.partitioning.remat(
        #         attn_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(3, 4, 5, 6),
        #     )

        #     mlp_block = flax.linen.partitioning.remat(
        #         mlp_block,
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #         static_argnums=(1,),
        #     )
        # hidden_states,
        # key_value_states: Optional[chex.Array] = None,
        # attention_mask = None,
        # casual_mask = None,
        # deterministic: bool = True,
        # init_cache: bool = False,
        # output_attentions: bool = False,

        self.attn = attn_block(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ln_2 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if config.add_cross_attention:
            self.crossattention = attn_block(
                config=config,
                dtype=dtype,
                causal=True,
                is_cross_attention=True,
            )
            self.ln_cross_attn = nnx.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.mlp = mlp_block(self.config, inner_dim, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states: Optional[chex.Array] = None,
        encoder_attention_mask: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        past_key_values: Optional[KVCache] = None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output, attn_weight = self.attn(
            hidden_states,
            None,
            attention_mask,
            past_key_values,
        )
        outputs = (attn_weight,)
        hidden_states = attn_output + residual
        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            # hidden_states
            # key_value_states: Optional[chex.Array] = None
            # attention_mask = None
            # casual_mask = None
            # deterministic: bool = True
            # init_cache: bool = False
            # output_attentions: bool = False

            attn_output, cross_attn_weight = self.crossattention(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                False,
            )
            # residual connection
            hidden_states = residual + attn_output
            outputs += (
                cross_attn_weight,  # add cross attentions if we output attention weights
            )
        else:
            outputs += (None,)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + outputs

        return outputs


class GPT2Model(BaseNNXModule):

    def __init__(
        self,
        config: GPT2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.embed_dim = self.config.hidden_size

        self.wte = nnx.Embed(
            config.vocab_size,
            self.embed_dim,
            embedding_init=nnx.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wpe = nnx.Embed(
            config.max_position_embeddings,
            self.embed_dim,
            embedding_init=nnx.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(
            rate=config.embd_pdrop,
            rngs=rngs,
        )
        self.h = [
            GPT2Block(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.ln_f = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self._causal_mask = None

    @property
    def causal_mask(self):
        if self._causal_mask:
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
                    dtype="bool",
                ),
                dtype="bool",
            )
        return self._causal_mask

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        encoder_hidden_states: Optional[chex.Array] = None,
        encoder_attention_mask: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )

        batch_size, sequence_length = input_ids.shape
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

        input_embeds = self.wte(input_ids.astype("i4"))
        position_embeds = self.wpe(position_ids.astype("i4"))

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        for idx, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                past_key_values=(
                    past_key_values[idx] if past_key_values is not None else None
                ),
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        outputs = (
            hidden_states,
            all_hidden_states,
            all_attentions,
            all_cross_attentions,
        )
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
        )

    def get_input_embeddings(self):
        return self.module.wte

    def set_input_embeddings(self, value):
        self.module.wte = value

    @property
    def can_generate(self):
        return False


class FlaxGPT2LMHeadModule(BaseNNXModule):
    def __init__(
        self,
        config: GPT2Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config=config)
        self.transformer = GPT2Model(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        encoder_hidden_states: Optional[chex.Array] = None,
        encoder_attention_mask: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.wte.embedding.value.T
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def get_output_embeddings(self):
        return self.module.lm_head

    def get_decoder(self):
        return self.module.transformer

    def get_input_embeddings(self):
        return self.module.transformer.wte

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.module.transformer = decoder

    def set_input_embeddings(self, value):
        self.module.transformer.wte = value
