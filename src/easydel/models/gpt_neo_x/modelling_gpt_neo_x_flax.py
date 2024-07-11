import math
from typing import List, Optional, Tuple, Union

import chex
import jax
from flax import nnx
from jax import numpy as jnp

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    control_mlp_sharding,
)
from easydel.models.gpt_neo_x.gpt_neo_x_configuration import (
    GPTNeoXConfig as GPTNeoXConfig,
)
from easydel.models.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from easydel.models.modelling_utils import BaseNNXModule


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end)  # type: ignore
    freqs = jnp.outer(t, freqs).astype(dtype)
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(
        *xq_out.shape[:-1], -1
    )

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(
        *xk_out.shape[:-1], -1
    )

    return xq_out.astype(dtype), xk_out.astype(dtype)


class GPTNeoXAttention(BaseAttentionModule):
    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.head_size = config.hidden_size // config.num_attention_heads
        self.freqs_cis = precompute_freqs_cis(
            dtype=self.dtype,
            dim=self.head_size,
            end=config.max_position_embeddings,
        )
        self.w_qkv = nnx.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.w_o = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attention_module = FlexibleAttentionModule(
            num_attention_heads=config.num_attention_heads,
            attention_dropout=0.0,
            head_dims=self.head_dim,
            precision=precision,
            attn_mechanism=config.attn_mechanism,
            mesh=config.mesh,
            sm_scale=1 / math.sqrt(self.head_dim),
            base_config=config,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.config.hidden_size,)
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

        query, key, value = jnp.split(
            self.w_qkv(hidden_states),
            indices_or_sections=3,
            axis=-1,
        )

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        freq = jnp.expand_dims(freqs_cis[position_ids], -1)
        query, k = apply_rotary_emb(
            query,
            key,
            freqs_cis=freq,
            dtype=self.dtype,
        )

        if past_key_values is not None:
            past_key_values.update(key_states=key, value_states=value)
            key, value, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        query_length, key_length = query.shape[1], key.shape[1]
        attention_bias = None
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, :key_length]
            attention_bias = jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(
                    attention_mask.shape,
                    jnp.finfo(self.dtype).min,
                ).astype(self.dtype),
            )
        key_length = key.shape[1]
        attentions = self.attention_module(
            query_states=query,
            key_states=key,
            value_states=value,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=True,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            segment_ids=segment_ids,
        )
        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.w_o(attn_output)
        return attn_output, attentions.attention_weights


class GPTNeoXMlp(nnx.Module):

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense_h_to_4h = nnx.Linear(
            self.config.hidden_size,
            self.config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.dense_4h_to_h = nnx.Linear(
            self.config.intermediate_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = control_mlp_sharding(
            hidden_states,
            self.config.partition_axis,
        )
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(hidden_states)))


class GPTNeoXBlock(nnx.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nnx.LayerNorm(
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = nnx.LayerNorm(
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention = GPTNeoXAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = GPTNeoXMlp(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
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
        attn, attn_weight = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            past_key_values=past_key_values,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )

        if self.use_parallel_residual:
            mlp = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp + hidden_states + attn
        else:
            hidden_states = attn + hidden_states
            hidden_states = (
                self.mlp(self.post_attention_layernorm(hidden_states)) + hidden_states
            )
        return hidden_states, attn_weight


class GPTNeoXModel(BaseNNXModule):

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_in = nnx.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            GPTNeoXBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.final_layer_norm = nnx.LayerNorm(
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self._causal_mask = None
        self._freqs_cis = None

    @property
    def freqs_cis(self):
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                self.config.rotary_dim or self.config.hidden_size,
                self.config.max_position_embeddings,
                theta=self.config.rotary_emb_base,
                dtype=self.dtype,
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
        input_ids,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        input_embeds: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
        extra_embedding: Optional[chex.Array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_input_embeds at the same time"
            )
        if input_embeds is None and input_ids is not None:
            input_embeds = self.embed_in(input_ids.astype("i4"))
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

        assert (
            sequence_length <= self.config.max_position_embeddings
        ), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"

        hidden_states = (
            input_embeds + extra_embedding
            if extra_embedding is not None
            else input_embeds
        )
        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weight = block(
                hidden_states=hidden_states,
                freqs_cis=self.freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=(
                    past_key_values[idx] if past_key_values is not None else None
                ),
                segment_ids=segment_ids,
            )
            if output_attentions:
                all_attentions += (attn_weight,)
        hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (
            hidden_states,
            all_hidden_states,
            all_attentions,
        )
        if return_dict:
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_states,
                all_hidden_states=outputs[1],
                attentions=outputs[2],
            )

        return tuple([v for v in outputs if v is not None])

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.module.wte = value


class GPTNeoXForCausalLM(BaseNNXModule):

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.transformer = GPTNeoXModel(
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
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        past_key_values: Optional[List[KVCache]] = None,
        input_embeds: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
        extra_embedding: Optional[chex.Array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_embeds=input_embeds,
            segment_ids=segment_ids,
            extra_embedding=extra_embedding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
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

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.transformer

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.module.transformer = decoder

    def set_input_embeddings(self, value):
        self.module.transformer.wte = value

    @property
    def can_generate(self):
        return True
