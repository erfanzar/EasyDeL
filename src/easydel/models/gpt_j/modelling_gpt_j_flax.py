import math
from functools import partial
from typing import Optional, Tuple, Union, List

import chex
import flax.linen.partitioning
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.utils import logging

from easydel.models.attention_module import FlexibleAttentionModule
from easydel.models.caching_utils import KVCache
from easydel.models.flax_modelling_utils import (
    ACT2FN,
    BaseAttentionModule,
    block_wise_ffn,
)
from easydel.models.gpt_j.gpt_j_configuration import GPTJConfig as GPTJConfig

logger = logging.get_logger(__name__)


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype(
        "float32"
    )
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out)


def rotate_every_two(tensor):
    rotate_half_tensor = jnp.stack(
        (-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1
    )
    rotate_half_tensor = rotate_half_tensor.reshape(
        rotate_half_tensor.shape[:-2] + (-1,)
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class GPTJAttention(BaseAttentionModule):
    def __init__(
        self,
        config: GPTJConfig,
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

        self.rotary_dim = config.rotary_dim

        linear = partial(
            nnx.Linear,
            self.embed_dim,
            self.embed_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            param_dtype=param_dtype,
            precision=precision,
        )

        self.q_proj, self.k_proj, self.v_proj = (
            linear(rngs=rngs),
            linear(rngs=rngs),
            linear(rngs=rngs),
        )
        self.out_proj = linear(rngs=rngs)

        self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop, rngs=rngs)
        # pos_embd_dim = self.rotary_dim or self.embed_dim

        self.attention_module = FlexibleAttentionModule(
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attn_pdrop,
            head_dims=self.head_dim,
            precision=precision,
            attn_mechanism=config.attn_mechanism,
            mesh=config.get_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
            base_config=config,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states: chex.Array,
        freqs_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        past_key_values: Optional[KVCache] = None,
        segment_ids: Optional[chex.Array] = None,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        sincos = jnp.take(freqs_cis, position_ids, axis=0)
        sincos = jnp.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            key = jnp.concatenate([k_rot, k_pass], axis=-1)
            query = jnp.concatenate([q_rot, q_pass], axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        if past_key_values is not None:
            past_key_values.update(key_states=key, value_states=value)
            key, value, attention_mask = past_key_values.get(
                attention_mask=attention_mask
            )

        query_length, key_length = query.shape[1], key.shape[1]
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
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attentions.attention_weights


class GPTJMLP(nnx.Module):
    def __init__(
        self,
        config: GPTJConfig,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.intermediate_size = intermediate_size
        embed_dim = config.hidden_size
        kernel_init = nnx.initializers.normal(config.initializer_range)

        self.fc_in = nnx.Linear(
            embed_dim,
            intermediate_size,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.fc_out = nnx.Linear(
            intermediate_size,
            embed_dim,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.act = ACT2FN[config.activation_function]
        self.dropout = nnx.Dropout(rate=config.resid_pdrop)

    def __call__(self, hidden_states):
        hidden_states = self.dropout(self.fc_out(self.act(self.fc_in(hidden_states))))
        return hidden_states


class GPTJBlock(nnx.Module):
    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        hidden_size = self.config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        attn_block = GPTJAttention
        mlp_block = GPTJMLP

        self.ln_1 = nnx.LayerNorm(
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.attn = attn_block(
            config,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp = mlp_block(
            config,
            inner_dim,
            dtype=dtype,
            param_dtype=dtype,
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
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, attn_weight = self.attn(
            hidden_states,
            freqs_cis,
            attention_mask,
            position_ids,
            past_key_values,
            segment_ids,
        )
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = attn_output + feed_forward_hidden_states + residual

        return (hidden_states, attn_weight)


class GPTJModel(nnx.Module):
    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = config.hidden_size
        self.wte = nnx.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=nnx.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(
            rate=self.config.embd_pdrop,
            rngs=rngs,
        )
        self.h = [
            GPTJBlock(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.ln_f = nnx.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        self._causal_mask = None
        self._freqs_cis = None

    @property
    def freqs_cis(self):
        if self._freqs_cis is None:
            self._freqs_cis = create_sinusoidal_positions(
                self.config.max_position_embeddings,
                self.config.rotary_dim or self.config.hidden_size,
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
            input_embeds = self.wte(input_ids.astype("i4"))
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

        hidden_states = self.dropout(input_embeds)

        for idx, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weight = block(
                attention_mask=attention_mask,
                freqs_cis=self.freqs_cis,
                hidden_states=hidden_states,
                past_key_values=(
                    past_key_values[idx] if past_key_values is not None else None
                ),
                position_ids=position_ids,
                segment_ids=segment_ids,
            )
            if output_attentions:
                all_attentions += (attn_weight,)

        hidden_states = self.ln_f(hidden_states)

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

    def get_input_embeddings(self):
        return self.module.wte

    def set_input_embeddings(self, value):
        self.module.wte = value


class GPTJForCausalLM(nnx.Module):
    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.transformer = GPTJModel(
            config,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            config.hidden_size,
            self.config.vocab_size,
            dtype=dtype,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_range),
            param_dtype=dtype,
            precision=precision,
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
            extra_embedding=extra_embedding,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.wte.embedding.value.T
            self.lm_head.kernel.value = shared_kernel
            lm_logits = self.lm_head(hidden_states)

        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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

    @property
    def can_generate(self):
        return True
