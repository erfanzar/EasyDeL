# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple

import flax.linen as nn
import flax.linen.partitioning
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ..flax_modelling_utils import ACT2FN, with_sharding_constraint, \
    get_dot_general_by_bits, ACT2FN, BaseJAXAttentionModule, get_gradient_checkpoint_policy, block_wise_ffn
from .gpt2_configuration import GPT2Config
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"


class FlaxConv1D(nn.Module):
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")
    dot_general: Optional[None] = None

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", jax.nn.initializers.normal(stddev=0.02), (self.features, inputs.shape[-1]))
        kernel = jnp.asarray(kernel.transpose(), self.dtype)
        if self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = lax.dot_general

        y = dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        if self.use_bias:
            bias = self.param("bias", jax.nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y


class FlaxGPT2Attention(BaseJAXAttentionModule):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.is_cross_attention:
            self.c_attn = FlaxConv1D(
                2 * self.embed_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
            )
            self.q_attn = FlaxConv1D(
                self.embed_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
            )
        else:
            self.c_attn = FlaxConv1D(
                3 * self.embed_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
            )
        self.c_proj = FlaxConv1D(
            self.embed_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
            self,
            hidden_states,
            key_value_states: Optional[jnp.ndarray] = None,
            attention_mask=None,
            casual_mask=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]
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

        if self.config.use_pjit_attention_force:
            query = with_sharding_constraint(query, jax.sharding.PartitionSpec(("dp", "fsdp"), None, "tp"))
            key = with_sharding_constraint(key, jax.sharding.PartitionSpec(("dp", "fsdp"), None, "tp"))
            value = with_sharding_constraint(value, jax.sharding.PartitionSpec(("dp", "fsdp"), None, "tp"))
        query_length, key_length = query.shape[1], key.shape[1]

        if self.causal:
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    casual_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = casual_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # combine masks if needed
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # transform boolean mask into float mask
        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        # usual dot product attention
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )
        if self.config.use_pjit_attention_force:
            attn_weights = with_sharding_constraint(attn_weights,
                                                    jax.sharding.PartitionSpec(("dp", "fsdp"), "tp", None, None))
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxGPT2MLP(nn.Module):
    config: GPT2Config
    intermediate_size: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        embed_dim = self.config.hidden_size
        self.c_fc = FlaxConv1D(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.c_proj = FlaxConv1D(
            embed_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.act = ACT2FN[self.config.activation_function]
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxGPT2Block(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype
        )

        attn_block = FlaxGPT2Attention
        mlp_block = FlaxGPT2MLP
        if self.config.gradient_checkpointing != "":
            attn_block = flax.linen.partitioning.remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(3, 4, 5, 6)
            )

            mlp_block = flax.linen.partitioning.remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )
        # hidden_states,
        # key_value_states: Optional[jnp.ndarray] = None,
        # attention_mask = None,
        # casual_mask = None,
        # deterministic: bool = True,
        # init_cache: bool = False,
        # output_attentions: bool = False,

        self.attn = attn_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        if self.config.add_cross_attention:
            self.crossattention = attn_block(
                config=self.config,
                dtype=self.dtype,
                causal=False,
                is_cross_attention=True
            )
            self.ln_cross_attn = nn.LayerNorm(
                epsilon=self.config.layer_norm_epsilon,
                dtype=self.dtype
            )

        self.mlp = mlp_block(self.config, inner_dim, dtype=self.dtype)

    def __call__(
            self,
            hidden_states,
            attention_mask=None,
            casual_mask=None,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        # hidden_states
        # key_value_states: Optional[jnp.ndarray] = None
        # attention_mask = None
        # casual_mask = None
        # deterministic: bool = True
        # init_cache: bool = False
        # output_attentions: bool = False

        attn_outputs = self.attn(
            hidden_states,
            None,
            attention_mask,
            casual_mask,
            deterministic,
            init_cache,
            output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
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
            # key_value_states: Optional[jnp.ndarray] = None
            # attention_mask = None
            # casual_mask = None
            # deterministic: bool = True
            # init_cache: bool = False
            # output_attentions: bool = False

            cross_attn_outputs = self.crossattention(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                casual_mask,
                deterministic,
                False,
                output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
                deterministic
            )
        else:
            feed_forward_hidden_states = self.mlp(
                hidden_states,
                deterministic,
            )
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + outputs

        return outputs


class FlaxGPT2PreTrainedModel(EasyDelFlaxPretrainedModel):
    config_class = GPT2Config
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
            self,
            config: GPT2Config,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            _do_init: bool = True,
            **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        tie_word_embeddings = self.module.config.tie_word_embeddings
        self.module.config.tie_word_embeddings = False
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
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
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)
        self.module.config.tie_word_embeddings = tie_word_embeddings
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
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    def __call__(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            add_params_field: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if encoder_hidden_states is not None and encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng
        inputs = {"params": params or self.params} if add_params_field is True else params or self.params

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
            encoder_hidden_states,
            encoder_attention_mask,
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
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


class FlaxGPT2BlockCollection(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        self.blocks = [
            FlaxGPT2Block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            ) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states,
            attention_mask=None,
            casual_mask=None,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                casual_mask=casual_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        return outputs


class FlaxGPT2Module(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.casual_mask = make_causal_mask(
            jnp.ones((1, getattr(self.config, "c_max_position_embeddings", self.config.max_position_embeddings)),
                     dtype="bool"), dtype="bool"
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxGPT2BlockCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            deterministic=True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        inputs_embeds = self.wte(input_ids.astype("i4"))
        position_embeds = self.wpe(position_ids.astype("i4"))

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.h(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            casual_mask=self.casual_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
        )


class FlaxGPT2Model(FlaxGPT2PreTrainedModel):
    module_class = FlaxGPT2Module

    def get_input_embeddings(self):
        return self.module.wte

    def set_input_embeddings(self, value):
        self.module.wte = value


class FlaxGPT2LMHeadModule(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest")

    def setup(self):
        self.transformer = FlaxGPT2Module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
            encoder_hidden_states: Optional[jnp.ndarray] = None,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
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


class FlaxGPT2LMHeadModel(FlaxGPT2PreTrainedModel):
    module_class = FlaxGPT2LMHeadModule

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

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask.astype("i4"), (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
