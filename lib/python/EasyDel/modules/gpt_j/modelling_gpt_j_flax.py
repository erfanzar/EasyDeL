# Note
# the GPT-j implemented by Huggingface is not supporting Partition spec s and using
# fully with pjit and required creating
# parameters even if you want to load already trained model so this one is the same
# but include those bugs / non-features fixed


# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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
""" GPT-J model configuration"""
import math
from functools import partial
from typing import Optional, Tuple, Union

import flax.linen.partitioning
from einops import einops
from fjformer import with_sharding_constraint
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.utils import logging

from fjformer.pallas_operations.efficient_attention import efficient_attention

from ..easy_attention import EasyAttention
from ..flax_modelling_utils import with_sharding_constraint, ACT2FN, BaseJAXAttentionModule, \
    get_gradient_checkpoint_policy, block_wise_ffn
from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel
import chex
from fjformer.bits import config as q_config, q_flax
from .gpt_j_configuration import GPTJConfig

logger = logging.get_logger(__name__)


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out)


def rotate_every_two(tensor):
    rotate_half_tensor = jnp.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1)
    rotate_half_tensor = rotate_half_tensor.reshape(rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class FlaxGPTJAttention(BaseJAXAttentionModule):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = None
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.rotary_dim = config.rotary_dim
        if self.config.bits is not None:
            _dot_general_cls = q_config.fully_quantized(
                fwd_bits=self.config.bits,
                bwd_bits=self.config.bits
            )
        else:
            _dot_general_cls = None

        dot_general_cls = q_flax.QDotGeneral(_dot_general_cls)
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dot_general=dot_general_cls,
            param_dtype=self.dtype,
            precision=self.precision
        )

        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")

        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(config.max_position_embeddings, pos_embd_dim)

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
            attention_dropout=self.config.attn_pdrop,
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
            sm_scale=1 / math.sqrt(self.head_dim)
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
            self,
            hidden_states,
            attention_mask,
            position_ids,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Force A local Sharding
        if self.config.use_pjit_attention_force:
            query = with_sharding_constraint(query, PartitionSpec(("dp", "fsdp"), None, "sp"))
            key = with_sharding_constraint(key, PartitionSpec(("dp", "fsdp"), None, "sp"))
            value = with_sharding_constraint(value, PartitionSpec(("dp", "fsdp"), None, "sp"))

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        sincos = jnp.take(self.embed_positions, position_ids, axis=0)
        sincos = jnp.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            key = jnp.concatenate([k_rot, k_pass], axis=-1)
            query = jnp.concatenate([q_rot, q_pass], axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        query_length, key_length = query.shape[1], key.shape[1]
        # usual dot product attention
        attentions = self.attention_performer.__call__(
            query_states=query,
            key_states=key,
            value_states=value,
            bias=attention_bias,
            causal=False,
            use_pjit_attention_force=self.config.use_pjit_attention_force,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
        )
        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        outputs = (attn_output, attentions.attention_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxGPTJMLP(nn.Module):
    config: GPTJConfig
    intermediate_size: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = None

    def setup(self):
        embed_dim = self.config.hidden_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        if self.config.bits is not None:
            _dot_general_cls = q_config.fully_quantized(
                fwd_bits=self.config.bits,
                bwd_bits=self.config.bits
            )
        else:
            _dot_general_cls = None

        dot_general_cls = q_flax.QDotGeneral(_dot_general_cls)
        self.fc_in = nn.Dense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            dot_general=dot_general_cls
        )
        self.fc_out = nn.Dense(
            embed_dim,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            dot_general=dot_general_cls
        )

        self.act = ACT2FN[self.config.activation_function]
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxGPTJBlock(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = None

    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )
        attn_block = FlaxGPTJAttention

        mlp_block = FlaxGPTJMLP

        if self.config.gradient_checkpointing != "":
            # hidden_states
            # attention_mask
            # position_ids
            # deterministic: bool = True
            # init_cache: bool = False
            # output_attentions: bool = False
            attn_block = flax.linen.partitioning.remat(
                attn_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(3, 4, 5)
            )

            mlp_block = flax.linen.partitioning.remat(
                mlp_block,
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
                static_argnums=(1,)
            )
        self.attn = attn_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=self.precision
        )

        self.mlp = mlp_block(
            self.config,
            inner_dim,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=self.precision
        )

    def __call__(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # hidden_states
        # attention_mask
        # position_ids
        # deterministic: bool = True
        # init_cache: bool = False
        # output_attentions: bool = False
        attn_outputs = self.attn(
            hidden_states,
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
        )
        attn_output = attn_outputs[0]
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
                deterministic
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states, deterministic)
        # residual connection
        hidden_states = attn_output + feed_forward_hidden_states + residual

        return (hidden_states,) + attn_outputs[1:]


class FlaxGPTJPreTrainedModel(EasyDelFlaxPretrainedModel):
    config_class = GPTJConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
            self,
            config: GPTJConfig,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[Union[str, jax.lax.Precision]] = None,
            _do_init: bool = True,
            **kwargs,
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **kwargs
        )
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        if params is None:
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
            return module_init_outputs
        else:
            return params

    def init_cache(self, batch_size, max_length):

        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params} if add_params_field else params or self.params

        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False
        rngs['params'] = jax.random.key(0)
        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxGPTJBlockCollection(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = None

    def setup(self):
        self.blocks = [
            FlaxGPTJBlock(
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
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxGPTJModule(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = None

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxGPTJBlockCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ln_f = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
            deterministic=True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        inputs_embeds = self.wte(input_ids.astype("i4"))

        hidden_states = self.dropout(inputs_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
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

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxGPTJModel(FlaxGPTJPreTrainedModel):
    module_class = FlaxGPTJModule

    def get_input_embeddings(self):
        return self.module.wte

    def set_input_embeddings(self, value):
        self.module.wte = value


class FlaxGPTJForCausalLMModule(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = None

    def setup(self):
        if self.config.bits is not None:
            _dot_general_cls = q_config.fully_quantized(
                fwd_bits=self.config.bits,
                bwd_bits=self.config.bits
            )
        else:
            _dot_general_cls = None

        dot_general_cls = q_flax.QDotGeneral(_dot_general_cls)
        self.transformer = FlaxGPTJModule(
            self.config,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=self.precision
        )
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dot_general=dot_general_cls,
            param_dtype=self.dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            position_ids,
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

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxGPTJForCausalLM(FlaxGPTJPreTrainedModel):
    module_class = FlaxGPTJForCausalLMModule

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

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
