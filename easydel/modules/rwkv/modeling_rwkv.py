# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import chex
import jax.lax
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import ModelOutput
from easydel.layers.linear import ParallelLinear

from .rwkv_configuration import RwkvConfig as RwkvConfig

# NOTE:Updated but wont work forsure, check this later.


@auto_pytree
class RwkvOutput(ModelOutput):
    last_hidden_state: chex.Array = None
    state: tuple[chex.Array, ...] | None = None
    hidden_states: tuple[chex.Array, ...] | None = None
    attentions: tuple[chex.Array, ...] | None = None


@auto_pytree
class RwkvCausalLMOutput(ModelOutput):
    logits: chex.Array = None
    state: list[chex.Array] | None = None
    hidden_states: tuple[chex.Array, ...] | None = None
    attentions: tuple[chex.Array, ...] | None = None


def init_state(hidden_size):
    zeros = jnp.zeros(hidden_size)
    min_values = jnp.full(hidden_size, -jnp.inf)
    time_mix_state = (zeros, zeros, zeros, min_values)
    channel_mix_state = zeros
    return time_mix_state, channel_mix_state


def rwkv_linear_attention(
    time_decay,
    time_first,
    key,
    value,
    state=None,
    return_state=False,
):
    current_sequence_length = key.shape[1]
    output = jnp.zeros_like(key)

    if state is None:
        num_state = jnp.zeros_like(key[:, 0], dtype=jnp.float32)
        den_state = jnp.zeros_like(key[:, 0], dtype=jnp.float32)
        max_state = jnp.zeros_like(key[:, 0], dtype=jnp.float32) - 1e38
    else:
        num_state, den_state, max_state = state

    time_decay = -jnp.exp(time_decay)

    for current_index in range(current_sequence_length):
        current_key = key[:, current_index].float()
        current_value = value[:, current_index]

        max_for_output = jnp.maximum(max_state, current_key + time_first)
        e1 = jnp.exp(max_state - max_for_output)
        e2 = jnp.exp(current_key + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_value
        denominator = e1 * den_state + e2
        output[:, current_index] = (numerator / denominator).astype(output.dtype)

        max_for_state = jnp.maximum(max_state + time_decay, current_key)
        e1 = jnp.exp(max_state + time_decay - max_for_state)
        e2 = jnp.exp(current_key - max_for_state)
        num_state = e1 * num_state + e2 * current_value
        den_state = e1 * den_state + e2
        max_state = max_for_state

    if return_state or state is not None:
        state = [num_state, den_state, max_state]

    return output, state


class RwkvSelfAttention(nn.Module):
    def __init__(
        self,
        config: RwkvConfig,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.config = config
        self.layer_id = layer_id
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        num_hidden_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size if config.attention_hidden_size is not None else hidden_size
        self.attention_hidden_size = attention_hidden_size

        ratio_0_to_1 = layer_id / (num_hidden_layers - 1)
        ratio_1_to_almost_0 = 1.0 - (layer_id / num_hidden_layers)
        zigzag = 0.5 * (jnp.arange(1, hidden_size + 1) % 3 - 1)
        time_first = jnp.full(hidden_size, math.log(0.3)) + zigzag
        h = jnp.arange(0, hidden_size)
        time_decay = -5 + 8 * (h / (hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        x = jnp.arange(hidden_size) / hidden_size

        time_mix_key = jnp.power(x, ratio_1_to_almost_0)
        time_mix_value = time_mix_key + 0.3 * ratio_0_to_1
        time_mix_receptance = jnp.power(x, 0.5 * ratio_1_to_almost_0)

        self.time_decay = nn.Param(time_decay.astype(self.param_dtype))
        self.time_first = nn.Param(time_first.astype(self.param_dtype))
        self.time_mix_key = nn.Param(time_mix_key.astype(self.param_dtype))
        self.time_mix_value = nn.Param(time_mix_value.astype(self.param_dtype))
        self.time_mix_receptance = nn.Param(time_mix_receptance.astype(self.param_dtype))

        self.key = ParallelLinear(
            hidden_size,
            attention_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.value = ParallelLinear(
            hidden_size,
            attention_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.receptance = ParallelLinear(
            hidden_size,
            attention_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.output = ParallelLinear(
            attention_hidden_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden: chex.Array,
        state: tuple[chex.Array, chex.Array, chex.Array, chex.Array],
    ):
        sx, aa, bb, pp = state
        c_x = jnp.concatenate(
            (jnp.expand_dims(sx, 0), hidden[:-1, :]),
        )
        key_x = hidden * self.time_mix_key.reshape(-1) + c_x * (1 - self.time_mix_key.reshape(-1))
        value_x = hidden * self.time_mix_value.reshape(-1) + c_x * (1 - self.time_mix_value.reshape(-1))
        receptance_x = hidden * self.time_mix_receptance.reshape(-1) + c_x * (1 - self.time_mix_receptance.reshape(-1))
        receptance_state = nn.sigmoid(self.receptance(receptance_x))
        key_states = self.key(key_x)
        value_states = self.value(value_x)

        def step(in_state, kv):
            (inner_aa, inner_bb, inner_p), (kk, vv) = in_state, kv
            ww = self.time_first.reshape(-1) + kk
            p = jnp.maximum(inner_p, ww)
            e1 = jnp.exp(inner_p - p)
            e2 = jnp.exp(ww - p)
            next_c_x = ((e1 * inner_aa + e2 * vv) / (e1 * inner_bb + e2)).astype(dtype=receptance_state.dtype)

            ww = -jnp.exp(self.time_decay.reshape(-1)) + inner_p
            p = jnp.maximum(ww, kk)
            e1 = jnp.exp(ww - p)
            e2 = jnp.exp(kk - p)
            inner_aa = e1 * inner_aa + e2 * vv
            inner_bb = e1 * inner_bb + e2
            inner_p = p
            next_inner_state = (inner_aa, inner_bb, inner_p)
            return next_inner_state, next_c_x

        (aa, bb, pp), c_x = jax.lax.scan(step, (aa, bb, pp), (key_states, value_states))
        out = hidden + self.output(receptance_state * c_x)
        next_state = (hidden[-1, :], aa, bb, pp)
        return out, next_state


class RwkvFeedForward(nn.Module):
    def __init__(
        self,
        config: RwkvConfig,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.config = config
        self.layer_id = layer_id
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        hidden_size = config.hidden_size
        layer_id = self.layer_id
        num_hidden_layers = self.config.num_hidden_layers
        intermediate_size = config.intermediate_size if config.intermediate_size is not None else 4 * config.hidden_size

        x = jnp.arange(hidden_size) / hidden_size

        ratio_1_to_almost_0 = 1.0 - (layer_id / num_hidden_layers)
        time_mix_key = jnp.power(x, ratio_1_to_almost_0)
        time_mix_receptance = jnp.power(x, 0.5 * ratio_1_to_almost_0)
        self.time_mix_key = nn.Param(time_mix_key.astype(self.param_dtype))
        self.time_mix_receptance = nn.Param(time_mix_receptance.astype(self.param_dtype))

        self.key = ParallelLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.receptance = ParallelLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.value = ParallelLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden, state):
        sx = jnp.concatenate((jnp.expand_dims(state, 0), hidden[:-1, :]))
        xk = hidden * self.time_mix_key.reshape(-1) + sx * (1 - self.time_mix_key.reshape(-1))
        xr = hidden * self.time_mix_receptance.reshape(-1) + sx * (1 - self.time_mix_receptance.reshape(-1))
        r = nn.sigmoid(self.receptance(xr))
        k = jnp.square(nn.relu(self.key(xk)))
        return r * self.value(k), hidden[-1, :]


class SingleStandRwkvBlock(nn.Module):
    def __init__(
        self,
        config: RwkvConfig,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.config = config
        self.layer_id = layer_id
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(
                epsilon=config.layer_norm_epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        self.ln1 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.ln2 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.attention = RwkvSelfAttention(
            config=config,
            layer_id=layer_id,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.feed_forward = RwkvFeedForward(
            config=config,
            layer_id=layer_id,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

    def __call__(self, hidden, state=None, output_attentions: bool = False):
        if state is None:
            state = init_state(self.config.hidden_size)

        self_state, ffd_state = state
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)

        attention, self_state = self.attention(
            self.ln1(hidden),
            state=self_state,
        )
        hidden = hidden + attention

        feed_forward, ffd_state = self.feed_forward(self.ln2(hidden), state=ffd_state)
        hidden = hidden + feed_forward

        outputs = (hidden, (self_state, ffd_state))
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs


RwkvBlock = nn.vmap(SingleStandRwkvBlock, in_axes=0, out_axes=0)


@register_module(TaskType.BASE_MODULE, config=RwkvConfig, model_type="rwkv")
class RwkvModel(EasyDeLBaseModule):
    def __init__(
        self,
        config: RwkvConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.blocks = self.blocks = [
            RwkvBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_id=idx,
                rngs=rngs,
            )
            for idx in range(self.config.num_hidden_layers)
        ]

        self.layers_are_rescaled = False
        self.deterministic = True
        self.ln_out = nn.LayerNorm(
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        state: list[chex.Array] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple | RwkvOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.deterministic else False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and state is None:
            shape = (
                inputs_embeds.shape[0],
                self.config.hidden_size,
                self.config.num_hidden_layers,
            )
            state = [
                jnp.zeros(
                    *shape,
                    dtype=inputs_embeds.dtype if i <= 1 else jnp.float32,
                )
                for i in range(5)
            ]
            state[4] -= 1e30

        hidden_states = inputs_embeds

        all_hidden_states = ()
        all_self_attentions = ()

        for idx, block in enumerate(self.blocks):
            hidden_states, state, attentions = block(hidden_states, state=state, output_attentions=output_attentions)

            if self.layers_are_rescaled and self.config.rescale_every > 0 and (idx + 1) % self.config.rescale_every == 0:
                hidden_states = hidden_states / 2

            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            if output_attentions:
                all_self_attentions = (*all_self_attentions, attentions)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return RwkvOutput(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@register_module(TaskType.CAUSAL_LM, config=RwkvConfig, model_type="rwkv")
class RwkvForCausalLM(EasyDeLBaseModule):
    def __init__(
        self,
        config: RwkvConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.rwkv = RwkvModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.head = ParallelLinear(
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
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        state: list[chex.Array] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple | RwkvCausalLMOutput:
        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = rwkv_outputs[0]

        logits = self.head(hidden_states)

        return RwkvCausalLMOutput(
            logits=logits,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )
