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
import typing as tp
from functools import partial

import jax.lax
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import ModelOutput
from easydel.infra.utils import ArrayParam
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear

from .rwkv_configuration import RwkvConfig as RwkvConfig

# NOTE:Updated but wont work forsure, check this later.


@auto_pytree
class RwkvOutput(ModelOutput):
    """Output type for RWKV model."""

    last_hidden_state: Array = None
    state: tuple[Array, ...] | None = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None


@auto_pytree
class RwkvCausalLMOutput(ModelOutput):
    """Output type for RWKV causal language model."""

    loss: Array | None = None
    logits: Array = None
    state: list[Array] | None = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None


def init_state(hidden_size: int, batch_size: int | None = None):
    """Create zeroed RWKV recurrent state tensors for a given hidden size.

    RWKV is recurrent over the sequence dimension but can be vectorized over the batch
    dimension. When `batch_size` is provided, the state tensors include the batch axis.
    """
    if batch_size is None:
        zeros = jnp.zeros((hidden_size,))
        min_values = jnp.full((hidden_size,), -1e38, dtype=jnp.float32)
    else:
        zeros = jnp.zeros((batch_size, hidden_size))
        min_values = jnp.full((batch_size, hidden_size), -1e38, dtype=jnp.float32)
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
    """Compute RWKV linear attention update with optional recurrent state."""
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
    """RWKV self-attention mechanism with linear complexity."""

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
        zigzag = 0.5 * (jnp.arange(1, attention_hidden_size + 1) % 3 - 1)
        time_first = jnp.full((attention_hidden_size,), math.log(0.3), dtype=jnp.float32) + zigzag.astype(jnp.float32)
        h = jnp.arange(attention_hidden_size, dtype=jnp.float32)
        time_decay = -5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        x = jnp.arange(hidden_size) / hidden_size

        time_mix_key = jnp.power(x, ratio_1_to_almost_0)
        time_mix_value = time_mix_key + 0.3 * ratio_0_to_1
        time_mix_receptance = jnp.power(x, 0.5 * ratio_1_to_almost_0)

        self.time_decay = ArrayParam.bound(
            shape=time_decay.shape,
            dtype=jnp.float32,
            init_method="zeros",
            key=None,
            value=time_decay.astype(jnp.float32),
        )
        self.time_first = ArrayParam.bound(
            shape=time_first.shape,
            dtype=jnp.float32,
            init_method="zeros",
            key=None,
            value=time_first.astype(jnp.float32),
        )
        self.time_mix_key = ArrayParam.bound(
            shape=time_mix_key.shape,
            dtype=self.param_dtype,
            init_method="zeros",
            key=None,
            value=time_mix_key.astype(self.param_dtype),
        )
        self.time_mix_value = ArrayParam.bound(
            shape=time_mix_value.shape,
            dtype=self.param_dtype,
            init_method="zeros",
            key=None,
            value=time_mix_value.astype(self.param_dtype),
        )
        self.time_mix_receptance = ArrayParam.bound(
            shape=time_mix_receptance.shape,
            dtype=self.param_dtype,
            init_method="zeros",
            key=None,
            value=time_mix_receptance.astype(self.param_dtype),
        )

        self.key = ColumnParallelLinear(
            hidden_size,
            attention_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.value = ColumnParallelLinear(
            hidden_size,
            attention_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.receptance = RowParallelLinear(
            hidden_size,
            attention_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.output = RowParallelLinear(
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
        hidden: Array,
        state: tuple[Array, Array, Array, Array],
    ):
        sx, aa, bb, pp = state
        # `hidden` is (batch, seq, hidden) during training/testing.
        c_x = jnp.concatenate(
            (jnp.expand_dims(sx, 1), hidden[:, :-1, :]),
            axis=1,
        )
        key_x = hidden * self.time_mix_key.reshape(1, 1, -1) + c_x * (1 - self.time_mix_key.reshape(1, 1, -1))
        value_x = hidden * self.time_mix_value.reshape(1, 1, -1) + c_x * (1 - self.time_mix_value.reshape(1, 1, -1))
        receptance_x = hidden * self.time_mix_receptance.reshape(1, 1, -1) + c_x * (
            1 - self.time_mix_receptance.reshape(1, 1, -1)
        )
        receptance_state = checkpoint_name(nn.sigmoid(self.receptance(receptance_x)), "attn_receptance")
        key_states = checkpoint_name(self.key(key_x), "attn_key")
        value_states = checkpoint_name(self.value(value_x), "attn_value")

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

        # Scan over the sequence axis (time), keeping batch dimension in the carry.
        xs = (jnp.swapaxes(key_states, 0, 1), jnp.swapaxes(value_states, 0, 1))
        (aa, bb, pp), c_x = jax.lax.scan(step, (aa, bb, pp), xs)
        c_x = jnp.swapaxes(c_x, 0, 1)

        out = checkpoint_name(self.output(receptance_state * c_x), "attn_output")
        next_state = (hidden[:, -1, :], aa, bb, pp)
        return out, next_state


class RwkvFeedForward(nn.Module):
    """RWKV feedforward network with channel mixing."""

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
        # HF reference uses the same exponent for key + receptance in the channel-mix block.
        time_mix_receptance = jnp.power(x, ratio_1_to_almost_0)
        self.time_mix_key = ArrayParam.bound(
            shape=time_mix_key.shape,
            dtype=self.param_dtype,
            init_method="zeros",
            key=None,
            value=time_mix_key.astype(self.param_dtype),
        )
        self.time_mix_receptance = ArrayParam.bound(
            shape=time_mix_receptance.shape,
            dtype=self.param_dtype,
            init_method="zeros",
            key=None,
            value=time_mix_receptance.astype(self.param_dtype),
        )

        self.key = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.receptance = RowParallelLinear(
            hidden_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.value = ColumnParallelLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden, state):
        sx = jnp.concatenate((jnp.expand_dims(state, 1), hidden[:, :-1, :]), axis=1)
        xk = hidden * self.time_mix_key.reshape(1, 1, -1) + sx * (1 - self.time_mix_key.reshape(1, 1, -1))
        xr = hidden * self.time_mix_receptance.reshape(1, 1, -1) + sx * (1 - self.time_mix_receptance.reshape(1, 1, -1))
        r = checkpoint_name(nn.sigmoid(self.receptance(xr)), "mlp_gate")
        k = checkpoint_name(jnp.square(nn.relu(self.key(xk))), "mlp_up")
        return checkpoint_name(r * self.value(k), "mlp_output"), hidden[:, -1, :]


class SingleStandRwkvBlock(nn.Module):
    """Single RWKV transformer block with attention and feedforward layers."""

    def __init__(
        self,
        config: RwkvConfig,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: nn.Rngs = None,
    ) -> None:
        self.config = config
        self.layer_id = layer_id
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
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
            rngs=rngs,
        )
        self.feed_forward = RwkvFeedForward(
            config=config,
            layer_id=layer_id,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden, state=None, use_cache: bool = False, output_attentions: bool = False):
        uses_global_state = isinstance(state, list)
        if state is None:
            state = init_state(self.config.hidden_size, batch_size=hidden.shape[0])

        if uses_global_state:
            ffd_state = state[0][:, :, self.layer_id]
            self_state = (
                state[1][:, :, self.layer_id],
                state[2][:, :, self.layer_id],
                state[3][:, :, self.layer_id],
                state[4][:, :, self.layer_id],
            )
        else:
            self_state, ffd_state = state
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)

        attention, self_state = self.attention(
            self.ln1(hidden),
            state=self_state,
        )
        hidden = checkpoint_name(hidden + attention, "residual")

        feed_forward, ffd_state = self.feed_forward(self.ln2(hidden), state=ffd_state)
        hidden = checkpoint_name(hidden + feed_forward, "layer_output")

        if uses_global_state:
            state[0] = state[0].at[:, :, self.layer_id].set(ffd_state)
            state[1] = state[1].at[:, :, self.layer_id].set(self_state[0])
            state[2] = state[2].at[:, :, self.layer_id].set(self_state[1])
            state[3] = state[3].at[:, :, self.layer_id].set(self_state[2])
            state[4] = state[4].at[:, :, self.layer_id].set(self_state[3])
            outputs = (hidden, state)
        else:
            outputs = (hidden, (self_state, ffd_state))
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs


# Use SingleStandRwkvBlock directly since nn.vmap on class constructors has issues
# with keyword argument resolution in recent Flax versions
RwkvBlock = SingleStandRwkvBlock


@register_module(TaskType.BASE_MODULE, config=RwkvConfig, model_type="rwkv")
class RwkvModel(EasyDeLBaseModule):
    """RWKV base model with embedding and transformer blocks."""

    def __init__(
        self,
        config: RwkvConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        dtype = jnp.float32
        param_dtype = jnp.float32
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
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        state: list[Array] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple | RwkvOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if self.deterministic else False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embeddings(input_ids), "embeddings")

        if use_cache and state is None:
            shape = (
                inputs_embeds.shape[0],
                self.config.hidden_size,
                self.config.num_hidden_layers,
            )
            state = [jnp.zeros(shape, dtype=inputs_embeds.dtype if i <= 1 else jnp.float32) for i in range(5)]
            state[4] -= 1e30

        hidden_states = inputs_embeds

        all_hidden_states = ()
        all_self_attentions = ()

        for idx, block in enumerate(self.blocks):
            hidden_states, state, attentions = block(
                hidden_states,
                state=state,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if self.layers_are_rescaled and self.config.rescale_every > 0 and (idx + 1) % self.config.rescale_every == 0:
                hidden_states = hidden_states / 2

            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            if output_attentions:
                all_self_attentions = (*all_self_attentions, attentions)

        hidden_states = checkpoint_name(self.ln_out(hidden_states), "model_output")

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return RwkvOutput(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def get_embedding(self):
        """Returns the embedding layer of the module."""
        return self.embeddings

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self


@register_module(TaskType.CAUSAL_LM, config=RwkvConfig, model_type="rwkv")
class RwkvForCausalLM(BaseCausalLMModule[RwkvModel, RwkvConfig]):
    """RWKV model with language modeling head for causal generation."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "rwkv"
    _config_class = RwkvConfig

    def __init__(
        self,
        config: RwkvConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        dtype = jnp.float32
        param_dtype = jnp.float32
        super().__init__(
            config=config,
            base_model_class=RwkvModel,
            base_model_name="rwkv",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_name="head",
            lm_head_bias=False,
        )

    @property
    def transform_fn(self):
        from easydel.layers.moe import BaseMoeModule, ParallelMoELinear
        from easydel.utils import traversals
        from easydel.utils.parameters_transformation import StateDictConverter

        embedding_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.Embed)]
        layernorm_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.LayerNorm)]
        moe_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, ParallelMoELinear)]
        moe_block_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, BaseMoeModule)]

        def _keep_fp32(array: Array, key: tuple):
            if key and key[-1] in {"time_decay", "time_first"}:
                return array.astype(jnp.float32)
            return array

        return partial(
            StateDictConverter.huggingface_to_easydel,
            embedding_layer_names=embedding_path,
            layernorm_names=layernorm_path,
            moe_names=list(set([names.split(".")[-1] for names in moe_path])),
            moe_block_names=list(set([names.split(".")[-1] for names in moe_block_path])),
            moe_block_path=moe_block_path,
            moe_path=moe_path,
            dtype=self.param_dtype,
            shard_fns=self._shard_fns,
            reform_param=self._get_reform_param(),
            callback=_keep_fp32,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings: int | None = None,
        attention_mask: Array | None = None,
        token_type_ids: Array | None = None,
        mask_info=None,
        state: list[Array] | None = None,
        inputs_embeds: Array | None = None,
        **kwargs,
    ) -> dict[str, tp.Any]:
        del max_length, pad_token_id, starts, shardings, token_type_ids, mask_info, kwargs
        del inputs_embeds

        if state is None:
            shape = (
                input_ids.shape[0],
                self.config.hidden_size,
                self.config.num_hidden_layers,
            )
            state = [jnp.zeros(shape, dtype=jnp.float32) for _ in range(5)]
            state[4] = state[4] - 1e30

        model_inputs: dict[str, tp.Any] = {"state": state}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        return model_inputs

    def update_inputs_for_generation(
        self,
        model_outputs: RwkvCausalLMOutput,
        model_kwargs: dict[str, tp.Any],
    ) -> dict[str, tp.Any]:
        model_kwargs["state"] = model_outputs.state
        return model_kwargs

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        state: list[Array] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
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

        logits = self.apply_lm_head(hidden_states)

        return RwkvCausalLMOutput(
            logits=logits,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )

    def get_lm_head(self):
        """Returns the language modeling head."""
        return self.head

    def get_embedding(self):
        """Returns the embedding layer of the module."""
        return self.rwkv.get_embedding()

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self.rwkv
