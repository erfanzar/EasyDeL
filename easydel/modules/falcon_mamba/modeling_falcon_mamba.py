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
"""FalconMamba models (EasyDeL / JAX).

Implements:
    - `FalconMambaModel`: base decoder-only backbone.
    - `FalconMambaForCausalLM`: causal LM head on top of the backbone.

The implementation mirrors the HuggingFace reference (PyTorch) but uses
EasyDeL's NNX modules, sharding-aware linear layers, and the unified
`RecurrentCache` interface for decoding.

HuggingFace reference:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py
"""

import functools

import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from einops import repeat
from flax import nnx as nn
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import RecurrentCache, RecurrentCacheConfig, RecurrentCacheView
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm as FalconMambaRMSNorm
from easydel.layers.operations import OperationMetadata
from easydel.layers.operations.modules import SSM1Op

from .falcon_mamba_configuration import FalconMambaConfig


def rms_forward(hidden_states: Array, *, variance_epsilon: float = 1e-6) -> Array:
    """RMS normalize without learnable weights (HF parity helper)."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * lax.rsqrt(variance + variance_epsilon)
    return hidden_states.astype(input_dtype)


@auto_pytree
class FalconMambaOutput(BaseModelOutput):
    """Output type for `FalconMambaModel`."""

    last_hidden_state: Array = None
    cache_params: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None


@auto_pytree
class FalconMambaCausalLMOutput(BaseModelOutput):
    """Output type for `FalconMambaForCausalLM`."""

    logits: Array = None
    cache_params: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None


class Conv1D(nn.Module):
    """Depthwise causal 1D convolution used by the mixer.

    Parameter layout matches HF after conversion:
        - `kernel`: [kernel_size, 1, channels]
    """

    def __init__(
        self,
        features: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        num_spatial_dims: int = 1,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        kernel_shape = (kernel_size, 1, features)
        self.kernel = ArrayParam.bound(
            shape=kernel_shape,
            dtype=param_dtype,
            init_method="lecun_normal",
            key=rngs.params(),
        )

        if use_bias:
            self.bias = ArrayParam.bound(
                shape=(features,),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.params(),
            )

        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.num_spatial_dims = num_spatial_dims
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

    def __call__(self, x):
        unbatched_rank = self.num_spatial_dims + 2
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank}, but input has shape {x.shape}.",
            )

        org_x_dtype = x.dtype
        rhs = jnp.asarray(jnp.swapaxes(self.kernel.value, 0, 2), dtype=self.dtype)
        x = lax.conv_general_dilated(
            lhs=x.astype(self.dtype),
            rhs=rhs,
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
            precision=self.precision,
        )

        if self.use_bias:
            x = x + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)

        return x.astype(org_x_dtype)


class FalconMambaMixer(nn.Module):
    """Selective SSM mixer used by FalconMamba blocks.

    This is a faithful, naive implementation of the reference algorithm:
    - Causal depthwise conv over the expanded stream.
    - Input-dependent (dt, B, C) from projections.
    - Recurrent scan over sequence to update the SSM state.

    The "fast path" (CUDA kernels / mamba-ssm) is not implemented here; this
    version is intended for correctness and portability.
    """

    def __init__(
        self,
        config: FalconMambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        hidden_size = config.hidden_size
        ssm_state_size = config.state_size
        intermediate_size = config.intermediate_size
        time_step_rank = int(config.time_step_rank)
        conv_kernel_size = config.conv_kernel

        self.conv1d = Conv1D(
            features=intermediate_size,
            use_bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=intermediate_size,
            padding=config.conv_kernel - 1,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.rms_eps = config.mixer_rms_eps

        dt_init_std = time_step_rank**-0.5 * config.time_step_scale
        if config.time_step_init_scheme == "constant":
            init_kernel_dt = nn.initializers.constant(dt_init_std, dtype=param_dtype)
        elif config.time_step_init_scheme == "random":

            def init_kernel_dt(key, shape, dtype):
                return (
                    jax.nn.initializers.uniform(scale=dt_init_std * 2, dtype=param_dtype)(key, shape, dtype)
                    - dt_init_std
                )

        else:
            init_kernel_dt = nn.initializers.normal(config.initializer_range, param_dtype)

        def init_bias_dt(key, shape, dtype):
            # Match HF init: sample dt uniformly in [min, max] then invert softplus.
            dt = jnp.exp(
                jax.random.uniform(
                    key=key,
                    shape=shape,
                    dtype=jnp.float32,
                )
                * (jnp.log(config.time_step_max) - jnp.log(config.time_step_min))
                + jnp.log(config.time_step_min)
            )
            dt = jnp.clip(dt, a_min=config.time_step_floor)
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt.astype(dtype)

        column_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        row_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.in_proj = column_linear(
            hidden_size,
            intermediate_size * 2,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.x_proj = column_linear(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            use_bias=False,
            rngs=rngs,
        )
        self.dt_proj = column_linear(
            time_step_rank,
            intermediate_size,
            use_bias=True,
            kernel_init=init_kernel_dt,
            bias_init=init_bias_dt,
            rngs=rngs,
        )
        # Contracting projection is typically sharded ROW-wise.
        self.out_proj = row_linear(
            intermediate_size,
            hidden_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        A = repeat(jnp.arange(1, ssm_state_size + 1, dtype=jnp.float32), "n -> d n", d=intermediate_size)
        self.A_log = ArrayParam.bound(
            shape=A.shape,
            dtype=param_dtype,
            init_method="zeros",
            key=None,
            value=jnp.log(A).astype(param_dtype),
        )
        self.D = ArrayParam.bound(
            shape=(intermediate_size,),
            dtype=param_dtype,
            init_method="ones",
            key=None,
        )

        self.ssm_state_size = ssm_state_size
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = conv_kernel_size
        self.time_step_rank = time_step_rank

        metadata = OperationMetadata(
            runtime_dtype=dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=config,
        )
        self.ssm_op = SSM1Op(metadata)

    def __call__(
        self,
        input_states: Array,
        cache_params: RecurrentCacheView | None = None,
        cache_position: Array | None = None,
        attention_mask: Array | None = None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = checkpoint_name(self.in_proj(input_states), name="ssm_input_proj")
        projected_states = jnp.swapaxes(projected_states, 2, 1)
        hidden_states, gate = jnp.split(projected_states, 2, axis=1)

        if attention_mask is not None:
            hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

        cache_view = cache_params
        if cache_params is not None and cache_params.recurrent_state is not None:
            ssm_state0 = cache_params.recurrent_state.astype(jnp.float32)
        else:
            ssm_state0 = jnp.zeros((batch_size, self.intermediate_size, self.ssm_state_size), dtype=jnp.float32)

        is_inference = seq_len == 1 and cache_params is not None

        if is_inference and cache_params is not None and cache_params.conv_state is not None:
            new_hidden = hidden_states[:, :, 0]
            conv_state, _, cache_view = cache_params.concatenate_to_cache(conv_state=new_hidden)

            conv_out_full = self.conv1d(conv_state)[..., : self.conv_kernel_size]
            conv_out = conv_out_full[:, :, self.conv_kernel_size - 1]
            hidden_states = jnp.expand_dims(self.act(conv_out).astype(dtype), -1)
        else:
            conv_input = hidden_states
            conv_out = self.conv1d(conv_input)[..., :seq_len]
            hidden_states = self.act(conv_out).astype(dtype)

            if cache_params is not None:
                if seq_len >= self.conv_kernel_size:
                    new_conv_state = conv_input[:, :, -self.conv_kernel_size :]
                else:
                    pad_width = self.conv_kernel_size - seq_len
                    new_conv_state = jnp.pad(conv_input, ((0, 0), (0, 0), (pad_width, 0)))
                cache_view = cache_params.replace(conv_state=new_conv_state) if cache_view is not None else cache_view

        if attention_mask is not None:
            hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

        ssm_parameters = checkpoint_name(self.x_proj(jnp.swapaxes(hidden_states, 2, 1)), name="ssm_x_proj")
        time_step, B, C = jnp.split(
            ssm_parameters,
            [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1,
        )

        B = rms_forward(B, variance_epsilon=self.rms_eps)
        C = rms_forward(C, variance_epsilon=self.rms_eps)
        time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)

        discrete_time_step = checkpoint_name(self.dt_proj(time_step), name="ssm_dt_proj")
        discrete_time_step = jax.nn.softplus(discrete_time_step)

        hidden_states_t = jnp.swapaxes(hidden_states, 1, 2)
        gate_t = jnp.swapaxes(gate, 1, 2)

        ssm_output = self.ssm_op(
            hidden_states=hidden_states_t,
            A=self.A_log.value,
            B=B,
            C=C,
            D=self.D.value,
            discrete_time_step=discrete_time_step,
            gate=gate_t,
            ssm_state=ssm_state0,
            activation=self.activation,
        )

        y = jnp.swapaxes(ssm_output.attention_outputs, 1, 2)

        if cache_view is not None:
            cache_view = cache_view.replace(recurrent_state=ssm_output.ssm_state.astype(dtype))

        contextualized_states = checkpoint_name(self.out_proj(jnp.swapaxes(y, 2, 1)), name="ssm_output_proj")
        return contextualized_states, cache_view


class FalconMambaBlock(nn.Module):
    def __init__(
        self,
        config: FalconMambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = FalconMambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        block = auto_remat(
            FalconMambaMixer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.mixer = block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

    def __call__(
        self,
        hidden_states: Array,
        cache_params: RecurrentCacheView | None = None,
        cache_position: Array | None = None,
        attention_mask: Array | None = None,
    ) -> Array:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hidden_states, cache_params = self.mixer(
            hidden_states,
            cache_params,
            cache_position,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        return hidden_states, cache_params


@register_module(TaskType.BASE_MODULE, config=FalconMambaConfig, model_type="falcon_mamba")
class FalconMambaModel(EasyDeLBaseModule):
    """FalconMamba backbone (token embeddings + stacked mixer blocks)."""

    def __init__(
        self,
        config: FalconMambaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            FalconMambaBlock(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm_f = FalconMambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        cache_params: RecurrentCache | None = None,
        output_hidden_states: bool | None = None,
        cache_position: Array | None = None,
        attention_mask: Array | None = None,
        **kwargs,
    ) -> tuple | FalconMambaOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if cache_params is None:
            cache_params = RecurrentCache.init_empty(len(self.layers))
        if attention_mask is None:
            if input_ids is not None and input_ids.shape[1] > 1:
                pad_id = getattr(self.config, "pad_token_id", None)
                if pad_id is None:
                    eos_id = getattr(self.config, "eos_token_id", None)
                    if isinstance(eos_id, (list, tuple)) and eos_id:
                        eos_id = eos_id[0]
                    pad_id = eos_id

                if pad_id is not None:
                    pad_id = int(pad_id)
                    valid = input_ids != pad_id
                    starts = jnp.sum(jnp.cumsum(valid, axis=-1) == 0, axis=-1)
                    positions = jnp.arange(input_ids.shape[1], dtype=jnp.int32)[None, :]
                    attention_mask = (positions >= starts[:, None]).astype("i4")
                else:
                    attention_mask = jnp.ones(inputs_embeds.shape[:2], dtype="i4")
            else:
                attention_mask = jnp.ones(inputs_embeds.shape[:2], dtype="i4")

        hidden_states = inputs_embeds
        for idx, block in enumerate(self.layers):
            hidden_states, cache_view = block(
                hidden_states=hidden_states,
                cache_params=cache_params.views[idx],
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            cache_params[idx] = cache_view
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return FalconMambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params,
            hidden_states=all_hidden_states,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        return self.embeddings


@register_module(TaskType.CAUSAL_LM, config=FalconMambaConfig, model_type="falcon_mamba")
class FalconMambaForCausalLM(BaseCausalLMModule[FalconMambaModel, FalconMambaConfig]):
    """FalconMamba causal language model (backbone + LM head)."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "falcon_mamba"
    _config_class = FalconMambaConfig

    def __init__(
        self,
        config: FalconMambaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(
            config=config,
            base_model_class=FalconMambaModel,
            base_model_name="backbone",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )

    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        cache_params: RecurrentCache | None = None,
        output_hidden_states: bool | None = None,
        apply_lm_head: bool = True,
        cache_position: Array | None = None,
        attention_mask: Array | None = None,
        **kwargs,
    ) -> tuple | FalconMambaCausalLMOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(outputs.last_hidden_state)

        return FalconMambaCausalLMOutput(
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["cache_params"] = model_outputs.cache_params
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        **kwargs,
    ):
        from eformer.escale import PartitionAxis

        cache_params = kwargs.get("cache_params", None)
        attention_mask = kwargs.get("attention_mask", None)
        cache_position = kwargs.get("cache_position", None)
        if cache_params is None:
            partition_axis = getattr(self.config, "partition_axis", None) or PartitionAxis()
            cache_config = RecurrentCacheConfig.create_for_mamba(
                num_hidden_layers=int(self.config.num_hidden_layers),
                partition_axis=partition_axis,
                batch_size=int(input_ids.shape[0]),
                intermediate_size=int(self.config.intermediate_size),
                ssm_state_size=int(self.config.state_size),
                conv_kernel_size=int(self.config.conv_kernel),
            )
            cache_params = RecurrentCache.init_cache(cache_config, dtype=self.dtype)

        return self.prepare_inputs_for_call(
            cache_params=cache_params,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        return self.backbone.get_decoder()

    def get_lm_head(self):
        return self.lm_head

    def get_embedding(self):
        return self.backbone.get_embedding()
