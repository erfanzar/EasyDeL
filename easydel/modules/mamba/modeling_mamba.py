# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


import functools
import itertools
import typing as tp

import jax
import jax.numpy as jnp
from eformer.common_types import Replicated
from eformer.pytree import auto_pytree
from einops import repeat
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array

from easydel.caching import RecurrentCache, RecurrentCacheConfig, RecurrentCacheView
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import ColumnParallelLinear, Embed
from easydel.layers import RMSNorm as MambaRMSNorm
from easydel.modules._base import BaseCausalLMModule
from easydel.operations import OperationMetadata
from easydel.operations.kernels import SSM1Op

from .mamba_configuration import MambaConfig as MambaConfig


def init_to_value(x, dtype):
    """Return initializer that fills parameters with a broadcasted constant."""
    return lambda _, shape, dtype: jnp.broadcast_to(jnp.asarray(x, dtype=dtype), shape)


@auto_pytree
class MambaOutput(BaseModelOutput):
    """Output container for the base Mamba model with cached state."""

    last_hidden_state: Array = None
    cache: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None


@auto_pytree
class MambaCausalLMOutput(BaseModelOutput):
    """Causal LM output including logits and cache for Mamba decoding."""

    logits: Array = None
    cache: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None
    last_hidden_state: Array | None = None


_T = tp.TypeVar("_T")


def create_tuple_parser(
    n: int,
) -> tp.Callable[[_T | tp.Sequence[_T]], tuple[_T, ...]]:
    """Normalize a scalar or sequence into a tuple of length ``n``."""

    def parse(x: _T | tp.Sequence[_T]) -> tuple[_T, ...]:
        if isinstance(x, tp.Sequence):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(f"x!=n ({x}!=({n}))")
        else:
            return tuple(itertools.repeat(x, n))

    return parse


class Lambda(nn.Module):
    """Convenience wrapper to insert callables into module pipelines."""

    fn: tp.Callable

    def __call__(self, x, **kwargs):
        return self.fn(x, **kwargs)


class MambaConv1D(nn.Module):
    """Minimal 1D convolution layer for Mamba mixer implementation.

    Implements a depthwise separable 1D convolution used in the Mamba
    selective state space model for local context aggregation before
    the SSM layer.
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
        """Initialize Mamba 1D convolution layer.

        Args:
            features (int): Number of output features (channels).
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 1.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to input. Defaults to 0.
            dilation (int, optional): Dilation rate for the kernel. Defaults to 1.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            use_bias (bool, optional): Whether to include a bias term. Defaults to True.
            num_spatial_dims (int, optional): Number of spatial dimensions. Defaults to 1.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
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

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for convolution parameters."""
        specs = {"kernel": Replicated}
        if getattr(self, "use_bias", False) and hasattr(self, "bias"):
            specs["bias"] = Replicated
        return specs

    def __call__(self, x: Array) -> Array:
        """Apply 1D convolution to input tensor.

        Args:
            x (Array): Input tensor of shape (batch, features, sequence_length).

        Returns:
            Array: Convolved output tensor of shape (batch, features, output_length).

        Raises:
            ValueError: If input tensor rank does not match expected rank.
        """
        unbatched_rank = self.num_spatial_dims + 2
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank}, but input has shape {x.shape}.",
            )
        org_x_dtype = x.dtype
        x = lax.conv_general_dilated(
            lhs=x.astype(self.dtype),
            rhs=jnp.asarray(jnp.swapaxes(self.kernel.value, 0, 2), dtype=self.dtype),
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
        )

        if self.use_bias:
            x = x + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)

        return x.astype(org_x_dtype)


class MambaMixer(nn.Module):
    """Core selective state space mixer for Mamba blocks.

    Implements the selective state space model (SSM) with input-dependent
    dynamics that enables content-aware sequence modeling. This is the
    key component that differentiates Mamba from traditional SSMs by
    making the state transitions depend on the input sequence.

    Attributes:
        config (MambaConfig): Model configuration.
        layer_idx (int): Index of this layer in the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Numerical precision for operations.
    """

    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Mamba selective state space mixer.

        Args:
            config (MambaConfig): Model configuration with SSM parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        hidden_size = config.hidden_size
        ssm_state_size = config.state_size
        intermediate_size = config.intermediate_size
        time_step_rank = config.time_step_rank
        conv_kernel_size = config.conv_kernel

        self.conv1d = MambaConv1D(
            features=intermediate_size,
            use_bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=intermediate_size,
            padding=config.conv_kernel - 1,
            rngs=rngs,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

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
            dt = jax.lax.clamp(
                config.time_step_floor,
                jnp.exp(
                    jax.random.normal(
                        key=key,
                        shape=shape,
                        dtype=jnp.float32,
                    )
                    * (jnp.log(config.time_step_max) - jnp.log(config.time_step_min))
                    + jnp.log(config.time_step_min)
                ),
                config.time_step_max,
            )
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt.astype(dtype)

        linear_class = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.in_proj = linear_class(
            hidden_size,
            intermediate_size * 2,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.x_proj = linear_class(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            use_bias=False,
            rngs=rngs,
        )
        self.dt_proj = linear_class(
            time_step_rank,
            intermediate_size,
            use_bias=True,
            kernel_init=init_kernel_dt,
            bias_init=init_bias_dt,
            rngs=rngs,
        )
        self.out_proj = linear_class(
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

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for state space parameters."""
        return {
            "A_log": Replicated,
            "D": Replicated,
        }

    def __call__(
        self,
        input_states: Array,
        cache: RecurrentCacheView | None = None,
        position_ids: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, RecurrentCacheView | None]:
        """Apply selective state space transformation.

        Processes input through the SSM with input-dependent B, C, and delta
        parameters, enabling content-aware sequence modeling with efficient
        linear-time complexity.

        Args:
            input_states (Array): Input hidden states of shape (batch, seq_len, hidden_dim).
            cache (RecurrentCacheView | None, optional): Cache containing previous
                convolution and SSM states for incremental decoding. Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens.
                Defaults to None.
            attention_mask (Array | None, optional): Mask to avoid processing
                padding tokens. Defaults to None.

        Returns:
            tuple[Array, RecurrentCacheView | None]: Tuple containing:
                - Contextualized hidden states of shape (batch, seq_len, hidden_dim)
                - Updated cache view for subsequent decoding steps
        """
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = checkpoint_name(self.in_proj(input_states), name="ssm_input_proj")
        projected_states = jnp.swapaxes(projected_states, 2, 1)
        hidden_states, gate = jnp.split(projected_states, 2, axis=1)

        if attention_mask is not None:
            hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

        cache_view = cache
        if cache is not None and cache.recurrent_state is not None:
            ssm_state = cache.recurrent_state
        else:
            ssm_state = jnp.zeros((batch_size, self.intermediate_size, self.ssm_state_size), dtype=dtype)

        is_inference = seq_len == 1 and cache is not None

        if is_inference and cache is not None and cache.conv_state is not None:
            new_hidden = hidden_states[:, :, 0]
            conv_state, _, cache_view = cache.concatenate_to_cache(conv_state=new_hidden)

            kernel = jnp.asarray(jnp.swapaxes(self.conv1d.kernel.value, 0, 2), dtype=dtype)
            kernel = kernel[:, 0, :]
            hidden_states = jnp.sum(conv_state * kernel[None, :, :], axis=-1)
            if self.conv1d.use_bias:
                hidden_states = hidden_states + jnp.asarray(self.conv1d.bias.value, dtype=dtype)[None, :]
            hidden_states = jnp.expand_dims(self.act(hidden_states).astype(dtype), -1)
        else:
            conv_input = hidden_states
            conv_out = self.conv1d(conv_input)[..., :seq_len]
            hidden_states = self.act(conv_out).astype(dtype)

            if cache is not None:
                if seq_len >= self.conv_kernel_size:
                    new_conv_state = conv_input[:, :, -self.conv_kernel_size :]
                else:
                    pad_width = self.conv_kernel_size - seq_len
                    new_conv_state = jnp.pad(conv_input, ((0, 0), (0, 0), (pad_width, 0)))

                cache_view = cache.replace(conv_state=new_conv_state) if cache_view is not None else cache_view

        if attention_mask is not None:
            hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

        ssm_parameters = checkpoint_name(self.x_proj(jnp.swapaxes(hidden_states, 2, 1)), name="ssm_x_proj")
        time_step, B, C = jnp.split(
            ssm_parameters,
            [
                self.time_step_rank,
                self.ssm_state_size + self.time_step_rank,
            ],
            axis=-1,
        )

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
            ssm_state=ssm_state,
            activation=self.activation,
        )

        scan_output = jnp.swapaxes(ssm_output.attention_outputs, 1, 2)

        if cache_view is not None:
            cache_view = cache_view.replace(recurrent_state=ssm_output.ssm_state)

        contextualized_states = checkpoint_name(self.out_proj(jnp.swapaxes(scan_output, 2, 1)), name="ssm_output_proj")
        return contextualized_states, cache_view


class MambaBlock(nn.Module):
    """Single Mamba layer combining normalization, SSM mixer, and residual connections.

    Implements a complete Mamba block with pre-normalization architecture,
    applying RMS normalization followed by the selective state space mixer
    and a residual connection.

    Attributes:
        config (MambaConfig): Model configuration.
        layer_idx (int): Index of this layer in the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Numerical precision for operations.
    """

    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Mamba block.

        Args:
            config (MambaConfig): Model configuration.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        block = auto_remat(
            MambaMixer,
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
        cache: RecurrentCacheView | None = None,
        position_ids: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, RecurrentCacheView | None]:
        """Forward pass through the Mamba block.

        Applies pre-normalization architecture: x + mixer(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            cache (RecurrentCacheView | None, optional): Cache for incremental decoding.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens.
                Defaults to None.
            attention_mask (Array | None, optional): Mask to avoid processing
                padding tokens. Defaults to None.

        Returns:
            tuple[Array, RecurrentCacheView | None]: Tuple containing:
                - Output hidden states of shape (batch, seq_len, hidden_dim)
                - Updated cache view for subsequent decoding steps
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hidden_states, cache = self.mixer(
            hidden_states,
            cache,
            position_ids,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        return hidden_states, cache


@register_module(TaskType.BASE_MODULE, config=MambaConfig, model_type="mamba")
class MambaModel(EasyDeLBaseModule):
    """Mamba selective state space model implementation.

    Implements the Mamba architecture, a novel state space model that
    achieves linear-time sequence modeling through selective state spaces.
    Unlike transformers, Mamba uses content-aware state transitions that
    enable efficient processing of long sequences.

    Attributes:
        config (MambaConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: MambaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Mamba base model.

        Args:
            config (MambaConfig): Model configuration with SSM parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nn.List(
            [
                MambaBlock(
                    config=config,
                    layer_idx=layer_idx,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        cache: RecurrentCache | None = None,
        position_ids: Array | None = None,
        attention_mask: Array | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> tuple | MambaOutput:
        """Forward pass through the Mamba base model.

        Processes input tokens through embedding, all Mamba blocks with SSM,
        and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch, seq_len, hidden_size). Defaults to None.
            cache (RecurrentCache | None, optional): Cache with previous SSM and
                convolution states for incremental decoding. Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens, shape
                (batch, seq_len). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid processing
                padding tokens, shape (batch, seq_len). Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden
                states from all layers. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            MambaOutput: Contains last_hidden_state, optional all hidden_states,
                and updated cache.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        sequence_length = inputs_embeds.shape[1]
        if attention_mask is None:
            attention_mask = jnp.ones((inputs_embeds.shape[0], sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")

        mask_info = MaskInfo.dynamic_init(
            mask_info=None,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = mask_info.q_position_ids
        if cache is None:
            cache = RecurrentCache.init_empty(len(self.layers))

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.layers):
            hidden_states, cache_view = block(
                hidden_states=hidden_states,
                cache=cache.views[idx],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            cache[idx] = cache_view
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache=cache,
            hidden_states=all_hidden_states,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embeddings


@register_module(TaskType.CAUSAL_LM, config=MambaConfig, model_type="mamba")
class MambaForCausalLM(BaseCausalLMModule[MambaModel, MambaConfig]):
    """Mamba model with a language modeling head for causal language modeling tasks.

    This model combines the Mamba selective state space backbone with a
    linear language modeling head to perform autoregressive text generation
    with linear-time complexity.

    Attributes:
        config (MambaConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "mamba"
    _config_class = MambaConfig

    def __init__(
        self,
        config: MambaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Mamba model for causal language modeling.

        Args:
            config (MambaConfig): Model configuration with SSM parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=MambaModel,
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
        cache: RecurrentCache | None = None,
        position_ids: Array | None = None,
        apply_lm_head: bool = True,
        attention_mask: Array | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> tuple | MambaCausalLMOutput:
        """Forward pass for causal language modeling.

        Processes input through the Mamba backbone and optionally applies
        the language modeling head to produce next-token logits.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch, seq_len, hidden_size). Defaults to None.
            cache (RecurrentCache | None, optional): Cache with previous SSM and
                convolution states for incremental decoding. Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens.
                Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language modeling head.
                Defaults to True.
            attention_mask (Array | None, optional): Boolean mask to avoid processing
                padding tokens. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden
                states from all layers. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            MambaCausalLMOutput: Contains logits (if apply_lm_head), last_hidden_state,
                optional all hidden_states, and updated cache.
        """
        mamba_outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(mamba_outputs.last_hidden_state)

        return MambaCausalLMOutput(
            logits=logits,
            cache=mamba_outputs.cache,
            hidden_states=mamba_outputs.hidden_states,
            last_hidden_state=mamba_outputs.last_hidden_state,
        )

    def update_inputs_for_generation(
        self,
        outputs: MambaOutput,
        model_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Update model inputs for the next generation step.

        Extracts the updated cache from model outputs and adds it to the
        model kwargs for the next forward pass during autoregressive generation.

        Args:
            outputs (MambaOutput): Model outputs from the current generation step.
            model_kwargs (dict[str, tp.Any]): Current model keyword arguments.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            dict[str, tp.Any]: Updated model kwargs with the new cache state.
        """
        model_kwargs["cache"] = outputs.get("cache", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        **kwargs,
    ):
        """Prepare model inputs for text generation.

        Initializes or retrieves the recurrent cache and prepares all
        necessary inputs for the generation loop.

        Args:
            input_ids: Input token IDs to start generation from.
            max_length (int): Maximum sequence length for generation.
            pad_token_id (int): Token ID used for padding.
            starts (int | None, optional): Starting position for generation.
                Defaults to None.
            **kwargs: Additional keyword arguments including cache, attention_mask,
                and position_ids.

        Returns:
            dict: Prepared inputs including cache, attention_mask, and position_ids.
        """
        from eformer.escale import PartitionAxis

        from easydel.caching import RecurrentCache

        cache = kwargs.get("cache", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if cache is None:
            partition_axis = getattr(self.config, "partition_axis", None) or PartitionAxis()
            cache_config = RecurrentCacheConfig.create_for_mamba(
                num_hidden_layers=self.config.num_hidden_layers,
                partition_axis=partition_axis,
                batch_size=int(input_ids.shape[0]),
                intermediate_size=int(self.config.intermediate_size),
                ssm_state_size=int(self.config.state_size),
                conv_kernel_size=int(self.config.conv_kernel),
            )
            cache = RecurrentCache.init_cache(cache_config, dtype=self.dtype)

        return self.prepare_inputs_for_call(
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.backbone.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.backbone.get_embedding()
