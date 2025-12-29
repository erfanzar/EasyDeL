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
import itertools
import typing as tp

import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import RecurrentCache, RecurrentCacheView
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm as FlaxMamba2RMSNorm
from easydel.layers.operations import OperationMetadata
from easydel.layers.operations.modules import SSM2Op

from .mamba2_configuration import Mamba2Config as Mamba2Config


def init_to_value(x, dtype):
    """Return a parameter initializer that fills tensors with a fixed value."""
    return lambda *_: x.astype(dtype)


@auto_pytree
class Mamba2Output(BaseModelOutput):
    """Output type for the base Mamba2 model including cache state."""

    last_hidden_state: Array = None
    cache_params: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None


@auto_pytree
class Mamba2CausalLMOutput(BaseModelOutput):
    """Causal language modeling output with logits and cached state."""

    logits: Array = None
    cache_params: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None


def pad_tensor_by_size(input_tensor: jnp.ndarray, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)
    """
    if input_tensor.ndim == 4:
        pad_width = [(0, 0), (0, pad_size), (0, 0), (0, 0)]
    else:
        pad_width = [(0, 0), (0, pad_size), (0, 0)]

    return jnp.pad(input_tensor, pad_width, mode="constant", constant_values=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.
    """
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if input_tensor.ndim == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(
            input_tensor.shape[0],
            -1,
            chunk_size,
            input_tensor.shape[2],
            input_tensor.shape[3],
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.shape[-1]
    input_tensor = jnp.expand_dims(input_tensor, axis=-1)
    input_tensor = jnp.tile(input_tensor, (1,) * (input_tensor.ndim - 1) + (chunk_size,))

    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
    input_tensor = jnp.where(mask, input_tensor, 0)

    tensor_segsum = jnp.cumsum(input_tensor, axis=-2)

    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
    tensor_segsum = jnp.where(mask, tensor_segsum, -jnp.inf)

    return tensor_segsum


_T = tp.TypeVar("_T")


def create_tuple_parser(
    n: int,
) -> tp.Callable[[_T | tp.Sequence[_T]], tuple[_T, ...]]:
    """Ensure a scalar or sequence is expanded into a tuple of length ``n``."""

    def parse(x: _T | tp.Sequence[_T]) -> tuple[_T, ...]:
        if isinstance(x, tp.Sequence):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(f"x!=n ({x}!=({n}))")
        else:
            return tuple(itertools.repeat(x, n))

    return parse


class Conv1D(nn.Module):
    """Lightweight 1D convolution wrapper used by the Mamba2 mixer."""

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
        """Initialize Conv1D layer.

        Args:
            features (int): Number of output features/channels.
            kernel_size (int, optional): Size of the convolving kernel. Defaults to 1.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding added to both sides of the input. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            groups (int, optional): Number of blocked connections from input to output. Defaults to 1.
            use_bias (bool, optional): Whether to add a learnable bias. Defaults to True.
            num_spatial_dims (int, optional): Number of spatial dimensions. Defaults to 1.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision for computations. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.kernel = ArrayParam.bound(
            shape=(kernel_size, 1, features),
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
        """Apply 1D convolution.

        Args:
            x (Array): Input tensor of shape [batch, features, length] for 1D convolution.

        Returns:
            Array: Convolved output with same shape as input (after considering stride/padding).

        Raises:
            ValueError: If input rank doesn't match expected rank (num_spatial_dims + 2).
        """
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
        )

        if self.use_bias:
            x = x + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)

        return x.astype(org_x_dtype)


class MambaRMSNormGated(nn.Module):
    """RMSNorm variant that optionally gates inputs before normalization."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize gated RMSNorm layer.

        Args:
            hidden_size (int): Dimensionality of the hidden states.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
        """
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.kernel = ArrayParam.bound(
            shape=(self.hidden_size,),
            dtype=self.dtype,
            init_method="ones",
            key=None,
        )

    def __call__(self, hidden_states, gate=None):
        """Apply gated RMS normalization.

        Args:
            hidden_states (Array): Input tensor of shape [batch, seq_len, hidden_size].
            gate (Array | None, optional): Gate tensor of same shape as hidden_states.
                If provided, applies SiLU gating before normalization. Defaults to None.

        Returns:
            Array: Normalized (and optionally gated) output with same shape as input.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)

        if gate is not None:
            gate = gate.astype(jnp.float32)
            hidden_states = hidden_states * jax.nn.silu(gate)

        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)

        return (self.kernel.value * hidden_states).astype(input_dtype)


class Mamba2Mixer(nn.Module):
    """Selective state space mixer powering the Mamba2 token mixing step."""

    def __init__(
        self,
        config: Mamba2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Mamba2Mixer.

        Args:
            config (Mamba2Config): Model configuration.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.norm_before_gate = config.norm_before_gate
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = Conv1D(
            features=self.conv_dim,
            kernel_size=self.config.conv_kernel,
            groups=self.conv_dim,
            stride=1,
            padding=self.config.conv_kernel - 1,
            use_bias=self.config.use_conv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads

        self.in_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            use_bias=self.config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        dt = jax.lax.clamp(
            self.config.time_step_floor,
            jnp.exp(
                jax.random.normal(
                    key=rngs.params(),
                    shape=(self.config.num_heads,),
                    dtype=self.param_dtype,
                )
                * (jnp.log(self.config.time_step_max) - jnp.log(self.config.time_step_min))
                + jnp.log(self.config.time_step_min)
            ).astype(jnp.float32),
            1e9,
        )

        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_bias = ArrayParam.bound(
            shape=inv_dt.shape,
            dtype=self.param_dtype,
            init_method="zeros",
            key=None,
            value=inv_dt.astype(self.param_dtype),
        )

        A_log_value = jnp.log(jnp.arange(1, self.num_heads + 1, dtype=jnp.float32)).astype(self.param_dtype)
        self.A_log = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=self.param_dtype,
            init_method="zeros",
            key=None,
            value=A_log_value,
        )
        self.D = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=self.param_dtype,
            init_method="ones",
            key=None,
        )

        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=self.layer_norm_epsilon,
            dtype=self.param_dtype,
        )
        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            use_bias=self.config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        metadata = OperationMetadata(
            runtime_dtype=dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=config,
        )
        self.ssm_op = SSM2Op(metadata)

    def __call__(
        self,
        input_states: Array,
        cache_params: RecurrentCacheView | None = None,
        cache_position: Array | None = None,
        attention_mask: Array | None = None,
    ):
        """Apply selective state space transformation.

        Args:
            input_states (Array): Input tensor of shape [batch, seq_len, hidden_size].
            cache_params (RecurrentCacheView | None, optional): Cache for efficient generation.
                Contains conv_state [batch, conv_dim, conv_kernel] and recurrent_state
                [batch, num_heads, head_dim, state_size]. Defaults to None.
            cache_position (Array | None, optional): Position indices for cache updates.
                Not used in current implementation. Defaults to None.
            attention_mask (Array | None, optional): Mask tensor of shape [batch, seq_len]
                where 1 indicates valid tokens and 0 indicates padding. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - contextualized_states (Array): Output tensor of shape [batch, seq_len, hidden_size].
                - cache_view (RecurrentCacheView | None): Updated cache state for next iteration.
        """
        dtype = input_states.dtype

        mask = None
        if (
            attention_mask is not None
            and attention_mask.shape[0] == input_states.shape[0]
            and attention_mask.shape[1] == input_states.shape[1]
            and attention_mask.shape[1] > 1
        ):
            mask = attention_mask.astype(dtype)
            input_states = (input_states * mask[:, :, None]).astype(dtype)

        batch_size, seq_len, _ = input_states.shape

        if self.num_heads % self.n_groups != 0:
            raise ValueError("Expected `num_heads` to be divisible by `n_groups` for Mamba2.")
        if self.intermediate_size != self.num_heads * self.head_dim:
            raise ValueError("Expected `intermediate_size == num_heads * head_dim` for Mamba2.")

        projected_states = checkpoint_name(self.in_proj(input_states), name="ssm_input_proj")
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, conv_in, dt = jnp.split(
            projected_states,
            [
                d_mlp,
                d_mlp * 2,
                d_mlp * 2 + self.intermediate_size,
                d_mlp * 2 + self.intermediate_size + self.conv_dim,
            ],
            axis=-1,
        )

        if mask is not None:
            gate = gate * mask[:, :, None]
            conv_in = conv_in * mask[:, :, None]
            dt = dt * mask[:, :, None]

        cache_view = cache_params

        # Convolution on conv_in (depthwise, causal). Cache stores conv input, not conv output.
        if seq_len == 1 and cache_params is not None and cache_params.conv_state is not None:
            new_token = conv_in[:, 0, :]  # [batch, conv_dim]
            conv_state, _, cache_view = cache_params.concatenate_to_cache(conv_state=new_token)

            conv_out_full = self.conv1d(conv_state)[..., : self.conv_kernel_size]  # [batch, conv_dim, k]
            conv_out = self.act(conv_out_full[:, :, self.conv_kernel_size - 1]).astype(dtype)[:, None, :]
        else:
            conv_out = self.conv1d(jnp.swapaxes(conv_in, 2, 1))[..., :seq_len]
            conv_out = self.act(jnp.swapaxes(conv_out, 2, 1)).astype(dtype)  # [batch, seq_len, conv_dim]

            if cache_params is not None:
                conv_in_t = conv_in.transpose(0, 2, 1)  # [batch, conv_dim, seq_len]
                if seq_len >= self.conv_kernel_size:
                    new_conv_state = conv_in_t[:, :, -self.conv_kernel_size :]
                else:
                    pad_width = self.conv_kernel_size - seq_len
                    new_conv_state = jnp.pad(conv_in_t, ((0, 0), (0, 0), (pad_width, 0)))
                cache_view = cache_params.replace(conv_state=new_conv_state)

        if mask is not None:
            conv_out = conv_out * mask[:, :, None]

        x, B, C = jnp.split(
            conv_out,
            [self.intermediate_size, self.intermediate_size + self.n_groups * self.ssm_state_size],
            axis=-1,
        )

        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).astype(jnp.float32)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).astype(jnp.float32)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).astype(jnp.float32)
        # Note: SSM2Op handles group expansion internally

        # Prepare dt with bias and clipping
        dt = dt.astype(jnp.float32)
        dt = jax.nn.softplus(dt + self.dt_bias.value.astype(jnp.float32))
        dt = jnp.clip(dt, self.time_step_limit[0], self.time_step_limit[1])

        if cache_params is not None and cache_params.recurrent_state is not None:
            ssm_state0 = cache_params.recurrent_state.astype(jnp.float32)
        else:
            ssm_state0 = None

        # Call SSM2Op
        ssm_output = self.ssm_op(
            x=x,  # [batch, seq_len, num_heads, head_dim]
            A=self.A_log.value,  # [num_heads] in log form
            B=B,  # [batch, seq_len, n_groups, ssm_state_size]
            C=C,  # [batch, seq_len, n_groups, ssm_state_size]
            D=self.D.value,  # [num_heads]
            dt=dt,  # [batch, seq_len, num_heads]
            gate=None,  # Gating handled by self.norm below
            ssm_state=ssm_state0,
            n_groups=self.n_groups,
            use_gated_rmsnorm=False,  # We use self.norm below
            precision=self.precision,
        )

        y = ssm_output.attention_outputs

        if cache_view is not None:
            cache_view = cache_view.replace(recurrent_state=ssm_output.ssm_state.astype(dtype))

        scan_output = self.norm(y, gate)
        contextualized_states = checkpoint_name(self.out_proj(scan_output.astype(dtype)), name="ssm_output_proj")
        return contextualized_states, cache_view


class Mamba2Block(nn.Module):
    """Single Mamba2 layer combining normalization, mixer, and residual path."""

    def __init__(
        self,
        config: Mamba2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Mamba2Block.

        Args:
            config (Mamba2Config): Model configuration.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = FlaxMamba2RMSNorm(
            dim=config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        block = Mamba2Mixer
        block = auto_remat(
            block,
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
        """Apply Mamba2 block with pre-normalization and residual connection.

        Args:
            hidden_states (Array): Input tensor of shape [batch, seq_len, hidden_size].
            cache_params (RecurrentCacheView | None, optional): Cache for generation. Defaults to None.
            cache_position (Array | None, optional): Position indices for cache. Defaults to None.
            attention_mask (Array | None, optional): Attention mask of shape [batch, seq_len]. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - hidden_states (Array): Output tensor of shape [batch, seq_len, hidden_size].
                - cache_params (RecurrentCacheView | None): Updated cache state.
        """
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


@register_module(TaskType.BASE_MODULE, config=Mamba2Config, model_type="mamba2")
class Mamba2Model(EasyDeLBaseModule):
    """Stacked Mamba2 mixer blocks with token embeddings and final norm."""

    def __init__(
        self,
        config: Mamba2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Mamba2Model.

        Args:
            config (Mamba2Config): Model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
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
        self.embeddings = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Mamba2Block(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm_f = FlaxMamba2RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
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
    ) -> tuple | Mamba2Output:
        """Forward pass through Mamba2 model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape [batch, seq_len].
                Mutually exclusive with inputs_embeds. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                [batch, seq_len, hidden_size]. Mutually exclusive with input_ids. Defaults to None.
            cache_params (RecurrentCache | None, optional): Cache containing states for all layers.
                Used for efficient autoregressive generation. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None (uses config value).
            cache_position (Array | None, optional): Position indices for cache updates. Defaults to None.
            attention_mask (Array | None, optional): Mask of shape [batch, seq_len] where 1 indicates
                valid tokens and 0 indicates padding. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Mamba2Output: Object containing:
                - last_hidden_state (Array): Final hidden states of shape [batch, seq_len, hidden_size].
                - cache_params (RecurrentCache | None): Updated cache for next iteration.
                - hidden_states (tuple[Array] | None): Hidden states from all layers if output_hidden_states=True.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified or both are None.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if cache_params is None:
            cache_params = RecurrentCache.init_empty(len(self.layers))
        if attention_mask is None:
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

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params,
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


@register_module(TaskType.CAUSAL_LM, config=Mamba2Config, model_type="mamba2")
class Mamba2ForCausalLM(BaseCausalLMModule[Mamba2Model, Mamba2Config]):
    """Mamba2 language model with tied backbone and projection head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "mamba2"
    _config_class = Mamba2Config

    def __init__(
        self,
        config: Mamba2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Mamba2ForCausalLM.

        Args:
            config (Mamba2Config): Model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Mamba2Model,
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
    ) -> tuple | Mamba2CausalLMOutput:
        """Forward pass for causal language modeling.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape [batch, seq_len].
                Mutually exclusive with inputs_embeds. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                [batch, seq_len, hidden_size]. Mutually exclusive with input_ids. Defaults to None.
            cache_params (RecurrentCache | None, optional): Cache containing states for all layers.
                Used for efficient autoregressive generation. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None (uses config value).
            apply_lm_head (bool, optional): Whether to apply language modeling head to produce logits.
                Set to False to get only hidden states. Defaults to True.
            cache_position (Array | None, optional): Position indices for cache updates. Defaults to None.
            attention_mask (Array | None, optional): Mask of shape [batch, seq_len] where 1 indicates
                valid tokens and 0 indicates padding. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Mamba2CausalLMOutput: Object containing:
                - logits (Array | None): Next-token prediction logits of shape [batch, seq_len, vocab_size]
                  if apply_lm_head=True, otherwise None.
                - cache_params (RecurrentCache | None): Updated cache for next iteration.
                - hidden_states (tuple[Array] | None): Hidden states from all layers if output_hidden_states=True.
        """
        mamba_outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_params=cache_params,
            cache_position=cache_position,
            output_hidden_states=output_hidden_states,
        )

        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(mamba_outputs.last_hidden_state)

        return Mamba2CausalLMOutput(
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        **kwargs,
    ):
        from eformer.escale import PartitionAxis

        from easydel.layers.caching import RecurrentCache, RecurrentCacheConfig

        cache_params = kwargs.get("cache_params", None)
        cache_position = kwargs.get("cache_position", None)
        attention_mask = kwargs.get("attention_mask", None)

        if cache_params is None:
            partition_axis = getattr(self.config, "partition_axis", None) or PartitionAxis()
            cache_config = RecurrentCacheConfig.create_for_mamba2(
                num_hidden_layers=int(self.config.num_hidden_layers),
                partition_axis=partition_axis,
                batch_size=int(input_ids.shape[0]),
                intermediate_size=int(self.config.intermediate_size),
                num_heads=int(self.config.num_heads),
                head_dim=int(self.config.head_dim),
                state_size=int(self.config.state_size),
                conv_kernel_size=int(self.config.conv_kernel),
                n_groups=int(self.config.n_groups),
            )
            cache_params = RecurrentCache.init_cache(cache_config, dtype=self.dtype)

        if attention_mask is None:
            attention_mask = jnp.ones(input_ids.shape, dtype="i4")

        return self.prepare_inputs_for_call(
            attention_mask=attention_mask,
            cache_params=cache_params,
            cache_position=cache_position,
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_outputs.cache_params.update_seq(1)
        model_kwargs["cache_params"] = model_outputs.cache_params
        return model_kwargs

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
