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

from __future__ import annotations

import functools
import typing as tp

import jax
import jax.numpy as jnp
from eformer import common_types
from flax import nnx as nn
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.attention import MaskInfo
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import HybridCache, HybridCacheView, OperationsMetadata
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm
from easydel.layers.operations import OperationMetadata
from easydel.layers.operations.modules import SSM2Op

from .falcon_h1_configuration import FalconH1Config


def compute_mup_vector(config: FalconH1Config) -> jnp.ndarray:
    """Compute the muP (maximal update parameterization) scaling vector for FalconH1 SSM.

    This function creates a scaling vector used to apply muP multipliers to different
    components of the SSM projection output. The vector applies different scaling
    factors to: z (gate), x (hidden), B (input), C (output), and dt (timestep) projections.

    Args:
        config (FalconH1Config): Model configuration containing SSM parameters including
            mamba_d_ssm, mamba_expand, hidden_size, mamba_n_groups, mamba_d_state,
            mamba_n_heads, and ssm_multipliers.

    Returns:
        jnp.ndarray: A scaling vector of shape (1, 1, vector_shape) where vector_shape
            equals 2 * intermediate_size + 2 * groups_time_state_size + num_heads.
            The vector contains different multipliers for each SSM component region.
    """
    intermediate_size = (
        int(config.mamba_d_ssm) if config.mamba_d_ssm is not None else int(config.mamba_expand * config.hidden_size)
    )
    groups_time_state_size = int(config.mamba_n_groups) * int(config.mamba_d_state)
    num_heads = int(config.mamba_n_heads)
    zxbcdt_multipliers = list(config.ssm_multipliers)

    vector_shape = 2 * intermediate_size + 2 * groups_time_state_size + num_heads
    mup_vector = jnp.ones((1, 1, vector_shape), dtype=jnp.float32)

    mup_vector = mup_vector.at[:, :, :intermediate_size].multiply(zxbcdt_multipliers[0])
    mup_vector = mup_vector.at[:, :, intermediate_size : 2 * intermediate_size].multiply(zxbcdt_multipliers[1])
    mup_vector = mup_vector.at[:, :, 2 * intermediate_size : 2 * intermediate_size + groups_time_state_size].multiply(
        zxbcdt_multipliers[2]
    )
    mup_vector = mup_vector.at[
        :, :, 2 * intermediate_size + groups_time_state_size : 2 * intermediate_size + 2 * groups_time_state_size
    ].multiply(zxbcdt_multipliers[3])
    mup_vector = mup_vector.at[:, :, 2 * intermediate_size + 2 * groups_time_state_size :].multiply(
        zxbcdt_multipliers[4]
    )
    return mup_vector


def apply_mask_to_padding_states(hidden_states: Array, attention_mask: Array | None) -> Array:
    """Apply attention mask to zero out hidden states at padding positions.

    This function ensures that padding tokens don't contribute to the SSM state
    by multiplying the hidden states with the attention mask. This is critical
    for Mamba-style SSMs where padding tokens should not affect the recurrent state.

    Args:
        hidden_states (Array): Hidden states tensor of shape (batch_size, seq_len, hidden_dim).
        attention_mask (Array | None): Boolean or float mask of shape (batch_size, seq_len)
            where 1/True indicates valid tokens and 0/False indicates padding.

    Returns:
        Array: Masked hidden states with padding positions zeroed out, same shape
            as input. Returns unchanged hidden_states if mask is None or incompatible.
    """
    if (
        attention_mask is not None
        and attention_mask.shape[0] == hidden_states.shape[0]
        and attention_mask.shape[1] == hidden_states.shape[1]
        and attention_mask.shape[1] > 1
    ):
        dtype = hidden_states.dtype
        return (hidden_states * attention_mask[:, :, None]).astype(dtype)
    return hidden_states


class FalconH1Attention(UnifiedAttention):
    """Multi-head attention layer for FalconH1 hybrid models.

    Implements grouped query attention (GQA) with rotary position embeddings (RoPE)
    for the FalconH1 architecture. This attention module applies muP-style key
    scaling via the key_multiplier parameter for improved training stability.

    Inherits from UnifiedAttention and customizes QKV post-processing to apply
    the key_multiplier scaling factor.

    Attributes:
        key_multiplier (float): Scaling factor applied to key states for muP.
    """

    def __init__(
        self,
        config: FalconH1Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize FalconH1 attention layer with RoPE and muP key scaling.

        Args:
            config (FalconH1Config): Model configuration with attention parameters.
            layer_idx (int): Index of this layer in the model stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matmuls.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state for initialization.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
        )
        self.key_multiplier = float(config.key_multiplier)

    def _postprocess_qkv(
        self,
        query_states: Float[Array, "batch_size seq_len num_heads head_dim"],
        key_states: Float[Array, "batch_size seq_len num_kv_heads head_dim"],
        value_states: Float[Array, "batch_size seq_len num_kv_heads head_dim"],
    ) -> tuple[
        Float[Array, "batch_size seq_len num_heads head_dim"],
        Float[Array, "batch_size seq_len num_kv_heads head_dim"],
        Float[Array, "batch_size seq_len num_kv_heads head_dim"],
    ]:
        """Apply muP key scaling after QKV projection.

        This hook is called after the QKV projections to apply the key_multiplier
        scaling factor for maximal update parameterization (muP). This scaling
        helps maintain consistent learning dynamics across different model widths.

        Args:
            query_states (Array): Query tensor of shape (batch, seq_len, num_heads, head_dim).
            key_states (Array): Key tensor of shape (batch, seq_len, num_kv_heads, head_dim).
            value_states (Array): Value tensor of shape (batch, seq_len, num_kv_heads, head_dim).

        Returns:
            tuple: A tuple of (query_states, scaled_key_states, value_states) where
                key_states has been multiplied by self.key_multiplier.
        """
        key_states = key_states * self.key_multiplier
        return query_states, key_states, value_states


class Conv1D(nn.Module):
    """Depthwise causal 1D convolution layer for the Mamba SSM mixer.

    Implements a 1D convolution with depthwise grouping for processing sequential
    features in the Mamba-style state space model. The convolution is applied
    along the sequence dimension with optional causal padding.

    Parameter layout matches HuggingFace after weight conversion:
        - `kernel`: shape [kernel_size, 1, channels]
        - `bias`: shape [channels] (optional)

    Attributes:
        features (int): Number of input/output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Convolution stride.
        padding (int): Amount of padding on each side.
        dilation (int): Dilation factor for the kernel.
        groups (int): Number of groups for grouped convolution.
        use_bias (bool): Whether to include a bias term.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: JAX precision setting for matmuls.
    """

    def __init__(
        self,
        features: int,
        kernel_size: int,
        *,
        rngs: nn.Rngs,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
    ):
        """Initialize the 1D convolution layer.

        Args:
            features (int): Number of input/output channels (depthwise).
            kernel_size (int): Size of the convolutional kernel along sequence dim.
            rngs (nn.Rngs): Random number generator state for initialization.
            stride (int, optional): Convolution stride. Defaults to 1.
            padding (int, optional): Padding on each side of input. Defaults to 0.
            dilation (int, optional): Kernel dilation factor. Defaults to 1.
            groups (int, optional): Number of groups for grouped convolution.
                Set to features for depthwise convolution. Defaults to 1.
            use_bias (bool, optional): Whether to add a bias term. Defaults to True.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision.
                Defaults to None.
        """
        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

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

    def __call__(self, x: Array) -> Array:
        """Apply 1D convolution to the input.

        Args:
            x (Array): Input tensor of shape (batch, channels, seq_len) where
                channels equals self.features.

        Returns:
            Array: Convolved output of shape (batch, channels, output_seq_len)
                where output_seq_len depends on padding, stride, and dilation.
        """
        rhs = jnp.asarray(jnp.swapaxes(self.kernel.value, 0, 2), dtype=self.dtype)
        y = lax.conv_general_dilated(
            lhs=x.astype(self.dtype),
            rhs=rhs,
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
            precision=self.precision,
        )
        if self.use_bias:
            y = y + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)
        return y.astype(x.dtype)


class FalconH1RMSNormGated(nn.Module):
    """Group RMS normalization with optional SiLU gating for FalconH1 Mamba blocks.

    Implements grouped RMS normalization where the hidden dimension is divided into
    groups and each group is normalized independently. Supports optional SiLU gating
    that can be applied before or after normalization based on the norm_before_gate
    parameter.

    This layer is used in the Mamba SSM mixer to normalize the output before
    the final projection, with gating providing additional non-linearity.

    Attributes:
        hidden_size (int): Total hidden dimension size.
        variance_epsilon (float): Small constant for numerical stability.
        n_groups (int): Number of groups for grouped normalization.
        norm_before_gate (bool): If True, apply norm then gate; else gate then norm.
        dtype (jnp.dtype): Data type for the learnable scale parameter.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        rngs: nn.Rngs,
        n_groups: int = 1,
        norm_before_gate: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize the gated group RMS normalization layer.

        Args:
            hidden_size (int): Size of the hidden dimension to normalize.
            eps (float): Small epsilon for numerical stability in rsqrt.
            rngs (nn.Rngs): Random number generator state for initialization.
            n_groups (int, optional): Number of groups to divide hidden_size into.
                Each group is normalized independently. Defaults to 1 (full RMSNorm).
            norm_before_gate (bool, optional): If True, normalizes first then applies
                SiLU gate. If False, applies gate first then normalizes. Defaults to True.
            dtype (jnp.dtype, optional): Data type for the scale parameter.
                Defaults to jnp.float32.
        """
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate
        self.dtype = dtype
        self.kernel = ArrayParam.bound(
            shape=(hidden_size,),
            dtype=dtype,
            init_method="ones",
            key=rngs.params(),
        )

    def __call__(self, hidden_states: Array, gate: Array | None = None) -> Array:
        """Apply grouped RMS normalization with optional SiLU gating.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_size)
                or (batch, hidden_size).
            gate (Array | None, optional): Optional gating tensor of same shape as
                hidden_states. When provided, applies SiLU(gate) * normalized_output
                (or vice versa based on norm_before_gate). Defaults to None.

        Returns:
            Array: Normalized (and optionally gated) output with same shape as input.
        """
        input_dtype = hidden_states.dtype
        hs = hidden_states.astype(jnp.float32)

        if (not self.norm_before_gate) and gate is not None:
            hs = hs * jax.nn.silu(gate.astype(jnp.float32))

        if hs.ndim == 3:
            batch_size, seq_len, dim = hs.shape
        else:
            batch_size, dim = hs.shape
            seq_len = 1
            hs = hs[:, None, :]

        hs = hs.reshape(batch_size, seq_len, self.n_groups, dim // self.n_groups)
        variance = jnp.mean(jnp.square(hs), axis=-1, keepdims=True)
        hs = hs * lax.rsqrt(variance + self.variance_epsilon)

        w = self.kernel.value.astype(jnp.float32).reshape(self.n_groups, dim // self.n_groups)
        hs = w[None, None, :, :] * hs
        hs = hs.reshape(batch_size, seq_len, dim)

        if self.norm_before_gate and gate is not None:
            hs = hs * jax.nn.silu(gate.astype(jnp.float32))

        if hidden_states.ndim != 3:
            hs = hs[:, 0, :]
        return hs.astype(input_dtype)


class FalconH1Mixer(nn.Module):
    """Mamba2-style selective state space model (SSM) mixer for FalconH1 blocks.

    Implements a Mamba2-style SSM layer that provides efficient sequence modeling
    through selective state spaces. The mixer consists of:
    1. Input projection to expand hidden states into gate, x/B/C (conv input), and dt
    2. Depthwise 1D convolution for local context
    3. SSM computation with learned A, B, C, D matrices and timestep dt
    4. Gated RMS normalization (optional) before output projection

    This module supports both prefill (parallel) and decode (sequential) modes
    with proper caching of convolution and recurrent states.

    Attributes:
        config (FalconH1Config): Model configuration.
        layer_idx (int): Index of this layer in the model stack.
        num_heads (int): Number of SSM heads.
        hidden_size (int): Model hidden dimension.
        ssm_state_size (int): Size of the SSM state (d_state).
        conv_kernel_size (int): Size of the causal convolution kernel.
        intermediate_size (int): Size of the expanded SSM dimension.
        n_groups (int): Number of groups for grouped operations.
        head_dim (int): Dimension per SSM head.
        chunk_size (int): Chunk size for chunked SSM computation.
    """

    def __init__(
        self,
        config: FalconH1Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the FalconH1 Mamba mixer.

        Args:
            config (FalconH1Config): Model configuration with SSM parameters including
                mamba_n_heads, mamba_d_state, mamba_d_conv, mamba_d_ssm, mamba_expand,
                mamba_n_groups, mamba_d_head, and mamba_chunk_size.
            layer_idx (int): Index of this layer in the model stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matmuls.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state for initialization.

        Raises:
            ValueError: If mamba_n_heads is not divisible by mamba_n_groups.
            ValueError: If intermediate_size != mamba_n_heads * mamba_d_head.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.num_heads = int(config.mamba_n_heads)
        self.hidden_size = int(config.hidden_size)
        self.ssm_state_size = int(config.mamba_d_state)
        self.conv_kernel_size = int(config.mamba_d_conv)
        self.intermediate_size = (
            int(config.mamba_expand * self.hidden_size) if config.mamba_d_ssm is None else int(config.mamba_d_ssm)
        )
        self.n_groups = int(config.mamba_n_groups)
        self.head_dim = int(config.mamba_d_head)
        self.chunk_size = int(config.mamba_chunk_size)

        if self.num_heads % self.n_groups != 0:
            raise ValueError("Expected `mamba_n_heads` to be divisible by `mamba_n_groups` for FalconH1.")
        if self.intermediate_size != self.num_heads * self.head_dim:
            raise ValueError("Expected `mamba_intermediate == mamba_n_heads * mamba_d_head` for FalconH1.")

        self.use_conv_bias = bool(config.mamba_conv_bias)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.use_bias = bool(config.mamba_proj_bias)

        self.layer_norm_epsilon = float(config.rms_norm_eps)
        self.mamba_rms_norm = bool(config.mamba_rms_norm)

        self.ssm_in_multiplier = float(config.ssm_in_multiplier)

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = Conv1D(
            features=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            use_bias=self.use_conv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.in_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.dt_bias = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.params(),
        )
        self.A_log = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.params(),
        )
        self.D = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=param_dtype,
            init_method="ones",
            key=rngs.params(),
        )

        if self.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=self.layer_norm_epsilon,
                n_groups=self.n_groups,
                norm_before_gate=bool(config.mamba_norm_before_gate),
                dtype=param_dtype,
                rngs=rngs,
            )

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            use_bias=bool(config.projectors_bias),
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
        hidden_states: Array,
        mask_info: MaskInfo,
        cache_view: HybridCacheView | None = None,
    ) -> tuple[Array, HybridCacheView | None]:
        """Process input through the Mamba SSM mixer.

        Applies the full Mamba2-style SSM computation pipeline:
        1. Input projection with muP scaling
        2. Split into gate, convolution input (x/B/C), and timestep (dt)
        3. Causal convolution with activation
        4. SSM computation (x, A, B, C, D, dt) -> y
        5. Gated normalization and output projection

        Supports both prefill mode (full sequence) and decode mode (single token)
        with proper caching of convolution states and SSM recurrent states.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_size).
            mask_info (MaskInfo): Mask information containing attention mask for
                handling padding tokens in the SSM computation.
            cache_view (HybridCacheView | None, optional): Cache view containing
                conv_state (for convolution) and recurrent_state (for SSM). When
                provided in decode mode, enables efficient single-token generation.
                Defaults to None.

        Returns:
            tuple[Array, HybridCacheView | None]: A tuple containing:
                - contextualized_states: Output tensor of shape (batch, seq_len, hidden_size)
                - updated_cache_view: Updated cache with new conv_state and recurrent_state,
                  or None if caching is disabled
        """
        q_mask = None
        if mask_info is not None:
            q_mask = mask_info.q_attention_mask
            if q_mask is not None and q_mask.shape[1] != hidden_states.shape[1]:
                q_mask = q_mask[:, : hidden_states.shape[1]]
        hidden_states = apply_mask_to_padding_states(hidden_states, q_mask)

        dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states * self.ssm_in_multiplier

        projected_states = checkpoint_name(self.in_proj(hidden_states), name="ssm_input_proj")
        mup_vector = compute_mup_vector(self.config).astype(projected_states.dtype)
        projected_states = projected_states * mup_vector

        gate, hidden_states_B_C, dt = jnp.split(
            projected_states,
            [
                self.intermediate_size,
                self.intermediate_size + self.conv_dim,
            ],
            axis=-1,
        )

        if q_mask is not None:
            # Mask *post-projection* so biases don't leak into padding positions.
            gate = apply_mask_to_padding_states(gate, q_mask)
            hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, q_mask)

        updated_cache_view = cache_view

        # Convolution with cache support for decode mode
        if seq_len == 1 and cache_view is not None and cache_view.conv_state is not None:
            new_token = hidden_states_B_C[:, 0, :]  # [batch, conv_dim]
            conv_state, _, updated_cache_view = cache_view.concatenate_to_cache(conv_state=new_token)

            # This keeps decode numerically consistent with the full convolution + slice.
            conv_out_full = self.conv1d(conv_state)[..., : self.conv_kernel_size]  # [batch, conv_dim, k]
            conv_out = self.act(conv_out_full[:, :, self.conv_kernel_size - 1]).astype(dtype)[:, None, :]
        else:
            # Prefill mode: use full convolution
            conv_out = self.conv1d(jnp.swapaxes(hidden_states_B_C, 2, 1))[..., :seq_len]
            conv_out = self.act(jnp.swapaxes(conv_out, 2, 1)).astype(dtype)

            # Update cache with conv_state for future decoding
            if cache_view is not None:
                conv_in_t = hidden_states_B_C.transpose(0, 2, 1)  # [batch, conv_dim, seq_len]
                if seq_len >= self.conv_kernel_size:
                    new_conv_state = conv_in_t[:, :, -self.conv_kernel_size :]
                else:
                    pad_width = self.conv_kernel_size - seq_len
                    new_conv_state = jnp.pad(conv_in_t, ((0, 0), (0, 0), (pad_width, 0)))
                updated_cache_view = cache_view.replace(conv_state=new_conv_state)

        if q_mask is not None:
            conv_out = apply_mask_to_padding_states(conv_out, q_mask)

        groups_time_state_size = self.n_groups * self.ssm_state_size
        x, B, C = jnp.split(conv_out, [self.intermediate_size, self.intermediate_size + groups_time_state_size], axis=-1)

        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).astype(jnp.float32)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).astype(jnp.float32)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).astype(jnp.float32)
        # Note: SSM2Op handles group expansion internally

        # Prepare dt with bias
        dt = jax.nn.softplus(dt.astype(jnp.float32) + self.dt_bias.value.astype(jnp.float32))
        if q_mask is not None and seq_len > 1:
            # For padding tokens, force dt=0 so the SSM state remains unchanged.
            dt = dt * q_mask[:, :, None].astype(dt.dtype)

        if cache_view is not None and cache_view.recurrent_state is not None:
            ssm_state0 = cache_view.recurrent_state.astype(jnp.float32)
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
            gate=None,  # Gating handled by self.norm below (if mamba_rms_norm) or manually
            ssm_state=ssm_state0,
            n_groups=self.n_groups,
            use_gated_rmsnorm=False,  # We handle norm/gating below
            precision=self.precision,
        )

        y = ssm_output.attention_outputs

        # Update cache with final SSM state
        if updated_cache_view is not None:
            updated_cache_view = updated_cache_view.replace(recurrent_state=ssm_output.ssm_state.astype(dtype))
        if self.mamba_rms_norm:
            scan_output = self.norm(y, gate)
        else:
            scan_output = y * jax.nn.silu(gate.astype(jnp.float32))

        contextualized_states = checkpoint_name(self.out_proj(scan_output.astype(dtype)), name="ssm_output_proj")
        return contextualized_states, updated_cache_view


class FalconH1MLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron for FalconH1 models.

    Implements a gated feedforward network using SwiGLU (Swish-Gated Linear Unit)
    activation. The architecture consists of parallel gate and up projections
    that are combined through element-wise multiplication with gating, followed
    by a down projection.

    The MLP applies muP-style scaling through gate_multiplier and down_multiplier
    parameters for improved training stability at scale.

    Attributes:
        config (FalconH1Config): Model configuration.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: JAX precision setting for matmuls.
        gate_multiplier (float): Scaling factor for gate projection (muP).
        down_multiplier (float): Scaling factor for down projection output (muP).
    """

    def __init__(
        self,
        config: FalconH1Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the SwiGLU MLP block.

        Args:
            config (FalconH1Config): Model configuration with MLP parameters including
                hidden_size, intermediate_size, mlp_bias, hidden_act, and mlp_multipliers.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matmuls.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state for initialization.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        column = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        row = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        self.gate_proj = column(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias, rngs=rngs)
        self.up_proj = column(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias, rngs=rngs)
        self.down_proj = row(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias, rngs=rngs)

        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def __call__(self, x: Array) -> Array:
        """Apply SwiGLU feedforward transformation with muP scaling.

        Computes: down_proj(up_proj(x) * act_fn(gate_proj(x) * gate_mul)) * down_mul

        Args:
            x (Array): Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Array: Transformed tensor of shape (batch, seq_len, hidden_size).
        """
        y = self.up_proj(x) * self.act_fn(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


class FalconH1DecoderLayer(nn.Module):
    """Single decoder layer for FalconH1 hybrid models.

    Implements a parallel hybrid architecture where the Mamba SSM mixer and
    transformer attention run in parallel on the same input, with their outputs
    summed together. This design combines the strengths of:
    - Mamba SSM: Efficient O(n) sequence modeling with selective state spaces
    - Transformer attention: Strong in-context learning and long-range dependencies

    Layer structure:
    1. Input LayerNorm
    2. Parallel computation:
       - Mamba SSM mixer (with muP scaling)
       - Multi-head attention (with muP key scaling)
    3. Sum of Mamba and attention outputs + residual
    4. Pre-FFN LayerNorm
    5. SwiGLU MLP + residual

    Attributes:
        config (FalconH1Config): Model configuration.
        layer_idx (int): Index of this layer in the model stack.
        attention_in_multiplier (float): muP scaling for attention input.
        ssm_out_multiplier (float): muP scaling for SSM output.
        attn_out_multiplier (float): muP scaling for attention output.
    """

    def __init__(
        self,
        config: FalconH1Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the FalconH1 decoder layer.

        Args:
            config (FalconH1Config): Model configuration with layer parameters.
            layer_idx (int): Index of this layer in the model stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matmuls.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state for initialization.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.feed_forward = FalconH1MLP(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mamba = FalconH1Mixer(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.self_attn = FalconH1Attention(
            config,
            layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.attention_in_multiplier = float(config.attention_in_multiplier)
        self.ssm_out_multiplier = float(config.ssm_out_multiplier)
        self.attn_out_multiplier = float(config.attention_out_multiplier)

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pre_ff_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        mask_info: MaskInfo | None,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: HybridCacheView | None = None,
        cache_metadata: OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the FalconH1 decoder layer.

        Processes input through parallel Mamba SSM and attention branches, combines
        their outputs with residual connection, then applies MLP with another residual.
        The cache is updated sequentially: first Mamba updates conv/recurrent states,
        then attention updates key/value states.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_size).
            mask_info (MaskInfo | None): Mask information for attention and SSM,
                containing causal masks and padding information.
            position_ids (Array): Position indices for RoPE of shape (batch, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode controlling optimization paths:
                - MODE_TRAIN: Training mode with full sequence processing
                - MODE_PREFILL: Prefill phase for generation (parallel processing)
                - MODE_DECODE: Decode phase for generation (sequential single tokens)
            cache_view (HybridCacheView | None, optional): Cache view containing:
                - key_states, value_states: For attention KV cache
                - conv_state: For Mamba convolution state
                - recurrent_state: For Mamba SSM state
                Defaults to None.
            cache_metadata (OperationsMetadata | None, optional): Metadata for
                cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights.
                Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies of
                shape (seq_len, head_dim). Defaults to None.

        Returns:
            DecoderLayerOutput: Contains:
                - hidden_states: Output tensor of shape (batch, seq_len, hidden_size)
                - attention_weight: Attention weights if output_attentions=True, else None
                - cache_view: Updated HybridCacheView with new states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Run Mamba SSM and attention in parallel (cache updates are sequential)
        # Mamba updates conv_state/recurrent_state
        mamba_hidden_states, updated_cache_view = self.mamba(
            hidden_states,
            mask_info=mask_info,
            cache_view=cache_view,
        )
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

        # Pass updated_cache_view so attention can update key/value while preserving mamba's state
        attn_output = self.self_attn(
            hidden_states * self.attention_in_multiplier,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=updated_cache_view,  # Contains mamba's conv/recurrent state
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            frequencies=frequencies,
        )
        attn_hidden_states = attn_output.attention_output * self.attn_out_multiplier

        # Use attention's cache_view as it contains both mamba's updates and attention's KV updates
        if attn_output.cache_view is not None:
            updated_cache_view = attn_output.cache_view

        # Parallel combination: sum of mamba and attention outputs
        hidden_states = residual + (mamba_hidden_states + attn_hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_output.attention_weight if output_attentions else None,
            cache_view=updated_cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=FalconH1Config, model_type="falcon_h1")
class FalconH1Model(EasyDeLBaseModule):
    """FalconH1 base model implementing a parallel hybrid Mamba-Attention architecture.

    FalconH1 is a state-of-the-art language model that combines Mamba2 selective state
    space models with transformer attention in a parallel hybrid design. Each layer
    processes input through both Mamba SSM and attention branches simultaneously,
    combining their outputs for enhanced sequence modeling.

    Key features:
    - Parallel hybrid architecture: SSM and attention run concurrently in each layer
    - Mamba2 SSM: Efficient O(n) sequence modeling with selective state spaces
    - Grouped Query Attention (GQA): Memory-efficient attention with RoPE
    - muP (maximal update parameterization): Scaling factors for stable training
    - HybridCache: Unified caching for both KV states and SSM recurrent states

    Model structure:
    1. Token embeddings with embedding_multiplier scaling
    2. Stack of FalconH1DecoderLayer blocks
    3. Final RMSNorm

    Attributes:
        config (FalconH1Config): Model configuration.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: JAX precision setting for matmuls.
        embedding_multiplier (float): muP scaling for embeddings.
        lm_head_multiplier (float): muP scaling for language model head output.
    """

    def __init__(
        self,
        config: FalconH1Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the FalconH1 base model.

        Args:
            config (FalconH1Config): Model configuration containing architecture
                parameters including vocab_size, hidden_size, num_hidden_layers,
                and all Mamba/attention hyperparameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matmuls.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state for initialization.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        layer_block = auto_remat(
            FalconH1DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = [
            layer_block(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.final_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.embedding_multiplier = float(config.embedding_multiplier)
        self.lm_head_multiplier = float(config.lm_head_multiplier)

    def __call__(
        self,
        input_ids: Array | None = None,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
        past_key_values: HybridCache | None = None,
        inputs_embeds: Array | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        mask_info: MaskInfo | None = None,
        cache_metadata: OperationsMetadata | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        """Perform forward pass through the FalconH1 transformer model.

        Processes input tokens through embeddings and stacked hybrid decoder layers,
        combining Mamba SSM and transformer attention at each layer. Supports both
        training and inference modes with efficient caching for generation.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape
                (batch_size, sequence_length). Either this or inputs_embeds must
                be provided, but not both. Defaults to None.
            attention_mask (Array | None, optional): Boolean mask of shape
                (batch_size, sequence_length) where True indicates valid tokens
                and False indicates padding. Defaults to None.
            position_ids (Array | None, optional): Position indices for RoPE of
                shape (batch_size, sequence_length). Auto-generated from mask if
                not provided. Defaults to None.
            past_key_values (HybridCache | None, optional): Cache containing:
                - key_states, value_states: For attention KV cache
                - conv_state: For Mamba convolution state
                - recurrent_state: For Mamba SSM state
                Enables efficient autoregressive generation. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed embeddings of shape
                (batch_size, sequence_length, hidden_size). Use instead of input_ids
                for custom embeddings. Defaults to None.
            use_cache (bool | None, optional): Whether to use and return cache for
                generation. Defaults to config.use_cache.
            output_attentions (bool | None, optional): Whether to return attention
                weights from all layers. Defaults to config.output_attentions.
            output_hidden_states (bool | None, optional): Whether to return hidden
                states from all layers. Defaults to config.output_hidden_states.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode:
                - MODE_TRAIN: Training with full sequences
                - MODE_PREFILL: Prefill phase for generation
                - MODE_DECODE: Token-by-token generation
                Auto-detected if None. Defaults to None.
            mask_info (MaskInfo | None, optional): Pre-computed mask information.
                If provided, overrides attention_mask. Defaults to None.
            cache_metadata (OperationsMetadata | None, optional): Metadata for
                cache operations. Defaults to None.
            **kwargs: Additional unused arguments (for compatibility).

        Returns:
            BaseModelOutput: Contains:
                - last_hidden_state: Final layer output of shape (batch, seq_len, hidden_size)
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
                - past_key_values: Updated HybridCache for next generation step

        Raises:
            ValueError: If neither input_ids nor inputs_embeds is provided, or both.
        """
        del kwargs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embedding_multiplier

        hidden_states = inputs_embeds
        _batch_size, seq_len = hidden_states.shape[:2]

        # Determine runtime mode
        if mode is None:
            if past_key_values is not None and seq_len == 1:
                mode = common_types.MODE_DECODE
            elif past_key_values is not None:
                mode = common_types.MODE_PREFILL
            else:
                mode = common_types.MODE_TRAIN

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            cache_view = None
            if past_key_values is not None:
                cache_view = past_key_values.get_view(layer_idx)

            layer_output = layer(
                hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
            )

            hidden_states = layer_output.hidden_states

            if past_key_values is not None and layer_output.cache_view is not None:
                past_key_values = past_key_values.update_view(layer_idx, layer_output.cache_view)

            if output_attentions and layer_output.attention_weight is not None:
                all_attentions += (layer_output.attention_weight,)

        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_decoder(self):
        """Return the decoder layers of the model.

        Returns:
            list[FalconH1DecoderLayer]: List of decoder layer modules.
        """
        return self.layers

    def get_embedding(self):
        """Return the token embedding layer of the model.

        Returns:
            nn.Embed: Token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=FalconH1Config, model_type="falcon_h1")
class FalconH1ForCausalLM(BaseCausalLMModule[FalconH1Model, FalconH1Config]):
    """FalconH1 model with a language modeling head for causal language modeling.

    This model combines the FalconH1 parallel hybrid architecture (Mamba2 SSM +
    transformer attention) with a language modeling head for autoregressive text
    generation. It supports efficient generation through HybridCache which maintains
    both attention KV states and Mamba recurrent states.

    Key features:
    - Parallel hybrid backbone: Mamba SSM and attention in each layer
    - muP (maximal update parameterization): Stable training at scale
    - HybridCache: Unified caching for efficient generation
    - Tied embeddings: Optional weight tying between embeddings and LM head

    Attributes:
        config (FalconH1Config): Model configuration.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: JAX precision setting for matmuls.
        model (FalconH1Model): The base transformer model.
        lm_head: Linear projection to vocabulary logits.
    """

    def __init__(
        self,
        config: FalconH1Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the FalconH1 causal language model.

        Args:
            config (FalconH1Config): Model configuration containing all architecture
                parameters and generation settings.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matmuls.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state for initialization.
        """
        super().__init__(
            config=config,
            base_model_class=FalconH1Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            tie_word_embeddings=config.tie_word_embeddings,
            lm_head_name="lm_head",
            lm_head_bias=False,
        )

    def __call__(
        self,
        input_ids: Array | None = None,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
        past_key_values: HybridCache | None = None,
        inputs_embeds: Array | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        mask_info: MaskInfo | None = None,
        cache_metadata: OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        **kwargs,
    ) -> CausalLMOutput:
        """Perform forward pass for causal language modeling.

        Processes input through the FalconH1 backbone and optionally applies the
        language modeling head to produce vocabulary logits. Supports efficient
        autoregressive generation with HybridCache.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape
                (batch_size, sequence_length). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask of shape
                (batch_size, sequence_length) for valid tokens. Defaults to None.
            position_ids (Array | None, optional): Position indices for RoPE.
                Defaults to None.
            past_key_values (HybridCache | None, optional): Cache containing
                attention KV states and Mamba recurrent states. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed embeddings.
                Defaults to None.
            use_cache (bool | None, optional): Whether to use/return cache.
                Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention
                weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all
                hidden states. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimization.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Pre-computed mask information.
                Defaults to None.
            cache_metadata (OperationsMetadata | None, optional): Cache operation
                metadata. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the LM head to produce
                logits. Set to False to get only hidden states. Defaults to True.
            **kwargs: Additional unused arguments (for compatibility).

        Returns:
            CausalLMOutput: Contains:
                - logits: Vocabulary logits of shape (batch, seq_len, vocab_size),
                  or None if apply_lm_head=False
                - hidden_states: Tuple of layer outputs if output_hidden_states=True
                - last_hidden_state: Final layer output
                - attentions: Tuple of attention weights if output_attentions=True
                - past_key_values: Updated HybridCache for next generation step
        """
        del kwargs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            mask_info=mask_info,
            cache_metadata=cache_metadata,
        )

        hidden_states = outputs.last_hidden_state
        logits = None
        if apply_lm_head:
            logits = self.lm_head(hidden_states) * self.model.lm_head_multiplier

        return CausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Array,
        max_length: int,
        pad_token_id: int,
        starts: Array | None = None,
        attention_mask: Array | None = None,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Prepare inputs for the first step of autoregressive generation.

        Initializes the HybridCache with appropriate layer types for the parallel
        hybrid architecture, computes mask information, and sets up position IDs.
        This method is called once at the beginning of generation.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, seq_len) for
                the prompt/prefix.
            max_length (int): Maximum total generation length (prompt + generated).
            pad_token_id (int): Token ID used for padding.
            starts (Array | None, optional): Starting positions for each sequence
                in the batch, used for variable-length prompts. Auto-computed from
                attention_mask if not provided. Defaults to None.
            attention_mask (Array | None, optional): Boolean mask of shape
                (batch_size, seq_len) indicating valid tokens. Defaults to None.
            **kwargs: Additional unused arguments (for compatibility).

        Returns:
            dict[str, tp.Any]: Dictionary containing prepared inputs:
                - past_key_values: Initialized HybridCache
                - mask_info: Padded MaskInfo for causal attention
                - position_ids: Position indices for RoPE
        """
        del kwargs
        batch_size, seq_length = input_ids.shape

        # Calculate starts if not provided
        if starts is None:
            if attention_mask is not None:
                starts = self.compute_prefill_length_from_mask(attention_mask.astype(jnp.bool_))
            else:
                starts = self.compute_prefill_length(input_ids, pad_token_id)

        cache = self.init_operations_cache(
            batch_size=batch_size,
            max_length=max_length,
            starts=starts,
        )

        # Setup mask info
        if attention_mask is not None:
            mask_info = MaskInfo.from_attention_mask(attention_mask)
        else:
            valid = input_ids != pad_token_id
            seg = jnp.where(valid, jnp.int32(0), jnp.int32(-1))
            mask_info = MaskInfo.from_segments(seg)
        mask_info = self._pad_maskinfo_to_maxlen(mask_info, max_length=max_length, make_causal=True)

        if attention_mask is not None:
            am = attention_mask.astype(jnp.bool_)
            position_ids = jnp.where(am, am.astype(jnp.int32).cumsum(axis=-1) - 1, 0)
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype=jnp.int32)[None, :], (batch_size, seq_length))

        return self.prepare_inputs_for_call(past_key_values=cache, mask_info=mask_info, position_ids=position_ids)

    def update_inputs_for_generation(
        self,
        model_outputs: CausalLMOutput,
        model_kwargs: dict[str, tp.Any],
    ) -> dict[str, tp.Any]:
        """Update model inputs for the next autoregressive generation step.

        Updates the cache and position IDs from the previous forward pass to prepare
        for generating the next token. This method is called after each token is
        generated during autoregressive decoding.

        Args:
            model_outputs (CausalLMOutput): Outputs from the previous forward pass,
                containing the updated cache with new KV states and Mamba recurrent states.
            model_kwargs (dict[str, tp.Any]): Current model keyword arguments including
                past_key_values and position_ids.

        Returns:
            dict[str, tp.Any]: Updated model keyword arguments with:
                - past_key_values: Updated HybridCache from model_outputs
                - position_ids: Incremented position for the next token
        """
        # Update cache
        if model_outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = model_outputs.past_key_values

        # Update position IDs for next token
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            # Increment by 1 for the next token
            model_kwargs["position_ids"] = position_ids[:, -1:] + 1

        return model_kwargs

    def get_decoder(self):
        """Return the decoder layers of the underlying model.

        Returns:
            list[FalconH1DecoderLayer]: List of decoder layer modules from the
                base FalconH1Model.
        """
        return self.model.get_decoder()

    def get_embedding(self):
        """Return the token embedding layer of the underlying model.

        Returns:
            nn.Embed: Token embedding layer from the base FalconH1Model.
        """
        return self.model.get_embedding()
