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
    """RoPE GQA attention used by FalconH1 blocks.

    Inherits from UnifiedAttention and applies key_multiplier via _postprocess_qkv.
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
        """Apply key_multiplier for muP scaling."""
        key_states = key_states * self.key_multiplier
        return query_states, key_states, value_states


class Conv1D(nn.Module):
    """Depthwise causal 1D convolution used by the Mamba mixer.

    Parameter layout matches HF after conversion:
        - `kernel`: [kernel_size, 1, channels]
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
    """Group RMSNorm with optional SiLU gating (FalconH1 Mamba block)."""

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
    """Naive Mamba2-style SSM mixer used in FalconH1 blocks."""

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
        """Forward pass with optional cache support for autoregressive generation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional mask for padding
            cache_view: Optional cache view for conv_state and recurrent_state

        Returns:
            Tuple of (output tensor, updated cache_view or None)
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
    """SwiGLU MLP used by FalconH1 blocks."""

    def __init__(
        self,
        config: FalconH1Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
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
        y = self.up_proj(x) * self.act_fn(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


class FalconH1DecoderLayer(nn.Module):
    """One FalconH1 decoder layer (Mamba mixer + attention + MLP).

    FalconH1 uses parallel hybrid architecture where Mamba SSM and attention
    run in parallel within each layer, then results are summed.
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
        """Forward pass for FalconH1 decoder layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            mask_info: Mask information for attention
            position_ids: Position indices for RoPE
            mode: Runtime mode (MODE_TRAIN, MODE_PREFILL, MODE_DECODE)
            cache_view: Optional HybridCacheView for caching (both KV and recurrent state)
            cache_metadata: Optional cache metadata
            output_attentions: Whether to return attention weights
            frequencies: Optional precomputed RoPE frequencies

        Returns:
            DecoderLayerOutput containing hidden states, attention weights, and updated cache
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
    """FalconH1 backbone (embeddings + stacked hybrid decoder layers)."""

    def __init__(
        self,
        config: FalconH1Config,
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
        """Forward pass for FalconH1 model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs for RoPE [batch, seq_len]
            past_key_values: HybridCache for KV and recurrent state caching
            inputs_embeds: Optional pre-computed embeddings
            use_cache: Whether to use/return cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            mode: Runtime mode (MODE_TRAIN, MODE_PREFILL, MODE_DECODE)
            mask_info: MaskInfo for attention masking
            cache_metadata: Cache metadata for operations

        Returns:
            BaseModelOutput with hidden states, attentions, and cache
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
        return self.layers

    def get_embedding(self):
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=FalconH1Config, model_type="falcon_h1")
class FalconH1ForCausalLM(BaseCausalLMModule[FalconH1Model, FalconH1Config]):
    """FalconH1 model with a language modeling head.

    FalconH1 is a parallel hybrid model combining Mamba2 SSM with transformer attention.
    This class provides generation support with proper HybridCache handling.
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
        """Forward pass for FalconH1 Causal LM.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs for RoPE
            past_key_values: HybridCache for KV and recurrent state
            inputs_embeds: Optional pre-computed embeddings
            use_cache: Whether to use/return cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            mode: Runtime mode (MODE_TRAIN, MODE_PREFILL, MODE_DECODE)
            mask_info: MaskInfo for attention masking
            cache_metadata: Cache metadata
            apply_lm_head: Whether to apply LM head

        Returns:
            CausalLMOutput with logits, hidden states, attentions, and cache
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
        """Prepare inputs for autoregressive generation.

        Creates a HybridCache with PARALLEL_HYBRID layer types for all layers.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_length: Maximum generation length
            pad_token_id: Padding token ID
            starts: Optional starting positions
            attention_mask: Optional attention mask

        Returns:
            Dictionary with prepared inputs for generation
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
        """Update inputs for the next generation step.

        Args:
            model_outputs: Outputs from the previous forward pass
            model_kwargs: Current model keyword arguments

        Returns:
            Updated model keyword arguments for next step
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
        return self.model.get_decoder()

    def get_embedding(self):
        return self.model.get_embedding()
