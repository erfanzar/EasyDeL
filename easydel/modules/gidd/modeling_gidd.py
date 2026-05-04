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

"""
This module provides the core components of the GIDD model, including:
- GiddMLP: A feed-forward network with squared ReLU activation
- GiddAttention: An attention mechanism with optional query-key normalization
- GiddRMSNorm: Root Mean Square normalization layer
- GiddLayer: A transformer layer combining attention and MLP components
- GiddModel: The base transformer model
- GiddForDiffusionLM: A version of the model adapted for diffusion language modeling

The implementation leverages JAX for efficient computation and supports various
optimizations including gradient checkpointing, mixed precision, and model parallelism.
"""

import typing as tp
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import ArrayParam, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule

from .gidd_configuration import GiddConfig


class GiddMLP(spx.Module):
    """Two-layer FFN with squared-ReLU (``ReLU(x)**2``) activation.

    Unlike the gated MLPs used by Llama / Gemma, GIDD uses a plain
    ``up -> activation -> down`` two-projection feedforward and replaces the
    usual SiLU/GeLU with **squared ReLU** (the ReLU squared elementwise).
    This activation has been shown by the Primer line of work to slightly
    outperform GeLU on autoregressive transformers and is also used by GIDD
    for its diffusion variant. Both projections are ``fan_in`` scaled to
    keep activation variance stable across the discrete denoising chain
    GIDD relies on for sampling.

    Attributes:
        config: Source ``GiddConfig`` (reads ``hidden_size``,
            ``intermediate_size``, ``mlp_bias``, ``init_scale``).
        dtype: Activation/compute dtype.
        param_dtype: Storage dtype for the projection kernels.
        precision: ``jax.lax.PrecisionLike`` for the two matmuls.
        up_proj: ColumnParallel ``hidden -> intermediate`` linear.
        down_proj: RowParallel ``intermediate -> hidden`` linear.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GIDD MLP block.

        Args:
            config (GiddConfig): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        column_parallel_linear = partial(
            ColumnParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self.config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self.config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
            rngs=rngs,
        )

        self.up_proj = column_parallel_linear(config.hidden_size, config.intermediate_size)
        self.down_proj = row_parallel_linear(config.intermediate_size, config.hidden_size)

    def forward(self, h: jnp.ndarray) -> jnp.ndarray:
        """Apply squared ReLU feedforward transformation.

        Args:
            h: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim]
        """
        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        h = checkpoint_name(self.up_proj(h), name="mlp_up")
        h = nn.relu(h) ** 2

        h = checkpoint_name(self.down_proj(h), name="mlp_down")

        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        return h


class GiddAttention(AttentionModule):
    """Bidirectional attention with log-init QK-norm scale for diffusion denoising.

    GIDD performs **non-causal** attention because the diffusion objective
    must let every position see every other position when refining a noisy
    sequence. The mask is a noise mask (which positions are *known* clean vs.
    diffusion-corrupted) rather than the lower-triangular causal mask used by
    autoregressive LMs. Beyond that, two GIDD-specific design choices live
    in this module:

    1. **QK normalization with log-position-init scale.** When
       ``config.use_qk_norm`` is true, the per-head QK softmax temperature
       is parameterised by a learned ``qk_scale`` of shape
       ``(1, 1, num_heads, 1)``, initialised to
       ``2 * log(max_position_embeddings)``. This log-scaled init is the
       Primer-style fix for head temperature collapse at long context: it
       starts the temperature high enough that early-training softmaxes do
       not over-sharpen and lets backprop tune each head's effective
       sharpness independently.
    2. **Fan-in scaled projections** with ``init_scale`` from ``config`` so
       the bidirectional attention preserves activation variance during the
       reverse diffusion process.

    The ``head_dim`` defaults to ``hidden_size // num_attention_heads`` but
    can be overridden via ``config.head_dim`` for non-square head geometry.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GIDD attention layer with optional query-key normalization.

        Args:
            config (GiddConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size

        # Calculate head dimension, allowing for explicit override in config
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", head_dim)

        # QK normalization settings
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_eps = config.qk_norm_eps

        if self.use_qk_norm:
            qk_scale_value = jnp.full(
                (1, 1, self.config.num_attention_heads, 1),
                2 * jnp.log(config.max_position_embeddings),
                dtype=self.param_dtype,
            )
            self.qk_scale = ArrayParam.bound(
                shape=(1, 1, self.config.num_attention_heads, 1),
                dtype=self.param_dtype,
                init_method="zeros",
                key=rngs.parameters,
                value=qk_scale_value,
            )
        else:
            # Fixed scale based on head dimension
            self.qk_scale = 1.0

        column_parallel_linear = partial(
            ColumnParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
        )

        self.q_proj = column_parallel_linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.k_proj = column_parallel_linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.v_proj = column_parallel_linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.o_proj = row_parallel_linear(config.num_attention_heads * self.head_dim, config.hidden_size, rngs=rngs)

        self.rotary = self.config.get_basic_rope(
            self.dtype,
            self.head_dim,
            self.head_dim,
            True,
        )

        self.attention_performer = FlexibleAttentionModule(
            base_config=self.config,
            softmax_scale=1.0 if self.use_qk_norm else 1.0 / self.head_dim**0.5,
            dropout_prob=0.0,
        )

    @jax.named_scope("gidd-SpecTrax-attention-concatenate")
    def concatenate(
        self,
        *,
        query: Array,
        key: Array,
        value: Array,
        mask_info: MaskInfo,
        noise_mask: Array | None,
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> tuple[Array, Array, Array, tp.Callable[[], Array]]:
        """Prepare and concatenate key, value, and attention mask for attention computation.

        Handles preprocessing of attention inputs including mask validation, reshaping,
        combining attention masks with noise masks, and creating attention bias initialization.

        Args:
            query (Array): Query tensor of shape [batch_size, seq_len, num_heads, head_dim].
            key (Array): Key tensor of shape [batch_size, seq_len, num_heads, head_dim].
            value (Array): Value tensor of shape [batch_size, seq_len, num_heads, head_dim].
            mask_info (MaskInfo): Mask information including attention mask.
            noise_mask (Array): Binary mask for noise tokens of shape [batch_size, seq_len].
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): View into
                the key/value cache for incremental decoding. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.

        Returns:
            tuple: Contains (key, value, mask_info, init_attention_bias, cache_view).
        """
        # Validate that query and key have matching sequence lengths
        assert query.shape[1] == key.shape[1], "Query and Key lengths must match for GIDD attention."

        attention_mask = mask_info.attention_mask
        if attention_mask is not None:
            if attention_mask.dtype != jnp.bool:
                warnings.warn("attention_mask should be a boolean array", stacklevel=1)
                attention_mask = (attention_mask == 1).astype("b1")

        # Expand attention mask to match attention computation dimensions
        assert attention_mask is not None, "attention_mask must not be None for GIDD attention"
        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = jnp.repeat(attention_mask, query.shape[1], -2)  # [Batch, 1, q_len, kv_len]

        if noise_mask is not None:
            if noise_mask.dtype != jnp.bool:
                warnings.warn("noise_mask should be a boolean array", stacklevel=1)
                noise_mask = (noise_mask == 1).astype("b1")

            noise_mask_q = jnp.expand_dims(noise_mask, axis=-1)
            noise_mask_kv = jnp.expand_dims(noise_mask, axis=-2)
            noise_attn_mask = jnp.expand_dims(noise_mask_q >= noise_mask_kv, axis=-3)

            # Combine with attention mask
            attention_mask = jnp.logical_and(attention_mask, noise_attn_mask)

        # Function to initialize attention bias
        def init_attention_bias():
            return jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

        return key, value, attention_mask, init_attention_bias, cache_view  # pyright: ignore[reportReturnType]

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply L2 normalization to query or key vectors.

        Args:
            x (jnp.ndarray): Input tensor of shape [..., num_heads, head_dim].

        Returns:
            jnp.ndarray: Normalized tensor of the same shape.
        """
        return x * jax.lax.rsqrt(jnp.square(x).sum(-1, keepdims=True) + self.qk_norm_eps)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        noise_mask: Array,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> tuple[Array, Array]:
        """Forward pass through the attention module.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, seq_len, hidden_size).
            mask_info (MaskInfo): Attention mask information including causal masks.
            noise_mask (Array): Binary mask for noise tokens of shape (batch_size, seq_len).
            position_ids (Array): Position indices for rotary embeddings, shape (batch_size, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): View into
                the key/value cache. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention_output, attention_weight, and cache_view.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        # Project inputs to Q, K, V
        query_states, key_states, value_states = (
            checkpoint_name(self.q_proj(hidden_states), name="attn_query"),
            checkpoint_name(self.k_proj(hidden_states), name="attn_key"),
            checkpoint_name(self.v_proj(hidden_states), name="attn_value"),
        )

        if self.use_qk_norm:
            query_states = self._norm(query_states)
            key_states = self._norm(key_states)

        qshape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        kv_shape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        query_states = query_states.reshape(qshape)
        key_states = key_states.reshape(kv_shape)
        value_states = value_states.reshape(kv_shape)

        (
            query_states,
            key_states,
            value_states,
        ) = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

        # Prepare inputs for attention computation
        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            noise_mask=noise_mask,
        )

        attentions = self.attention_performer.forward(
            query_states=query_states * self.qk_scale,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=False,
        )

        # Project attention outputs back to hidden dimension
        attn_output = checkpoint_name(
            self.o_proj(self.shard_attention_prod(attn_output=self._merge_heads(attentions.attention_outputs))),
            name="attn_output",
        )

        return AttentionLayerOutput(  # pyright: ignore[reportReturnType]
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class GiddRMSNorm(spx.Module):
    """RMSNorm with the ``(1 + weight)`` scale, zero-init weight, fp32 variance.

    Same formulation as Gemma's RMSNorm (variance reduced in float32, scale
    applied as ``1 + weight`` so the layer is the identity at initialization)
    used here for the diffusion decoder. The zero-init weight is essential
    for diffusion training because the reverse process is sensitive to the
    initial noise level — a layer that is the identity at step 0 lets the
    denoiser's residual stream survive the early gradient updates intact.
    """

    kernel_init = staticmethod(jax.nn.initializers.ones)

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        """Initialize GIDD RMSNorm layer.

        Args:
            config (GiddConfig): Model configuration with normalization parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
        """
        self.config = config
        self.epsilon = self.config.rms_norm_eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.weight = ArrayParam.bound(
            shape=(self.config.hidden_size,),
            dtype=param_dtype,
            init_method="zeros",
            key=None,
        )

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply RMSNorm to the input tensor.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Normalized tensor of the same shape.
        """
        variance = hidden_states.astype(jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)

        # Normalize and apply scale
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        return (1 + self.weight.value.astype(self.dtype)) * jnp.asarray(hidden_states, dtype=self.dtype)


class GiddLayer(spx.Module):
    """Pre-norm bidirectional transformer block with optional residual gating.

    Standard pre-norm shape ``x + attn(norm(x))`` followed by ``x +
    mlp(norm(x))`` but with two GIDD-specific knobs: attention is
    bidirectional (the diffusion denoiser must see future positions to
    refine the corrupted ones), and the residual additions optionally pass
    through a per-layer scalar gate so the model can learn how much of each
    sub-block's contribution to inject into the chain — useful when the
    same network is reused across many denoising steps and the effective
    residual magnitude needs to vary per timestep.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        resid_scale: float = 1.0,
    ):
        """Initialize GIDD transformer layer.

        Args:
            config (GiddConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            resid_scale (float, optional): Scaling factor for residual connections. Defaults to 1.0.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.resid_scale = resid_scale

        self.self_attn: GiddAttention = GiddAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp: GiddMLP = GiddMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = GiddRMSNorm(config=config, dtype=dtype, param_dtype=param_dtype)
        self.post_attention_layernorm = GiddRMSNorm(config=config, dtype=dtype, param_dtype=param_dtype)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        noise_mask: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the transformer layer.

        Applies pre-normalization architecture: x + scale*attn(norm(x)) followed by x + scale*mlp(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, seq_len, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, seq_len).
            noise_mask (Array): Binary mask for noise tokens of shape (batch_size, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, and cache view.
        """
        # Self-attention block with residual connection
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            noise_mask,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = hidden_states + self.resid_scale * attn_outputs.attention_output

        # Feed-forward block with residual connection
        feed_forward_input = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            # Use block-wise computation for memory efficiency
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)
        hidden_states = hidden_states + self.resid_scale * feed_forward_hidden_states

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=GiddConfig, model_type="Gidd")
class GiddModel(EasyDeLBaseModule):
    """GIDD model implementation for diffusion language modeling.

    This implements the GIDD transformer architecture, utilizing transformer blocks
    with RMSNorm, rotary position embeddings, and bidirectional attention with noise masking.

    Attributes:
        config (GiddConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GIDD base model.

        Args:
            config (GiddConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Calculate residual scale factor
        self.resid_scale = config.resid_scale / config.num_hidden_layers

        self.embed_tokens = Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.emb_init_scale),
            rngs=rngs,
        )

        remat_layer_block = auto_remat(
            GiddLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for _ in range(self.config.num_hidden_layers):
            with self.assign_layer_stage(_, total_layers=self.config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                        resid_scale=self.resid_scale,
                    )
                )

        final_layer_idx = max(0, self.config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=self.config.num_hidden_layers):
            self.norm = GiddRMSNorm(config=config, dtype=dtype, param_dtype=param_dtype)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        log_snr: Array | None = None,
        noise_mask: Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the GIDD base model.

        Processes input tokens through embedding, all transformer layers with RoPE and RMSNorm,
        and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, seq_len, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, seq_len). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, seq_len). Defaults to None.
            log_snr (Array | None, optional): Log signal-to-noise ratio for diffusion. Defaults to None.
            noise_mask (Array | None, optional): Binary mask for noise tokens. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        # Validate input
        if (input_ids is None) ^ (inputs_embeds is None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Validate sequence length
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        # Start with input embeddings
        hidden_states = inputs_embeds

        # Determine runtime mode
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )

        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        def _layer_loop(block, carry):
            hidden_states, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    noise_mask=noise_mask,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=self.frequencies,
                )

            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)

            return hidden_states, all_hidden_states, all_attentions, idx + 1

        hidden_states, all_hidden_states, all_attentions, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, 0),
            trace=True,
        )
        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """Returns the encoder part of the model's graph definition.

        Decoder-Only models don't have an encoder.

        Raises:
            NotImplementedError: Always raised as this is a decoder-only model.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Returns the decoder part of the model's graph definition.

        Returns:
            GiddModel: The model itself, as it is a decoder-only model.
        """
        return self

    def get_lm_head(self):
        """Returns the language model head of the module.

        Base Models don't have a Language Model Head.

        Raises:
            NotImplementedError: Always raised as the base model does not have an LM head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Returns the embedding layer of the module.

        Returns:
            Embed: The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.DIFFUSION_LM, config=GiddConfig, model_type="Gidd")
class GiddForDiffusionLM(EasyDeLBaseModule):
    """GIDD model with a language modeling head for diffusion language modeling tasks.

    This model extends the base GiddModel with a language modeling head for
    generation tasks in the context of diffusion language modeling.

    Attributes:
        config (GiddConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GIDD model for diffusion language modeling.

        Args:
            config (GiddConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Initialize base model
        self.model = GiddModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Initialize language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.head_init_scale),
            precision=self.precision,
            rngs=rngs,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        log_snr: Array | None = None,
        noise_mask: Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass through the GIDD diffusion language model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, seq_len, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, seq_len). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, seq_len). Defaults to None.
            log_snr (Array | None, optional): Log signal-to-noise ratio for diffusion. Defaults to None.
            noise_mask (Array | None, optional): Binary mask for noise tokens. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language modeling head. Defaults to True.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.

        Returns:
            CausalLMOutput: Contains logits (if apply_lm_head), hidden_states, last_hidden_state,
                attentions, and past_key_values.
        """
        # Get outputs from base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            log_snr=log_snr,
            noise_mask=noise_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        # Apply language modeling head if requested
        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(hidden_states)

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self):
        """Returns the encoder part of the model's graph definition.

        Decoder-Only models don't have an encoder.

        Raises:
            NotImplementedError: Always raised as this is a decoder-only model.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Returns the decoder part of the model's graph definition.

        Returns:
            GiddModel: The base model, which serves as the decoder.
        """
        return self.model

    def get_lm_head(self):
        """Returns the language model head of the module.

        Returns:
            ColumnParallelLinear: The language modeling head.
        """
        return self.lm_head

    def get_embedding(self):
        """Returns the embedding layer of the module.

        Returns:
            Embed: The token embedding layer from the base model.
        """
        return self.model.embed_tokens
