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

"""Unified Attention Module.

Single base class supporting multiple attention mechanisms:
- Standard RoPE-based attention (Llama, Mistral, most models)
- Multi-head Latent Attention (DeepSeek V2/V3)
- ALiBi positional bias (Falcon, MPT)

Features:
- Multiple forward paths (forward, forward_mla, forward_alibi)
- Post-processing hooks for Q/K normalization
- Automatic routing based on config
- Support for fused and separate QKV projections
- Sliding window attention
- KV caching
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar

import jax
import jax.numpy as jnp
from eformer import common_types
from einops import rearrange
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from easydel.infra.modeling_outputs import AttentionLayerOutput

from .attention import AttentionModule, FlexibleAttentionModule
from .caching import (
    OperationsMetadata,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCacheView,
    TransformerMetadata,
)
from .linear import ColumnParallelLinear, RowParallelLinear
from .norms import RMSNorm

if TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig
else:
    EasyDeLBaseConfig = object


Cfg = TypeVar("BaseModuleCfg", bound=EasyDeLBaseConfig)


def apply_rotary_pos_emb(
    q: Float[Array, "batch_size num_heads seq_len head_dim"],
    k: Float[Array, "batch_size num_kv_heads seq_len head_dim"],
    cos: Float[Array, "max_seq_len head_dim"],
    sin: Float[Array, "max_seq_len head_dim"],
    position_ids: Int[Array, "batch_size seq_len"],
) -> tuple[
    Float[Array, "batch_size num_heads seq_len head_dim"],
    Float[Array, "batch_size num_kv_heads seq_len head_dim"],
]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_kv_heads, seq_len, head_dim]
            cos: Cosine component of RoPE [max_seq_len, head_dim]
            sin: Sine component of RoPE [max_seq_len, head_dim]
            position_ids: Position indices [batch_size, seq_len]

    Returns:
            Tuple of (rotated_query, rotated_key)
    """
    cos_gathered: Float[Array, "batch_size seq_len head_dim"] = cos[position_ids]
    sin_gathered: Float[Array, "batch_size seq_len head_dim"] = sin[position_ids]
    cos_expanded: Float[Array, "batch_size 1 seq_len head_dim"] = jnp.expand_dims(cos_gathered, axis=1)
    sin_expanded: Float[Array, "batch_size 1 seq_len head_dim"] = jnp.expand_dims(sin_gathered, axis=1)

    def rotate_half(
        x: Float[Array, "batch_size num_heads_any seq_len head_dim"],
    ) -> Float[Array, "batch_size num_heads_any seq_len head_dim"]:
        """Rotate half the hidden dims of the input."""
        half_dim: int = x.shape[-1] // 2
        x1: Float[Array, "batch_size num_heads_any seq_len half_head_dim"] = x[..., :half_dim]
        x2: Float[Array, "batch_size num_heads_any seq_len half_head_dim"] = x[..., half_dim:]
        rotated: Float[Array, "batch_size num_heads_any seq_len head_dim"] = jnp.concatenate((-x2, x1), axis=-1)
        return rotated

    # Apply rotary embeddings
    q_rotated: Float[Array, "batch_size num_heads seq_len head_dim"] = rotate_half(q)
    k_rotated: Float[Array, "batch_size num_kv_heads seq_len head_dim"] = rotate_half(k)
    q_embed: Float[Array, "batch_size num_heads seq_len head_dim"] = (q * cos_expanded) + (q_rotated * sin_expanded)
    k_embed: Float[Array, "batch_size num_kv_heads seq_len head_dim"] = (k * cos_expanded) + (k_rotated * sin_expanded)
    return q_embed, k_embed


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Calculate YaRN mscale factor.

    Args:
            scale: Scaling factor
            mscale: Base mscale value

    Returns:
            Computed mscale factor
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * jnp.log(scale) + 1.0


class UnifiedAttention(AttentionModule, Generic[Cfg]):
    """Unified attention module supporting multiple attention mechanisms.

    A single base class that handles all common attention patterns through:
    1. Multiple forward paths (methods) for different attention types
    2. Post-processing hooks for Q/K normalization and other transformations
    3. Automatic routing based on configuration

    Supported Attention Types:
        - standard: RoPE-based attention (default, most models)
        - mla: Multi-head Latent Attention (DeepSeek V2/V3)
        - alibi: ALiBi positional bias (Falcon, MPT)

    Key Features:
        - Q/K normalization via _postprocess_qkv() hook
        - Fused or separate QKV projections
        - Sliding window attention
        - Grouped Query Attention (GQA)
        - KV caching support
        - Residual dropout

    Model-specific customizations via override methods:
        - define_network(): Custom projection structure (e.g., fused QKV)
        - _postprocess_qkv(): Apply Q/K normalization or other transformations
        - _create_rotary(): Custom RoPE configuration
        - _create_alibi_slopes(): Custom ALiBi slopes
        - forward(), forward_mla(), forward_alibi(): Full attention path override

    Example:
        >>> # Standard attention with Q/K norm
        >>> class Gemma3Attention(UnifiedAttention):
        ...     def _postprocess_qkv(self, q, k, v):
        ...         q = self.q_norm(q)
        ...         k = self.k_norm(k)
        ...         return q, k, v
        >>>
        >>> # MLA attention (DeepSeek)
        >>> class DeepseekV2Attention(UnifiedAttention):
        ...     def __init__(self, config, ...):
        ...         config.attention_type = 'mla'
        ...         super().__init__(config, ...)
        >>>
        >>> # ALiBi attention (Falcon)
        >>> class FalconAttention(UnifiedAttention):
        ...     def __init__(self, config, ...):
        ...         config.attention_type = 'alibi'
        ...         super().__init__(config, ...)
        >>>
        >>> # Custom projection names (e.g., matching HuggingFace names)
        >>> class CustomAttention(UnifiedAttention):
        ...     projection_mapping: ClassVar = {
        ...         "query_projection": "query_projection",
        ...         "key_projection": "key_projection",
        ...         "value_projection": "value_projection",
        ...         "output_projection": "output_projection",
        ...         "query_key_value_projection": "query_key_value_projection",
        ...     }
    """

    norms_mapping: ClassVar[dict[str, str]] = {
        "query_normalization": "q_norm",
        "key_normalization": "k_norm",
        "value_normalization": "v_norm",
        "output_normalization": "o_norm",
    }
    projection_mapping: ClassVar[dict[str, str]] = {
        "query_projection": "q_proj",
        "key_projection": "k_proj",
        "value_projection": "v_proj",
        "output_projection": "o_proj",
        "query_key_value_projection": "qkv_proj",
        # MLA-specific projections (DeepSeek V2/V3)
        "mla_q_proj": "q_proj",
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
    }

    def __init__(
        self,
        config: Cfg,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
        attention_type: Literal["standard", "mla", "alibi"] = "standard",
        causal: bool = True,
        sliding_window: int | tuple[int, int] | None = None,
        use_qk_norm: bool = False,
        use_fused_qkv: bool = False,
        use_gqa: bool = False,
        use_mla_lora: bool = False,
    ) -> None:
        """Initialize unified attention module.

        Args:
            config: Model configuration
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: JAX precision setting
            rngs: Random number generators
            attention_type: Type of attention mechanism ("standard", "mla", "alibi")
            causal: Whether to use causal (autoregressive) attention masking
        """
        super().__init__(config=config)

        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.layer_idx = layer_idx

        self.attention_type = attention_type
        self.causal = causal
        self.sliding_window = sliding_window
        self.use_qk_norm = use_qk_norm
        self.use_fused_qkv = use_fused_qkv
        self.use_gqa = use_gqa
        self.use_mla_lora = use_mla_lora

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.define_network(config, dtype, param_dtype, precision, rngs)

    def define_network(
        self,
        config: Cfg,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        precision: jax.lax.Precision,
        rngs: PRNGKeyArray,
    ) -> None:
        """Define network structure.

        Override this to customize projection structure (e.g., fused QKV).
        Default creates separate Q/K/V/O projections.

        Args:
            config: Model configuration
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: JAX precision setting
            rngs: Random number generators
        """
        if self.use_fused_qkv:
            setattr(
                self,
                self.projection_mapping["query_key_value_projection"],
                self._create_fused_qkv_proj(config, dtype, param_dtype, precision, rngs),
            )
        else:
            setattr(
                self,
                self.projection_mapping["query_projection"],
                self._create_q_proj(config, dtype, param_dtype, precision, rngs),
            )
            setattr(
                self,
                self.projection_mapping["key_projection"],
                self._create_k_proj(config, dtype, param_dtype, precision, rngs),
            )
            setattr(
                self,
                self.projection_mapping["value_projection"],
                self._create_v_proj(config, dtype, param_dtype, precision, rngs),
            )

        setattr(
            self,
            self.projection_mapping["output_projection"],
            self._create_o_proj(config, dtype, param_dtype, precision, rngs),
        )

        if self.attention_type == "alibi":
            self._create_alibi_slopes(config)
        else:
            self.rotary = self._create_rotary(config, dtype)

        self.attention_performer = self._create_attention_performer(config, rngs)

        if self.use_qk_norm:
            setattr(
                self,
                self.norms_mapping["query_normalization"],
                self._create_q_norm(config, dtype, param_dtype, rngs),
            )

            setattr(
                self,
                self.norms_mapping["key_normalization"],
                self._create_k_norm(config, dtype, param_dtype, rngs),
            )

        if hasattr(config, "resid_pdrop") and config.resid_pdrop > 0:
            self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def _create_q_proj(
        self,
        config: Cfg,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        precision: jax.lax.Precision,
        rngs: PRNGKeyArray,
    ) -> ColumnParallelLinear:
        """Create query projection layer."""
        return ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_k_proj(
        self,
        config: Cfg,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        precision: jax.lax.Precision,
        rngs: PRNGKeyArray,
    ) -> ColumnParallelLinear:
        """Create key projection layer."""
        return ColumnParallelLinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_v_proj(
        self,
        config: Cfg,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        precision: jax.lax.Precision,
        rngs: PRNGKeyArray,
    ) -> ColumnParallelLinear:
        """Create value projection layer."""
        return ColumnParallelLinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_o_proj(
        self,
        config: Cfg,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        precision: jax.lax.Precision,
        rngs: PRNGKeyArray,
    ) -> RowParallelLinear:
        """Create output projection layer."""
        return RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_fused_qkv_proj(
        self,
        config: Cfg,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        precision: jax.lax.Precision,
        rngs: PRNGKeyArray,
    ) -> ColumnParallelLinear:
        """Create fused QKV projection (Phi3, DBRX, MPT style).

        Override this for models with fused QKV projections.
        """
        qkv_size: int = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        return ColumnParallelLinear(
            config.hidden_size,
            qkv_size,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_rotary(self, config: Cfg, dtype: DTypeLike):
        """Create rotary position embedding layer.

        Override for custom RoPE configuration (partial rotary, custom theta, etc.).

        Returns:
            Rotary position embedding module from config
        """
        return config.get_basic_rope(
            dtype=dtype,
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            base=getattr(config, "rope_theta", 10000.0),
        )

    def _apply_rotary(
        self,
        query_states: Float[Array, "batch_size seq_len num_heads head_dim"],
        key_states: Float[Array, "batch_size seq_len num_kv_heads head_dim"],
        position_ids: Int[Array, "batch_size seq_len"],
        frequencies: Float[Array, "max_seq_len head_dim"] | None = None,
    ) -> tuple[
        Float[Array, "batch_size seq_len num_heads head_dim"],
        Float[Array, "batch_size seq_len num_kv_heads head_dim"],
    ]:
        """Apply rotary position embeddings to Q and K."""
        if self.rotary is None:
            return query_states, key_states
        q_rotated: Float[Array, "batch_size seq_len num_heads head_dim"]
        k_rotated: Float[Array, "batch_size seq_len num_kv_heads head_dim"]
        q_rotated, k_rotated = self.rotary(
            query=query_states,
            key=key_states,
            positions=position_ids,
            frequencies=frequencies,
        )
        return q_rotated, k_rotated

    def _create_alibi_slopes(self, config: Cfg) -> None:
        """Create ALiBi slope values for positional bias.

        Override for custom ALiBi configuration.
        """
        n: int = self.num_heads
        slopes = jnp.array([2 ** (-8 * (i + 1) / n) for i in range(n)])
        self.alibi_slopes = slopes

    def _compute_alibi_bias(self, sequence_length: int) -> Float[Array, "num_heads seq_len seq_len"]:
        """Compute ALiBi positional bias matrix.

        Args:
            sequence_length: Length of the sequence

        Returns:
            ALiBi bias tensor [num_heads, seq_len, seq_len]
        """
        # Create position indices
        positions = jnp.arange(sequence_length)
        # Compute relative positions (broadcasting)
        relative_positions: Int[Array, "seq_len seq_len"] = positions[None, :] - positions[:, None]
        # Apply slopes
        alibi_bias: Float[Array, "num_heads seq_len seq_len"] = (
            relative_positions[None, :, :] * self.alibi_slopes[:, None, None]
        )
        return alibi_bias

    def _create_attention_performer(self, config: Cfg, rngs: nn.Rngs) -> FlexibleAttentionModule:
        """Create attention performer module.

        Override for custom attention dropout or softmax scale.

        Returns:
            FlexibleAttentionModule instance
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=getattr(config, "attention_dropout", 0.0),
        )

    def _create_q_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs) -> RMSNorm:
        """Create query normalization layer.

        Override for custom Q normalization (LayerNorm vs RMSNorm, custom eps, etc.).

        Returns:
            RMSNorm instance for query normalization
        """
        return RMSNorm(
            self.head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs) -> RMSNorm:
        """Create key normalization layer.

        Override for custom K normalization.

        Returns:
            RMSNorm instance for key normalization
        """
        return RMSNorm(
            self.head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_v_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs) -> RMSNorm:
        """Create value normalization layer.

        Override for custom V normalization (LayerNorm vs RMSNorm, custom eps, etc.).

        Returns:
            RMSNorm instance for value normalization
        """
        return RMSNorm(
            self.head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_o_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs) -> RMSNorm:
        """Create output normalization layer.

        Override for custom output normalization.

        Returns:
            RMSNorm instance for output normalization
        """
        return RMSNorm(
            self.head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @property
    def query_projection(self) -> ColumnParallelLinear:
        """Get query projection (for fused QKV support)."""
        return getattr(self, self.projection_mapping["query_projection"])

    @property
    def key_projection(self) -> ColumnParallelLinear:
        """Get key projection (for fused QKV support)."""
        return getattr(self, self.projection_mapping["key_projection"])

    @property
    def value_projection(self) -> ColumnParallelLinear:
        """Get value projection (for fused QKV support)."""
        return getattr(self, self.projection_mapping["value_projection"])

    @property
    def output_projection(self) -> RowParallelLinear:
        """Get output projection."""
        return getattr(self, self.projection_mapping["output_projection"])

    @property
    def query_key_value_projection(self) -> ColumnParallelLinear:
        """Get fused QKV projection."""
        return getattr(self, self.projection_mapping["query_key_value_projection"])

    @property
    def query_normalization(self) -> RMSNorm:
        """Get query normalization layer."""
        return getattr(self, self.norms_mapping["query_normalization"])

    @property
    def key_normalization(self) -> RMSNorm:
        """Get key normalization layer."""
        return getattr(self, self.norms_mapping["key_normalization"])

    @property
    def value_normalization(self) -> RMSNorm:
        """Get value normalization layer."""
        return getattr(self, self.norms_mapping["value_normalization"])

    @property
    def output_normalization(self) -> RMSNorm:
        """Get output normalization layer."""
        return getattr(self, self.norms_mapping["output_normalization"])

    @property
    def mla_q_proj(self) -> ColumnParallelLinear:
        """Get MLA query projection (non-LoRA)."""
        return getattr(self, self.projection_mapping["mla_q_proj"])

    @property
    def mla_q_a_proj(self) -> ColumnParallelLinear:
        """Get MLA query A projection (LoRA)."""
        return getattr(self, self.projection_mapping["mla_q_a_proj"])

    @property
    def mla_q_a_layernorm(self) -> RMSNorm:
        """Get MLA query A layer normalization."""
        return getattr(self, self.projection_mapping["mla_q_a_layernorm"])

    @property
    def mla_q_b_proj(self) -> ColumnParallelLinear:
        """Get MLA query B projection (LoRA)."""
        return getattr(self, self.projection_mapping["mla_q_b_proj"])

    @property
    def mla_kv_a_proj_with_mqa(self) -> ColumnParallelLinear:
        """Get MLA KV A projection with MQA."""
        return getattr(self, self.projection_mapping["mla_kv_a_proj_with_mqa"])

    @property
    def mla_kv_a_layernorm(self) -> RMSNorm:
        """Get MLA KV A layer normalization."""
        return getattr(self, self.projection_mapping["mla_kv_a_layernorm"])

    @property
    def mla_kv_b_proj(self) -> ColumnParallelLinear:
        """Get MLA KV B projection."""
        return getattr(self, self.projection_mapping["mla_kv_b_proj"])

    def _preprocess_qkv(
        self,
        query_states: Float[Array, "batch_size seq_len num_heads_times_head_dim"],
        key_states: Float[Array, "batch_size seq_len num_kv_heads_times_head_dim"],
        value_states: Float[Array, "batch_size seq_len num_kv_heads_times_head_dim"],
    ) -> tuple[
        Float[Array, "batch_size seq_len num_heads_times_head_dim"],
        Float[Array, "batch_size seq_len num_kv_heads_times_head_dim"],
        Float[Array, "batch_size seq_len num_kv_heads_times_head_dim"],
    ]:
        """Pre-process Q/K/V after projection before reshape/RoPE/sharding.

        **KEY METHOD**: Override this to apply Q/K normalization or other transformations.
        By default, applies Q/K norm if configured, otherwise returns unchanged.

        Pattern for Q/K normalization:
            >>> def _preprocess_qkv(self, q, k, v):
            ...     if hasattr(self, 'q_norm'):
            ...         q = self.q_norm(q)
            ...         k = self.k_norm(k)
            ...     return q, k, v

        Args:
            query_states: Query tensor [batch_size, seq_len, num_heads * head_dim]
            key_states: Key tensor [batch_size, seq_len, num_kv_heads * head_dim]
            value_states: Value tensor [batch_size, seq_len, num_kv_heads * head_dim]

        Returns:
            Tuple of (processed_query, processed_key, processed_value)
        """
        return query_states, key_states, value_states

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
        """Post-process Q/K/V after projection and reshape, before RoPE/sharding.

        **KEY METHOD**: Override this to apply Q/K normalization or other transformations.
        By default, applies Q/K norm if configured, otherwise returns unchanged.

        Pattern for Q/K normalization:
            >>> def _postprocess_qkv(self, q, k, v):
            ...     if hasattr(self, 'q_norm'):
            ...         q = self.q_norm(q)
            ...         k = self.k_norm(k)
            ...     return q, k, v

        Args:
            query_states: Query tensor [batch_size, seq_len, num_heads, head_dim]
            key_states: Key tensor [batch_size, seq_len, num_kv_heads, head_dim]
            value_states: Value tensor [batch_size, seq_len, num_kv_heads, head_dim]

        Returns:
            Tuple of (processed_query, processed_key, processed_value)
        """
        return query_states, key_states, value_states

    def _merge_heads(
        self, hidden_states: Float[Array, "batch_size seq_len num_heads head_dim"]
    ) -> Float[Array, "batch_size seq_len hidden_dim"]:
        """Merge attention heads back to hidden dimension.

        Args:
            hidden_states: Attention output tensor [batch_size, seq_len, num_heads, head_dim]

        Returns:
            Merged tensor [batch_size, seq_len, hidden_dim] where hidden_dim = num_heads * head_dim
        """
        batch_size: int = hidden_states.shape[0]
        seq_len: int = hidden_states.shape[1]
        merged = hidden_states.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return merged

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Standard RoPE-based attention (default path).

        Used by most models: Llama, Mistral, Gemma, Qwen, etc.

        Flow:
            1. Project Q/K/V
            2. Reshape to multi-head format
            3. POST-PROCESS: Apply Q/K norm via _postprocess_qkv()
            4. Apply sharding
            5. Apply RoPE
            6. KV cache concatenation
            7. Compute attention
            8. Merge heads and output projection
            9. Optional residual dropout

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            mask_info: Mask information for attention
            position_ids: Position indices for RoPE
            mode: Runtime mode (train/eval/infer)
            cache_view: Optional cache view for KV caching
            cache_metadata: Optional cache metadata
            output_attentions: Whether to return attention weights
            frequencies: Optional precomputed RoPE frequencies
            alibi: Optional external ALiBi positional bias (unused in standard attention)

        Returns:
            AttentionLayerOutput with attention output and optional weights
        """
        batch_size: int = hidden_states.shape[0]
        sequence_length: int = hidden_states.shape[1]

        if self.use_fused_qkv:
            qkv = checkpoint_name(self.query_key_value_projection(hidden_states), "attn_qkv")
            if self.use_gqa:
                qkv_states = rearrange(
                    qkv,
                    "b q (h gs d) -> b q h gs d",
                    gs=2 + self.num_key_value_groups,
                    d=self.head_dim,
                )
                query_states = rearrange(qkv_states[..., : self.num_key_value_groups, :], "b q h gs d -> b q (h gs) d")
                key_states: Float[Array, "batch_size kvseq_len num_kv_heads head_dim"] = qkv_states[..., -2, :]
                value_states: Float[Array, "batch_size kvseq_len num_kv_heads head_dim"] = qkv_states[..., -1, :]
            else:
                query_states, key_states, value_states = jnp.split(qkv, indices_or_sections=3, axis=-1)
        else:
            query_states = checkpoint_name(self.query_projection(hidden_states), "attn_query")
            key_states = checkpoint_name(self.key_projection(hidden_states), "attn_key")
            value_states = checkpoint_name(self.value_projection(hidden_states), "attn_value")

        query_states, key_states, value_states = self._preprocess_qkv(query_states, key_states, value_states)

        query_states: Float[Array, "batch_size seq_len num_heads head_dim"] = query_states.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        key_states: Float[Array, "batch_size kvseq_len num_kv_heads head_dim"] = key_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states: Float[Array, "batch_size kvseq_len num_kv_heads vhead_dim"] = value_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)
        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self.sliding_window
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            sliding_window=sliding_window_for_kernel,
        )

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

        attentions: AttentionLayerOutput = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=causal_for_kernel,
            sliding_window=sliding_window_for_kernel,
            softmax_aux=softmax_aux,
        )

        if attentions.cache_view is not None:
            cache_view = attentions.cache_view

        attention_out = self._merge_heads(attentions.attention_outputs)
        attn_output = self.shard_attention_prod(attention_out)
        attn_output = checkpoint_name(self.output_projection(attn_output), "attn_output")

        if hasattr(self, "resid_dropout"):
            attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )

    def forward_mla(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Multi-head Latent Attention forward path (DeepSeek V2/V3).

        Models using MLA should set config.attention_type = 'mla'.

        MLA uses LoRA-style compression for queries and compressed KV projections.
        """
        bsz, q_len, _ = hidden_states.shape

        if not self.use_mla_lora:
            q = checkpoint_name(self.mla_q_proj(hidden_states), name="attn_query")
        else:
            q = checkpoint_name(
                self.mla_q_b_proj(
                    self.mla_q_a_layernorm(checkpoint_name(self.mla_q_a_proj(hidden_states), name="attn_query_a"))
                ),
                name="attn_query",
            )

        # Reshape and transpose: [batch, heads, seq, dim]
        q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

        # Split query into nope (non-positional) and pe (positional) parts
        q_nope, q_pe = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]

        # KV projection with compression
        compressed_kv = self.mla_kv_a_proj_with_mqa(hidden_states)
        k_pe = compressed_kv[..., self.kv_lora_rank :]
        compressed_kv = compressed_kv[..., : self.kv_lora_rank]

        # Reshape k_pe for RoPE
        k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        # Decompress KV
        kv = (
            self.mla_kv_b_proj(self.mla_kv_a_layernorm(compressed_kv))
            .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(0, 2, 1, 3)
        )

        k_nope = kv[..., : self.qk_nope_head_dim]
        value_states = kv[..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.v_head_dim]

        # Apply RoPE directly to MLA format [batch, heads, seq, rope_dim]
        if frequencies is not None:
            # Extract array from ModuleCaches if needed
            freq_array = frequencies.value if hasattr(frequencies, "value") else frequencies
            # Gather frequencies for all positions: position_ids is [batch, seq], freq_array is [max_seq, rope_dim*2]
            # Result: [batch, seq, rope_dim*2]
            freqs = freq_array[position_ids]
            # Split into cos/sin: each [batch, seq, rope_dim]
            cos, sin = jnp.split(freqs, 2, -1)
            # Expand for heads dimension: [batch, 1, seq, rope_dim]
            cos = cos[:, None, :, :].astype(q_pe.dtype)
            sin = sin[:, None, :, :].astype(q_pe.dtype)
            # Apply neox-style (split) rotation
            q1, q2 = jnp.split(q_pe, 2, axis=-1)
            k1, k2 = jnp.split(k_pe, 2, axis=-1)
            q_pe = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
            k_pe = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

        # Recombine nope and pe parts for query
        query_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), q_pe.dtype)
        query_states = query_states.at[..., : self.qk_nope_head_dim].set(q_nope)
        query_states = query_states.at[..., self.qk_nope_head_dim :].set(q_pe)

        # Recombine nope and pe parts for key
        key_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), k_pe.dtype)
        key_states = key_states.at[..., : self.qk_nope_head_dim].set(k_nope)
        key_states = key_states.at[..., self.qk_nope_head_dim :].set(k_pe)

        # Transpose back to [batch, seq, heads, dim]
        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        # Concatenate with KV cache

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self.sliding_window
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
        )

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=causal_for_kernel,
            sliding_window=sliding_window_for_kernel,
            softmax_aux=softmax_aux,
        )

        # Merge heads and project output
        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.output_projection(attn_output), name="attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )

    def forward_alibi(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """ALiBi positional bias attention forward path (Falcon, MPT).

        Uses Attention with Linear Biases (ALiBi) instead of RoPE for positional information.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        query_states = checkpoint_name(self.query_projection(hidden_states), "attn_query")
        key_states = checkpoint_name(self.key_projection(hidden_states), "attn_key")
        value_states = checkpoint_name(self.value_projection(hidden_states), "attn_value")

        query_states = query_states.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)

        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self.sliding_window
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None
        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
        )

        if alibi is not None:
            alibi_bias = alibi
        else:
            alibi_bias = self._compute_alibi_bias(key_states.shape[1])  # Use full KV length after cache

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=alibi_bias,  # KEY: Use ALiBi bias instead of RoPE
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=causal_for_kernel,
            sliding_window=sliding_window_for_kernel,
            softmax_aux=softmax_aux,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.output_projection(attn_output), "attn_output")

        if hasattr(self, "resid_dropout"):
            attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Main entry point - routes to appropriate forward method.

        Automatically selects the correct attention mechanism based on config.attention_type.
        Models usually don't need to override this unless they have special routing logic.

        Args:
            hidden_states: Input tensor
            mask_info: Mask information
            position_ids: Position indices
            mode: Runtime mode
            cache_view: Optional cache view
            cache_metadata: Optional cache metadata
            output_attentions: Whether to return attention weights
            frequencies: Optional precomputed frequencies
            alibi: Optional external ALiBi positional bias

        Returns:
            AttentionLayerOutput
        """
        if self.attention_type == "mla":
            return self.forward_mla(
                hidden_states,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
                alibi,
            )
        elif self.attention_type == "alibi":
            return self.forward_alibi(
                hidden_states,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                alibi,
            )
        else:
            return self.forward(
                hidden_states,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
                alibi,
            )

    # Operation access properties for dynamic discovery
    @property
    def operation_executor(self):
        """Get the OperationExecutor from attention_performer.

        Returns:
            OperationExecutor if attention_performer exists, None otherwise.
        """
        if hasattr(self, "attention_performer") and self.attention_performer is not None:
            return self.attention_performer.operation_executor
        return None

    @property
    def operation(self):
        """Get the primary operation from attention_performer."""
        if hasattr(self, "attention_performer") and self.attention_performer is not None:
            return self.attention_performer.operation
        return None

    @property
    def operation_requirements(self):
        """Get requirements from the attention performer.

        Returns:
            OperationRequirements if attention_performer exists, default otherwise.
        """
        from easydel.layers.operations.requirements import OperationRequirements

        if hasattr(self, "attention_performer") and self.attention_performer is not None:
            return self.attention_performer.operation_requirements
        return OperationRequirements.default()

    @property
    def requires_cache(self) -> bool:
        """Whether this attention layer requires cache."""
        if hasattr(self, "attention_performer") and self.attention_performer is not None:
            return self.attention_performer.requires_cache
        return True
