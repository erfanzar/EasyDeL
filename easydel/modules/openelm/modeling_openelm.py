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

"""OpenELM (Open Efficient Language Model) implementation.

Implements Apple's OpenELM decoder family. The defining trait is *layer-wise
scaling*: per-layer ``num_query_heads`` / ``num_kv_heads`` and per-layer FFN
multipliers, all derived from ``qkv_multipliers`` / ``ffn_multipliers`` in the
config. Uses fused QKV projections, optional QK normalization, RoPE,
RMSNorm, and a SwiGLU-style FFN.

Exports:
    - ``OpenELMMultiHeadCausalAttention``: per-layer-shaped causal attention.
    - ``OpenELMFeedForwardNetwork``: per-layer-shaped FFN.
    - ``OpenELMDecoderLayer``: one transformer block.
    - ``OpenELMModel``: base transformer trunk.
    - ``OpenELMForCausalLM``: causal LM head wrapper.
"""

import importlib
import types
import typing
from functools import cached_property

import jax
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
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
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RMSNorm, RowParallelLinear
from easydel.layers.attention import UnifiedAttention
from easydel.modules._base import BaseCausalLMModule

from .openelm_configuration import OpenELMConfig, make_divisible


def _openelm_decoder_layer_block(config: OpenELMConfig) -> type["OpenELMDecoderLayer"]:
    """Return the decoder-layer class to instantiate for this config.

    OpenELM generation rebuilds a no-checkpoint graph for traced decoding. That
    only works if the original decoder-layer class stays unmodified, so we wrap a
    throwaway subclass when gradient checkpointing is enabled instead of mutating
    ``OpenELMDecoderLayer`` in place.

    Args:
        config: OpenELM model configuration. Its ``gradient_checkpointing``
            and ``gradient_checkpointing_targets`` fields control whether a
            rematerialized subclass is created.

    Returns:
        Either the plain ``OpenELMDecoderLayer`` class (when checkpointing is
        disabled) or a dynamically created ``auto_remat``-wrapped subclass.
    """
    policy = getattr(config, "gradient_checkpointing", None)
    policy_value = getattr(policy, "value", policy)
    if policy is None or str(policy_value).lower() in {"", "none"}:
        return OpenELMDecoderLayer

    remat_layer_block = types.new_class(
        "OpenELMDecoderLayerRemat",
        (OpenELMDecoderLayer,),
    )
    remat_layer_block.__module__ = OpenELMDecoderLayer.__module__
    # Register the dynamic class in its module so spectrax can resolve it
    # during jax.eval_shape / bind round-trips.
    try:
        parent_module = importlib.import_module(remat_layer_block.__module__)
        if not hasattr(parent_module, remat_layer_block.__name__):
            setattr(parent_module, remat_layer_block.__name__, remat_layer_block)
    except ImportError:
        pass
    return auto_remat(
        remat_layer_block,
        policy=policy,
        save_names=config.gradient_checkpointing_targets,
        exclude_names=config.gradient_checkpointing_targets,
    )


class OpenELMMultiHeadCausalAttention(UnifiedAttention):
    """Per-layer-scaled GQA attention block — OpenELM's defining architectural feature.

    Unlike LLaMA-style transformers where every layer has the same head
    counts, OpenELM allocates a *different* ``num_query_heads`` and
    ``num_kv_heads`` to each layer (read here from
    ``config.num_query_heads[layer_idx]`` /
    ``config.num_kv_heads[layer_idx]`` produced by the config's layer-wise
    scaling formula). This module temporarily mutates ``config.num_attention_heads``
    and ``config.num_key_value_heads`` so that the parent
    :class:`UnifiedAttention` initializer sees the layer-local values when
    sizing the QKV / output projections, then restores the originals so
    other layers can pick their own counts. The QKV projection is
    *fused* into a single ``ColumnParallelLinear`` of width
    ``(num_q_heads + num_k_heads + num_v_heads) * head_dim`` and split
    along the head axis afterwards. Optional Q/K-norm
    (``normalize_qk_projections``) and rotary position embeddings are
    handled identically to LLaMA.

    The ``projection_mapping`` ClassVar overrides the base mapping so the
    fused QKV layer is exposed under the upstream OpenELM names
    (``qkv_proj`` / ``out_proj``) for state-dict compatibility.

    Attributes:
        layer_idx (int): Index used to look up per-layer head counts.
        num_q_heads (int): Number of query heads on this layer.
        num_k_heads, num_v_heads (int): Number of K/V heads (GQA).
        head_dim (int): Per-head dimension (constant across layers).
        transformer_dim (int): Equal to ``config.model_dim`` — kept under
            this name to mirror the upstream codebase.
    """

    projection_mapping: typing.ClassVar = dict(UnifiedAttention.projection_mapping)
    projection_mapping.update(
        {
            "query_key_value_projection": "qkv_proj",
            "output_projection": "out_proj",
        }
    )

    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize OpenELM attention layer with per-layer head configuration.

        Args:
            config (OpenELMConfig): Model configuration with per-layer attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer, used to determine head counts.
        """
        self.layer_idx = layer_idx
        self.num_q_heads = config.num_query_heads[layer_idx]
        self.num_k_heads = config.num_kv_heads[layer_idx]
        self.num_v_heads = config.num_kv_heads[layer_idx]
        self.head_dim = config.head_dim
        original_num_heads = getattr(config, "num_attention_heads", None)
        original_num_kv_heads = getattr(config, "num_key_value_heads", None)
        config.num_attention_heads = self.num_q_heads
        config.num_key_value_heads = self.num_k_heads
        try:
            super().__init__(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
                attention_type="standard",
                causal=True,
                use_fused_qkv=True,
                use_gqa=True,
            )
        finally:
            if original_num_heads is None:
                delattr(config, "num_attention_heads")
            else:
                config.num_attention_heads = original_num_heads
            if original_num_kv_heads is None:
                delattr(config, "num_key_value_heads")
            else:
                config.num_key_value_heads = original_num_kv_heads
        # Override base head bookkeeping with per-layer values
        self.num_heads = self.num_q_heads
        self.num_key_value_heads = self.num_k_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.transformer_dim = config.model_dim

    def define_network(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> None:
        """Define the attention network components.

        Creates the fused QKV projection, output projection, optional QK normalization
        layers, rotary embeddings, and attention performer.

        Args:
            config (OpenELMConfig): Model configuration.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Numerical precision for operations.
            rngs (spx.Rngs): Random number generator state.
        """
        self.qkv_proj = ColumnParallelLinear(
            config.model_dim,
            (self.num_q_heads + self.num_k_heads + self.num_v_heads) * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.out_proj = RowParallelLinear(
            self.num_q_heads * self.head_dim,
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

        if config.normalize_qk_projections:
            self.q_norm = RMSNorm(
                dim=self.head_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                eps=1e-6,
                rngs=rngs,
            )
            self.k_norm = RMSNorm(
                dim=self.head_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                eps=1e-6,
                rngs=rngs,
            )
        else:
            self.q_norm = None
            self.k_norm = None
        self.rotary = self._create_rotary(config, dtype)
        self.attention_performer = self._create_attention_performer(config, rngs)

    def _postprocess_qkv(
        self,
        query_states: jnp.ndarray,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply optional QK normalization to query and key states.

        Args:
            query_states (jnp.ndarray): Query tensor.
            key_states (jnp.ndarray): Key tensor.
            value_states (jnp.ndarray): Value tensor.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Normalized query, key, and value states.
        """
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        return query_states, key_states, value_states

    def _create_rotary(self, config: OpenELMConfig, dtype: jnp.dtype):
        """Create rotary position embedding for this attention layer.

        Args:
            config (OpenELMConfig): Model configuration with RoPE parameters.
            dtype (jnp.dtype): Data type for the rotary embeddings.

        Returns:
            Rotary embedding module for applying positional information.
        """
        return config.get_basic_rope(
            dtype,
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            base=config.rope_freq_constant,
        )

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
        """Forward pass through the attention layer.

        Computes multi-head attention with rotary position embeddings and optional
        KV caching for efficient autoregressive generation.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, seq_len, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view
                for KV states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.
            alibi (Array | None, optional): ALiBi attention biases. Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention output, optional attention weights, and cache view.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        qkv = checkpoint_name(self.query_key_value_projection(hidden_states), "attn_qkv")
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        query_states = qkv[:, :, : self.num_q_heads, :]
        key_states = qkv[:, :, self.num_q_heads : self.num_q_heads + self.num_k_heads, :]
        value_states = qkv[:, :, self.num_q_heads + self.num_k_heads :, :]

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

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.output_projection(attn_output), name="attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class OpenELMFeedForwardNetwork(spx.Module):
    """Per-layer-scaled FFN supporting both vanilla MLP and SwiGLU.

    OpenELM picks each layer's intermediate dimension as
    ``make_divisible(ffn_multipliers[layer_idx] * model_dim, ffn_dim_divisor)``,
    so the FFN width grows monotonically through depth (per Apple's
    "layer-wise scaling" recipe). Two FFN flavours are supported and chosen
    by ``config.ffn_with_glu``:

    * **GLU** (default): ``proj_1`` produces ``2 * intermediate_dim`` wide
      output that is split halfway into a gate and value stream; the
      activation is applied only to the gate, the streams are multiplied,
      and ``proj_2`` projects back to ``model_dim``. With
      ``activation_fn_name="swish"`` this collapses to the standard
      SwiGLU.
    * **Plain MLP**: ``proj_1 -> activation -> proj_2`` of width
      ``intermediate_dim``.

    Both variants use ``ColumnParallelLinear`` on the expanding side and
    ``RowParallelLinear`` on the contracting side for tensor-parallel
    sharding of the intermediate activation.

    Attributes:
        proj_1 (ColumnParallelLinear): Expanding projection. Width is
            ``2 * intermediate_dim`` for GLU or ``intermediate_dim`` for plain.
        proj_2 (RowParallelLinear): Contracting projection back to ``model_dim``.
        ffn_with_glu (bool): Mirrors ``config.ffn_with_glu``.
        act (Callable): Activation looked up via ``ACT2FN[activation_fn_name]``.
        layer_idx (int): Layer index used for the per-layer width lookup.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initializes the OpenELMFeedForwardNetwork module.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            layer_idx (int): The index of the current decoder layer.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = ColumnParallelLinear(
                config.model_dim,
                2 * intermediate_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            self.proj_2 = RowParallelLinear(
                intermediate_dim,
                config.model_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            self.ffn_with_glu = True
        else:
            self.proj_1 = ColumnParallelLinear(
                config.model_dim,
                intermediate_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            self.proj_2 = RowParallelLinear(
                intermediate_dim,
                config.model_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply feedforward transformation with optional gated linear unit.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            Array: Transformed hidden states of shape (batch_size, seq_len, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        if self.ffn_with_glu:
            y_12 = checkpoint_name(self.proj_1(hidden_states), "mlp_gate")
            y_1, y_2 = jnp.split(y_12, 2, axis=-1)
            hidden_states = checkpoint_name(self.proj_2(self.act(y_1) * y_2), "mlp_down")
        else:
            proj_1_out = checkpoint_name(self.proj_1(hidden_states), "mlp_up")
            hidden_states = checkpoint_name(self.proj_2(self.act(proj_1_out)), "mlp_down")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class OpenELMDecoderLayer(spx.Module):
    """One OpenELM block: pre-norm per-layer-scaled attention + per-layer-scaled FFN.

    Both sub-layers use the same pre-norm residual recipe as LLaMA but
    every shape (head counts, FFN width) is *layer-specific* — see
    :class:`OpenELMMultiHeadCausalAttention` and
    :class:`OpenELMFeedForwardNetwork` for details. This makes the block
    fairly conventional structurally; the novelty lives entirely in the
    sizing schedules computed in :class:`OpenELMConfig`.

    Attributes:
        attn (OpenELMMultiHeadCausalAttention): Per-layer GQA attention.
        ffn (OpenELMFeedForwardNetwork): Per-layer-scaled (Sw)GLU FFN.
        attn_norm (RMSNorm): Pre-attention RMSNorm.
        ffn_norm (RMSNorm): Pre-FFN RMSNorm.
        layer_idx (int): Layer index used by both sub-layers for schedule
            lookup.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initializes the OpenELMDecoderLayer.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            layer_idx (int): The index of the current decoder layer.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx

        self.attn = OpenELMMultiHeadCausalAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.ffn = OpenELMFeedForwardNetwork(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.ffn_norm = RMSNorm(
            self.config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            eps=1e-6,
            rngs=rngs,
        )
        self.attn_norm = RMSNorm(
            self.config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            eps=1e-6,
            rngs=rngs,
        )

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
    ):
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + ffn(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view
                for KV states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, and cache view.
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = checkpoint_name(residual + attn_outputs.attention_output, "residual")

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.ffn,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.ffn(hidden_states)
        hidden_states = checkpoint_name(residual + feed_forward_hidden_states, "layer_output")
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


@register_module(TaskType.BASE_MODULE, config=OpenELMConfig, model_type="openelm")
class OpenELMModel(EasyDeLBaseModule):
    """OpenELM base trunk: token embeddings + layer-wise-scaled decoder stack + final RMSNorm.

    Each decoder layer in this trunk is *individually* sized — see
    :class:`OpenELMConfig` for the multipliers that materialize per-layer
    head counts and FFN widths. The attention layer-stage assignment uses
    :func:`spx.assign_stage` so the heterogeneous per-layer shapes can
    still be partitioned across pipeline stages cleanly.

    Attributes:
        config (OpenELMConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (spx.Rngs): Random number generators.
        token_embeddings (Embed): Embedding layer for input tokens.
        layers (tp.List[OpenELMDecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initializes the OpenELMModel.

        Args:
            config (OpenELMConfig): The configuration object for the OpenELM model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.token_embeddings = Embed(
            config.vocab_size,
            config.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        remat_layer_block = _openelm_decoder_layer_block(config)
        self.layers = nn.ModuleList([])
        for i in range(self.config.num_transformer_layers):
            with spx.assign_stage(total=self.config.num_transformer_layers, current=i):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        layer_idx=i,
                        rngs=rngs,
                    )
                )
        self.norm = RMSNorm(
            config.model_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            eps=1e-6,
            rngs=rngs,
        )
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = ColumnParallelLinear(
                config.model_dim,
                config.vocab_size,
                use_bias=False,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
            )
        self.num_transformer_layers = config.num_transformer_layers

    @cached_property
    def frequencies(self):
        """Compute and cache rotary position embedding frequencies.

        Returns:
            Array: Precomputed frequency tensor for rotary position embeddings.
        """
        return self.config.get_basic_frequencies(
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            base=self.config.rope_freq_constant,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the OpenELM base model.

        Processes input tokens through embedding, all decoder layers with RoPE and RMSNorm,
        and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, model_dim). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
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
            ValueError: If both input_ids and inputs_embeds are None or both are provided.
            AssertionError: If sequence_length exceeds max_context_length.
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = checkpoint_name(self.token_embeddings(input_ids.astype("i4")), "embeddings")
        else:
            raise ValueError("you should specify inputs_embeds or input_ids one of them")
        sequence_length = inputs_embeds.shape[1]

        assert sequence_length <= self.config.max_context_length, (
            f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_context_length} got {sequence_length})"
        )
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            inputs_embeds,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        def _layer_loop(layer, carry):
            hidden_states, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    position_ids=position_ids,
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
        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
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
        return self.token_embeddings


@register_module(TaskType.CAUSAL_LM, config=OpenELMConfig, model_type="openelm")
class OpenELMForCausalLM(BaseCausalLMModule[OpenELMModel, OpenELMConfig]):
    """OpenELM model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with per-layer head configuration,
    causal attention masks, and optional input-output embedding sharing for efficient
    autoregressive language generation.

    Attributes:
        config (OpenELMConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "openelm"
    _config_class = OpenELMConfig

    def __init__(
        self,
        config: OpenELMConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize OpenELM model for causal language modeling.

        Args:
            config (OpenELMConfig): Model configuration with OpenELM-specific parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=OpenELMModel,
            base_model_name="transformer",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )
