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

"""MosaicML Pretrained Transformer (MPT) implementation.

Implements MosaicML's MPT decoder family — a GPT-style architecture with no
position embeddings (replaced by Attention with Linear Biases / ALiBi), a fused
QKV projection, optional QK LayerNorm, and a configurable LayerNorm variant
(``low_precision_layernorm``). Supports causal language modeling.

Exports:
    - ``MptMLP``: GELU feed-forward block with residual.
    - ``MptAttention``: ALiBi attention with fused QKV.
    - ``MptBlock``: a single transformer block.
    - ``MptModel``: base transformer trunk with cached ALiBi tensor.
    - ``MptForCausalLM``: causal LM head wrapper.
    - ``build_mpt_alibi_tensor``: helper to build ALiBi bias tensors.
"""

import math
from functools import cached_property, partial

import jax
import spectrax as spx
from einops import rearrange
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
from easydel.infra.utils import auto_remat
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseCausalLMModule

from .mosaic_configuration import MptConfig as MptConfig


class MptMLP(spx.Module):
    """Two-layer GELU FFN for MPT, returning the post-residual output directly.

    Unlike most modern decoder MLPs (gated SwiGLU), MPT uses a plain
    ``up -> GELU(exact) -> down`` feed-forward of width
    ``expansion_ratio * d_model``. Dropout is applied only on the residual
    branch (matching MPT's original training recipe), and the residual is
    *added inside* this module rather than by the parent block — that is
    why ``forward`` accepts and returns the post-add hidden state.

    Attributes:
        up_proj (ColumnParallelLinear): ``d_model -> expansion_ratio * d_model``.
        down_proj (ColumnParallelLinear): ``expansion_ratio * d_model -> d_model``.
        hidden_dropout (nn.Dropout): Dropout applied to the down-projected
            output before the residual add (rate ``attn_config.attn_pdrop``).
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initializes the MptMLP module.

        Args:
            config (MptConfig): The configuration object for the MPT model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
        self.config = config
        linear_class = partial(
            ColumnParallelLinear,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.up_proj = linear_class(
            self.config.hidden_size,
            self.config.expansion_ratio * self.config.hidden_size,
            rngs=rngs,
        )
        self.down_proj = linear_class(
            self.config.expansion_ratio * self.config.hidden_size,
            self.config.hidden_size,
            rngs=rngs,
        )
        self.hidden_dropout = nn.Dropout(
            self.config.attn_config.attn_pdrop,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        residual: Float[Array, "batch seq_len hidden_dim"],
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the MptMLP module.

        Applies a two-layer feed-forward network with GELU activation.
        The computation is: dropout(down_proj(gelu(up_proj(hidden_states)))) + residual.

        Args:
            hidden_states: Input hidden states of shape (batch_size, sequence_length, hidden_dim).
            residual: Residual connection tensor of shape (batch_size, sequence_length, hidden_dim)
                to be added to the output.

        Returns:
            Output hidden states with residual connection of shape (batch_size, sequence_length, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        up = jax.nn.gelu(checkpoint_name(self.up_proj(hidden_states), name="mlp_up"), approximate=False)
        hidden_states = checkpoint_name(self.down_proj(up), name="mlp_down")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return self.hidden_dropout(hidden_states) + residual


class MptAttention(UnifiedAttention):
    """MPT multi-head attention with ALiBi position biases.

    Differences from a standard transformer attention:

    * **Fused QKV**: a single linear ``Wqkv`` of width ``3 * d_model`` is
      split into Q, K, V slices instead of three separate projections.
    * **No RoPE / learned positional embeddings**: positional information
      enters as a per-head ALiBi bias matrix
      ``-m_h * (i - j)`` for ``j <= i`` added directly to the QK scores
      before softmax. The slopes ``m_h`` form a geometric sequence capped
      by ``alibi_bias_max`` and are precomputed once per model in
      :func:`build_mpt_alibi_tensor`.
    * **Optional QK-LayerNorm** (``qk_ln``) applied to Q and K before the
      score computation for training stability.
    * **No bias** on Wqkv / out_proj when ``no_bias`` is set (the default).

    The module overrides ``forward_alibi`` on :class:`UnifiedAttention` so
    that the precomputed ALiBi slopes plus the live attention mask are
    combined into a single additive bias before the softmax kernel.

    Attributes:
        Wqkv (ColumnParallelLinear): Fused QKV projection ``d_model -> 3 * d_model``.
        out_proj (RowParallelLinear): Output projection ``d_model -> d_model``.
        resid_dropout (nn.Dropout): Residual-stream dropout applied to the
            attention output.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize MPT attention with ALiBi support.

        Args:
            config: Configuration object for the MPT model.
            dtype: Data type for computation. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX operations. Defaults to None.
            rngs: Random number generators.
            layer_idx: Index of this layer in the transformer stack.
        """
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="alibi",
            causal=True,
        )

    def define_network(
        self,
        config: MptConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ):
        """Define MPT-specific network with fused QKV projection.

        Creates the query/key/value projection layer (Wqkv), output projection (out_proj),
        dropout, attention performer, and ALiBi slopes.

        Args:
            config: Configuration object for the MPT model.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: Precision setting for JAX operations.
            rngs: Random number generators.
        """
        self.Wqkv = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size * 3,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        self.resid_dropout = nn.Dropout(
            config.attn_config.attn_pdrop,
            rngs=rngs,
        )

        self.attention_performer = self._create_attention_performer(config, rngs)
        self._create_alibi_slopes(config)

    def _create_attention_performer(self, config: MptConfig, rngs: spx.Rngs):
        """Create attention performer with MPT-specific settings.

        Args:
            config: Configuration object for the MPT model.
            rngs: Random number generators.

        Returns:
            FlexibleAttentionModule configured for MPT attention.
        """
        softmax_scale = config.attn_config.softmax_scale
        if softmax_scale is None:
            softmax_scale = 1 / math.sqrt(self.head_dim)

        return FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=float(config.attn_config.attn_pdrop) if config.attn_config.attn_pdrop is not None else 0.0,
            base_config=config,
            softmax_scale=softmax_scale,
        )

    def _compute_alibi_bias(self, sequence_length):
        """Compute ALiBi positional bias tensor.

        Args:
            sequence_length: Maximum sequence length for the ALiBi tensor.

        Returns:
            ALiBi bias tensor of shape (1, num_heads, sequence_length, sequence_length).
        """
        config: MptConfig = self.config
        return build_mpt_alibi_tensor(config.n_heads, sequence_length, config.attn_config.alibi_bias_max)

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
        """Forward pass with ALiBi positional bias and fused QKV projection.

        Implements attention with ALiBi (Attention with Linear Biases) for positional
        encoding. Uses a fused QKV projection for efficiency.

        Important: ALiBi does not enforce causality by itself, so causal masking
        is applied via `mask_info` and the `causal` flag.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_dim).
            mask_info: Mask information for attention computation.
            position_ids: Position indices of shape (batch_size, seq_len).
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
            cache_view: Cache view for key/value states in generation.
            cache_metadata: Metadata for cache handling.
            output_attentions: Whether to return attention weights.
            alibi: Optional pre-computed ALiBi bias tensor. If None, computed internally.

        Returns:
            AttentionLayerOutput containing attention output, optional attention weights,
            and updated cache view.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        mixed_qkv = checkpoint_name(self.Wqkv(hidden_states), "attn_qkv")
        query_states, key_states, value_states = jnp.split(mixed_qkv, 3, -1)

        query_states = rearrange(query_states, "b s (h d) -> b s h d", h=self.config.n_heads)
        key_states = rearrange(key_states, "b s (h d) -> b s h d", h=self.config.n_heads)
        value_states = rearrange(value_states, "b s (h d) -> b s h d", h=self.config.n_heads)

        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        (
            key_states,
            value_states,
            mask_info,
            _,
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

        if alibi is None:
            alibi_bias = self._compute_alibi_bias(self.config.max_seq_len)
        else:
            alibi_bias = alibi

        alibi_bias = jnp.asarray(alibi_bias, dtype=self.dtype)
        if alibi_bias.ndim == 3:
            alibi_bias = alibi_bias[None, ...]
        elif alibi_bias.ndim == 2:
            alibi_bias = alibi_bias[None, :, None, :]

        q_len = query_states.shape[1]
        kv_len = key_states.shape[1]

        if alibi_bias.shape[-1] != kv_len:
            start_k = max(0, alibi_bias.shape[-1] - kv_len)
            alibi_bias = alibi_bias[..., start_k:]

        if alibi_bias.shape[0] == 1 and batch_size != 1:
            alibi_bias = jnp.broadcast_to(alibi_bias, (batch_size, *alibi_bias.shape[1:]))

        if alibi_bias.shape[-2] == 1 and q_len != 1:
            alibi_bias = jnp.broadcast_to(alibi_bias, (*alibi_bias.shape[:-2], q_len, kv_len))
        elif alibi_bias.shape[-2] != q_len:
            start_q = max(0, alibi_bias.shape[-2] - q_len)
            alibi_bias = alibi_bias[..., start_q:, :]

        attention = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=alibi_bias,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=None,
            mask_info=mask_info,
            causal=causal_for_kernel,
        )

        attn_output = self.shard_attention_prod(
            attention.attention_outputs.reshape(batch_size, sequence_length, self.config.hidden_size)
        )
        attn_output = checkpoint_name(self.out_proj(attn_output), name="attn_output")
        attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attention.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class MptBlock(spx.Module):
    """One MPT decoder block: pre-norm attention plus pre-norm GELU FFN.

    Layout (input ``x``)::

        h = x + Dropout(attn(LN1(x), alibi_bias=...))
        out = h + Dropout(MLP(LN2(h)))         # residual added inside MptMLP

    Both LayerNorms are MPT's "low-precision LayerNorm" (epsilon =
    ``layer_norm_epsilon``, optional bias per ``use_norm_bias``). Note the
    asymmetry: the attention residual is added *here* (with a residual-only
    dropout), while the FFN residual is added *inside* :class:`MptMLP` —
    that matches the upstream MosaicML implementation.

    Attributes:
        norm_1, norm_2 (LayerNorm): Pre-attention and pre-FFN LayerNorms.
        attn (MptAttention): The ALiBi multi-head attention.
        ffn (MptMLP): GELU feed-forward (returns post-residual hidden state).
        resid_attn_dropout (nn.Dropout): Dropout applied to the attention
            output before adding the residual.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initializes the MptBlock module.

        Args:
            config (MptConfig): The configuration object for the MPT model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.norm_1 = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.attn = MptAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.norm_2 = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.ffn = MptMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.dropout_rate = self.config.attn_config.attn_pdrop
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate, rngs=rngs)

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
        position_bias: Float[Array, "batch heads seq_len seq_len"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass of the MptBlock.

        Applies pre-norm architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual.
        Uses ALiBi positional encoding through the position_bias parameter.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_dim).
            mask_info: Mask information for attention computation.
            position_ids: Position indices of shape (batch_size, seq_len).
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
            cache_view: Cache view for key/value states in generation.
            cache_metadata: Metadata for cache handling.
            output_attentions: Whether to return attention weights.
            frequencies: Unused, kept for interface compatibility.
            position_bias: ALiBi positional bias tensor of shape (batch, heads, seq_len, seq_len).

        Returns:
            DecoderLayerOutput containing hidden states, optional attention weights,
            and updated cache view.
        """
        attn_outputs = self.attn(
            self.norm_1(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
            alibi=position_bias,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs.attention_output) + hidden_states
        output = self.ffn(self.norm_2(hidden_states), hidden_states)

        return DecoderLayerOutput(
            hidden_states=output,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8):
    """Precompute the per-head ALiBi position-bias tensor.

    ALiBi (Press, Smith & Lewis, 2022) replaces sinusoidal/learned position
    embeddings with an additive bias on the attention scores

    .. math::
        \\text{score}_{ij} \\mathrel{+}= -m_h \\, \\max(i - j, 0)

    where :math:`m_h` is a fixed per-head slope from a geometric sequence.
    Slopes are chosen to be ``2^{-(8 / H_2) k}`` for ``k = 1, …, H_2`` with
    ``H_2`` the next power of two ≥ ``num_heads``; when the next power of
    two overshoots, the function interleaves the even and odd entries and
    truncates to ``num_heads`` to preserve the slope schedule used in the
    original MPT paper. Because the slopes are fixed (not learned) and the
    bias is purely a function of ``i - j``, MPT can extrapolate to context
    lengths longer than seen at training time.

    Args:
        num_heads: Number of attention heads ``H``.
        sequence_length: Maximum context length to materialize.
        alibi_bias_max: Cap on the largest slope (default ``8`` matches MPT).

    Returns:
        jax.Array: Tensor of shape ``(1, num_heads, 1, sequence_length)``
        of non-positive position biases ready to be broadcast into the
        attention score matrix.
    """
    alibi = jnp.arange(
        1 - sequence_length,
        1,
        dtype="i4",
    ).reshape(
        1,
        1,
        1,
        sequence_length,
    )
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = jnp.arange(1, num_heads_power_of_2 + 1, dtype=jnp.int32).astype("float32")
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / jnp.pow(2, base)
    slopes = slopes.reshape(
        1,
        num_heads_power_of_2,
        1,
        1,
    )

    if num_heads_power_of_2 != num_heads:
        slopes = jnp.concat(
            [slopes[:, 1::2, ...], slopes[:, ::2, ...]],
            axis=1,
        )[:, :num_heads, ...]

    alibi = alibi * slopes
    return alibi


@register_module(TaskType.BASE_MODULE, config=MptConfig, model_type="mpt")
class MptModel(EasyDeLBaseModule):
    """MPT base trunk: token embeddings + N MptBlocks + final LayerNorm.

    No positional embedding table is allocated — MPT relies entirely on the
    cached ALiBi bias produced by :func:`build_mpt_alibi_tensor` and reused
    across every block (computed lazily once per forward to a length that
    bounds the input). Tied LM head behaviour is governed by ``use_lm_head``
    in the config (note the unusual semantic: in MPT, ``use_lm_head=True``
    means the LM logits come from the input embedding rather than a separate
    matrix). The trunk uses :class:`LayerNorm` rather than RMSNorm because
    that is what MPT was trained with.

    Attributes:
        wte (Embed): Token embedding ``(vocab_size, d_model)``.
        emb_drop (nn.Dropout): Embedding dropout (rate ``emb_prob_drop``).
        blocks (nn.ModuleList[MptBlock]): ``n_layers`` decoder blocks.
        norm_f (LayerNorm): Final LayerNorm at ``layer_norm_epsilon``.
        alibi (jax.Array | None): Lazily-cached ALiBi position-bias tensor.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initializes the MptModel.

        Args:
            config (MptConfig): The configuration object for the MPT model.
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
        self.wte = Embed(
            num_embeddings=config.vocab_size,
            features=config.d_model,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        remat_layer_block = auto_remat(
            MptBlock,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.blocks = nn.ModuleList([])
        for i in range(self.config.n_layers):
            with spx.assign_stage(total=self.config.n_layers, current=i):
                self.blocks.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        self.norm_f = LayerNorm(
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=config.layer_norm_epsilon,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )

    @cached_property
    def alibi(self):
        """Compute and cache the ALiBi positional bias tensor.

        Returns:
            ALiBi tensor of shape (1, num_heads, max_seq_len, max_seq_len) containing
            position-dependent attention biases.
        """
        return build_mpt_alibi_tensor(
            sequence_length=self.config.max_seq_len,
            num_heads=self.config.n_heads,
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
        """Forward pass through the MPT transformer model.

        Processes input tokens through learned token embeddings (no explicit positional
        embeddings - uses ALiBi), multiple transformer blocks, and final layer normalization.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Either this
                or `inputs_embeds` must be provided but not both.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length,
                hidden_size). Use instead of `input_ids` for custom embeddings.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) indicating
                which tokens to attend to (True) and which to ignore (False).
            mask_info: Pre-computed mask information. If provided, overrides `attention_mask`.
            position_ids: Position indices of shape (batch_size, sequence_length). Used for
                ALiBi bias computation.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER). Auto-detected if None.
            past_key_values: Cached key/value states for efficient autoregressive generation.
            cache_metadata: Metadata for paged attention mechanisms.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer output of shape (batch, seq_len, hidden_size)
                - past_key_values: Updated cache for next generation step
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are provided, or if neither
                is provided.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.astype("i4"))
        sequence_length = inputs_embeds.shape[1]

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

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.blocks))

        def _layer_loop(block, carry):
            hidden_states, all_hidden_states, all_attentions, idx = carry
            with self._layer_stage_context(idx, layers=self.blocks):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=None,
                    position_bias=self.alibi,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.blocks)
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            return hidden_states, all_hidden_states, all_attentions, idx + 1

        hidden_states, all_hidden_states, all_attentions, _ = self.blocks.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, 0),
            trace=True,
        )
        hidden_states = self.norm_f(hidden_states)
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
        return self.wte


@register_module(TaskType.CAUSAL_LM, config=MptConfig, model_type="mpt")
class MptForCausalLM(BaseCausalLMModule[MptModel, MptConfig]):
    """MPT model with a language modeling head for causal language modeling.

    This model extends the base MptModel by adding a linear layer on top to
    predict the next token in a sequence, making it suitable for causal language
    modeling tasks. It uses ALiBi for positional encoding.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (spx.Rngs): Random number generators.
        transformer (MptModel): The base MPT transformer model.
        lm_head (nn.Linear): Linear layer for language modeling predictions.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "mpt"
    _config_class = MptConfig

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the MPT causal language model.

        Args:
            config: Configuration object for the MPT model.
            dtype: Data type for computation. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX operations. Defaults to None.
            rngs: Random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=MptModel,
            base_model_name="transformer",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=config.use_bias if hasattr(config, "use_bias") else False,
        )
