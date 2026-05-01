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

"""Mixtral Mixture-of-Experts transformer implementation.

Implements Mistral AI's Mixtral architecture: a decoder-only transformer where
the standard MLP is replaced by a sparse Mixture-of-Experts block. A per-token
top-k router selects ``num_experts_per_tok`` experts out of ``num_local_experts``
SwiGLU experts; an auxiliary load-balancing loss is exposed via
``output_router_logits``. Attention layers use grouped-query attention and a
configurable sliding-window mask.

Exports:
    - ``MixtralAttention``: standard / sliding-window attention.
    - ``MixtralMoEMlp``, ``MixtralSparseMoeBlock``: per-expert MLP and routing block.
    - ``MixtralDecoderLayer``: one transformer block.
    - ``MixtralModel``: base transformer trunk.
    - ``MixtralForCausalLM``: causal LM head wrapper with auxiliary MoE loss.
    - ``MixtralForSequenceClassification``: classification head wrapper.
"""

import typing

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
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNorm,
    RowParallelMoELinear,
)
from easydel.layers.attention import UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .mixtral_configuration import MixtralConfig as MixtralConfig


class MixtralAttention(UnifiedAttention):
    """Mixtral attention: GQA + sliding-window + RoPE.

    Mixtral inherits the default ``"standard"`` attention implementation
    from :class:`UnifiedAttention` and only adds three twists relative to
    LLaMA-style attention:

    - **Sliding-window causal mask**: tokens can only attend to the most
      recent ``config.sliding_window`` keys, dropping attention cost from
      O(L²) to O(L·W) per layer. Long-range mixing comes from stacking
      layers (each token sees a window of size ``W`` per layer, so after
      ``L`` layers the receptive field is ``L·W``).
    - **Grouped-query attention** with ``num_key_value_heads = 8`` (a
      quarter of ``num_attention_heads`` in 8x7B), shrinking KV cache
      memory by 4×.
    - **High RoPE base** (``rope_theta = 1e6``) for long-context
      extrapolation up to ~130k tokens.

    Inherited capabilities:
    """

    def __init__(
        self,
        config: MixtralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Mixtral attention with sliding window configuration.

        Args:
            config (MixtralConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window,
        )

    def _create_rotary(self, config: MixtralConfig, dtype: jnp.dtype):
        """Build the rotary embedding for this attention layer.

        Args:
            config (MixtralConfig): Model configuration providing ``rope_theta`` and
                any optional ``rope_scaling`` settings.
            dtype (jnp.dtype): Computation dtype for the rotary tables.

        Returns:
            Rotary module configured with this layer's head dimension.
        """
        return config.get_basic_rope(dtype, self.head_dim)


class MixtralMoEMlp(spx.Module):
    """Batched SwiGLU expert bank used inside :class:`MixtralSparseMoeBlock`.

    Mixtral keeps the original LLaMA SwiGLU shape (w1 = gate, w3 = up,
    w2 = down) but stacks ``num_local_experts`` copies along an extra
    leading axis so all experts can be evaluated as a single grouped GEMM
    via :class:`ColumnParallelMoELinear` / :class:`RowParallelMoELinear`.
    Per token / per assigned expert the forward computes
    ``y = w2(silu(w1(x)) * w3(x))``. Tokens arrive already permuted into
    expert-major layout (the parent block does the permutation), and
    ``group_sizes`` tells the GEMM how many tokens go to each slice.
    Tensor parallelism shards the hidden axis; expert parallelism shards
    the leading expert axis (the parent toggles between EP and TP via
    ``use_expert_tensor_mode``).

    Attributes:
        w1 (ColumnParallelMoELinear): Gate projection ``hidden -> intermediate``.
        w3 (ColumnParallelMoELinear): Up projection ``hidden -> intermediate``.
        w2 (RowParallelMoELinear): Down projection ``intermediate -> hidden``.
        act_fn (Callable): Per-expert activation (typically SiLU).
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "w1.weight", "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2)},
                {"name": "w3.weight", "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2)},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat(
                (gate.transpose(-1, -2), up.transpose(-1, -2)),
                dim=1,
            ),
        },
        "down_proj$": {
            "splits": [
                {"name": "w2.weight", "spliter": lambda x: x.swapaxes(-1, -2)},
            ],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
    }

    def __init__(
        self,
        config: MixtralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mixtral MoE MLP block.

        Args:
            config (MixtralConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.w1 = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.w2 = RowParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.w3 = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply SwiGLU feedforward transformation for MoE.

        Args:
            x (Array): Input tensor for the expert MLP.
            group_sizes (Array): Sizes of token groups assigned to each expert.
            sorted_experts (Array | None, optional): Sorted expert indices for routing.
                Defaults to None.

        Returns:
            Array: Transformed hidden states after expert MLP processing.
        """
        hidden_states = checkpoint_name(self.act_fn(self.w1(x, group_sizes, sorted_experts)), "mlp_gate")
        hidden_states = checkpoint_name(hidden_states * self.w3(x, group_sizes, sorted_experts), "mlp_up")
        outputs = checkpoint_name(self.w2(hidden_states, group_sizes, sorted_experts), "mlp_down")
        return checkpoint_name(outputs, "mlp_output")


class MixtralSparseMoeBlock(BaseMoeModule):
    """Top-k softmax-routed sparse-MoE FFN at the heart of Mixtral.

    Routing protocol (per token ``x``):

    1. ``logits = router(x)`` produces a vector of length ``num_local_experts``.
    2. Add Uniform(``-router_jitter_noise``, ``+router_jitter_noise``) noise
       during training (only) — this is the canonical Mixtral exploration
       trick that prevents premature expert collapse.
    3. ``probs = softmax(logits)``; pick the top-``num_experts_per_tok``
       experts and renormalize their weights to sum to one.
    4. Permute tokens into expert-major layout, run :class:`MixtralMoEMlp`
       in a single grouped GEMM, then unpermute and combine with the
       weighted sum :math:`\\sum_e w_e \\, \\text{expert}_e(x)`.

    Auxiliary load-balancing loss (Switch Transformer-style)
    :math:`\\mathcal{L}_{aux} = N_e \\sum_i f_i p_i` is exposed via
    ``router_logits`` so that :class:`BaseCausalLMModule` can fold it into
    the final training loss with coefficient ``router_aux_loss_coef``.
    Note: unlike DeepSeek-V3 / Llama-4, Mixtral has *no* shared expert —
    every token is processed by exactly ``num_experts_per_tok`` routed
    experts and that's it.

    Attributes:
        gate (ColumnParallelLinear): Single-layer router producing
            ``hidden_size -> num_local_experts`` logits.
        experts (MixtralMoEMlp): Batched SwiGLU expert bank.
        jitter_noise (float): ``router_jitter_noise``; controls training
            exploration noise.
    """

    def __init__(
        self,
        config: MixtralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mixtral Sparse MoE block.

        Args:
            config (MixtralConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=getattr(config, "router_aux_loss_coef", None),
            rzl_coef=getattr(config, "router_z_loss_coef", None),
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )

        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        # Router/gate
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(),
        )

        # Expert MLPs
        self.experts = MixtralMoEMlp(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, hidden_state: Array) -> tuple[Array, Array]:
        """Forward pass through the Sparse MoE block.

        Routes input tokens to selected experts and combines their outputs.

        Args:
            hidden_state (Array): Input hidden states of shape (batch, seq_len, hidden_dim).

        Returns:
            tuple[Array, Array]: A tuple containing:
                - Output hidden states after MoE processing (batch, seq_len, hidden_dim).
                - Router logits for auxiliary loss computation (batch, seq_len, num_experts).
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_state,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.w1.weight.value,
            wu_kernel=self.experts.w3.weight.value,
            wd_kernel=self.experts.w2.weight.value,
            act_fn=self.experts.act_fn,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class MixtralDecoderLayer(spx.Module):
    """One Mixtral block: pre-norm sliding-window attention + sparse-MoE FFN.

    Layout (input ``x``)::

        x = x + self_attn(input_layernorm(x))
        x = x + block_sparse_moe(post_attention_layernorm(x))   # also returns router_logits

    Both norms are :class:`RMSNorm` at ``rms_norm_eps``. The FFN sub-layer
    is a :class:`MixtralSparseMoeBlock` that returns both the combined
    expert output and the per-token router logits — the parent
    :class:`MixtralModel` collects those logits across layers so the
    auxiliary load-balancing loss can be computed once at the LM-head
    level (see ``output_router_logits`` on the model forward).

    Attributes:
        self_attn (MixtralAttention): Sliding-window GQA attention.
        block_sparse_moe (MixtralSparseMoeBlock): Top-k MoE FFN.
        input_layernorm, post_attention_layernorm (RMSNorm): Pre-attention
            and pre-FFN RMSNorms.
    """

    # Accept HF MoE tensor naming (`mlp.*`) directly during state-dict conversion
    # so conversion does not depend on test-only key remaps.
    reform_param: typing.ClassVar = {
        "mlp.gate.weight$": {
            "splits": [{"name": "block_sparse_moe.gate.weight", "spliter": lambda x: x.swapaxes(-1, -2)}],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
        "mlp.experts.gate_up_proj$": {
            "splits": [
                {
                    "name": "block_sparse_moe.experts.w1.weight",
                    "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2),
                },
                {
                    "name": "block_sparse_moe.experts.w3.weight",
                    "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2),
                },
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat(
                (gate.transpose(-1, -2), up.transpose(-1, -2)),
                dim=1,
            ),
        },
        "mlp.experts.down_proj$": {
            "splits": [{"name": "block_sparse_moe.experts.w2.weight", "spliter": lambda x: x.swapaxes(-1, -2)}],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
    }

    def __init__(
        self,
        config: MixtralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Mixtral decoder layer.

        Args:
            config (MixtralConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.self_attn = MixtralAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.block_sparse_moe = MixtralSparseMoeBlock(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + moe(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal and sliding window masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits, and cache view.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        attn_outputs = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=MixtralConfig, model_type="mixtral")
class MixtralModel(EasyDeLBaseModule):
    """The base Mixtral model transformer.

    This class represents the core transformer architecture of the Mixtral model,
    consisting of an embedding layer, multiple MixtralDecoderLayer layers (with sparse MoE),
    and a final layer normalization.

    Attributes:
        config (MixtralConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (spx.Rngs): Random number generators.
        embed_tokens (Embed): Embedding layer for input tokens.
        layers (tp.List[MixtralDecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: MixtralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mixtral base model.

        Args:
            config (MixtralConfig): Model configuration.
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

        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        remat_layer_block = auto_remat(
            MixtralDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for idx in range(config.num_hidden_layers):
            with spx.assign_stage(total=config.num_hidden_layers, current=idx):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                        layer_idx=idx,
                    )
                )

        self.norm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        trace: bool = False,
    ) -> MoeModelOutput:
        """Forward pass through the Mixtral base model.

        Processes input tokens through embedding, all decoder layers with sliding window
        attention, Sparse MoE, and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None (uses config.output_attentions).
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None (uses config.output_hidden_states).
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None (uses config.output_router_logits).
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

        Returns:
            MoeModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                optional router_logits, and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
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

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
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

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views

        trace_layers = self._layer_scan_trace(
            trace,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=output_router_logits or needs_trace_cache,
        )
        cache_views = views if trace_layers else None

        def _run_layer(block, carry):
            hs, cv, ah, aa, ar, idx = carry
            if output_hidden_states:
                ah = (*ah, hs)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hs,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=self.frequencies,
                )
            hs = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)
            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_attentions:
                aa = (*aa, layer_outputs.attention_weight)
            if output_router_logits and layer_outputs.router_logits is not None:
                ar = (*ar, layer_outputs.router_logits)
            return hs, cv, ah, aa, ar, idx + 1

        init_carry = (hidden_states, cache_views, all_hidden_states, all_self_attns, all_router_logits, 0)
        hidden_states, _, all_hidden_states, all_self_attns, all_router_logits, _ = self.layers.scan(
            _run_layer,
            init_carry,
            trace=trace_layers,
        )
        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
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
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=MixtralConfig, model_type="mixtral")
class MixtralForCausalLM(BaseCausalLMModule[MixtralModel, MixtralConfig]):  # type: ignore
    """Mixtral model with a language modeling head for causal language modeling tasks.

    This model is a sparse MoE transformer-based language model with causal attention masks
    and sliding window attention applied to perform autoregressive language generation.

    Attributes:
        config (MixtralConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "mixtral"
    _config_class = MixtralConfig

    def __init__(
        self,
        config: MixtralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mixtral model for causal language modeling.

        Args:
            config (MixtralConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=MixtralModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass through the Mixtral causal language model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply language model head projection.
                Defaults to True.

        Returns:
            MoeCausalLMOutput: Contains logits, optional hidden_states, optional attentions,
                optional router_logits, auxiliary loss, and updated past_key_values.
        """
        return self.forward_moe(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            aux_loss_fn=self._compute_aux_loss,
        )

    def _compute_aux_loss(self, outputs, attention_mask):
        """Compute auxiliary load balancing loss from router logits.

        Args:
            outputs: Model outputs containing router_logits from MoE layers.
            attention_mask: Attention mask to exclude padding tokens from loss computation.

        Returns:
            Optional auxiliary loss value for load balancing, or None if router_logits unavailable.
        """
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None
        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=self.config.num_local_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )
        return aux_loss + (aux_loss * self.config.router_aux_loss_coef)


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=MixtralConfig, model_type="mixtral")
class MixtralForSequenceClassification(BaseSequenceClassificationModule[MixtralModel, MixtralConfig]):  # type: ignore
    """Mixtral model for sequence classification tasks.

    This class extends the base Mixtral model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
        config (MixtralConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "mixtral"
    _config_class = MixtralConfig

    def __init__(
        self,
        config: MixtralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mixtral model for sequence classification.

        Args:
            config (MixtralConfig): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=MixtralModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            classifier_name="score",
            classifier_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the Mixtral sequence classification model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

        Returns:
            SequenceClassifierOutput: Contains classification logits, optional hidden_states,
                optional attentions, auxiliary loss, and updated past_key_values.
        """
        transformer_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jnp.argmax(jnp.equal(input_ids, self.config.pad_token_id).astype("i4"), -1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        aux_loss = None
        if output_router_logits and transformer_outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=transformer_outputs.router_logits,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            aux_loss = aux_loss * self.config.router_aux_loss_coef

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=aux_loss,
        )
