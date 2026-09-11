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

"""Configuration classes for the GLM-5-Next (GLM-5.3-Flash) model family.

GLM-5-Next is a multimodal MoE decoder that pairs every layer with one of two
attention mechanisms scheduled 3:1:

- **KDA linear attention** (``"linear_attention"``) — Kimi-style gated delta
  rule with a fused depthwise short conv, *per-channel* log decay, and gated
  RMSNorm output (``hc_mult`` residual streams via mHC hyper-connections wrap
  every block).
- **DSA/MLA sparse attention** (``"deepseek_sparse_attention"``) — DeepSeek
  style Multi-head Latent Attention whose KV pages are restricted each step to
  the ``index_topk`` tokens chosen by a k-pooled lightning indexer (groups of
  ``index_kpool`` keys are pooled with a softmax-gated average and scored with
  ReLU dot-product heads).

The text stack is fully NoPE (``qk_rope_head_dim=0`` — no rotary embeddings
anywhere), routes through a sigmoid grouped-top-k MoE with a correction bias
(288 routed + 1 shared experts, top-8), clamps every SwiGLU gate/up
projection at ``swiglu_limit``, and carries ``hc_mult=4`` mHC
hyper-connection streams through the residual path.

Defaults match `zai-org/GLM-5.3-Flash` on the Hugging Face Hub.
"""

import typing
import typing as tp

from easydel.caching.hybrid import FULL_ATTENTION, KDA_LINEAR_ATTENTION
from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

LINEAR_ATTENTION_LAYER_TYPE = "linear_attention"
DEEPSEEK_SPARSE_ATTENTION_LAYER_TYPE = "deepseek_sparse_attention"

GLM5_NEXT_TEXT_LAYER_TYPES: tuple[str, str] = (
    LINEAR_ATTENTION_LAYER_TYPE,
    DEEPSEEK_SPARSE_ATTENTION_LAYER_TYPE,
)
GLM5_NEXT_MLP_LAYER_TYPES: tuple[str, str] = ("dense", "sparse")
GLM5_NEXT_INDEXER_TYPES: tuple[str, str] = ("full", "shared")


@register_config("glm5_next_text")
class Glm5NextTextConfig(EasyDeLBaseConfig):
    """Configuration class for the GLM-5-Next text decoder.

    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden representations (the mHC stream width).
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimensionality of the dense MLP intermediate layer.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of each MoE expert's intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 45):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for the MLA/DSA layers.
        num_key_value_heads (`int`, *optional*, defaults to 64):
            Number of key-value heads (equal to ``num_attention_heads`` for MLA).
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts that always process every token.
        n_routed_experts (`int`, *optional*, defaults to 288):
            Number of routed experts in MoE layers.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor applied to routed expert weights after normalisation.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the low-rank KV compression in MLA.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the low-rank query decomposition (required for DSA).
        qk_rope_head_dim (`int`, *optional*, defaults to 0):
            RoPE subspace of Q/K heads. Must stay ``0`` — GLM-5-Next is NoPE.
        qk_nope_head_dim (`int`, *optional*, defaults to 256):
            Non-positional Q/K head dimension.
        v_head_dim (`int`, *optional*, defaults to 256):
            Value head dimension.
        n_group (`int`, *optional*, defaults to 1):
            Number of expert groups for grouped top-k routing.
        topk_group (`int`, *optional*, defaults to 1):
            Number of groups activated per token.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Routed experts activated per token.
        norm_topk_prob (`bool`, *optional*, defaults to ``True``):
            Whether to renormalise top-k routing weights.
        hidden_act (`str`, *optional*, defaults to ``"silu"``):
            Activation used in MLP gate projections.
        max_position_embeddings (`int`, *optional*, defaults to 1048576):
            Maximum sequence length the model supports.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialisation.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalisation layers.
        use_cache (`bool`, *optional*, defaults to ``True``):
            Whether to return past key/values for caching.
        pad_token_id (`int`, *optional*, defaults to 154820):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning-of-stream token id (unused by the GLM-5 tokenizer).
        eos_token_id (`int | list[int]`, *optional*):
            End-of-stream token id(s).
        tie_word_embeddings (`bool`, *optional*, defaults to ``False``):
            Whether to tie input embeddings with the LM head.
        mlp_layer_types (`list[str]`, *optional*):
            Per-layer feed-forward schedule; ``"dense"`` or ``"sparse"``.
            Defaults to the first three layers dense and the remainder sparse.
        attention_bias (`bool`, *optional*, defaults to ``False``):
            Whether attention projections carry biases.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention weights.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of sparse-attention positions selected by the DSA indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            DSA indexer projection head dimension.
        index_n_heads (`int`, *optional*, defaults to 32):
            Number of DSA indexer heads.
        index_kpool (`int`, *optional*, defaults to 16):
            Pool size of the compressed key groups selected by the DSA indexer.
        index_kpool_always_select_tail (`bool`, *optional*, defaults to ``True``):
            Whether the incomplete trailing pool is always included.
        layer_types (`list[str]`, *optional*):
            Per-layer attention schedule; ``"linear_attention"`` (KDA) or
            ``"deepseek_sparse_attention"`` (MLA+DSA). Defaults to KDA except
            every 4th layer (``idx % 4 == 3``). ``"full_attention"`` entries
            are accepted and mapped to ``"deepseek_sparse_attention"``.
        indexer_types (`list[str]`, *optional*):
            Per-layer DSA indexer mode; ``"full"`` runs the layer's indexer,
            ``"shared"`` reuses the previous full layer's top-k selection.
            Derived from ``index_topk_pattern`` (or the freq/offset schedule,
            all-``"full"`` by default) when omitted.
        index_topk_pattern (`str | list[str]`, *optional*):
            Compact indexer schedule (``"F"``/``"S"`` string or list) used to
            derive ``indexer_types``; overrides the freq/offset schedule.
        index_topk_freq (`int`, *optional*, defaults to 1):
            Period of ``"full"`` layers when deriving ``indexer_types``.
        index_skip_topk_offset (`int`, *optional*, defaults to 2):
            Layer offset applied before the freq modulo when deriving
            ``indexer_types``.
        swiglu_limit (`float`, *optional*, defaults to 10.0):
            Clamp limit applied to every SwiGLU gate/up projection.
        linear_head_dim (`int`, *optional*, defaults to 128):
            Head dimension of the KDA linear-attention layers.
        linear_num_heads (`int`, *optional*, defaults to 64):
            Number of heads in the KDA linear-attention layers.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the KDA fused depthwise short convolution.
        linear_lower_bound (`float`, *optional*, defaults to -5.0):
            Lower bound applied to the KDA forget-gate log decay.
        hc_mult (`int`, *optional*, defaults to 4):
            Number of mHC hyper-connection residual streams.
        hc_eps (`float`, *optional*, defaults to 1e-6):
            Numerical floor used by mHC Sinkhorn normalisation.
        hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
            Sinkhorn iterations used by mHC stream routing.
        output_router_logits (`bool`, *optional*, defaults to ``False``):
            Whether MoE layers should return router logits for aux losses.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Weight of the MoE router load-balancing aux loss.
        linear_attn_config (`dict`, *optional*):
            HF-style linear-attention dict; folded into ``linear_head_dim``,
            ``linear_num_heads``, ``linear_conv_kernel_dim`` and
            ``linear_lower_bound`` for checkpoint compatibility.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Kept for base-config compatibility; unused (the model is NoPE).
        rope_scaling (`dict`, *optional*):
            Kept for base-config compatibility; unused.
        rope_parameters (`dict`, *optional*):
            Kept for base-config compatibility; unused.
    """

    model_type: str = "glm5_next_text"
    base_config_key: str = "text_config"
    keys_to_ignore_at_inference: tp.ClassVar = ["past_key_values"]
    attribute_map: tp.ClassVar = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_shared_experts: int = 1,
        n_routed_experts: int = 288,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 0,
        mla_cache_rope_width: int = 128,
        indexer_max_rows: int = 8,
        qk_nope_head_dim: int = 256,
        v_head_dim: int = 256,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = 154820,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        mlp_layer_types: list[str] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_kpool: int = 16,
        index_kpool_always_select_tail: bool = True,
        layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        index_topk_pattern: str | list[str] | None = None,
        index_topk_freq: int = 1,
        index_skip_topk_offset: int = 2,
        swiglu_limit: float = 10.0,
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float | None = -5.0,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        linear_attn_config: dict[str, typing.Any] | None = None,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, typing.Any] | None = None,
        rope_parameters: dict[str, typing.Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize a GLM-5-Next text-decoder configuration.

        Mirrors HF ``Glm5NextTextConfig.__post_init__``: derives the 3:1
        KDA/DSA ``layer_types`` schedule, the per-layer ``indexer_types``
        schedule, and folds a legacy ``linear_attn_config`` dict, then runs
        architecture validation (MLA head symmetry, k-pool divisibility,
        required ``q_lora_rank``, NoPE).

        Args:
            See the class docstring for the full parameter table.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.

        Raises:
            ValueError: If any schedule has the wrong length or unknown
                entries, if ``n_routed_experts`` is not divisible by
                ``n_group``, or if an architecture invariant (head symmetry,
                ``index_kpool`` divisibility, ``q_lora_rank`` presence, NoPE)
                is violated.
        """
        # Composite-checkpoint absorption: `zai-org/GLM-5.3-Flash` ships a
        # multimodal config (`model_type: "glm5_next"`) whose text fields live
        # under a nested `text_config` dict. When this text config is built
        # from such a file (composite→text registry alias), flatten the nested
        # dict into the flat keyword arguments; explicitly passed flat values
        # win. Composite-only keys (`vision_config`, image/video token ids)
        # stay in kwargs and land as plain attributes. `quantization_config`
        # (the fp8 recipe, at the composite top level) is captured separately
        # and re-applied after init: the loader field expects an
        # EasyQuantizer-style object, so the raw HF dict would be filtered.
        nested_text = kwargs.pop("text_config", None)
        if isinstance(nested_text, dict):
            nested_text = {k: v for k, v in nested_text.items() if k not in ("model_type", "architectures")}
            kwargs = {**nested_text, **kwargs}
        kwargs.pop("vision_config", None)
        hf_quantization_config = kwargs.pop("quantization_config", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        # Cache-side rope width for the MLA ragged-pages kernel (see
        # ``_create_mla_ragged_page_cache_config``): zero information, keeps
        # the Pallas rope component 128-aligned for NoPE serving.
        self.mla_cache_rope_width = mla_cache_rope_width
        # Per-token indexer packed-state width ([key | gate | valid]); read by
        # ``_create_mla_ragged_page_cache_config`` to size the per-request
        # indexer-state sidecar on the MLA ragged cache views.
        self.indexer_packed_dim = 2 * index_head_dim + 1
        # Request slots covered by the sidecar; must be >= eSurge
        # ``max_num_seqs`` (the serving batch width).
        self.indexer_max_rows = indexer_max_rows
        self.qk_nope_head_dim = qk_nope_head_dim
        # HF convention: `head_dim` tracks the RoPE dim (0 here); the effective
        # Q/K width lives on `qk_head_dim`.
        self.head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.swiglu_limit = swiglu_limit
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters

        if mlp_layer_types is None:
            dense_layers = min(3, num_hidden_layers)
            mlp_layer_types = ["dense"] * dense_layers + ["sparse"] * (num_hidden_layers - dense_layers)
        self.mlp_layer_types = mlp_layer_types

        # Per-layer attention schedule. Held back from ``super().__init__`` and
        # validated manually afterwards (same pattern as DeepSeek-V4): the
        # generic base validator would reject GLM-5's KDA / DSA layer-type
        # strings inside ``super().__init__``. Kept in locals (not attributes)
        # so ``to_dict`` stays clean of derivation-only inputs.
        pending_layer_types = layer_types
        pending_indexer_types = indexer_types
        pending_index_topk_pattern = index_topk_pattern
        pending_index_topk_freq = index_topk_freq
        pending_index_skip_topk_offset = index_skip_topk_offset

        if linear_attn_config is not None:
            self.linear_head_dim = linear_attn_config.get("head_dim", self.linear_head_dim)
            self.linear_num_heads = linear_attn_config.get("num_heads", self.linear_num_heads)
            self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
            gate_lower_bound = linear_attn_config.get("gate_lower_bound", self.linear_lower_bound)
            if linear_attn_config.get("safe_gate", True) and gate_lower_bound is None:
                gate_lower_bound = -5.0
            self.linear_lower_bound = gate_lower_bound

        # Always materialize the KDA cache geometry dict: the generation mixin's
        # ``create_kda_cache_config`` reads ``linear_attn_config['num_heads'] /
        # ['head_k_dim'] / ['head_v_dim'] / ['d_conv']`` to size the KDA
        # recurrent + conv states (its own defaults are Kimi's 128/128/4 and
        # are wrong for this family).
        self.linear_attn_config = {
            "num_heads": self.linear_num_heads,
            "head_k_dim": self.linear_head_dim,
            "head_v_dim": self.linear_head_dim,
            "d_conv": self.linear_conv_kernel_dim,
        }

        # The nested `text_config` merge above can re-introduce keys that are
        # also explicit named parameters; keep the explicit value when set,
        # otherwise adopt the merged one (avoids "multiple values" errors).
        explicit = {
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "tie_word_embeddings": tie_word_embeddings,
        }
        for name, value in list(explicit.items()):
            if name in kwargs:
                merged_value = kwargs.pop(name)
                if value is None:
                    explicit[name] = merged_value

        super().__init__(**explicit, **kwargs)
        # Assigned after ``super().__init__`` for the same reason as
        # DeepSeek-V4: the base strict-dataclass validators collected at
        # decoration time would run the generic layer-type check first.
        self.layer_types = self._resolve_layer_types(pending_layer_types)
        self.indexer_types = self._resolve_indexer_types(
            indexer_types=pending_indexer_types,
            index_topk_pattern=pending_index_topk_pattern,
            index_topk_freq=pending_index_topk_freq,
            index_skip_topk_offset=pending_index_skip_topk_offset,
        )
        if hf_quantization_config is not None and self.quantization_config is None:
            # Keep the raw HF fp8 recipe verbatim (quant_method/fmt/
            # weight_block_size/modules_to_not_convert): serving re-applies
            # fp8/w4a16 quantization from it via `apply_quantization`.
            self.quantization_config = hf_quantization_config
        self.validate_layer_type()
        self.validate_architecture()

    def _resolve_layer_types(self, layer_types: list[str] | None) -> list[str]:
        """Derive the per-layer attention schedule.

        Args:
            layer_types: Explicit schedule from the constructor, or ``None``
                to derive the 3:1 KDA/DSA default.

        Returns:
            List of ``"linear_attention"`` / ``"deepseek_sparse_attention"``
            entries, one per decoder layer. ``"full_attention"`` entries map
            to ``"deepseek_sparse_attention"`` (HF BC behaviour).
        """
        if layer_types is None:
            layer_types = [
                LINEAR_ATTENTION_LAYER_TYPE if idx % 4 != 3 else DEEPSEEK_SPARSE_ATTENTION_LAYER_TYPE
                for idx in range(self.num_hidden_layers)
            ]
        return [DEEPSEEK_SPARSE_ATTENTION_LAYER_TYPE if lt == FULL_ATTENTION else lt for lt in layer_types]

    def _resolve_indexer_types(
        self,
        indexer_types: list[str] | None,
        index_topk_pattern: str | list[str] | None,
        index_topk_freq: int,
        index_skip_topk_offset: int,
    ) -> list[str]:
        """Derive the per-layer DSA indexer schedule.

        Args:
            indexer_types: Explicit schedule from the constructor, or
                ``None`` to derive one.
            index_topk_pattern: Compact ``"F"``/``"S"`` schedule overriding
                the freq/offset derivation, or ``None``.
            index_topk_freq: Period of ``"full"`` layers in the freq/offset
                derivation.
            index_skip_topk_offset: Layer offset applied before the freq
                modulo.

        Returns:
            List of ``"full"`` / ``"shared"`` entries, one per decoder layer.
            With the defaults the derivation is all-``"full"``.
        """
        if indexer_types is None:
            if index_topk_pattern is not None:
                indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in index_topk_pattern]
                    if isinstance(index_topk_pattern, str)
                    else list(index_topk_pattern)
                )
            else:
                freq = max(index_topk_freq, 1)
                offset = index_skip_topk_offset
                indexer_types = [
                    "full" if (max(idx - offset + 1, 0) % freq) == 0 else "shared"
                    for idx in range(self.num_hidden_layers)
                ]
        return list(indexer_types)

    def validate_layer_type(self) -> None:
        """Validate the per-layer schedules against GLM-5-Next's own types.

        Raises:
            ValueError: On a length mismatch with ``num_hidden_layers`` or an
                unknown ``layer_types`` / ``mlp_layer_types`` /
                ``indexer_types`` entry.
        """
        num_hidden_layers = getattr(self, "num_hidden_layers", None)
        if num_hidden_layers is None:
            return
        for name, types, allowed in (
            ("layer_types", getattr(self, "layer_types", None), GLM5_NEXT_TEXT_LAYER_TYPES),
            ("mlp_layer_types", getattr(self, "mlp_layer_types", None), GLM5_NEXT_MLP_LAYER_TYPES),
            ("indexer_types", getattr(self, "indexer_types", None), GLM5_NEXT_INDEXER_TYPES),
        ):
            if types is None:
                continue
            if len(types) != num_hidden_layers:
                raise ValueError(f"`num_hidden_layers` ({num_hidden_layers}) must equal `len({name})` ({len(types)}).")
            bad = sorted({t for t in types if t not in allowed})
            if bad:
                raise ValueError(f"`{name}` entries must be one of {allowed} for GLM-5-Next; got {bad}.")

    def validate_architecture(self) -> None:
        """Validate GLM-5-Next architecture invariants (HF parity).

        Raises:
            ValueError: If MLA head counts disagree, ``index_kpool`` is not a
                positive divisor of ``index_topk``, ``q_lora_rank`` is
                missing, or the attention carries a RoPE subspace (the model
                is NoPE).
        """
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be the same as "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}.")
        if self.index_topk % self.index_kpool != 0:
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")
        if self.q_lora_rank is None:
            raise ValueError("For DSA usage in the attention layers, the `q_lora_rank` is strictly required!")
        if self.qk_rope_head_dim > 0:
            raise ValueError(
                f"Expecting NoPE for the DSA attention layers, but got {self.qk_rope_head_dim} as RoPE dim."
            )
        if self.n_routed_experts is not None and self.n_group > 0 and self.n_routed_experts % self.n_group != 0:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must be divisible by n_group ({self.n_group})."
            )

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Check whether ``layer_idx`` is a KDA linear-attention layer.

        Args:
            layer_idx: Zero-based decoder layer index.

        Returns:
            True for ``"linear_attention"`` layers, False otherwise.
        """
        return self.layer_types[layer_idx] == LINEAR_ATTENTION_LAYER_TYPE

    def is_dsa_layer(self, layer_idx: int) -> bool:
        """Check whether ``layer_idx`` is a DSA/MLA sparse-attention layer.

        Args:
            layer_idx: Zero-based decoder layer index.

        Returns:
            True for ``"deepseek_sparse_attention"`` layers, False otherwise.
        """
        return self.layer_types[layer_idx] == DEEPSEEK_SPARSE_ATTENTION_LAYER_TYPE

    def get_layer_types(self) -> tuple[str, ...]:
        """Get the per-layer cache schedule for :class:`HybridCache` init.

        Returns:
            Tuple of cache-level layer types — ``"kda_linear_attention"`` for
            KDA layers (conv + recurrent state) and ``"full_attention"`` for
            DSA/MLA layers (KV pages).
        """
        return tuple(
            KDA_LINEAR_ATTENTION if self.is_kda_layer(idx) else FULL_ATTENTION for idx in range(self.num_hidden_layers)
        )

    def get_mask_details(self) -> dict[int, typing.Any]:
        """Report per-layer attention mask details for engine accounting.

        KDA layers have no token-KV pages at all (they are reported as
        linear/full for grouping); DSA layers are causal full attention whose
        top-k restriction is applied inside the layer, not by the mask.

        Returns:
            Dict mapping layer index to :class:`~easydel.infra.utils.AttnMaskDetail`.
        """
        from easydel.infra.utils import AttnMaskDetail, AttnMaskType

        # NOTE: KDA layers must report FULL, not LINEAR — the eSurge cache
        # spec dispatcher (`create_kv_cache_specs_from_config`) only knows
        # FULL/SLIDING/CHUNK and treats linear layers as full-attention
        # groups for scheduler grouping (same as `AttnMaskType.from_hf`).
        mapping: dict[int, typing.Any] = {}
        for layer_idx in range(self.num_hidden_layers):
            mapping[layer_idx] = AttnMaskDetail(mask_type=AttnMaskType.FULL, size=None, chunks=None)
        return mapping


@register_config("glm5_next_vision")
class Glm5NextVisionConfig(EasyDeLBaseConfig):
    """Configuration class for the GLM-5-Next vision encoder.

    A 24-layer ViT with axial rotary position embeddings and a patch merger
    whose SwiGLU projections carry the same ``swiglu_limit`` clamp as the
    text stack.

    Args:
        depth (`int`, *optional*, defaults to 24):
            Number of transformer layers in the vision encoder.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder hidden states.
        hidden_act (`str`, *optional*, defaults to ``"silu"``):
            Activation used in the vision MLPs.
        attention_bias (`bool`, *optional*, defaults to ``True``):
            Whether vision attention projections carry biases.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for vision attention weights.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in each vision layer.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels (RGB).
        image_size (`int`, *optional*, defaults to 336):
            Input image resolution.
        patch_size (`int`, *optional*, defaults to 14):
            Size of each image patch for the patch embedding.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for vision RMS normalisation layers.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Factor for spatial downsampling of visual features.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Temporal patch size for video processing.
        out_hidden_size (`int`, *optional*, defaults to 1536):
            Output projection dimension fed into the language model.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the vision MLP intermediate layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialisation.
        rope_parameters (`dict`, *optional*):
            Axial rotary-embedding parameters for the vision encoder.
        projection_intermediate_size (`int`, *optional*, defaults to 10240):
            Inner width of the patch-merger SwiGLU MLP.
        swiglu_limit (`float`, *optional*, defaults to 10.0):
            Clamp limit applied to the patch-merger SwiGLU projections.
    """

    model_type: str = "glm5_next_vision"
    base_config_key: str = "vision_config"
    attribute_map: tp.ClassVar = {"num_attention_heads": "num_heads"}

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        hidden_act: str = "silu",
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        num_heads: int = 16,
        in_channels: int = 3,
        image_size: int = 336,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 1536,
        intermediate_size: int = 4096,
        initializer_range: float = 0.02,
        rope_parameters: dict[str, typing.Any] | None = None,
        projection_intermediate_size: int = 10240,
        swiglu_limit: float = 10.0,
        **kwargs,
    ) -> None:
        """Initialize the GLM-5-Next vision encoder configuration.

        Args:
            See the class docstring for the full parameter table.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.rms_norm_eps = rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters
        self.projection_intermediate_size = projection_intermediate_size
        self.swiglu_limit = swiglu_limit


@register_config("glm5_next")
class Glm5NextConfig(EasyDeLBaseConfig):
    """Configuration class for the GLM-5-Next multimodal model.

    Thin wrapper around :class:`Glm5NextTextConfig` and
    :class:`Glm5NextVisionConfig` plus the special-token ids that mark image
    and video spans inside the text stream. Flat (text-only) checkpoints that
    store text fields at the top level are supported: when ``text_config`` is
    omitted the remaining keyword arguments are forwarded to the text config.

    Args:
        text_config (`dict | Glm5NextTextConfig`, *optional*):
            Configuration for the text decoder. A dict is converted to
            :class:`Glm5NextTextConfig`; ``None`` builds the text config from
            the remaining top-level kwargs (HF flat-checkpoint behaviour) or
            the class defaults.
        vision_config (`dict | Glm5NextVisionConfig`, *optional*):
            Configuration for the vision encoder. A dict is converted to
            :class:`Glm5NextVisionConfig`; ``None`` builds the class defaults.
        image_token_id (`int`, *optional*, defaults to 154854):
            Token id used as the image placeholder in the input stream.
        video_token_id (`int`, *optional*, defaults to 154855):
            Token id used as the video placeholder in the input stream.
        image_start_token_id (`int`, *optional*, defaults to 154830):
            Token id marking the start of an image sequence.
        image_end_token_id (`int`, *optional*, defaults to 154831):
            Token id marking the end of an image sequence.
        video_start_token_id (`int`, *optional*, defaults to 154832):
            Token id marking the start of a video sequence.
        video_end_token_id (`int`, *optional*, defaults to 154833):
            Token id marking the end of a video sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to ``False``):
            Whether to tie input embeddings with the LM head.
    """

    model_type: str = "glm5_next"
    sub_configs: tp.ClassVar = {
        "vision_config": Glm5NextVisionConfig,
        "text_config": Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference: tp.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: dict[str, typing.Any] | Glm5NextTextConfig | None = None,
        vision_config: dict[str, typing.Any] | Glm5NextVisionConfig | None = None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a GLM-5-Next configuration.

        Args:
            text_config: Text-decoder config or dict; ``None`` falls back to
                the remaining kwargs (flat text-only checkpoints) or defaults.
            vision_config: Vision-encoder config or dict; ``None`` falls back
                to the class defaults.
            image_token_id: Image placeholder token id.
            video_token_id: Video placeholder token id.
            image_start_token_id: Image-sequence start token id.
            image_end_token_id: Image-sequence end token id.
            video_start_token_id: Video-sequence start token id.
            video_end_token_id: Video-sequence end token id.
            tie_word_embeddings: Whether to tie input embeddings with the LM
                head.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`; when
                ``text_config`` is ``None`` they are also forwarded to the
                text sub-config.
        """
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self._fix_parent_kws(text_config, kwargs))
        elif text_config is None:
            # Flat (text-only) GLM-5.3-Flash checkpoints store the text fields
            # at the top level; forward them so `text_config` is populated.
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self._fix_parent_kws(vision_config, kwargs))
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = True) -> Glm5NextTextConfig:
        """Return the text decoder configuration.

        Args:
            decoder: Unused; kept for API compatibility with upstream HF
                configs.

        Returns:
            The text sub-config.
        """
        del decoder
        return self.text_config  # pyright: ignore[reportReturnType]

    def get_vision_config(self) -> Glm5NextVisionConfig:
        """Return the vision encoder configuration.

        Returns:
            The vision sub-config.
        """
        return self.vision_config  # pyright: ignore[reportReturnType]


__all__ = [
    "Glm5NextConfig",
    "Glm5NextTextConfig",
    "Glm5NextVisionConfig",
]
