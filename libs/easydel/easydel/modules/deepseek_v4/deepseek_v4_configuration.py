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

"""Configuration for DeepSeek-V4.

DeepSeek-V4 is a genuinely novel decoder architecture combining:

- Manifold-constrained hyper-connections (mHC): the residual stream is a stack
  of ``hc_mult`` parallel streams mixed per sub-layer through learned
  Sinkhorn-projected doubly-stochastic matrices.
- Shared-KV multi-query attention (``K == V``, single KV head) with per-head
  learnable attention sinks, partial *interleaved* RoPE on the trailing
  ``qk_rope_head_dim`` channels, an inverse output rotation, and a grouped
  low-rank output projection (``o_groups`` block-diagonal ``o_a_proj`` +
  ``o_b_proj``).
- Three per-layer attention regimes (``layer_types``): plain sliding-window
  attention, Compressed Sparse Attention (CSA, compress rate 4 + Lightning
  Indexer top-k), and Heavily Compressed Attention (HCA, compress rate 128).
  Compressed entries are concatenated after the token KVs with an additive
  ``block_bias`` mask.
- A 256-expert top-6 MoE on every layer with ``sqrtsoftplus`` scoring, an
  ``e_score_correction_bias`` selection bias, clamped SwiGLU experts
  (``swiglu_limit``), a dense shared expert, and hash routing
  (``mlp_layer_types == "hash_moe"``) on the first layers where expert indices
  come from a frozen ``tid2eid[input_ids]`` table.

This EasyDeL port implements the numerically-faithful *stateless* forward
(training / prefill, ``use_cache=False`` semantics). Decode-time caching of the
compressor branches is not implemented yet — see
``.claude/projects/port_hyv3_dsv4_mistral4.md``.
"""

import typing as tp

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

DEEPSEEK_V4_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)

_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}

DEEPSEEK_V4_MLP_LAYER_TYPES = ("hash_moe", "moe")


@register_config("deepseek_v4")
class DeepseekV4Config(EasyDeLBaseConfig):
    """Configuration class for DeepSeek-V4 (HF ``model_type="deepseek_v4"``).

    Args:
        vocab_size: Token vocabulary size.
        hidden_size: Residual-stream dimension (per hyper-connection stream).
        moe_intermediate_size: Inner width of each routed expert and of the
            shared expert.
        num_hidden_layers: Decoder block count.
        num_attention_heads: Query head count.
        num_key_value_heads: Always ``1`` — V4 uses shared-KV MQA where the
            single ``kv_proj`` output is read as both key and value.
        head_dim: Per-head dimension (512 for V4-Flash/Pro).
        q_lora_rank: Rank of the query LoRA bottleneck (``q_a_proj`` output).
        num_experts_per_tok: Routed experts selected per token.
        n_routed_experts: Total routed experts per layer.
        n_shared_experts: Shared expert count (informational; the shared MLP
            width is ``moe_intermediate_size``).
        scoring_func: Router activation; V4 uses ``"sqrtsoftplus"``
            (``sqrt(softplus(logits))``).
        norm_topk_prob: Whether top-k routing weights are re-normalized.
        routed_scaling_factor: Multiplier on normalized routing weights.
        max_position_embeddings: Context-length bound.
        rope_theta: RoPE base for the "main" rotary (sliding layers).
        layer_types: Per-layer attention schedule drawn from
            ``DEEPSEEK_V4_LAYER_TYPES``. Defaults to 2x HCA bootstrap followed
            by HCA/CSA interleave (HF V4 default).
        compress_rates: Per-layer-type compression rate mapping. Defaults to
            ``{"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}``.
        compress_rope_theta: RoPE base for the "compress" rotary (CSA/HCA
            layers and their compressors).
        hc_mult: Number of parallel hyper-connection residual streams.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations projecting the mHC
            combine matrix onto doubly-stochastic matrices.
        hc_eps: Numerical floor used throughout the mHC math.
        mlp_layer_types: Per-layer MoE schedule from
            ``DEEPSEEK_V4_MLP_LAYER_TYPES``; defaults to 3x ``hash_moe`` then
            ``moe``.
        swiglu_limit: Clamp for expert/shared-MLP gate (max) and up (+-) paths.
        sliding_window: Sliding window applied to the token-KV part on all
            layers.
        o_groups: Head-group count of the grouped output projection.
        o_lora_rank: Per-group intermediate width of the grouped output
            projection.
        index_n_heads: Lightning Indexer query head count.
        index_head_dim: Lightning Indexer head dimension.
        index_topk: Compressed entries kept per query by the indexer.
        num_nextn_predict_layers: MTP layer count in upstream checkpoints
            (never instantiated here; ``mtp.*`` keys are ignored).
        output_router_logits: Whether forward returns router logits.
        router_aux_loss_coef: Load-balancing aux-loss coefficient (unused by
            the reference forward; kept for config parity).
        router_jitter_noise: Router jitter (unused; config parity).
        hidden_act: Activation in expert MLPs.
        initializer_range: Stddev for weight init.
        rms_norm_eps: Epsilon for every RMSNorm (weighted and unweighted).
        use_cache: HF-config parity flag; caching is not implemented in this
            port and cache initialization raises ``NotImplementedError``.
        pad_token_id: Padding token id.
        bos_token_id: BOS token id.
        eos_token_id: EOS token id.
        tie_word_embeddings: Tie input embeddings with the LM head.
        rope_parameters: Optional nested ``{"main": {...}, "compress": {...}}``
            rope parameter mapping (or a flat yarn-style dict that is folded
            into the "compress" entry, mirroring HF ``__post_init__``).
        partial_rotary_factor: Fraction of ``head_dim`` receiving RoPE.
            Defaults to ``64 / 512`` (i.e. ``qk_rope_head_dim = 64``).
        attention_bias: Bias flag on attention projections (V4: ``False``).
        mlp_bias: Bias flag on MLP projections (V4: ``False``).
        attention_dropout: Attention dropout probability.
        compress_ratios: Legacy per-layer int list (0/4/128) folded into
            ``layer_types``.
        compress_rate_csa: Legacy scalar folded into ``compress_rates``.
        compress_rate_hca: Legacy scalar folded into ``compress_rates``.
        num_hash_layers: Legacy hash-MoE layer count folded into
            ``mlp_layer_types``.
        qk_rope_head_dim: Legacy rope width folded into
            ``partial_rotary_factor``.
        **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference: tp.ClassVar = ["past_key_values"]
    attribute_map: tp.ClassVar = {
        "num_local_experts": "n_routed_experts",
        # HF's DeepseekV4Config maps `intermediate_size` to the MoE width (V4 ships no
        # dense-MLP width); HF modeling code reads `config.intermediate_size` through it.
        "intermediate_size": "moe_intermediate_size",
    }

    _default_compress_rates: tp.ClassVar = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    _default_num_hash_layers: tp.ClassVar[int] = 3
    _default_partial_rotary_factor: tp.ClassVar[float] = 64 / 512

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1024,
        num_experts_per_tok: int = 6,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        scoring_func: str = "sqrtsoftplus",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        layer_types: list[str] | None = None,
        compress_rates: dict | None = None,
        compress_rope_theta: float = 160000.0,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1.0e-6,
        mlp_layer_types: list[str] | None = None,
        swiglu_limit: float = 10.0,
        sliding_window: int = 128,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        num_nextn_predict_layers: int = 1,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1.0e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        partial_rotary_factor: float | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        attention_dropout: float = 0.0,
        compress_ratios: list[int] | None = None,
        compress_rate_csa: int | None = None,
        compress_rate_hca: int | None = None,
        num_hash_layers: int | None = None,
        qk_rope_head_dim: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.swiglu_limit = swiglu_limit
        self.sliding_window = sliding_window
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout

        n = num_hidden_layers

        # `compress_rates`: dict keyed by layer type; legacy scalar overrides fold in.
        if compress_rates is None:
            compress_rates = dict(self._default_compress_rates)
        else:
            compress_rates = dict(compress_rates)
        if compress_rate_csa is not None:
            compress_rates["compressed_sparse_attention"] = compress_rate_csa
        if compress_rate_hca is not None:
            compress_rates["heavily_compressed_attention"] = compress_rate_hca
        self.compress_rates = compress_rates

        # `layer_types`: explicit > legacy per-layer `compress_ratios` (0/4/128) >
        # V4 default (2x HCA bootstrap + HCA/CSA interleave).
        if layer_types is None and compress_ratios is not None:
            layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in compress_ratios]
        if layer_types is None:
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention" for i in range(max(n - 2, 0))
            ]
            layer_types = ["heavily_compressed_attention"] * min(n, 2) + interleave
        layer_types = list(layer_types[:n])

        # `mlp_layer_types`: first `num_hash_layers` are hash_moe, rest moe.
        if mlp_layer_types is None:
            n_hash = num_hash_layers if num_hash_layers is not None else self._default_num_hash_layers
            mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
        mlp_layer_types = list(mlp_layer_types[:n])

        # `partial_rotary_factor` = legacy `qk_rope_head_dim / head_dim` when given.
        if partial_rotary_factor is None:
            partial_rotary_factor = (
                qk_rope_head_dim / head_dim if qk_rope_head_dim is not None else self._default_partial_rotary_factor
            )
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_rope_head_dim = int(head_dim * partial_rotary_factor)

        # V4 keys `rope_parameters` by rope-type label ("main" / "compress"), NOT by
        # `layer_types`. Fold a flat yarn-style dict into the "compress" entry the
        # same way HF `__post_init__` does. YaRN compress layers force
        # `attention_factor=1.0` (the reference never multiplies cos/sin by yarn's
        # derived mscale).
        rp = dict(rope_parameters or {})
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            rope_parameters = {"main": dict(rp["main"]), "compress": dict(rp["compress"])}
        else:
            yarn = {k: v for k, v in rp.items() if k not in ("main", "compress")}
            main = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress = {
                **yarn,
                "rope_theta": compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress.setdefault("rope_type", compress.get("type", "default"))
            if compress["rope_type"] == "yarn":
                compress.setdefault("attention_factor", 1.0)
            rope_parameters = {"main": main, "compress": compress}
        # Keep a private, normalization-proof copy: EasyDeLBaseConfig.__setattr__
        # rope backfilling treats non-layer-type-keyed dicts as flat and may add
        # top-level keys to `rope_parameters`; the modeling code reads through
        # `v4_rope_params()` which prefers this private copy.
        self._v4_rope_parameters = {k: dict(v) for k, v in rope_parameters.items()}
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # Assigned after ``super().__init__``: transformers 5.x collects the
        # BASE class's strict-dataclass validators at decoration time, so the
        # generic ``validate_layer_type`` (which rejects V4's layer-type
        # strings) would run against these values inside ``super().__init__``.
        # The V4-specific override below is invoked manually instead.
        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.validate_layer_type()

    def validate_layer_type(self):
        """Narrow layer-type validation to DeepSeek-V4's own attention/MLP types.

        The base transformers validator only knows the generic
        ``ALLOWED_LAYER_TYPES``; V4 ships ``compressed_sparse_attention`` /
        ``heavily_compressed_attention`` (plus ``hash_moe``/``moe`` for the
        MLP schedule), so this override mirrors HF ``DeepseekV4Config``.

        Raises:
            ValueError: On a length mismatch or an unknown layer type.
        """
        num_hidden_layers = getattr(self, "num_hidden_layers", None)
        if num_hidden_layers is None:
            return
        for name, types, allowed in (
            ("layer_types", getattr(self, "layer_types", None), DEEPSEEK_V4_LAYER_TYPES),
            ("mlp_layer_types", getattr(self, "mlp_layer_types", None), DEEPSEEK_V4_MLP_LAYER_TYPES),
        ):
            if types is None:
                continue
            if len(types) != num_hidden_layers:
                raise ValueError(f"`num_hidden_layers` ({num_hidden_layers}) must equal `len({name})` ({len(types)}).")
            bad = [t for t in types if t not in allowed]
            if bad:
                raise ValueError(f"`{name}` entries must be one of {allowed} for DeepSeek-V4; got {bad}.")

    def get_mask_details(self) -> dict:
        """Report every layer as sliding-window attention for engine accounting.

        DeepSeek-V4's token-KV state is a ``sliding_window`` ring on ALL
        layers (CSA/HCA layers add fixed-shape compressor state that lives in
        the :class:`~easydel.caching.CompressedWindowCache` slots, not in KV
        pages), and the generic ``AttnMaskType.from_hf`` mapping does not
        know V4's compressed layer-type strings. Serving engines therefore
        see a uniform sliding-window mask profile, which keeps their
        page-budget accounting bounded; the model itself never reads pages.

        Returns:
            dict: Layer index -> :class:`~easydel.infra.utils.AttnMaskDetail`
            with ``SLIDING`` masks of size ``sliding_window``.
        """
        from easydel.infra.utils import AttnMaskDetail, AttnMaskType

        return {
            layer_idx: AttnMaskDetail(
                mask_type=AttnMaskType.SLIDING,
                size=self.sliding_window,
                chunks=None,
            )
            for layer_idx in range(self.num_hidden_layers)
        }

    def v4_rope_params(self, label: str) -> dict:
        """Return the rope-parameter sub-dict for a rope-type label.

        Args:
            label: Either ``"main"`` (sliding layers) or ``"compress"``
                (CSA/HCA layers and compressors).

        Returns:
            dict: Rope parameters containing at least ``rope_type``,
            ``rope_theta`` and ``partial_rotary_factor``.

        Raises:
            KeyError: If ``label`` cannot be resolved.
        """
        private = getattr(self, "_v4_rope_parameters", None)
        if isinstance(private, dict) and isinstance(private.get(label), dict):
            return private[label]
        rp = getattr(self, "rope_parameters", None) or {}
        sub = rp.get(label)
        if isinstance(sub, dict):
            return sub
        raise KeyError(f"DeepSeek-V4 rope parameters for label {label!r} not found.")


__all__ = ["DeepseekV4Config"]
