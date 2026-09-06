# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Configuration classes for Qwen4-Exp (Qwen3.8-Flash-Next).

Three classes mirror the checkpoint layout:

- :class:`Qwen4ExpTextConfig` (``qwen4_exp_text``): the hybrid
  GatedDeltaNet/QSA MoE language model with hyper-connections, PLE n-gram
  embeddings and an optional MTP head.
- :class:`Qwen4ExpVisionConfig` (``qwen4_exp_vision``): the Qwen3-VL-style
  vision tower.
- :class:`Qwen4ExpConfig` (``qwen4_exp``): the multimodal composition.

Field names follow the released ``config.json`` verbatim so
``from_pretrained`` binds without a renaming pass.
"""

import typing
from collections.abc import Mapping

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

logger = get_logger(__name__)


@register_config("qwen4_exp_vision")
class Qwen4ExpVisionConfig(EasyDeLBaseConfig):
    """Configuration for the Qwen4-Exp vision encoder.

    Structurally the Qwen3-VL vision tower: learned (interpolated) position
    embeddings, per-image block-diagonal attention, and a PatchMerger
    projecting to the text hidden size.

    Args:
        depth: Number of vision transformer blocks. Defaults to 27.
        hidden_size: Vision hidden width. Defaults to 1152.
        hidden_act: MLP activation. Defaults to "gelu_pytorch_tanh".
        intermediate_size: Vision MLP inner width. Defaults to 4304.
        num_heads: Attention heads. Defaults to 16.
        in_channels: Image channels. Defaults to 3.
        patch_size: Patch size. Defaults to 16.
        spatial_merge_size: Merger downsampling. Defaults to 2.
        temporal_patch_size: Video temporal patch. Defaults to 2.
        out_hidden_size: Projected output width (text hidden). Defaults to 2560.
        num_position_embeddings: Learned position grid (48 x 48). Defaults to 2304.
        deepstack_visual_indexes: Deepstack layers; empty for Qwen4-Exp.
        initializer_range: Init std. Defaults to 0.02.
    """

    model_type = "qwen4_exp_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 2560,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize the vision config; ``**kwargs`` go to EasyDeLBaseConfig."""
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = [] if deepstack_visual_indexes is None else deepstack_visual_indexes
        self.initializer_range = initializer_range

        self.embed_dim = hidden_size


@register_config("qwen4_exp_text")
class Qwen4ExpTextConfig(EasyDeLBaseConfig):
    """Configuration for the Qwen4-Exp language model.

    Architecture (from the released Qwen3.8-Flash-Next config):

    - 48 layers of ``3 x linear_attention (GatedDeltaNet) -> 1 x
      qwen_sparse_attention`` (``full_attention_interval: 4``); the released
      checkpoint spells the sparse layers ``full_attention`` and they are
      remapped here, matching the reference config.
    - QSA full attention: 24 Q / 2 KV heads of ``head_dim`` 256, partial RoPE
      (``partial_rotary_factor`` 0.25, interleaved mRoPE sections), sigmoid
      output gate fused into ``q_proj``, and a block top-k indexer
      (``indexer_*``).
    - MoE on every layer: 512 experts, 10 active + 1 gated shared expert,
      softmax routing with ``norm_topk_prob``.
    - Hyper-connections: ``hc_count`` residual streams with low-rank gated
      mixing (``hc_lowrank``); the model-level ``hyper_connection_mixer`` is
      also the final norm/collapse.
    - PLE: hashed n-gram embeddings injected on ``ple_layer_ids`` (1-indexed).

    Args:
        vocab_size: Vocabulary size. Defaults to 248320.
        hidden_size: Hidden width. Defaults to 2560.
        num_hidden_layers: Decoder layers. Defaults to 48.
        num_attention_heads: Full-attention query heads. Defaults to 24.
        num_key_value_heads: Full-attention KV heads. Defaults to 2.
        head_dim: Full-attention head width. Defaults to 256.
        full_attention_interval: Every Nth layer is sparse-full attention.
        linear_*: GatedDeltaNet dims (key/value heads, head dims, conv kernel).
        indexer_*: QSA indexer dims (None disables QSA).
        hc_count / hc_lowrank: Hyper-connection stream count / mixer rank.
        ple_*: PLE layer ids (1-indexed), embed width, conv kernel.
        ngram_*: Hashed n-gram table sizing (size, heads, vocab base, divisor,
            shard parts).
        mtp / mtp_num_hidden_layers: MTP head sub-config and depth.
        norm_topk_prob: Renormalize router top-k weights. Defaults to True.
        output_gate_type: GDN output gate activation ("sigmoid" or "silu").
    """

    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2560,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000000.0,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        full_attention_interval: int = 4,
        layer_types: list[str] | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        mamba_ssm_dtype: str = "float32",
        indexer_n_heads: int | None = 4,
        indexer_kv_heads: int | None = 1,
        indexer_head_dim: int | None = 128,
        indexer_budget: int | None = 2048,
        indexer_compress_ratio: int | None = 4,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        ple_layer_ids: list[int] | None = None,
        ple_embed_dim: int | None = None,
        ple_conv_kernel_size: int = 4,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_ngram_vocab_size_divisible_by: int = 128,
        seed: int = 1234,
        split_ngram_parts: int = 128,
        ngram_table_dtype: str | None = None,
        ngram_sharding_axis: str = "tp",
        moe_intermediate_size: int = 640,
        shared_expert_intermediate_size: int = 640,
        num_experts_per_tok: int = 10,
        num_experts: int = 512,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        output_gate_type: str | None = "sigmoid",
        mtp: dict | None = None,
        mtp_num_hidden_layers: int = 0,
        mtp_use_dedicated_embeddings: bool = False,
        mtp_loss_coef: float = 0.3,
        intermediate_size: int | None = None,
        # GDN runtime knobs consumed by the shared Qwen3NextLinearAttention;
        # Qwen4-Exp always stores split projections.
        linear_attention_separate_proj: bool = True,
        linear_attention_merged_split_proj: bool = False,
        use_grouped_gdr_prefill: bool = True,
        use_ragged_gdr: bool = True,
        ragged_gdr_chunk_size: int = 16,
        force_recurrent_scan_prefill: bool = False,
        recurrent_scan_prefill_max_seq_len: int = 64,
        force_dp_recurrent_scan_prefill: bool = False,
        dp_recurrent_scan_prefill_max_seq_len: int = 64,
        attn_output_gate: bool = True,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bos_token_id: int | None = 248044,
        eos_token_id: int | list[int] | None = 248044,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the text config; ``**kwargs`` go to EasyDeLBaseConfig."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size

        # RoPE: partial, interleaved mRoPE. ``rope_parameters`` (HF v5) and
        # ``rope_scaling`` (legacy) are the same dict; EasyDeLBaseConfig keeps
        # them in sync. Defaults reproduce the released checkpoint.
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            }
        self.partial_rotary_factor = float(rope_parameters.get("partial_rotary_factor", 1.0))
        # The rotary modules and the frequencies cache read the top-level
        # ``rope_theta`` field, so reconcile it with the authoritative
        # ``rope_parameters["rope_theta"]`` here (not only in get_text_config)
        # or every table is built with the wrong base.
        self.rope_theta = float(rope_parameters.get("rope_theta", rope_theta))

        if full_attention_interval <= 0:
            raise ValueError(f"full_attention_interval must be positive, got {full_attention_interval}.")
        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            layer_types = [
                "linear_attention" if (i + 1) % full_attention_interval else "qwen_sparse_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            # The released checkpoint spells sparse layers "full_attention".
            layer_types = ["qwen_sparse_attention" if lt == "full_attention" else lt for lt in layer_types]
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length must equal num_hidden_layers ({num_hidden_layers}), got {len(layer_types)}."
            )
        bad = sorted(set(layer_types) - {"linear_attention", "qwen_sparse_attention"})
        if bad:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {bad}.")
        # Assigned after ``super().__init__``: transformers 5.x collects the
        # BASE class's strict-dataclass validators at decoration time, so the
        # generic ``validate_layer_type`` (which does not know
        # "qwen_sparse_attention") would run inside ``super().__init__``.
        # ``self.layer_types`` is set and validated manually below instead.

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.mamba_ssm_dtype = mamba_ssm_dtype
        if linear_attention_separate_proj and linear_attention_merged_split_proj:
            raise ValueError("linear_attention_separate_proj and linear_attention_merged_split_proj are exclusive")
        self.linear_attention_separate_proj = linear_attention_separate_proj
        self.linear_attention_merged_split_proj = linear_attention_merged_split_proj
        self.use_grouped_gdr_prefill = bool(use_grouped_gdr_prefill)
        self.use_ragged_gdr = use_ragged_gdr
        self.ragged_gdr_chunk_size = ragged_gdr_chunk_size
        self.force_recurrent_scan_prefill = force_recurrent_scan_prefill
        self.recurrent_scan_prefill_max_seq_len = recurrent_scan_prefill_max_seq_len
        self.force_dp_recurrent_scan_prefill = force_dp_recurrent_scan_prefill
        self.dp_recurrent_scan_prefill_max_seq_len = dp_recurrent_scan_prefill_max_seq_len
        self.attn_output_gate = bool(attn_output_gate)
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size

        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio

        if hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {hc_count}.")
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank

        self.ple_layer_ids = sorted(set(ple_layer_ids)) if ple_layer_ids else []
        self.ple_embed_dim = self.hidden_size if ple_embed_dim is None else ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.seed = seed
        self.split_ngram_parts = split_ngram_parts
        self.ngram_table_dtype = ngram_table_dtype
        if ngram_sharding_axis not in {"dp", "fsdp", "ep", "tp", "sp"}:
            raise ValueError(f"ngram_sharding_axis must be one of dp/fsdp/ep/tp/sp, got {ngram_sharding_axis!r}.")
        self.ngram_sharding_axis = ngram_sharding_axis
        # GDN conv, PLE conv, and the PLE n-gram context are separate states.
        self.number_of_conv_states = 3 if self.ple_layer_ids else 1

        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        if output_gate_type is not None and output_gate_type not in ("sigmoid", "silu"):
            raise ValueError(f"Unsupported Qwen4-Exp output gate activation: {output_gate_type}.")
        self.output_gate_type = output_gate_type

        if mtp is not None and not isinstance(mtp, dict):
            raise TypeError(f"mtp must be a mapping or None, got {type(mtp).__name__}.")
        self.mtp = None if mtp is None else dict(mtp)
        if self.mtp is not None:
            mtp_num_hidden_layers = int(self.mtp.get("num_hidden_layers", mtp_num_hidden_layers))
            mtp_types = self.mtp.get("layer_types")
            if mtp_types is not None and any(t not in {"qwen_sparse_attention", "full_attention"} for t in mtp_types):
                raise ValueError("Qwen4-Exp MTP currently supports QSA/full_attention layers only.")
            if mtp_types is not None and len(mtp_types) != mtp_num_hidden_layers:
                raise ValueError("mtp.layer_types length must equal mtp.num_hidden_layers.")
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = bool(mtp_use_dedicated_embeddings)
        if mtp_loss_coef < 0:
            raise ValueError(f"mtp_loss_coef must be non-negative, got {mtp_loss_coef}.")
        self.mtp_loss_coef = float(mtp_loss_coef)

        rope_theta = float(rope_parameters.get("rope_theta", rope_theta))
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_parameters=rope_parameters,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        # Validations that read special-token ids run after super().__init__()
        # (which is where eos_token_id/bos_token_id are assigned).
        self._validate_qsa()
        self._validate_ple(layer_types)
        self.layer_types = layer_types
        self.validate_layer_type()

    def _validate_qsa(self) -> None:
        """Port of the reference QSA invariant checks (all-or-nothing fields)."""
        fields = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        values = {name: getattr(self, name) for name in fields}
        if all(v is None for v in values.values()):
            return
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}.")
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}.")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1.")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio.")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > self.indexer_head_dim:
            raise ValueError(
                f"Qwen4-Exp attention RoPE dimensions must fit the QSA index head: rotary_dim={rotary_dim}, "
                f"indexer_head_dim={self.indexer_head_dim}."
            )

    def _validate_ple(self, layer_types: list[str]) -> None:
        """Port of the reference PLE invariant checks (1-indexed linear layers)."""
        if not self.ple_layer_ids:
            return
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if ngram_heads <= 0 or self.ple_embed_dim <= 0 or self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "ple_embed_dim and the total number of n-gram heads must be positive, and ple_embed_dim must be "
                f"divisible by the number of heads: {self.ple_embed_dim} % {ngram_heads} != 0."
            )
        invalid = [i for i in self.ple_layer_ids if i < 1 or i > self.num_hidden_layers]
        if invalid:
            raise ValueError(
                f"ple_layer_ids must contain one-indexed ids in [1, {self.num_hidden_layers}], got {invalid}."
            )
        non_linear = [i for i in self.ple_layer_ids if layer_types[i - 1] != "linear_attention"]
        if non_linear:
            raise ValueError(
                f"Qwen4-Exp PLE is only supported on linear_attention layers, got PLE on layers {non_linear}."
            )
        if self.eos_token_id is None or (isinstance(self.eos_token_id, list) and not self.eos_token_id):
            raise ValueError("eos_token_id must be set when Qwen4-Exp PLE layers are enabled.")

    def validate_layer_type(self) -> None:
        """Qwen4-Exp layer-type vocabulary check.

        Overrides the generic transformers validator (invoked at
        ``super().__init__`` time on the base class) which does not know
        ``qwen_sparse_attention``; see the assignment note in ``__init__``.
        """
        bad = sorted(set(self.layer_types) - {"linear_attention", "qwen_sparse_attention"})
        if bad:
            raise ValueError(
                f"The `layer_types` entries must be in ('linear_attention', 'qwen_sparse_attention') "
                f"for Qwen4-Exp but got {sorted(set(self.layer_types))}"
            )

    @property
    def qsa_enabled(self) -> bool:
        """Whether QSA indexer fields are populated (sparse full attention on)."""
        return self.indexer_n_heads is not None

    @property
    def rotary_dim(self) -> int:
        """Number of leading head channels rotated by RoPE."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_d_inner(self) -> int:
        """GatedDeltaNet conv width: ``key_dim * 2 + value_dim``."""
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        return key_dim * 2 + value_dim

    @property
    def linear_d_state(self) -> int:
        """Per-head recurrent-state dimension of the linear layers."""
        return self.linear_value_head_dim

    @property
    def ple_layer_indices_0based(self) -> dict[int, int]:
        """Map of 0-indexed decoder layer -> PLE layer index (empty without PLE)."""
        return {layer_id - 1: i for i, layer_id in enumerate(self.ple_layer_ids)}

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """True for sparse-full (QSA) attention layers, False for GDN layers."""
        return self.layer_types[layer_idx] == "qwen_sparse_attention"

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Qwen4-Exp uses MoE on every layer."""
        return True


@register_config("qwen4_exp")
class Qwen4ExpConfig(EasyDeLBaseConfig):
    """Top-level Qwen4-Exp multimodal configuration.

    Args:
        vision_config: Vision sub-config (dict or Qwen4ExpVisionConfig).
        text_config: Text sub-config (dict or Qwen4ExpTextConfig).
        image_token_id: Image placeholder token. Defaults to 248056.
        video_token_id: Video placeholder token. Defaults to 248057.
        vision_start_token_id / vision_end_token_id: Vision delimiters.
        language_model_only: Skip the vision tower (text-only serving).
        tie_word_embeddings: Tied embeddings. Defaults to False.
    """

    model_type = "qwen4_exp"
    sub_configs: typing.ClassVar = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vision_config: Mapping[str, typing.Any] | Qwen4ExpVisionConfig | None = None,
        text_config: Mapping[str, typing.Any] | Qwen4ExpTextConfig | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        language_model_only: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the multimodal config; ``**kwargs`` go to EasyDeLBaseConfig."""
        if isinstance(vision_config, dict):
            # The released checkpoint tags the vision sub-config with the
            # parent's model_type; normalize it (the reference does the same).
            if vision_config.get("model_type") == "qwen4_exp":
                vision_config = {**vision_config, "model_type": "qwen4_exp_vision"}
            self.vision_config = self.sub_configs["vision_config"](**self._fix_parent_kws(vision_config, kwargs))
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self._fix_parent_kws(text_config, kwargs))
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.language_model_only = language_model_only
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = False) -> Qwen4ExpTextConfig:
        """Return the text decoder configuration.

        Args:
            decoder: Part of the HF v5 ``get_text_config`` protocol; the text
                config is the decoder, so the flag is accepted and ignored.
        """
        return self.text_config


__all__ = ["Qwen4ExpConfig", "Qwen4ExpTextConfig", "Qwen4ExpVisionConfig"]
