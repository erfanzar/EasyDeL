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

"""DeepSeek-V4.1 configuration (HF ``model_type="deepseek_v41"``, text tower ``deepseek_v41_text``).

DeepSeek-V4.1 keeps V4's shared-KV MQA with sinks, grouped output projection, mHC hyper-connections and
``sqrtsoftplus`` MoE, but changes the attention schedule and adds new components:

- Per-layer integer ``compress_ratios`` (0 = sliding window only, ``r`` = window plus ``r``-to-1 compressed KV),
  listed for the backbone layers followed by the MTP layers.
- Compressed KV is produced only by ``kv_source_layer_ids`` and re-used by every later layer with a compressed
  ratio; indexer top-k selections come from ``index_source_layer_ids``; one ``candidate_source_layer_id`` publishes
  a block pre-filter for later index sources.
- Engram n-gram hash embeddings on ``engram_layer_ids`` whose table sizes are derived from a prime-bucket layout.
- DSpark MTP layers and a vision tower (``vision_config``).

Semantics follow DeepSeek's reference implementation (``inference/model.py`` in the model repository); see
``.xerxes/projects/deepseek-v41/SPEC.md``. transformers does not ship this architecture.
"""

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.layers.engram._hashing import engram_bucket_primes, engram_table_rows

# Released DeepSeek-V4.1-Flash schedule: 40 backbone layers + 3 MTP layers.
_DEFAULT_COMPRESS_RATIOS: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)

_TEXT_DEFAULTS: dict[str, tp.Any] = {
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": True,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "gate_temp": 1.0,
    "output_router_logits": False,
    "router_aux_loss_coef": 0.001,
    "sliding_window": 128,
    "compress_ratios": None,
    "kv_source_layer_ids": (2, 8, 14, 20),
    "index_source_layer_ids": (2, 8, 14, 20, 24, 28, 32, 36),
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": (1, 14),
    "engram_num_embeddings": None,
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16_000_000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "engram_table_memory_kind": "device",
    "engram_table_dtype": "param",
    "quantize_expert_chunk": None,
    "indexer_aux_loss_coef": 0.0,
    "ced_prefill_skip": True,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": (37, 38, 39),
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "image_token_id": 129264,
    # Read by the reused DeepSeek-V4 MoE / expert / hyper-connection modules.
    "mlp_bias": False,
    "hc_use_fused_coefficients": False,
    "mlp_layer_types": None,
}

#: Defaults of the nested ``vision_config`` (released ``config.json``; ``rms_norm_eps`` is hard-coded in ``vision.py``).
#: ``num_hidden_layers=0`` keeps vision disabled unless the layer count is given explicitly.
_VISION_DEFAULTS: dict[str, tp.Any] = {
    "num_hidden_layers": 0,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000.0,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936,
    "max_wh_ratio": None,
    "rms_norm_eps": 1e-6,
}

_LIST_FIELDS = (
    "compress_ratios",
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "engram_layer_ids",
    "engram_num_embeddings",
    "dspark_target_layer_ids",
)


@register_config("deepseek_v41")
class DeepseekV41Config(EasyDeLBaseConfig):
    """Configuration for DeepSeek-V4.1 (text tower; composite HF checkpoints are absorbed).

    All text fields keep the Hugging Face ``config.json`` names (see ``_TEXT_DEFAULTS`` for the released
    defaults). A composite checkpoint config (``model_type="deepseek_v41"`` with nested ``text_config`` /
    ``vision_config``) is flattened: precedence is explicit flat kwargs > nested ``text_config`` > defaults.
    ``vision_config`` is kept as a plain dict for the vision wrapper; the raw HF ``quantization_config`` (fp8
    32x32 blocks, fp4 experts) is preserved verbatim for load-time decoding.

    Cross-layer ownership (reference ``Attention.__init__`` / ``SharedAttentionRuntime``):

    - a layer with ratio > 0 attends to the compressed KV written by the most recent KV source at or before it,
      and that source must have the same ratio;
    - only index sources run the indexer; other compressed layers re-use the most recent top-k selection;
    - an index source owns indexer keys only when it is also a KV source, otherwise it reads its KV source's keys;
    - index sources after ``candidate_source_layer_id`` restrict their scores to its published block candidates.

    Optional training / memory features (defaults preserve the reference behaviour):

    - ``engram_table_memory_kind``: ``"device"`` (default) keeps the engram embedding tables in accelerator memory;
      ``"pinned_host"`` stores them in pinned host memory (non-trainable ``buffers`` leaves, same names) and gathers
      only the looked-up rows to the device.
    - ``engram_table_dtype``: ``"param"`` (default) stores each engram table as one ``param_dtype`` parameter;
      ``"fp8"`` keeps the released layout (float8_e4m3fn ``embed.weight`` + uint8 e8m0 ``embed.scale`` buffers,
      row-sharded over all devices, dequantized per looked-up row). Use it with ``engram_table_memory_kind="pinned_host"``
      and fill the tables with ``easydel.layers.engram.load_engram_tables``.
    - ``quantize_expert_chunk``: ``None`` (default) quantizes each stacked expert kernel in one step at load; an int
      quantizes that many experts at a time so the fp32 temporaries stay bounded (result is identical).
    - ``indexer_aux_loss_coef``: ``0.0`` (default) leaves the indexer without a training signal (hard top-k). A
      positive value adds ``coef * sum_layers KL(stop_grad(dense head-summed attention over compressed entries) ||
      softmax(indexer scores))`` to the loss (DSA indexer warm-up objective), updating only the indexer parameters.
    - ``ced_prefill_skip``: ``True`` (default) lets eSurge prefill skip the window-only decoder layers after the last
      KV source on steps whose tokens all lie outside their receptive field (exact; see
      ``modeling_deepseek_v41._ced_decoder_tail``). ``False`` always runs every layer.
    """

    model_type: str = "deepseek_v41"
    keys_to_ignore_at_inference: tp.ClassVar = ["past_key_values"]
    attribute_map: tp.ClassVar = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }
    # ``rope_parameters`` is keyed by rope-type label, not by ``layer_types``. Transformers >= 5.17 only treats the
    # dict as nested when its keys are in ``_rope_type_labels`` (``RotaryEmbeddingConfigMixin.nested_rope_parameter_keys``);
    # without this it validates ``{"main", "compress"}`` as one flat dict and warns about both keys (same as HF V4).
    _rope_type_labels: tp.ClassVar[tuple[str, ...]] = ("main", "compress")

    def validate_rope(self):
        """Validate the ``main`` and ``compress`` rope sub-dicts separately (mirrors HF ``DeepseekV4Config``).

        The yarn/llama3/longrope validators read ``self.rope_parameters[...]`` directly, so ``rope_parameters`` points
        at the sub-dict under validation and is restored afterwards.
        """
        rope_parameters_dict = getattr(self, "rope_parameters", None) or {}
        ignore_keys = getattr(self, "ignore_keys_at_rope_validation", None)
        for label in self._rope_type_labels:
            rope_parameters = rope_parameters_dict.get(label)
            if not isinstance(rope_parameters, dict):
                continue
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            rope_parameters["rope_type"] = rope_type
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            if validation_fn is None:
                continue
            self.rope_parameters = rope_parameters
            try:
                validation_fn(rope_parameters, ignore_keys=ignore_keys)
            finally:
                self.rope_parameters = rope_parameters_dict

    def __init__(self, **kwargs):
        nested = kwargs.pop("text_config", None)
        nested = {k: v for k, v in (nested or {}).items() if k not in ("model_type", "architectures")}
        merged: dict[str, tp.Any] = {**_TEXT_DEFAULTS, **nested, **kwargs}
        vision_config = merged.pop("vision_config", None)
        hf_quantization_config = merged.pop("quantization_config", None)

        for name in _TEXT_DEFAULTS:
            value = merged.pop(name)
            if name in _LIST_FIELDS and value is not None:
                value = [int(v) for v in value]
            setattr(self, name, value)
        if self.compress_ratios is None:
            self.compress_ratios = list(_DEFAULT_COMPRESS_RATIOS)
        if len(self.compress_ratios) == self.num_hidden_layers:
            # MTP layers are window-only; allow configs that list only the backbone.
            self.compress_ratios = self.compress_ratios + [0] * self.num_nextn_predict_layers

        # V4.1 has no hash-routed MoE: every backbone and MTP layer uses learned top-k routing. The reused V4 MoE block
        # selects its router from this list.
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["moe"] * (self.num_hidden_layers + self.num_nextn_predict_layers)
        else:
            self.mlp_layer_types = [str(t) for t in self.mlp_layer_types]

        self.partial_rotary_factor = self.qk_rope_head_dim / self.head_dim
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self._v4_rope_parameters = self._build_rope_parameters(
            merged.pop("rope_parameters", None), merged.pop("rope_scaling", None)
        )
        self.rope_parameters = {k: dict(v) for k, v in self._v4_rope_parameters.items()}
        self.validate_schedule()

        merged.setdefault("bos_token_id", 0)
        merged.setdefault("eos_token_id", 1)
        merged.setdefault("tie_word_embeddings", False)
        super().__init__(**merged)
        self.vision_config = dict(vision_config) if isinstance(vision_config, dict) else vision_config
        if hf_quantization_config is not None:
            self.quantization_config = hf_quantization_config

    def _build_rope_parameters(self, rope_parameters: dict | None, rope_scaling: dict | None) -> dict:
        """Split RoPE settings into the ``main`` (window-only layers) and ``compress`` (ratio > 0) entries.

        The flat HF ``rope_scaling`` yarn dict applies to compressed layers only; window-only layers use plain
        RoPE at ``rope_theta``. Yarn here never rescales cos/sin (``attention_factor=1``), as in the reference.
        """
        rp = dict(rope_parameters or rope_scaling or {})
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            return {"main": dict(rp["main"]), "compress": dict(rp["compress"])}
        yarn = {k: v for k, v in rp.items() if k not in ("main", "compress", "rope_theta")}
        legacy_type = yarn.pop("type", None)
        yarn.setdefault("rope_type", legacy_type or "default")
        main = {"rope_type": "default", "rope_theta": float(self.rope_theta)}
        compress = {**yarn, "rope_theta": float(self.compress_rope_theta)}
        if compress["rope_type"] == "yarn":
            compress.setdefault("attention_factor", 1.0)
        for entry in (main, compress):
            entry["partial_rotary_factor"] = self.partial_rotary_factor
        return {"main": main, "compress": compress}

    def validate_schedule(self) -> None:
        """Check the compression / sharing schedule against the reference ownership rules.

        Raises:
            ValueError: On any inconsistent length, out-of-range layer id, or ownership violation.
        """
        n, mtp = self.num_hidden_layers, self.num_nextn_predict_layers
        ratios = self.compress_ratios
        if len(ratios) != n + mtp:
            raise ValueError(
                f"compress_ratios must list num_hidden_layers ({n}) or num_hidden_layers + num_nextn_predict_layers "
                f"({n + mtp}) entries; got {len(ratios)}."
            )
        if any(r < 0 for r in ratios):
            raise ValueError(f"compress_ratios must be non-negative; got {ratios}.")
        if any(r != 0 for r in ratios[n:]):
            raise ValueError("MTP (DSpark) layers are window-only: their compress_ratios must be 0.")
        for name in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids", "dspark_target_layer_ids"):
            bad = [i for i in getattr(self, name) if not 0 <= i < n]
            if bad:
                raise ValueError(f"{name} must index backbone layers [0, {n}); got {bad}.")
        for src in self.kv_source_layer_ids:
            if ratios[src] == 0:
                raise ValueError(f"KV source layer {src} has compress ratio 0 and cannot produce compressed KV.")
        for layer in range(n):
            if ratios[layer] == 0:
                continue
            src = self.kv_source_layer(layer)
            if src is None:
                raise ValueError(f"Layer {layer} has compress ratio {ratios[layer]} but no KV source at or before it.")
            if ratios[src] != ratios[layer]:
                raise ValueError(
                    f"Layer {layer} (ratio {ratios[layer]}) reads compressed KV from layer {src} "
                    f"(ratio {ratios[src]}); consumer and source ratios must match."
                )
            topk_src = self.topk_source_layer(layer)
            if topk_src is None:
                raise ValueError(f"Compressed layer {layer} has no index source at or before it.")
            if ratios[topk_src] != ratios[layer]:
                raise ValueError(
                    f"Layer {layer} (ratio {ratios[layer]}) reuses the top-k selection of layer {topk_src} "
                    f"(ratio {ratios[topk_src]}); they index different compressed caches."
                )
        for src in self.index_source_layer_ids:
            if ratios[src] == 0:
                raise ValueError(f"Index source layer {src} has compress ratio 0.")
            key_src = self.kv_source_layer(src)
            if key_src not in self.index_source_layer_ids:
                raise ValueError(
                    f"Index source layer {src} scores against the indexer keys of KV source {key_src}, which must "
                    "itself be an index source (only an index source that compresses its own KV builds keys)."
                )
        cand = self.candidate_source_layer_id
        if cand is not None and cand >= 0 and cand not in self.index_source_layer_ids:
            raise ValueError(f"candidate_source_layer_id {cand} must be one of index_source_layer_ids.")
        for src in self.index_source_layer_ids:
            if self.uses_candidates(src) and ratios[src] != ratios[cand]:
                raise ValueError(
                    f"Index source {src} (ratio {ratios[src]}) masks with candidates from layer {cand} "
                    f"(ratio {ratios[cand]}); they index different compressed caches."
                )
        if self.engram_num_embeddings is not None and len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings must have one entry per engram layer.")
        if self.engram_table_memory_kind not in ("device", "pinned_host"):
            raise ValueError(
                f"engram_table_memory_kind must be 'device' or 'pinned_host'; got {self.engram_table_memory_kind!r}."
            )
        if self.engram_table_dtype not in ("param", "fp8"):
            raise ValueError(f"engram_table_dtype must be 'param' or 'fp8'; got {self.engram_table_dtype!r}.")
        if self.quantize_expert_chunk is not None and int(self.quantize_expert_chunk) < 1:
            raise ValueError(f"quantize_expert_chunk must be None or >= 1; got {self.quantize_expert_chunk}.")
        if float(self.indexer_aux_loss_coef) < 0.0:
            raise ValueError(f"indexer_aux_loss_coef must be >= 0; got {self.indexer_aux_loss_coef}.")
        if len(self.mlp_layer_types) != n + mtp or any(t != "moe" for t in self.mlp_layer_types):
            raise ValueError(
                f"DeepSeek-V4.1 uses learned top-k routing on all {n + mtp} layers (no hash MoE): mlp_layer_types "
                f"must be {n + mtp} x 'moe'; got {self.mlp_layer_types}."
            )

    def layer_compress_ratio(self, layer_idx: int) -> int:
        """Compression ratio of a backbone (or MTP) layer; 0 means sliding window only."""
        return int(self.compress_ratios[layer_idx])

    def rope_label(self, layer_idx: int) -> str:
        """``"compress"`` for compressed layers (yarn, ``compress_rope_theta``), ``"main"`` otherwise."""
        return "compress" if self.layer_compress_ratio(layer_idx) > 0 else "main"

    def is_kv_source(self, layer_idx: int) -> bool:
        """Whether the layer runs a compressor and writes the shared compressed KV."""
        return layer_idx < self.num_hidden_layers and layer_idx in self.kv_source_layer_ids

    def is_index_source(self, layer_idx: int) -> bool:
        """Whether the layer runs the indexer and publishes a new top-k selection."""
        return layer_idx < self.num_hidden_layers and layer_idx in self.index_source_layer_ids

    def kv_source_layer(self, layer_idx: int) -> int | None:
        """Most recent KV source at or before ``layer_idx`` for a compressed layer, else ``None``."""
        if layer_idx >= self.num_hidden_layers or self.layer_compress_ratio(layer_idx) == 0:
            return None
        earlier = [s for s in self.kv_source_layer_ids if s <= layer_idx]
        return max(earlier) if earlier else None

    def topk_source_layer(self, layer_idx: int) -> int | None:
        """Most recent index source at or before ``layer_idx`` for a compressed layer, else ``None``."""
        if layer_idx >= self.num_hidden_layers or self.layer_compress_ratio(layer_idx) == 0:
            return None
        earlier = [s for s in self.index_source_layer_ids if s <= layer_idx]
        return max(earlier) if earlier else None

    def index_key_source_layer(self, layer_idx: int) -> int | None:
        """Layer owning the indexer keys an index source scores against (itself when it is a KV source)."""
        if not self.is_index_source(layer_idx):
            return None
        return self.kv_source_layer(layer_idx)

    def owns_index_keys(self, layer_idx: int) -> bool:
        """Whether the layer carries indexer ``wk``/``k_norm`` parameters."""
        return self.is_index_source(layer_idx) and self.is_kv_source(layer_idx)

    def uses_candidates(self, layer_idx: int) -> bool:
        """Whether an index source restricts its scores to the candidate source's published blocks."""
        cand = self.candidate_source_layer_id
        return self.is_index_source(layer_idx) and cand is not None and 0 <= cand < layer_idx

    def engram_primes(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Prime bucket moduli ``[engram layer][ngram size - 2][head]``."""
        return engram_bucket_primes(
            len(self.engram_layer_ids), self.engram_max_ngram_size, self.engram_n_heads, self.engram_vocab_size
        )

    def engram_table_rows(self) -> list[int]:
        """Rows of each engram table: the sum of its prime buckets.

        Raises:
            ValueError: If ``engram_num_embeddings`` is set and disagrees with the derived layout (the hash
                offsets would then address the wrong rows).
        """
        return engram_table_rows(self)

    @property
    def vision_enabled(self) -> bool:
        """Whether the config carries a vision tower (``vision_config`` with ``num_hidden_layers > 0``).

        The layer count must be given explicitly: a ``vision_config`` without ``num_hidden_layers`` (or an empty
        one) leaves vision disabled, mirroring the reference ``ModelArgs.vision_enabled`` (``vision_n_layers > 0``).
        """
        vision_config = getattr(self, "vision_config", None)
        return isinstance(vision_config, dict) and int(vision_config.get("num_hidden_layers") or 0) > 0

    def vision_params(self) -> dict[str, tp.Any]:
        """Vision tower / aligner settings with the released ``vision_config`` values as defaults.

        Only normalizes (fills defaults, coerces types); structural validation (head divisibility, even rotary
        width) happens when the vision modules are built so a bare config never raises here.

        Returns:
            Dict with ``num_hidden_layers``, ``hidden_size``, ``num_attention_heads``, ``intermediate_size``,
            ``patch_size``, ``rope_theta``, ``downsample_ratio``, ``max_image_tokens``, ``min_pixels``,
            ``max_wh_ratio`` and ``rms_norm_eps``.
        """
        raw = getattr(self, "vision_config", None)
        raw = raw if isinstance(raw, dict) else {}
        params = {name: raw.get(name, default) for name, default in _VISION_DEFAULTS.items()}
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
            "patch_size",
            "downsample_ratio",
            "max_image_tokens",
            "min_pixels",
        ):
            params[name] = int(params[name])
        params["rope_theta"] = float(params["rope_theta"])
        params["rms_norm_eps"] = float(params["rms_norm_eps"])
        return params

    def get_mask_details(self) -> dict:
        """Report every backbone layer as sliding-window attention for engine page accounting.

        As in DeepSeek-V4, token KV is a ``sliding_window`` ring on every layer; compressed state lives in the
        compressed-window cache, not in KV pages.
        """
        from easydel.infra.utils import AttnMaskDetail, AttnMaskType

        return {
            layer_idx: AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=self.sliding_window, chunks=None)
            for layer_idx in range(self.num_hidden_layers)
        }

    def v4_rope_params(self, label: str) -> dict:
        """Rope parameters for ``"main"`` or ``"compress"`` (same accessor name as DeepSeek-V4 for reuse)."""
        private = getattr(self, "_v4_rope_parameters", None)
        if isinstance(private, dict) and isinstance(private.get(label), dict):
            return private[label]
        sub = (getattr(self, "rope_parameters", None) or {}).get(label)
        if isinstance(sub, dict):
            return sub
        raise KeyError(f"DeepSeek-V4.1 rope parameters for label {label!r} not found.")


__all__ = ["DeepseekV41Config"]
