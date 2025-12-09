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

import typing as tp

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("seed_oss")
class SeedOssConfig(EasyDeLBaseConfig):
    """
    Configuration class for the Seed OSS decoder-only transformer.

    The architecture follows a GPT-style stack with:
    - Pre-attention RMSNorm and post-attention RMSNorm
    - Rotary position embeddings with optional scaling
    - Gated SiLU feed-forward network
    - Optional sliding-window attention per-layer

    Default hyper-parameters are aligned with the public Seed OSS checkpoints.
    """

    model_type = "seed_oss"

    def __init__(
        self,
        *,
        vocab_size: int = 200704,
        hidden_size: int = 7168,
        intermediate_size: int = 20480,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 56,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1.0e-5,
        rope_theta: float = 1_000_000.0,
        rope_scaling: tp.Mapping[str, tp.Any] | None = None,
        tie_word_embeddings: bool = False,
        attention_dropout: float = 0.0,
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        use_cache: bool = True,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int | None = None,
        layer_types: list[str] | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        gradient_checkpointing_targets: tuple[str, ...] | None = None,
        scan_layers: bool = True,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        attention_bias: bool = True,
        attention_out_bias: bool = False,
        residual_dropout=0.1,
        mlp_bias: bool = False,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        if max_window_layers is None:
            max_window_layers = num_hidden_layers

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = dict(rope_scaling) if rope_scaling is not None else None
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.layer_types = layer_types
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.residual_dropout = residual_dropout
        self.mlp_bias = mlp_bias
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if (self.use_sliding_window and self.sliding_window is not None and i < self.max_window_layers)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_targets = gradient_checkpointing_targets or ()
        self.scan_layers = scan_layers
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        # Seed OSS uses attention bias on Q/K/V projections but not on O projection.
        self.attention_bias = True

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, tp.Any], ...]:
        """
        Partition rules for optimised sharding.

        Mirrors the standard decoder-only layout:
            - Embed/lm_head sharded column-wise.
            - Attention QKV projections column-wise, output row-wise.
            - MLP gate/up column-wise, down row-wise.
            - Normalisation parameters replicated.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"layers.*/self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"layers.*/self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"layers.*/self_attn/(q_proj|k_proj|v_proj)/bias", pmag.resolve(Replicated)),
            (r"layers.*/self_attn/o_proj/bias", pmag.resolve(Replicated)),
            (r"layers.*/mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"layers.*/mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"layers.*/mlp/.*_proj/bias", pmag.resolve(Replicated)),
            (r"layers.*/(input_layernorm|post_attention_layernorm)/kernel", pmag.resolve(Replicated)),
            (r"norm/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Return per-layer attention mask settings."""
        mapping: dict[int, AttnMaskDetail] = {}
        for layer_idx, layer_type in enumerate(self.layer_types or ()):
            mapping[layer_idx] = AttnMaskDetail(
                mask_type=AttnMaskType.from_hf(layer_type),
                size=self.sliding_window,
            )
        return mapping


__all__ = ["SeedOssConfig"]
