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

"""Configuration class for the Seed-OSS model family.

Defines :class:`SeedOssConfig`, the EasyDeL configuration object for
ByteDance's Seed-OSS dense decoder-only architecture (RMSNorm, GQA,
SwiGLU MLP, RoPE with optional scaling, optional sliding-window
attention via per-layer ``layer_types``).
"""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("seed_oss")
class SeedOssConfig(EasyDeLBaseConfig):
    """Configuration class for the Seed-OSS decoder-only transformer.

    The architecture follows a GPT-style stack with:

    - Pre-attention RMSNorm and post-attention RMSNorm
    - Rotary position embeddings with optional scaling
    - Gated SiLU (SwiGLU) feed-forward network
    - Optional sliding-window attention per-layer

    Default hyper-parameters are aligned with the public Seed-OSS checkpoints.

    Args:
        vocab_size (`int`, *optional*, defaults to 200704):
            Vocabulary size of the Seed-OSS tokenizer.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 20480):
            Dimensionality of the MLP intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of transformer decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 56):
            Number of query attention heads.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads (GQA). Defaults to
            ``num_attention_heads`` (MHA) when not provided.
        head_dim (`int`, *optional*):
            Dimensionality of each attention head. Defaults to
            ``hidden_size // num_attention_heads`` when not provided.
        hidden_act (`str`, *optional*, defaults to ``"silu"``):
            Activation function used in the gated MLP.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length the model supports.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMSNorm layers.
        rope_theta (`float`, *optional*, defaults to 1_000_000.0):
            Base frequency for rotary position embeddings.
        rope_scaling (`Mapping`, *optional*):
            RoPE scaling configuration (e.g. ``{"type": "yarn", "factor": 4.0}``).
        tie_word_embeddings (`bool`, *optional*, defaults to ``False``):
            Whether to tie input and output embedding weights.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability on attention weights.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            Dropout probability on residual connections.
        embd_pdrop (`float`, *optional*, defaults to 0.0):
            Dropout probability on the input embeddings.
        use_cache (`bool`, *optional*, defaults to ``True``):
            Whether to cache key/value tensors for incremental decoding.
        use_sliding_window (`bool`, *optional*, defaults to ``False``):
            Whether to enable sliding-window attention on the first
            ``max_window_layers`` layers.
        sliding_window (`int`, *optional*):
            Sliding-window size (only used when ``use_sliding_window=True``).
        max_window_layers (`int`, *optional*):
            Number of leading layers that use sliding-window attention.
            Defaults to ``num_hidden_layers`` when not provided.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type (``"full_attention"`` or
            ``"sliding_attention"``). Auto-derived from the sliding-window
            settings when not provided.
        gradient_checkpointing (`EasyDeLGradientCheckPointers`, *optional*):
            Gradient-checkpointing policy. Defaults to
            :attr:`EasyDeLGradientCheckPointers.NONE`.
        gradient_checkpointing_targets (`tuple[str, ...]`, *optional*):
            Optional named tensors to save through checkpointing.
        scan_layers (`bool`, *optional*, defaults to ``True``):
            Whether to scan the decoder stack with ``lax.scan``.
        use_scan_mlp (`bool`, *optional*, defaults to ``False``):
            Whether to chunk the MLP forward pass to reduce peak memory.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            Chunk size used when ``use_scan_mlp`` is enabled.
        bits (`int`, *optional*):
            Optional quantization bit-width (forwarded to
            :class:`EasyDeLBaseConfig`).
        attention_bias (`bool`, *optional*, defaults to ``True``):
            Whether the Q/K/V projections carry a bias term.
        attention_out_bias (`bool`, *optional*, defaults to ``False``):
            Whether the output projection carries a bias term.
        residual_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability applied at residual connections.
        mlp_bias (`bool`, *optional*, defaults to ``False``):
            Whether the MLP projections carry bias terms.
        **kwargs: Additional keyword arguments forwarded to
            :class:`EasyDeLBaseConfig`.
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
        rope_scaling: Mapping[str, tp.Any] | None = None,
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
        residual_dropout: float = 0.1,
        mlp_bias: bool = False,
        **kwargs,
    ):
        """Initialize SeedOssConfig.

        See the class docstring for parameter semantics. Defaults are
        chosen to match the public Seed-OSS checkpoints; in particular
        ``num_key_value_heads`` defaults to ``num_attention_heads``,
        ``head_dim`` defaults to ``hidden_size // num_attention_heads``,
        and ``max_window_layers`` defaults to ``num_hidden_layers`` when
        not provided. ``**kwargs`` are forwarded to
        :class:`EasyDeLBaseConfig`.
        """
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
                (
                    "sliding_attention"
                    if (self.use_sliding_window and self.sliding_window is not None and i < self.max_window_layers)
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]

        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_targets = gradient_checkpointing_targets or ()
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            scan_layers=scan_layers,
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Materialise per-layer attention-mask metadata.

        Walks ``layer_types`` (one entry per decoder layer) and
        produces a mapping ``layer_idx -> AttnMaskDetail`` where the
        mask type is derived from the HuggingFace string descriptor
        and the sliding-window size is shared across all sliding
        layers via :attr:`sliding_window`.

        The returned mapping is consumed by EasyDeL's attention
        dispatcher to pick the right kernel per layer (full vs.
        sliding-window causal).

        Returns:
            dict[int, AttnMaskDetail]: One entry per layer present in
            ``layer_types``; an empty dict when ``layer_types`` is
            ``None``.
        """
        mapping: dict[int, AttnMaskDetail] = {}
        for layer_idx, layer_type in enumerate(self.layer_types or ()):
            mapping[layer_idx] = AttnMaskDetail(
                mask_type=AttnMaskType.from_hf(layer_type),
                size=self.sliding_window,
            )
        return mapping


__all__ = ["SeedOssConfig"]
