"""Attention mechanisms and decoder layer utilities.

This subpackage consolidates:
- FlexibleAttentionModule, AttentionModule and AttentionMechanisms
- UnifiedAttention base class
- BaseDecoderLayer and block_wise_ffn
"""

from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]

from ._decoder_base import BaseDecoderLayer, block_wise_ffn
from ._flexible import (
    AttentionMechanisms,
    AttentionModule,
    FlexibleAttentionModule,
    get_optimal_config,
    tpu_version_check,
)
from ._unified import UnifiedAttention, apply_rotary_pos_emb, yarn_get_mscale

__all__ = (
    "AttentionMechanisms",
    "AttentionModule",
    "BaseDecoderLayer",
    "FlexibleAttentionModule",
    "MaskInfo",
    "UnifiedAttention",
    "apply_rotary_pos_emb",
    "block_wise_ffn",
    "get_optimal_config",
    "tpu_version_check",
    "yarn_get_mscale",
)
