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

"""Attention mechanisms and decoder layer utilities.

Public entry point for everything attention-related in EasyDeL. The
subpackage owns three layers of abstraction stacked on top of each other:

* :class:`AttentionMechanisms` and :class:`FlexibleAttentionModule` — the
  thin dispatcher that routes Q/K/V plus a bag of optional knobs (mask,
  bias, sliding-window, soft-caps, ...) to the appropriate registered
  kernel (FlashAttention 2, Splash, Ring, ragged-page, MLA, etc.).
* :class:`AttentionModule` — abstract sharding / mask / cache helper used
  by concrete attention implementations.
* :class:`UnifiedAttention` — the canonical base class used by ~70 model
  attention implementations across the repository. Subclasses override a
  handful of ``_create_*`` / ``_postprocess_qkv`` hooks instead of
  reimplementing the full forward path.

Decoder-layer helpers (:class:`BaseDecoderLayer`, :func:`blockwise_ffn`)
live alongside attention because the canonical pre-LN ``h + sublayer(norm(h))``
patterns are tightly coupled to the attention module's outputs.
"""

from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]

from ._decoder_base import BaseDecoderLayer, blockwise_ffn
from ._flexible import (
    AttentionMechanisms,
    AttentionModule,
    FlexibleAttentionModule,
    get_optimal_config,
    tpu_version_check,
)
from ._unified import UnifiedAttention

__all__ = (
    "AttentionMechanisms",
    "AttentionModule",
    "BaseDecoderLayer",
    "FlexibleAttentionModule",
    "MaskInfo",
    "UnifiedAttention",
    "blockwise_ffn",
    "get_optimal_config",
    "tpu_version_check",
)
