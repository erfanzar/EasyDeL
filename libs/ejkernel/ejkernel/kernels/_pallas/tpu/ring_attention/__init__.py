# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Ring Attention using Splash Attention kernels.

This module provides a ring attention implementation that distributes attention
computation across devices using a ring topology, with JAX's splash attention
kernel as the inner attention mechanism.

Features:
- Distributed ring communication via lax.ppermute
- All splash attention mask types (Causal, ChunkedCausal, Local, Full)
- Custom mask builder support
- Attention sinks (softmax_aux/sinks)
- Logits soft capping
- Full backward pass with custom VJP
"""

from ._interface import ring_attention
from ._pallas_impl_bwd import (
    RING_AXIS,
    BlockSizes,
    RingSplashAttentionKernel,
    SegmentIds,
    make_ring_attention,
    ring_splash_attention,
)

__all__ = [
    "RING_AXIS",
    "BlockSizes",
    "RingSplashAttentionKernel",
    "SegmentIds",
    "make_ring_attention",
    "ring_attention",
    "ring_splash_attention",
]
