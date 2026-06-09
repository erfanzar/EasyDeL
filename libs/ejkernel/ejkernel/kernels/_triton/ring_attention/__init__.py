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

"""Ring Attention using Triton Block-sparse Attention.

This module provides a ring attention implementation that distributes attention
computation across devices using a ring topology, with the Triton flash
attention kernel historically used as the inner mechanism. The current
implementation uses Triton block-sparse attention to correctly handle explicit
positions/segment IDs under distributed causal/window masking.

Features:
- Distributed ring communication via lax.ppermute
- Causal + sliding window masking with explicit positions/segment IDs
- Logits soft capping
- Full backward pass with custom VJP
"""

from ._interface import ring_attention
from ._triton_impl_bwd import RingFlashResiduals, ring_flash_attention_call

__all__ = ("RingFlashResiduals", "ring_attention", "ring_flash_attention_call")
