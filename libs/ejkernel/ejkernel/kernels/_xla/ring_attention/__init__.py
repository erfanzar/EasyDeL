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


"""XLA backend for Ring Attention distributed attention.

This submodule provides XLA-optimized implementation of Ring Attention
for processing long sequences across multiple devices in a ring topology.

Key Features:
    - Distributed attention across multiple devices
    - Ring communication pattern for memory efficiency
    - Support for causal masking with blockwise patterns
    - Compatible with JAX's pjit for multi-device execution

Algorithm:
    Ring Attention distributes sequence chunks across devices,
    computing attention incrementally while passing key-value
    blocks around the ring to minimize memory per device.

Reference:
    Ring Attention with Blockwise Transformers for Near-Infinite Context
    https://arxiv.org/abs/2310.01889
"""

from ._interface import ring_attention
from ._xla_impl_bwd import _ring_attention_bwd as ring_attention_xla_bwd
from ._xla_impl_fwd import _ring_attention_fwd as ring_attention_xla_fwd

__all__ = [
    "ring_attention",
    "ring_attention_xla_bwd",
    "ring_attention_xla_fwd",
]
