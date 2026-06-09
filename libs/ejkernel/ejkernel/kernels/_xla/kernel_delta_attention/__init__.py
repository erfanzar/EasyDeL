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


"""XLA backend for Kernel Delta Attention (DeltaNet).

This submodule provides XLA-optimized implementation of kernel delta attention,
a linear attention variant using delta rule updates for state management.

Key Features:
    - O(N) complexity through recurrent state updates
    - Delta rule for selective memory management
    - Configurable decay factors for temporal weighting
    - Support for chunked, recurrent, and single-step modes

Reference:
    DeltaNet: Conditional State Space Models with Selective State Updates
"""

from ._interface import kda, kda_decay, kernel_delta_attention
from ._xla_impl_fwd import _chunk_kda_fwd as kernel_delta_attention_xla_chunk_fwd
from ._xla_impl_fwd import _recurrent_kda_fwd as kernel_delta_attention_xla_recurrent_fwd
from ._xla_impl_fwd import _single_step_kda_fwd as kernel_delta_attention_xla_single_step_fwd

__all__ = (
    "kda",
    "kda_decay",
    "kernel_delta_attention",
    "kernel_delta_attention_xla_chunk_fwd",
    "kernel_delta_attention_xla_recurrent_fwd",
    "kernel_delta_attention_xla_single_step_fwd",
)
