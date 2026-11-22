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


"""XLA backend for SSM2 (Mamba2-style) selective state space.

This submodule provides XLA-optimized implementations of SSM2,
the Mamba2 selective state space model architecture.

Key characteristics:
- 1D A vector: [num_heads] (per-head scalar)
- SSM state shape: [batch, num_heads, head_dim, ssm_state_size]
- O(N) complexity through sequential processing
"""

from ._interface import _ssm2_bwd_rule as ssm2_xla_bwd
from ._interface import _ssm2_fwd_rule as ssm2_xla_fwd
from ._interface import state_space_v2

__all__ = ["ssm2_xla_bwd", "ssm2_xla_fwd", "state_space_v2"]
