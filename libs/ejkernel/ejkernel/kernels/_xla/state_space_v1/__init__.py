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


"""XLA backend for SSM1 (Mamba1-style) selective state space.

This submodule provides XLA-optimized implementations of SSM1,
the original Mamba selective state space model architecture.

Key characteristics:
- 2D A matrix: [intermediate_size, ssm_state_size]
- SSM state shape: [batch, intermediate_size, ssm_state_size]
- O(N) complexity through sequential processing
"""

from ._interface import _ssm1_bwd_rule as ssm1_xla_bwd
from ._interface import _ssm1_fwd_rule as ssm1_xla_fwd
from ._interface import state_space_v1

__all__ = [
    "ssm1_xla_bwd",
    "ssm1_xla_fwd",
    "state_space_v1",
]
