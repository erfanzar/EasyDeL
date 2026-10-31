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


"""XLA backend for recurrent linear attention.

Implements O(N) linear attention through sequential state updates.  Supports
multiple gating variants (GLA, Lightning attention, key/value gating) and
variable-length packed sequences.

Exports:
    - ``recurrent``: Public API registered in the kernel registry.
    - ``recurrent_xla_fwd`` (``_recurrent_fwd``): VJP forward rule; saves
      residuals for the backward pass.
    - ``recurrent_xla_bwd`` (``_recurrent_bwd``): VJP backward rule;
      computes gradients for Q, K, V, gates, and the initial hidden state.
"""

from ._interface import (
    _recurrent_bwd as recurrent_xla_bwd,
)
from ._interface import (
    _recurrent_fwd as recurrent_xla_fwd,
)
from ._interface import (
    recurrent,
)

__all__ = [
    "recurrent",
    "recurrent_xla_bwd",
    "recurrent_xla_fwd",
]
