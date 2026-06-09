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

"""TileLang backend for grouped matrix multiplication v3.

v3 extends v1/v2 with optional per-group block-wise weight scaling
(``rhs_scale``), per-group bias addition (``rhs_bias``), and output
accumulation into an existing tensor (``existing_out``).  It also provides
a full native VJP for all differentiable inputs.

Exports:
    grouped_matmulv3: GPU-accelerated grouped matmul with scale/bias support.
"""

from ._interface import grouped_matmulv3

__all__ = ["grouped_matmulv3"]
