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

"""TileLang backend for Gated Linear Attention (GLA).

GLA is implemented by routing through the shared tile-lang recurrent kernel
(``recurrent._impl.recurrent_tilelang``) with per-head key-side gate decay
enabled via the ``g`` / ``g_gamma`` arguments.

Exports:
    recurrent_gla: GPU-accelerated GLA forward + backward.
"""

from ._interface import recurrent_gla

__all__ = ["recurrent_gla"]
