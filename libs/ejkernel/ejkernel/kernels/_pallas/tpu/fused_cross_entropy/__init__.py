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

"""TPU Pallas fused cross-entropy package.

Importing this package exposes and registers the TPU Pallas implementation of
``fused_cross_entropy``. The implementation supports sparse integer targets,
masked row-block skipping, DMA-backed HBM-to-VMEM streaming, analytic backward,
and vocab-parallel execution when called from ``shard_map`` with a TP axis.
"""

from ._interface import fused_cross_entropy

__all__ = ["fused_cross_entropy"]
