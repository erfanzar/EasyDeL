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

"""Pallas TPU all-gather matmul kernel.

Performs a bidirectional ring all-gather on the LHS shard ``x`` across a
TPU device mesh and then computes ``all_gather(x) @ y`` entirely on-device,
overlapping communication with MXU computation.

Public API:
    all_gather_matmul: Registered under ``Platform.PALLAS / Backend.TPU``.
        Custom VJP uses a fused reduce-scatter matmul for the LHS gradient
        and a standard ``lax.all_gather`` + dot for the RHS gradient.
"""

from ._interface import all_gather_matmul

__all__ = ("all_gather_matmul",)
