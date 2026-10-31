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

"""XLA all-gather matmul kernel.

Exposes ``all_gather_matmul``, which gathers a row-sharded LHS across a JAX
device mesh and multiplies by a local column-sharded RHS.  The backward pass
uses ``lax.psum_scatter`` so gradients match the input shard shapes.
"""

from ._interface import all_gather_matmul

__all__ = ("all_gather_matmul",)
