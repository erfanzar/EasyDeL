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

"""XLA backend for fused reduce-scatter matrix multiplication.

Computes ``reduce_scatter(x @ y.T, scatter_dim=0)`` over a device mesh using
XLA ``psum_scatter``.  Provides a custom VJP so gradients flow correctly
through the collective operation.

Key exports:
    - ``reduce_scatter_matmul``: Public entry point registered in the kernel
      registry.
"""

from ._interface import reduce_scatter_matmul

__all__ = ("reduce_scatter_matmul",)
