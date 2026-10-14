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

"""Re-export shim for ``grouped_matmulv2``.

The function and its kernel-registry entry are defined in
:mod:`ejkernel.kernels._tilelang.grouped_matmul._interface`.  This module
simply re-exports the symbol so that the ``grouped_matmulv2`` sub-package
has a canonical import path.
"""

from ..grouped_matmul._interface import grouped_matmulv2

__all__ = ["grouped_matmulv2"]
