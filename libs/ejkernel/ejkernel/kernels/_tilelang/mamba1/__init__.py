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

"""TileLang ``mamba1`` alias package for ``state_space_v1``.

Re-exports the ``state_space_v1`` function under the name ``mamba1``.
The kernel registration for all three names (``state_space_v1``, ``mamba1``,
``ssm1``) happens in
:mod:`ejkernel.kernels._tilelang.state_space_v1._interface`.

Exports:
    mamba1: Mamba-1 selective scan (alias for ``state_space_v1``).
"""

from ._interface import mamba1

__all__ = ["mamba1"]
