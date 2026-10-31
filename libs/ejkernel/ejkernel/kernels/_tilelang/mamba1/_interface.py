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

"""Re-export shim that exposes ``state_space_v1`` under the name ``mamba1``.

The kernel-registry entry for ``"mamba1"`` is created by a stacked
``@kernel_registry.register`` decorator in
:mod:`ejkernel.kernels._tilelang.state_space_v1._interface`; importing from
this module is sufficient to trigger that registration.
"""

from ..state_space_v1._interface import state_space_v1 as mamba1

__all__ = ["mamba1"]
