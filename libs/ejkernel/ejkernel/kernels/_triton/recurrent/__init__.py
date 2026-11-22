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


"""Triton backend for Recurrent Attention.

This submodule provides GPU-optimized recurrent linear attention using
Triton kernels for O(N) complexity sequential processing.

Key Features:
    - Linear complexity through state recurrence
    - Custom forward and backward Triton kernels
    - State caching for incremental inference
    - Support for various gating mechanisms
"""

from ._interface import (
    _bwd_call as recurrent_gpu_bwd,
)
from ._interface import (
    _fwd_call as recurrent_gpu_fwd,
)
from ._interface import (
    recurrent,
)

__all__ = [
    "recurrent",
    "recurrent_gpu_bwd",
    "recurrent_gpu_fwd",
]
