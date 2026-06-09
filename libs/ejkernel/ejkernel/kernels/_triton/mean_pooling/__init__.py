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


"""Triton backend for Mean Pooling.

This submodule provides GPU-optimized mean pooling using Triton kernels
for efficient sequence embedding aggregation.

Key Features:
    - Sequence mean pooling with mask support
    - Efficient handling of variable-length sequences
    - Custom forward and backward Triton kernels
    - Support for packed sequence processing
"""

from ._interface import (
    _bwd_call as mean_pooling_gpu_bwd,
)
from ._interface import (
    _fwd_call as mean_pooling_gpu_fwd,
)
from ._interface import (
    mean_pooling,
)

__all__ = [
    "mean_pooling",
    "mean_pooling_gpu_bwd",
    "mean_pooling_gpu_fwd",
]
