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


"""Pallas TPU backend for Grouped Matrix Multiplication v2.

This submodule provides an improved TPU-optimized grouped matrix
multiplication using Pallas with enhanced tiling strategies.

Key Features:
    - Improved tiling for better TPU utilization
    - Dynamic tile size selection via LutFn
    - Enhanced prefetching for memory latency hiding
    - Support for transposed RHS matrices
"""

from ._interface import grouped_matmulv2

__all__ = ("grouped_matmulv2",)
