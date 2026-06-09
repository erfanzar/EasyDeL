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

"""Pallas TPU backend for Grouped Matrix Multiplication v3.

This submodule provides the v3 grouped matrix multiplication kernel for TPU,
adapted from the upstream ``gmm_v2`` kernel.  It replaces the v2
``buffered_pallas_call`` dispatch with JAX's ``emit_pipeline`` and supports
optional fused activations (SiLU, GeLU, SwiGLUoai) and per-group RHS
scale/bias for quantised weight formats.
"""

from ._interface import grouped_matmulv3

__all__ = ("grouped_matmulv3",)
