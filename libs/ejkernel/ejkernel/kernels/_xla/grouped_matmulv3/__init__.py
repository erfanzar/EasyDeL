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

"""XLA backend for Grouped Matrix Multiplication v3 (GMM v3).

This submodule provides the XLA implementation of grouped GEMM v3, which
extends the base ``grouped_matmul`` with support for optional per-group
block-float scale (``rhs_scale``) and bias (``rhs_bias``) tensors.

When ``rhs_scale`` or ``rhs_bias`` is provided, the computation falls back to
a vmap-based pure-JAX reference to keep all parameters differentiable.
Otherwise, ``jax.lax.ragged_dot_general`` is used for efficiency.

Registered keys: ``"grouped_matmulv3"`` (XLA platform, any backend).
"""

from ._interface import grouped_matmulv3

__all__ = ("grouped_matmulv3",)
