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


"""GPU backend for scaled dot-product attention (cuDNN implementation).

Provides a thin wrapper around ``jax.nn.dot_product_attention`` with
``implementation="cudnn"``.  Registered under both ``Platform.PALLAS`` and
``Platform.TRITON`` in the kernel registry so that either platform selector
resolves to this cuDNN-backed implementation.

Despite the module path placing it under ``_pallas``, the actual compute
kernel is cuDNN — JAX's Pallas/Triton layer is only used for registration
routing, not for lowering the attention computation itself.

Public API:
    scaled_dot_product_attention: Supports causal masking, sliding-window
        local attention, GQA/MQA, bias tensors, and variable-length
        packed sequences via cumulative sequence-length arrays.
"""

from ._interface import scaled_dot_product_attention

__all__ = ("scaled_dot_product_attention",)
