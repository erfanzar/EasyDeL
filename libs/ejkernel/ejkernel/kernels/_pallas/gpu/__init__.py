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


"""Pallas GPU kernel implementations (Triton backend).

Provides GPU-optimized attention kernels written with JAX Pallas and compiled
via the Triton backend.  Import from this package requires Triton to be
installed; the parent ``_pallas`` package suppresses ``ModuleNotFoundError``
when Triton is absent.

Available kernels:
    ragged_decode_attention: Single-token decode attention for variable-length
        (ragged) KV caches.  Implements online-softmax tiling over the key
        sequence with configurable head/key block sizes.
    scaled_dot_product_attention: Full scaled dot-product attention delegating
        to ``jax.nn.dot_product_attention`` with ``implementation="cudnn"``.
        Supports causal masking, sliding-window, GQA/MQA, and variable-length
        packed sequences.
"""

from .ragged_decode_attention import ragged_decode_attention
from .scaled_dot_product_attention import scaled_dot_product_attention

__all__ = ("ragged_decode_attention", "scaled_dot_product_attention")
