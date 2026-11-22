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

"""TurboQuant: Two-stage vector quantization for KV cache compression.

Implements the TurboQuant algorithm (ICLR 2026, arXiv:2504.19874) which
compresses key-value caches using:

1. Random rotation + Lloyd-Max scalar quantization (MSE-optimal)
2. QJL residual correction for unbiased inner products on keys

Usage:
    >>> from ejkernel.quantization.turboquant import (
    ...     solve_lloyd_max,
    ...     generate_rotation_matrix,
    ...     generate_projection_matrix,
    ...     turboquant_compress_keys,
    ...     turboquant_compress_values,
    ...     turboquant_dequantize_values,
    ...     turboquant_asymmetric_scores,
    ...     pack_4bit, unpack_4bit,
    ...     pack_signs, unpack_signs,
    ... )
"""

from .codebook import LloydMaxCodebook, solve_lloyd_max
from .matrices import generate_projection_matrix, generate_rotation_matrix
from .ops import (
    turboquant_asymmetric_scores,
    turboquant_compress_keys,
    turboquant_compress_values,
    turboquant_dequantize_values,
)
from .packing import pack_4bit, pack_signs, unpack_4bit, unpack_signs

__all__ = [
    "LloydMaxCodebook",
    "generate_projection_matrix",
    "generate_rotation_matrix",
    "pack_4bit",
    "pack_signs",
    "solve_lloyd_max",
    "turboquant_asymmetric_scores",
    "turboquant_compress_keys",
    "turboquant_compress_values",
    "turboquant_dequantize_values",
    "unpack_4bit",
    "unpack_signs",
]
