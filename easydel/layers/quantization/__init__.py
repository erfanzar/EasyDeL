# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

# Re-export eformer quantization types for convenience
from eformer.ops.quantization import Array8B, ArrayNF4, QuantizationConfig, QuantizationType, quantize, straight_through

from .linear_8bit import Linear8bit
from .linear_nf4 import LinearNF4
from .quantizers import EasyDeLQuantizationConfig, EasyQuantizer

__all__ = (
    "Array8B",
    "ArrayNF4",
    "EasyDeLQuantizationConfig",
    "EasyQuantizer",
    "Linear8bit",
    "LinearNF4",
    "QuantizationConfig",
    "QuantizationType",
    "quantize",
    "straight_through",
)
