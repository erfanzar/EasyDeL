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

from ._configs import QuantizationConfig, QuantizationType
from ._quants import EasyQuantizer, quantize
from ._straight_through import (
    straight_through,
    straight_through_1bit,
    straight_through_8bit,
    straight_through_mxfp4,
    straight_through_mxfp8,
    straight_through_nf4,
    straight_through_nvfp8,
)

__all__ = (
    "EasyQuantizer",
    "QuantizationConfig",
    "QuantizationType",
    "quantize",
    "straight_through",
    "straight_through_1bit",
    "straight_through_8bit",
    "straight_through_mxfp4",
    "straight_through_mxfp8",
    "straight_through_nf4",
    "straight_through_nvfp8",
)
