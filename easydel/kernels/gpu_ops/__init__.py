# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

# Implementation by @erfanzar,
# with a few bug fixes and adjustments.
from .pallas_gemm import gpu_matmul
from .pallas_gqa_flash_attention_2 import pallas_gqa_flash_attention2_gpu
from .pallas_mha_flash_attention_2 import pallas_mha_flash_attention2_gpu
from .triton_gemm import gemm
from .triton_gqa_flash_attention_2 import triton_gqa_flash_attention2_gpu

__all__ = [
	"triton_gqa_flash_attention2_gpu",
	"pallas_gqa_flash_attention2_gpu",
	"pallas_mha_flash_attention2_gpu",
	"gemm",
	"gpu_matmul",
]
