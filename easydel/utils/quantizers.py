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

from typing import Literal, Optional

import chex
import jax
from fjformer.dtypes import Array8Bit, ArrayNF4, A8Q, A4Q

DEFAULT_QUANTIZATION_PATTERN = (
	"(wo|wq|wk|wv|q_proj|k_proj|v_proj|o_proj|w1|w2|w3|"
	"gate_proj|up_proj|down_proj|dense_4h_to_h|dense_h_to_4h|query_key_value|wqkv|Wqkv|"
	"dense|proj_1|proj_2|out_proj|qkv_proj)"
)


class EasyQuantizer:
	def __init__(
		self,
		quantization_method: Literal["nf4", "8bit", "a8q", "a4q"] = "nf4",
		quantization_platform: Optional[Literal["jax", "triton", "pallas"]] = "jax",
		**kwargs,
	) -> None:
		self.scalar_block_size = 32
		self.block_size = 128
		self.quantization_method = quantization_method
		self.quantization_platform = quantization_platform

	def __call__(self, array) -> chex.Array:
		match self.quantization_method:
			case "8bit":
				return Array8Bit.quantize(
					array=array,
					platform=self.quantization_platform,
					q8=64,
				)
			case "a8q":
				return A8Q.quantize(array=array, q8=32)
			case "a4q":
				return A4Q.quantize(array=array, q8=64)
			case "nf4":
				should_be_quantized = True
				if array.shape[0] % 128 != 0:
					should_be_quantized = False
				if array.ndim <= 2 and should_be_quantized:
					return ArrayNF4.quantize(array=array.astype(jax.numpy.float32))
				return array
			case _:
				raise ValueError(f"unknown quantization_method {self.quantization_method}.")
