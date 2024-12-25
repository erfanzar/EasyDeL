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
from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from flax.nnx.module import Module
from flax.nnx.nn import initializers
from flax.typing import (
	Dtype,
	PrecisionLike,
)

Array = jax.Array
Axis = int
Size = int


default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


class QauntModule(Module):
	def __init__(
		self,
		dtype: tp.Optional[Dtype] = None,
		param_dtype: Dtype = jnp.float32,
		precision: PrecisionLike = None,
	):
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

	@staticmethod
	def metadata():
		raise NotImplementedError()

	@staticmethod
	def quantization_mapping():
		raise NotImplementedError()
