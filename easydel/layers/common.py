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

from typing import Optional, Union

from flax import linen as nn
from jax import lax
from jax import numpy as jnp


class Conv1D(nn.Module):
	features: int
	kernel_size: int = 1
	stride: int = 1
	padding: int = 0
	dilation: int = 1
	groups: int = 1
	use_bias: bool = True
	num_spatial_dims: int = 1
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, lax.Precision]] = None

	@nn.compact
	def __call__(self, x):
		kernel = self.param(
			"kernel",
			nn.initializers.lecun_normal(dtype=self.param_dtype),
			(self.features, 1, self.kernel_size),
			self.param_dtype,
		)
		unbatched_rank = self.num_spatial_dims + 2
		if x.ndim != unbatched_rank:
			raise ValueError(
				f"Input to `Conv` needs to have rank {unbatched_rank},"
				f" but input has shape {x.shape}.",
			)

		x = lax.conv_general_dilated(
			lhs=x,
			rhs=jnp.asarray(kernel, dtype=self.dtype),
			window_strides=(self.stride,),
			padding=((self.padding, self.padding),),
			rhs_dilation=(self.dilation,),
			feature_group_count=self.groups,
		)
		if self.use_bias:
			bias = self.param(
				"bias", nn.initializers.zeros_init(), (self.features,), self.param_dtype
			)
			x = x + jnp.asarray(bias.reshape(1, -1, 1), dtype=self.dtype)
		return x
