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

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.infra.utils import (
	ACT2FN,
	control_mlp_sharding,
	get_dot_general_by_bits,
)

from .deepseek_configuration import DeepseekV3Config


class DeepseekV3MLP(nn.Module):
	def __init__(
		self,
		config: DeepseekV3Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		hidden_size=None,
		intermediate_size=None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
		self.intermediate_size = (
			config.intermediate_size if intermediate_size is None else intermediate_size
		)
		linear_class = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=self.config.mlp_bias,
			kernel_init=jax.nn.initializers.normal(config.initializer_range),
			precision=precision,
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.gate_proj = linear_class(hidden_size, intermediate_size)
		self.down_proj = linear_class(intermediate_size, hidden_size)
		self.up_proj = linear_class(hidden_size, intermediate_size)
		self.dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)
		self.act_fn = ACT2FN[config.hidden_act]

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		hidden_states = self.down_proj(
			self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)
		hidden_states = self.dropout(hidden_states)
		return hidden_states
