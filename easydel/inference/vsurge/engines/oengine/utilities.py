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

from functools import partial

import jax
from jax import numpy as jnp


from .._utils import apply_temperature, apply_top_k, apply_top_p


@partial(jax.vmap, in_axes=(0, 0, 0, 0), out_axes=(0))
def _apply_filters(logits, top_p, top_k, temperature):
	logits = jnp.expand_dims(logits, 0)
	logits = apply_temperature(logits, temperature.astype(logits.dtype))
	logits = apply_top_k(logits, top_k)
	logits = apply_top_p(logits, top_p.astype(logits.dtype))
	return logits[0]

