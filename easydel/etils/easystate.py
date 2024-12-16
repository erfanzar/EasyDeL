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

import jax
import optax
from flax import nnx as nn
from flax import struct


from easydel.etils.etils import get_logger

logger = get_logger(__name__)


class EasyDeLState(struct.PyTreeNode):
	"""
	**EasyDeLState A Snapshot of Your EasyDeL Model**

	The `EasyDeLState` class acts like a comprehensive container that holds all the essential information about your EasyDeL
	model at a given point in time. Think of it as a snapshot of your model. It includes
	"""

	step: int | jax.Array
	graphdef: nn.GraphDef
	grephstate: nn.GraphState
	tx: optax.GradientTransformation = struct.field(pytree_node=False)
	opt_state: tp.Optional[optax.OptState] = struct.field(pytree_node=True)

	def apply_gradients(self, *, grads, **kwargs):
		"""
		Applies gradients to the model parameters and updates the optimizer state.

		This function is typically called during training to update the model based on the computed gradients.

		Args:
		    grads: A dictionary of gradients, where keys correspond to model parameters.
		    **kwargs: Additional keyword arguments.

		Returns:
		    EasyDeLState: An updated EasyDeLState object with modified parameters and optimizer state.
		"""

		grads_with_opt = grads
		params_with_opt = self.params

		updates, new_opt_state = self.tx.update(
			grads_with_opt, self.opt_state, params_with_opt
		)
		new_params_with_opt = optax.apply_updates(params_with_opt, updates)
		new_params = new_params_with_opt
		return self.replace(
			step=self.step + 1,
			params=new_params,
			opt_state=new_opt_state,
			**kwargs,
		)

	@classmethod
	def create(cls, **kwargs): ...
