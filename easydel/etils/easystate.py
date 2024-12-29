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

import os
import pathlib
import typing as tp

import jax
import optax
from flax import nnx as nn
from flax import struct

from easydel.etils.etils import get_logger

if tp.TYPE_CHECKING:
	from easydel.infra.base_module import EasyDeLBaseModule
else:
	EasyDeLBaseModule = tp.Any

logger = get_logger(__name__)
WEIGHTS_NAME = "easydel-model.parameters"
OPTIMIZER_NAME = "easydel-optstate.parameters"


class EasyDeLState(struct.PyTreeNode):
	"""
	**EasyDeLState A Snapshot of Your EasyDeL Model**

	The `EasyDeLState` class acts like a comprehensive container that holds all the essential information about your EasyDeL
	model at a given point in time. Think of it as a snapshot of your model. It includes
	"""

	step: int | jax.Array
	graphdef: nn.GraphDef
	graphstate: nn.GraphState
	graphother: nn.GraphState
	tx: optax.GradientTransformation = struct.field(pytree_node=False)
	opt_state: tp.Optional[optax.OptState] = struct.field(pytree_node=True)

	def apply_gradients(self, *, grads):
		"""
		Applies gradients to the model parameters and updates the optimizer state.
		This function is typically called during training to update the model based on the computed gradients.

		Args:
		    grads: A dictionary of gradients, where keys correspond to model parameters.

		Returns:
		    EasyDeLState: An updated EasyDeLState object with modified parameters and optimizer state.
		"""
		assert self.opt_state is not None
		assert self.tx is not None
		# updates: Updates, state: OptState, params: Optional[Params] = None
		updates, new_opt_state = self.tx.update(
			updates=grads,
			state=self.opt_state,
			params=self.graphstate,
		)
		graphstate = optax.apply_updates(self.graphstate, updates)
		return self.replace(
			step=self.step + 1,
			graphstate=graphstate,
			opt_state=new_opt_state,
		)

	@classmethod
	def create(
		cls,
		*,  # Force keyword arguments
		step: tp.Optional[int] = None,
		graphdef: tp.Optional[nn.GraphDef] = None,
		graphstate: tp.Optional[nn.GraphState] = None,
		graphother: tp.Optional[nn.GraphState] = None,
		model: tp.Optional[nn.Module] = None,
		tx: tp.Optional[optax.GradientTransformation] = None,
		opt_state: tp.Optional[optax.OptState] = None,
		init_opt_state: bool = False,
	) -> EasyDeLState:
		"""
		Create an instance with flexible initialization options.

		Args:
				step: Optional number of training steps.
				graphdef: Optional graph definition.
				graphstate: Optional graph state.
				graphother: Optional graph *others.
				model: Optional neural network module.
				tx: Optional gradient transformation.
				opt_state: Optional optimizer state.

		Raises:
				ValueError: If initialization parameters are inconsistent.
		"""
		# Validate mutual exclusivity of model and graph-related parameters
		graph_params_provided = (
			graphdef is not None or graphstate is not None or graphother is not None
		)

		if model is not None and graph_params_provided:
			raise ValueError(
				"Cannot provide both a model and graph-related parameters. "
				"Choose either model or (graphdef, graphstate)."
			)

		if model is not None:
			graphdef, graphstate, graphother = nn.split(model, nn.Param, ...)

		if graphdef is not None and graphstate is None and graphother is None:
			raise ValueError(
				"When providing graphdef, (graphstate, graphother) must also be provided.",
			)

		if graphstate is not None and graphdef is None and graphother is None:
			raise ValueError(
				"When providing graphstate, (graphdef, graphother) must also be provided.",
			)
		if graphother is not None and graphdef is None and graphstate is None:
			raise ValueError(
				"When providing graphother, (graphstate, graphdef) must also be provided.",
			)
		if init_opt_state and opt_state is not None:
			raise ValueError(
				"When passing `init_opt_state` as `True` you can't also provide `opt_state`"
			)
		if init_opt_state and tx is None:
			raise ValueError(
				"When passing `init_opt_state` as `True` you have to also provide `tx`."
			)

		if init_opt_state:
			opt_state = tx.init(graphstate)
		if step is None:
			step = 0

		return cls(
			step=step,
			graphdef=graphdef,
			graphstate=graphstate,
			graphother=graphother,
			tx=tx,
			opt_state=opt_state,
		)

	def merge(self, tree) -> EasyDeLBaseModule:
		return nn.merge(self.graphdef, tree, self.graphother)

	@property
	def model(self) -> EasyDeLBaseModule:
		return nn.merge(self.graphdef, self.graphstate, self.graphother)

	def save_state(
		self,
		save_directory: tp.Union[str, os.PathLike],
		float_dtype: tp.Optional[jax.numpy.dtype] = None,
		verbose: bool = True,
		mismatch_allowed: bool = True,
		save_optimizer: bool = True,
	):
		save_directory = pathlib.Path(save_directory)
		self.model.save_pretrained(
			save_directory=str(save_directory),
			gather_fns=self.model._gather_fns,
			float_dtype=float_dtype,
			mismatch_allowed=mismatch_allowed,
			verbose=verbose,
		)
		# Fix this
		# if save_optimizer:
		# 	CheckpointManager.save_checkpoint(
		# 		state=self.opt_state,
		# 		path=str(save_directory / OPTIMIZER_NAME),
		# 		float_dtype=float_dtype,
		# 		gather_fns=self.model._gather_fns,
		# 		verbose=verbose,
		# 		mismatch_allowed=mismatch_allowed,
		# 	)

	def load_state(self): ...

	def __repr__(self):
		return "EasyDeLState-" + str(self.model)

	__str__ = __repr__
