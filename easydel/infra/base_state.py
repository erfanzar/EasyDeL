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
from functools import partial

import jax
import optax
from flax import nnx as nn
from flax import struct

from easydel.utils.traversals import specs_to_name_sharding

if tp.TYPE_CHECKING:
	from jax.sharding import Mesh, PartitionSpec

	from .base_module import EasyDeLBaseModule, PartitionLike
else:
	EasyDeLBaseModule = tp.Any
	PartitionSpec = tp.Any
	PartitionLike = tp.Any
	Mesh = tp.Any

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

	def init_tx(
		self,
		tx: optax.GradientTransformation,
		partition_rules: PartitionLike = None,
	) -> EasyDeLState:
		"""
		Initialize the optimizer state with the given gradient transformation.

		Args:
			tx (optax.GradientTransformation): A gradient transformation to initialize the optimizer state.
			partition_rules (Optional[Any], optional): Rules for partitioning the optimizer state. Defaults to None.

		Returns:
			EasyDeLState: An updated EasyDeLState object with the new gradient transformation and sharded optimizer state.
		"""

		if partition_rules is None:
			partition_rules = self.model.config.get_partition_rules()

		from easydel.escale import match_partition_rules

		eval_opt_state = jax.eval_shape(lambda: tx.init(self.graphstate))
		partition_specs = match_partition_rules(partition_rules, eval_opt_state)
		named_shardings = specs_to_name_sharding(partition_specs, self.model.mesh)

		@partial(jax.jit, out_shardings=named_shardings)
		def make():
			return tx.init(self.graphstate)

		opt_state = make()
		return self.replace(tx=tx, opt_state=opt_state)

	def shard_optimizer_state(
		self,
		opt_state: tp.Optional[tp.Any] = None,
		partition_rules: PartitionLike = None,
	) -> tp.Any:
		"""
		Shards the optimizer state according to the provided partition rules.

		Args:
			opt_state (Optional[Any]): The optimizer state to be sharded. If None, the method will use `self.opt_state`.
																Raises a ValueError if both `opt_state` and `self.opt_state` are None.
			partition_rules (Optional[Any]): The partition rules to be used for sharding. If None, the method will use
																			the partition rules from `self.model.config`.

		Returns:
			Any: The sharded optimizer state.

		Raises:
			ValueError: If both `opt_state` and `self.opt_state` are None.
		"""
		if opt_state is None and self.opt_state is None:
			raise ValueError("Optimizer state is not initialized.")
		if opt_state is None:
			opt_state = self.opt_state
		if partition_rules is None:
			partition_rules = self.model.config.get_partition_rules()

		from easydel.escale import make_shard_and_gather_fns, match_partition_rules

		partition_specs = match_partition_rules(partition_rules, opt_state)
		shard_fns, _ = make_shard_and_gather_fns(partition_specs)
		opt_state = jax.tree_util.tree_map(
			lambda f, o: f(o),
			shard_fns,
			opt_state,
		)
		return self.replace(opt_state=opt_state)

	def gather_optimizer_state(self, partition_rules=None):
		assert self.opt_state is not None, "Optimizer state is not initialized."
		if partition_rules is None:
			partition_rules = self.model.config.get_partition_rules()

		from easydel.escale import make_shard_and_gather_fns, match_partition_rules

		partition_specs = match_partition_rules(partition_rules, self.opt_state)
		_, gather = make_shard_and_gather_fns(partition_specs)
		self = self.replace(
			opt_state=jax.tree_util.tree_map(
				lambda f, o: f(o),
				gather,
				self.opt_state,
			)
		)
		return self

	def merge(self, tree) -> EasyDeLBaseModule:
		return nn.merge(self.graphdef, tree, self.graphother)

	def merge_to_state(self, tree) -> EasyDeLState:
		return self.replace(graphstate=tree)

	@property
	def model(self) -> EasyDeLBaseModule:
		return nn.merge(self.graphdef, self.graphstate, self.graphother)

	@property
	def size(self) -> int:
		"""
		Calculates the total size of the optimizer state and model graph state.

		Returns:
		    int: The total size in bytes.
		"""

		def calculate_size(pytree):
			if pytree is None:
				return 0
			leaves, _ = jax.tree_util.tree_flatten(pytree)
			return sum(
				leaf.size * leaf.itemsize
				for leaf in leaves
				if isinstance(leaf, jax.numpy.ndarray)
			)

		opt_state_size = calculate_size(self.opt_state)
		graphstate_size = calculate_size(self.graphstate)
		return opt_state_size + graphstate_size

	def save_state(
		self,
		save_directory: tp.Union[str, os.PathLike],
		float_dtype: tp.Optional[jax.numpy.dtype] = None,
		verbose: bool = True,
		mismatch_allowed: bool = True,
		save_optimizer: bool = True,
		enable: tp.Optional[bool] = None,
	):
		save_directory = pathlib.Path(save_directory)
		self.model.save_pretrained(
			save_directory=str(save_directory),
			gather_fns=self.model._gather_fns,
			float_dtype=float_dtype,
			mismatch_allowed=mismatch_allowed,
			verbose=verbose,
			enable=enable,
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

	def load_state(
		self,
		load_directory: tp.Union[str, os.PathLike],
		verbose: bool = True,
	): ...

	def shard_with_shape(self, shape) -> EasyDeLState:
		"""shard current state with a given shape"""
		from easydel.escale import with_sharding_constraint

		self = nn.from_tree(
			jax.tree_util.tree_map(
				lambda arr, sharding: with_sharding_constraint(
					arr,
					sharding,
				),
				nn.to_tree(self),
				nn.to_tree(shape),
			)
		)
		return self

	def shard_state(self, partition_rules: PartitionLike = None) -> EasyDeLState:
		"""
		Shards the entire state, according to the provided partition rules.

		Args:
			partition_rules (Optional[Any]): The partition rules to be used for sharding. If None, the method will use
																			the partition rules from `self.model.config`.

		Returns:
			EasyDeLState: An updated EasyDeLState object with the sharded state.
		"""

		if self.opt_state is not None:
			self = self.shard_optimizer_state(partition_rules=partition_rules)
		self = self.shard_model(partition_rules=partition_rules)
		return self

	def gather_state(self):
		"""
		Gathers the entire state.

		Returns:
			EasyDeLState: An updated EasyDeLState object with the gathered state.
		"""
		if self.opt_state is not None:
			self = self.gather_optimizer_state()
		self = self.gather_model()
		return self

	def gather_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	) -> EasyDeLState:
		"""
		Gathers the model according to the provided partition rules.

		Returns:
			EasyDeLState: An updated EasyDeLState object with the gathered model.
		"""
		from easydel.escale import make_shard_and_gather_fns, match_partition_rules

		rules = partition_rules or self.model._get_partition_rules(None)
		mesh = mesh or self.model._get_mesh(None)
		partition_specs = match_partition_rules(
			rules=rules,
			tree=self.graphstate,
		)
		_, gather_fns = make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=mesh,
		)
		graphstate = jax.tree_util.tree_map(
			lambda f, o: f(o),
			gather_fns,
			self.graphstate,
		)
		graphother = jax.tree_util.tree_map(
			lambda f, o: f(o),
			gather_fns,
			self.graphother,
		)
		self = self.replace(graphstate=graphstate, graphother=graphother)
		return self

	def shard_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	) -> EasyDeLState:
		"""
		Shards the model according to the provided partition rules.

		Args:
			partition_rules (Optional[Any]): The partition rules to be used for sharding. If None, the method will use
																			the partition rules from `self.model.config`.
			mesh (Optional[Mesh]): The mesh to be used for sharding. If None, the method will use the mesh from `self.model`.

		Returns:
			EasyDeLState: An updated EasyDeLState object with the sharded model.
		"""

		rules = partition_rules or self.model._get_partition_rules(None)
		mesh = mesh or self.model._get_mesh(None)

		def appy_sharding_on_tree(tree):
			from easydel.escale import make_shard_and_gather_fns, match_partition_rules

			partition_specs = match_partition_rules(rules, tree)
			shard_fns, _ = make_shard_and_gather_fns(partition_specs, mesh)
			return jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, tree)

		graphstate = appy_sharding_on_tree(self.graphstate)
		graphother = appy_sharding_on_tree(self.graphother)

		self = self.replace(graphstate=graphstate, graphother=graphother)
		return self

	@property
	def shardings(self):
		"""
		Returns the sharding information for the state.

		Returns:
			Any: The sharding information.
		"""
		return nn.from_tree(
			jax.tree_util.tree_map(
				lambda x: x.sharding if hasattr(x, "sharding") else None,
				nn.to_tree(self),
			)
		)

	def __repr__(self):
		return "EasyDeLState-" + str(self.model)

	__str__ = __repr__
