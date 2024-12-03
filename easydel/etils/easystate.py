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

import copy
import gc
import os
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import fjformer
import fjformer.checkpoint
import jax.tree_util
import optax
from fjformer.dtypes import Array8Bit
from flax import core, struct, traverse_util
from flax.core import FrozenDict
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from safetensors._safetensors_rust import SafetensorError

from easydel.etils.auto_tx import get_optimizer_and_scheduler
from easydel.etils.errors import EasyDeLRuntimeError
from easydel.etils.etils import AVAILABLE_OPTIMIZERS, AVAILABLE_SCHEDULERS, get_logger
from easydel.etils.partition_module import PartitionAxis

logger = get_logger(__name__)
TYPE_SEP = "<*TYPE*>"
VALUE_SEP = "<*VALUE*>"

STRING_REP = "{type}" + TYPE_SEP + "{key}" + VALUE_SEP + "{value}"
DEFAULT_ES_VAL = -1


def revert_type_back(tp, val):
	"""Reverts a value to its original type after deserialization."""
	if tp == "int":
		val = int(val)
	elif tp == "float":
		val = float(val)
	elif tp == "dict":
		val = dict(eval(val))
	elif tp == "bool":
		val = bool(val)
	elif tp == "list":
		val = list(eval(val))
	elif tp == "str":
		val = str(val)
	elif tp == "NoneType":
		val = None
	return val


def break_format(key, value):
	"""Breaks down a serialized key-value pair."""
	k, v = [None] * 2
	if value == DEFAULT_ES_VAL:
		try:
			tp, rs = key.split(TYPE_SEP)
			k, v = rs.split(VALUE_SEP)
			v = revert_type_back(tp=tp, val=v)
			return k, v
		except (KeyError, ValueError):
			...
	else:
		k = key
		v = value
	return k, v


class EasyDeLState(struct.PyTreeNode):
	"""
	**EasyDeLState A Snapshot of Your EasyDeL Model**

	The `EasyDeLState` class acts like a comprehensive container that holds all the essential information about your EasyDeL
	model at a given point in time. Think of it as a snapshot of your model. It includes:

	* **Training Progress:**
	    * `step`: Tracks the current training step.
	* **Model Itself:**
	    * `module`:  Holds the actual instance of your EasyDeL model.
	    * `module_config`: Stores the model's configuration settings.
	    * `module_config_args`:  Keeps track of arguments used to create the configuration (useful for reloading).
	    * `apply_fn`:  References the core function that applies your model to data.
	* **Learned Parameters:**
	    * `params`: Contains the trained weights and biases of your model.
	* **Optimizer Information:**
	    * `tx`: Stores the optimizer you're using to update the model's parameters (e.g., AdamW).
	    * `opt_state`: Keeps track of the optimizer's internal state (this is important for things like momentum in
	      optimizers).
	    * `tx_init`: Remembers the initial settings used to create the optimizer (again, for reloading purposes).
	* **Additional Settings:**
	    * `hyperparameters`:  Provides a flexible place to store other hyperparameters related to your model or training
	      process.

	**Key Capabilities of EasyDeLState:**

	* **Initialization (`create`)**: Lets you create a brand new `EasyDeLState` to start training.
	* **Loading (`load`, `load_state`, `from_pretrained`)**: Enables you to reload a saved model from a checkpoint file or
	  even a pre-trained model from a repository like Hugging Face Hub.
	* **Saving (`save_state`)**: Allows you to save your model's current state, including its parameters and optimizer
	  state.
	* **Optimizer Management (`apply_gradients`, `free_opt_state`, `init_opt_state`)**: Provides methods for updating the
	  model's parameters using gradients, releasing optimizer memory, and re-initializing the optimizer if needed.
	* **Sharding (`shard_params`)**:  Helps you distribute your model's parameters efficiently across multiple devices (
	  important for training large models).
	* **PyTorch Conversion (`to_pytorch`)**:  Gives you a way to convert your EasyDeL model to its PyTorch equivalent.

	**In Essence:**

	`EasyDeLState` streamlines the process of managing, saving, loading, and even converting your EasyDeL models. It ensures
	that you can easily work with your models and maintain consistency throughout your machine learning workflow.

	State of an EasyDeL model, including parameters, optimizer state, and other metadata.

	This class inherits from `flax.struct.PyTreeNode` to support JAX's tree-based data structures.

	Attributes:
	    step (int): Current training step.
	    module (Optional[EasyDeLBaseModule]): An instance of an EasyDeL model.
	    module_config (Optional[EasyDeLBaseConfig]): Configuration of the EasyDeL model.
	    module_config_args (Optional[dict]): Dictionary of arguments used to initialize the model configuration.
	    apply_fn (Callable): Function to apply the model to input data.
	    params (core.FrozenDict[str, Any]): Model parameters, stored as a frozen dictionary.
	    tx (optax.GradientTransformation): Optimizer used for training.
	    opt_state (Optional[optax.OptState]): Optimizer state.
	    tx_init (Optional[dict]): Dictionary containing optimizer and scheduler initialization parameters.
	    hyperparameters (Optional[dict]): Dictionary to store any additional hyperparameters.

	Example Usage::


	>>> # Initialize an EasyDeLState object
	>>> state = EasyDeLState.create(
	...    apply_fn=model.__call__,
	...    params=model.params,
	...    tx=optax.adam(learning_rate=1e-3),
	...     module=model,
	...     module_config=model.config
	>>> )

	"""

	step: int
	module: Optional[
		"easydel.modules.modeling_utils.EasyDeLBaseModule"  # type:ignore # noqa
	] = struct.field(pytree_node=False)
	module_config: Optional[
		"easydel.modules.modeling_utils.EasyDeLBaseConfig"  # type:ignore # noqa
	] = struct.field(pytree_node=False)
	module_config_args: Optional[dict] = struct.field(pytree_node=True)
	apply_fn: Callable = struct.field(pytree_node=False)
	params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
	tx: optax.GradientTransformation = struct.field(pytree_node=False)
	opt_state: Optional[optax.OptState] = struct.field(pytree_node=True)
	tx_init: Optional[dict] = struct.field(pytree_node=True)
	hyperparameters: Optional[dict] = struct.field(pytree_node=True)

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
		if OVERWRITE_WITH_GRADIENT in grads:
			grads_with_opt = grads["params"]
			params_with_opt = self.params["params"]
		else:
			grads_with_opt = grads
			params_with_opt = self.params

		updates, new_opt_state = self.tx.update(
			grads_with_opt, self.opt_state, params_with_opt
		)
		new_params_with_opt = optax.apply_updates(params_with_opt, updates)
		if OVERWRITE_WITH_GRADIENT in grads:
			new_params = {
				"params": new_params_with_opt,
				OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
			}
		else:
			new_params = new_params_with_opt
		return self.replace(
			step=self.step + 1,
			params=new_params,
			opt_state=new_opt_state,
			**kwargs,
		)

	@classmethod
	def create(
		cls,
		*,
		apply_fn: Callable,
		params: Union[core.FrozenDict[str, Any], Mapping[str, Any]],
		tx: Optional[optax.GradientTransformation] = None,
		tx_init: Optional[dict] = None,
		hyperparameters: Optional[dict] = None,
		module: Optional["EasyDeLBaseModule"] = None,  # type:ignore #noqa
		module_config: Optional["EasyDeLBaseConfig"] = None,  # type:ignore #noqa
		module_config_args: Optional[dict] = None,
		**kwargs,
	):
		"""
		Creates a new EasyDeLState object.

		This class method is used to initialize an EasyDeLState object from model parameters, optimizer, and other configurations.

		Args:
		    apply_fn (Callable): A function that applies the model to a batch of data.
		    params (Union[core.FrozenDict[str, Any], Mapping[str, Any]]): Model parameters.
		    tx (optax.GradientTransformation): An optax optimizer.
		    tx_init (Optional[dict]): A dictionary of optimizer initialization parameters.
		    hyperparameters (Optional[dict]): A dictionary of additional hyperparameters.
		    module (Optional[EasyDeLBaseModule]): An instance of an EasyDeL model.
		    module_config (Optional[EasyDeLBaseConfig]): An instance of an EasyDeL model configuration.
		    module_config_args (Optional[dict]): A dictionary of arguments used to initialize the model configuration.
		    **kwargs: Additional keyword arguments.

		Returns:
		    EasyDeLState: A new EasyDeLState object.
		"""
		if hyperparameters is None:
			hyperparameters = {}
		params_with_opt = params["params"] if OVERWRITE_WITH_GRADIENT in params else params
		if tx is None:
			logger.warn(
				"`tx` is set to None in case that you want to add a tx. (use `state.replace_tx`)"
			)
		opt_state = tx.init(params_with_opt) if tx is not None else None
		if module_config is not None:
			module_config = copy.deepcopy(module_config)
			cls.safe_dict(module_config.__dict__)
		return cls(
			step=0,
			apply_fn=apply_fn,
			module=module,
			params=params,
			tx=tx,
			opt_state=opt_state,
			tx_init=cls.safe_dict(tx_init or {}),
			hyperparameters=hyperparameters,
			module_config=module_config,
			module_config_args=None,
			**kwargs,
		)

	def replace_tx(self, tx: optax.GradientTransformation):
		"""
		replaces or set a new optimizer/tx
		"""
		return self.replace(tx=tx)

	@classmethod
	def load(
		cls,
		*,
		apply_fn: Callable,
		params: Union[core.FrozenDict[str, Any], Mapping[str, Any]],
		step: int = 0,
		opt_state: Optional[optax.OptState] = None,
		tx_init: Optional[dict] = None,
		hyperparameters: Optional[dict] = None,
		module: Optional["EasyDeLBaseModule"] = None,  # type:ignore #noqa
		module_config: Optional["EasyDeLBaseConfig"] = None,  # type:ignore #noqa
		module_config_args: Optional[dict] = None,
		**kwargs,
	):
		"""
		Loads an EasyDeLState object from a checkpoint.

		This class method is used to load a previously saved EasyDeLState object from a checkpoint file. This can be useful for resuming training or inference.

		Args:
		    apply_fn (Callable): A function that applies the model to a batch of data.
		    params (Union[core.FrozenDict[str, Any], Mapping[str, Any]]): Model parameters.
		    step (int, optional): The training step to resume from. Defaults to 0.
		    opt_state (Optional[optax.OptState], optional): The optimizer state to load. Defaults to None.
		    tx_init (Optional[dict], optional): A dictionary of optimizer initialization parameters. Defaults to None.
		    hyperparameters (Optional[dict], optional): A dictionary of additional hyperparameters. Defaults to None.
		    module (Optional[EasyDeLBaseModule], optional): An instance of an EasyDeL model. Defaults to None.
		    module_config (Optional[EasyDeLBaseConfig], optional): An instance of an EasyDeL model configuration. Defaults to None.
		    module_config_args (Optional[dict], optional): A dictionary of arguments used to initialize the model configuration. Defaults to None.
		    **kwargs: Additional keyword arguments.

		Returns:
		    EasyDeLState: A loaded EasyDeLState object.
		"""
		if module_config is not None:
			module_config = copy.deepcopy(module_config)

		if tx_init is None:
			tx_init = {}
		tx_init = copy.deepcopy(tx_init)
		tx_init = cls.unsafe_dict(tx_init)

		tx_init["optimizer"] = cls.search("optimizer", tx_init, "adamw")
		tx_init["scheduler"] = cls.search("scheduler", tx_init, "none")
		tx_init["steps"] = cls.search("steps", tx_init, 1e6)

		def fix_dict_types(input_dict):
			fixed_dict = input_dict.copy()

			# Fix extra_optimizer_kwargs
			if "extra_optimizer_kwargs" in fixed_dict:
				fixed_dict["extra_optimizer_kwargs"] = eval(
					fixed_dict["extra_optimizer_kwargs"]
				)

			# Fix gradient_accumulation_steps
			if "gradient_accumulation_steps" in fixed_dict:
				fixed_dict["gradient_accumulation_steps"] = int(
					fixed_dict["gradient_accumulation_steps"]
				)

			# Fix steps
			if "steps" in fixed_dict:
				fixed_dict["steps"] = int(fixed_dict["steps"])

			# Fix warmup_steps
			if "warmup_steps" in fixed_dict:
				fixed_dict["warmup_steps"] = int(fixed_dict["warmup_steps"])

			return fixed_dict

		try:
			tx, sc = get_optimizer_and_scheduler(**tx_init)
		except TypeError:
			tx, sc = get_optimizer_and_scheduler(**fix_dict_types(tx_init))
		if hyperparameters is None:
			hyperparameters = {}

		if module_config is not None:
			hyperparameters = cls.create_hyperparameters(module_config.model_type)
			cls.safe_dict(module_config.__dict__)
		return cls(
			step=step,
			apply_fn=apply_fn,
			params=params,
			tx=tx,
			opt_state=opt_state,
			tx_init=cls.safe_dict(tx_init),
			hyperparameters=hyperparameters,
			module=module,
			module_config=module_config,
			module_config_args=None,
			**kwargs,
		)

	@classmethod
	def load_state(
		cls,
		checkpoint_path: Union[str, os.PathLike],
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[str, jax.lax.Precision]] = None,
		init_optimizer_state: bool = False,
		state_shard_fns: Optional[Mapping[str, Callable]] = None,
		verbose: bool = False,
		input_shape: Tuple = (1, 1),
		config_kwargs: Optional[dict] = None,
		sharding_axes_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		sharding_axes_dims: Sequence[int] = (1, -1, 1, 1),
		module_config: Optional["EasyDeLBaseConfig"] = None,  # type:ignore #noqa
		safe: bool = False,
		auto_shard_state: bool = False,
		partition_rules: Optional[Tuple[Tuple[str, PartitionSpec]]] = None,
		depth_target: Optional[List[str]] = None,
	):
		"""
		Loads an EasyDeLState object from a checkpoint file.

		This class method is used to load a pre-trained EasyDeL model and its associated state from a checkpoint file.

		Args:
		    checkpoint_path (Union[str, os.PathLike]): Path to the checkpoint file.
		    dtype (jnp.dtype, optional): The data type to use for the model parameters. Defaults to jnp.float32.
		    param_dtype (jnp.dtype, optional): The data type to use for the model parameters during training. Defaults to jnp.float32.
		    precision (Optional[Union[str, jax.lax.Precision]], optional): The precision to use for computations. Defaults to None.
		    init_optimizer_state (bool, optional): Whether to initialize the optimizer state. Defaults to False.
		    state_shard_fns (Optional[Mapping[str, Callable]], optional): Functions to shard the model state. Defaults to None.
		    verbose (bool, optional): Whether to print verbose output during loading. Defaults to False.
		    input_shape (Tuple, optional): The shape of the input data. Defaults to (1, 1).
		    config_kwargs (Optional[dict], optional): Keyword arguments to pass to the model configuration. Defaults to None.
		    sharding_axes_names (Sequence[str], optional): Names of the axes for sharding. Defaults to ("dp", "fsdp", "tp", "sp").
		    sharding_axes_dims (Sequence[int], optional): Dimensions of the axes for sharding. Defaults to (1, -1, 1, 1).
		    module_config (Optional[EasyDeLBaseConfig], optional): An instance of an EasyDeL model configuration. Defaults to None.
		    auto_shard_state (bool, optional): Whether to automatically shard the model state. Defaults to False.
		    partition_rules (Optional[Tuple[Tuple[str, PartitionSpec]]], optional): Rules for partitioning the model parameters. Defaults to None.
		    depth_target (Optional[List[str]], optional): Target depth for partitioning. Defaults to None.
		    safe (bool): whenever to load your model with safetensors and your checkpoints are saved with safe=True. Defaults to True.
		Returns:
		    EasyDeLState: A loaded EasyDeLState object.
		"""
		if depth_target is None:
			depth_target = ["params", "params"]
		from fjformer.sharding import create_mesh

		from easydel.modules.auto_causal_language_model import (
			AutoShardAndGatherFunctions,
			get_modules_by_type,
		)

		mesh = create_mesh(sharding_axes_dims, sharding_axes_names)
		if auto_shard_state:
			assert module_config is not None, (
				"`module_config` is None, in case of using"
				" `auto_shard_state=True` you should pass Module Config"
			)
			state_shard_fns, state_gather_fns = AutoShardAndGatherFunctions.from_config(
				config=module_config,
				partition_rules=partition_rules,
				input_shape=input_shape,  # type:ignore
				flatten=False,
				depth_target=depth_target,
			)

		with mesh:
			if safe:
				try:
					checkpoint, _ = fjformer.CheckpointManager.load_checkpoint_safe(
						path=checkpoint_path,
						shard_fns=state_shard_fns,
						verbose=verbose,
					)
				except SafetensorError as e:
					raise SafetensorError(
						e + " Make Sure your checkpoint file saved with "
					) from None

			else:
				checkpoint = fjformer.CheckpointManager.load_checkpoint(
					path=checkpoint_path,
					shard_fns=state_shard_fns,
					verbose=verbose,
				)

			checkpoint["params"] = flatten_dict(checkpoint["params"])
			for k in list(checkpoint["params"].keys()):
				# The data conversion is not performed here because it causes double memory allocation
				# until the loop is completed. This can be problematic for GPUs with limited VRAM
				# (e.g., 24GB), making it crucial to avoid such heavy loop allocations.
				#
				# The following commented-out code would convert the data types, but it's avoided
				# due to the above-mentioned reason:
				# checkpoint["params"] = jax.tree_util.tree_map(
				#     lambda x: (
				#         jax.lax.convert_element_type(x, param_dtype)
				#         if (hasattr(x, "dtype") and x.dtype != param_dtype)
				#         else x
				#     ),
				#     checkpoint["params"],
				# )

				x = checkpoint["params"][k]
				if hasattr(x, "dtype") and x.dtype != param_dtype:
					x_converted = jax.lax.convert_element_type(x, param_dtype)
					checkpoint["params"][k] = x_converted
					del x
					gc.collect()

			checkpoint["params"] = unflatten_dict(checkpoint["params"])
			hyperparameters = checkpoint.get("hyperparameters")
			cfg, module, convertor = get_modules_by_type(
				model_type=cls.get_model_type(hyperparameters)
			)
			checkpoint.pop("module_config", None)
			if checkpoint["module_config_args"] is not None:
				cfg_behave = cls.unsafe_dict(checkpoint.get("module_config_args", {}))
				cfg_behave.pop("id2label", None)
				cfg_behave.pop("label2id", None)
				cfg_behave.pop("torch_dtype", None)
				for k, v in cfg_behave.items():
					if v is None:
						cfg_behave.pop(k, None)
					elif v == "None":
						cfg_behave[k] = None
					elif isinstance(v, str):
						if v.startswith("{") or v.startswith("(") or v.startswith("PartitionSpec"):
							cfg_behave[k] = eval(v)
				module_config = cfg.from_dict(cfg_behave)
				if config_kwargs is not None:
					for k, v in config_kwargs.items():
						setattr(module_config, k, v)
				module_in = module(
					config=module_config,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					input_shape=input_shape,
					_do_init=False,
				)
			else:
				raise TypeError("Om seems like i couldn't read model correctly ;(")
			state = cls.load(
				apply_fn=module_in.__call__,
				module=module_in,
				module_config=module_config,
				**checkpoint,
			)
			state = state.replace(
				module_config_args=None  # removing because it's not needed anymore
			)
			if init_optimizer_state:
				state = state.init_opt_state()
		return state

	@classmethod
	def get_model_type(cls, dictionary):
		return cls.find_key("model_type", dictionary)

	def save_state(
		self,
		filename: Union[str, os.PathLike],
		save_optimizer: bool = False,
		checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
		verbose: bool = False,
		safe: bool = False,
		gather_fns: dict[Callable] = None,
		float_dtype: Union[str, jax.numpy.dtype] = None,
	):
		"""
		Saves the EasyDeLState object to a checkpoint file.

		Args:
		    filename (Union[str, os.PathLike]): The name of the checkpoint file.
		    save_optimizer (bool, optional): Whether to save the optimizer state. Defaults to False.
		    checkpoint_dir (Optional[Union[str, os.PathLike]], optional): The directory to save the checkpoint to. Defaults to None.
		    verbose (bool, optional): Whether to print verbose output during saving. Defaults to False.
		    gather_fns (dict[Callable], optional): Functions to gather sharded parameters. Defaults to None.
		    float_dtype (Union[str, jax.numpy.dtype], optional): The data type to use for floating-point numbers in the saved checkpoint. Defaults to None.
		    safe (bool): whenever to load your model with safetensors and your checkpoints are saved with safe=True. Defaults to True.
		"""
		state = self
		if not save_optimizer:
			state = self.replace(opt_state=None)
		state = state.replace(
			module_config_args={
				k: v
				for k, v in state.module.config.__dict__.items()
				if isinstance(v, (int, bool, float))
			}
		)
		if safe:
			fjformer.CheckpointManager.save_checkpoint_safe(
				state=state,
				path=(
					os.path.join(checkpoint_dir, filename)
					if checkpoint_dir is not None
					else filename
				),
				verbose=verbose,
				gather_fns=gather_fns,
				float_dtype=float_dtype,
			)
		else:
			fjformer.CheckpointManager.save_state_to_file(
				state=state,
				path=(
					os.path.join(checkpoint_dir, filename)
					if checkpoint_dir is not None
					else filename
				),
				verbose=verbose,
				gather_fns=gather_fns,
				float_dtype=float_dtype,
			)

	def free_opt_state(self) -> "EasyDeLState":
		"""
		Frees the memory allocated for the optimizer state.

		Returns:
		    EasyDeLState: A new EasyDeLState object with the optimizer state set to None.
		"""
		return self.replace(opt_state=None)

	def init_opt_state(self) -> "EasyDeLState":
		"""
		Initializes the optimizer state.

		Returns:
		    EasyDeLState: A new EasyDeLState object with an initialized optimizer state.
		"""
		assert (
			self.tx is not None
		), "`tx` is set to None you have to first add an optimizer."
		if self.opt_state is None:
			params_with_opt = (
				self.params["params"] if OVERWRITE_WITH_GRADIENT in self.params else self.params
			)
			opt_state = self.tx.init(params_with_opt)

			return self.replace(opt_state=opt_state)
		return self

	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		filename: Optional[str] = None,
		optimizer: AVAILABLE_OPTIMIZERS = "adamw",
		scheduler: AVAILABLE_SCHEDULERS = "none",
		tx_init: Optional[dict] = None,
		device: Optional[jax.Device] = None,
		dtype: jax.numpy.dtype = jax.numpy.float32,
		param_dtype: jax.numpy.dtype = jax.numpy.float32,
		precision: Optional[jax.lax.Precision] = None,
		sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
		sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
		partition_axis: Optional[PartitionAxis] = None,
		shard_attention_computation: bool = True,
		auto_shard_params: bool = False,
		input_shape: Sequence[int] = (1, 1),
		backend: Optional[str] = None,
		init_optimizer_state: bool = False,
		free_optimizer_state: bool = True,
		verbose: bool = True,
		state_shard_fns: Optional[Mapping[str, Callable]] = None,
		config_kwargs: Optional[Mapping[str, Any]] = None,
		**kwargs,
	) -> "EasyDeLState":
		"""
		Loads a pre-trained EasyDeL model and its state.

		This class method can load a model from either a local checkpoint or a remote repository like Hugging Face Hub.

		Args:
		    pretrained_model_name_or_path (str): The name of the pre-trained model or the path to the local checkpoint.
		    filename (Optional[str], optional): The name of the file to load from Hugging Face Hub. Defaults to None.
		    optimizer (AVAILABLE_OPTIMIZERS, optional): The optimizer to use for training. Defaults to "adamw".
		    scheduler (AVAILABLE_SCHEDULERS, optional): The learning rate scheduler to use during training. Defaults to "none".
		    tx_init (Optional[dict], optional): A dictionary of optimizer initialization parameters. Defaults to None.
		    device (optional): The device to load the model on. Defaults to None -> jax.devices('cpu')[0].
		    dtype (jax.numpy.dtype, optional): The data type to use for the model parameters. Defaults to jax.numpy.float32.
		    param_dtype (jax.numpy.dtype, optional): The data type to use for the model parameters during training. Defaults to jax.numpy.float32.
		    precision (Optional[jax.lax.Precision], optional): The precision to use for computations. Defaults to jax.lax.Precision("fastest").
		    sharding_axis_dims (Sequence[int], optional): The dimensions of the axes for sharding. Defaults to (1, -1, 1, 1).
		    sharding_axis_names (Sequence[str], optional): The names of the axes for sharding. Defaults to ("dp", "fsdp", "tp", "sp").
		    partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
		    shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
		    auto_shard_params (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
		    input_shape (Sequence[int], optional): The shape of the input data. Defaults to (1, 1).
		    backend (Optional[str], optional): The backend to use for computations. Defaults to None.
		    init_optimizer_state (bool, optional): Whether to initialize the optimizer state. Defaults to False.
		    free_optimizer_state (bool, optional): Whether to free the optimizer state after loading. Defaults to True.
		    verbose (bool, optional): Whether to print verbose output. Defaults to True.
		    state_shard_fns (Optional[Mapping[str, Callable]], optional): Functions to shard the model state. Defaults to None.
		    config_kwargs (Optional[Mapping[str, Any]], optional): Keyword arguments to pass to the model configuration. Defaults to None.
		    **kwargs: Additional keyword arguments.

		Returns:
		    EasyDeLState: A loaded EasyDeLState object.

		Raises:
		    EasyDeLRuntimeError: If both `free_optimizer_state` and `init_optimizer_state` are set to True.
		"""
		if device is None:
			device = jax.devices("cpu")[0]
		if precision is None:
			precision = jax.lax.Precision("fastest")
		if partition_axis is None:
			partition_axis = PartitionAxis()
		if free_optimizer_state and init_optimizer_state:
			raise EasyDeLRuntimeError(
				"You can't use `free_optimizer_state` and `init_optimizer_state` True at same Time"
			)

		if filename is None:
			from easydel.modules.auto_causal_language_model import AutoEasyDeLModelForCausalLM

			model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
				pretrained_model_name_or_path,
				device=device,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				sharding_axis_dims=sharding_axis_dims,
				sharding_axis_names=sharding_axis_names,
				partition_axis=partition_axis,
				shard_attention_computation=shard_attention_computation,
				input_shape=input_shape,  # type:ignore
				backend=backend,
				config_kwargs=config_kwargs,
				auto_shard_params=auto_shard_params,
				**kwargs,
			)
			if tx_init is None:
				tx_init = {}

			tx_init["optimizer"] = optimizer
			tx_init["scheduler"] = scheduler

			state = cls.load(
				apply_fn=model.__call__,
				params=FrozenDict({"params": params}),
				step=0,
				opt_state=None,
				tx_init=tx_init,
				hyperparameters=None,
				module=model,
				module_config=model.config,
				module_config_args=model.config.to_dict(),
			)
		else:
			with jax.default_device(device):
				from huggingface_hub import hf_hub_download

				checkpoint_path = hf_hub_download(
					repo_id=pretrained_model_name_or_path,
					filename=filename,
				)
				state = cls.load_state(
					checkpoint_path=checkpoint_path,
					init_optimizer_state=init_optimizer_state,
					verbose=verbose,
					state_shard_fns=state_shard_fns,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					input_shape=input_shape,  # type: ignore
				)
		if init_optimizer_state:
			with jax.default_device(device):
				state = state.init_opt_state()
		if free_optimizer_state:
			state = state.free_opt_state()
		return state

	def to_8bit(self, quantization_fields=None):
		if quantization_fields is None:
			quantization_fields = ["kernel"]

		def quantize_params(params: dict) -> dict:
			"""Quantizes model parameters using Array8Bit.

			Args:
			    params: A dictionary of model parameters.

			Returns:
			    A dictionary of quantized model parameters.
			"""

			def q(path: str, array: Any) -> Array8Bit:
				"""Quantizes a single parameter array."""
				path = [p for p in path[0].key]
				for field in quantization_fields:
					if field in path:
						return Array8Bit.quantize(array, qk=64)
				return array

			return traverse_util.unflatten_dict(
				jax.tree_util.tree_map_with_path(
					q,
					traverse_util.flatten_dict(params),
				)
			)

		self = self.replace(params=quantize_params(self.params))  # type:ignore #noqa
		return self

	def shard_params(
		self,
		fully_sharded_data_parallel: bool = True,
		shard_fns: Optional[Mapping[str, Callable]] = None,
		dtype: Union[jax.numpy.dtype, str] = "bf16",
		mesh: Optional[Mesh] = None,
		rules: Optional[Sequence[Mapping[str, PartitionSpec]]] = None,
	):
		"""
		Shards the model parameters across devices.

		Args:
		    fully_sharded_data_parallel (bool, optional): Whether to use fully sharded data parallelism. Defaults to True.
		    shard_fns (Optional[Mapping[str, Callable]], optional): Functions to shard the model parameters. Defaults to None.
		    dtype (Union[jax.numpy.dtype, str], optional): The data type to use for sharded parameters. Defaults to "bf16".
		    mesh (Optional[Mesh], optional): The JAX mesh to use for sharding. Defaults to None.
		    rules (Optional[Sequence[Mapping[str, PartitionSpec]]], optional): Rules for partitioning the model parameters. Defaults to None.

		Returns:
		    EasyDeLState: An EasyDeLState object with sharded parameters.

		Raises:
		    EasyDeLRuntimeError: If `shard_fns` and `rules` are both None and the model does not have a `module_config`.
		"""
		dtype = fjformer.checkpoint.get_dtype(dtype)

		if mesh is None:
			mesh = self.module_config.mesh
		assert mesh is not None, "consider passing mesh"
		with mesh:
			if shard_fns is None and self.module_config is None and rules is None:
				raise EasyDeLRuntimeError(
					"the model doesn't carrying `module_config` you should pass `shard_fns` or `rules`"
				)
			elif shard_fns is None and rules is not None or self.module_config is not None:
				from fjformer import make_shard_and_gather_fns, match_partition_rules

				rules = rules or self.module_config.get_partition_rules(
					fully_sharded_data_parallel
				)
				partition_specs = match_partition_rules(
					rules=rules,
					params=self.params,
				)
				shard_fns, gather_fns = make_shard_and_gather_fns(
					partition_specs=partition_specs,
					mesh=mesh,
				)
				return self.replace(
					params=jax.tree_util.tree_map(
						lambda f, p: f(p),
						shard_fns,
						self.params,
					)
				)

	@staticmethod
	def create_hyperparameters(model_type: str):
		"""it's the only way we can dump xla compiler"""
		return {
			STRING_REP.format(type="str", key="model_type", value=model_type): DEFAULT_ES_VAL
		}

	@staticmethod
	def safe_dict(dictionary: dict):
		for k in list(dictionary.keys()):
			val = dictionary.get(k)
			if not isinstance(val, (int, bool)):
				val = dictionary.pop(k)
				string_value_format = STRING_REP.format(
					type=type(val).__name__, key=k, value=val
				)
				dictionary[string_value_format] = DEFAULT_ES_VAL
		return dictionary

	@staticmethod
	def unsafe_dict(dictionary: dict):
		result = {}
		for k in list(dictionary.keys()):
			if VALUE_SEP in k and TYPE_SEP in k:
				v = dictionary[k]
				key, value = break_format(key=k, value=v)
				result[key] = value
			else:
				result[k] = dictionary[k]
		return result

	def __str__(self):
		"""The __str__ function is called when you call str(object) or print(object).
		The __repr__ function is called when you type the object name in the interpreter.
		If no __str__ method exists, Python will use __repr__ as a fallback.

		Args:
		    self: Refer to the object itself

		Returns:
		    string
		"""
		params_size = sum(
			getattr(n, "size", 0) for n in jax.tree_util.tree_flatten(self.params)[0]
		)
		opt_state_size = sum(
			getattr(n, "size", 0) for n in jax.tree_util.tree_flatten(self.opt_state)[0]
		)

		def make_depth(mdl=None):
			if mdl is not None:
				try:
					return (
						mdl.__str__().replace("\n", "\n\t" "") if hasattr(mdl, "__str__") else None
					)
				except TypeError:
					...
			return mdl

		optimizer = self.tx_init.get("optimizer", None)
		scheduler = self.tx_init.get("scheduler", None)

		if optimizer is None:
			optimizer = self.find_key("optimizer", self.tx_init)
		if scheduler is None:
			scheduler = self.find_key("scheduler", self.tx_init)

		string = (
			f"{self.__class__.__name__}("
			f"\n\tstep = {self.step}"
			f"\n\tmodule = {make_depth(self.module.__class__.__name__)}"
			f"\n\tmodule_config = {make_depth(self.module_config)}"
			f"\n\tapply_fn: Callable = {make_depth(self.apply_fn)}"
			f"\n\tparams : {params_size} Parameters"
			f"\n\ttx = {optimizer} Optimizer with {scheduler} Scheduler"
			f"\n\topt_state : {opt_state_size} Parameters"
			f"\n\thyperparameters : {self.hyperparameters}"
			f"\n)"
		)
		return string

	@classmethod
	def search(cls, key, dictionary: dict, default: Any = None):
		"""
		Searches for a key in a dictionary, handling serialized keys.

		Args:
		    key: The key to search for.
		    dictionary (dict): The dictionary to search within.
		    default: The default value to return if the key is not found.

		Returns:
		    The value associated with the key, or the default value if the key is not found.
		"""
		req = dictionary.get(key, None)
		if req is None:
			req = cls.find_key(key, dictionary)
		return req or default

	@staticmethod
	def find_key(key, dictionary: dict) -> Union[str, None]:
		"""
		Finds a key within a dictionary, considering serialized keys.

		This method iterates through the dictionary, checking for both regular and serialized keys.

		Args:
		    key: The key to find.
		    dictionary (dict): The dictionary to search.

		Returns:
		    The value associated with the key, or None if not found.
		"""
		result = None
		for k, v in dictionary.items():
			k_, v_ = break_format(key=k, value=v)
			if k_ == key:
				result = v_
				break
		return result

	def serialize(self):
		"""
		Prepares the state for JAX sharding and saving by serializing specific attributes.

		This method modifies the `module_config` in-place, converting non-trivial data types into
		strings to ensure compatibility with JAX sharding and checkpointing mechanisms.
		"""
		for k, v in self.safe_dict(self.module_config.__dict__).items():
			setattr(self.module_config, k, v)

	def un_serialize(self):
		"""
		Reverses the serialization process, restoring the original data types of `module_config` attributes.

		This method iterates through the `module_config` attributes, identifying and converting
		serialized string values back to their appropriate data types.
		"""
		for k, v in self.unsafe_dict(self.module_config.__dict__).items():
			setattr(self.module_config, k, v)

	def to_pytorch(
		self,
		base_hf_auto_class=None,
		easystate_to_huggingface_model_kwargs: Optional[dict] = None,
	):
		"""
		Converts the EasyDeL model to a PyTorch model.

		Args:
		    base_hf_auto_class: The base Hugging Face AutoModel class to use for conversion.
		    easystate_to_huggingface_model_kwargs: Keyword arguments to pass to the conversion method.

		Returns:
		    A PyTorch model equivalent to the EasyDeL model.
		"""
		if base_hf_auto_class is None:
			from transformers import AutoModelForCausalLM as base_hf_auto_class
		return self.module.to_pytorch(
			params=self.params,
			base_hf_auto_class=base_hf_auto_class,
			easystate_to_huggingface_model_kwargs=easystate_to_huggingface_model_kwargs,
		)

	def __repr__(self):
		"""The __repr__ function is the &quot;official&quot; string representation of an object.
		It's what you get when you type the object name at the Python prompt, or pass it to str().
		The goal of __repr__ is to be unambiguous: if eval(repr(x)) == x, then __repr__ should return a string that
		looks like a valid Python expression that could be used to recreate an object with the same value (
		given an appropriate environment). If this is not possible, a string formatted using %s
		formatting is also acceptable.

		Args:
		    self: Represent the instance of the class

		Returns:
		    A string that is a valid python expression
		"""
		return self.__str__()

	def generate(
		self,
		input_ids: jnp.ndarray,
		generation_config: Optional[
			"transformers.GenerationConfig"  # noqa # type:ignore
		] = None,
		prng_key: Optional[jnp.ndarray] = None,
		trace: bool = True,
		logits_processor: Optional[
			"transformers.FlaxLogitsProcessorList"  # noqa # type:ignore
		] = None,
		**kwargs,
	):
		r"""
		Generates sequences of token ids for models with a language modeling head.

		Parameters:
		    input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
		        The sequence used as a prompt for the generation.
		    generation_config (`~generation.GenerationConfig`, *optional*):
		        The generation configuration to be used as base parametrization for the generation call. `**kwargs`
		        passed to generate matching the attributes of `generation_config` will override them. If
		        `generation_config` is not provided, the default will be used, which had the following loading
		        priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
		        configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
		        default values, whose documentation should be checked to parameterize generation.
		    trace (`bool`, *optional*, defaults to `True`):
		        Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
		        considerably slower runtime.
		    params (`Dict[str, jnp.ndarray]`, *optional*):
		        Optionally the model parameters can be passed. Can be useful for parallelized generation.
		    logits_processor (`FlaxLogitsProcessorList `, *optional*):
		        Custom logits processors that complement the default logits processors built from arguments and
		        generation config. If a logit processor is passed that is already created with the arguments or a
		        generation config an error is thrown. This feature is intended for advanced users.
		    kwargs (`Dict[str, Any]`, *optional*):
		        Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
		        forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
		        specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

		Return:
		    [`~utils.ModelOutput`].

		"""
		params = self.params.get("params", None)
		if params is None:
			params = self.params
		return self.module.generate(
			input_ids=input_ids,
			generation_config=generation_config,
			prng_key=prng_key,
			trace=trace,
			params={"params": params},
			logits_processor=logits_processor,
			**kwargs,
		)
