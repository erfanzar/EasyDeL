import copy
import os
from typing import Any, Callable, Optional, Mapping, Sequence, Tuple

import fjformer
import jax.tree_util
import termcolor
from flax import core
from flax import struct
from flax.core import FrozenDict
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
import optax
from .auto_tx import get_optimizer_and_scheduler
from ..etils import AVAILABLE_SCHEDULERS, AVAILABLE_OPTIMIZERS, EasyDelRuntimeError
from ..modules.easydel_modelling_utils import EasyDelFlaxPretrainedModel, EasyDelPretrainedConfig
from jax.sharding import Mesh, PartitionSpec

TYPE_SEP = "<*TYPE*>"
VALUE_SEP = "<*VALUE*>"

STRING_REP = "{type}" + TYPE_SEP + "{key}" + VALUE_SEP + "{value}"
DEFAULT_ES_VAL = -1


def revert_type_back(tp, val):
    # assert tp in ["int", "float", "dict", "bool", "list", "str"], f"type {tp} not supported"
    if tp == "int":
        val = int(val)
    elif tp == "float":
        val = float(val)
    elif tp == "dict":
        val = dict(val)
    elif tp == "bool":
        val = bool(val)
    elif tp == "list":
        val = list(val)
    elif tp == "str":
        val = str(val)
    return val


def break_format(key, value):
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


class EasyDelState(struct.PyTreeNode):
    step: int
    module: Optional[EasyDelFlaxPretrainedModel] = struct.field(pytree_node=False)
    module_config: Optional[EasyDelPretrainedConfig] = struct.field(pytree_node=False)
    module_config_args: Optional[dict] = struct.field(pytree_node=True)
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = struct.field(pytree_node=True)
    tx_init: Optional[dict] = struct.field(pytree_node=True)
    hyperparameters: Optional[dict] = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, **kwargs):

        """
        The apply_gradients function is the core of the optimizer. It takes in a dictionary of gradients,
        and returns an updated version of itself with new parameters and state. The function also updates
        the step count.

        :param self: Refer to the current instance of the class
        :param *: Unpack the grads dictionary into positional arguments
        :param grads: Pass in the gradients of the loss function with respect to each parameter
        :param kwargs: Pass in additional arguments to the function
        :return: A new State with the updated parameters and params
        """
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                'params': new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT]
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
            params: core.FrozenDict[str, Any] | Mapping[str, Any],
            tx: optax.GradientTransformation,
            tx_init: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            module: Optional[EasyDelFlaxPretrainedModel] = None,
            module_config: Optional[EasyDelPretrainedConfig] = None,
            module_config_args: Optional[dict] = None,
            **kwargs
    ):

        """
        The create function is used to create a new instance of the class.

        :param cls: Create a new instance of the class
        :param *: Pass a list of parameters to the function
        :param apply_fn: Callable: Apply the model to a batch of data
        :param params: core.FrozenDict[str,Any] | Mapping[str,Any]: Pass in the parameters of the model
        :param tx: optax.GradientTransformation: Initialize the optimizer
        :param tx_init: Optional[dict]: Initialize the optimizer
        :param hyperparameters: Optional[dict]: Pass hyperparameters to the state for init
        :param module: Optional[EasyDelFlaxPretrainedModel]: Pass the module to be used int state
        :param module_config: Optional[EasyDelPretrainedConfig]: Pass in the module config
        :param module_config_args: Optional[dict]: Store the config args of the model
        :param kwargs: Pass in additional parameters to the
        :return: A EasyDelState object
        """
        if hyperparameters is None:
            hyperparameters = {}
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_state = tx.init(params_with_opt)
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
            tx_init=cls.safe_dict(tx_init),
            hyperparameters=hyperparameters,
            module_config=module_config,
            module_config_args=None,
            **kwargs,
        )

    @classmethod
    def load(
            cls,
            *,
            apply_fn: Callable,
            params: core.FrozenDict[str, Any] | Mapping[str, Any],
            step: int = 0,
            opt_state: Optional[optax.OptState] = None,
            tx_init: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            module: Optional[EasyDelFlaxPretrainedModel] = None,
            module_config: Optional[EasyDelPretrainedConfig] = None,
            module_config_args: Optional[dict] = None,
            **kwargs
    ):

        """
        The load function is used to load a saved state of the Model and optimizer or Model Only.

        :param cls: Make the function a class method
        :param *: Pass in a variable number of arguments
        :param step: int: Keep track of the number of steps that have been taken
        :param apply_fn: Callable: Apply the optimizer to the model
        :param params: core.FrozenDict[str,Any] | Mapping[str,Any]: Pass in the parameters of the model
        :param opt_state: Optional[optax.OptState]: optimizer state
        :param tx_init: Optional[dict]: Pass the hyperparameters to the optimizer
        :param hyperparameters: Optional[dict]: Load hyperparameters from the state dict
        :param module: Optional[EasyDelFlaxPretrainedModel]: Pass in the module
        :param module_config: Optional[EasyDelPretrainedConfig]: Pass the module config
        :param module_config_args: Optional[dict]: Pass the config_args to the model
        :param kwargs: Pass in any additional parameters that may be needed for the model
        :return: A new instance of the class
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

        try:
            tx, sc = get_optimizer_and_scheduler(
                **tx_init
            )
        except TypeError:
            termcolor.cprint(
                "Couldn't load past optimizer State initializing new one with default Optimizer and Scheduler",
                color="red", force_color=True
            )
            tx, sc = get_optimizer_and_scheduler(
                optimizer="adamw",
                scheduler="none",
                steps=10000,
            )
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
            checkpoint_path: str | os.PathLike,
            init_optimizer_state: bool = False,
            state_shard_fns: Optional[Mapping[str, Callable]] = None,
            verbose: bool = False
    ):

        """    
        The load_state function is a class method that loads the state of an EasyDelModel from a checkpoint.
        
        :param cls: Create an instance of the class
        :param checkpoint_path: str | os.PathLike: Specify the path to the checkpoint file
        :param init_optimizer_state: bool: Initialize the optimizer if it's not Initialized yet (if it Initialized the option
        will be ignored )
        :param state_shard_fns: Optional[Mapping[str,Callable]]: Specify the function that will be used 
        to shard the loaded state
        :param verbose: bool: Print out the progress of loading
        :return: A state object
        """
        from ..modules.auto_easydel_model import get_modules_by_type

        checkpoint = fjformer.CheckpointManager.load_checkpoint(
            path=checkpoint_path,
            shard_fns=state_shard_fns,
            verbose=verbose,
        )

        hyperparameters = checkpoint.get("hyperparameters")
        cfg, module, convertor = get_modules_by_type(model_type=cls.get_model_type(hyperparameters))
        module_config = checkpoint.pop("module_config", None)
        if checkpoint["module_config_args"] is not None:
            module_config = cfg.from_dict(checkpoint.get("module_config_args", {}))

        state = cls.load(
            apply_fn=module.__call__,
            module=module,
            module_config=module_config,
            **checkpoint
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
            filename: str | os.PathLike,
            save_optimizer: bool = False,
            checkpoint_dir: Optional[str | os.PathLike] = None,
            verbose: bool = False,
            gather_fns: dict[Callable] = None,
            float_dtype: str | jax.numpy.dtype = None,
    ):

        """
        The save_state function saves the state of a model to disk.

        :param self: Pass the object itself to the function
        :param filename: str | os.PathLike: Specify the name of the file to save
        :param save_optimizer: bool: Determine whether to save the optimizer state or not
        :param checkpoint_dir: Optional[str | os.PathLike]: Specify the directory where the checkpoint is saved
        :param verbose: bool: Print out the path of the saved file
        :param gather_fns: dict[Callable]: Specify a dictionary of functions that can be used to gather
        :param float_dtype: str | jax.numpy.dtype: Specify the precision of the saved model
        :param : Save the optimizer state
        :return: None
        """
        state = self
        if not save_optimizer:
            state = self.replace(
                opt_state=None
            )
        fjformer.CheckpointManager.save_state_to_file(
            state=state,
            path=os.path.join(checkpoint_dir, filename) if checkpoint_dir is not None else filename,
            verbose=verbose,
            gather_fns=gather_fns,
            float_dtype=float_dtype,
        )

    def free_opt_state(self) -> "EasyDelState":

        """
        The free_opt_state function is used to free the memory allocated by a previous call to setopt.
        It should be called after all the options have been set, and before you perform any of the transfers.


        :param self: Represent the instance of the class
        :return: A new state with the opt_state field set to none
        """
        return self.replace(
            opt_state=None
        )

    def init_opt_state(self) -> "EasyDelState":

        """
        The init_opt_state function initializes the optimizer state.
        :param self: Make the object callable, and params is used to pass in a dictionary of parameters
        :return: A new instance of the class with opt_state initialized
        """
        if self.opt_state is None:
            params_with_opt = (
                self.params['params'] if OVERWRITE_WITH_GRADIENT in self.params else self.params
            )
            opt_state = self.tx.init(params_with_opt)

            return self.replace(
                opt_state=opt_state
            )
        return self

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            filename: Optional[str] = None,
            optimizer: AVAILABLE_OPTIMIZERS = "adamw",
            scheduler: AVAILABLE_SCHEDULERS = "none",
            tx_init: Optional[dict] = None,
            device=jax.devices('cpu')[0],
            dtype: jax.numpy.dtype = jax.numpy.float32,
            param_dtype: jax.numpy.dtype = jax.numpy.float32,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            key_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            value_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None, None),
            attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            input_shape: Sequence[int] = (1, 1),
            backend: Optional[str] = None,
            init_optimizer_state: bool = False,
            free_optimizer_state: bool = True,
            verbose: bool = True,
            state_shard_fns: Optional[Mapping[str, Callable]] = None,
            config_kwargs: Optional[Mapping[str, Any]] = None,
            **kwargs
    ) -> "EasyDelState":

        """
        The from_pretrained function is a helper function to quickly load a pretrained model and its associated configuration.
        This method takes care of returning the correct model class instance based on the `model_type` property in the
        config object, or when it's missing, falling back to using pattern matching on the
         `pretrained_model_name_or_path` string:

        :param cls: Refer to the class that is being defined
        :param pretrained_model_name_or_path: str: Load the pretrained model
        :param filename: Optional[str]: Specify the name of the file to download from huggingface hub
        :param optimizer: AVAILABLE_OPTIMIZERS: Specify the optimizer used for training
        :param scheduler: AVAILABLE_SCHEDULERS: Specify the name of the scheduler to use
        :param tx_init: Optional[dict]: Pass the hyperparameters of the optimizer
        :param device: Specify the device on which to run the model
        :param dtype: jax.numpy.dtype: Specify the dtype of the model parameters
        :param param_dtype: jax.numpy.dtype: Specify the data type of the parameters
        :param precision: jax.lax.Precision: Control the precision of the calculation
        :param sharding_axis_dims: Sequence[int]: Specify the dimension of each axis
        :param sharding_axis_names: Sequence[str]: Specify the names of the axes in each shard
        :param query_partition_spec: PartitionSpec: Specify the partitioning of the query matrix
        :param key_partition_spec: PartitionSpec: Specify the partitioning of the key matrix
        :param value_partition_spec: PartitionSpec: Specify the partitioning of the value tensor
        :param bias_partition_spec: PartitionSpec: Specify the partitioning of the bias
        :param attention_partition_spec: PartitionSpec: Partition the attention weights
        :param use_shard_map: bool: Determine whether to use shard_map or not
        :param input_shape: Sequence[int]: Specify the shape of the input to be used for training
        :param backend: Optional[str]: Specify the backend used for the model
        :param init_optimizer_state: bool: Initialize the optimizer state
        :param free_optimizer_state: bool: Free the optimizer state from memory
        :param verbose: bool: Print the progress of loading the model
        :param state_shard_fns: Optional[Mapping[str,Callable]]: Specify the function to use for sharding the state
        :param kwargs: Pass keyword arguments to the function
        :param config_kwargs: Optional[Mapping[str, Any]]: Config kwargs to be added to config before creating module
        :return: An `EasyDelState` object
        """
        if free_optimizer_state and init_optimizer_state:
            raise EasyDelRuntimeError(
                "You can't use `free_optimizer_state` and `init_optimizer_state` True at same Time"
            )

        if filename is None:
            from ..modules.auto_easydel_model import AutoEasyDelModelForCausalLM

            model, params = AutoEasyDelModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                device=device,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                sharding_axis_dims=sharding_axis_dims,
                sharding_axis_names=sharding_axis_names,
                query_partition_spec=query_partition_spec,
                key_partition_spec=key_partition_spec,
                value_partition_spec=value_partition_spec,
                bias_partition_spec=bias_partition_spec,
                attention_partition_spec=attention_partition_spec,
                use_shard_map=use_shard_map,
                input_shape=input_shape,
                backend=backend,
                config_kwargs=config_kwargs,
                **kwargs
            )
            if tx_init is None:
                tx_init = {}

            tx_init["optimizer"] = optimizer
            tx_init["scheduler"] = scheduler

            state = cls.load(
                apply_fn=model.__call__,
                params=FrozenDict({'params': params}),
                step=0,
                opt_state=None,
                tx_init=tx_init,
                hyperparameters=None,
                module=model,
                module_config=model.config,
                module_config_args=model.config.to_dict()
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
                    state_shard_fns=state_shard_fns
                )
        if init_optimizer_state:
            with jax.default_device(device):
                state = state.init_opt_state()
        if free_optimizer_state:
            state = state.free_opt_state()
        return state

    def shard_params(
            self,
            fully_sharded_data_parallel: bool = True,
            shard_fns: Optional[Mapping[str, Callable]] = None,
            dtype: jax.numpy.dtype | str = "bf16",
            mesh: Optional[Mesh] = None,
            rules: Optional[Sequence[Mapping[str, PartitionSpec]]] = None
    ):
        dtype = fjformer.get_dtype(dtype)
        if shard_fns is None and self.module_config is None and rules is None:
            raise EasyDelRuntimeError(
                "the model doesn't carrying `module_config` you should pass `shard_fns` or `rules`"
            )
        elif shard_fns is None and rules is not None or self.module_config is not None:
            from fjformer import match_partition_rules, make_shard_and_gather_fns
            rules = rules or self.module_config.get_partition_rules(fully_sharded_data_parallel)
            partition_specs = match_partition_rules(
                rules=rules, params=self.params
            )
            shard_fns, gather_fns = make_shard_and_gather_fns(
                partition_specs=partition_specs,
                dtype_specs=dtype
            )
        if mesh is None:
            mesh = self.module_config.jax_mesh()
        with mesh:
            return self.replace(
                params=jax.tree_util.tree_map(
                    lambda f, p: f(p), shard_fns, self.params
                )
            )

    @staticmethod
    def create_hyperparameters(model_type: str):
        """
        it's the only way we can dump xla compiler
        """
        return {
            STRING_REP.format(
                type="str",
                key="model_type",
                value=model_type
            ): DEFAULT_ES_VAL
        }

    @staticmethod
    def safe_dict(dictionary: dict):
        for k in list(dictionary.keys()):
            val = dictionary.get(k)
            if not isinstance(val, (int, bool)):
                val = dictionary.pop(k)
                string_value_format = STRING_REP.format(
                    type=type(val).__name__,
                    key=k,
                    value=val
                )
                dictionary[string_value_format] = DEFAULT_ES_VAL
        return dictionary

    @staticmethod
    def unsafe_dict(dictionary: dict):
        result = {}
        for k in list(dictionary.keys()):
            v = dictionary[k]
            key, value = break_format(key=k, value=v)
            result[key] = value
        return result

    def __str__(self):

        """
        The __str__ function is called when you call str(object) or print(object).
        The __repr__ function is called when you type the object name in the interpreter.
        If no __str__ method exists, Python will use __repr__ as a fallback.

        :param self: Refer to the object itself
        :return: string
        """
        params_size = sum(getattr(n, "size", 0) for n in jax.tree_util.tree_flatten(self.params)[0])
        opt_state_size = sum(getattr(n, "size", 0) for n in jax.tree_util.tree_flatten(self.opt_state)[0])

        def make_depth(mdl=None):
            if mdl is not None:
                try:
                    return mdl.__str__().replace(
                        "\n",
                        "\n\t"
                        ""
                    ) if hasattr(mdl, "__str__") else None
                except TypeError:
                    ...
            return mdl

        optimizer = self.tx_init.get("optimizer", None)
        scheduler = self.tx_init.get("scheduler", None)

        if optimizer is None:
            optimizer = self.find_key(
                "optimizer",
                self.tx_init
            )
        if scheduler is None:
            scheduler = self.find_key(
                "scheduler",
                self.tx_init
            )

        string = (
            f"{self.__class__.__name__}("
            f"\n\tstep = {self.step}"
            f"\n\tmodule = {make_depth(self.module)}"
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
        req = dictionary.get(key, None)
        if req is None:
            req = cls.find_key(key, dictionary)
        return req or default

    @staticmethod
    def find_key(key, dictionary: dict) -> str | None:
        result = None
        for k, v in dictionary.items():
            k_, v_ = break_format(key=k, value=v)
            if k_ == key:
                result = v_
                break
        return result

    def __repr__(self):

        """
        The __repr__ function is the &quot;official&quot; string representation of an object.
        It's what you get when you type the object name at the Python prompt, or pass it to str().
        The goal of __repr__ is to be unambiguous: if eval(repr(x)) == x, then __repr__ should return a string that
        looks like a valid Python expression that could be used to recreate an object with the same value (
        given an appropriate environment). If this is not possible, a string formatted using %s
        formatting is also acceptable.

        :param self: Represent the instance of the class
        :return: A string that is a valid python expression
        """
        return self.__str__()
