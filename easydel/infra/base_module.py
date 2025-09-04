# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import re
import typing as tp
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, partial
from re import Pattern
from typing import Self

import chex
import flax
import flax.nnx
import flax.struct
import jax
import jax.extend
import jax.tree_util
from eformer.escale import make_shard_and_gather_fns, match_partition_rules
from eformer.loggings import get_logger
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.utils import traversals
from easydel.utils.traversals import flatten_dict, is_flatten, unflatten_dict

from .base_config import EasyDeLBaseConfig
from .etils import EasyDeLGradientCheckPointers, EasyDeLQuantizationMethods
from .loss_utils import LOSS_MAPPING, ForCausalLMLoss, ForSequenceClassificationLoss, LossConfig, LossMetrics
from .mixins import BaseModuleProtocol, EasyBridgeMixin, EasyGenerationMixin

if tp.TYPE_CHECKING:
    from easydel.infra.base_state import EasyDeLState
    from easydel.layers.linear import ParallelLinear

PartitionLike = tp.Mapping[str, tp.Callable] | tp.Mapping[tuple, tp.Callable] | None


logger = get_logger(__name__)

BaseConf = EasyDeLBaseConfig


@dataclass
class ParameterTransformRule:
    """Rule for transforming MoE parameter names and tensors."""

    pattern: str | Pattern
    replacement: str
    tensor_transform: Callable | None = None
    consolidate_experts: bool = False


class EasyDeLBaseModule(nn.Module, BaseModuleProtocol, EasyBridgeMixin, EasyGenerationMixin):
    """
    Base class for EasyDeL modules, providing common functionalities for model initialization,
    parameter handling, and integration with the EasyDeL ecosystem.
    """

    config_class: type[BaseConf]
    base_model_prefix: str
    config: BaseConf | None = None
    _model_task: str | None = None
    _model_type: str | None = None
    _parameter_transform_rules: tp.ClassVar[list[ParameterTransformRule]] = []

    def __init__(
        self,
        config: BaseConf,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: lax.PrecisionLike,
        rngs: nn.Rngs,
    ):
        """Initialize the EasyDeLBaseModule.

        Sets up the base module with configuration and data types.
        Subclasses should call this in their __init__ method.

        Args:
            config: Model configuration with architecture parameters.
            dtype: Data type for computations (e.g., float32, bfloat16).
            param_dtype: Data type for model parameters.
            precision: Precision setting for JAX operations.
            rngs: Random number generators for initialization.

        Note:
            This method should be called by all subclasses to properly
            initialize the base functionality.
        """
        self.config: BaseConf = config
        self.dtype: jnp.dtype = dtype
        self.param_dtype: jnp.dtype = param_dtype
        self.precision: lax.PrecisionLike = precision
        self.rngs: nn.Rngs = rngs

        # these useless call's are just here to init values in graphdef
        _ = self.graphtree_shape
        _ = self.graphtree_params_shape
        _ = self.mesh
        _ = self.model_task
        _ = self.model_type

    @property
    def parameters(self: Self) -> dict:
        """
        Retrieves the parameters of the module as a dictionary.

        This property iterates through the module and its submodules, extracting
        variables marked as `nn.Param` and returning them in a flat dictionary
        where keys represent the parameter path.

        Returns:
            tp.Dict: A dictionary containing the module's parameters.
        """
        from easydel.utils.traversals import iter_module_search

        parameters = {}
        for key, value in iter_module_search(self, nn.Param):
            parameters[key] = value.value
        return parameters

    @property
    def graphstate_type(self: Self):
        return nn.LoRAParam if self.lora_is_enabled else nn.Param

    def split_module(self: Self):
        return nn.split(self, self.graphstate_type, ...)

    @staticmethod
    def merge_module(graphdef: nn.GraphDef, graphstate: nn.GraphState, graphother: nn.GraphState):
        return nn.merge(graphdef, graphstate, graphother)

    @property
    def graphdef(self: Self) -> nn.GraphDef:
        """
        Returns the graph definition (structure without parameters) of the module.

        Uses `flax.nnx.split` to separate the graph definition from the state (parameters).

        Returns:
            nn.GraphDef: The graph definition of the module.
        """
        return nn.split(self, self.graphstate_type, ...)[0]

    @property
    def graphstate(self: Self) -> nn.GraphState:
        """
        Returns the graph state (parameters) of the module.

        Uses `flax.nnx.split` to separate the state (parameters) from the graph definition.

        Returns:
            nn.GraphState: The graph state containing the module's parameters.
        """
        return nn.split(self, self.graphstate_type, ...)[1]

    @property
    def graphother(self: Self) -> nn.GraphState:
        """
        Returns any other state variables in the module (non-parameters).

        Uses `flax.nnx.split` to separate non-parameter state variables.

        Returns:
            nn.GraphState: The graph state containing non-parameter variables.
        """
        return nn.split(self, self.graphstate_type, ...)[-1]

    @property
    def graphtree_params_shape(self: Self) -> dict:
        """
        Computes and returns the shapes of the module's parameters as a nested dictionary.

        It uses `nnx.eval_shape` to determine the shapes without actual computation,
        then extracts the shape information from the resulting graph state.

        Returns:
            tp.Dict: A nested dictionary mirroring the parameter structure, containing their shapes.
        """
        graphtree = nn.eval_shape(lambda: nn.split(self, self.graphstate_type, ...)[1])

        flattened_tree = flatten_dict(graphtree)

        param_shapes = {key: val.value for key, val in flattened_tree.items()}
        return unflatten_dict(param_shapes)

    @property
    def graphtree_shape(self: Self) -> dict:
        """
        Computes and returns the shapes of all state variables (including non-parameters) in the module.

        Uses `nnx.eval_shape` on the entire module state (parameters and others)
        and extracts the shape information.

        Returns:
            tp.Dict: A nested dictionary mirroring the module's state structure, containing the shapes.
        """
        graphtree = nn.eval_shape(lambda: nn.split(self)[1])

        flattened_tree = flatten_dict(graphtree)

        param_shapes = {key: getattr(val, "value", val) for key, val in flattened_tree.items()}
        return unflatten_dict(param_shapes)

    @property
    def mesh(self: Self) -> jax.sharding.Mesh:
        """
        Retrieves the JAX device mesh from the module's configuration.

        Returns:
            jax.sharding.Mesh: The device mesh defined in `self.config.mesh`.
        """
        return self.config.mesh

    @property
    def model_task(self: Self) -> str | None:
        """
        Returns the specific task associated with this model instance (e.g., 'causal-language-model').

        Returns:
            tp.Optional[str]: The model task identifier, or None if not set.
        """
        return self._model_task

    @property
    def model_type(self: Self) -> str | None:
        """
        Returns the specific type of this model instance (e.g., 'llama', 'mistral').

        Returns:
            tp.Optional[str]: The model type identifier, or None if not set.
        """
        return self._model_type

    @property
    def params(self: Self) -> dict:
        """
        Returns the parameters and other state variables of the module as a dictionary.

        Uses `flax.nnx.split` to get the combined state (parameters and others).

        Returns:
            tp.Dict: A dictionary containing all state variables of the module.
        """
        return nn.split(self)[-1]

    @cached_property
    def causal_mask(self: Self) -> jnp.ndarray:
        """
        Retrieves or computes the basic causal attention mask from the configuration.

        Uses `self.config.get_basic_causal_mask()` and caches the result.

        Returns:
            jnp.ndarray: The causal attention mask, potentially cached.
        """
        return self.config.get_basic_causal_mask()

    @cached_property
    def frequencies(self: Self) -> jnp.ndarray:
        """
        Retrieves or computes the frequency components (e.g., for RoPE) from the configuration.

        Uses `self.config.get_basic_frequencies()` and caches the result.

        Returns:
            jnp.ndarray: The frequency components, potentially cached.
        """
        return self.config.get_basic_frequencies()

    @cached_property
    def inv_frequencies(self: Self) -> jnp.ndarray:
        """
        Retrieves or computes the inv-frequency components (e.g., for RoPE) from the configuration.

        Uses `self.config.get_basic_inv_frequencies()` and caches the result.

        Returns:
            jnp.ndarray: The inv-frequency components, potentially cached.
        """
        return self.config.get_basic_inv_frequencies()

    @cached_property
    def static_arguments(self: Self) -> tuple:
        """
        Retrieves or computes static arguments needed for the module's `__call__` method.

        Uses `self.get_static_arguments()` and caches the result. Static arguments
        are typically those that don't change during execution and can be pre-computed.

        Returns:
            tp.Tuple: A tuple of static arguments.
        """
        return self.get_static_arguments()

    @cached_property
    def lossfn_type(self: Self):
        if getattr(self.config, "loss_type", None) is not None:
            loss_type = self.config.loss_type
        elif getattr(self, "loss_type", None) is not None:
            loss_type = self.loss_type
        else:
            loss_type = self.__class__.__name__
            if loss_type not in LOSS_MAPPING:
                loss_groups = f"({'|'.join(LOSS_MAPPING)})"
                loss_type = re.findall(loss_groups, self.__class__.__name__)
                if len(loss_type) > 0:
                    loss_type = loss_type[0]
                else:
                    loss_type = None
        if loss_type is None or (loss_type not in LOSS_MAPPING and getattr(self.config, "loss_type", None) is not None):
            warnings.warn(
                f"`loss_type={loss_type}` was set in the config but it is unrecognised."
                f"Using the default loss: `ForCausalLMLoss`.",
                stacklevel=1,
            )
            loss_type = "ForCausalLM"
        return loss_type

    @cached_property
    def loss_function(self: Self):
        """
        Determines and returns the appropriate loss function based on the configuration or model type.

        It prioritizes `config.loss_type`, then `self.loss_type`, and finally tries to infer
        the loss type from the class name. If no suitable loss function is found, it defaults
        to `ForCausalLMLoss` and issues a warning.

        Returns:
            tp.Callable: The selected loss function (e.g., `ForCausalLMLoss`, `ForSequenceClassificationLoss`).
        """

        return LOSS_MAPPING[self.lossfn_type]

    @property
    def module_dtype(self: Self) -> jnp.dtype:
        """
        Determines the data type of the module's parameters.

        It inspects the flattened parameter state to find the dtype of the first parameter encountered.

        Returns:
            jnp.dtype: The data type of the module's parameters.
        """
        params_state = nn.split(self, self.graphstate_type, ...)[1].flat_state()
        return jax.tree_util.tree_leaves(params_state)[0].dtype

    def compute_complex_rotary(self, position_ids: jax.Array) -> jnp.ndarray:
        frequencies = jnp.transpose(
            self.inv_frequencies[None, :, None] @ position_ids[:, None, :].astype("f4"),
            (0, 2, 1),
        )
        return jnp.exp(1j * frequencies)

    def to_dtype(self: Self, dtype: jnp.dtype) -> Self:
        """
        Converts the module's parameters to the specified data type.

        It iterates through the module's parameters (excluding quantization-related ones)
        and casts them to the target `dtype`. It also updates the `param_dtype` attribute
        of the module and its submodules if they exist.

        Args:
            dtype (jnp.dtype): The target data type for the parameters.

        Returns:
            Self: The module instance with parameters converted to the specified dtype.
        """
        from easydel.utils.traversals import iter_module_search

        gdef, state, others = nn.split(self, self.graphstate_type, ...)

        def _map(path, val: nn.VariableState):
            if val.value is not None:
                if not path[-1].startswith("quant_"):
                    val.value = val.value.astype(dtype)
            return val

        state.update(state.map(_map))
        self = nn.merge(gdef, state, others)

        for _path, module in iter_module_search(self):
            if hasattr(module, "param_dtype"):
                module.param_dtype = dtype
        return self

    def half(self: Self, change_runtime_dtype: bool = True) -> Self:
        """
        Converts the module's parameters to half-precision (float16).

        Optionally also changes the runtime computation dtype (`self.dtype`) to float16.

        Args:
            change_runtime_dtype (bool): If True, also sets `self.dtype` to `jnp.float16`.
                Defaults to True.

        Returns:
            Self: The module instance with parameters (and potentially runtime dtype) set to float16.
        """
        if change_runtime_dtype:
            self = self._reformat_runtime_dtype(jnp.float16)
        return self._reformat_dtype(jnp.float16)

    def float(self: Self, change_runtime_dtype: bool = True) -> Self:
        """
        Converts the module's parameters to single-precision (float32).

        Optionally also changes the runtime computation dtype (`self.dtype`) to float32.

        Args:
            change_runtime_dtype (bool): If True, also sets `self.dtype` to `jnp.float32`.
                Defaults to True.

        Returns:
            Self: The module instance with parameters (and potentially runtime dtype) set to float32.
        """
        if change_runtime_dtype:
            self = self._reformat_runtime_dtype(jnp.float32)
        return self._reformat_dtype(jnp.float32)

    def _reformat_runtime_dtype(self: Self, dtype) -> Self:
        """
        Internal helper to change the runtime computation data type (`dtype`) of the module and its submodules.

        Args:
            dtype (jnp.dtype): The target runtime data type.

        Returns:
            Self: The module instance with updated runtime dtype.
        """
        from easydel.utils.traversals import iter_module_search

        for _path, module in iter_module_search(self):
            if hasattr(module, "dtype"):
                if str(type(module.dtype)).endswith("lax_numpy._ScalarMeta'>"):  # dont change numpy based dtypes
                    module.dtype = dtype
        self.dtype = dtype
        return self

    def _reformat_dtype(self: Self, dtype) -> Self:
        """
        Internal helper to change the data type of the module's parameters (`param_dtype`).

        Casts floating-point parameters to the target `dtype`.

        Args:
            dtype (jnp.dtype): The target parameter data type.

        Returns:
            Self: The module instance with updated parameter dtype.
        """
        from easydel.utils.traversals import iter_module_search

        gdef, gtree, others = nn.split(self, self.graphstate_type, ...)

        def _map(array):
            if array.dtype in [
                jnp.bfloat16,
                jnp.float16,
                jnp.float32,
                jnp.float64,
                jnp.float_,
            ]:
                array = array.astype(dtype)
            return array

        gtree = jax.tree_util.tree_map(_map, gtree)
        others = jax.tree_util.tree_map(_map, others)
        self = nn.merge(gdef, gtree, others)

        for _path, module in iter_module_search(self):
            if hasattr(module, "param_dtype"):
                if isinstance(module.param_dtype, jnp.dtype):
                    module.param_dtype = dtype

        self.param_dtype = dtype
        return self

    def _match_partition_rules(self, partition_rules: tp.Any = None):
        """
        Matches the provided or configured partition rules against the module's parameter shapes.

        Args:
            partition_rules (tp.Any, optional): The partition rules to use. If None, uses rules
                from the configuration. Defaults to None.

        Returns:
            tp.Any: The partition specifications matched to the parameter tree.
        """
        return match_partition_rules(
            rules=self._get_partition_rules(partition_rules),
            tree=self.graphtree_params_shape,
        )

    @property
    def _specs_sharding(self: Self):
        """
        Extracts the PartitionSpec part from the NamedSharding of each parameter.

        Returns:
            tp.Dict: A nested dictionary mirroring the parameter structure, containing PartitionSpecs.
        """

        def _map(array):
            if hasattr(array, "sharding"):
                sharding = array.sharding
                if isinstance(sharding, NamedSharding):
                    return sharding.spec
            return PartitionSpec()

        return nn.from_tree(jax.tree_util.tree_map(_map, nn.to_tree(self)))

    @property
    def _shardings(self: Self):
        """
        Extracts the sharding information (PartitionSpec or NamedSharding) for each parameter.

        Returns:
            tp.Dict: A nested dictionary mirroring the parameter structure, containing the sharding info.
        """
        return nn.from_tree(
            jax.tree_util.tree_map(
                lambda x: x.sharding if hasattr(x, "sharding") else PartitionSpec(),
                nn.to_tree(self),
            )
        )

    @property
    def _named_shardings(self: Self):
        """
        Extracts the NamedSharding object (if present) for each parameter.

        Returns:
            tp.Dict: A nested dictionary mirroring the parameter structure, containing NamedSharding or None.
        """
        return nn.from_tree(
            jax.tree_util.tree_map(
                lambda x: x.sharding if hasattr(x, "sharding") else None,
                nn.to_tree(self),
            )
        )

    def _get_mesh(self, mesh: Mesh | None = None) -> Mesh:
        """
        Retrieves the JAX device mesh, prioritizing the provided argument over the configuration.

        Args:
            mesh (tp.Optional[Mesh]): A potential JAX device mesh.

        Returns:
            Mesh: The resolved JAX device mesh.

        Raises:
            ValueError: If no mesh is provided and none is found in the configuration.
        """
        if mesh is None:
            if not hasattr(self, "config") or not hasattr(self.config, "mesh") or self.config.mesh is None:
                raise ValueError("A mesh must be provided, either as an argument or through the model config.")
            return self.config.mesh
        return mesh

    def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
        """
        Retrieves the partitioning rules, prioritizing the provided argument over the configuration.

        Args:
            partition_rules (PartitionLike): Potential partitioning rules.

        Returns:
            PartitionLike: The resolved partitioning rules.

        Raises:
            ValueError: If no rules are provided and none can be obtained from the configuration.
        """
        if partition_rules is None:
            if not hasattr(self, "config"):
                raise ValueError("Partition rules must be provided either as an argument or through the model config.")

            return self.config.get_partition_rules(fully_sharded_data_parallel=True)
        return partition_rules

    def _apply_sharding_fns(
        self: Self,
        sharding_fns: tp.Mapping[str, tp.Callable],
    ) -> Self:
        """
        Applies sharding or gathering functions to the module's parameters.

        Args:
            sharding_fns (tp.Mapping[str, tp.Callable]): A mapping from flattened parameter paths
                to sharding or gathering functions.

        Returns:
            Self: The module instance with sharding/gathering functions applied to its parameters.
        """
        gdef, state, others = nn.split(self, self.graphstate_type, ...)
        sharding_fns = flatten_dict(sharding_fns)
        _shard_keys = list(sharding_fns.keys())

        def _map(path, val: nn.VariableState):
            if val.value is not None and path in _shard_keys:
                fn = sharding_fns[path]
                val.value = fn(val.value)
            return val

        state.update(state.map(_map))
        others.update(others.map(_map))
        self = nn.merge(gdef, state, others)
        return self

    def shard_model(
        self: Self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
        overlay_fns: tp.Mapping[str, tp.Callable] | None = None,
    ) -> Self:
        """
        Shards the model's parameters according to the specified rules and mesh.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules. If None, uses config rules.
                Defaults to None.
            mesh (tp.Optional[Mesh], optional): JAX device mesh. If None, uses config mesh. Defaults to None.
            overlay_fns (tp.Optional[tp.Mapping[str, tp.Callable]], optional): Additional functions to apply,
                potentially overriding default sharding for specific parameters. Defaults to None.

        Returns:
            Self: The sharded model instance.
        """
        mesh = self._get_mesh(mesh)
        partition_rules = self._get_partition_rules(partition_rules)
        partition_specs = match_partition_rules(rules=partition_rules, tree=self.graphtree_params_shape)
        shard_fns, _ = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
        if overlay_fns is not None:
            shard_fns.update(overlay_fns)
        self = self._apply_sharding_fns(shard_fns)
        return self

    def gather_model(
        self: Self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
        overlay_fns: tp.Mapping[str, tp.Callable] | None = None,
    ) -> Self:
        """
        Gathers the model's parameters from potentially distributed devices to the host or a single device.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules used to determine how parameters
                were originally sharded. If None, uses config rules. Defaults to None.
            mesh (tp.Optional[Mesh], optional): JAX device mesh from which to gather. If None, uses config mesh.
                Defaults to None.
            overlay_fns (tp.Optional[tp.Mapping[str, tp.Callable]], optional): Additional functions to apply,
                potentially overriding default gathering for specific parameters. Defaults to None.

        Returns:
            Self: The model instance with gathered parameters.
        """
        mesh = self._get_mesh(mesh)
        partition_rules = self._get_partition_rules(partition_rules)
        partition_specs = match_partition_rules(
            rules=partition_rules,
            tree=self.graphtree_params_shape,
        )
        _, gather_fns = make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=mesh,
        )

        if overlay_fns is not None:
            gather_fns.update(overlay_fns)
        return self._apply_sharding_fns(gather_fns)

    @property
    def _shard_fns(self: Self):
        """
        Generates the dictionary of sharding functions based on the module's configuration.

        Returns:
            tp.Mapping: A mapping from flattened parameter paths to sharding functions.
        """
        mesh = self._get_mesh(None)
        partition_specs = match_partition_rules(
            rules=self._get_partition_rules(None),
            tree=self.graphtree_params_shape,
        )
        return make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=mesh,
        )[0]

    @property
    def _gather_fns(self: Self):
        """
        Generates the dictionary of gathering functions based on the module's configuration.

        Returns:
            tp.Mapping: A mapping from flattened parameter paths to gathering functions.
        """
        mesh = self._get_mesh(None)
        partition_specs = match_partition_rules(
            rules=self._get_partition_rules(None),
            tree=self.graphtree_params_shape,
        )
        return make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=mesh,
        )[1]

    def apply_out_shardings(self, out_shardings):
        splits = self.split_module()

        @partial(jax.jit, out_shardings=out_shardings)
        def _call(graphstate, graphother):
            return graphstate, graphother

        splits[1:] = _call(*splits[1:])
        return self.merge_module(*splits)

    def fully_shard(self: Self, partition_rules: PartitionLike = None) -> Self:
        """
        Applies JAX sharding constraints to all parameters based on the partition rules.

        This function ensures that parameters are explicitly marked with their intended sharding,
        which can be useful for performance and correctness checks. It uses `ejit` with
        `out_shardings` to enforce the constraints.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules. If None, uses config rules.
                Defaults to None.

        Returns:
            Self: The model instance with sharding constraints applied.
        """

        class ShardState(flax.struct.PyTreeNode):
            graphdef: nn.GraphDef
            graphstate: nn.GraphState

        gdef, gstate = nn.split(self)
        mock = ShardState(graphdef=gdef, graphstate=gstate)
        shardings = jax.tree_util.tree_map(
            lambda x: NamedSharding(mesh=self.mesh, spec=x),
            match_partition_rules(self._get_partition_rules(partition_rules), nn.eval_shape(lambda: mock)),
        )

        @partial(jax.jit, out_shardings=shardings)
        def _call(cl):
            return cl

        mock = _call(mock)
        self = nn.merge(mock.graphdef, mock.graphstate)
        return self

    def fully_gather(self: Self) -> Self:
        """
        Applies JAX sharding constraints to gather all parameters onto the host or a single device.

        This function marks all parameters to have no sharding (PartitionSpec()). It uses `ejit`
        with `out_shardings` to enforce these gathering constraints.

        Returns:
            Self: The model instance with gathering constraints applied.
        """

        class ShardState(flax.struct.PyTreeNode):
            graphdef: nn.GraphDef
            graphstate: nn.GraphState

        gdef, gstate = nn.split(self)
        mock = ShardState(graphdef=gdef, graphstate=gstate)
        shardings = jax.tree_util.tree_map(
            lambda x: NamedSharding(mesh=self.mesh, spec=PartitionSpec()),
            match_partition_rules(self._get_partition_rules(None), nn.eval_shape(lambda: mock)),
        )

        @partial(jax.jit, out_shardings=shardings)
        def _call(cl):
            return cl

        mock = _call(mock)
        self = nn.merge(mock.graphdef, mock.graphstate)
        return self

    def quantize(
        self: Self,
        method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.A8BIT,
        block_size: int = 128,
        quantization_pattern: str | None = None,
        quantize_tensors: bool = True,
        verbose: bool | None = None,
    ) -> Self:
        """
        Applies quantization to the module's linear layers or tensors.

        Args:
            method (EasyDeLQuantizationMethods, optional): The quantization algorithm to use
                (e.g., A8BIT, NF4). Defaults to EasyDeLQuantizationMethods.A8BIT.
            block_size (int, optional): The block size for quantization methods that support it.
                Defaults to 128.
            quantization_pattern (tp.Optional[str], optional): A regular expression to match
                parameter names that should be quantized. If None, uses a default pattern.
                Defaults to None.
            quantize_tensors (bool, optional): If True, quantizes the tensor values directly.
                If False (currently default behavior in implementation), replaces Linear layers
                with their quantized equivalents. Defaults to True (though implementation differs).
            verbose (tp.Optional[bool], optional): If True, logs information during the quantization process.
                Defaults to True only on process index 0.

        Returns:
            Self: The quantized model instance.
        """
        from easydel.layers.quantization.quantizers import EasyQuantizer

        quantizer = EasyQuantizer(
            quantization_method=method,
            block_size=block_size,
            quantization_pattern=quantization_pattern,
        )
        if verbose is None:
            verbose = jax.process_index() == 0
        if quantize_tensors:
            ...
        else:
            self = quantizer.quantize_linears(
                self,
                quantization_pattern=quantization_pattern,
                verbose=verbose,
            )
        return self

    def to_state(self, state_class: type[EasyDeLState] | None = None) -> EasyDeLState:
        """
        Converts the current module instance into an `EasyDeLState` object.

        This is useful for saving and managing the model's state, including parameters
        and potentially optimizer state (though optimizer state is typically added later).

        Returns:
            EasyDeLState: An EasyDeLState object representing the current model state.
        """
        if state_class is None:
            from easydel.infra.base_state import EasyDeLState

            state_class = EasyDeLState

        @partial(jax.jit, donate_argnums=(1, 2), static_argnums=(0,))
        def _create_state(gstruct, gstate, gother):
            return state_class.create(
                step=0,
                model=self.merge_module(gstruct, gstate, gother),
            )

        return _create_state(*self.split_module())

    def to_torch(self, **kwargs):
        """
        Converts the EasyDeL module to its equivalent Hugging Face PyTorch model.

        Requires the corresponding PyTorch model class to be available and registered.
        Uses utility functions to transfer parameters from JAX to PyTorch format.

        Args:
            **kwargs: Additional keyword arguments passed to the parameter transformation function.

        Returns:
            torch.nn.Module: The equivalent Hugging Face PyTorch model with loaded weights.
        """
        from easydel.utils.parameters_transformation import ModelConverter

        return ModelConverter.easydel_to_huggingface(
            module=self,
            base_huggingface_module=self.get_torch_loader()._model_mapping[type(self.config)],
            config=self.config,
            dtype=self.param_dtype,
            **kwargs,
        )

    def prepare_inputs_for_call(self, **kwargs):
        """
        Prepares keyword arguments before passing them to the module's `__call__` method.

        This base implementation simply returns the kwargs as is. Subclasses can override
        this to modify or add arguments as needed (e.g., for generation).

        Args:
            **kwargs: The keyword arguments intended for `__call__`.

        Returns:
            dict: The prepared keyword arguments.
        """
        return kwargs

    def get_static_arguments(self: Self) -> tuple:
        """
        Returns a tuple of static arguments required by the module's `__call__` method.

        Static arguments are those that don't change across calls and can be potentially
        cached or handled differently by JIT compilation. This base implementation returns
        an empty tuple. Subclasses should override this if they have static arguments.

        Returns:
            tp.Tuple: A tuple containing static arguments.
        """
        return ()

    def get_encoder(self: Self) -> nn.Module | EasyDeLBaseModule:
        """
        Returns the encoder part of the model's graph definition.

        This is useful for models that have a distinct encoder component, such as
        encoder-decoder architectures. The base implementation returns the full graph definition..
        """
        raise NotImplementedError()

    def get_decoder(self: Self) -> nn.Module | EasyDeLBaseModule:
        """
        Returns the decoder part of the model's graph definition.

        This is useful for models that have a distinct decoder component, such as
        encoder-decoder architectures. The base implementation returns the full graph definition.
        """
        raise NotImplementedError()

    def get_lm_head(self: Self) -> ParallelLinear:
        """
        Returns the language model head of the module.

        This is useful for models that have a separate head for language modeling tasks.
        The base implementation returns the full graph definition.
        """
        raise NotImplementedError()

    def get_embedding(self: Self) -> nn.Module | nn.Embed:
        """
        Returns the embedding layer of the module.

        This is useful for models that have a separate embedding layer for input tokens.
        The base implementation returns the full graph definition.
        """
        raise NotImplementedError()

    @classmethod
    def sequential_init(cls: type[Self], **kwargs) -> Self:
        """Initialize model parameters sequentially with proper sharding.

        This method performs lazy initialization followed by sequential parameter
        initialization with appropriate sharding for distributed training. It's
        particularly useful for large models that need memory-efficient initialization.

        The method:
        1. Creates a lazy (shape-only) version of the model
        2. Iterates through all modules and initializes their parameters
        3. Applies proper sharding based on partition rules

        Args:
            **kwargs: Arguments passed to lazy_init, including:
                - config: Model configuration
                - dtype: Computation dtype
                - param_dtype: Parameter dtype
                - precision: JAX precision setting
                - rngs: Random number generators (required)

        Returns:
            Self: Fully initialized model with properly sharded parameters.

        Example:
            >>> config = LlamaConfig(hidden_size=1024, num_hidden_layers=4)
            >>> model = LlamaModel.sequential_init(
            ...     config=config,
            ...     dtype=jnp.float32,
            ...     param_dtype=jnp.float32,
            ...     rngs=nn.Rngs(0)
            ... )
        """
        from easydel.utils.traversals import iter_module_search

        def _shard(x):
            return x

        rng = kwargs.get("rngs", flax.nnx.Rngs(44))
        lazy_model = cls.lazy_init(**kwargs)
        partition_rules = lazy_model.config.get_partition_rules()
        for path, module in iter_module_search(lazy_model, flax.nnx.Module):
            if path:
                joined_path = "/".join([str(p) for p in path])
                a = jnp.ones((1,))
                partition_spec = jax.tree_util.tree_map(
                    lambda x: NamedSharding(lazy_model.mesh, x),
                    match_partition_rules(
                        partition_rules,
                        {
                            joined_path + "/kernel": a,
                            joined_path + "/bias": a,
                            joined_path + "/embedding": a,
                            joined_path + "/scale": a,
                        },
                        strict=False,
                    ),
                )

                shardings = {
                    "kernel": partition_spec[joined_path + "/kernel"],
                    "bias": partition_spec[joined_path + "/bias"],
                    "embedding": partition_spec[joined_path + "/embedding"],
                    "scale": partition_spec[joined_path + "/scale"],
                }
            if hasattr(module, "kernel") and hasattr(module, "kernel_init"):
                arr = module.kernel_init(
                    key=rng.param(),
                    shape=module.kernel.value.shape,
                    dtype=module.kernel.value.dtype,
                )
                arr = jax.jit(_shard, out_shardings=shardings["kernel"])(arr)
                if isinstance(module.kernel, flax.nnx.Param):
                    module.kernel.value = arr
                else:
                    module.kernel = arr
            if hasattr(module, "bias") and hasattr(module, "bias_init") and module.bias is not None:
                arr = module.bias_init(
                    key=rng.param(),
                    shape=module.bias.value.shape,
                    dtype=module.bias.value.dtype,
                )
                arr = jax.jit(_shard, out_shardings=shardings["bias"])(arr)
                module.bias.value = arr

            if hasattr(module, "embedding") and hasattr(module, "embedding_init"):
                arr = module.embedding_init(
                    key=rng.param(),
                    shape=module.embedding.value.shape,
                    dtype=module.embedding.value.dtype,
                )
                arr = jax.jit(_shard, out_shardings=shardings["embedding"])(arr)
                module.embedding.value = arr

            if hasattr(module, "scale") and hasattr(module, "scale_init"):
                arr = module.scale_init(
                    key=rng.param(),
                    shape=module.scale.value.shape,
                    dtype=module.scale.value.dtype,
                )
                arr = jax.jit(_shard, out_shardings=shardings["scale"])(arr)
                module.scale.value = arr

            if hasattr(module, "rngs"):
                module.rngs = rng.fork()
        for path, module in iter_module_search(lazy_model, nn.Param):
            if path and type(module.value) is jax.ShapeDtypeStruct:
                logger.warning("found empty array at " + ("/".join([str(s) for s in path])))

        return lazy_model

    @classmethod
    def lazy_init(cls: type[Self], **kwargs) -> Self:
        """
        Performs a "lazy" initialization using `nnx.eval_shape`.

        This initializes the module structure and determines parameter shapes without
        actually allocating memory for the parameters. Useful for inspecting the model
        structure or preparing for sharding.

        Args:
            **kwargs: Keyword arguments passed to the class constructor.

        Returns:
            Self: A module instance with initialized structure but potentially abstract parameters.
        """
        rngs = kwargs.pop("rngs", None)

        def _init(rngs):
            return cls(**kwargs, rngs=rngs)

        return nn.eval_shape(_init, rngs=rngs)

    def merge_lora_params(self: Self, pytree: dict) -> Self:
        """
        Merges LoRA parameters from a pytree into the base model's parameters.

        Args:
            pytree (tp.Dict): A dictionary (pytree) containing the LoRA parameters (A and B matrices)
                structured similarly to the base model's parameters.

        Returns:
            Self: The module instance with LoRA parameters merged into the base weights.
        """
        from easydel.infra.utils import merge_lora_params

        self = merge_lora_params(self, pytree)
        return self

    def split_lora_params(self: Self) -> dict:
        """
        Splits merged LoRA parameters back out from the base model's parameters.

        This function assumes LoRA parameters were previously merged using `merge_lora_params`
        or a similar process that stored the original base weights and LoRA weights appropriately.

        Returns:
            tp.Dict: A pytree containing the extracted LoRA parameters (A and B matrices).
                The base model parameters are restored to their original (pre-merge) state.
        """
        from easydel.infra.utils import split_lora_params

        pytree = split_lora_params(self)
        return pytree

    @property
    def lora_is_enabled(self: Self):
        for _, tensor in nn.iter_graph(self):
            if isinstance(tensor, nn.LoRAParam):
                return True
        return False

    def apply_lora_to_layers(
        self: Self,
        lora_rank: int,
        lora_pattern: str | None = None,
        verbose: bool = False,
        rngs: nn.Rngs | None = None,
    ) -> Self:
        """
        Applies Low-Rank Adaptation (LoRA) layers to the specified linear layers within the module.

        Replaces targeted `flax.linen.Dense` layers with `easydel.layers.lora.LoraLinear`
        layers, initializing the LoRA matrices (A and B).

        Args:
            lora_rank (int): The rank of the LoRA decomposition.
            lora_pattern (tp.Optional[str], optional): A regular expression to match the names
                of the `Dense` layers to apply LoRA to. If None, applies to common attention
                and MLP layers. Defaults to None.
            verbose (bool, optional): If True, prints information about which layers are being
                modified. Defaults to False.
            rngs (tp.Optional[nn.Rngs], optional): JAX random number generators for initializing
                LoRA matrices. If None, default RNGs might be used. Defaults to None.

        Returns:
            Self: The module instance with LoRA layers applied.
        """
        from easydel.infra.utils import apply_lora_to_layers

        self = apply_lora_to_layers(
            self,
            lora_pattern=lora_pattern,
            lora_rank=lora_rank,
            rngs=rngs,
            verbose=verbose,
        )
        return self

    def unwrap_lora_to_layers(self: Self, verbose: bool = False) -> Self:
        """
        Reverts the application of LoRA layers, restoring the original linear layers.

        Replaces `easydel.layers.lora.LoraLinear` layers with their original `flax.linen.Dense`
        counterparts, discarding the LoRA matrices.

        Args:
            verbose (bool, optional): If True, prints information about which layers are being
                reverted. Defaults to False.

        Returns:
            Self: The module instance with LoRA layers removed and original layers restored.
        """
        from easydel.infra.utils import unwrap_lora_to_layers

        self = unwrap_lora_to_layers(self, verbose=verbose)
        return self

    @property
    def transform_fn(self):
        """Transform function with rules."""
        from easydel.layers.moe import BaseMoeModule, ParallelMoELinear
        from easydel.utils import traversals
        from easydel.utils.parameters_transformation import StateDictConverter

        embedding_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.Embed)]
        layernorm_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.LayerNorm)]
        moe_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, ParallelMoELinear)]
        moe_block_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, BaseMoeModule)]

        return partial(
            StateDictConverter.huggingface_to_easydel,
            embedding_layer_names=embedding_path,
            layernorm_names=layernorm_path,
            moe_names=list(set([names.split(".")[-1] for names in moe_path])),
            moe_block_names=list(set([names.split(".")[-1] for names in moe_block_path])),
            moe_block_path=moe_block_path,
            moe_path=moe_path,
            dtype=self.param_dtype,
            shard_fns=self._shard_fns,
        )

    @property
    def _generate_compatible_graphdef(self: Self):
        """
        Creates a graph definition compatible with generation tasks.

        Often, generation requires specific configurations (like disabling gradient checkpointing).
        This method creates a temporary, generation-compatible configuration, performs a lazy
        initialization with it, and extracts the resulting graph definition.

        Returns:
            nn.GraphDef: A graph definition suitable for use during generation.
        """
        from copy import deepcopy

        adjusted_config = deepcopy(self.config)
        adjusted_config.gradient_checkpointing = EasyDeLGradientCheckPointers.NONE
        dummy = type(self).lazy_init(
            config=adjusted_config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=self.rngs,
        )
        gdef, _, _ = nn.split(dummy, self.graphstate_type, ...)
        return gdef

    @property
    def _generate_compatible_graphother(self: Self):
        """
        Creates the 'other' state (non-parameters) compatible with generation tasks.

        Similar to `_generate_compatible_graphdef`, this creates a temporary,
        generation-compatible configuration, lazy-initializes, and extracts the 'other'
        state variables, ensuring they have concrete values instead of meta-placeholders.

        Returns:
            nn.GraphState: A graph state containing non-parameter variables suitable for generation.
        """
        from copy import deepcopy

        adjusted_config = deepcopy(self.config)
        adjusted_config.gradient_checkpointing = EasyDeLGradientCheckPointers.NONE
        dummy = type(self).lazy_init(
            config=adjusted_config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=self.rngs,
        )
        _, _, gother = nn.split(dummy, self.graphstate_type, ...)
        gother = traversals.recreate_meta_values(gother)
        return gother

    @property
    def params_sharding(self: Self) -> dict:
        """
        Retrieves the sharding annotation for each parameter in the module.

        Returns:
            tp.Dict: A nested dictionary mirroring the parameter structure, containing the
                     sharding information (e.g., NamedSharding, PartitionSpec) for each parameter,
                     or None if unsharded.
        """
        return jax.tree_util.tree_map(
            lambda x: x.sharding if hasattr(x, "sharding") else None,
            self.split_params_dict(),
        )

    def merge_params(self, tree):
        """
        Merges a given parameter state tree back into the module.

        Reconstructs the module using its existing graph definition and 'other' state,
        but replaces the parameter state with the provided `tree`.

        Args:
            tree: A pytree (likely a `nn.GraphState`) containing the parameters to merge.

        Returns:
            EasyDeLBaseModule: The module instance with the new parameters merged in.
        """
        gdef, _, gother = nn.split(self, self.graphstate_type, ...)
        self = nn.merge(gdef, tree, gother)
        return self

    def split_params(self: Self):
        """
        Splits the module and returns the parameter state.

        Uses `nnx.split` to extract the `GraphState` containing the parameters.

        Returns:
            nn.GraphState: The parameter state of the module.
        """
        return nn.split(self, self.graphstate_type, ...)[1]

    def split_params_dict(
        self,
        extract_fn: tp.Callable | None = None,
        remove_none: bool = True,
    ) -> dict:
        """
        Splits the module parameters and returns them as a nested dictionary.

        Extracts the parameter state, converts it to a plain dictionary (removing `VariableState`
        wrappers), and optionally removes entries with `None` values.

        Args:
            extract_fn (tp.Optional[tp.Callable], optional): A function to apply to each parameter
                during extraction. Defaults to None.
            remove_none (bool, optional): If True, removes key-value pairs where the value is `None`.
                Defaults to True.

        Returns:
            tp.Dict: A nested dictionary containing the module's parameters.
        """
        flat_params = flatten_dict(self.split_params().to_pure_dict(extract_fn=extract_fn))
        if remove_none:
            flat_params = {
                k: v.value if hasattr(v, "value") else v
                for k, v in flat_params.items()
                if (v.value if hasattr(v, "value") else v) is not None
            }
        else:
            flat_params = {k: v.value if hasattr(v, "value") else v for k, v in flat_params.items()}
        return unflatten_dict(flat_params)

    def merge_params_dict(self: Self, params_dict: dict) -> Self:
        """
        Merges parameters from a dictionary back into the module's state.

        Updates the module's current parameter state with values from the provided dictionary.

        Args:
            params_dict (tp.Dict): A nested dictionary containing the parameters to merge.
                The structure should match the module's parameter structure.

        Returns:
            Self: The module instance with the parameters from the dictionary merged in.

        Raises:
            KeyError: If a key from `params_dict` is not found in the module's current state.
        """
        current_state = self.split_params().flat_state()
        if not is_flatten(params_dict):
            params_dict = flatten_dict(params_dict)
        for key, value in params_dict.items():
            if key in current_state:
                current_state[key].value = value
            else:
                raise KeyError(f"Parameter key {key} not found in the current model state.")
        self = self.merge_params(unflatten_dict(current_state))
        return self

    def flops_per_token(
        self,
        sequence_length: int | None = None,
        include_loss: bool = True,
        include_backward: bool = False,
    ) -> float:
        """
        Calculates the total FLOPs (Floating Point Operations) for the module per token.

        This method should be implemented by subclasses to provide a
        module-specific FLOPs calculation.

        Returns:
            float: The total FLOPs for the module.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        from .utils import ActivationType, FlopCalcConfig, flops_per_token

        try:
            config = self.config
            text_config = getattr(config, "text_config", config)
            vision_config = getattr(config, "vision_config", config)
            if sequence_length is None:
                sequence_length = text_config.granted_mask_max_position_embedding

            num_heads = text_config.num_attention_heads
            hidden_dim = text_config.hidden_size
            fconf = FlopCalcConfig(
                hidden_dim=hidden_dim,
                intermediate_dim=text_config.intermediate_size,
                num_layers=text_config.num_hidden_layers,
                num_heads=num_heads,
                activation_type=getattr(text_config, "hidden_act", ActivationType.SILU),
                head_dim=getattr(text_config, "head_dim", hidden_dim // num_heads),
                kv_heads=getattr(text_config, "num_key_value_heads", num_heads),
                seq_len=sequence_length,
                task=self._model_task,
                vocab_size=text_config.vocab_size,
                include_loss=include_loss,
                num_labels=getattr(text_config, "num_labels", 0),
                num_experts=getattr(text_config, "num_local_experts", 0),
                num_experts_per_tok=getattr(text_config, "num_experts_per_tok", 0),
                glu=getattr(text_config, "glu_mlp", True),
                vision_hidden_dim=getattr(vision_config, "hidden_size", 0),
                vision_intermediate_dim=getattr(vision_config, "intermediate_size", 0),
                vision_num_heads=getattr(vision_config, "num_attention_heads", 0),
                vision_num_layers=getattr(vision_config, "num_hidden_layers", 0),
                vision_seq_len=getattr(vision_config, "max_position_embeddings", 0),
            )

            flops = flops_per_token(fconf)
            if include_backward:
                flops *= 3
        except Exception:
            warnings.warn("Calculating Flops Failed!", stacklevel=1)
            flops = 1
        return flops

    def _flop(self, *args, **kwargs) -> float | None:
        """
        Estimates the FLOPs (Floating Point Operations) for a single forward pass (`__call__`).

        Uses JAX's `make_jaxpr` to get the computation graph and then analyzes it
        using `easydel.infra.utils.count_flop_jaxpr` to estimate FLOPs.

        Args:
            *args: Positional arguments to pass to `__call__`.
            **kwargs: Keyword arguments to pass to `__call__`.

        Returns:
            tp.Optional[float]: The estimated FLOP count, or None if calculation fails.
        """
        from .utils import count_flop_jaxpr

        return count_flop_jaxpr(jax.make_jaxpr(self.__call__)(*args, **kwargs))

    @property
    def pure_transform_fn(self: Self):
        """
        Returns a pure transformation function for PyTorch state dicts to EasyDeL parameters.

        Similar to `transform_fn`, but this version does *not* include sharding functions.
        It identifies embedding and LayerNorm layers and returns a partial function
        (`torch_dict_to_easydel_params`) configured only with layer names and dtype.

        Returns:
            tp.Callable: A partial function for converting a PyTorch state dict without applying sharding.
        """
        from easydel.layers.moe import BaseMoeModule, ParallelMoELinear
        from easydel.utils import traversals
        from easydel.utils.parameters_transformation import StateDictConverter

        embedding_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.Embed)]
        layernorm_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.LayerNorm)]
        moe_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, ParallelMoELinear)]
        moe_block_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, BaseMoeModule)]

        return partial(
            StateDictConverter.huggingface_to_easydel,
            embedding_layer_names=embedding_path,
            layernorm_names=layernorm_path,
            moe_names=list(set([names.split(".")[-1] for names in moe_path])),
            moe_block_names=list(set([names.split(".")[-1] for names in moe_block_path])),
            moe_block_path=moe_block_path,
            moe_path=moe_path,
            dtype=self.param_dtype,
        )

    @property
    def _default_loss_config(self: Self) -> LossConfig | None:
        """
        Provides a default LossConfig for the module, if applicable.

        Subclasses can override this property to return a default `LossConfig`
        instance specific to the model's task (e.g., setting `num_labels` for
        sequence classification).

        Returns:
            tp.Optional[LossConfig]: The default loss configuration, or None.
        """
        return None

    @_default_loss_config.setter
    def _default_loss_config(self, val):
        """Setter for the default loss config (internal use)."""
        return val

    def compute_loss(
        self,
        *,
        labels: chex.Array | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
        **batch,
    ) -> tuple[tp.Any, LossMetrics]:
        """
        Computes the loss for the model given a batch of inputs and labels.

        This method performs a forward pass using the provided `batch` arguments,
        then calculates the loss using the determined `loss_function`. It handles
        potential label inference (e.g., using `input_ids` as labels for Causal LM)
        and default loss configurations.

        Args:
            labels (tp.Optional[chex.Array], optional): The target labels. If None and the task is Causal LM,
                `input_ids` from the batch might be used. Defaults to None.
            loss_config (tp.Optional[LossConfig], optional): Specific configuration for the loss calculation.
                If None, defaults might be inferred (e.g., for sequence classification). Defaults to None.
            loss_kwargs (tp.Optional[tp.Dict], optional): Additional keyword arguments to pass directly
                to the loss function. Defaults to None.
            **batch: Keyword arguments representing the input batch (e.g., `input_ids`, `attention_mask`).

        Returns:
            tp.Tuple[tp.Any, LossMetrics]: A tuple containing:
                - The model's output ( Pytree typically including logits, hidden states etc.)
                - A `LossMetrics` object containing the calculated loss and potentially other metrics.

        Raises:
            AssertionError: If labels are required for the loss function but are not provided or inferred.
            AssertionError: If sequence classification loss is used without `num_labels` in the config.
        """
        if labels is None and self.loss_function.__name__ == ForCausalLMLoss.__name__:
            labels = batch.get("input_ids", None)

        if self.loss_function.__name__ == ForSequenceClassificationLoss.__name__:
            if loss_config is None:
                assert hasattr(self.config, "num_labels"), (
                    "in order to use `SequenceClassification` Models in `EasyDeL` you first need to attach"
                    " `num_labels` to model `config`"
                )
                loss_config = LossConfig(num_labels=self.config.num_labels)

        assert labels is not None, "`labels` can not be `None` for computing loss."
        loss_kwargs = loss_kwargs or {}
        outputs = self(**batch)

        loss_output: LossMetrics = self.loss_function(
            labels=labels,
            config=loss_config,
            paxis=self.config.partition_axis,
            **loss_kwargs,
            **outputs,
            **batch,
        )
        if hasattr(outputs, "aux_loss"):
            if outputs.aux_loss is not None:
                loss_output.loss = loss_output.loss + outputs.aux_loss
        outputs = outputs.replace(loss=loss_output.loss)
        return outputs, loss_output

    def apply_lm_head(self, hidden_states: chex.Array) -> chex.Array:
        """
        Apply the language model head to transform hidden states into logits.

        Args:
            hidden_states: Input hidden states from the transformer model.
                Shape should be [..., hidden_size].

        Returns:
            Output logits over the vocabulary. Shape will be [..., vocab_size].
        """
        tie_embeddings = next(
            (
                getattr(self.config, key)
                for key in ["tie_word_embeddings", "use_lm_head", "share_input_output_layers"]
                if hasattr(self.config, key)
            ),
            False,
        )
        w = self.get_embedding().embedding.value.T if tie_embeddings else None
        return self.get_lm_head()(hidden_states, w=w)
