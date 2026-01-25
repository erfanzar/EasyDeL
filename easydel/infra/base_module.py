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

"""Base module implementation for EasyDeL models.

This module provides the core foundation for all EasyDeL neural network models,
implementing essential functionality for model initialization, parameter management,
sharding, quantization, and integration with the broader EasyDeL ecosystem.

The EasyDeLBaseModule class serves as the base class that all EasyDeL models inherit from,
providing:
- Parameter management and state handling
- Model sharding and gathering for distributed training
- Quantization and LoRA support
- Loss computation framework
- Integration with HuggingFace models
- Generation capabilities through mixins

Key Classes:
    EasyDeLBaseModule: The base class for all EasyDeL models, providing common
        functionality for parameter handling, sharding, and model operations.
    ParameterTransformRule: Data class defining rules for transforming parameter
        names and tensors, particularly useful for MoE models.

Example:
    >>> from easydel.infra import EasyDeLBaseModule, EasyDeLBaseConfig
    >>> import flax.nnx as nn
    >>>
    >>> class MyModel(EasyDeLBaseModule):
    ...     def __init__(self, config, dtype, param_dtype, precision, rngs):
    ...         super().__init__(config, dtype, param_dtype, precision, rngs)
    ...         # Initialize model layers
    ...         self.layer = nn.Linear(config.hidden_size, config.hidden_size)
    ...
    ...     def __call__(self, inputs):
    ...         return self.layer(inputs)
    >>>
    >>> # Create and use the model
    >>> config = EasyDeLBaseConfig(hidden_size=768)
    >>> model = MyModel(
    ...     config=config,
    ...     dtype=jnp.float32,
    ...     param_dtype=jnp.float32,
    ...     precision='highest',
    ...     rngs=nn.Rngs(0)
    ... )

The module integrates with JAX's sharding system for distributed training,
supports various quantization methods, and provides utilities for converting
between EasyDeL and HuggingFace model formats.
"""

from __future__ import annotations

import hashlib
import re
import typing as tp
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial
from re import Pattern
from typing import Self, Unpack

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
from jaxtyping import Array, Float, Int

from easydel.infra.utils import ArrayParam
from easydel.utils import traversals
from easydel.utils.traversals import flatten_dict, is_flatten, unflatten_dict

from .base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict
from .etils import EasyDeLGradientCheckPointers
from .loss_utils import LOSS_MAPPING, ForCausalLMLoss, ForSequenceClassificationLoss, LossConfig, LossMetrics
from .mixins import BaseModuleProtocol, EasyBridgeMixin, EasyGenerationMixin, OperationCacheMixin
from .modeling_outputs import EmbeddingInfo

if tp.TYPE_CHECKING:
    from easydel.infra.base_state import EasyDeLState
    from easydel.layers.components import Embed, ParallelLinear, QuantizationConfig


PartitionLike = tp.Mapping[str, tp.Callable] | tp.Mapping[tuple, tp.Callable] | None
"""Type alias for partition rule specifications.

Can be a mapping from parameter name patterns (as strings or tuples) to
partition specification functions, or None for default partitioning.
"""


logger = get_logger(__name__)

BaseConf = EasyDeLBaseConfig
"""Alias for EasyDeLBaseConfig for backward compatibility."""


@dataclass
class ParameterTransformRule:
    """Rule for transforming parameter names and tensors during model conversion.

    This dataclass defines transformation rules that can be applied to parameter
    names and their associated tensor values during model conversion or loading.
    It is particularly useful for handling Mixture of Experts (MoE) models where
    parameter naming conventions may differ between frameworks.

    Attributes:
        pattern: Regular expression pattern or string to match parameter names.
            Can be a compiled Pattern object or a string that will be used for
            matching against parameter paths during conversion.
        replacement: String to replace matched patterns in parameter names.
            Supports regex replacement syntax (e.g., r'\\1' for capture groups).
        tensor_transform: Optional callable to transform the tensor values.
            Should take a tensor and return a transformed tensor of potentially
            different shape or dtype. If None, no transformation is applied
            to the tensor values, only the name is changed.
        consolidate_experts: Whether to consolidate multiple expert parameters
            into a single tensor. When True, parameters matching the pattern
            from multiple experts will be stacked into a single array with
            an additional expert dimension. Defaults to False.

    Example:
        >>> # Rule to rename and transpose expert weights
        >>> rule = ParameterTransformRule(
        ...     pattern=r"expert_(\\d+)\\.weight",
        ...     replacement=r"experts.\\1.weight",
        ...     tensor_transform=lambda x: x.transpose(),
        ...     consolidate_experts=True
        ... )
        >>>
        >>> # Simple renaming rule without tensor transformation
        >>> simple_rule = ParameterTransformRule(
        ...     pattern="old_layer_name",
        ...     replacement="new_layer_name"
        ... )

    Note:
        When consolidate_experts is True, the pattern should capture the expert
        index so that parameters can be properly grouped and stacked.
    """

    pattern: str | Pattern
    replacement: str
    tensor_transform: Callable | None = None
    consolidate_experts: bool = False


class EasyDeLBaseModule(nn.Module, EasyBridgeMixin, EasyGenerationMixin, OperationCacheMixin, BaseModuleProtocol):
    """Base class for all EasyDeL neural network modules.

    EasyDeLBaseModule provides the foundational functionality for all EasyDeL models,
    including parameter management, distributed training support, quantization,
    LoRA adaptation, and integration with HuggingFace models. It inherits from
    flax.nnx.Module and multiple mixins that provide additional capabilities.

    This class should be subclassed to create specific model architectures. Subclasses
    must implement the __call__ method and may override various hooks for customization.

    Attributes:
        config_class: The configuration class associated with this model type.
            Should be a subclass of EasyDeLBaseConfig.
        base_model_prefix: String prefix used when loading/saving weights,
            typically matches the HuggingFace model prefix.
        config: The model configuration instance containing architecture parameters.
        _model_task: String identifier for the model's task (e.g., 'causal-language-model').
            Set automatically based on the model class.
        _model_type: String identifier for the model type (e.g., 'llama', 'mistral').
            Set automatically based on the configuration.
        _parameter_transform_rules: Class-level list of ParameterTransformRule instances
            for handling parameter name/value transformations during conversion.

    Example:
        >>> class MyCustomModel(EasyDeLBaseModule):
        ...     config_class = MyCustomConfig
        ...     base_model_prefix = "model"
        ...
        ...     def __init__(self, config, dtype, param_dtype, precision, rngs):
        ...         super().__init__(config, dtype, param_dtype, precision, rngs)
        ...         self.embed = nn.Embed(config.vocab_size, config.hidden_size)
        ...         self.layers = [
        ...             TransformerBlock(config, dtype, param_dtype, precision, rngs)
        ...             for _ in range(config.num_hidden_layers)
        ...         ]
        ...
        ...     def __call__(self, input_ids, attention_mask=None):
        ...         hidden_states = self.embed(input_ids)
        ...         for layer in self.layers:
        ...             hidden_states = layer(hidden_states, attention_mask)
        ...         return hidden_states

    Note:
        - Always call super().__init__() in subclass constructors to properly
          initialize base functionality.
        - The mesh and partition rules from config are used for distributed training.
        - Use lazy_init() for memory-efficient model initialization.
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

        Sets up the base module with configuration and data types. This method
        initializes core attributes and triggers the computation of various
        cached properties. Subclasses should call this in their __init__ method
        before initializing their own layers.

        Args:
            config: Model configuration object containing architecture parameters
                such as hidden_size, num_layers, etc. Must be an instance of
                EasyDeLBaseConfig or a subclass.
            dtype: Data type for computations during forward pass. Common choices
                include jnp.float32, jnp.bfloat16, or jnp.float16.
            param_dtype: Data type for storing model parameters. May differ from
                dtype for mixed-precision training (e.g., bfloat16 params with
                float32 computation).
            precision: JAX precision setting for matrix operations. Can be 'highest',
                'high', 'default', or a jax.lax.Precision enum value.
            rngs: Flax random number generator container for parameter initialization
                and stochastic operations like dropout.

        Example:
            >>> config = LlamaConfig(hidden_size=1024, num_hidden_layers=4)
            >>> model = LlamaModel(
            ...     config=config,
            ...     dtype=jnp.bfloat16,
            ...     param_dtype=jnp.bfloat16,
            ...     precision='high',
            ...     rngs=nn.Rngs(42)
            ... )

        Note:
            This method should be called by all subclasses to properly
            initialize the base functionality. Failing to call super().__init__()
            will result in missing core attributes.
        """
        self.config: BaseConf = config
        self.dtype: jnp.dtype = dtype
        self.param_dtype: jnp.dtype = param_dtype
        self.precision: lax.PrecisionLike = precision
        self.rngs: nn.Rngs = rngs

        _ = self.graphtree_shape
        _ = self.graphtree_params_shape
        _ = self.mesh
        _ = self.model_task
        _ = self.model_type

    @property
    def parameters(self: Self) -> dict:
        """Retrieve the module's parameters as a nested dictionary.

        This property iterates through the module and its submodules, extracting
        all variables marked as nn.Param and returning them in a flat dictionary
        where keys represent the parameter path (e.g., 'layers.0.attention.qkv.kernel').

        Returns:
            dict: A dictionary mapping parameter paths (strings) to their values
                (JAX arrays). The paths use dot notation to represent the module
                hierarchy.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> params = model.parameters
            >>> print(params.keys())
            dict_keys(['embed.embedding', 'layers.0.attention.q_proj.kernel', ...])
            >>> print(params['embed.embedding'].shape)
            (32000, 4096)

        Note:
            This creates a new dictionary on each access. For performance-critical
            code that needs repeated parameter access, consider caching the result.
        """
        from easydel.utils.traversals import iter_module_search

        parameters = {}
        for key, value in iter_module_search(self, nn.Param):
            parameters[key] = value.value
        return parameters

    @property
    def graphstate_type(self: Self):
        """Determine the parameter type based on LoRA enablement status.

        Returns the appropriate parameter type class for use with nnx.split().
        If LoRA is enabled on the model, returns nn.LoRAParam to properly
        separate LoRA parameters; otherwise returns nn.Param.

        Returns:
            type: nn.LoRAParam if LoRA is enabled on this model instance,
                otherwise nn.Param.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> print(model.graphstate_type)
            <class 'flax.nnx.variables.Param'>
            >>>
            >>> model = model.apply_lora_to_layers(lora_rank=8)
            >>> print(model.graphstate_type)
            <class 'flax.nnx.lora.LoRAParam'>
        """
        return nn.LoRAParam if self.lora_is_enabled else nn.Param

    def split_module(self: Self):
        """Split the module into graph definition and state components.

        Uses flax.nnx.split to decompose the module into its structural definition
        (GraphDef) and two state components: parameters and other state variables.
        This is useful for functional transformations, serialization, and
        manipulation of model state.

        Returns:
            tuple: A tuple of (GraphDef, GraphState, GraphState) where:
                - GraphDef: The module's structure without values
                - GraphState (first): The parameter state (nn.Param or nn.LoRAParam)
                - GraphState (second): Other state variables (non-parameters)

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> graphdef, params, others = model.split_module()
            >>> # Modify parameters
            >>> modified_params = jax.tree_map(lambda x: x * 0.5, params)
            >>> # Reconstruct model
            >>> new_model = model.merge_module(graphdef, modified_params, others)
        """
        return nn.split(self, self.graphstate_type, ...)

    @staticmethod
    def merge_module(graphdef: nn.GraphDef, graphstate: nn.GraphState, graphother: nn.GraphState):
        """Merge graph components back into a complete module.

        Reconstructs a complete module instance from its decomposed components.
        This is the inverse operation of split_module().

        Args:
            graphdef: The module's graph definition containing the structural
                information (layer types, connections) without parameter values.
            graphstate: The module's parameter state containing the trainable
                parameters (weights and biases).
            graphother: The module's non-parameter state containing other
                variables like batch statistics or cached values.

        Returns:
            EasyDeLBaseModule: The reconstructed module instance with all
                components merged back together.

        Example:
            >>> graphdef, params, others = model.split_module()
            >>> # Apply some transformation to parameters
            >>> new_params = jax.tree_map(lambda x: x.astype(jnp.float16), params)
            >>> # Merge back into a complete model
            >>> new_model = EasyDeLBaseModule.merge_module(graphdef, new_params, others)
        """
        return nn.merge(graphdef, graphstate, graphother)

    @property
    def graphdef(self: Self) -> nn.GraphDef:
        """Get the graph definition (structure without parameters) of the module.

        Uses flax.nnx.split to separate the graph definition from the state.
        The graph definition contains the module's structural information
        including layer types, shapes, and connections, but no parameter values.

        Returns:
            nn.GraphDef: The graph definition of the module, which can be used
                to reconstruct the module structure or for functional transformations.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> gdef = model.graphdef
            >>> # The graphdef can be used with different parameter states
            >>> new_model = nn.merge(gdef, new_params, new_others)
        """
        return nn.split(self, self.graphstate_type, ...)[0]

    @property
    def graphstate(self: Self) -> nn.GraphState:
        """Get the graph state (parameters) of the module.

        Uses flax.nnx.split to separate the parameter state from the graph
        definition. Returns the trainable parameters as a GraphState object
        that can be manipulated, serialized, or used in functional transformations.

        Returns:
            nn.GraphState: The graph state containing the module's trainable
                parameters (weights, biases, embeddings, etc.).

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> state = model.graphstate
            >>> # Access individual parameters
            >>> flat_state = state.flat_state()
            >>> print(flat_state.keys())
        """
        return nn.split(self, self.graphstate_type, ...)[1]

    @property
    def graphother(self: Self) -> nn.GraphState:
        """Get any other state variables in the module (non-parameters).

        Uses flax.nnx.split to separate non-parameter state variables from
        the graph definition and parameters. This includes variables like
        batch normalization statistics, cached values, or other mutable state.

        Returns:
            nn.GraphState: The graph state containing non-parameter variables
                such as batch statistics, caches, or other mutable state.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> other_state = model.graphother
            >>> # May contain cached values or other non-trainable state
        """
        return nn.split(self, self.graphstate_type, ...)[-1]

    @property
    def graphtree_params_shape(self: Self) -> dict:
        """Compute the shapes of the module's parameters as a nested dictionary.

        Uses nnx.eval_shape to determine parameter shapes without allocating
        memory for the actual parameter values. This is useful for inspecting
        model architecture, calculating memory requirements, or preparing
        sharding specifications.

        Returns:
            dict: A nested dictionary mirroring the parameter structure, where
                each leaf contains a jax.ShapeDtypeStruct with shape and dtype
                information instead of actual array values.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> shapes = model.graphtree_params_shape
            >>> print(shapes['layers']['0']['attention']['q_proj']['kernel'])
            ShapeDtypeStruct(shape=(4096, 4096), dtype=float32)
        """
        graphtree = nn.eval_shape(lambda: nn.split(self, self.graphstate_type, ...)[1])

        flattened_tree = flatten_dict(graphtree)

        param_shapes = {key: val.value for key, val in flattened_tree.items()}
        return unflatten_dict(param_shapes)

    @property
    def graphtree_shape(self: Self) -> dict:
        """Compute the shapes of all state variables (including non-parameters).

        Uses nnx.eval_shape on the entire module state (both parameters and
        other variables) and extracts shape information. This provides a
        complete view of all arrays in the model.

        Returns:
            dict: A nested dictionary mirroring the module's complete state
                structure, where each leaf contains shape and dtype information
                for all variables, not just trainable parameters.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> all_shapes = model.graphtree_shape
            >>> # Includes both parameters and other state like batch stats
        """
        graphtree = nn.eval_shape(lambda: nn.split(self)[1])

        flattened_tree = flatten_dict(graphtree)

        param_shapes = {key: getattr(val, "value", val) for key, val in flattened_tree.items()}
        return unflatten_dict(param_shapes)

    @property
    def mesh(self: Self) -> jax.sharding.Mesh:
        """Get the JAX device mesh from the module's configuration.

        Returns the mesh used for distributed training and sharding operations.
        The mesh defines how arrays are partitioned across devices.

        Returns:
            jax.sharding.Mesh: The device mesh defined in self.config.mesh,
                which specifies the topology of devices for data and model
                parallelism.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> print(model.mesh.devices.shape)
            (8,)  # Example: 8 devices in the mesh
            >>> print(model.mesh.axis_names)
            ('dp', 'fsdp', 'tp', 'sp')
        """
        return self.config.mesh

    @property
    def explicit_mesh(self: Self) -> jax.sharding.Mesh:
        """Get the explicit-axis JAX device mesh from the module's configuration.

        Returns the explicit mesh variant where axes are explicitly named
        and managed. This is useful for advanced sharding strategies.

        Returns:
            jax.sharding.Mesh: The explicit-axis device mesh defined in
                self.config.explicit_mesh.
        """
        return self.config.explicit_mesh

    @property
    def manual_mesh(self: Self) -> jax.sharding.Mesh:
        """Get the manual-axis JAX device mesh from the module's configuration.

        Returns the manual mesh variant where axis handling is done manually
        by the user. This provides maximum flexibility for custom sharding.

        Returns:
            jax.sharding.Mesh: The manual-axis device mesh defined in
                self.config.manual_mesh.
        """
        return self.config.manual_mesh

    def mesh_call(self: Self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Call the module under the configured JAX mesh context.

        This is a convenience method equivalent to `with self.mesh: self(*args, **kwargs)`.
        It ensures that all operations within __call__ respect the mesh sharding
        configuration.

        Args:
            *args: Positional arguments to pass to __call__.
            **kwargs: Keyword arguments to pass to __call__.

        Returns:
            Any: The output from __call__, with appropriate sharding applied
                based on the mesh configuration.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> # These are equivalent:
            >>> output1 = model.mesh_call(input_ids, attention_mask=mask)
            >>> with model.mesh:
            ...     output2 = model(input_ids, attention_mask=mask)

        Note:
            This method uses self.mesh only. For explicit_mesh or manual_mesh,
            enter those contexts explicitly when needed.
        """
        with self.mesh:
            return self(*args, **kwargs)

    @property
    def model_task(self: Self) -> str | None:
        """Get the task identifier for this model instance.

        Returns the specific task this model is designed for, such as
        'causal-language-model', 'sequence-classification', etc. This is
        used for selecting appropriate loss functions and training procedures.

        Returns:
            str | None: The model task identifier string, or None if not set.
                Common values include 'ForCausalLM', 'ForSequenceClassification',
                'ForTokenClassification', etc.

        Example:
            >>> model = LlamaForCausalLM(config, dtype, param_dtype, precision, rngs)
            >>> print(model.model_task)
            'ForCausalLM'
        """
        return self._model_task

    @property
    def model_type(self: Self) -> str | None:
        """Get the model type identifier for this model instance.

        Returns the specific architecture type of this model, such as
        'llama', 'mistral', 'qwen2', etc. This is typically derived from
        the configuration and used for model identification.

        Returns:
            str | None: The model type identifier string, or None if not set.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> print(model.model_type)
            'llama'
        """
        return self._model_type

    @property
    def params(self: Self) -> dict:
        """Get the parameters and other state variables as a dictionary.

        Uses flax.nnx.split to get the combined state (both parameters and
        other variables) as a single dictionary.

        Returns:
            dict: A dictionary containing all state variables of the module,
                including both trainable parameters and other state.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> all_state = model.params
        """
        return nn.split(self)[-1]

    @cached_property
    def causal_mask(self: Self) -> jnp.ndarray:
        """Get or compute the basic causal attention mask from configuration.

        Retrieves the causal attention mask from the configuration, computing
        and caching it on first access. The mask prevents attention to future
        positions in autoregressive models.

        Returns:
            jnp.ndarray: The causal attention mask with shape typically
                (1, 1, max_position_embeddings, max_position_embeddings).
                Values are 0 for positions that can be attended to and
                large negative values for positions that should be masked.

        Note:
            This is a cached property - the mask is computed once and stored
            for subsequent accesses.
        """
        return self.config.get_basic_causal_mask()

    @cached_property
    def frequencies(self: Self) -> jnp.ndarray:
        """Get or compute the frequency components for rotary embeddings.

        Retrieves the frequency components used in Rotary Position Embeddings
        (RoPE) from the configuration, computing and caching on first access.

        Returns:
            jnp.ndarray: The frequency components for RoPE with shape
                (max_position_embeddings, head_dim // 2) or similar depending
                on the RoPE implementation.

        Note:
            This is a cached property - frequencies are computed once and
            stored for subsequent accesses.
        """
        return self.config.get_basic_frequencies()

    @cached_property
    def inv_frequencies(self: Self) -> jnp.ndarray:
        """Get or compute the inverse frequency components for rotary embeddings.

        Retrieves the inverse frequency components used in Rotary Position
        Embeddings (RoPE) from the configuration, computing and caching on
        first access.

        Returns:
            jnp.ndarray: The inverse frequency components for RoPE.

        Note:
            This is a cached property - inverse frequencies are computed once
            and stored for subsequent accesses.
        """
        return self.config.get_basic_inv_frequencies()

    @cached_property
    def static_arguments(self: Self) -> tuple:
        """Get static arguments needed by the module's __call__ method.

        Retrieves static arguments that don't change during execution and can
        be pre-computed. These are typically used for JIT compilation optimization.

        Returns:
            tuple: A tuple of static arguments. The default implementation
                returns an empty tuple; subclasses should override this if
                they have static arguments.

        Note:
            This is a cached property - arguments are computed once and stored.
        """
        return self.get_static_arguments()

    @cached_property
    def lossfn_type(self: Self):
        """Determine the loss function type for this model.

        Determines the appropriate loss function type based on (in order of
        priority):
        1. config.loss_type attribute if set
        2. self.loss_type attribute if set
        3. Class name pattern matching against known loss types
        4. Defaults to 'ForCausalLM' if not determined

        Returns:
            str: String identifier for the loss function type (e.g., 'ForCausalLM',
                'ForSequenceClassification', 'ForTokenClassification').

        Note:
            If an unrecognized loss_type is set in config, a warning is issued
            and 'ForCausalLM' is used as fallback.
        """
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
        """Get the appropriate loss function based on configuration or model type.

        Determines and returns the loss function class based on the lossfn_type.
        The function is looked up in the LOSS_MAPPING registry.

        Returns:
            Callable: The selected loss function class (e.g., ForCausalLMLoss,
                ForSequenceClassificationLoss) that can be called to compute
                the loss given model outputs and labels.

        Example:
            >>> model = LlamaForCausalLM(config, dtype, param_dtype, precision, rngs)
            >>> loss_fn = model.loss_function
            >>> print(loss_fn.__name__)
            'ForCausalLMLoss'
        """

        return LOSS_MAPPING[self.lossfn_type]

    @property
    def module_dtype(self: Self) -> jnp.dtype:
        """Determine the data type of the module's parameters.

        Inspects the flattened parameter state to find the dtype of the first
        parameter encountered. This reflects the actual storage dtype of
        the model's parameters.

        Returns:
            jnp.dtype: The data type of the module's parameters (e.g.,
                jnp.float32, jnp.bfloat16, jnp.float16).

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> print(model.module_dtype)
            bfloat16
        """
        params_state = nn.split(self, self.graphstate_type, ...)[1].flat_state()
        return jax.tree_util.tree_leaves(params_state)[0].dtype

    def compute_complex_rotary(self, position_ids: jax.Array) -> jnp.ndarray:
        """Compute complex-valued rotary position embeddings.

        Computes the complex exponential of frequencies for rotary embeddings
        given position indices. This is used in models that use complex-valued
        RoPE implementation.

        Args:
            position_ids: Position indices to compute embeddings for, with
                shape (batch_size, sequence_length).

        Returns:
            jnp.ndarray: Complex exponential of frequencies with shape
                (batch_size, sequence_length, head_dim // 2). The result
                contains complex numbers that can be used to rotate query
                and key vectors.

        Example:
            >>> position_ids = jnp.arange(128)[None, :]  # (1, 128)
            >>> freqs_cis = model.compute_complex_rotary(position_ids)
            >>> print(freqs_cis.shape)
            (1, 128, 64)  # Assuming head_dim=128
        """
        frequencies = jnp.transpose(
            self.inv_frequencies[None, :, None] @ position_ids[:, None, :].astype("f4"),
            (0, 2, 1),
        )
        return jnp.exp(1j * frequencies)

    def to_dtype(self: Self, dtype: jnp.dtype) -> Self:
        """Convert the module's parameters to the specified data type.

        Iterates through the module's parameters (excluding quantization-related
        ones like quant_*) and casts them to the target dtype. Also updates the
        param_dtype attribute of the module and all its submodules.

        Args:
            dtype: The target data type for the parameters (e.g., jnp.float32,
                jnp.bfloat16, jnp.float16).

        Returns:
            Self: The module instance with parameters converted to the specified
                dtype. Note that this returns a new module instance.

        Example:
            >>> model = LlamaModel(config, jnp.float32, jnp.float32, precision, rngs)
            >>> model = model.to_dtype(jnp.bfloat16)
            >>> print(model.module_dtype)
            bfloat16

        Note:
            Quantization-related parameters (those starting with 'quant_') are
            not converted to preserve their specific formats.
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
        """Convert the module's parameters to half-precision (float16).

        Convenience method to convert all parameters to float16. Optionally
        also changes the runtime computation dtype to float16.

        Args:
            change_runtime_dtype: If True, also sets self.dtype to jnp.float16
                for runtime computations. Defaults to True.

        Returns:
            Self: The module instance with parameters (and potentially runtime
                dtype) set to float16.

        Example:
            >>> model = LlamaModel(config, jnp.float32, jnp.float32, precision, rngs)
            >>> model = model.half()
            >>> print(model.module_dtype)
            float16

        Note:
            For better numerical stability on TPUs, consider using bfloat16
            instead of float16.
        """
        if change_runtime_dtype:
            self = self._reformat_runtime_dtype(jnp.float16)
        return self._reformat_dtype(jnp.float16)

    def float(self: Self, change_runtime_dtype: bool = True) -> Self:
        """Convert the module's parameters to single-precision (float32).

        Convenience method to convert all parameters to float32. Optionally
        also changes the runtime computation dtype to float32.

        Args:
            change_runtime_dtype: If True, also sets self.dtype to jnp.float32
                for runtime computations. Defaults to True.

        Returns:
            Self: The module instance with parameters (and potentially runtime
                dtype) set to float32.

        Example:
            >>> model = model.float()  # Convert to float32
            >>> print(model.module_dtype)
            float32
        """
        if change_runtime_dtype:
            self = self._reformat_runtime_dtype(jnp.float32)
        return self._reformat_dtype(jnp.float32)

    def _reformat_runtime_dtype(self: Self, dtype) -> Self:
        """Change the runtime computation dtype of the module and submodules.

        Internal helper method that updates the dtype attribute (used for
        computations during forward pass) of this module and all its submodules.

        Args:
            dtype: The target runtime data type (e.g., jnp.float32, jnp.bfloat16).

        Returns:
            Self: The module instance with updated runtime dtype.

        Note:
            This is an internal method. Use half(), float(), or to_dtype()
            for the public API.
        """
        from easydel.utils.traversals import iter_module_search

        for _path, module in iter_module_search(self):
            if hasattr(module, "dtype"):
                if str(type(module.dtype)).endswith("lax_numpy._ScalarMeta'>"):  # dont change numpy based dtypes
                    module.dtype = dtype
        self.dtype = dtype
        return self

    def _reformat_dtype(self: Self, dtype) -> Self:
        """Change the data type of the module's parameters.

        Internal helper method that casts all floating-point parameters to
        the target dtype. Non-floating-point parameters are left unchanged.

        Args:
            dtype: The target parameter data type (e.g., jnp.float32, jnp.bfloat16).

        Returns:
            Self: The module instance with updated parameter dtype.

        Note:
            This is an internal method. Use half(), float(), or to_dtype()
            for the public API.
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
        """Match partition rules against the module's parameter shapes.

        Matches the provided or configured partition rules against the module's
        parameter tree to generate PartitionSpec assignments for each parameter.

        Args:
            partition_rules: The partition rules to use for matching. If None,
                uses rules from self.config.get_partition_rules(). Defaults to None.

        Returns:
            dict: A nested dictionary mapping parameter paths to PartitionSpec
                objects that define how each parameter should be sharded.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> specs = model._match_partition_rules()
            >>> print(specs['layers']['0']['attention']['q_proj']['kernel'])
            PartitionSpec('fsdp', 'tp')
        """
        return match_partition_rules(
            rules=self._get_partition_rules(partition_rules),
            tree=self.graphtree_params_shape,
        )

    @property
    def _specs_sharding(self: Self):
        """Extract the PartitionSpec from each parameter's NamedSharding.

        Returns a nested dictionary where each leaf contains the PartitionSpec
        portion of the parameter's sharding annotation, or an empty PartitionSpec
        if the parameter is not sharded.

        Returns:
            dict: A nested dictionary mirroring the parameter structure,
                containing PartitionSpec objects for each parameter.
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
        """Extract the sharding information for each parameter.

        Returns a nested dictionary containing the sharding information
        (PartitionSpec or NamedSharding) for each parameter in the module.

        Returns:
            dict: A nested dictionary mirroring the parameter structure,
                containing the sharding info (or empty PartitionSpec if unsharded).
        """
        return nn.from_tree(
            jax.tree_util.tree_map(
                lambda x: x.sharding if hasattr(x, "sharding") else PartitionSpec(),
                nn.to_tree(self),
            )
        )

    @property
    def _named_shardings(self: Self):
        """Extract the NamedSharding object for each parameter.

        Returns a nested dictionary containing the NamedSharding object
        (if present) for each parameter, or None for unsharded parameters.

        Returns:
            dict: A nested dictionary mirroring the parameter structure,
                containing NamedSharding objects or None.
        """
        return nn.from_tree(
            jax.tree_util.tree_map(
                lambda x: x.sharding if hasattr(x, "sharding") else None,
                nn.to_tree(self),
            )
        )

    def _get_mesh(self, mesh: Mesh | None = None) -> Mesh:
        """Retrieve the JAX device mesh, with fallback to configuration.

        Gets the mesh to use for sharding operations, prioritizing the provided
        argument over the mesh in the configuration.

        Args:
            mesh: A JAX device mesh to use. If None, uses self.config.mesh.

        Returns:
            Mesh: The resolved JAX device mesh.

        Raises:
            ValueError: If no mesh is provided and none is found in the
                configuration (self.config.mesh is None or config doesn't exist).

        Example:
            >>> mesh = model._get_mesh()  # Uses config mesh
            >>> custom_mesh = Mesh(devices, axis_names)
            >>> mesh = model._get_mesh(custom_mesh)  # Uses provided mesh
        """
        if mesh is None:
            if not hasattr(self, "config") or not hasattr(self.config, "mesh") or self.config.mesh is None:
                raise ValueError("A mesh must be provided, either as an argument or through the model config.")
            return self.config.mesh
        return mesh

    def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
        """Retrieve partition rules, with fallback to configuration.

        Gets the partition rules to use for sharding, prioritizing the provided
        argument over rules from the configuration.

        Args:
            partition_rules: Partition rules to use. If None, calls
                self.config.get_partition_rules(fully_sharded_data_parallel=True).

        Returns:
            PartitionLike: The resolved partition rules as a mapping from
                parameter name patterns to PartitionSpec generators.

        Raises:
            ValueError: If no rules are provided and the configuration doesn't
                support partition rule generation.

        Example:
            >>> rules = model._get_partition_rules(None)  # Uses config rules
            >>> custom_rules = [(".*kernel", ("fsdp", "tp"))]
            >>> rules = model._get_partition_rules(custom_rules)  # Uses provided
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
        """Apply sharding or gathering functions to the module's parameters.

        Internal method that applies a mapping of functions to transform
        parameters. Used by shard_model() and gather_model() to distribute
        or collect parameters across devices.

        Args:
            sharding_fns: A mapping from flattened parameter paths (tuples)
                to transformation functions. Each function takes a parameter
                array and returns a transformed (sharded or gathered) array.

        Returns:
            Self: The module instance with sharding/gathering functions applied
                to its parameters.

        Note:
            Parameters that are not callable (e.g., pre-sharded NF4 arrays)
            are left unchanged.
        """
        gdef, state, others = nn.split(self, self.graphstate_type, ...)
        sharding_fns = flatten_dict(sharding_fns)
        _shard_keys = list(sharding_fns.keys())

        def _map(path, val: nn.VariableState):
            if val.value is not None and path in _shard_keys:
                fn = sharding_fns[path]
                if callable(fn):
                    val.value = fn(val.value)
                else:
                    # NOTE: It should smt like NF4 or 8bit array if it's not callable and that mean it's already pre-sharded.
                    ...
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
        """Shard the model's parameters according to partition rules and mesh.

        Distributes the model's parameters across devices according to the
        specified partition rules and device mesh. This is the primary method
        for preparing a model for distributed training.

        Args:
            partition_rules: Partitioning rules specifying how to shard each
                parameter. If None, uses rules from config. Defaults to None.
            mesh: JAX device mesh defining the device topology. If None, uses
                mesh from config. Defaults to None.
            overlay_fns: Additional transformation functions that override
                the default sharding for specific parameters. Keys are parameter
                paths, values are transformation functions. Defaults to None.

        Returns:
            Self: The model instance with sharded parameters.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> # Shard using config's mesh and rules
            >>> model = model.shard_model()
            >>>
            >>> # Shard with custom mesh
            >>> devices = jax.devices()
            >>> mesh = Mesh(devices, ('dp',))
            >>> model = model.shard_model(mesh=mesh)

        Note:
            After sharding, each parameter will be distributed across devices
            according to its PartitionSpec. Access to the full parameter
            requires gathering (see gather_model()).
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
        """Gather the model's parameters from distributed devices to host.

        Collects sharded parameters from across devices and consolidates them,
        typically to a single device or the host. This is the inverse of
        shard_model() and is useful for saving checkpoints or inference.

        Args:
            partition_rules: Partitioning rules that were used to shard the
                parameters. If None, uses rules from config. Defaults to None.
            mesh: JAX device mesh from which to gather parameters. If None,
                uses mesh from config. Defaults to None.
            overlay_fns: Additional transformation functions that override
                the default gathering for specific parameters. Defaults to None.

        Returns:
            Self: The model instance with gathered (non-distributed) parameters.

        Example:
            >>> # After distributed training
            >>> model = model.gather_model()
            >>> # Now parameters are on a single device and can be saved
            >>> model.save_pretrained("checkpoint/")

        Note:
            Gathering is typically slower than keeping parameters distributed,
            so it should only be done when necessary (e.g., checkpointing).
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
        """Generate sharding functions based on the module's configuration.

        Creates a dictionary of sharding functions that can be used to
        distribute parameters across devices according to the partition rules.

        Returns:
            Mapping: A mapping from flattened parameter paths to sharding
                functions that transform arrays to their sharded form.
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
        """Generate gathering functions based on the module's configuration.

        Creates a dictionary of gathering functions that can be used to
        collect distributed parameters back to a single device.

        Returns:
            Mapping: A mapping from flattened parameter paths to gathering
                functions that collect sharded arrays.
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
        """Apply output sharding specifications to the module state.

        Uses JIT compilation with out_shardings to enforce specific sharding
        constraints on the module's state.

        Args:
            out_shardings: Sharding specifications to apply to the module's
                graphstate and graphother components.

        Returns:
            Self: Module with sharding constraints applied to its state.

        Example:
            >>> shardings = jax.tree_map(
            ...     lambda x: NamedSharding(mesh, PartitionSpec()),
            ...     model.split_module()[1:]
            ... )
            >>> model = model.apply_out_shardings(shardings)
        """
        splits = self.split_module()

        @partial(jax.jit, out_shardings=out_shardings)
        def _call(graphstate, graphother):
            return graphstate, graphother

        splits[1:] = _call(*splits[1:])
        return self.merge_module(*splits)

    def fully_shard(self: Self, partition_rules: PartitionLike = None) -> Self:
        """Apply JAX sharding constraints to all parameters.

        Ensures that all parameters are explicitly marked with their intended
        sharding based on partition rules. Uses jax.jit with out_shardings
        to enforce the constraints. This is useful for performance optimization
        and correctness verification.

        Args:
            partition_rules: Partitioning rules to use. If None, uses rules
                from config. Defaults to None.

        Returns:
            Self: The model instance with explicit sharding constraints applied
                to all parameters.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> model = model.fully_shard()
            >>> # All parameters now have explicit sharding annotations
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
        """Apply JAX sharding constraints to gather all parameters.

        Marks all parameters to have no sharding (PartitionSpec()), effectively
        gathering them to the host or a single device. Uses jax.jit with
        out_shardings to enforce these gathering constraints.

        Returns:
            Self: The model instance with gathering constraints applied,
                where all parameters are replicated (not sharded).

        Example:
            >>> model = model.fully_gather()
            >>> # All parameters are now replicated across devices
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
        quantization_config: QuantizationConfig | None = None,
        quantize_tensors: bool = False,
        quantize_modules: bool | None = None,
        verbose: bool | None = None,
        raise_error: bool = True,
    ) -> Self:
        """Apply quantization to the module's linear layers or tensors.

        Quantizes the model using the specified configuration. Can either
        replace Linear layers with their quantized equivalents (module-level)
        or quantize tensor values directly (tensor-level).

        Args:
            quantization_config: Configuration specifying quantization dtype,
                block_size, and pattern. If None, uses default INT8 quantization.
            quantize_tensors: If True, quantizes tensor values directly without
                changing module structure. Defaults to False.
            quantize_modules: If True, replaces Linear layers with quantized
                equivalents (e.g., Linear8bit, LinearNF4). Defaults to True
                when quantize_tensors is False.
            verbose: If True, logs information during quantization. Defaults
                to True only on process index 0.
            raise_error: If True, raises error when both quantize_tensors and
                quantize_modules are False. Defaults to True.

        Returns:
            Self: The quantized model instance.

        Raises:
            ValueError: If both quantize_tensors and quantize_modules are True.
            ValueError: If both are False and raise_error is True.

        Example:
            >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
            >>> # INT8 quantization
            >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
            >>> model = model.quantize(quantization_config=config)
            >>>
            >>> # NF4 quantization with custom block size
            >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
            >>> model = model.quantize(quantization_config=config)

        Note:
            Module-level quantization (quantize_modules=True) typically provides
            better performance as it can fuse dequantization with computation.
        """
        from easydel.layers.components import EasyQuantizer, QuantizationConfig, QuantizationType

        if quantization_config is None:
            quantization_config = QuantizationConfig(dtype=QuantizationType.INT8)

        quantizer = EasyQuantizer(quantization_config=quantization_config)
        if quantize_modules is None:
            quantize_modules = not quantize_tensors
        if quantize_modules and quantize_tensors:
            raise ValueError("`quantize_tensors` and `quantize_modules` both can't be True.")

        if verbose is None:
            verbose = jax.process_index() == 0
        if quantize_tensors:
            self = quantizer.quantize_model_tensors(self)
        elif quantize_modules:
            self = quantizer.quantize_modules(self, verbose=verbose)
        elif raise_error:
            raise ValueError(
                "both `quantize_modules` and `quantize_tensors` can't be False at a time u can pass `raise_error=False` to skip this error."
            )
        return self

    def to_state(self, state_class: type[EasyDeLState] | None = None) -> EasyDeLState:
        """Convert the module instance into an EasyDeLState object.

        Creates an EasyDeLState that encapsulates the model's parameters and
        configuration for saving, loading, and training operations. The state
        includes the model graph definition and parameters.

        Args:
            state_class: Optional custom state class to use. Must be a subclass
                of EasyDeLState. If None, uses the default EasyDeLState class.

        Returns:
            EasyDeLState: An EasyDeLState object representing the current model
                state, with step initialized to 0 and no optimizer state.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> state = model.to_state()
            >>> # State can be saved and loaded
            >>> state.save_state("checkpoint/")
            >>>
            >>> # Can also use custom state class
            >>> state = model.to_state(state_class=MyCustomState)
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
        """Convert the EasyDeL module to its HuggingFace PyTorch equivalent.

        Creates a HuggingFace PyTorch model and transfers the parameters from
        this JAX model to PyTorch format. Requires the corresponding PyTorch
        model class to be available and registered.

        Args:
            **kwargs: Additional keyword arguments passed to the parameter
                transformation function (e.g., device specification).

        Returns:
            torch.nn.Module: The equivalent HuggingFace PyTorch model with
                weights loaded from this JAX model.

        Example:
            >>> model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b")
            >>> torch_model = model.to_torch()
            >>> # torch_model is now a HuggingFace LlamaModel in PyTorch

        Note:
            Requires PyTorch and the corresponding HuggingFace model to be
            installed. The conversion handles parameter name mapping and
            tensor transposition automatically.
        """
        from easydel.utils.parameters_transformation import ModelConverter

        return ModelConverter.easydel_to_huggingface(
            module=self,
            base_huggingface_module=self.get_torch_loader()._model_mapping[type(self.config)],
            config=self.config,
            dtype=self.param_dtype,
            reform_param=self._get_reform_param(),
            **kwargs,
        )

    def prepare_inputs_for_call(self, **kwargs):
        """Prepare keyword arguments before passing to __call__.

        Hook method that can modify or add arguments before they are passed
        to the module's __call__ method. The base implementation returns
        kwargs unchanged; subclasses can override for custom preprocessing.

        Args:
            **kwargs: The keyword arguments intended for __call__.

        Returns:
            dict: The prepared keyword arguments, potentially modified.

        Example:
            >>> # In a subclass:
            >>> def prepare_inputs_for_call(self, **kwargs):
            ...     # Add default values
            ...     kwargs.setdefault('use_cache', True)
            ...     return kwargs
        """
        return kwargs

    def get_static_arguments(self: Self) -> tuple:
        """Get static arguments required by the module's __call__ method.

        Returns a tuple of static arguments that don't change across calls
        and can be potentially cached or handled differently by JIT compilation.
        Subclasses should override this if they have static arguments.

        Returns:
            tuple: A tuple containing static arguments. The default
                implementation returns an empty tuple.

        Example:
            >>> # In a subclass with static config:
            >>> def get_static_arguments(self):
            ...     return (self.config.use_flash_attention,)
        """
        return ()

    def get_encoder(self: Self) -> nn.Module | EasyDeLBaseModule:
        """Return the encoder component of the model.

        Should be overridden by encoder-decoder models to return their
        encoder component. Useful for tasks that only need the encoder,
        such as feature extraction or embedding generation.

        Returns:
            nn.Module | EasyDeLBaseModule: The encoder module.

        Raises:
            NotImplementedError: If the model does not implement an encoder.
                Decoder-only models should not override this method.

        Example:
            >>> # For encoder-decoder models like T5:
            >>> encoder = model.get_encoder()
            >>> encoder_outputs = encoder(input_ids)
        """
        raise NotImplementedError()

    def get_decoder(self: Self) -> nn.Module | EasyDeLBaseModule:
        """Return the decoder component of the model.

        Should be overridden by encoder-decoder models to return their
        decoder component. Useful for tasks that need access to the
        decoder separately from the encoder.

        Returns:
            nn.Module | EasyDeLBaseModule: The decoder module.

        Raises:
            NotImplementedError: If the model does not implement a decoder.
                Encoder-only models should not override this method.

        Example:
            >>> # For encoder-decoder models:
            >>> decoder = model.get_decoder()
            >>> outputs = decoder(input_ids, encoder_hidden_states=enc_out)
        """
        raise NotImplementedError()

    def get_lm_head(self: Self) -> ParallelLinear:
        """Return the language model head of the model.

        Should be overridden by language models to return their output
        projection layer that maps hidden states to vocabulary logits.

        Returns:
            ParallelLinear: The language model head layer.

        Raises:
            NotImplementedError: If the model does not have a language
                model head. Base models without LM heads should not
                override this method.

        Example:
            >>> lm_head = model.get_lm_head()
            >>> logits = lm_head(hidden_states)  # Shape: (batch, seq, vocab)
        """
        raise NotImplementedError()

    def get_embedding(self: Self) -> nn.Module | Embed:
        """Return the input embedding layer of the model.

        Should be overridden by models to return their token embedding
        layer. Useful for weight tying or accessing embeddings directly.

        Returns:
            nn.Module | Embed: The embedding layer that converts token IDs
                to dense vectors.

        Raises:
            NotImplementedError: If the model does not have an embedding
                layer accessible through this method.

        Example:
            >>> embedding = model.get_embedding()
            >>> embeds = embedding(input_ids)  # Shape: (batch, seq, hidden)
        """
        raise NotImplementedError()

    def compute_embedding(self: Self, input_ids: Int[Array, "..."], *args, **kwargs) -> Float[Array, "..."]:
        """Compute input embeddings from token IDs.

        By default, calls the embedding layer returned by get_embedding().
        Vision-language models can override this hook to incorporate
        multimodal embeddings or other model-specific preprocessing.

        Args:
            input_ids: Token IDs to embed, typically with shape
                (batch_size, sequence_length).
            *args: Additional positional arguments for subclass implementations.
            **kwargs: Additional keyword arguments for subclass implementations.

        Returns:
            Float[Array, "..."]: The embedded representations, typically with
                shape (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If input_ids is None.

        Example:
            >>> embeds = model.compute_embedding(input_ids)
            >>> print(embeds.shape)
            (2, 128, 4096)  # (batch, seq_len, hidden_size)
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")
        return self.get_embedding()(jnp.asarray(input_ids, dtype="i4"))

    def compute_embedding_with_info(
        self: Self, input_ids: Int[Array, "..."], *args, **kwargs
    ) -> tuple[Float[Array, "..."], EmbeddingInfo | None]:
        """Compute input embeddings and optional auxiliary information.

        The default implementation returns (compute_embedding(...), None).
        Multimodal models can override this to return extra tensors needed
        to reproduce the full forward pass when providing inputs_embeds
        directly (e.g., DeepStack visual features, mRoPE indices).

        Args:
            input_ids: Token IDs to embed.
            *args: Additional positional arguments for subclass implementations.
            **kwargs: Additional keyword arguments for subclass implementations.

        Returns:
            tuple: A tuple of (embeddings, info) where:
                - embeddings: The embedded token representations
                - info: Optional EmbeddingInfo containing auxiliary data,
                    or None for text-only models

        Example:
            >>> embeds, info = model.compute_embedding_with_info(input_ids, images=images)
            >>> # For VLMs, info may contain visual features and position info
        """
        return self.compute_embedding(input_ids, *args, **kwargs), None

    @classmethod
    def sequential_init(cls: type[Self], **kwargs) -> Self:
        """Initialize model parameters sequentially with proper sharding.

        Performs lazy initialization followed by sequential parameter
        initialization with appropriate sharding for distributed training.
        This is particularly useful for large models that need memory-efficient
        initialization where creating all parameters at once would exceed
        available memory.

        The method:
        1. Creates a lazy (shape-only) version of the model
        2. Iterates through all modules and initializes their parameters one by one
        3. Applies proper sharding based on partition rules to each parameter

        Args:
            **kwargs: Arguments passed to lazy_init, including:
                - config: Model configuration
                - dtype: Computation dtype
                - param_dtype: Parameter dtype
                - precision: JAX precision setting
                - rngs: Random number generators (defaults to Rngs(44) if not provided)

        Returns:
            Self: Fully initialized model with properly sharded parameters.

        Example:
            >>> config = LlamaConfig(hidden_size=4096, num_hidden_layers=32)
            >>> # This won't OOM even for very large models
            >>> model = LlamaModel.sequential_init(
            ...     config=config,
            ...     dtype=jnp.bfloat16,
            ...     param_dtype=jnp.bfloat16,
            ...     precision='high',
            ...     rngs=nn.Rngs(0)
            ... )

        Note:
            This method is slower than regular initialization but allows
            initializing models that would otherwise not fit in memory.
        """
        from easydel.utils.traversals import iter_module_search

        def _shard(x):
            return x

        rng = kwargs.get("rngs", flax.nnx.Rngs(44))
        lazy_model = cls.lazy_init(**kwargs)
        partition_rules = lazy_model.config.get_partition_rules()
        for path, module in iter_module_search(lazy_model, (flax.nnx.Module, ArrayParam)):
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
                            joined_path: a,
                        },
                        strict=False,
                    ),
                )
                shardings = {
                    "kernel": partition_spec[joined_path + "/kernel"],
                    "bias": partition_spec[joined_path + "/bias"],
                    "embedding": partition_spec[joined_path + "/embedding"],
                    "scale": partition_spec[joined_path + "/scale"],
                    "raw": partition_spec[joined_path],
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
            if hasattr(module, "resure"):
                module.resure(rng.param(), shard_fn=jax.jit(_shard, out_shardings=shardings["raw"]))
        for path, module in iter_module_search(lazy_model, nn.Param):
            if path and type(module.value) is jax.ShapeDtypeStruct:
                logger.warning(f"({type(module).__name__}) found empty array at " + ("/".join([str(s) for s in path])))

        return lazy_model

    @classmethod
    def lazy_init(cls: type[Self], **kwargs) -> Self:
        """Perform lazy initialization using nnx.eval_shape.

        Initializes the module structure and determines parameter shapes
        without actually allocating memory for the parameters. This is
        useful for inspecting model structure, preparing sharding specs,
        or initializing very large models incrementally.

        Args:
            **kwargs: Keyword arguments passed to the class constructor,
                including config, dtype, param_dtype, precision, and rngs.

        Returns:
            Self: A module instance with initialized structure but abstract
                parameters (jax.ShapeDtypeStruct instead of actual arrays).

        Example:
            >>> config = LlamaConfig(hidden_size=4096, num_hidden_layers=32)
            >>> lazy_model = LlamaModel.lazy_init(
            ...     config=config,
            ...     dtype=jnp.bfloat16,
            ...     param_dtype=jnp.bfloat16,
            ...     precision='high',
            ...     rngs=nn.Rngs(0)
            ... )
            >>> # Inspect shapes without allocating memory
            >>> print(lazy_model.graphtree_params_shape)

        Note:
            The returned model cannot be used for computation directly.
            Use sequential_init() or regular __init__ for usable models.
        """
        rngs = kwargs.pop("rngs", None)

        def _init(rngs):
            return cls(**kwargs, rngs=rngs)

        return nn.eval_shape(_init, rngs=rngs)

    def merge_lora_params(self: Self, pytree: dict) -> Self:
        """Merge LoRA parameters from a pytree into the base model.

        Combines LoRA low-rank adaptation matrices with the base model's
        weights. The LoRA update is computed as: W_new = W + A @ B * scaling

        Args:
            pytree: A dictionary (pytree) containing the LoRA parameters
                (A and B matrices) structured similarly to the base model's
                parameters.

        Returns:
            Self: The module instance with LoRA parameters merged into
                the base weights.

        Example:
            >>> # After training LoRA adapters
            >>> lora_params = load_lora_params("lora_checkpoint/")
            >>> model = model.merge_lora_params(lora_params)
            >>> # Model now has LoRA weights baked into base weights

        See Also:
            split_lora_params: Inverse operation to extract LoRA params.
            apply_lora_to_layers: Apply LoRA to specific layers.
        """
        from easydel.infra.utils import merge_lora_params

        self = merge_lora_params(self, pytree)
        return self

    def split_lora_params(self: Self) -> dict:
        """Split merged LoRA parameters back out from the base model.

        Extracts LoRA adaptation matrices that were previously merged using
        merge_lora_params() or a similar process. Restores the base model
        weights to their original pre-merge state.

        Returns:
            dict: A pytree containing the extracted LoRA parameters
                (A and B matrices) that can be saved or reapplied later.

        Example:
            >>> # Extract LoRA params for saving
            >>> lora_params = model.split_lora_params()
            >>> save_lora_params(lora_params, "lora_checkpoint/")
            >>> # Base model weights are restored

        See Also:
            merge_lora_params: Inverse operation to merge LoRA params.
        """
        from easydel.infra.utils import split_lora_params

        pytree = split_lora_params(self)
        return pytree

    @property
    def lora_is_enabled(self: Self):
        """Check if LoRA (Low-Rank Adaptation) is enabled for this module.

        Iterates through the module's graph to detect any LoRAParam instances,
        indicating that LoRA adaptation has been applied.

        Returns:
            bool: True if any LoRA parameters are found in the module,
                False otherwise.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> print(model.lora_is_enabled)
            False
            >>> model = model.apply_lora_to_layers(lora_rank=8)
            >>> print(model.lora_is_enabled)
            True
        """
        for _, tensor in nn.iter_graph(self):
            if isinstance(tensor, nn.LoRAParam):
                return True
        return False

    @property
    def is_quantized(self) -> bool:
        """Check if the model contains any quantized layers or parameters.

        Iterates through the model graph to detect quantized components,
        including 8-bit linear layers, NF4 linear layers, and quantized
        arrays (Array8B, ArrayNF4, Array1B).

        Returns:
            bool: True if the model contains any quantized components,
                False otherwise.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> print(model.is_quantized)
            False
            >>> model = model.quantize()
            >>> print(model.is_quantized)
            True
        """
        from eformer.ops.quantization import Array1B, Array8B, ArrayNF4

        from easydel.layers.components import (
            ColumnParallelLinearQuantized,
            ParallelLinearQuantized,
            RowParallelLinearQuantized,
        )

        for _, tensor in nn.iter_graph(self):
            if isinstance(
                tensor,
                (
                    RowParallelLinearQuantized,
                    ColumnParallelLinearQuantized,
                    ParallelLinearQuantized,
                ),
            ):
                return True

            possible_tensor = getattr(tensor, "value", getattr(tensor, "raw_value", tensor))

            if isinstance(possible_tensor, (Array8B, ArrayNF4, Array1B)):
                return True

            if getattr(possible_tensor, "dtype", None) in [
                jnp.float8_e4m3,
                jnp.float8_e5m2,
                jnp.float4_e2m1fn,
            ]:
                return True
        return False

    def apply_lora_to_layers(
        self: Self,
        lora_rank: int,
        lora_pattern: str | None = None,
        verbose: bool = False,
        rngs: nn.Rngs | None = None,
    ) -> Self:
        """Apply Low-Rank Adaptation (LoRA) to specified linear layers.

        Replaces matching Linear layers with LoRA-enabled equivalents that
        have low-rank A and B matrices for efficient fine-tuning.

        Args:
            lora_rank: The rank of the LoRA decomposition. Lower ranks use
                less memory but have less capacity. Common values: 4, 8, 16, 32.
            lora_pattern: Regular expression to match the names of Linear
                layers to apply LoRA to. If None, applies to common attention
                and MLP layers. Defaults to None.
            verbose: If True, prints information about which layers are being
                modified. Defaults to False.
            rngs: JAX random number generators for initializing LoRA matrices.
                If None, uses default RNGs. Defaults to None.

        Returns:
            Self: The module instance with LoRA layers applied.

        Example:
            >>> # Apply LoRA to attention layers only
            >>> model = model.apply_lora_to_layers(
            ...     lora_rank=8,
            ...     lora_pattern=r".*attention.*(q_proj|v_proj).*",
            ...     verbose=True
            ... )
            >>>
            >>> # Apply LoRA to all linear layers
            >>> model = model.apply_lora_to_layers(lora_rank=16)

        See Also:
            unwrap_lora_to_layers: Remove LoRA and restore original layers.
            merge_lora_params: Merge LoRA weights into base weights.
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
        """Revert LoRA layers to their original linear layers.

        Replaces LoraLinear layers with their original flax.nnx.Linear
        counterparts, discarding the LoRA A and B matrices. The base
        weights are preserved.

        Args:
            verbose: If True, prints information about which layers are being
                reverted. Defaults to False.

        Returns:
            Self: The module instance with LoRA layers removed and original
                Linear layers restored.

        Example:
            >>> model = model.apply_lora_to_layers(lora_rank=8)
            >>> # ... training ...
            >>> model = model.unwrap_lora_to_layers(verbose=True)
            >>> # Model now has regular Linear layers

        See Also:
            apply_lora_to_layers: Apply LoRA to layers.
        """
        from easydel.infra.utils import unwrap_lora_to_layers

        self = unwrap_lora_to_layers(self, verbose=verbose)
        return self

    def _get_reform_param(self) -> dict[str, tp.Any]:
        """Collect reform_param configurations from submodules.

        Traverses the module tree to collect reform_param configurations
        used for parameter transformation during model conversion.

        Returns:
            dict[str, Any]: A dictionary mapping fully qualified parameter
                paths to their reform configuration dictionaries.

        Note:
            This is an internal method used during model conversion.
        """
        from easydel.utils import traversals

        reform_param = {}
        for path, module in traversals.iter_module_search(self, nn.Module):
            if hasattr(module, "reform_param") and module.reform_param:
                path_str = ".".join(map(str, path))
                for key, value in module.reform_param.items():
                    full_key = f"{path_str}.{key}" if path_str else key
                    new_value = value.copy()
                    new_splits = []
                    for split in value["splits"]:
                        new_split = split.copy()
                        split_name = split["name"]
                        new_split["name"] = f"{path_str}.{split_name}" if path_str else split_name
                        new_splits.append(new_split)
                    new_value["splits"] = new_splits

                    reform_param[full_key] = new_value
        return reform_param

    @property
    def transform_fn(self):
        """Create a transformation function for HuggingFace to EasyDeL conversion.

        Identifies special layers (embeddings, LayerNorm, MoE) and returns a
        configured transformation function with sharding rules applied.

        Returns:
            Callable: A partial function (StateDictConverter.huggingface_to_easydel)
                configured with layer information, dtype, and sharding functions.

        Example:
            >>> transform_fn = model.transform_fn
            >>> easydel_params = transform_fn(hf_state_dict)
        """
        from easydel.layers.components import BaseMoeModule, Embed, ParallelMoELinear
        from easydel.utils import traversals
        from easydel.utils.parameters_transformation import StateDictConverter

        embedding_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.Embed)]
        embedding_path.extend([".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, Embed)])
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
            reform_param=self._get_reform_param(),
        )

    @property
    def _generate_compatible_graphdef(self: Self):
        """Create a graph definition compatible with generation tasks.

        Generation often requires specific configurations (like disabling
        gradient checkpointing). This method creates a temporary generation-
        compatible configuration, performs lazy initialization, and extracts
        the resulting graph definition.

        Returns:
            nn.GraphDef: A graph definition suitable for use during generation,
                with gradient checkpointing disabled.
        """

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
        """Create non-parameter state compatible with generation tasks.

        Similar to _generate_compatible_graphdef, creates a temporary
        generation-compatible configuration, lazy-initializes, and extracts
        the non-parameter state variables with concrete values.

        Returns:
            nn.GraphState: A graph state containing non-parameter variables
                suitable for generation, with meta-placeholders replaced by
                concrete values.
        """

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
        """Get the sharding annotation for each parameter.

        Returns:
            dict: A nested dictionary mirroring the parameter structure,
                containing the sharding information (e.g., NamedSharding,
                PartitionSpec) for each parameter, or None if unsharded.

        Example:
            >>> shardings = model.params_sharding
            >>> print(shardings['layers']['0']['attention']['q_proj']['kernel'])
            NamedSharding(mesh=..., spec=PartitionSpec('fsdp', 'tp'))
        """
        return jax.tree_util.tree_map(
            lambda x: x.sharding if hasattr(x, "sharding") else None,
            self.split_params_dict(),
        )

    def merge_params(self, tree):
        """Merge a parameter state tree back into the module.

        Reconstructs the module using its existing graph definition and
        'other' state, but replaces the parameter state with the provided tree.

        Args:
            tree: A pytree (typically nn.GraphState) containing the parameters
                to merge into the module.

        Returns:
            EasyDeLBaseModule: The module instance with new parameters merged in.

        Example:
            >>> # Modify parameters externally
            >>> params = model.split_params()
            >>> modified_params = jax.tree_map(lambda x: x * 0.9, params)
            >>> model = model.merge_params(modified_params)
        """
        gdef, _, gother = nn.split(self, self.graphstate_type, ...)
        self = nn.merge(gdef, tree, gother)
        return self

    def split_params(self: Self):
        """Split the module and return the parameter state.

        Uses nnx.split to extract the GraphState containing only the
        trainable parameters (nn.Param or nn.LoRAParam).

        Returns:
            nn.GraphState: The parameter state of the module.

        Example:
            >>> params = model.split_params()
            >>> # params is a GraphState that can be manipulated
            >>> flat_params = params.flat_state()
        """
        return nn.split(self, self.graphstate_type, ...)[1]

    def split_params_dict(
        self,
        extract_fn: tp.Callable | None = None,
        remove_none: bool = True,
    ) -> dict:
        """Split module parameters and return as a nested dictionary.

        Extracts the parameter state, converts it to a plain dictionary
        (removing VariableState wrappers), and optionally removes None values.

        Args:
            extract_fn: Optional function to apply to each parameter during
                extraction. Defaults to None.
            remove_none: If True, removes key-value pairs where the value
                is None. Defaults to True.

        Returns:
            dict: A nested dictionary containing the module's parameters
                as plain arrays (not wrapped in VariableState).

        Example:
            >>> params_dict = model.split_params_dict()
            >>> # params_dict is a regular nested dict
            >>> kernel = params_dict['layers']['0']['attention']['q_proj']['kernel']
            >>> print(type(kernel))  # jax.Array, not VariableState
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
        """Merge parameters from a dictionary into the module's state.

        Updates the module's current parameter state with values from the
        provided dictionary. The dictionary structure should match the
        module's parameter structure.

        Args:
            params_dict: A nested dictionary containing the parameters to merge.
                Can be either nested or flattened (will be detected automatically).

        Returns:
            Self: The module instance with parameters from the dictionary merged.

        Raises:
            KeyError: If a key from params_dict is not found in the module's
                current state.

        Example:
            >>> # Load parameters from a file
            >>> params_dict = load_params("checkpoint.pkl")
            >>> model = model.merge_params_dict(params_dict)
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
        """Calculate the FLOPs (Floating Point Operations) per token.

        Estimates the computational cost of processing one token through
        the model, useful for performance benchmarking and cost estimation.

        Args:
            sequence_length: Sequence length to use for the calculation.
                If None, uses granted_mask_max_position_embedding from config.
            include_loss: Whether to include loss computation in the count.
                Defaults to True.
            include_backward: Whether to include backward pass FLOPs.
                If True, multiplies forward FLOPs by 3 (typical ratio).
                Defaults to False.

        Returns:
            float: The estimated FLOPs per token. Returns 1 if calculation fails.

        Example:
            >>> flops = model.flops_per_token(sequence_length=2048)
            >>> print(f"FLOPs per token: {flops:.2e}")
            FLOPs per token: 1.23e+12
            >>>
            >>> # Include backward pass for training cost
            >>> train_flops = model.flops_per_token(include_backward=True)

        Note:
            This is an estimate based on standard transformer operations.
            Actual FLOPs may vary depending on implementation details.
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
        """Estimate FLOPs for a single forward pass using JAX's make_jaxpr.

        Uses JAX's computation graph analysis to estimate the floating point
        operations required for one forward pass with the given arguments.

        Args:
            *args: Positional arguments to pass to __call__.
            **kwargs: Keyword arguments to pass to __call__.

        Returns:
            float | None: The estimated FLOP count, or None if calculation fails.

        Note:
            This provides a more accurate estimate than flops_per_token() but
            requires actually tracing the computation graph.
        """
        from .utils import count_flop_jaxpr

        return count_flop_jaxpr(jax.make_jaxpr(self.__call__)(*args, **kwargs))

    @property
    def pure_transform_fn(self: Self):
        """Get a pure transformation function without sharding.

        Similar to transform_fn, but does not include sharding functions.
        Returns a partial function configured only with layer names and dtype.

        Returns:
            Callable: A partial function (StateDictConverter.huggingface_to_easydel)
                for converting PyTorch state dicts without applying sharding.

        Example:
            >>> transform_fn = model.pure_transform_fn
            >>> # Convert without sharding
            >>> easydel_params = transform_fn(hf_state_dict, shard_fns=None)
        """
        from easydel.layers.components import BaseMoeModule, Embed, ParallelMoELinear
        from easydel.utils import traversals
        from easydel.utils.parameters_transformation import StateDictConverter

        embedding_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.Embed)]
        embedding_path.extend([".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, Embed)])
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
            reform_param=self._get_reform_param(),
        )

    @property
    def _default_loss_config(self: Self) -> LossConfig | None:
        """Get the default LossConfig for this module.

        Subclasses can override this property to return a default LossConfig
        instance specific to the model's task.

        Returns:
            LossConfig | None: The default loss configuration, or None.
        """
        return None

    @_default_loss_config.setter
    def _default_loss_config(self, val):
        """Setter for the default loss config (internal use).

        Args:
            val: The value to set (not actually stored, just for API compatibility).

        Returns:
            The input value.
        """
        return val

    def compute_loss(
        self,
        *,
        labels: Array | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
        **batch,
    ) -> tuple[tp.Any, LossMetrics]:
        """Compute the loss for the model given inputs and labels.

        Performs a forward pass using the provided batch arguments, then
        calculates the loss using the determined loss function. Handles
        label inference for causal LM and default configurations for
        sequence classification.

        Args:
            labels: The target labels. For Causal LM, if None, uses input_ids
                from the batch. Defaults to None.
            loss_config: Specific configuration for loss calculation. For
                sequence classification, defaults to using num_labels from
                config. Defaults to None.
            loss_kwargs: Additional keyword arguments to pass directly to
                the loss function. Defaults to None.
            **batch: Keyword arguments representing the input batch
                (e.g., input_ids, attention_mask, pixel_values).

        Returns:
            tuple: A tuple containing:
                - outputs: The model's output (dataclass with logits, hidden_states, etc.)
                - LossMetrics: Object containing the calculated loss and metrics.

        Raises:
            AssertionError: If labels are required but not provided or inferred.
            AssertionError: If sequence classification loss is used without
                num_labels in config.

        Example:
            >>> outputs, loss_metrics = model.compute_loss(
            ...     input_ids=input_ids,
            ...     attention_mask=attention_mask,
            ...     labels=labels
            ... )
            >>> print(f"Loss: {loss_metrics.loss}")
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

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language model head to transform hidden states to logits.

        Computes output logits over the vocabulary from the final hidden states.
        Handles weight tying if configured (shares weights between embedding
        and output projection).

        Args:
            hidden_states: Input hidden states from the transformer model
                with shape (..., hidden_size).

        Returns:
            Array: Output logits over the vocabulary with shape (..., vocab_size).

        Example:
            >>> # Get hidden states from model
            >>> hidden_states = model.model(input_ids).last_hidden_state
            >>> # Apply LM head to get logits
            >>> logits = model.apply_lm_head(hidden_states)
            >>> print(logits.shape)
            (2, 128, 32000)  # (batch, seq, vocab)
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

    def update_module(
        self,
        recursive_update: bool = False,
        **kwargs: Unpack[EasyDeLBaseConfigDict],
    ):
        """Update the module configuration and reinitialize structure.

        Creates a new lazy module with updated configuration while preserving
        the current parameter state. Useful for changing model behavior
        without reinitializing weights.

        Args:
            recursive_update: If True, recursively apply the same updates to
                any nested config objects that are subclasses of EasyDeLBaseConfig.
                Defaults to False.
            **kwargs: Configuration parameters to update (e.g., attn_mechanism,
                gradient_checkpointing).

        Returns:
            Self: Updated module with new configuration and same parameter values.

        Example:
            >>> # Change attention mechanism
            >>> model = model.update_module(attn_mechanism='flash')
            >>>
            >>> # Disable gradient checkpointing
            >>> model = model.update_module(
            ...     gradient_checkpointing=EasyDeLGradientCheckPointers.NONE
            ... )

        Note:
            This modifies the config in place. Use new_graphdef() if you need
            to preserve the original config.
        """
        config = self.config
        for k, v in kwargs.items():
            setattr(config, k, v)
        if recursive_update:
            for attr_name in dir(config):
                if attr_name.startswith("_"):
                    continue
                attr_value = getattr(config, attr_name, None)
                if isinstance(attr_value, EasyDeLBaseConfig):
                    for k, v in kwargs.items():
                        if hasattr(attr_value, k):
                            setattr(attr_value, k, v)
        module = self.lazy_init(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=self.rngs,
        )
        self = self.merge_module(module.graphdef, self.graphstate, self.graphother)
        return self

    def new_graphdef(
        self,
        recursive_update: bool = False,
        **kwargs: Unpack[EasyDeLBaseConfigDict],
    ):
        """Create a new module with updated configuration.

        Creates a new lazy module with updated configuration while preserving
        the current parameter state. Unlike update_module(), this does not
        modify the original config.

        Args:
            recursive_update: If True, recursively apply the same updates to
                any nested config objects that are subclasses of EasyDeLBaseConfig.
                Defaults to False.
            **kwargs: Configuration parameters to update. Applied to a copy
                of the current configuration.

        Returns:
            nn.GraphDef: A new graph definition with updated configuration
                that can be merged with existing parameters.

        Example:
            >>> # Get a new graphdef with different settings
            >>> new_gdef = model.new_graphdef(attn_mechanism='flash')
            >>> # Merge with existing parameters
            >>> new_model = nn.merge(new_gdef, model.graphstate, model.graphother)
        """
        config = deepcopy(self.config)
        for k, v in kwargs.items():
            setattr(config, k, v)
        if recursive_update:
            for attr_name in dir(config):
                if attr_name.startswith("_"):
                    continue
                attr_value = getattr(config, attr_name, None)
                if isinstance(attr_value, EasyDeLBaseConfig):
                    for k, v in kwargs.items():
                        if hasattr(attr_value, k):
                            setattr(attr_value, k, v)
        module = self.lazy_init(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=self.rngs,
        )
        return module.graphdef

    def __hash__(self):
        """Compute a hash of the module for caching and comparison.

        Returns:
            int: Hash value based on the module's state and configuration.

        Note:
            This delegates to static_hash(None).
        """
        return self.static_hash(None)

    def static_hash(self, pop_things: list[str] | None = None):
        """Compute a deterministic hash of the module's state and configuration.

        Creates a hash based on the module's parameters (graphstate),
        non-parameter state (graphother), and configuration dictionary.
        Useful for caching compiled functions or identifying state changes.

        Args:
            pop_things: Optional list of configuration keys to exclude from
                the hash. Useful when certain config fields (e.g., 'attn_mechanism')
                shouldn't affect the cache key.

        Returns:
            int: A signed integer hash value computed from the model's state
                and configuration using MD5.

        Example:
            >>> # Hash without excluding any config keys
            >>> hash1 = model.static_hash()
            >>>
            >>> # Hash excluding attention mechanism
            >>> hash2 = model.static_hash(["attn_mechanism"])
            >>>
            >>> # Hashes may be equal if only attn_mechanism differs
            >>> print(hash1 == hash2)

        Note:
            The hash is deterministic - identical states produce identical hashes.
        """
        from ejkernel.callib._ejit import _get_args_signature

        dict_config = self.config.to_dict()
        if pop_things:
            for pops in pop_things:
                dict_config.pop(pops)
        tree_hash = _get_args_signature((self.graphstate, self.graphother), dict_config)
        bytes_in = hashlib.md5((tree_hash).encode("utf-8")).digest()
        return int.from_bytes(bytes_in, byteorder="big", signed=True)
