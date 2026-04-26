# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
    >>> import spectrax as spx
    >>> from spectrax import nn
    >>>
    >>> class MyModel(EasyDeLBaseModule):
    ...     def __init__(self, config, dtype, param_dtype, precision, rngs):
    ...         super().__init__(config, dtype, param_dtype, precision, rngs)
    ...         # Initialize model layers
    ...         self.layer = nn.Linear(config.hidden_size, config.hidden_size)
    ...
    ...     def forward(self, inputs):
    ...         return self.layer(inputs)
    >>>
    >>> # Create and use the model
    >>> config = EasyDeLBaseConfig(hidden_size=768)
    >>> model = MyModel(
    ...     config=config,
    ...     dtype=jnp.float32,
    ...     param_dtype=jnp.float32,
    ...     precision='highest',
    ...     rngs=spx.Rngs(0)
    ... )

The module integrates with JAX's sharding system for distributed training,
supports various quantization methods, and provides utilities for converting
between EasyDeL and HuggingFace model formats.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import re
import typing as tp
import warnings
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial, wraps
from re import Pattern
from typing import Self, Unpack

import jax
import jax.tree_util
import spectrax as spx
from eformer.loggings import get_logger
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float, Int
from spectrax import make_shard_and_gather_fns, match_partition_rules, nn
from spectrax.common_types import NOT_GIVEN

from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ArrayParam, materialize_meta_leaves
from easydel.layers.norms import LayerNorm
from easydel.utils import traversals
from easydel.utils.traversals import flatten_dict, is_flatten, unflatten_dict

from .base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict
from .etils import EasyDeLGradientCheckPointers
from .loss_utils import (
    LOSS_MAPPING,
    ForCausalLMLoss,
    ForSequenceClassificationLoss,
    LossConfig,
    LossMetrics,
    resolve_loss_strategy,
)
from .mixins import BaseModuleProtocol, EasyBridgeMixin, EasyGenerationMixin, OperationCacheMixin
from .mixins.protocol import printify_module
from .modeling_outputs import EmbeddingInfo
from .utils import device_put_or_shard_abstract, sanitize_partition_spec_for_shape

if tp.TYPE_CHECKING:
    from easydel.infra.base_state import EasyDeLState
    from easydel.layers import Embed, ParallelLinear, QuantizationConfig


PartitionLike = Mapping[str, tp.Callable] | Mapping[tuple, tp.Callable] | tuple[tuple[str, tp.Any], ...] | None
"""Type alias for partition rule specifications.

Can be a mapping from parameter name patterns (as strings or tuples) to
partition specification functions, or None for default partitioning.
"""


logger = get_logger(__name__)


BaseConf = EasyDeLBaseConfig
"""Alias for EasyDeLBaseConfig for backward compatibility."""


def _looks_like_config(value: tp.Any) -> bool:
    return isinstance(value, EasyDeLBaseConfig) or (
        value is not None and hasattr(value, "runtime_sharding_resolver") and hasattr(value, "mesh")
    )


def _resolve_init_config(
    init: tp.Callable, instance: tp.Any, args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any]
) -> tp.Any:
    if _looks_like_config(kwargs.get("config")):
        return kwargs["config"]

    try:
        bound = inspect.signature(init).bind_partial(instance, *args, **kwargs)
    except Exception:
        bound = None
    if bound is not None:
        config = bound.arguments.get("config")
        if _looks_like_config(config):
            return config

    for value in args:
        if _looks_like_config(value):
            return value
    return None


def _is_reusable_context(value: tp.Any) -> bool:
    return hasattr(value, "__enter__") and hasattr(value, "__exit__")


@contextlib.contextmanager
def _parameter_init_sharding_context(config: tp.Any):
    if not _looks_like_config(config):
        yield
        return

    mesh = config.mesh
    resolver = config.runtime_sharding_resolver

    if hasattr(resolver, "with_mesh"):
        resolver = resolver.with_mesh(mesh)

    def place(value: tp.Any, metadata: dict[str, tp.Any], explicit_sharding: bool) -> tp.Any | None:
        if not metadata or not isinstance(value, jax.Array):
            return None

        has_layout_metadata = any(key in metadata for key in ("sharding", "tensor_layout"))
        existing = getattr(value, "sharding", None)
        if isinstance(existing, NamedSharding) and not explicit_sharding and not has_layout_metadata:
            return None

        shape = tuple(int(dim) for dim in value.shape)
        try:
            named = resolver.named_sharding_for_metadata(metadata, shape=shape, mesh=mesh)
        except (TypeError, ValueError):
            return None
        if named is None:
            return None
        if existing is not None and existing == named:
            return value
        return jax.device_put(value, named)

    with contextlib.ExitStack() as stack:
        if _is_reusable_context(mesh):
            stack.enter_context(mesh)
        if hasattr(resolver, "logical_axis_rules"):
            stack.enter_context(resolver.logical_axis_rules())
        stack.enter_context(spx.variable_init_placement(place))
        yield


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


class EasyDeLLayerStackMixin:
    """Shared repeated-layer cache, scan, and pipeline-stage helpers."""

    config: tp.Any

    def _layer_cache_view_at(
        self: Self,
        cache_views: tp.Any,
        layer_idx: tp.Any,
        *,
        enabled: bool,
        cache: tp.Any = None,
    ) -> tp.Any:
        """Read a mutable per-layer cache view for trace/cache execution."""
        if not enabled:
            return None
        if cache is not None:
            return cache[layer_idx]
        if cache_views is None:
            return None
        return cache_views[layer_idx]

    def _layer_cache_view_update(
        self: Self,
        cache_views: tp.Any,
        layer_idx: tp.Any,
        new_view: tp.Any,
        *,
        enabled: bool,
        cache: tp.Any = None,
    ) -> tp.Any:
        """Update mutable per-layer cache views only on cache/trace paths."""
        if not enabled or new_view is None:
            return cache_views
        if cache is not None:
            cache[layer_idx] = new_view
        elif cache_views is not None:
            cache_views[layer_idx] = new_view
        return cache_views

    def _pipeline_stage_count(self: Self) -> int:
        """Return the logical pipeline width from the canonical SpectraX mesh."""
        return int(self.config.mesh.shape["pp"]) * int(self.config.pipeline_virtual_stages)

    def _pipeline_physical_stage_count(self: Self) -> int:
        """Return the physical pipeline width from the canonical SpectraX mesh."""
        return int(self.config.mesh.shape["pp"])

    def _layer_physical_stage_assignment(self: Self, layer_idx: int, total_layers: int) -> tuple[int, int]:
        """Resolve a layer position to the physical PP owner used for sharding.

        Logical virtual stages are ordered as ``virt * pp + rank`` by the
        default SpectraX virtual schedules. Parameter metadata stores the
        physical owner so creation-time sharding matches the schedule instead
        of requiring runtime cross-rank parameter moves.
        """
        physical_pp = self._pipeline_physical_stage_count()
        if physical_pp <= 1:
            return 0, 1
        logical_pp = self._pipeline_stage_count()
        logical_stage = min(logical_pp - 1, (int(layer_idx) * logical_pp) // int(total_layers))
        layout = self.config.pipeline_stage_layout
        if layout == "contiguous":
            virtual_stages = max(1, int(self.config.pipeline_virtual_stages))
            return min(physical_pp - 1, logical_stage // virtual_stages), physical_pp
        if layout == "loop":
            offset = logical_stage % physical_pp
            virtual_stage = logical_stage // physical_pp
            physical_stage = offset if virtual_stage % 2 == 0 else physical_pp - 1 - offset
            return physical_stage, physical_pp
        return logical_stage % physical_pp, physical_pp

    @contextlib.contextmanager
    def _assign_layer_stage(self: Self, layer_idx: int, *, total_layers: int):
        """Assign subsequently-created variables to the layer's physical PP owner."""
        current, total = self._layer_physical_stage_assignment(layer_idx, total_layers)
        with spx.assign_stage(total=total, current=current):
            yield

    def _layer_scan_trace(
        self: Self,
        trace: bool,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        cache: tp.Any = None,
        cache_views: tp.Iterable[tp.Any] | None = None,
        extra: bool = False,
    ) -> bool:
        """Resolve whether a repeated-layer stack must use the Python trace path."""
        del cache
        has_cache_views = cache_views is not None and any(view is not None for view in cache_views)
        return (
            trace
            or output_hidden_states
            or output_attentions
            or has_cache_views
            or extra
            or not self.config.scan_layers
            or self._pipeline_stage_count() > 1
        )

    def _mark_layer_stage_boundary(
        self: Self,
        hidden_states: tp.Any,
        layer_idx: tp.Any,
        *,
        layers: tp.Sized | None = None,
    ) -> tp.Any:
        """Mark dynamic pipeline stage boundaries for repeated layer stacks."""
        try:
            idx = int(layer_idx)
        except (TypeError, ValueError):
            return hidden_states
        total = len(layers if layers is not None else self.layers)  # type: ignore[attr-defined]
        pp = self._pipeline_stage_count()
        if pp <= 1 or idx + 1 >= total:
            return hidden_states
        current = min(pp - 1, (idx * pp) // total)
        nxt = min(pp - 1, ((idx + 1) * pp) // total)
        if current != nxt:
            edge_sharding = self._layer_stage_boundary_sharding(hidden_states)
            return spx.sxstage_iter(hidden_states, stage=current, sharding=edge_sharding)
        return hidden_states

    def _layer_stage_boundary_sharding(self: Self, hidden_states: tp.Any) -> PartitionSpec | None:
        """Resolve the activation sharding contract for PP stage edges."""
        if not hasattr(hidden_states, "shape"):
            return None
        try:
            return self.config.runtime_sharding_resolver.with_mesh(self.config.mesh).resolve(
                dynamic_axes=spx.common_types.HiddenStateSharding,
                shape=tuple(hidden_states.shape),
            )
        except Exception:
            return None


class EasyDeLBaseModule(
    EasyDeLLayerStackMixin,
    spx.Module,
    EasyBridgeMixin,
    EasyGenerationMixin,
    OperationCacheMixin,
    BaseModuleProtocol,
):
    """Base class for all EasyDeL neural network modules.

    EasyDeLBaseModule provides the foundational functionality for all EasyDeL models,
    including parameter management, distributed training support, quantization,
    LoRA adaptation, and integration with HuggingFace models. It inherits from
    spx.Module and multiple mixins that provide additional capabilities.

    This class should be subclassed to create specific model architectures. Subclasses
    must implement the forward method and may override various hooks for customization.

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
        ...         self.layers = nn.ModuleList([
        ...             TransformerBlock(config, dtype, param_dtype, precision, rngs)
        ...             for _ in range(config.num_hidden_layers)
        ...         ])
        ...
        ...     def forward(self, input_ids, attention_mask=None):
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

    config_class: type[BaseConf] | None
    base_model_prefix: str
    config: BaseConf | None
    _model_task: str | None
    _model_type: str | None
    _parameter_transform_rules: tp.ClassVar[list[ParameterTransformRule]] = []

    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__init_subclass__(**kwargs)
        original_init = cls.__dict__.get("__init__")
        if original_init is None or getattr(original_init, "_easydel_init_sharding_wrapped", False):
            return

        @wraps(original_init)
        def init_with_parameter_sharding(self, *args, **kwargs):
            config = _resolve_init_config(original_init, self, args, kwargs)
            with _parameter_init_sharding_context(config):
                original_init(self, *args, **kwargs)

        init_with_parameter_sharding._easydel_init_sharding_wrapped = True
        with contextlib.suppress(Exception):
            init_with_parameter_sharding.__signature__ = inspect.signature(original_init)
        cls.__init__ = init_with_parameter_sharding

    def __init__(
        self,
        config: BaseConf,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: lax.PrecisionLike,
        rngs: spx.Rngs,
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
            rngs: SpecTrax random number generator container for parameter initialization
                and stochastic operations like dropout.

        Example:
            >>> config = LlamaConfig(hidden_size=1024, num_hidden_layers=4)
            >>> model = LlamaModel(
            ...     config=config,
            ...     dtype=jnp.bfloat16,
            ...     param_dtype=jnp.bfloat16,
            ...     precision='high',
            ...     rngs=spx.Rngs(42)
            ... )

        Note:
            This method should be called by all subclasses to properly
            initialize the base functionality. Failing to call super().__init__()
            will result in missing core attributes.
        """
        super().__init__()
        self.config: BaseConf = config
        self.dtype: jnp.dtype = dtype
        self.param_dtype: jnp.dtype = param_dtype
        self.precision: lax.PrecisionLike = precision
        self.rngs: spx.Rngs = rngs

        _ = self.graphtree_shape
        _ = self.graphtree_parameters_shape
        _ = self.mesh
        self.register_context(mesh=self.mesh)
        _ = self.model_task
        _ = self.model_type

    def __str__(self) -> str:
        return printify_module(self)

    def __repr__(self) -> str:
        return printify_module(self)

    @property
    def default_trainable_selector(self: Self) -> spx.SelectorSugar:
        """Return the canonical default trainable selector for this module."""
        return "parameters"

    def _resolve_trainable_selector(
        self: Self, trainable_selector: spx.SelectorSugar | None = None
    ) -> spx.SelectorSugar:
        """Resolve ``trainable_selector`` against the module default."""
        return self.default_trainable_selector if trainable_selector is None else trainable_selector

    def _partition_trainable_state(
        self: Self,
        state: spx.State,
        *,
        trainable_selector: spx.SelectorSugar | None = None,
    ) -> tuple[spx.State, spx.State]:
        """Partition ``state`` into selected trainables and the remaining state."""
        selector = spx.as_selector(self._resolve_trainable_selector(trainable_selector))
        return selector.partition_state(self, state)

    @property
    def parameters(self: Self) -> spx.State:
        """Return the default trainable state for this module.

        By default this is the ``"parameters"`` collection only. Models that
        want LoRA-only or combined training should pass an explicit
        ``trainable_selector`` to :meth:`split_parameters`,
        :meth:`split_module`, or :meth:`to_state`.
        """
        return self.split_parameters()

    def parameter_values(
        self: Self,
        *,
        selector: spx.SelectorSugar | None = None,
        extract_fn: tp.Callable | None = None,
        remove_none: bool = True,
    ) -> dict[str, tp.Any]:
        """Return a flat ``path -> array`` mapping for the selected trainables."""
        flat_values: dict[str, tp.Any] = {}
        for _collection, path, value in self.split_parameters(selector).items():
            leaf = extract_fn(value) if extract_fn is not None else value
            leaf = leaf.value if hasattr(leaf, "value") else leaf
            if remove_none and leaf is None:
                continue
            flat_values[path] = leaf
        return flat_values

    def abstract_parameter_leaves(
        self: Self,
        *,
        selector: spx.SelectorSugar | None = None,
    ) -> tuple[tuple[str, str, tuple[tp.Any, ...], tp.Any], ...]:
        """Return selected trainable leaves that are still abstract placeholders."""
        missing: list[tuple[str, str, tuple[tp.Any, ...], tp.Any]] = []
        for collection, path, value in self.split_parameters(selector).items():
            leaf = value.value if hasattr(value, "value") else value
            if isinstance(leaf, jax.ShapeDtypeStruct):
                missing.append((collection, path, tuple(leaf.shape), leaf.dtype))
        return tuple(missing)

    def assert_parameters_materialized(
        self: Self,
        *,
        selector: spx.SelectorSugar | None = None,
        context: str = "parameter materialization",
    ) -> Self:
        """Fail if a real-weight path still contains lazy ShapeDtypeStruct leaves."""
        missing = self.abstract_parameter_leaves(selector=selector)
        if not missing:
            return self

        preview_items = [
            f"{collection}/{path} shape={shape} dtype={dtype}" for collection, path, shape, dtype in missing[:8]
        ]
        preview = "; ".join(preview_items)
        if len(missing) > len(preview_items):
            preview = f"{preview}; ... +{len(missing) - len(preview_items)} more"
        raise ValueError(
            f"{context} left {len(missing)} abstract trainable parameter leaf/leaves. "
            "This means checkpoint conversion/loading did not materialize required weights. "
            f"First missing leaves: {preview}"
        )

    def split_module(
        self: Self,
        *,
        trainable_selector: spx.SelectorSugar | None = None,
    ) -> tuple[spx.GraphDef, spx.State, spx.State]:
        """Split the module into graph definition, selected trainables, and the remainder."""
        gdef, state = spx.export(self)
        graphstate, graphother = self._partition_trainable_state(state, trainable_selector=trainable_selector)
        graphother = materialize_meta_leaves(graphother, seed=42)
        return gdef, graphstate, graphother

    def merge_module(
        self: Self,
        graphdef: spx.GraphDef,
        graphstate: spx.State,
        graphother: spx.State,
    ) -> Self:
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
            >>> new_model = model.merge_module(graphdef, new_params, others)
        """
        full_state = graphstate.merge(graphother, copy=True)
        bound = spx.bind(graphdef, full_state)
        object.__setattr__(bound, "_spx_opaque", dict(self._spx_opaque))
        for opaque_name in self._spx_attr_order:
            if opaque_name not in bound._spx_attr_order:
                bound._spx_attr_order.append(opaque_name)
        # Preserve training mode across bind/reconstruct
        bound.train(self.training)
        return bound

    @property
    def graphdef(self: Self) -> spx.GraphDef:
        """Get the graph definition (structure without parameters) of the module.

        Uses spx.export to separate the graph definition from the state.
        The graph definition contains the module's structural information
        including layer types, shapes, and connections, but no parameter values.

        Returns:
            spx.GraphDef: The graph definition of the module, which can be used
                to reconstruct the module structure or for functional transformations.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> gdef = model.graphdef
            >>> # The graphdef can be used with different parameter states
            >>> new_model = spx.bind(gdef, new_params.merge(new_others, copy=True))
        """
        return spx.export(self)[0]

    @property
    def graphstate(self: Self) -> spx.State:
        """Return the default selected trainable state for the module."""
        return self.split_parameters()

    @property
    def graphother(self: Self) -> spx.State:
        """Return the complement of :attr:`graphstate` in the exported module state."""
        _, state = spx.export(self)
        _graphstate, graphother = self._partition_trainable_state(state)
        return graphother

    @property
    def graphtree_parameters_shape(self: Self) -> dict:
        """Compute shape metadata for the default selected trainable state."""
        graphtree = jax.eval_shape(lambda: self.split_parameters())
        return graphtree.raw()

    @property
    def graphtree_shape(self: Self) -> dict:
        """Compute the shapes of all state variables (including non-parameters).

        Uses jax.eval_shape on the entire module state (both parameters and
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
        _, state = spx.export(self)
        graphtree = jax.eval_shape(lambda: state)
        return graphtree.raw()

    @property
    def mesh(self: Self) -> spx.SpxMesh:
        """Get the SpectraX device mesh from the module's configuration.

        Returns the mesh used for distributed training and sharding operations.
        The mesh defines how arrays are partitioned across devices.

        Returns:
            spx.SpxMesh: The device mesh defined in self.config.mesh,
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
    def explicit_mesh(self: Self) -> spx.SpxMesh:
        """Get the explicit-axis SpectraX device mesh from the module's configuration.

        Returns the explicit mesh variant where axes are explicitly named
        and managed. This is useful for advanced sharding strategies.

        Returns:
            spx.SpxMesh: The explicit-axis device mesh defined in
                self.config.explicit_mesh.
        """
        return self.config.explicit_mesh

    @property
    def manual_mesh(self: Self) -> spx.SpxMesh:
        """Get the manual-axis SpectraX device mesh from the module's configuration.

        Returns the manual mesh variant where axis handling is done manually
        by the user. This provides maximum flexibility for custom sharding.

        Returns:
            spx.SpxMesh: The manual-axis device mesh defined in
                self.config.manual_mesh.
        """
        return self.config.manual_mesh

    def mesh_call(self: Self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Call the module under the configured JAX mesh context.

        This is a convenience method equivalent to `with self.mesh: self(*args, **kwargs)`.
        It ensures that all operations within the forward pass respect the mesh sharding
        configuration.

        Args:
            *args: Positional arguments to pass to the module.
            **kwargs: Keyword arguments to pass to the module.

        Returns:
            Any: The module output, with appropriate sharding applied
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

    @cached_property
    def causal_mask(self: Self) -> tp.Any:
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
    def frequencies(self: Self) -> tp.Any:
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
    def inv_frequencies(self: Self) -> tp.Any:
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
        """Get static arguments needed by the module's forward method.

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

    @cached_property
    def loss_strategy(self: Self):
        """Get the planning-aware loss strategy for the resolved loss function."""

        return resolve_loss_strategy(self.loss_function)

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
        leaves = jax.tree_util.tree_leaves(self.split_parameters())
        return leaves[0].dtype

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

        gdef, params, others = self.split_module()

        def _map_leaf(leaf):
            if leaf is not None:
                leaf = leaf.astype(dtype)
            return leaf

        params = jax.tree_util.tree_map(_map_leaf, params)
        self = self.merge_module(gdef, params, others)

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

        gdef, params, others = self.split_module()

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

        params = jax.tree_util.tree_map(_map, params)
        others = jax.tree_util.tree_map(_map, others)
        self = self.merge_module(gdef, params, others)

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
                uses config rules when available, otherwise resolves automatic
                sharding rules. Defaults to None.

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
            tree=self.graphtree_parameters_shape,
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

        gdef, state = spx.export(self)
        new_state = jax.tree_util.tree_map(_map, state)
        return spx.bind(gdef, new_state)

    @property
    def _shardings(self: Self):
        """Extract the sharding information for each parameter.

        Returns a nested dictionary containing the sharding information
        (PartitionSpec or NamedSharding) for each parameter in the module.

        Returns:
            dict: A nested dictionary mirroring the parameter structure,
                containing the sharding info (or empty PartitionSpec if unsharded).
        """
        gdef, state = spx.export(self)
        new_state = jax.tree_util.tree_map(
            lambda x: x.sharding if hasattr(x, "sharding") else PartitionSpec(),
            state,
        )
        return spx.bind(gdef, new_state)

    @property
    def _named_shardings(self: Self):
        """Extract the NamedSharding object for each parameter.

        Returns a nested dictionary containing the NamedSharding object
        (if present) for each parameter, or None for unsharded parameters.

        Returns:
            dict: A nested dictionary mirroring the parameter structure,
                containing NamedSharding objects or None.
        """
        gdef, state = spx.export(self)
        new_state = jax.tree_util.tree_map(
            lambda x: x.sharding if hasattr(x, "sharding") else None,
            state,
        )
        return spx.bind(gdef, new_state)

    def _get_mesh(self, mesh: spx.SpxMesh | None = None) -> spx.SpxMesh:
        """Retrieve the SpectraX device mesh, with fallback to configuration.

        Gets the mesh to use for sharding operations, prioritizing the provided
        argument over the mesh in the configuration.

        Args:
            mesh: A SpectraX device mesh to use. If None, uses self.config.mesh.

        Returns:
            spx.SpxMesh: The resolved SpectraX device mesh.

        Raises:
            ValueError: If no mesh is provided and none is found in the
                configuration (self.config.mesh is None or config doesn't exist).

        Example:
            >>> mesh = model._get_mesh()  # Uses config mesh
            >>> custom_mesh = Mesh(devices, axis_names)
            >>> mesh = model._get_mesh(custom_mesh)  # Uses provided mesh
        """
        if mesh is None:
            return self.config.mesh
        return mesh

    @property
    def runtime_sharding_resolver(self):
        """Return the model's runtime sharding resolver."""
        return self.config.runtime_sharding_resolver.with_mesh(self.config.mesh)

    @staticmethod
    def _state_to_flat_items(state: tp.Any) -> dict[tuple[str, str], tp.Any]:
        if isinstance(state, spx.State):
            return {(collection, path): leaf for collection, path, leaf in state.items()}
        if isinstance(state, Mapping):
            flat_state = flatten_dict(state)
            output: dict[tuple[str, str], tp.Any] = {}
            for path, leaf in flat_state.items():
                if not isinstance(path, tuple) or len(path) < 2:
                    continue
                output[(str(path[0]), ".".join(str(part) for part in path[1:]))] = leaf
            return output
        raise TypeError(f"Unsupported state container: {type(state).__name__}")

    @staticmethod
    def _flat_items_to_state(flat_items: Mapping[tuple[str, str], tp.Any]) -> spx.State:
        nested: dict[str, dict[str, tp.Any]] = {}
        for (collection, path), leaf in flat_items.items():
            nested.setdefault(collection, {})[path] = leaf
        return spx.State(nested)

    @staticmethod
    def _metadata_rule_path_aliases(path: str) -> tuple[str, ...]:
        """Return compatibility aliases for canonical metadata-derived paths."""
        parts = path.split(".")
        if not parts:
            return (path,)

        aliases = [tuple(parts)]
        lowered_path = ".".join(parts).lower()
        terminal = parts[-1]

        if terminal == "weight":
            aliases.append((*parts[:-1], "kernel"))
            if "norm" in lowered_path:
                aliases.append((*parts[:-1], "scale"))
            if "embed" in lowered_path:
                aliases.append((*parts[:-1], "embedding"))

        deduped: list[str] = []
        seen: set[str] = set()
        for alias_parts in aliases:
            alias_path = ".".join(alias_parts)
            if alias_path in seen:
                continue
            seen.add(alias_path)
            deduped.append(alias_path)
        return tuple(deduped)

    @staticmethod
    def _compatibility_rule_alias_patterns(pattern: str) -> tuple[str, ...]:
        """Return legacy terminal-name aliases for auto-generated regex patterns."""
        aliases = [pattern]
        lowered_pattern = pattern.lower()
        if "/weight" in pattern:
            aliases.append(pattern.replace("/weight", "/kernel"))
            if "norm" in lowered_pattern:
                aliases.append(pattern.replace("/weight", "/scale"))
            if "embed" in lowered_pattern:
                aliases.append(pattern.replace("/weight", "/embedding"))
        deduped: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            deduped.append(alias)
        return tuple(deduped)

    def _metadata_partition_specs_state(self, mesh: Mesh | None = None) -> spx.State:
        """Build a sparse partition-spec state tree from variable metadata."""
        mesh = self._get_mesh(mesh)
        resolver = self.runtime_sharding_resolver.with_mesh(mesh)
        graph_collections = self.graphstate.collections()
        data: dict[str, dict[str, tp.Any]] = {}

        for path, var in spx.iter_variables(self):
            if getattr(var, "kind", None) not in graph_collections:
                continue
            value = getattr(var, "value", None)
            shape = tuple(value.shape) if hasattr(value, "shape") else NOT_GIVEN
            spec = resolver.partition_spec_for_variable(var, shape=shape, mesh=mesh)
            if spec is None:
                continue
            data.setdefault(var.kind, {})[path] = spec

        return spx.State(data)

    def _metadata_partition_rules(self, mesh: Mesh | None = None) -> tuple[tuple[str, tp.Any], ...]:
        """Bridge metadata-derived parameter specs into legacy regex rules.

        Emits one slash-form regex per (path, leaf-name-alias) pair. The
        rules are matched against slash-separated string paths (the form
        produced by ``spectrax.match_partition_rules`` via its internal
        ``_path_to_string`` and by the on-disk Checkpointer paths), so we
        do not emit tuple-form ``(DictKey(key='...'), ...)`` regexes —
        nothing in the matching pipeline uses that representation.
        """
        mesh = self._get_mesh(mesh)
        flat_specs = self._state_to_flat_items(self._metadata_partition_specs_state(mesh=mesh))
        rules: list[tuple[str, tp.Any]] = []
        seen: set[tuple[str, tp.Any]] = set()

        for (_collection, path), spec in flat_specs.items():
            if not isinstance(spec, PartitionSpec):
                continue
            for aliased_path in self._metadata_rule_path_aliases(path):
                regex_parts: list[str] = []
                for part in aliased_path.split("."):
                    regex_parts.append(r"\d+" if part.isdigit() else re.escape(part))
                pattern = r"^(?:.*/)?" + "/".join(regex_parts) + r"(?:/.*)?$"
                key = (pattern, spec)
                if key in seen:
                    continue
                seen.add(key)
                rules.append((pattern, spec))
        return tuple(rules)

    def _get_legacy_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
        """Retrieve legacy regex-based partition rules."""

        def _normalize_rules_for_variable_leaves(rules: PartitionLike) -> PartitionLike:
            normalized: list[tuple[str, tp.Any]] = []
            for pattern, spec in rules:
                if isinstance(pattern, str):
                    if (
                        pattern.endswith("(?:/.*)?$")
                        or pattern.endswith("/.*$")
                        or pattern.endswith("(?:/value)?$")
                        or pattern.endswith("/value$")
                    ):
                        normalized_pattern = pattern
                    elif pattern.endswith("$"):
                        normalized_pattern = pattern[:-1] + r"(?:/.*)?$"
                    else:
                        normalized_pattern = pattern
                else:
                    normalized_pattern = pattern
                normalized.append((normalized_pattern, spec))
            return tuple(normalized)

        if partition_rules is not None:
            return _normalize_rules_for_variable_leaves(partition_rules)

        if not hasattr(self, "config"):
            raise ValueError("Partition rules must be provided either as an argument or through the model config.")

        try:
            rules = self.config.get_partition_rules(fully_sharded_data_parallel=True)
        except TypeError:
            rules = self.config.get_partition_rules()
        except NotImplementedError:
            rules = None

        if rules is None:
            rules = ((".*", PartitionSpec()),)

        return _normalize_rules_for_variable_leaves(rules)

    def _parameter_partition_specs(
        self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
    ) -> spx.State:
        """Return the canonical partition-spec tree for model parameters."""
        mesh = self._get_mesh(mesh)

        if partition_rules is not None:
            specs = match_partition_rules(
                rules=self._get_partition_rules(partition_rules),
                tree=self.graphtree_parameters_shape,
            )
        else:
            metadata_specs = self._metadata_partition_specs_state(mesh=mesh)
            graph_flat = self._state_to_flat_items(self.graphtree_parameters_shape)
            metadata_flat = self._state_to_flat_items(metadata_specs)

            if metadata_flat and len(metadata_flat) == len(graph_flat):
                specs = metadata_specs
            else:
                legacy_specs = match_partition_rules(
                    rules=self._get_legacy_partition_rules(None),
                    tree=self.graphtree_parameters_shape,
                )
                merged = self._state_to_flat_items(legacy_specs)
                merged.update(metadata_flat)
                specs = self._flat_items_to_state(merged)

        flat_specs = self._state_to_flat_items(specs)
        flat_shapes = self._state_to_flat_items(self.graphtree_parameters_shape)
        adjusted = 0
        for key, spec in list(flat_specs.items()):
            if not isinstance(spec, PartitionSpec):
                continue
            shape_obj = flat_shapes.get(key)
            shape = tuple(getattr(shape_obj, "shape", ()))
            if not shape:
                continue
            safe_spec = sanitize_partition_spec_for_shape(spec=spec, shape=shape, mesh=mesh)
            if safe_spec != spec:
                flat_specs[key] = safe_spec
                adjusted += 1
        if adjusted:
            specs = self._flat_items_to_state(flat_specs)
        if adjusted:
            logger.warning("Adjusted %d non-divisible parameter sharding specs.", adjusted)
        return specs

    def _apply_partition_specs_to_state(
        self,
        state: spx.State,
        partition_specs: spx.State,
        *,
        mesh: Mesh | None = None,
    ) -> spx.State:
        mesh = self._get_mesh(mesh)
        resolver = self.runtime_sharding_resolver.with_mesh(mesh)
        flat_state = self._state_to_flat_items(state)
        flat_specs = self._state_to_flat_items(partition_specs)
        variables = {(var.kind, path): var for path, var in spx.iter_variables(self)}
        updated: dict[tuple[str, str], tp.Any] = {}

        for key, leaf in flat_state.items():
            spec = flat_specs.get(key)
            if isinstance(spec, PartitionSpec) and hasattr(leaf, "shape"):
                var = variables.get(key)
                if var is not None:
                    sharding = resolver.named_sharding_for_variable(var, shape=tuple(leaf.shape), mesh=mesh)
                else:
                    sharding = None
                if sharding is None:
                    sharding = resolver.named_sharding_for_spec(spec, shape=tuple(leaf.shape), mesh=mesh)
                updated[key] = device_put_or_shard_abstract(leaf, sharding)
            else:
                updated[key] = leaf

        return self._flat_items_to_state(updated)

    def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
        """Return compatibility regex rules with metadata-derived rules first."""
        if partition_rules is not None:
            return self._get_legacy_partition_rules(partition_rules)

        metadata_rules = self._metadata_partition_rules(mesh=self._get_mesh(None))
        legacy_rules = self._get_legacy_partition_rules(None)
        if not metadata_rules:
            compatibility_rules: list[tuple[str, tp.Any]] = []
            seen: set[tuple[str, tp.Any]] = set()
            for pattern, spec in legacy_rules:
                for alias_pattern in self._compatibility_rule_alias_patterns(pattern):
                    key = (alias_pattern, spec)
                    if key in seen:
                        continue
                    seen.add(key)
                    compatibility_rules.append((alias_pattern, spec))
            return tuple(compatibility_rules)

        combined: list[tuple[str, tp.Any]] = list(metadata_rules)
        for pattern, spec in legacy_rules:
            if pattern in (".*", r".*"):
                continue
            combined.append((pattern, spec))
        compatibility_rules = []
        seen: set[tuple[str, tp.Any]] = set()
        for pattern, spec in combined:
            for alias_pattern in self._compatibility_rule_alias_patterns(pattern):
                key = (alias_pattern, spec)
                if key in seen:
                    continue
                seen.add(key)
                compatibility_rules.append((alias_pattern, spec))
        compatibility_rules.append((".*", PartitionSpec()))
        return tuple(compatibility_rules)

    def resolve_shardings_automatically(
        self,
        *,
        mesh: Mesh | None = None,
    ) -> tuple[tuple[str, jax.sharding.NamedSharding], ...]:
        """Return per-variable ``(regex, NamedSharding)`` rules.

        One rule per ``(variable, leaf-name-alias)`` pair: the regex pins
        the *literal* path (including each layer index) and the
        ``NamedSharding`` is the variable's live placement, produced by
        ``named_sharding_for_variable``.

        Layer indices are NOT collapsed to ``\\d+`` -- each layer gets
        its own rule.  Under pipeline parallelism that's the only way to
        carry per-stage information through ``(regex, sharding)`` rules:
        layer 0 may live on stage 0's submesh while layer 9 lives on
        stage 1's submesh, and a single ``\\d+``-collapsed rule cannot
        represent both.

        Args:
            mesh: Optional mesh override.  Defaults to the model's
                configured mesh.

        Returns:
            Tuple of ``(slash-form regex, NamedSharding)`` pairs.  The
            regex matches the variable's slash-separated string path
            (the form ``spectrax.match_partition_rules`` and the
            on-disk ``Checkpointer`` use), so the returned rules can be
            consumed by either pipeline.
        """
        mesh = self._get_mesh(mesh)
        resolver = self.runtime_sharding_resolver.with_mesh(mesh)
        graph_collections = self.graphstate.collections()

        rules: list[tuple[str, jax.sharding.NamedSharding]] = []
        seen: set[str] = set()
        for path, var in spx.iter_variables(self):
            if getattr(var, "kind", None) not in graph_collections:
                continue
            value = getattr(var, "value", None)
            shape = tuple(value.shape) if hasattr(value, "shape") else None
            if shape is None:
                continue
            ns = resolver.named_sharding_for_variable(var, shape=shape, mesh=mesh)
            if ns is None:
                continue
            for aliased_path in self._metadata_rule_path_aliases(path):
                # Pin literal indices -- per-layer NamedShardings can differ
                # under PP, so collapsing layers/0, layers/1, ... to ``\d+``
                # would lose the per-stage submesh info.
                regex_parts = [re.escape(part) for part in aliased_path.split(".")]
                pattern = r"^(?:.*/)?" + "/".join(regex_parts) + r"(?:/.*)?$"
                if pattern in seen:
                    continue
                seen.add(pattern)
                rules.append((pattern, ns))
        return tuple(rules)

    def _apply_sharding_fns(
        self: Self,
        sharding_fns: Mapping[str, tp.Callable],
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
        gdef, params, others = self.split_module()
        sharding_fns = flatten_dict(sharding_fns)
        _shard_keys = list(sharding_fns.keys())

        def _apply_state(state: spx.State) -> spx.State:
            new_data: dict[str, dict[str, tp.Any]] = {}
            for c, p, leaf in state.items():
                path = tuple((c + "/" + p).split("/"))
                if leaf is not None and path in _shard_keys:
                    fn = sharding_fns[path]
                    if callable(fn):
                        leaf = fn(leaf)
                    # else: pre-sharded, leave as-is
                new_data.setdefault(c, {})[p] = leaf
            return spx.State(new_data)

        params = _apply_state(params)
        others = _apply_state(others)
        self = self.merge_module(gdef, params, others)
        return self

    def shard_model(
        self: Self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
        overlay_fns: Mapping[str, tp.Callable] | None = None,
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
        gdef, graphstate, graphother = self.split_module()
        partition_specs = self._parameter_partition_specs(partition_rules=partition_rules, mesh=mesh)
        graphstate = self._apply_partition_specs_to_state(graphstate, partition_specs, mesh=mesh)
        self = self.merge_module(gdef, graphstate, graphother)
        if overlay_fns is not None:
            self = self._apply_sharding_fns(overlay_fns)
        return self

    def gather_model(
        self: Self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
        overlay_fns: Mapping[str, tp.Callable] | None = None,
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
        gdef, graphstate, graphother = self.split_module()
        partition_specs = self._parameter_partition_specs(partition_rules=partition_rules, mesh=mesh)
        _, gather_fns = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
        graphstate = jax.tree_util.tree_map(lambda f, o: f(o), gather_fns, graphstate)
        self = self.merge_module(gdef, graphstate, graphother)
        if overlay_fns is not None:
            self = self._apply_sharding_fns(overlay_fns)
        return self

    def _make_shard_fns(self: Self):
        """Build sanitized shard functions from partition rules."""
        mesh = self._get_mesh(None)
        partition_specs = self._parameter_partition_specs(None, mesh=mesh)
        shard_fns, _ = make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=mesh,
        )
        return shard_fns

    @property
    def _shard_fns(self: Self):
        """Generate sharding functions based on the module's configuration.

        Returns:
            Mapping: A mapping from flattened parameter paths to sharding
                functions that transform arrays to their sharded form.
        """
        return self._make_shard_fns()

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
            ...     lambda x: replicated_named_sharding(mesh),
            ...     model.split_module()[1:]
            ... )
            >>> model = model.apply_out_shardings(shardings)
        """
        splits = self.split_module()

        # # @erfanzar NOTE: spx.jit (not jax.jit) so MPMD meshes route to
        # sxjit and per-stage out_shardings land on the right submesh.
        @partial(spx.jit, mesh=self.mesh, out_shardings=out_shardings)
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

        # # @erfanzar NOTE: delegate to ``_apply_partition_specs_to_state`` so
        # placement goes through ``named_sharding_for_variable`` (variable-aware,
        # MPMD-correct).  Hand-rolling ``NamedSharding(self.mesh, spec)`` here
        # would collapse per-stage submeshes back to the full mesh on PP runs.
        gdef, gstate = spx.export(self)
        partition_specs = self._parameter_partition_specs(partition_rules, mesh=self.mesh)
        gstate = self._apply_partition_specs_to_state(gstate, partition_specs, mesh=self.mesh)
        self = spx.bind(gdef, gstate)
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
        # # @erfanzar NOTE: spx.jit so MPMD meshes hit sxjit -- gathering
        # across stage submeshes onto the full mesh is a cross-mesh transfer
        # the MPMD runtime knows how to schedule.
        gdef, gstate = spx.export(self)
        partition_specs = self._parameter_partition_specs(None, mesh=self.mesh)
        shardings = jax.tree_util.tree_map(
            lambda x: replicated_named_sharding(self.mesh),
            partition_specs,
        )

        @partial(spx.jit, mesh=self.mesh, out_shardings=shardings)
        def _apply(state):
            return state

        gstate = _apply(gstate)
        self = spx.bind(gdef, gstate)
        return self

    def quantize(
        self: Self,
        quantization_config: QuantizationConfig | None = None,
        apply_quantization: bool = True,
        verbose: bool | None = None,
        raise_error: bool = True,
    ) -> Self:
        """Apply quantization to the module's linear layers.

        Quantizes the model using the specified configuration by replacing
        Linear layers with their quantized equivalents (module-level).

        Args:
            quantization_config: Configuration specifying quantization dtype,
                group_size, and pattern. If None, uses default INT8 quantization.
            apply_quantization: If True, replaces Linear layers with quantized
                equivalents (e.g., Linear8bit, LinearNF4). Defaults to True.
            verbose: If True, logs information during quantization. Defaults
                to True only on process index 0.
            raise_error: If True, raises error when apply_quantization is False.
                Defaults to True.

        Returns:
            Self: The quantized model instance.

        Raises:
            ValueError: If apply_quantization is False and raise_error is True.

        Example:
            >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
            >>> # INT8 quantization
            >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
            >>> model = model.quantize(quantization_config=config)
            >>>
            >>> # NF4 quantization with custom block size
            >>> config = QuantizationConfig(dtype=QuantizationType.NF4, group_size=64)
            >>> model = model.quantize(quantization_config=config)

        Note:
            Module-level quantization (apply_quantization=True) typically provides
            better performance as it can fuse dequantization with computation.
        """
        from easydel.layers import EasyQuantizer, QuantizationConfig, QuantizationType

        if quantization_config is None:
            quantization_config = QuantizationConfig(dtype=QuantizationType.INT8)

        quantizer = EasyQuantizer(quantization_config=quantization_config)

        if verbose is None:
            verbose = jax.process_index() == 0
        if apply_quantization:
            self = quantizer.apply_quantization(self, verbose=verbose)
        elif raise_error:
            raise ValueError(
                "`apply_quantization` can't be False when quantization is requested; pass `raise_error=False` to skip."
            )
        return self

    def to_state(
        self,
        state_class: type[EasyDeLState] | None = None,
        *,
        trainable_selector: spx.SelectorSugar | None = None,
    ) -> EasyDeLState:
        """Convert the module instance into an EasyDeLState object.

        Creates an EasyDeLState that encapsulates the model's parameters and
        configuration for saving, loading, and training operations. The state
        includes the model graph definition and parameters.

        Args:
            state_class: Optional custom state class to use. Must be a subclass
                of EasyDeLState. If None, uses the default EasyDeLState class.
            trainable_selector: Selector describing which collections belong in
                the state's ``graphstate``. Defaults to
                :attr:`default_trainable_selector`.

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
        gstruct, gstate, gother = self.split_module(trainable_selector=trainable_selector)
        return state_class.create(
            step=0,
            graphdef=gstruct,
            graphstate=gstate,
            graphother=gother,
        )

    def to_torch(self, **kwargs):
        """Convert the EasyDeL module to its HuggingFace PyTorch equivalent.

        Creates a HuggingFace PyTorch model and transfers the parameters from
        this JAX model to PyTorch format. Requires the corresponding PyTorch
        model class to be available and registered.

        Args:
            **kwargs: Additional keyword arguments passed to the parameter
                transformation function (e.g., device specification).

        Returns:
            torch.spx.Module: The equivalent HuggingFace PyTorch model with
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
        """Prepare keyword arguments before passing to the module.

        Hook method that can modify or add arguments before they are passed
        to the module. The base implementation returns
        kwargs unchanged; subclasses can override for custom preprocessing.

        Args:
            **kwargs: The keyword arguments intended for the module call.

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
        """Get static arguments required by the module's forward method.

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

    def get_encoder(self: Self) -> spx.Module | EasyDeLBaseModule:
        """Return the encoder component of the model.

        Should be overridden by encoder-decoder models to return their
        encoder component. Useful for tasks that only need the encoder,
        such as feature extraction or embedding generation.

        Returns:
            spx.Module | EasyDeLBaseModule: The encoder module.

        Raises:
            NotImplementedError: If the model does not implement an encoder.
                Decoder-only models should not override this method.

        Example:
            >>> # For encoder-decoder models like T5:
            >>> encoder = model.get_encoder()
            >>> encoder_outputs = encoder(input_ids)
        """
        raise NotImplementedError()

    def get_decoder(self: Self) -> spx.Module | EasyDeLBaseModule:
        """Return the decoder component of the model.

        Should be overridden by encoder-decoder models to return their
        decoder component. Useful for tasks that need access to the
        decoder separately from the encoder.

        Returns:
            spx.Module | EasyDeLBaseModule: The decoder module.

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

    def get_embedding(self: Self) -> spx.Module | Embed:
        """Return the input embedding layer of the model.

        Should be overridden by models to return their token embedding
        layer. Useful for weight tying or accessing embeddings directly.

        Returns:
            spx.Module | Embed: The embedding layer that converts token IDs
                to dense vectors.

        Raises:
            NotImplementedError: If the model does not have an embedding
                layer accessible through this method.

        Example:
            >>> embedding = model.get_embedding()
            >>> embeds = embedding(input_ids)  # Shape: (batch, seq, hidden)
        """
        raise NotImplementedError()

    def compute_embedding(self: Self, input_ids: Int[Array, "..."] | None, *args, **kwargs) -> Float[Array, "..."]:
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
            ...     rngs=spx.Rngs(0)
            ... )

        Note:
            This method is slower than regular initialization but allows
            initializing models that would otherwise not fit in memory.
        """
        from easydel.utils.traversals import iter_module_search

        rng = kwargs.get("rngs", spx.Rngs(44))
        lazy_model = cls.lazy_init(**kwargs)
        partition_rules = lazy_model._get_partition_rules(None)
        # # @erfanzar NOTE: variable-aware sharding via the runtime resolver.
        # The resolver's ``named_sharding_for_variable`` reads each Variable's
        # metadata (including the per-stage assignment recorded by
        # ``spx.assign_stage`` during ``lazy_init``), so PP placements are
        # preserved.  Hand-rolling ``NamedSharding(lazy_model.mesh, spec)``
        # against a regex-derived PartitionSpec would collapse stages back to
        # the full mesh.
        resolver = lazy_model.runtime_sharding_resolver
        full_mesh = lazy_model.mesh

        def _sharding_for(var, fallback_spec, shape):
            if var is not None and hasattr(var, "metadata"):
                ns = resolver.named_sharding_for_variable(var, shape=shape, mesh=full_mesh)
                if ns is not None:
                    return ns
            if shape is None or not isinstance(fallback_spec, PartitionSpec):
                return replicated_named_sharding(full_mesh)
            return spx.get_corrected_named_sharding(shape, fallback_spec, raise_mesh_error=False)

        for path, module in iter_module_search(lazy_model, (spx.Module, ArrayParam)):
            if not path:
                continue
            joined_path = "/".join([str(p) for p in path])
            a = jnp.ones((1,))
            partition_spec = match_partition_rules(
                partition_rules,
                {
                    joined_path + "/weight": a,
                    joined_path + "/bias": a,
                    joined_path: a,
                },
                strict=False,
            )

            weight_var = module.weight if hasattr(module, "weight") and module.weight is not None else None
            bias_var = module.bias if hasattr(module, "bias") and module.bias is not None else None
            shardings = {
                "weight": _sharding_for(
                    weight_var,
                    partition_spec[joined_path + "/weight"],
                    tuple(weight_var.value.shape) if weight_var is not None else None,
                ),
                "bias": _sharding_for(
                    bias_var,
                    partition_spec[joined_path + "/bias"],
                    tuple(bias_var.value.shape) if bias_var is not None else None,
                ),
                "raw": _sharding_for(None, partition_spec[joined_path], None),
            }

            def _init_array_param(param: ArrayParam, key: jax.Array) -> jax.Array:
                """Initialize an ArrayParam using its stored init metadata."""
                direct_initializers = {"zeros", "ones"}
                if param.init_method in direct_initializers:
                    init_fn = getattr(jax.nn.initializers, param.init_method)
                else:
                    init_fn = getattr(
                        jax.nn.initializers,
                        param.init_method,
                        jax.nn.initializers.normal,
                    )(**(param.init_kwargs or {}))
                return init_fn(key, param.value.shape, param.value.dtype)

            if hasattr(module, "weight") and hasattr(module, "kernel_init"):
                arr = module.kernel_init(
                    key=rng.param,
                    shape=module.weight.value.shape,
                    dtype=module.weight.value.dtype,
                )
                arr = jax.device_put(arr, shardings["weight"])
                if isinstance(module.weight, spx.Parameter):
                    module.weight.value = arr
                else:
                    module.weight = arr
            # Fallback for ArrayParam weights without kernel_init.
            elif hasattr(module, "weight") and isinstance(module.weight, ArrayParam):
                if isinstance(module.weight.value, jax.ShapeDtypeStruct):
                    arr = _init_array_param(module.weight, rng.param)
                    arr = jax.device_put(arr, shardings["weight"])
                    module.weight.value = arr

            if hasattr(module, "bias") and hasattr(module, "bias_init") and module.bias is not None:
                arr = module.bias_init(
                    key=rng.param,
                    shape=module.bias.value.shape,
                    dtype=module.bias.value.dtype,
                )
                arr = jax.device_put(arr, shardings["bias"])
                module.bias.value = arr
            # Handle modules (e.g., nn.LayerNorm) that don't store init fns.
            if hasattr(module, "bias") and module.bias is not None:
                bias = module.bias
                if isinstance(bias, spx.Parameter) and isinstance(bias.value, jax.ShapeDtypeStruct):
                    arr = jax.nn.initializers.zeros(rng.param, bias.value.shape, bias.value.dtype)
                    arr = jax.device_put(arr, shardings["bias"])
                    bias.value = arr
                elif isinstance(bias, ArrayParam) and isinstance(bias.value, jax.ShapeDtypeStruct):
                    arr = _init_array_param(bias, rng.param)
                    arr = jax.device_put(arr, shardings["bias"])
                    bias.value = arr

            if hasattr(module, "weight") and hasattr(module, "embedding_init"):
                arr = module.embedding_init(
                    key=rng.param,
                    shape=module.weight.value.shape,
                    dtype=module.weight.value.dtype,
                )
                arr = jax.device_put(arr, shardings["weight"])
                module.weight.value = arr

            if hasattr(module, "weight") and hasattr(module, "scale_init"):
                arr = module.scale_init(
                    key=rng.param,
                    shape=module.weight.value.shape,
                    dtype=module.weight.value.dtype,
                )
                arr = jax.device_put(arr, shardings["weight"])
                module.weight.value = arr
            # Handle modules (e.g., nn.LayerNorm) that don't store init fns.
            elif hasattr(module, "weight") and module.weight is not None:
                weight = module.weight
                if isinstance(weight, spx.Parameter) and isinstance(weight.value, jax.ShapeDtypeStruct):
                    arr = jax.nn.initializers.ones(rng.param, weight.value.shape, weight.value.dtype)
                    arr = jax.device_put(arr, shardings["weight"])
                    weight.value = arr
                elif isinstance(weight, ArrayParam) and isinstance(weight.value, jax.ShapeDtypeStruct):
                    arr = _init_array_param(weight, rng.param)
                    arr = jax.device_put(arr, shardings["weight"])
                    weight.value = arr

            # General fallback: initialize any ArrayParam attributes that are
            # still ShapeDtypeStruct (e.g. A_log, D, dt_bias in Mamba layers).
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue
                if isinstance(attr, ArrayParam) and isinstance(attr.value, jax.ShapeDtypeStruct):
                    arr = _init_array_param(attr, rng.param)
                    # No per-attribute sharding info available here; just assign.
                    attr.value = arr

            if hasattr(module, "rngs"):
                module.rngs = rng.fork(1)[0]
            if hasattr(module, "resure"):
                _raw_sharding = shardings["raw"]
                module.resure(rng.param, shard_fn=lambda x, _s=_raw_sharding: jax.device_put(x, _s))
        parameter_collections = {"parameters", "lora"} if lazy_model.lora_is_enabled else {"parameters"}
        for path, module in spx.iter_variables(lazy_model):
            if module.kind not in parameter_collections:
                continue
            if path and type(module.value) is jax.ShapeDtypeStruct:
                if isinstance(module, ArrayParam):
                    # ArrayParam stores init info in its own attributes.
                    direct_initializers = {"zeros", "ones"}
                    if module.init_method in direct_initializers:
                        init_fn = getattr(jax.nn.initializers, module.init_method)
                    else:
                        init_fn = getattr(
                            jax.nn.initializers,
                            module.init_method,
                            jax.nn.initializers.normal,
                        )(**(module.init_kwargs or {}))
                    arr = init_fn(rng.param, module.value.shape, module.value.dtype)
                    module.value = arr
                else:
                    logger.warning(
                        f"({type(module).__name__}) found empty array at " + ("/".join([str(s) for s in path]))
                    )

        return lazy_model

    @classmethod
    def lazy_init(cls: type[Self], **kwargs) -> Self:
        """Perform lazy initialization using jax.eval_shape.

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
            ...     rngs=spx.Rngs(0)
            ... )
            >>> # Inspect shapes without allocating memory
            >>> print(lazy_model.graphtree_parameters_shape)

        Note:
            The returned model cannot be used for computation directly.
            Use sequential_init() or regular __init__ for usable models.
        """
        rngs = kwargs.pop("rngs", None)

        def _init(rngs):
            return cls(**kwargs, rngs=rngs)

        return jax.eval_shape(_init, rngs=rngs)

    def merge_lora_parameters(self: Self, pytree: dict) -> Self:
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
            >>> model = model.merge_lora_parameters(lora_params)
            >>> # Model now has LoRA weights baked into base weights

        See Also:
            split_lora_parameters: Inverse operation to extract LoRA params.
            apply_lora_to_layers: Apply LoRA to specific layers.
        """
        from easydel.infra.utils import merge_lora_parameters

        self = merge_lora_parameters(self, pytree)
        return self

    def split_lora_parameters(self: Self) -> tp.Any:
        """Split merged LoRA parameters back out from the base model.

        Extracts LoRA adaptation matrices that were previously merged using
        merge_lora_parameters() or a similar process. Restores the base model
        weights to their original pre-merge state.

        Returns:
            dict: A pytree containing the extracted LoRA parameters
                (A and B matrices) that can be saved or reapplied later.

        Example:
            >>> # Extract LoRA params for saving
            >>> lora_params = model.split_lora_parameters()
            >>> save_lora_params(lora_params, "lora_checkpoint/")
            >>> # Base model weights are restored

        See Also:
            merge_lora_parameters: Inverse operation to merge LoRA params.
        """
        from easydel.infra.utils import split_lora_parameters

        pytree = split_lora_parameters(self)
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
        for _, tensor in spx.iter_variables(self):
            if isinstance(tensor, nn.LoraParameter):
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

        from easydel.layers import (
            ColumnParallelLinearQuantized,
            ParallelLinearQuantized,
            RowParallelLinearQuantized,
        )

        # Check 1: any child module is a quantized layer type.
        # This works for lazy models where variables hold ShapeDtypeStruct.
        for _, module in spx.iter_modules(self):
            if isinstance(
                module,
                (
                    RowParallelLinearQuantized,
                    ColumnParallelLinearQuantized,
                    ParallelLinearQuantized,
                ),
            ):
                return True

        # Check 2: any variable value is a quantized array type.
        # This works for fully-materialized models.
        for _, var in spx.iter_variables(self):
            val = getattr(var, "value", getattr(var, "raw_value", var))
            if isinstance(val, (Array8B, ArrayNF4, Array1B)):
                return True
            if getattr(val, "dtype", None) in [
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
        rngs: spx.Rngs | None = None,
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
            merge_lora_parameters: Merge LoRA weights into base weights.
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

        Replaces LoraLinear layers with their original spectrax.nn.Linear
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
        for path, module in traversals.iter_module_search(self, spx.Module):
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

    def _build_transform_fn(self, shard_fns=None):
        """Build a HuggingFace-to-EasyDeL transformation function.

        Args:
            shard_fns: Optional sharding functions. If None, no sharding is applied.

        Returns:
            Callable: A partial function (StateDictConverter.huggingface_to_easydel).
        """
        from easydel.layers import BaseMoeModule, Embed, ParallelMoELinear
        from easydel.utils import traversals
        from easydel.utils.parameters_transformation import StateDictConverter

        embedding_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, nn.Embed)]
        embedding_path.extend([".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, Embed)])
        layernorm_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, LayerNorm)]
        moe_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, ParallelMoELinear)]
        moe_block_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(self, BaseMoeModule)]

        kwargs = dict(
            embedding_layer_names=embedding_path,
            layernorm_names=layernorm_path,
            moe_names=list(set([names.split(".")[-1] for names in moe_path])),
            moe_block_names=list(set([names.split(".")[-1] for names in moe_block_path])),
            moe_block_path=moe_block_path,
            moe_path=moe_path,
            dtype=self.param_dtype,
            reform_param=self._get_reform_param(),
        )
        if shard_fns is not None:
            kwargs["shard_fns"] = shard_fns
        return partial(StateDictConverter.huggingface_to_easydel, **kwargs)

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
        return self._build_transform_fn(shard_fns=self._shard_fns)

    @property
    def _generate_compatible_graphdef(self: Self):
        """Create a graph definition compatible with generation tasks.

        Generation often requires specific configurations (like disabling
        gradient checkpointing). This method creates a temporary generation-
        compatible configuration, performs lazy initialization, and extracts
        the resulting graph definition.

        Returns:
            spx.GraphDef: A graph definition suitable for use during generation,
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
        gdef, _ = spx.export(dummy)
        return gdef

    @property
    def _generate_compatible_graphother(self: Self):
        """Create non-parameter state compatible with generation tasks.

        Similar to _generate_compatible_graphdef, creates a temporary
        generation-compatible configuration, lazy-initializes, and extracts
        the non-parameter state variables with concrete values.

        Returns:
            spx.State: A graph state containing non-parameter variables
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
        _, state = spx.export(dummy)
        _graphstate, gother = self._partition_trainable_state(state)
        gother = traversals.recreate_meta_values(gother)
        return gother

    @property
    def parameters_sharding(self: Self) -> dict:
        """Get the sharding annotation for the default selected trainable values.

        Returns:
            dict: A flat ``path -> sharding`` mapping for the default
                trainable selector.
        """
        return jax.tree_util.tree_map(
            lambda x: x.sharding if hasattr(x, "sharding") else None,
            self.parameter_values(),
        )

    def merge_parameters(
        self: Self,
        tree: spx.State,
        *,
        selector: spx.SelectorSugar | None = None,
    ) -> Self:
        """Merge a selected trainable state tree back into the module."""
        gdef, _graphstate, gother = self.split_module(trainable_selector=selector)
        self = self.merge_module(gdef, tree, gother)
        return self

    def split_parameters(self, selector: spx.SelectorSugar | None = None) -> spx.State:
        """Return the selected trainable state as an :class:`spx.State`."""
        _, state = spx.export(self)
        graphstate, _graphother = self._partition_trainable_state(state, trainable_selector=selector)
        return graphstate

    def merge_parameter_values(
        self: Self,
        parameter_values: dict,
        *,
        selector: spx.SelectorSugar | None = None,
    ) -> Self:
        """Merge a flat or nested parameter-values mapping into the selected state."""
        current_state = self.split_parameters(selector).flat_state()
        if not is_flatten(parameter_values):
            parameter_values = flatten_dict(parameter_values)
        for key, value in parameter_values.items():
            if key in current_state:
                current_state[key].value = value
            else:
                raise KeyError(f"Parameter key {key} not found in the current model state.")
        self = self.merge_parameters(unflatten_dict(current_state), selector=selector)
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
            logger.warning_once("Calculating Flops Failed!")
            flops = 1
        return flops

    def _flop(self, *args, **kwargs) -> float | None:
        """Estimate FLOPs for a single forward pass using JAX's make_jaxpr.

        Uses JAX's computation graph analysis to estimate the floating point
        operations required for one forward pass with the given arguments.

        Args:
            *args: Positional arguments to pass to the module.
            **kwargs: Keyword arguments to pass to the module.

        Returns:
            float | None: The estimated FLOP count, or None if calculation fails.

        Note:
            This provides a more accurate estimate than flops_per_token() but
            requires actually tracing the computation graph.
        """
        from .utils import count_flop_jaxpr

        return count_flop_jaxpr(jax.make_jaxpr(self.forward)(*args, **kwargs))

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
        return self._build_transform_fn(shard_fns=None)

    @property
    def _default_loss_config(self: Self) -> LossConfig | None:
        """Get the default LossConfig for this module.

        Subclasses can override this property to return a default LossConfig
        instance specific to the model's task.

        Returns:
            LossConfig | None: The default loss configuration, or None.
        """
        return None

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
                if not hasattr(self.config, "num_labels"):
                    raise ValueError(
                        "in order to use `SequenceClassification` Models in `EasyDeL` you first need to attach"
                        " `num_labels` to model `config`"
                    )
                loss_config = LossConfig(num_labels=self.config.num_labels)

        if labels is None:
            raise ValueError("`labels` can not be `None` for computing loss.")
        loss_kwargs = loss_kwargs or {}
        forward_batch = batch
        try:
            call_signature = inspect.signature(self.forward)
        except (TypeError, ValueError):
            call_signature = None

        if call_signature is not None:
            call_parameters = call_signature.parameters
            if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in call_parameters.values()):
                accepted_keys = set(call_parameters.keys())
                forward_batch = {key: value for key, value in batch.items() if key in accepted_keys}

        # Most task heads accept either token IDs or input embeddings, not both simultaneously.
        if forward_batch.get("input_ids", None) is not None and forward_batch.get("inputs_embeds", None) is not None:
            forward_batch = dict(forward_batch)
            forward_batch.pop("inputs_embeds", None)

        forward_plan = self.loss_strategy.plan_forward(
            module=self,
            labels=labels,
            loss_config=loss_config,
            batch=batch,
            loss_kwargs=loss_kwargs,
        )
        if forward_plan.forward_kwargs:
            forward_batch = dict(forward_batch)
            forward_batch.update(forward_plan.forward_kwargs)

        outputs = self(**forward_batch)

        loss_output: LossMetrics = self.loss_strategy.compute(
            module=self,
            outputs=outputs,
            labels=labels,
            loss_config=loss_config,
            batch=batch,
            loss_kwargs=loss_kwargs,
            paxis=None if self.mesh.is_mpmd else self.config.partition_axis,
            forward_plan=forward_plan,
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
        w = self.get_embedding().weight.value.T if tie_embeddings else None
        return self.get_lm_head()(hidden_states, w=w)

    def make_lm_head_fn(self) -> "Callable[[Array], Array]":
        """Return a trace-safe callable that projects hidden states to logits.

        The returned function can safely be called from inside any JAX
        traced region — ``jax.lax.scan``, ``jax.lax.fori_loop``,
        ``jax.checkpoint``, and their compositions — without triggering
        ``ValueError``.

        **Why this is needed:**  When gradient checkpointing is enabled the
        LM-head linear layer is wrapped with ``nn.remat``.  SpecTrax's remat
        uses a split/merge protocol that *mutates* Variables
        (``update_from_state``) — which fails when the call site is inside
        a different JAX trace level (e.g. ``lax.scan`` body under
        ``jax.grad``).  This method bypasses the ``nn.remat`` wrapper by
        calling the head's ``native_forward`` directly (reads-only on SpecTrax
        Variables — no mutation, no trace-context check).

        The default implementation resolves tied-embedding weights and
        calls ``native_forward`` on the LM head.  Model subclasses that
        add post-processing (e.g. logit soft-capping or scaling) should
        override this method so the returned function reproduces the same
        semantics.

        Trainers should call this method **once** before entering a
        traced loop and use the returned function inside the loop body.

        Returns:
            A callable ``fn(hidden_states) -> logits`` that preserves
            all model-specific projection semantics.
        """
        head = self.get_lm_head()

        tie_embeddings = next(
            (
                getattr(self.config, key)
                for key in ["tie_word_embeddings", "use_lm_head", "share_input_output_layers"]
                if hasattr(self.config, key)
            ),
            False,
        )
        w = self.get_embedding().weight.value.T if tie_embeddings else None

        _native_forward = getattr(head, "native_forward", None)

        if _native_forward is None:

            def _native_forward(hidden_states: "Array", *, w: "Array | None" = None) -> "Array":
                if w is None:
                    return head(hidden_states)
                return head(hidden_states, w=w)

        def _project(hidden_states: "Array") -> "Array":
            return _native_forward(hidden_states, w=w)

        return _project

    @staticmethod
    def _recursive_config_children(config: EasyDeLBaseConfig) -> tuple[EasyDeLBaseConfig, ...]:
        """Collect nested config objects without evaluating dynamic properties.

        We intentionally avoid ``dir(config)`` + ``getattr(config, ...)`` here,
        because that can evaluate computed properties (for example mesh accessors)
        and trigger side effects during graph-def rebuild.
        """
        config_dict = getattr(config, "__dict__", None)
        if not isinstance(config_dict, dict):
            return ()

        out: list[EasyDeLBaseConfig] = []
        seen_ids: set[int] = set()

        sub_configs = config_dict.get("sub_configs")
        if isinstance(sub_configs, dict):
            for attr_name in sub_configs.keys():
                sub_cfg = config_dict.get(attr_name, None)
                if isinstance(sub_cfg, EasyDeLBaseConfig):
                    sub_cfg_id = id(sub_cfg)
                    if sub_cfg_id not in seen_ids:
                        out.append(sub_cfg)
                        seen_ids.add(sub_cfg_id)

        for value in config_dict.values():
            if isinstance(value, EasyDeLBaseConfig):
                value_id = id(value)
                if value_id not in seen_ids:
                    out.append(value)
                    seen_ids.add(value_id)

        return tuple(out)

    @staticmethod
    def _apply_recursive_config_updates(config: EasyDeLBaseConfig, updates: Mapping[str, tp.Any]) -> None:
        """Apply overrides to nested configs discovered from concrete attributes."""
        for sub_cfg in EasyDeLBaseModule._recursive_config_children(config):
            sub_cfg_dict = getattr(sub_cfg, "__dict__", None)
            for key, value in updates.items():
                if (isinstance(sub_cfg_dict, dict) and key in sub_cfg_dict) or hasattr(type(sub_cfg), key):
                    try:
                        setattr(sub_cfg, key, value)
                    except AttributeError:
                        # Skip read-only attributes/properties while applying broad overrides.
                        continue

    @staticmethod
    def _normalize_rebuild_quantization_config(config: EasyDeLBaseConfig):
        quantization_config = getattr(config, "quantization_config", None)
        if quantization_config is None:
            return None

        from easydel.layers import QuantizationConfig

        if isinstance(quantization_config, dict):
            quantization_config = QuantizationConfig(**quantization_config)
            config.quantization_config = quantization_config
        return quantization_config

    def _lazy_init_rebuilt_module(self, config: EasyDeLBaseConfig):
        module = self.lazy_init(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=self.rngs,
        )

        quantization_config = self._normalize_rebuild_quantization_config(config)
        if quantization_config is None or not self.is_quantized:
            return module

        from easydel.layers import EasyQuantizer

        return EasyQuantizer(quantization_config=quantization_config).apply_quantization(module, verbose=False)

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
            self._apply_recursive_config_updates(config, kwargs)
        module = self._lazy_init_rebuilt_module(config)
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
            spx.GraphDef: A new graph definition with updated configuration
                that can be merged with existing parameters.

        Example:
            >>> # Get a new graphdef with different settings
            >>> new_gdef = model.new_graphdef(attn_mechanism='flash')
            >>> # Merge with existing parameters
            >>> new_model = spx.bind(new_gdef, model.graphstate.merge(model.graphother, copy=True))
        """
        config = deepcopy(self.config)
        for k, v in kwargs.items():
            setattr(config, k, v)
        if recursive_update:
            self._apply_recursive_config_updates(config, kwargs)
        module = self._lazy_init_rebuilt_module(config)
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
        from ejkernel.callib._ejit import _get_args_signature  # pyright: ignore[reportMissingTypeStubs]

        dict_config = self.config.to_dict()
        if pop_things:
            for pops in pop_things:
                dict_config.pop(pops)
        tree_hash = _get_args_signature((self.graphstate, self.graphother), dict_config)
        bytes_in = hashlib.md5((tree_hash).encode("utf-8")).digest()
        return int.from_bytes(bytes_in, byteorder="big", signed=True)


__all__ = (
    "EasyDeLBaseConfig",
    "EasyDeLBaseConfigDict",
    "EasyDeLBaseModule",
    "ParameterTransformRule",
)
