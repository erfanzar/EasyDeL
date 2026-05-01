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

"""EasyDeL Infrastructure Module.

Core infrastructure components for building and managing neural network models
in EasyDeL. This module provides the foundational classes and utilities that
all EasyDeL models are built upon.

Key Components:
    - Base configuration system for model parameters
    - Base module class with common model functionality
    - State management for model parameters and optimization
    - Loss computation utilities
    - Mixin classes for generation and bridging
    - Partitioning and sharding utilities

Classes:
    EasyDeLBaseConfig: Base configuration class for all models
    EasyDeLBaseConfigDict: Dictionary-based configuration
    EasyDeLBaseModule: Base class for all model implementations
    EasyDeLState: State container for model parameters and optimizer state
    LossConfig: Configuration for loss computation
    AxisPolicy: Canonical semantic-axis configuration
    PartitionAxis: Legacy axis specification compatibility surface
    RuntimeShardingResolver: Runtime sharding lowering helper
    TensorLayout: Parameter metadata for compound logical layouts

Utilities:
    PyTree: JAX pytree utilities
    auto_pytree: Decorator for automatic pytree registration
    Rngs: Random number generator management
    sharding: SpectraX sharding utilities for distributed training

Example:
    >>> from easydel.infra import (
    ...     EasyDeLBaseConfig,
    ...     EasyDeLBaseModule,
    ...     EasyDeLState
    ... )
    >>> # Create a configuration
    >>> config = EasyDeLBaseConfig(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     num_hidden_layers=12
    ... )
    >>> # Initialize a model (subclass of EasyDeLBaseModule)
    >>> model = MyModel(config, dtype=jnp.float32)
    >>> # Create state for training
    >>> state = EasyDeLState.create(
    ...     model=model,
    ...     config=config,
    ...     optimizer=optimizer
    ... )

Note:
    This module provides the infrastructure that enables:
    - Efficient distributed training
    - Automatic mixed precision
    - Gradient checkpointing
    - Model quantization
    - State serialization and deserialization
"""

import typing as tp

import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import PyTree, auto_pytree
from spectrax import PartitionAxis
from spectrax import sharding as _spectrax_sharding

from easydel.typings import ConfigDataclass

from .sharding import AxisPolicy, RuntimeShardingResolver, TensorLayout, logical_axis_rules, sharding_for_layout

Rngs = spx.Rngs
sharding = _spectrax_sharding

if tp.TYPE_CHECKING:  # pragma: no cover
    from .base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict
    from .base_module import EasyDeLBaseModule
    from .base_state import EasyDeLState
    from .elarge import BenchmarkConfig, eLargeModel
    from .loss_utils import LossConfig


def init_cluster():
    """
    Initialize the distributed computing cluster for EasyDeL.

    This function sets up the distributed configuration for JAX-based distributed
    computing. It attempts to initialize the JAX distributed runtime using the
    DistributedConfig from eformer.

    Note:
        This function should be called once at the beginning of your program
        before any distributed operations.
    """
    import os as _os

    import jax as _jax

    _logger = get_logger("EasyDeL-Distributed")
    _os.environ.setdefault("JAX_ENABLE_PREEMPTION_SERVICE", "true")
    _jax.config.update("jax_enable_preemption_service", True)

    if _jax.distributed.is_initialized():
        _logger.info("JAX distributed already initialized; using existing setup.")
        return

    from eformer.executor import DistributedConfig as _DistributedConfig

    try:
        _DistributedConfig().initialize()
    except RuntimeError:
        if _jax.distributed.is_initialized():
            _logger.info("JAX distributed already initialized; using existing setup.")
            return
        raise


__all__ = (
    "AxisPolicy",
    "BenchmarkConfig",
    "EasyDeLBaseConfig",
    "EasyDeLBaseConfigDict",
    "EasyDeLBaseModule",
    "EasyDeLState",
    "LossConfig",
    "PartitionAxis",
    "PyTree",
    "Rngs",
    "RuntimeShardingResolver",
    "TensorLayout",
    "auto_pytree",
    "eLargeModel",
    "init_cluster",
    "logical_axis_rules",
    "sharding",
    "sharding_for_layout",
)


def __getattr__(name: str):  # pragma: no cover
    """Lazy attribute loader to avoid import cycles in core modules."""
    if name in {"EasyDeLBaseConfig", "EasyDeLBaseConfigDict"}:
        from .base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict

        return {"EasyDeLBaseConfig": EasyDeLBaseConfig, "EasyDeLBaseConfigDict": EasyDeLBaseConfigDict}[name]
    if name == "EasyDeLBaseModule":
        from .base_module import EasyDeLBaseModule

        return EasyDeLBaseModule
    if name == "EasyDeLState":
        from .base_state import EasyDeLState

        return EasyDeLState
    if name == "LossConfig":
        from .loss_utils import LossConfig

        return LossConfig
    if name == "eLargeModel":
        from .elarge import eLargeModel

        return eLargeModel
    if name == "BenchmarkConfig":
        from .elarge import BenchmarkConfig

        return BenchmarkConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover
    """Return the union of module globals and ``__all__`` for ``dir()`` callers.

    Returns:
        list[str]: Sorted attribute names, including lazy attributes resolved
            by :func:`__getattr__` that are not present in ``globals()``.
    """
    return sorted(set(globals().keys()) | set(__all__))
