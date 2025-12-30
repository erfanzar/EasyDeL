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
    PartitionAxis: Axis specification for model parallelism

Utilities:
    PyTree: JAX pytree utilities
    auto_pytree: Decorator for automatic pytree registration
    Rngs: Random number generator management
    escale: Scaling utilities for distributed training

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

from eformer import escale
from eformer.escale import PartitionAxis
from eformer.executor import DistributedConfig
from eformer.loggings import get_logger
from eformer.pytree import PyTree, auto_pytree
from flax.nnx import Rngs

if tp.TYPE_CHECKING:  # pragma: no cover
    from .base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict
    from .base_module import EasyDeLBaseModule
    from .base_state import EasyDeLState
    from .elarge_model import eLargeModel
    from .loss_utils import LossConfig


def init_cluster():
    """
    Initialize the distributed computing cluster for EasyDeL.

    This function sets up the distributed configuration for JAX-based distributed
    computing. It attempts to initialize the JAX distributed runtime using the
    DistributedConfig from eformer.

    The function handles two types of failures gracefully:
    - RuntimeError: Occurs when JAX distributed is already initialized manually
    - General Exception: Occurs when running in single-process mode

    Both failures are logged as warnings and do not raise exceptions, allowing
    the code to continue execution in both distributed and single-process scenarios.

    Note:
        This function should be called once at the beginning of your program
        before any distributed operations.
    """
    _logger = get_logger("EasyDeL-Distributed")

    try:
        DistributedConfig().initialize()
    except RuntimeError:
        _logger.warn("Failed to initialize jax-dist if you have initialized that manually you can ignore this warning")
    except Exception:  # maybe it's a single process
        _logger.warn("Failed to initialize jax-dist")


__all__ = (
    "EasyDeLBaseConfig",
    "EasyDeLBaseConfigDict",
    "EasyDeLBaseModule",
    "EasyDeLState",
    "LossConfig",
    "PartitionAxis",
    "PyTree",
    "Rngs",
    "auto_pytree",
    "eLargeModel",
    "escale",
    "init_cluster",
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
        from .elarge_model import eLargeModel

        return eLargeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(__all__))
