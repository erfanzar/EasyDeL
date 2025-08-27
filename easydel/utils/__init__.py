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

"""EasyDeL Utilities Module.

Provides essential utilities and helper functions for the EasyDeL framework,
including compilation optimization, checkpoint management, data handling,
logging, and various development tools.

Submodules:
    analyze_memory: Memory analysis and profiling tools
    checkpoint_managers: Model checkpoint saving/loading utilities
    cli_helpers: Command-line interface helpers
    compiling_utils: JAX compilation caching and optimization
    data_managers: Dataset loading and management
    helpers: General helper functions and timing utilities
    lazy_import: Lazy module importing utilities
    parameters_transformation: Parameter transformation tools
    rngs_utils: Random number generator utilities
    traversals: PyTree traversal utilities

Key Components:
    AsyncCheckpointManager: Manage model checkpoints
    DataManager: Handle dataset loading and mixing
    Timer/Timers: Performance timing utilities
    ejit: Enhanced JIT compilation with caching
    LazyModule: Lazy module loading

Example:
    >>> from easydel.utils import Timer, ejit, get_logger
    >>>
    >>> # Use timer for performance measurement
    >>> with Timer("model_forward"):
    ...     output = model(input)
    >>>
    >>> # Use enhanced JIT with caching
    >>> @ejit
    ... def optimized_function(x):
    ...     return x @ x.T
    >>>
    >>> # Get configured logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting training...")

Note:
    This module provides the foundational utilities that support
    efficient development and deployment of EasyDeL models.
"""

from eformer.paths import ePath, ePathLike

from . import analyze_memory, compiling_utils, traversals
from .compiling_utils import ejit, load_cached_functions, load_compiled_fn, save_compiled_fn
from .data_managers import (
    DataManager,
    DatasetLoadError,
    DatasetMixture,
    DatasetType,
    TextDatasetInform,
    VisualDatasetInform,
)
from .helpers import Timer, Timers, capture_time, check_bool_flag, get_cache_dir
from .lazy_import import LazyModule, is_package_available

__all__ = (
    "DataManager",
    "DatasetLoadError",
    "DatasetMixture",
    "DatasetType",
    "LazyModule",
    "TextDatasetInform",
    "Timer",
    "Timers",
    "VisualDatasetInform",
    "analyze_memory",
    "capture_time",
    "check_bool_flag",
    "compiling_utils",
    "ePath",
    "ePathLike",
    "ejit",
    "get_cache_dir",
    "is_package_available",
    "load_cached_functions",
    "load_compiled_fn",
    "save_compiled_fn",
    "traversals",
)
