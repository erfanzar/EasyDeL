# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""
eformer: A JAX-based library for efficient transformer implementations.

The eformer library provides utilities for building, training, and deploying
transformer models with JAX. It includes support for:

- Distributed training and sharding strategies via the escale module
- Quantization support (NF4, INT8) through the jaximus implicit array system
- Argument parsing utilities for dataclass-based configurations
- Universal path handling for local and cloud storage (GCS)
- Logging and profiling utilities

Submodules:
    aparser: Dataclass-based argument parsing utilities.
    common_types: Type aliases and constants for JAX/sharding configurations.
    escale: Distributed execution and mesh creation utilities.
    executor: Training execution helpers.
    jaximus: Implicit array system for lazy/quantized array handling.
    loggings: Logging utilities with colored output and progress tracking.
    mpric: Multi-process RPC utilities.
    optimizers: Optimizer implementations and wrappers.
    paths: Universal path utilities for local and cloud storage.
    pytree: JAX pytree utilities.
    serialization: Model serialization and checkpointing.

Example:
    >>> import eformer
    >>> from eformer.jaximus import ImplicitArray, implicit
    >>> from eformer.paths import ePath
    >>>
    >>> # Load data from GCS or local filesystem
    >>> data = ePath("gs://bucket/data.npy").read_bytes()
    >>>
    >>> # Use implicit arrays for memory-efficient operations
    >>> @implicit
    ... def forward(x, weights):
    ...     return x @ weights
"""

from logging import getLogger as _getLogger

_getLogger("jax.experimental.array_serialization.serialization").setLevel(40)

__version__ = "0.0.101"

__all__ = (
    "aparser",
    "common_types",
    "escale",
    "executor",
    "jaximus",
    "loggings",
    "mpric",
    "optimizers",
    "paths",
    "pytree",
    "serialization",
)
