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

"""Checkpoint management utilities.

Provides tools for saving and loading model checkpoints with support
for various storage backends including local filesystem and cloud storage.

Classes:
    CheckpointManager: Main checkpoint management class
    EasyPath: Universal path abstraction for multiple backends

Types:
    EasyPathLike: Type hint for path-like objects

Constants:
    ALLOWED_DATA_TYPES: Supported data types for checkpointing
    DTYPE_TO_STRING_MAP: Mapping from JAX dtypes to strings
    STRING_TO_DTYPE_MAP: Mapping from strings to JAX dtypes

Key Features:
    - Unified API for local and cloud storage
    - Efficient streaming of large checkpoints
    - Automatic dtype conversion and preservation
    - Support for PyTree structures
    - Checkpoint versioning and rotation

Example:
    >>> from easydel.utils.checkpoint_managers import CheckpointManager
    >>>
    >>> manager = CheckpointManager(
    ...     checkpoint_dir="checkpoints/",
    ...     max_to_keep=5
    ... )
    >>>
    >>> # Save checkpoint
    >>> manager.save_checkpoint(
    ...     state=model_state,
    ...     step=1000
    ... )
    >>>
    >>> # Load latest checkpoint
    >>> state = manager.load_checkpoint()
"""

from .path_utils import EasyPath, EasyPathLike
from .streamer import ALLOWED_DATA_TYPES, DTYPE_TO_STRING_MAP, STRING_TO_DTYPE_MAP, CheckpointManager

__all__ = (
    "ALLOWED_DATA_TYPES",
    "DTYPE_TO_STRING_MAP",
    "STRING_TO_DTYPE_MAP",
    "CheckpointManager",
    "EasyPath",
    "EasyPathLike",
)
