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


"""Serialization module for eFormer.

This module provides efficient checkpoint saving and loading with support for:
- TensorStore backend for large-scale storage
- Async operations for parallel I/O
- Sharding for distributed arrays
- Google Cloud Storage support
- SafeTensors format compatibility
"""

from . import fsspec_utils
from .async_manager import AsyncCheckpointManager
from .base_manager import CheckpointManager
from .checkpointer import Checkpointer, CheckpointInterval
from .serialization import tree_deserialize_leaves, tree_serialize_leaves

__all__ = (
    "AsyncCheckpointManager",
    "CheckpointInterval",
    "CheckpointManager",
    "Checkpointer",
    "fsspec_utils",
    "tree_deserialize_leaves",
    "tree_serialize_leaves",
)
