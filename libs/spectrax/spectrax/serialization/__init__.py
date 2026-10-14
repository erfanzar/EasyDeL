# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Serialization module for Spectrax (TensorStore-only).

Provides efficient checkpoint saving and loading built on top of JAX's
:class:`GlobalAsyncCheckpointManager` with support for:
- TensorStore backend for large-scale distributed storage
- Async operations for parallel I/O
- Sharding preservation for distributed arrays (TP/FSDP)
- Structured PyTree saves with treedef recovery
"""

from .async_manager import AsyncCheckpointManager
from .checkpointer import (
    Checkpointer,
    CheckpointInterval,
    find_latest_checkpoint,
    read_checkpoint_metadata,
)
from .serialization import tree_deserialize_leaves, tree_serialize_leaves

__all__ = (
    "AsyncCheckpointManager",
    "CheckpointInterval",
    "Checkpointer",
    "find_latest_checkpoint",
    "read_checkpoint_metadata",
    "tree_deserialize_leaves",
    "tree_serialize_leaves",
)
