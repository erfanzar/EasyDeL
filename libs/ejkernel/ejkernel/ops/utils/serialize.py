# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Simple JSON serialization utilities for configuration persistence.

This module provides basic JSON serialization functions for converting
Python objects to and from JSON strings. It includes support for dataclasses
and provides a fallback to string representation for complex objects.

Functions:
    to_json: Convert Python objects to JSON strings
    from_json: Parse JSON strings back to Python objects

These utilities are used primarily for:
    - Configuration serialization in persistent caches
    - Simple object persistence across program runs
    - Debug output and logging of complex objects

Note:
    For more comprehensive serialization needs (including JAX arrays,
    functions, and complex nested structures), use the fingerprint.stable_json
    function which provides deterministic serialization.

Example Usage:
    >>> @dataclass
    >>> class Config:
    ...     lr: float = 0.01
    ...     batch_size: int = 32
    >>>
    >>> config = Config(lr=0.001, batch_size=64)
    >>> json_str = to_json(config)
    >>> restored = from_json(json_str)
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


def to_json(obj: Any) -> str:
    """Convert a Python object to a JSON string.

    Provides JSON serialization with support for dataclasses and fallback
    string representation for objects that aren't directly JSON-serializable.

    Args:
        obj: Object to serialize to JSON

    Returns:
        JSON string representation of the object

    Note:
        - Dataclasses are converted using dataclasses.asdict()
        - Other non-serializable objects fall back to repr()
        - Output uses sorted keys and compact separators for consistency
    """

    def default(o):
        """JSON serialization fallback for dataclasses and non-serializable objects."""
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return repr(o)

    return json.dumps(obj, default=default, sort_keys=True, separators=(",", ":"))


def from_json(s: str) -> Any:
    """Parse a JSON string and return the corresponding Python object.

    Args:
        s: JSON string to parse

    Returns:
        Python object parsed from the JSON string

    Raises:
        json.JSONDecodeError: If the string is not valid JSON

    Note:
        This is a simple wrapper around json.loads() and returns
        standard Python types (dict, list, str, int, float, bool, None).
        It does not attempt to reconstruct original object types.
    """
    return json.loads(s)
