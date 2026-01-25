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

"""Compatibility shim for vLLM-style imports.

This module provides a compatibility layer for vLLM-style imports, allowing code
ported from vLLM to work seamlessly with EasyDeL's internal structure.

vLLM keeps tool parsing helpers under `tool_parsers/utils.py`, while EasyDeL
keeps them under `easydel.inference.tools.utils`. This module re-exports
EasyDeL's utilities to maintain API compatibility with vLLM's layout.

Re-exported Functions:
    consume_space: Skip whitespace characters in a string starting from an index.
    extract_intermediate_diff: Extract the difference between two JSON strings.
    find_all_indices: Find all occurrences of a substring within a string.
    find_common_prefix: Find the common prefix between two strings.
    find_common_suffix: Find the common suffix between two strings.
    is_complete_json: Check if a string represents complete, valid JSON.
    partial_json_loads: Parse potentially incomplete JSON with configurable options.

Example:
    >>> from easydel.inference.tools.parsers.utils import is_complete_json
    >>> is_complete_json('{"key": "value"}')
    True
    >>> is_complete_json('{"key": ')
    False

Note:
    This module exists primarily for compatibility. For new code, consider
    importing directly from `easydel.inference.tools.utils`.
"""

from __future__ import annotations

from ..utils import (
    consume_space,
    extract_intermediate_diff,
    find_all_indices,
    find_common_prefix,
    find_common_suffix,
    is_complete_json,
    partial_json_loads,
)

__all__ = [
    "consume_space",
    "extract_intermediate_diff",
    "find_all_indices",
    "find_common_prefix",
    "find_common_suffix",
    "is_complete_json",
    "partial_json_loads",
]
