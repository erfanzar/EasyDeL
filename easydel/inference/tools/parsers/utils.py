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

vLLM keeps tool parsing helpers under `tool_parsers/utils.py`.
EasyDeL keeps them under `easydel.inference.tools.utils`.

This module re-exports EasyDeL's utilities to make it easier to keep
ported parsers aligned with upstream vLLM implementations.
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
