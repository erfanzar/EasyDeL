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

"""Compatibility shim for serving-style imports.

This module provides a compatibility layer for serving-style imports, allowing code
ported from external parser layouts to work seamlessly with EasyDeL's internal structure.

External parser packages keep tool parsing helpers under `tool_parsers/utils.py`, while EasyDeL
keeps them under `easydel.inference.tools.utils`. This module re-exports
EasyDeL's utilities to maintain API compatibility with that layout.

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

import json
import re
from uuid import uuid4

from ...openai_api_modules import DeltaFunctionCall, DeltaToolCall
from ..utils import (
    consume_space,
    extract_intermediate_diff,
    find_all_indices,
    find_common_prefix,
    find_common_suffix,
    is_complete_json,
    partial_json_loads,
)


def stream_json_calls(calls: list[dict], previous: list[dict], sent: list[str], ids: list[str], complete: bool):
    """Emit every top-level call, keeping stable identities and argument prefixes.

    Only parsed call objects are inspected; nested argument keys cannot create calls.
    Incomplete argument serialization is withheld until its prefix is stable.
    """
    deltas = []
    for index, call in enumerate(calls):
        while len(sent) <= index:
            sent.append("")
        while len(ids) <= index:
            ids.append("")
        name = call.get("name")
        if not isinstance(name, str) or not name:
            continue
        first = not ids[index]
        if first:
            ids[index] = f"call_{uuid4().hex}"
        args = call.get("arguments")
        fragment = ""
        if args is not None:
            serialized = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            if complete or index < len(calls) - 1:
                stable = serialized
            elif index < len(previous) and previous[index].get("arguments") is not None:
                old = previous[index]["arguments"]
                old = old if isinstance(old, str) else json.dumps(old, ensure_ascii=False)
                # A partial parser synthesizes closing syntax. Even identical
                # snapshots may represent a still-open empty object or string.
                stable = find_common_prefix(old, serialized).rstrip('"} ]')
                # Numeric tokens may still grow into exponents (1 -> 1e-2),
                # changing their serialization rather than simply appending.
                if not isinstance(args, str):
                    stable = re.sub(r"(?<=[ :\[,])[-+0-9.eE]+$", "", stable)
            else:
                stable = ""
            if stable.startswith(sent[index]):
                fragment = stable[len(sent[index]) :]
                sent[index] += fragment
        if first or fragment:
            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=ids[index] if first else None,
                    type="function" if first else None,
                    function=DeltaFunctionCall(name=name if first else None, arguments=fragment or None),
                )
            )
    previous[:] = calls
    return deltas


__all__ = [
    "consume_space",
    "extract_intermediate_diff",
    "find_all_indices",
    "find_common_prefix",
    "find_common_suffix",
    "is_complete_json",
    "partial_json_loads",
]
