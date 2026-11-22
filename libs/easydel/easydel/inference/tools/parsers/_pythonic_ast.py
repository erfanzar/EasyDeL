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

"""Shared AST helpers for pythonic-style tool-call parsers.

The pythonic, OLMo-3 and Llama-4 pythonic parsers all turn ``name(arg=value, ...)`` calls into
OpenAI ``ToolCall`` objects through Python's ``ast`` module, and stream them by completing
partial Python source. This module is the single implementation of that machinery.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from ...openai_api_modules import DeltaFunctionCall, DeltaToolCall, FunctionCall, ToolCall
from ..utils import stable_json_stream_prefix


class UnexpectedAstError(Exception):
    """Raised when parsed output does not match the expected pythonic tool-call structure.

    Signals that the AST is not a list of function calls, that an argument is not a supported
    literal, or that brackets are mismatched while completing partial source.
    """


_NAME_LITERALS: dict[str, Any] = {"null": None, "true": True, "false": False}


def _is_json_constant(value: Any) -> bool:
    """Return True if *value* is a JSON-serializable scalar constant."""
    return value is None or isinstance(value, bool | int | float | str)


def get_parameter_value(val: ast.expr) -> Any:
    """Extract a Python value from an AST expression node.

    Recursively processes AST nodes to extract literal values: JSON-compatible constants,
    signed numeric constants (``-1``, ``+2.5``), dicts, lists, and the JSON-style names
    ``null``/``true``/``false``.

    Args:
        val: The AST expression node to extract the value from.

    Returns:
        The Python value represented by the AST node.

    Raises:
        UnexpectedAstError: If the node contains non-literal expressions, non-JSON constants
            (such as ``bytes``, ``complex``, or ``Ellipsis``), or invalid dict keys.
    """
    if isinstance(val, ast.Constant):
        if not _is_json_constant(val.value):
            raise UnexpectedAstError(f"Unsupported constant literal type: {type(val.value).__name__}")
        return val.value
    if isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.USub | ast.UAdd):
        operand = val.operand
        if (
            isinstance(operand, ast.Constant)
            and isinstance(operand.value, int | float)
            and not isinstance(operand.value, bool)
        ):
            return -operand.value if isinstance(val.op, ast.USub) else operand.value
        raise UnexpectedAstError("Signed tool call arguments must be numeric literals")
    if isinstance(val, ast.Dict):
        if not all(isinstance(k, ast.Constant) and _is_json_constant(k.value) for k in val.keys):
            raise UnexpectedAstError("Dict tool call arguments must have JSON-compatible literal keys")
        return {k.value: get_parameter_value(v) for k, v in zip(val.keys, val.values, strict=False)}
    if isinstance(val, ast.List):
        return [get_parameter_value(v) for v in val.elts]
    if isinstance(val, ast.Name) and val.id in _NAME_LITERALS:
        return _NAME_LITERALS[val.id]
    raise UnexpectedAstError("Tool call arguments must be literals")


def handle_single_tool(call: ast.Call) -> ToolCall:
    """Convert an AST Call node to a ToolCall object.

    Args:
        call: The AST Call node representing a function call.

    Returns:
        A ToolCall with the function name and JSON-encoded keyword arguments.

    Raises:
        UnexpectedAstError: If the call's func is not a simple name or an argument is not a
            supported literal.
    """
    if not isinstance(call.func, ast.Name):
        raise UnexpectedAstError("Invalid tool call name")
    if call.args:
        raise UnexpectedAstError("Positional tool call arguments are not supported")
    function_name = call.func.id
    arguments = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise UnexpectedAstError("Starred keyword arguments are not supported")
        arguments[keyword.arg] = get_parameter_value(keyword.value)
    try:
        serialized_arguments = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise UnexpectedAstError("Tool call arguments must be JSON-serializable") from exc
    return ToolCall(
        type="function",
        function=FunctionCall(name=function_name, arguments=serialized_arguments),
    )


def parse_pythonic_call_list(source: str) -> list[ToolCall] | None:
    """Parse ``[fn(k=v), ...]`` strictly as a non-empty list of valid pythonic tool calls.

    Returns ``None`` on syntax error, empty lists, non-call elements, positional/starred
    arguments, or non-literal argument values. Runs in linear time without regex backtracking.
    """
    stripped = source.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        module = ast.parse(stripped)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
        return None
    parsed = module.body[0].value
    if not isinstance(parsed, ast.List) or not parsed.elts:
        return None
    if not all(isinstance(elt, ast.Call) for elt in parsed.elts):
        return None
    try:
        return [handle_single_tool(elt) for elt in parsed.elts]
    except (UnexpectedAstError, RecursionError, MemoryError, ValueError):
        return None


def _count_top_level_delimiters(segment: str, sep: str) -> tuple[int, int]:
    """Count top-level ``sep`` and ``,`` tokens outside strings and nested brackets."""
    depth = 0
    quote: str | None = None
    escaped = False
    sep_count = 0
    comma_count = 0

    for char in segment:
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in {"[", "(", "{"}:
            depth += 1
        elif char in {"]", ")", "}"}:
            if depth > 0:
                depth -= 1
        elif depth == 0:
            if char == sep:
                sep_count += 1
            elif char == ",":
                comma_count += 1

    return sep_count, comma_count


def make_valid_python(text: str) -> tuple[str, str] | None:
    """Complete partial Python source by appending the closing brackets/quotes it needs.

    Args:
        text: Partial Python text that may be incomplete.

    Returns:
        ``(completed_text, added_text)`` where ``added_text`` is what was appended, or ``None``
        when the text ends in an incomplete parameter name or value (for example ``=`` or ``:``)
        and more tokens are needed.

    Raises:
        UnexpectedAstError: If brackets are mismatched.
    """
    bracket_stack: list[tuple[str, int]] = []
    escaped = False
    for idx, char in enumerate(text):
        # Strings own their contents, including brackets and the other quote.
        # Toggle escape state so an even run of backslashes does not escape a quote.
        if bracket_stack and bracket_stack[-1][0] in {"'", '"'}:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == bracket_stack[-1][0]:
                bracket_stack.pop()
            continue
        if char in {"[", "(", "{", "'", '"'}:
            bracket_stack.append((char, idx))
        elif char in {"]", ")", "}"}:
            opening = {"]": "[", ")": "(", "}": "{"}[char]
            if not bracket_stack or bracket_stack.pop()[0] != opening:
                raise UnexpectedAstError("Mismatched brackets")
    if escaped:
        # A trailing escape needs another token before a quote can close it.
        return None

    in_string = bool(bracket_stack and bracket_stack[-1][0] in {"'", '"'})
    if not in_string:
        text = text.rstrip()
        if text.endswith("=") or text.endswith(":"):
            return None
        if text.endswith(","):
            text = text[:-1].rstrip()
        if text.endswith("=") or text.endswith(":"):
            return None

    if bracket_stack and bracket_stack[-1][0] == "{":
        inner_dict_text = text[bracket_stack[-1][1] + 1 :]
        num_keys, num_commas = _count_top_level_delimiters(inner_dict_text, ":")
        if num_keys <= num_commas:
            return None
    if bracket_stack and bracket_stack[-1][0] == "(":
        inner_params_text = text[bracket_stack[-1][1] + 1 :]
        num_full_param_names, num_commas = _count_top_level_delimiters(inner_params_text, "=")
        if num_full_param_names <= num_commas:
            return None
    if (
        in_string
        and len(bracket_stack) >= 2
        and bracket_stack[-2][0] == "{"
    ):
        inner_dict_text = text[bracket_stack[-2][1] + 1 : bracket_stack[-1][1]]
        num_keys, num_commas = _count_top_level_delimiters(inner_dict_text, ":")
        if num_keys <= num_commas:
            return None
    if len(bracket_stack) == 1 and bracket_stack[0][0] == "[" and not text.endswith("[") and not text.endswith(")"):
        return None

    added_text = ""
    for char, _idx in reversed(bracket_stack):
        if char == "[":
            added_text += "]"
        elif char == "(":
            added_text += ")"
        elif char == "{":
            added_text += "}"
        elif char == "'":
            added_text += "'"
        elif char == '"':
            added_text += '"'

    return text + added_text, added_text


def compute_tool_delta(
    previously_sent_args: str, new_call: ToolCall, index: int, withheld_suffix: str
) -> DeltaToolCall | None:
    """Compute the delta between previously sent and current tool-call arguments.

    Args:
        previously_sent_args: JSON argument text already streamed to the client.
        new_call: The current ToolCall with the latest arguments.
        index: The index of this tool call in the response.
        withheld_suffix: Characters to withhold from the end of the arguments (incomplete syntax
            added only to make the source parseable).

    Returns:
        A DeltaToolCall with the function name on the first delta, only the argument growth on
        later deltas, or ``None`` when there is nothing new to send.
    """
    new_call_args = new_call.function.arguments
    if withheld_suffix:
        if new_call_args.endswith(withheld_suffix):
            new_call_args = new_call_args[: -len(withheld_suffix)]
        new_call_args = stable_json_stream_prefix(new_call_args)
    if not previously_sent_args:
        return DeltaToolCall(
            id=new_call.id,
            type="function",
            index=index,
            function=DeltaFunctionCall(name=new_call.function.name, arguments=new_call_args),
        )

    if not new_call_args.startswith(previously_sent_args):
        return None
    arg_diff = new_call_args[len(previously_sent_args) :]
    return DeltaToolCall(id=None, index=index, function=DeltaFunctionCall(arguments=arg_diff)) if arg_diff else None
