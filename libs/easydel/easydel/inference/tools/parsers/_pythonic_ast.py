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


class UnexpectedAstError(Exception):
    """Raised when parsed output does not match the expected pythonic tool-call structure.

    Signals that the AST is not a list of function calls, that an argument is not a supported
    literal, or that brackets are mismatched while completing partial source.
    """


_NAME_LITERALS: dict[str, Any] = {"null": None, "true": True, "false": False}


def get_parameter_value(val: ast.expr) -> Any:
    """Extract a Python value from an AST expression node.

    Recursively processes AST nodes to extract literal values: constants, signed numeric
    constants (``-1``, ``+2.5``), dicts, lists, and the JSON-style names ``null``/``true``/``false``.

    Args:
        val: The AST expression node to extract the value from.

    Returns:
        The Python value represented by the AST node.

    Raises:
        UnexpectedAstError: If the node contains non-literal expressions or dict keys that are
            not constants.
    """
    if isinstance(val, ast.Constant):
        return val.value
    if isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.USub | ast.UAdd):
        operand = val.operand
        if (
            isinstance(operand, ast.Constant)
            and isinstance(operand.value, int | float | complex)
            and not isinstance(operand.value, bool)
        ):
            return -operand.value if isinstance(val.op, ast.USub) else operand.value
        raise UnexpectedAstError("Signed tool call arguments must be numeric literals")
    if isinstance(val, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in val.keys):
            raise UnexpectedAstError("Dict tool call arguments must have literal keys")
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
    function_name = call.func.id
    arguments = {}
    for keyword in call.keywords:
        arguments[keyword.arg] = get_parameter_value(keyword.value)
    return ToolCall(
        type="function",
        function=FunctionCall(name=function_name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


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
    bracket_stack: list[str] = []
    escaped = False
    for char in text:
        # Strings own their contents, including brackets and the other quote.
        # Toggle escape state so an even run of backslashes does not escape a quote.
        if bracket_stack and bracket_stack[-1] in {"'", '"'}:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == bracket_stack[-1]:
                bracket_stack.pop()
            continue
        if char in {"[", "(", "{", "'", '"'}:
            bracket_stack.append(char)
        elif char in {"]", ")", "}"}:
            opening = {"]": "[", ")": "(", "}": "{"}[char]
            if not bracket_stack or bracket_stack.pop() != opening:
                raise UnexpectedAstError("Mismatched brackets")
    if escaped:
        # A trailing escape needs another token before a quote can close it.
        return None

    text = text.rstrip()
    if text.endswith("=") or text.endswith(":"):
        return None
    if bracket_stack and bracket_stack[-1] == "{":
        trailing_dict_text = text[: text.rfind("{")]
        num_keys = trailing_dict_text.count(":")
        num_values = trailing_dict_text.count(",")
        if num_keys <= num_values:
            return None
    if bracket_stack and bracket_stack[-1] == "(":
        trailing_params_text = text[: text.rfind("(")]
        num_full_param_names = trailing_params_text.count("=")
        num_full_param_values = trailing_params_text.count(",")
        if num_full_param_names <= num_full_param_values:
            return None
    if text.endswith(","):
        text = text[:-1]
    if bracket_stack and bracket_stack[-1] == "[" and not text.endswith("[") and not text.endswith(")"):
        return None

    added_text = ""
    for char in reversed(bracket_stack):
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
    if withheld_suffix and new_call_args.endswith(withheld_suffix):
        new_call_args = new_call_args[: -len(withheld_suffix)]
    if not previously_sent_args:
        return DeltaToolCall(
            id=new_call.id,
            type="function",
            index=index,
            function=DeltaFunctionCall(name=new_call.function.name, arguments=new_call_args),
        )

    arg_diff = new_call_args[len(previously_sent_args) :]
    return DeltaToolCall(id=None, index=index, function=DeltaFunctionCall(arguments=arg_diff)) if arg_diff else None
