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

"""Shared tool-schema lookup and parameter coercion for XML-style tool parsers.

The Qwen3-Coder, Seed-OSS and MiniMax-M2 parsers all receive parameter values as raw
strings and must look up the tool definition in the request to coerce them to the
JSON-Schema declared type. This module is the single implementation of that logic.
"""

from __future__ import annotations

import ast
import json
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from eformer.loggings import get_logger
from jsonschema import validators
from jsonschema.exceptions import ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable

from ...openai_api_modules import ToolDefinition

logger = get_logger(__name__)


def get_arguments_config(func_name: str, tools: list[ToolDefinition] | None) -> dict:
    """Extract the parameter properties schema for a function.

    Looks up ``func_name`` in the provided tool definitions and returns the ``properties``
    dict from its JSON Schema parameters (or the parameters dict itself when it has no
    ``properties`` key), which maps parameter names to their type metadata.

    Args:
        func_name: The name of the function to look up.
        tools: List of tool definitions from the request, or None.

    Returns:
        A dict mapping parameter names to their schema definitions, or an empty dict if the
        function is not found or has no schema.
    """
    if tools is None:
        return {}
    for config in tools:
        if not hasattr(config, "type") or not (hasattr(config, "function") and hasattr(config.function, "name")):
            continue
        if config.type == "function" and config.function.name == func_name:
            if not hasattr(config.function, "parameters"):
                return {}
            params = config.function.parameters
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            elif isinstance(params, dict):
                return params
            else:
                return {}
    logger.debug("Tool '%s' is not defined in the tools list.", func_name)
    return {}


def lookup_tool_properties(func_name: str | None, tools: list[ToolDefinition] | None) -> dict[str, Any]:
    """Return the ``properties`` mapping of a tool's parameter schema.

    Unlike :func:`get_arguments_config`, a parameters dict without a ``properties`` key yields
    an empty dict rather than the parameters dict itself.

    Args:
        func_name: The function name to look up.
        tools: Tool definitions from the request, or None.

    Returns:
        The ``properties`` dict, or an empty dict when the tool or schema is missing.
    """
    if not tools or func_name is None:
        return {}
    for tool in tools:
        if hasattr(tool, "function") and tool.function.name == func_name and hasattr(tool.function, "parameters"):
            params = tool.function.parameters
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            return {}
    return {}


def _union_param_types(schema: dict) -> set[str]:
    """Collect explicit and enum/const-inferred types from nested union branches.

    Args:
        schema: A parameter schema, possibly containing anyOf/oneOf branches.

    Returns:
        Normalized candidate type names. Leaves without type or value constraints
        default to string, matching the ordinary parameter conversion fallback.
    """
    declared = schema.get("type", [])
    types = {declared} if isinstance(declared, str) else {t for t in declared if isinstance(t, str)}
    if "type" not in schema:
        values = schema.get("enum", [])
        if "const" in schema:
            values = [*values, schema["const"]]
        json_types = {
            type(None): "null",
            bool: "boolean",
            int: "integer",
            float: "number",
            dict: "object",
            list: "array",
        }
        types.update(json_types.get(type(value), "string") for value in values)
    for keyword in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(keyword, []):
            if isinstance(branch, dict):
                types.update(_union_param_types(branch))
    return {t.strip().lower() for t in types} or {"string"}


def _param_value_has_type(value: Any, param_type: str) -> bool:
    """Check a coerced value against a schema type, without treating bool as int."""
    param_type = param_type.strip().lower()
    if param_type == "null":
        return value is None
    if param_type in {"string", "str", "text", "varchar", "char", "enum"}:
        return isinstance(value, str)
    if param_type.startswith(("int", "uint", "long", "short", "unsigned")):
        return type(value) is int or (type(value) is float and value.is_integer())
    if param_type.startswith(("num", "float")):
        return type(value) in (int, float)
    if param_type in {"boolean", "bool", "binary"}:
        return type(value) is bool
    if param_type == "object" or param_type.startswith("dict"):
        return isinstance(value, dict)
    if param_type in {"array", "arr"} or param_type.startswith("list"):
        return isinstance(value, list)
    return False


def _matches_union_constraints(value: Any, schema: dict | bool) -> bool:
    """Validate complete JSON-Schema branch membership, not just its scalar type.

    Structural, logical, conditional and referenced constraints all distinguish
    oneOf branches. Use the schema's standard validator instead of duplicating a
    partial validator. Compare decimal JSON numbers exactly (including enum,
    const and multipleOf), and never fetch remote references from request schemas.
    """
    if isinstance(schema, bool):
        return schema

    def exact_numbers(obj):
        if type(obj) is float:
            return Fraction(str(obj))
        if isinstance(obj, list):
            return [exact_numbers(item) for item in obj]
        if isinstance(obj, dict):
            return {key: exact_numbers(item) for key, item in obj.items()}
        return obj

    def declared_types(node):
        if isinstance(node, dict):
            declared = node.get("type", [])
            if isinstance(declared, str):
                yield declared
            elif isinstance(declared, list):
                yield from (kind for kind in declared if isinstance(kind, str))
            for child in node.values():
                yield from declared_types(child)
        elif isinstance(node, list):
            for child in node:
                yield from declared_types(child)

    def has_type(instance, kind):
        # Preserve the parser's existing scalar aliases (int/float/bool/etc.)
        # while the standard validator handles every other schema constraint.
        if isinstance(instance, Fraction):
            instance = 1 if instance.denominator == 1 else 1.5
        return _param_value_has_type(instance, kind)

    base = validators.validator_for(schema)
    checker = base.TYPE_CHECKER.redefine_many(
        {
            kind: lambda _checker, instance, kind=kind: has_type(instance, kind)
            for kind in {"integer", "number", *declared_types(schema)}
        }
    )
    validator = validators.extend(base, type_checker=checker)
    try:
        validator(exact_numbers(schema), registry=Registry()).validate(exact_numbers(value))
    except (ValidationError, Unresolvable):
        return False
    return True


def _lossless_json_value(value: Any) -> Any:
    """Convert exact parsed JSON decimals to serializable numbers without rounding.

    Integral decimals become arbitrary-precision integers. Fractional decimals must
    survive JSON's float string round trip; otherwise callers preserve the raw text
    instead of silently changing a tool argument or its enum/const membership.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Nonfinite JSON number")
        if value == value.to_integral_value():
            return int(value)
        converted = float(value)
        if Decimal(str(converted)) != value:
            raise ValueError("JSON number would lose precision")
        return converted
    if isinstance(value, list):
        return [_lossless_json_value(member) for member in value]
    if isinstance(value, dict):
        return {key: _lossless_json_value(member) for key, member in value.items()}
    return value


def _convert_union_param_value(value: str, schema: dict, param_name: str, func_name: str) -> Any:
    """Coerce a union parameter only when the result has a permitted type.

    Args:
        value: Raw XML parameter text.
        schema: The parameter's union schema.
        param_name: Parameter name for conversion diagnostics.
        func_name: Tool name for conversion diagnostics.

    Returns:
        A permitted typed value, or the unchanged text if no typed conversion succeeds.
        String branches are fallbacks, not permission to JSON-decode arbitrary text.
    """
    types = _union_param_types(schema)
    if value.lower() == "null":
        return None if "null" in types and _matches_union_constraints(None, schema) else value
    # Try non-string branches first, independent of schema branch ordering. Numeric
    # candidates must preserve the input before testing enum/const membership: using
    # the legacy scalar float converter can round a nonmember into an allowed value.
    for param_type in sorted(types):
        if param_type in {"null", "string", "str", "text", "varchar", "char", "enum"}:
            continue
        if param_type in {"boolean", "bool", "binary"} and value.lower() not in {"true", "false"}:
            continue
        if _param_value_has_type(1, param_type):
            try:
                converted = _lossless_json_value(Decimal(value))
            except (InvalidOperation, ValueError, TypeError, OverflowError):
                continue
        elif _param_value_has_type({}, param_type) or _param_value_has_type([], param_type):
            try:
                exact = json.loads(value, parse_float=Decimal, parse_constant=Decimal)
            except (json.JSONDecodeError, InvalidOperation):
                # Preserve Python-literal support, but recover numeric source text
                # before literal_eval can hide rounding in a nested float constant.
                source = value.strip()
                try:
                    expression = ast.parse(source, mode="eval")
                    for node in ast.walk(expression):
                        if isinstance(node, ast.Constant) and type(node.value) is float:
                            node.value = _lossless_json_value(Decimal(ast.get_source_segment(source, node)))
                    converted = ast.literal_eval(expression)
                except (SyntaxError, ValueError, TypeError, InvalidOperation, OverflowError):
                    continue
            else:
                try:
                    converted = _lossless_json_value(exact)
                except (ValueError, OverflowError):
                    continue
        else:
            converted = convert_param_value(value, param_name, {param_name: {"type": param_type}}, func_name)
        if _param_value_has_type(converted, param_type) and _matches_union_constraints(converted, schema):
            return converted
    return value


def convert_param_value(param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
    """Convert a raw string parameter value to its schema-defined type.

    Uses the parameter's type from ``param_config`` to coerce the string into the appropriate
    Python type (int, float, bool, dict, list, etc.). For anyOf/oneOf unions and type lists,
    converts only to a permitted type (including types inferred from enum/const values),
    preserving raw text for nullable strings and failed conversions. Single-type schemas
    retain the raw-string/``ast.literal_eval`` fallback.

    Args:
        param_value: The raw string value to convert.
        param_name: Name of the parameter (for schema lookup and logging).
        param_config: Parameter properties schema from the tool definition.
        func_name: Name of the parent function (for logging).

    Returns:
        The converted value in the appropriate Python type, or the original string if conversion
        is not possible.
    """
    schema = param_config.get(param_name)
    if isinstance(schema, dict) and ("anyOf" in schema or "oneOf" in schema or isinstance(schema.get("type"), list)):
        return _convert_union_param_value(param_value, schema, param_name, func_name)
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        if param_config != {}:
            logger.debug(
                "Parsed parameter '%s' is not defined in the tool parameters for tool '%s', directly "
                "returning the string value.",
                param_name,
                func_name,
            )
        return param_value
    if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
        param_type = str(param_config[param_name]["type"]).strip().lower()
    else:
        param_type = "string"
    if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            logger.debug(
                "Parsed value '%s' of parameter '%s' is not an integer in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        try:
            float_param_value = float(param_value)
            return float_param_value if float_param_value - int(float_param_value) != 0 else int(float_param_value)
        except (ValueError, TypeError, OverflowError):
            logger.debug(
                "Parsed value '%s' of parameter '%s' is not a float in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type in ["boolean", "bool", "binary"]:
        param_value = param_value.lower()
        if param_value not in ["true", "false"]:
            logger.debug(
                "Parsed value '%s' of parameter '%s' is not a boolean (`true` or `false`) in tool '%s', "
                "degenerating to false.",
                param_value,
                param_name,
                func_name,
            )
        return param_value == "true"
    else:
        if param_type in ["object", "array", "arr"] or param_type.startswith("dict") or param_type.startswith("list"):
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be parsed with json.loads in tool '%s', "
                    "will try other methods to parse it.",
                    param_value,
                    param_name,
                    func_name,
                )
        try:
            param_value = ast.literal_eval(param_value)
        except (ValueError, SyntaxError, TypeError):
            logger.debug(
                "Parsed value '%s' of parameter '%s' cannot be converted via Python ast.literal_eval() "
                "in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
        return param_value
