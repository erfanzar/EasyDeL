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

"""Public XML parser regressions for union branches with implicit JSON types."""

from __future__ import annotations

import json
import re

import pytest
from easydel.inference.tools import ToolParserManager

from .test_tool_parsers import _request, _Tokenizer

pytestmark = pytest.mark.cpu_ok


class _XMLTokenizer(_Tokenizer):
    _TOKENS = (*_Tokenizer._TOKENS, "<tool_call>", "</tool_call>")


@pytest.mark.parametrize("parser_name", ["qwen3_coder", "qwen3_xml", "seed_oss"])
@pytest.mark.parametrize("streaming", [False, True], ids=["batch", "streaming"])
@pytest.mark.parametrize(
    "schema, value, expected",
    [
        pytest.param({"anyOf": [{"enum": [1, 2]}, {"type": "null"}]}, "1", 1, id="integer-enum"),
        pytest.param({"anyOf": [{"enum": [True, False]}, {"type": "null"}]}, "true", True, id="boolean-enum"),
        pytest.param({"anyOf": [{"enum": [{"x": 1}]}, {"type": "null"}]}, '{"x":1}', {"x": 1}, id="object-enum"),
        pytest.param({"anyOf": [{"enum": [[1, 2]]}, {"type": "null"}]}, "[1,2]", [1, 2], id="array-enum"),
        pytest.param(
            {"anyOf": [{"const": {"x": [9007199254740992.0]}}, {"type": "null"}]},
            '{"x":[9007199254740993.0]}',
            '{"x":[9007199254740993.0]}',
            id="nested-large-nonmember-not-rounded-to-const",
        ),
        pytest.param(
            {"anyOf": [{"enum": [[1.5]]}, {"type": "null"}]},
            "[1.5000000000000001]",
            "[1.5000000000000001]",
            id="nested-fractional-nonmember-not-rounded-to-enum",
        ),
        pytest.param(
            {"anyOf": [{"enum": [{"x": [1, 0.3]}]}, {"type": "null"}]},
            '{"x":[1e0,0.3]}',
            {"x": [1, 0.3]},
            id="nested-equivalent-numbers",
        ),
        pytest.param(
            {"anyOf": [{"const": {"x": [9007199254740992.0]}}, {"type": "null"}]},
            "{'x':[9007199254740993.0]}",
            "{'x':[9007199254740993.0]}",
            id="python-literal-nonmember-not-rounded-to-const",
        ),
        pytest.param(
            {"anyOf": [{"enum": [{"x": [1, -0.3, True]}]}, {"type": "null"}]},
            "{'x':[1e0,-0.3,True]}",
            {"x": [1, -0.3, True]},
            id="python-literal-container-fallback-preserved",
        ),
        pytest.param({"oneOf": [{"const": 7}, {"type": "null"}]}, "7", 7, id="integer-const"),
        pytest.param({"anyOf": [{"enum": [None]}, {"type": "integer"}]}, "null", None, id="null-enum"),
        pytest.param({"anyOf": [{"enum": ["123"]}, {"type": "null"}]}, "123", "123", id="string-enum-control"),
        pytest.param({"oneOf": [{"const": "true"}, {"type": "null"}]}, "true", "true", id="string-const-control"),
        pytest.param({"anyOf": [{"const": 1.5}, {"type": "null"}]}, "1.5", 1.5, id="number-const"),
        pytest.param({"anyOf": [{"enum": [1]}, {"type": "null"}]}, "1.0", 1, id="integer-enum-decimal"),
        pytest.param({"oneOf": [{"const": 1}, {"type": "null"}]}, "1e0", 1, id="integer-const-exponent"),
        pytest.param({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "3e0", 3, id="typed-integer-exponent"),
        pytest.param(
            {"anyOf": [{"const": 1e23}, {"type": "null"}]},
            "100000000000000000000000",
            100000000000000000000000,
            id="large-equivalent-decimal-const",
        ),
        pytest.param(
            {"anyOf": [{"const": 1e23}, {"type": "null"}]},
            "99999999999999991611392",
            "99999999999999991611392",
            id="large-float-binary-approximation-is-not-const",
        ),
        pytest.param(
            {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "1/2",
            "1/2",
            id="fraction-literal-is-not-a-json-number",
        ),
        pytest.param(
            {"anyOf": [{"const": 9007199254740992.0}, {"type": "null"}]},
            "9007199254740993",
            "9007199254740993",
            id="large-nonmember-not-rounded-to-const",
        ),
        pytest.param(
            {"anyOf": [{"enum": [9007199254740992.0]}, {"type": "null"}]},
            "9007199254740993.0",
            "9007199254740993.0",
            id="large-decimal-nonmember-not-rounded-to-enum",
        ),
        pytest.param(
            {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "9007199254740993",
            9007199254740993,
            id="large-integer-number-preserved-exactly",
        ),
        pytest.param(
            {"anyOf": [{"const": 1.5}, {"type": "null"}]},
            "1.5000000000000001",
            "1.5000000000000001",
            id="fractional-nonmember-not-rounded-to-const",
        ),
        pytest.param(
            {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": -1}]},
            "3",
            3,
            id="disjoint-integer-ranges-positive",
        ),
        pytest.param(
            {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": -1}]},
            "-3",
            -3,
            id="disjoint-integer-ranges-negative",
        ),
        pytest.param(
            {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": 5}]},
            "3",
            "3",
            id="overlapping-integer-ranges",
        ),
        pytest.param(
            {"oneOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "number", "maximum": 0}]},
            "0",
            0,
            id="exclusive-numeric-boundary",
        ),
        pytest.param(
            {"oneOf": [{"type": "number", "exclusiveMaximum": 0}, {"type": "number", "minimum": 0}]},
            "0",
            0,
            id="exclusive-upper-boundary",
        ),
        pytest.param(
            {"anyOf": [{"type": "integer"}, {"type": "null"}], "minimum": 4},
            "3",
            "3",
            id="containing-numeric-bound",
        ),
        pytest.param(
            {"oneOf": [{"type": "number", "multipleOf": 0.1}, {"type": "number", "minimum": 1}]},
            "0.3",
            0.3,
            id="decimal-multiple-of",
        ),
        pytest.param({"oneOf": [{"enum": [1]}, True]}, "1", "1", id="oneof-true-branch-is-ambiguous"),
        pytest.param({"oneOf": [{"enum": [1]}, False]}, "1", 1, id="oneof-false-branch-never-matches"),
        pytest.param({"oneOf": [{"enum": [None]}, True]}, "null", "null", id="oneof-null-true-branch-is-ambiguous"),
        pytest.param({"anyOf": [{"enum": [1]}, False]}, "1", 1, id="anyof-false-branch-never-matches"),
        pytest.param(
            {"oneOf": [{"enum": [1]}, {"type": "integer"}]},
            "1.0",
            "1.0",
            id="equivalent-integral-number-oneof-overlap",
        ),
        pytest.param(
            {"anyOf": [{"const": 1.5}, {"type": "null"}]},
            "1.5e0",
            1.5,
            id="fractional-const-exponent",
        ),
        pytest.param(
            {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "Infinity",
            "Infinity",
            id="nonfinite-number",
        ),
        pytest.param({"oneOf": [{"const": False}, {"type": "null"}]}, "false", False, id="boolean-const"),
        pytest.param({"anyOf": [{"const": None}, {"type": "integer"}]}, "null", None, id="null-const"),
        pytest.param({"anyOf": [{"enum": [1, True, None, "x"]}]}, "true", True, id="heterogeneous-enum"),
        pytest.param({"anyOf": [{"enum": [1, 2]}, {"type": "null"}]}, "3", "3", id="nonmember-integer"),
        pytest.param({"anyOf": [{"const": 7}, {"type": "null"}]}, "8", "8", id="nonmember-const"),
        pytest.param({"anyOf": [{"enum": [1]}, {"type": "null"}]}, "true", "true", id="bool-is-not-integer"),
        pytest.param({"anyOf": [{"enum": [True]}, {"type": "null"}]}, "1", "1", id="integer-is-not-bool"),
        pytest.param({"anyOf": [{"enum": [0]}, {"type": "null"}]}, "false", "false", id="false-is-not-zero"),
        pytest.param(
            {"anyOf": [{"enum": [{"x": [1]}]}, {"type": "null"}]},
            '{"x":[true]}',
            '{"x":[true]}',
            id="nested-bool-is-not-integer",
        ),
        pytest.param({"anyOf": [{"enum": [[True]]}, {"type": "null"}]}, "[1]", "[1]", id="nested-integer-is-not-bool"),
        pytest.param({"anyOf": [{"enum": ["123"]}, {"type": "boolean"}]}, "123", "123", id="string-enum-not-number"),
        pytest.param({"anyOf": [{"enum": [1, None]}], "enum": [1]}, "null", "null", id="containing-enum-constraint"),
        pytest.param(
            {"anyOf": [{"type": "integer", "enum": [1]}, {"type": "string"}]},
            "2",
            "2",
            id="typed-branch-enum-constraint",
        ),
        pytest.param(
            {"anyOf": [{"anyOf": [{"const": {"x": [1, True]}}]}, {"type": "null"}]},
            '{"x":[1,true]}',
            {"x": [1, True]},
            id="nested-implicit-union",
        ),
        pytest.param(
            {"oneOf": [{"enum": [1]}, {"const": 1}, {"type": "string"}]},
            "1",
            "1",
            id="oneof-ambiguous-value",
        ),
        pytest.param(
            {"oneOf": [{"enum": [{"x": 1}]}, {"type": "object", "required": ["y"]}]},
            '{"x":1}',
            {"x": 1},
            id="object-enum-versus-required-key",
        ),
        pytest.param(
            {"oneOf": [{"enum": [{"kind": "a"}]}, {"type": "object", "properties": {"kind": {"const": "b"}}}]},
            '{"kind":"a"}',
            {"kind": "a"},
            id="object-enum-versus-discriminator",
        ),
        pytest.param(
            {"oneOf": [{"const": 1}, {"type": "integer", "not": {"const": 1}}]},
            "1",
            1,
            id="not-distinguishes-branches",
        ),
        pytest.param(
            {"oneOf": [{"enum": [1]}, {"type": "integer", "allOf": [{"minimum": 2}]}]},
            "1",
            1,
            id="allof-distinguishes-branches",
        ),
        pytest.param(
            {"oneOf": [{"type": "object", "required": ["a"]}, {"type": "object", "required": ["b"]}]},
            '{"a":1}',
            {"a": 1},
            id="disjoint-required-object-keys",
        ),
        pytest.param(
            {"oneOf": [{"type": "object", "required": ["a"]}, {"type": "object", "required": ["b"]}]},
            '{"a":1,"b":2}',
            '{"a":1,"b":2}',
            id="overlapping-required-object-keys",
        ),
        pytest.param(
            {"oneOf": [{"type": "array", "maxItems": 1}, {"type": "array", "minItems": 2}]},
            "[1]",
            [1],
            id="disjoint-array-lengths",
        ),
        pytest.param(
            {"oneOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "array", "items": {"type": "string"}}]},
            "[1]",
            [1],
            id="array-item-constraints",
        ),
        pytest.param(
            {
                "oneOf": [
                    {"type": "object", "additionalProperties": {"type": "integer"}},
                    {"type": "object", "additionalProperties": {"type": "string"}},
                ]
            },
            '{"a":1}',
            {"a": 1},
            id="additional-property-constraints",
        ),
        pytest.param(
            {
                "oneOf": [
                    {"type": "object", "dependentRequired": {"a": ["b"]}},
                    {"type": "object", "required": ["a"], "not": {"required": ["b"]}},
                ]
            },
            '{"a":1}',
            {"a": 1},
            id="dependent-required-constraints",
        ),
        pytest.param(
            {"oneOf": [{"type": "array", "contains": {"const": 1}}, {"type": "array", "contains": {"const": 2}}]},
            "[1]",
            [1],
            id="array-contains-constraints",
        ),
        pytest.param(
            {"oneOf": [{"type": "integer", "if": {"minimum": 0}, "then": {"minimum": 2}, "else": False}, {"const": 1}]},
            "1",
            1,
            id="conditional-constraints",
        ),
        pytest.param(
            {"$defs": {"two": {"const": 2}}, "oneOf": [{"type": "integer", "$ref": "#/$defs/two"}, {"const": 1}]},
            "1",
            1,
            id="local-reference-constraints",
        ),
    ],
)
def test_implicit_union_schema_preserves_declared_values(parser_name, streaming, schema, value, expected):
    """Both extraction APIs must preserve enum/const values without explicit type fields."""
    parser = ToolParserManager.get_tool_parser(parser_name)(_XMLTokenizer())
    request = _request([("f", {"type": "object", "properties": {"x": schema}})])
    tag = "seed:tool_call" if parser_name == "seed_oss" else "tool_call"
    prefix = "<seed:think>x</seed:think>" if parser_name == "seed_oss" else ""
    output = f"{prefix}<{tag}>\n<function=f>\n<parameter=x>{value}</parameter>\n</function>\n</{tag}>"
    if streaming:
        names = []
        arguments = ""
        previous = ""
        for part in re.split(r"(</?(?:seed:)?(?:think|tool_call)>)", output):
            pieces = [part] if re.fullmatch(r"</?(?:seed:)?(?:think|tool_call)>", part) else list(part)
            for piece in pieces:
                delta = parser.extract_tool_calls_streaming(previous, previous + piece, piece, [], [], [], request)
                previous += piece
                for tool in (delta.tool_calls if delta else None) or []:
                    assert tool.index == 0
                    if tool.function:
                        if tool.function.name:
                            names.append(tool.function.name)
                        arguments += tool.function.arguments or ""
        assert names == ["f"]
    else:
        result = parser.extract_tool_calls(output, request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "f"
        arguments = result.tool_calls[0].function.arguments
    parsed = json.loads(arguments)
    assert parsed == {"x": expected}
    assert type(parsed["x"]) is type(expected)
