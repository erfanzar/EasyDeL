# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-independent tool registry, schema extraction, and built-in tools.

The registry and built-ins live together so callers can use ``make_tool``
without importing any trainer algorithm. The legacy agentic MoshPit module
re-exports these same objects and therefore shares registrations and types.
"""

from __future__ import annotations

import abc
import copy
import inspect
import json
import re
import types
import typing as tp
from dataclasses import dataclass, field
from typing import Union, get_args, get_origin

_TOOL_REGISTRY: dict[str, type[Tool]] = {}

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
    tuple: "array",
    set: "array",
    bytes: "string",
}


def _annotation_to_schema(annotation: tp.Any) -> dict[str, tp.Any]:
    """Translate Python annotations into valid, recursively typed JSON Schema."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        return {"anyOf": [_annotation_to_schema(arg) for arg in args]}
    if origin is tp.Annotated:
        return _annotation_to_schema(args[0])
    if origin is tp.Literal:
        return {"enum": list(args)}
    if origin in (list, tuple, set):
        schema: dict[str, tp.Any] = {"type": "array"}
        if args:
            if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
                schema.update(
                    prefixItems=[_annotation_to_schema(arg) for arg in args], minItems=len(args), maxItems=len(args)
                )
            else:
                schema["items"] = _annotation_to_schema(args[0])
        return schema
    if origin is dict:
        return {"type": "object", "additionalProperties": _annotation_to_schema(args[1])} if args else {"type": "object"}
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}
    # Unknown classes and Any have no inferred JSON constraint. Never emit a
    # Python class name as a JSON Schema type.
    return {}


def function_to_json(func: tp.Callable) -> dict[str, tp.Any]:
    """Extract an OpenAI function schema from annotations and the docstring.

    Args:
        func: Callable to inspect. Its ``__tool_schema__`` override, if
            present, takes precedence over signature and docstring extraction.

    Returns:
        Function schema with name, description, properties and required keys.

    Raises:
        ValueError: If the signature cannot be extracted.
    """
    if hasattr(func, "__tool_schema__"):
        schema = func.__tool_schema__
        return {
            "type": "function",
            "function": {
                "name": schema.get("name", getattr(func, "__name__", "unknown")),
                "description": schema.get("description", func.__doc__ or ""),
                "parameters": schema.get(
                    "parameters",
                    {"type": "object", "properties": {}, "required": []},
                ),
            },
        }

    try:
        signature = inspect.signature(func)
        type_hints = tp.get_type_hints(func)
    except (ValueError, TypeError) as e:
        fn_name = getattr(func, "__name__", repr(func))
        raise ValueError(f"Failed to get signature for {fn_name}: {e}") from e

    docstring = func.__doc__ or ""
    first_line = docstring.strip().split("\n")[0] if docstring.strip() else ""
    param_descriptions: dict[str, str] = {}
    param_pattern = r"(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=\n\s*\w+(?:\s*\([^)]+\))?\s*:|$)"
    for param_name, description in re.findall(param_pattern, docstring, re.DOTALL | re.MULTILINE):
        param_descriptions[param_name.strip()] = description.strip()

    parameters: dict[str, dict[str, tp.Any]] = {}
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        param_info: dict[str, tp.Any] = {"type": "string"}
        annotation = type_hints.get(param.name, param.annotation)
        if annotation != inspect.Parameter.empty:
            param_info = _annotation_to_schema(annotation)
        if param.name in param_descriptions:
            param_info["description"] = param_descriptions[param.name]
        parameters[param.name] = param_info

    required = [
        p.name for p in signature.parameters.values() if p.name != "self" and p.default == inspect.Parameter.empty
    ]
    fn_name = getattr(func, "__name__", "unknown")
    if fn_name == "call":
        fn_name = getattr(getattr(func, "__self__", None), "name", fn_name)
        if fn_name == "call":
            owner = getattr(func, "__qualname__", "").split(".")[0]
            fn_name = owner.lower().removesuffix("tool") if owner else "unknown"
    return {
        "type": "function",
        "function": {
            "name": fn_name,
            "description": first_line,
            "parameters": {"type": "object", "properties": parameters, "required": required},
        },
    }


def register_tool(name: str) -> tp.Callable[[type[Tool]], type[Tool]]:
    """Register a tool class under a dispatch name and return it unchanged."""

    def decorator(cls: type[Tool]) -> type[Tool]:
        """Register the decorated class in the shared registry."""
        _TOOL_REGISTRY[name] = cls
        return cls

    return decorator


def make_tool(name: str, **kwargs: tp.Any) -> Tool:
    """Instantiate a registered tool, forwarding constructor kwargs.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in _TOOL_REGISTRY:
        available = ", ".join(sorted(_TOOL_REGISTRY.keys()))
        raise KeyError(f"Tool '{name}' not found. Available tools: {available}")
    return _TOOL_REGISTRY[name](**kwargs)


def list_tools() -> list[str]:
    """Return all registered names in sorted order."""
    return sorted(_TOOL_REGISTRY.keys())


class Tool(abc.ABC):
    """Base tool with JSON dispatch and annotation-driven function schemas."""

    @abc.abstractmethod
    def call(self, **kwargs: tp.Any) -> str:
        """Execute tool-specific arguments and return a text observation."""

    def clone_for_episode(self) -> Tool:
        """Create an episode-owned tool, preserving configuration, not live state.

        Override this for tools with non-copyable resources such as clients or
        locks. Stateful tools should also override ``reset_episode`` so reusing
        an environment starts a fresh episode without affecting other members.
        """
        tool = copy.deepcopy(self)
        tool.reset_episode()
        return tool

    def reset_episode(self) -> None:
        """Clear episode-local state; stateless tools need no reset."""
        return None

    @property
    def name(self) -> str:
        """Return the registered name or derive one from the class name."""
        for reg_name, cls in _TOOL_REGISTRY.items():
            if cls is type(self):
                return reg_name
        cls_name = type(self).__name__
        return cls_name.lower().removesuffix("tool") or cls_name.lower()

    def normalize_arguments(self, arguments: str) -> dict[str, tp.Any]:
        """Convert JSON objects, scalars, or legacy raw text to keyword arguments."""
        first_param = next(iter(inspect.signature(self.call).parameters), None)
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = arguments
        if isinstance(parsed, dict):
            return parsed
        return {first_param: parsed} if first_param else {}

    def execute(self, arguments: str) -> str:
        """Parse JSON arguments, with raw-string fallback, and invoke ``call``."""
        return self.call(**self.normalize_arguments(arguments))

    def schema(self) -> dict[str, tp.Any]:
        """Return the full OpenAI schema using this tool's dispatch name."""
        result = function_to_json(self.call)
        result["function"]["name"] = self.name
        return result

    @property
    def chat_schema(self) -> dict[str, tp.Any]:
        """Return the bare function schema accepted by HF chat templates."""
        return self.schema()["function"]


@register_tool("python_code")
class PythonCodeTool(Tool):
    """Execute Python in a resource-limited, timed, out-of-process sandbox."""

    def __init__(self, timeout: float = 10.0, max_output_length: int = 4096):
        """Configure the wall-clock timeout and captured output length."""
        self._timeout = timeout
        self._max_output_length = max_output_length

    def call(self, code: str) -> str:
        """Execute a Python code snippet and return everything printed to stdout/stderr. Use print() to produce output. You have access to the full Python standard library including math, json, re, itertools, collections, fractions, decimal, statistics, etc. For example: 'import math; print(math.factorial(10))' or 'print(sum(range(1, 101)))'.

        code: Complete Python code to execute. Must use print() to produce visible output.
        """
        from .sandbox import Sandbox

        res = Sandbox(timeout=self._timeout, max_output=self._max_output_length).run_python(code)
        output = res.stdout
        if res.stderr:
            output += ("\n" if output else "") + res.stderr
        if res.timed_out:
            output += f"\nError: code execution timed out after {self._timeout}s"
        if len(output) > self._max_output_length:
            output = output[: self._max_output_length] + "\n... (output truncated)"
        return output or "(no output)"


@register_tool("calculator")
class CalculatorTool(Tool):
    """Evaluate arithmetic and math functions inside a bounded child process."""

    def __init__(self, timeout: float = 2.0, max_output_length: int = 4096):
        """Configure resource limits without restricting supported math functions."""
        self._timeout = timeout
        self._max_output_length = max_output_length

    def call(self, expression: str) -> str:
        """Evaluate a mathematical expression and return the numeric result. Supports all standard arithmetic (+, -, *, /, //, %, **), comparisons, and math functions: abs, round, min, max, pow, sqrt, sin, cos, tan, asin, acos, atan, atan2, log, log2, log10, exp, ceil, floor, factorial, gcd, lcm, comb, perm, degrees, radians, hypot, isqrt. Constants: pi, e, tau, inf, nan.

        expression: Mathematical expression to evaluate, e.g. 'sqrt(2) * pi', 'factorial(10)', 'log2(1024)', '3**4 + sin(0.5)'.
        """
        import math

        allowed_names: dict[str, tp.Any] = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sum": sum,
            "len": len,
            "range": range,
            "int": int,
            "float": float,
            "bool": bool,
            "True": True,
            "False": False,
        }
        for name in (
            "sqrt",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "log",
            "log2",
            "log10",
            "exp",
            "ceil",
            "floor",
            "factorial",
            "gcd",
            "lcm",
            "comb",
            "perm",
            "degrees",
            "radians",
            "hypot",
            "isqrt",
            "pi",
            "e",
            "tau",
            "inf",
            "nan",
        ):
            if hasattr(math, name):
                allowed_names[name] = getattr(math, name)
        from .sandbox import Sandbox

        code = (
            "import ast, builtins, math\n"
            f"expression = {expression!r}\n"
            "tree = ast.parse(expression, mode='eval')\n"
            "if any(isinstance(node, ast.Attribute) for node in ast.walk(tree)):\n"
            "    raise ValueError('Attribute access is not allowed')\n"
            f"names = {list(allowed_names)!r}\n"
            "allowed = {name: getattr(math, name, getattr(builtins, name, None)) for name in names}\n"
            "print(eval(compile(tree, '<calculator>', 'eval'), {'__builtins__': {}}, allowed))\n"
        )
        result = Sandbox(timeout=self._timeout, memory_mb=256, max_output=self._max_output_length).run_python(code)
        if result.timed_out:
            return f"Error: calculation timed out after {self._timeout}s"
        if not result.ok:
            return f"Error: {result.stderr.strip() or f'calculator exited with code {result.returncode}'}"
        return result.stdout.strip()


@register_tool("bash")
class BashTool(Tool):
    """Run shell commands, including pipes and redirects, in a timed sandbox."""

    def __init__(self, timeout: float = 30.0, max_output_length: int = 8192, shell: str = "/bin/bash"):
        """Configure wall-clock timeout, captured output length, and shell."""
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._shell = shell

    def call(self, command: str) -> str:
        """Run a bash shell command and return its stdout and stderr output. Supports pipes, redirects, environment variables, and all standard Unix utilities (grep, awk, sed, curl, find, etc.).

        command: The shell command to execute, e.g. 'ls -la /tmp', 'echo hello | wc -c', 'curl -s https://example.com'.
        """
        from .sandbox import Sandbox

        res = Sandbox(timeout=self._timeout, max_output=self._max_output_length).run_bash(command, shell=self._shell)
        if res.timed_out:
            output = f"Error: command timed out after {self._timeout}s"
        else:
            output = res.stdout
            if res.stderr:
                output += f"\n[stderr]\n{res.stderr}"
            if res.returncode not in (0, None):
                output += f"\n[exit code: {res.returncode}]"
        if len(output) > self._max_output_length:
            output = output[: self._max_output_length] + "\n... (output truncated)"
        return output or "(no output)"


@register_tool("regex")
class RegexTool(Tool):
    """Extract full Python regular-expression matches in a bounded child process."""

    def __init__(self, timeout: float = 2.0, max_output_length: int = 4096):
        """Configure the matching deadline and maximum returned output length."""
        self._timeout = timeout
        self._max_output_length = max_output_length

    def call(self, pattern: str, text: str) -> str:
        """Search text with a Python regular expression and return all matches as a JSON array. Uses re.findall(), so if the pattern contains capture groups, only the captured portions are returned. Supports the full Python regex syntax including lookaheads, named groups, etc.

        pattern: Python regular expression pattern, e.g. r'\\d+' for numbers, r'(\\w+)@(\\w+\\.\\w+)' for emails, r'(?<=price: )\\$[\\d.]+' for prices after 'price: '.
        text: The text to search through.
        """
        from .sandbox import Sandbox

        code = (
            "import json, re, sys\n"
            "pattern, text = json.load(sys.stdin)\n"
            "try:\n"
            "    matches = re.findall(pattern, text)\n"
            "    print(json.dumps(matches, ensure_ascii=False) if matches else 'No matches found.')\n"
            "except re.error as exc:\n"
            "    print(f'Regex error: {exc}')\n"
        )
        result = Sandbox(timeout=self._timeout, memory_mb=256, max_output=self._max_output_length + 1).run_python(
            code, stdin=json.dumps([pattern, text])
        )
        if result.timed_out:
            return f"Regex error: matching timed out after {self._timeout}s"
        if not result.ok:
            return f"Regex error: {result.stderr.strip() or f'matcher exited with code {result.returncode}'}"
        output = result.stdout.strip()
        if len(output) > self._max_output_length:
            return output[: self._max_output_length] + "\n... (output truncated)"
        return output


@register_tool("json_processor")
class JSONProcessorTool(Tool):
    """Parse JSON, list object keys, and query nested values by dotted paths."""

    def call(self, operation: str, data: str, path: str = "") -> str:
        """Process JSON data. Three operations are available: 'parse' validates and pretty-prints the JSON; 'query' extracts a nested value by dot-separated key path (use numeric indices for arrays, e.g. 'results.0.name'); 'keys' lists all top-level keys of a JSON object.

        operation: One of 'parse', 'query', or 'keys'.
        data: The JSON string to process.
        path: Dot-separated key path for the 'query' operation, e.g. 'user.address.city' or 'items.2.price'. Not needed for 'parse' or 'keys'.
        """
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError as e:
            return f"JSON parse error: {e}"
        if operation == "parse":
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        elif operation == "keys":
            if isinstance(parsed, dict):
                return json.dumps(list(parsed.keys()))
            return "Error: data is not a JSON object."
        elif operation == "query":
            return self._query(parsed, path)
        else:
            return f"Unknown operation: {operation}. Use 'parse', 'query', or 'keys'."

    @staticmethod
    def _query(data: tp.Any, path: str) -> str:
        """Walk object keys/list indices and return the selected JSON subtree."""
        keys = path.split(".") if path else []
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                try:
                    current = current[int(key)]
                except (ValueError, IndexError):
                    return f"Error: invalid index '{key}' for list of length {len(current)}."
            else:
                return f"Error: key '{key}' not found."
        return json.dumps(current, indent=2, ensure_ascii=False) if not isinstance(current, str) else current


@register_tool("wikipedia")
class WikipediaTool(Tool):
    """Look up Wikipedia summaries in a configurable language."""

    def __init__(self, language: str = "en", max_length: int = 4096):
        """Configure the API language and response character limit."""
        self._language = language
        self._max_length = max_length

    def call(self, query: str) -> str:
        """Look up a topic on Wikipedia and return a summary of the article. Returns the article title and the first few paragraphs (the extract). Useful for factual questions about people, places, events, science, history, etc.

        query: The topic to look up, e.g. 'Albert Einstein', 'Python programming language', 'Pythagorean theorem'.
        """
        import urllib.parse
        import urllib.request

        url = f"https://{self._language}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EasyDeL-AgenticTool/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            title = data.get("title", query)
            extract = data.get("extract", "No summary available.")
            result = f"# {title}\n\n{extract}"
            if len(result) > self._max_length:
                result = result[: self._max_length] + "\n... (truncated)"
            return result
        except Exception as e:
            return f"Wikipedia lookup failed: {type(e).__name__}: {e}"


@register_tool("web_fetch")
class WebFetchTool(Tool):
    """Fetch HTTP(S) content and strip scripts, styles, and HTML markup."""

    def __init__(self, timeout: float = 15.0, max_length: int = 8192):
        """Configure the HTTP timeout and response character limit."""
        self._timeout = timeout
        self._max_length = max_length

    def call(self, url: str) -> str:
        """Fetch a URL via HTTP GET and return the page content as plain text with HTML tags stripped. Useful for reading web pages, API endpoints that return HTML, documentation pages, etc.

        url: Full URL to fetch, must start with http:// or https://, e.g. 'https://example.com/page'.
        """
        import urllib.request

        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EasyDeL-AgenticTool/1.0"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > self._max_length:
                content = content[: self._max_length] + "\n... (truncated)"
            return content or "(empty page)"
        except Exception as e:
            return f"Fetch failed: {type(e).__name__}: {e}"


@register_tool("file_read")
class FileReadTool(Tool):
    """Read local text with optional realpath-based directory restrictions."""

    def __init__(self, allowed_dirs: list[str] | None = None, max_length: int = 16384):
        """Configure allowed directories (empty permits all) and read limit."""
        self._allowed_dirs = allowed_dirs or []
        self._max_length = max_length

    def call(self, path: str) -> str:
        """Read and return the contents of a local file. Supports any text file format. Tilde (~) is expanded to the home directory.

        path: Absolute or relative path to the file, e.g. '/tmp/data.csv', '~/notes.txt', 'src/main.py'.
        """
        import os

        path = os.path.expanduser(path)
        if self._allowed_dirs:
            resolved_path = os.path.realpath(path)
            allowed_roots = [os.path.realpath(os.path.expanduser(d)) for d in self._allowed_dirs]
            if not any(os.path.commonpath((resolved_path, root)) == root for root in allowed_roots):
                return f"Error: path '{path}' is outside allowed directories."
            path = resolved_path
        try:
            with open(path) as f:
                content = f.read(self._max_length)
            if len(content) == self._max_length:
                content += "\n... (truncated)"
            return content
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"


@register_tool("unit_converter")
class UnitConverterTool(Tool):
    """Convert length, weight, temperature, and volume units."""

    _CONVERSIONS: tp.ClassVar[dict[tuple[str, str], tp.Callable[[float], float]]] = {
        ("km", "mi"): lambda x: x * 0.621371,
        ("mi", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("m", "cm"): lambda x: x * 100,
        ("cm", "m"): lambda x: x / 100,
        ("m", "km"): lambda x: x / 1000,
        ("km", "m"): lambda x: x * 1000,
        ("ft", "in"): lambda x: x * 12,
        ("in", "ft"): lambda x: x / 12,
        ("mi", "ft"): lambda x: x * 5280,
        ("ft", "mi"): lambda x: x / 5280,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("kg", "g"): lambda x: x * 1000,
        ("g", "kg"): lambda x: x / 1000,
        ("lb", "oz"): lambda x: x * 16,
        ("oz", "lb"): lambda x: x / 16,
        ("c", "f"): lambda x: x * 9 / 5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5 / 9,
        ("c", "k"): lambda x: x + 273.15,
        ("k", "c"): lambda x: x - 273.15,
        ("f", "k"): lambda x: (x - 32) * 5 / 9 + 273.15,
        ("k", "f"): lambda x: (x - 273.15) * 9 / 5 + 32,
        ("l", "gal"): lambda x: x * 0.264172,
        ("gal", "l"): lambda x: x * 3.78541,
        ("l", "ml"): lambda x: x * 1000,
        ("ml", "l"): lambda x: x / 1000,
        ("cm", "in"): lambda x: x * 0.393701,
        ("in", "cm"): lambda x: x * 2.54,
    }

    def call(self, value: float, from_unit: str, to_unit: str) -> str:
        """Convert a numeric value from one unit to another. Supported units: length (km, mi, m, ft, cm, in), weight (kg, lb, g, oz), temperature (c, f, k), volume (l, gal, ml). Unit names are case-insensitive.

        value: The numeric value to convert.
        from_unit: The source unit, e.g. 'km', 'lb', 'c', 'gal'.
        to_unit: The target unit to convert to, e.g. 'mi', 'kg', 'f', 'l'.
        """
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        key = (from_unit, to_unit)
        if key in self._CONVERSIONS:
            result = self._CONVERSIONS[key](float(value))
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        if from_unit == to_unit:
            return f"{value} {from_unit} (no conversion needed)"
        supported = ", ".join(f"{a}->{b}" for a, b in self._CONVERSIONS)
        return f"Unsupported conversion: {from_unit} -> {to_unit}. Supported: {supported}"


@register_tool("notepad")
class NotepadTool(Tool):
    """Persistent per-instance scratchpad with write/append/read/clear actions."""

    def __init__(self):
        """Initialize an empty scratchpad."""
        self._content: str = ""

    def reset_episode(self) -> None:
        """Discard this scratchpad's notes at an episode boundary."""
        self._content = ""

    def call(self, action: str, text: str = "") -> str:
        """A persistent notepad for storing intermediate results, plans, or reasoning across multiple turns. Use 'write' to replace all content, 'append' to add to existing content, 'read' to retrieve current content, or 'clear' to erase everything. The notepad persists for the entire episode.

        action: One of 'write' (replace content), 'append' (add to end), 'read' (retrieve current content), or 'clear' (erase all).
        text: The text to write or append. Not needed for 'read' or 'clear'.
        """
        if action == "write":
            self._content = text
            return "Notes saved."
        elif action == "append":
            self._content += ("\n" if self._content else "") + text
            return "Notes appended."
        elif action == "read":
            return self._content if self._content else "(notepad is empty)"
        elif action == "clear":
            self._content = ""
            return "Notes cleared."
        else:
            return f"Unknown action: {action}. Use 'write', 'append', 'read', or 'clear'."


@dataclass
class FunctionTool(Tool):
    """Wrap a Python function with optional explicit name and schema overrides."""

    func: tp.Callable[..., str] = field(repr=False)
    tool_name: str = "custom"
    tool_schema: dict[str, tp.Any] | None = field(default=None)

    @property
    def name(self) -> str:
        """Return the configured dispatch name."""
        return self.tool_name

    def call(self, **kwargs: tp.Any) -> str:
        """Forward keyword arguments to the wrapped function."""
        return self.func(**kwargs)

    def normalize_arguments(self, arguments: str) -> dict[str, tp.Any]:
        """Normalize against the wrapped function, not ``call(**kwargs)``."""
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = arguments
        if isinstance(parsed, dict):
            return parsed
        first_param = next(iter(inspect.signature(self.func).parameters), None)
        return {first_param: parsed} if first_param else {}

    def execute(self, arguments: str) -> str:
        """Parse JSON with raw-string fallback and invoke the wrapped function."""
        return self.func(**self.normalize_arguments(arguments))

    def schema(self) -> dict[str, tp.Any]:
        """Return the explicit override or an automatically extracted schema."""
        if self.tool_schema is not None:
            return self.tool_schema
        result = function_to_json(self.func)
        result["function"]["name"] = self.name
        return result
