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

"""Tool registry and built-in tools for agentic training.

This module provides a tool abstraction and registry for agentic MoshPit
training. Tools are callable objects that the agent can invoke during
multi-turn interactions to gather information or perform computations.

Each tool defines a ``call`` method with proper type annotations and
docstrings. The tool schema (OpenAI function-calling format) is
**automatically extracted** from the ``call`` method's signature and
docstring via ``function_to_json`` — no manual schema dicts needed.

Built-in tools:
    - ``python_code``: Sandboxed Python code execution with timeout.
    - ``calculator``: Safe arithmetic expression evaluation.
    - ``bash``: Shell command execution with timeout and output capture.
    - ``regex``: Pattern matching and extraction on text.
    - ``json_processor``: JSON parsing, querying, and transformation.
    - ``wikipedia``: Wikipedia article lookup and summarization.
    - ``web_fetch``: HTTP GET requests with content extraction.
    - ``file_read``: Read file contents from allowed paths.
    - ``unit_converter``: Physical unit and currency conversions.
    - ``notepad``: Persistent scratch-pad for multi-turn reasoning.

Custom tools can be registered using the ``register_tool`` decorator
or by subclassing ``Tool``.

Example:
    >>> @register_tool("greet")
    ... class GreetTool(Tool):
    ...     def call(self, name: str, greeting: str = "Hello") -> str:
    ...         '''Greet someone by name.
    ...         name: The person's name.
    ...         greeting: Greeting word to use.
    ...         '''
    ...         return f"{greeting}, {name}!"
    >>> tool = GreetTool()
    >>> tool.chat_schema
    {'name': 'greet', 'description': 'Greet someone by name.', ...}
"""

from __future__ import annotations

import abc
import inspect
import json
import re
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


def function_to_json(func: tp.Callable) -> dict[str, tp.Any]:
    """Convert a Python function into an OpenAI function-calling schema.

    Extracts the function's name, description (from docstring), and
    parameters with their types and descriptions from the signature
    and docstring.

    If the function has a ``__tool_schema__`` attribute it is used
    directly, which is useful for dynamically created tool wrappers.

    Parameter descriptions are parsed from the docstring using the
    pattern ``param_name: description`` (one per line).

    Args:
        func: The function to convert.

    Returns:
        An OpenAI function-calling schema dict::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            }

    Raises:
        ValueError: If the function signature cannot be extracted.
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
            origin = get_origin(annotation)
            args = get_args(annotation)

            if origin is Union:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1 and type(None) in args:
                    param_info["type"] = _TYPE_MAP.get(non_none[0], "string")
                else:
                    param_info["type"] = _TYPE_MAP.get(non_none[0], "string") if non_none else "string"
            elif origin in (list, tuple, set):
                param_info["type"] = "array"
                if args:
                    param_info["items"] = {"type": _TYPE_MAP.get(args[0], "string")}
            elif annotation in _TYPE_MAP:
                param_info["type"] = _TYPE_MAP[annotation]
            else:
                param_info["type"] = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)

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
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def register_tool(name: str) -> tp.Callable[[type[Tool]], type[Tool]]:
    """Register a tool class in the global registry.

    Args:
        name: Unique name for the tool (used in tool call parsing).

    Returns:
        Decorator that registers the tool class.

    Example:
        >>> @register_tool("greet")
        ... class GreetTool(Tool):
        ...     def call(self, name: str) -> str:
        ...         '''Greet someone.
        ...         name: The person's name.
        ...         '''
        ...         return f"Hello, {name}!"
    """

    def decorator(cls: type[Tool]) -> type[Tool]:
        """Register ``cls`` under the captured name and return it unchanged.

        Args:
            cls: The :class:`Tool` subclass being registered.

        Returns:
            ``cls``, returned so the decorator chain stays composable.
        """
        _TOOL_REGISTRY[name] = cls
        return cls

    return decorator


def make_tool(name: str, **kwargs: tp.Any) -> Tool:
    """Create a tool instance by registered name.

    Args:
        name: Registered tool name.
        **kwargs: Arguments passed to the tool constructor.

    Returns:
        Tool instance.

    Raises:
        KeyError: If no tool is registered with the given name.
    """
    if name not in _TOOL_REGISTRY:
        available = ", ".join(sorted(_TOOL_REGISTRY.keys()))
        raise KeyError(f"Tool '{name}' not found. Available tools: {available}")
    return _TOOL_REGISTRY[name](**kwargs)


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return sorted(_TOOL_REGISTRY.keys())


class Tool(abc.ABC):
    """Abstract base class for tools used in agentic training.

    Subclass this and implement ``call`` with proper type annotations
    and a docstring. The schema is automatically extracted — no manual
    schema dicts needed.

    The ``call`` method IS the tool's function. Its:
      - **Name** becomes ``tool.name`` (derived from the registered name).
      - **Docstring first line** becomes the tool description.
      - **Parameter annotations** become the JSON schema types.
      - **``param: description``** lines in the docstring become
        parameter descriptions.

    ``execute(arguments)`` parses the JSON arguments string and
    delegates to ``call(**parsed_args)``.

    ``schema()`` returns the full OpenAI function-calling dict,
    auto-generated from ``call``'s signature.

    ``chat_schema`` returns the bare function dict that
    ``tokenizer.apply_chat_template(messages, tools=...)`` expects.

    Example:
        >>> @register_tool("add")
        ... class AddTool(Tool):
        ...     def call(self, a: int, b: int) -> str:
        ...         '''Add two numbers.
        ...         a: First number.
        ...         b: Second number.
        ...         '''
        ...         return str(a + b)
        >>> tool = AddTool()
        >>> tool.execute('{"a": 2, "b": 3}')
        '5'
        >>> tool.chat_schema["name"]
        'add'
    """

    @abc.abstractmethod
    def call(self, **kwargs: tp.Any) -> str:
        """The tool's callable entry point.

        Subclasses must override this with a concrete signature
        (typed parameters, docstring with param descriptions).
        The base class uses this method's signature to auto-generate
        the tool schema and to dispatch ``execute()`` calls.
        """

    @property
    def name(self) -> str:
        """Tool name, derived from the registry or class name.

        Override this property to provide a custom name.
        """
        for reg_name, cls in _TOOL_REGISTRY.items():
            if cls is type(self):
                return reg_name
        cls_name = type(self).__name__
        return cls_name.lower().removesuffix("tool") or cls_name.lower()

    def execute(self, arguments: str) -> str:
        """Parse JSON arguments and delegate to ``call``.

        Args:
            arguments: JSON string of tool arguments from the model.

        Returns:
            Text result from ``call``.
        """
        try:
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                parsed = {next(iter(inspect.signature(self.call).parameters)): parsed}
        except (json.JSONDecodeError, StopIteration):
            first_param = next(iter(inspect.signature(self.call).parameters), None)
            parsed = {first_param: arguments} if first_param else {}
        return self.call(**parsed)

    def schema(self) -> dict[str, tp.Any]:
        """Return the full OpenAI function-calling schema, auto-extracted from ``call``.

        Returns:
            Dict with ``"type"`` and ``"function"`` keys::

                {
                    "type": "function",
                    "function": {
                        "name": "...",
                        "description": "...",
                        "parameters": { ... }
                    }
                }
        """
        result = function_to_json(self.call)
        result["function"]["name"] = self.name
        return result

    @property
    def chat_schema(self) -> dict[str, tp.Any]:
        """Return the bare function dict for ``apply_chat_template``.

        HuggingFace tokenizers expect tools as a list of bare function
        dicts (without the ``"type": "function"`` wrapper).

        This is what gets passed to
        ``tokenizer.apply_chat_template(messages, tools=[tool.chat_schema, ...])``.
        """
        return self.schema()["function"]


@register_tool("python_code")
class PythonCodeTool(Tool):
    """Sandboxed Python code execution.

    Runs Python code and captures stdout/stderr. The code has access
    to the full standard library (math, json, re, collections, etc.).
    Uses ``SIGALRM`` for timeout on POSIX systems.

    Args:
        timeout: Maximum execution time in seconds.
        max_output_length: Maximum length of captured output.
    """

    def __init__(self, timeout: float = 10.0, max_output_length: int = 4096):
        """Configure the Python tool's execution sandbox.

        Args:
            timeout: Maximum wall-clock time per ``exec`` call (seconds).
            max_output_length: Maximum number of characters returned from
                stdout/stderr capture; longer outputs are truncated.
        """
        self._timeout = timeout
        self._max_output_length = max_output_length

    def call(self, code: str) -> str:
        """Execute a Python code snippet and return everything printed to stdout/stderr. Use print() to produce output. You have access to the full Python standard library including math, json, re, itertools, collections, fractions, decimal, statistics, etc. For example: 'import math; print(math.factorial(10))' or 'print(sum(range(1, 101)))'.

        code: Complete Python code to execute. Must use print() to produce visible output.
        """
        import contextlib
        import io
        import signal

        output = io.StringIO()
        try:

            def _timeout_handler(signum, frame):
                """Raise ``TimeoutError`` from a ``SIGALRM`` interrupt.

                Args:
                    signum: Signal number (ignored).
                    frame: Current stack frame (ignored).

                Raises:
                    TimeoutError: Always.
                """
                raise TimeoutError(f"Code execution timed out after {self._timeout}s")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(self._timeout))
            try:
                restricted_globals: dict[str, tp.Any] = {"__builtins__": __builtins__}
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(code, restricted_globals)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception as e:
            output.write(f"\nError: {type(e).__name__}: {e}")

        result = output.getvalue()
        if len(result) > self._max_output_length:
            result = result[: self._max_output_length] + "\n... (output truncated)"
        return result or "(no output)"


@register_tool("calculator")
class CalculatorTool(Tool):
    """Mathematical expression evaluator with full math support.

    Evaluates expressions using Python's ``math`` module. Supports
    arithmetic, trigonometry, logarithms, factorials, constants
    (pi, e, tau, inf), and more.
    """

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

        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"


@register_tool("bash")
class BashTool(Tool):
    """Shell command execution.

    Runs a bash command and returns the combined stdout and stderr.
    Supports pipes, redirects, and all standard shell features.

    Args:
        timeout: Maximum execution time in seconds.
        max_output_length: Maximum length of captured output.
        shell: Shell binary to use.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_output_length: int = 8192,
        shell: str = "/bin/bash",
    ):
        """Configure the bash tool's subprocess sandbox.

        Args:
            timeout: Maximum wall-clock time per command (seconds).
            max_output_length: Maximum bytes of combined stdout/stderr
                to return; longer outputs are truncated.
            shell: Path to the shell binary used to interpret commands.
        """
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._shell = shell

    def call(self, command: str) -> str:
        """Run a bash shell command and return its stdout and stderr output. Supports pipes, redirects, environment variables, and all standard Unix utilities (grep, awk, sed, curl, find, etc.).

        command: The shell command to execute, e.g. 'ls -la /tmp', 'echo hello | wc -c', 'curl -s https://example.com'.
        """
        import subprocess

        try:
            result = subprocess.run(
                [self._shell, "-c", command],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
        except subprocess.TimeoutExpired:
            output = f"Error: command timed out after {self._timeout}s"
        except Exception as e:
            output = f"Error: {type(e).__name__}: {e}"

        if len(output) > self._max_output_length:
            output = output[: self._max_output_length] + "\n... (output truncated)"
        return output or "(no output)"


@register_tool("regex")
class RegexTool(Tool):
    """Regular expression search and extraction."""

    def call(self, pattern: str, text: str) -> str:
        """Search text with a Python regular expression and return all matches as a JSON array. Uses re.findall(), so if the pattern contains capture groups, only the captured portions are returned. Supports the full Python regex syntax including lookaheads, named groups, etc.

        pattern: Python regular expression pattern, e.g. r'\\d+' for numbers, r'(\\w+)@(\\w+\\.\\w+)' for emails, r'(?<=price: )\\$[\\d.]+' for prices after 'price: '.
        text: The text to search through.
        """
        try:
            matches = re.findall(pattern, text)
            if not matches:
                return "No matches found."
            return json.dumps(matches, ensure_ascii=False)
        except re.error as e:
            return f"Regex error: {e}"


@register_tool("json_processor")
class JSONProcessorTool(Tool):
    """JSON parsing, querying, and inspection."""

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
        """Walk a parsed JSON value with a dotted path and return the result.

        Args:
            data: Parsed JSON value (dict / list / primitive).
            path: Dot-separated key path; numeric segments index into
                lists.

        Returns:
            The selected sub-tree pretty-printed as JSON, the raw string
            for string leaves, or an error message when the path cannot
            be resolved.
        """
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
    """Wikipedia article lookup.

    Args:
        language: Wikipedia language code (default ``"en"``).
        max_length: Maximum character length of the returned extract.
    """

    def __init__(self, language: str = "en", max_length: int = 4096):
        """Configure the Wikipedia summary tool.

        Args:
            language: ISO language code used in the API host
                (e.g. ``"en"`` for English).
            max_length: Maximum number of characters returned from the
                article extract; longer extracts are truncated.
        """
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
    """Fetch web page content via HTTP GET.

    Args:
        timeout: HTTP request timeout in seconds.
        max_length: Maximum response body length.
    """

    def __init__(self, timeout: float = 15.0, max_length: int = 8192):
        """Configure the HTTP fetch tool.

        Args:
            timeout: ``urlopen`` timeout in seconds.
            max_length: Maximum number of characters of response body
                returned; longer bodies are truncated.
        """
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
    """Read local file contents.

    Args:
        allowed_dirs: Restrict reads to these directory prefixes.
            Empty list allows all paths.
        max_length: Maximum characters to read from the file.
    """

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        max_length: int = 16384,
    ):
        """Configure the file-read sandbox.

        Args:
            allowed_dirs: Optional list of absolute directories that the
                tool is allowed to read from.  An empty list (the
                default) disables the directory check entirely.
            max_length: Maximum number of characters returned from a
                single file; longer files are truncated.
        """
        self._allowed_dirs = allowed_dirs or []
        self._max_length = max_length

    def call(self, path: str) -> str:
        """Read and return the contents of a local file. Supports any text file format. Tilde (~) is expanded to the home directory.

        path: Absolute or relative path to the file, e.g. '/tmp/data.csv', '~/notes.txt', 'src/main.py'.
        """
        import os

        path = os.path.expanduser(path)

        if self._allowed_dirs:
            abs_path = os.path.abspath(path)
            if not any(abs_path.startswith(os.path.abspath(d)) for d in self._allowed_dirs):
                return f"Error: path '{path}' is outside allowed directories."

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
    """Physical unit converter.

    Converts between common units of length (km, mi, m, ft, cm, in),
    weight (kg, lb), temperature (c, f, k), and volume (l, gal).
    """

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
    """Persistent scratch-pad that survives across turns within an episode."""

    def __init__(self):
        """Initialize the notepad with an empty buffer."""
        self._content: str = ""

    def call(self, action: str, text: str = "") -> str:
        """A persistent notepad for storing intermediate results, plans, or reasoning across multiple turns. Use 'write' to replace all content, 'append' to add to existing content, 'read' to retrieve current content, or 'clear' to erase everything. The notepad persists for the entire episode.

        action: One of 'write' (replace content), 'append' (add to end), 'read' (retrieve content), or 'clear' (erase all).
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
    """Wrap a plain Python function as a Tool.

    The function's signature and docstring are used to auto-generate
    the tool schema. If the function lacks annotations, provide an
    explicit ``tool_schema`` override.

    Args:
        func: The function to wrap. Its signature is inspected for
            schema generation.
        tool_name: Name for the tool (overrides auto-detection).
        tool_schema: Optional explicit schema dict (bypasses
            auto-extraction from ``func``).
    """

    func: tp.Callable[..., str] = field(repr=False)
    tool_name: str = "custom"
    tool_schema: dict[str, tp.Any] | None = field(default=None)

    @property
    def name(self) -> str:
        """Return the configured tool name (overrides registry detection)."""
        return self.tool_name

    def call(self, **kwargs: tp.Any) -> str:
        """Delegate to the wrapped function."""
        return self.func(**kwargs)

    def execute(self, arguments: str) -> str:
        """Parse JSON and call the wrapped function."""
        try:
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                parsed = {next(iter(inspect.signature(self.func).parameters)): parsed}
        except (json.JSONDecodeError, StopIteration):
            first_param = next(iter(inspect.signature(self.func).parameters), None)
            parsed = {first_param: arguments} if first_param else {}
        return self.func(**parsed)

    def schema(self) -> dict[str, tp.Any]:
        """Return schema from explicit override or auto-extract from ``func``."""
        if self.tool_schema is not None:
            return self.tool_schema
        result = function_to_json(self.func)
        result["function"]["name"] = self.name
        return result
