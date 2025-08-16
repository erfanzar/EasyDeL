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

"""Function calling support for inference engines.

This module provides function calling capabilities for LLMs, allowing them
to interact with external tools and APIs through a structured protocol.

Classes:
    Function: Represents a callable function with schema
    FunctionCall: Parsed function call from model output
    FunctionCallingManager: Manages function registry and execution
    FunctionCallingConfig: Configuration for function calling

Example:
    >>> from easydel.inference.function_calling import FunctionCallingManager, Function
    >>>
    >>> manager = FunctionCallingManager()
    >>>
    >>> # Register a function
    >>> @manager.register_function
    ... def get_weather(location: str, unit: str = "celsius"):
    ...     '''Get the current weather for a location.'''
    ...     return {"temp": 20, "condition": "sunny"}
    >>>
    >>> # Or register with schema
    >>> weather_func = Function(
    ...     name="get_weather",
    ...     description="Get weather information",
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {
    ...             "location": {"type": "string"},
    ...             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    ...         },
    ...         "required": ["location"]
    ...     },
    ...     implementation=get_weather_impl
    ... )
    >>> manager.register(weather_func)
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FunctionCallingMode(Enum):
    """Mode for function calling behavior."""

    AUTO = "auto"  # Model decides whether to call functions
    NONE = "none"  # No function calling
    REQUIRED = "required"  # Must call at least one function
    SPECIFIC = "specific"  # Must call a specific function


@dataclass
class Function:
    """Represents a callable function with JSON schema.

    Attributes:
        name: Function name
        description: Human-readable description
        parameters: JSON Schema for parameters
        implementation: Actual callable implementation
        examples: Optional usage examples
    """

    name: str
    description: str
    parameters: dict[str, Any]
    implementation: Callable | None = None
    examples: list[str] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def to_hermes_format(self) -> str:
        """Convert to Hermes-style format for Qwen models."""
        schema = self.to_json_schema()
        return json.dumps(schema, indent=2)


@dataclass
class FunctionCall:
    """Parsed function call from model output.

    Attributes:
        name: Function name to call
        arguments: Parsed arguments dictionary
        raw_arguments: Raw argument string from model
        id: Optional call ID for tracking
    """

    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    id: str | None = None

    @classmethod
    def from_model_output(cls, output: str) -> FunctionCall | None:
        """Parse function call from model output.

        Supports multiple formats:
        - JSON format: {"name": "func", "arguments": {...}}
        - XML format: <function>func</function><arguments>{...}</arguments>
        - Markdown format: ```function\nfunc({...})\n```
        """
        # Try JSON format first
        try:
            # Look for JSON-like function call
            json_match = re.search(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}', output, re.DOTALL)
            if json_match:
                call_data = json.loads(json_match.group())
                return cls(
                    name=call_data["name"],
                    arguments=call_data.get("arguments", {}),
                    raw_arguments=json.dumps(call_data.get("arguments", {})),
                    id=call_data.get("id"),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        # Try XML format
        xml_name_match = re.search(r"<function>([^<]+)</function>", output)
        xml_args_match = re.search(r"<arguments>([^<]+)</arguments>", output)
        if xml_name_match and xml_args_match:
            try:
                return cls(
                    name=xml_name_match.group(1).strip(),
                    arguments=json.loads(xml_args_match.group(1)),
                    raw_arguments=xml_args_match.group(1),
                )
            except json.JSONDecodeError:
                pass

        # Try markdown code block format
        md_match = re.search(r"```function\n([^(]+)\(([^)]*)\)\n```", output)
        if md_match:
            try:
                return cls(
                    name=md_match.group(1).strip(),
                    arguments=json.loads(md_match.group(2)) if md_match.group(2) else {},
                    raw_arguments=md_match.group(2) or "{}",
                )
            except json.JSONDecodeError:
                pass

        return None


@dataclass
class FunctionCallingConfig:
    """Configuration for function calling behavior.

    Attributes:
        mode: Function calling mode
        specific_function: Name of specific function if mode is SPECIFIC
        max_iterations: Maximum function call iterations
        include_thinking: Include reasoning in prompts
        parallel_calls: Allow parallel function execution
        error_handling: How to handle function errors
    """

    mode: FunctionCallingMode = FunctionCallingMode.AUTO
    specific_function: str | None = None
    max_iterations: int = 5
    include_thinking: bool = False
    parallel_calls: bool = False
    error_handling: str = "return_error"  # or "retry", "skip"


class FunctionCallingManager:
    """Manages function registry and execution for LLMs.

    Provides registration, discovery, and execution of functions
    that can be called by language models.

    Example:
        >>> manager = FunctionCallingManager()
        >>>
        >>> @manager.register_function
        ... def calculate(expression: str):
        ...     return eval(expression)
        >>>
        >>> # Generate prompt with functions
        >>> prompt = manager.create_prompt(
        ...     "What is 2+2?",
        ...     include_functions=True
        ... )
        >>>
        >>> # Parse and execute function calls
        >>> output = model.generate(prompt)
        >>> result = manager.execute_function_calls(output)
    """

    def __init__(self, config: FunctionCallingConfig | None = None):
        """Initialize function calling manager.

        Args:
            config: Function calling configuration
        """
        self.config = config or FunctionCallingConfig()
        self.functions: dict[str, Function] = {}
        self.execution_history: list[dict[str, Any]] = []

    def register(self, function: Function) -> None:
        """Register a function.

        Args:
            function: Function to register
        """
        self.functions[function.name] = function

    def register_function(self, func: Callable) -> Callable:
        """Decorator to register a Python function.

        Automatically extracts schema from function signature.

        Args:
            func: Function to register

        Returns:
            The original function
        """
        # Extract function metadata
        sig = inspect.signature(func)
        params_schema = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Infer type from annotation
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:  # noqa: E721
                    param_type = "integer"
                elif param.annotation == float:  # noqa: E721
                    param_type = "number"
                elif param.annotation == bool:  # noqa: E721
                    param_type = "boolean"
                elif param.annotation == list:  # noqa: E721
                    param_type = "array"
                elif param.annotation == dict:  # noqa: E721
                    param_type = "object"

            params_schema["properties"][param_name] = {"type": param_type, "description": f"Parameter {param_name}"}

            # Check if required
            if param.default == inspect.Parameter.empty:
                params_schema["required"].append(param_name)

        # Create Function object
        function = Function(
            name=func.__name__,
            description=func.__doc__ or f"Function {func.__name__}",
            parameters=params_schema,
            implementation=func,
        )

        self.register(function)
        return func

    def create_prompt(
        self,
        user_message: str,
        system_message: str | None = None,
        include_functions: bool = True,
        function_names: list[str] | None = None,
    ) -> str:
        """Create a prompt with function definitions.

        Args:
            user_message: User's input message
            system_message: Optional system message
            include_functions: Whether to include function definitions
            function_names: Specific functions to include (None for all)

        Returns:
            Formatted prompt with function definitions
        """
        prompt_parts = []

        # Add system message
        if system_message:
            prompt_parts.append(f"System: {system_message}")
        elif include_functions:
            prompt_parts.append(
                "You are a helpful assistant with access to functions. "
                "Use them when needed to answer questions accurately."
            )

        # Add function definitions
        if include_functions:
            functions_to_include = function_names or list(self.functions.keys())
            available_functions = []

            for func_name in functions_to_include:
                if func_name in self.functions:
                    func = self.functions[func_name]
                    available_functions.append(func.to_json_schema())

            if available_functions:
                prompt_parts.append("\nAvailable functions:")
                prompt_parts.append(json.dumps(available_functions, indent=2))
                prompt_parts.append('\nTo use a function, respond with:\n{"name": "function_name", "arguments": {...}}')

        # Add user message
        prompt_parts.append(f"\nUser: {user_message}")

        return "\n".join(prompt_parts)

    def parse_function_calls(self, model_output: str) -> list[FunctionCall]:
        """Parse function calls from model output.

        Args:
            model_output: Raw model output

        Returns:
            List of parsed function calls
        """
        calls = []

        # Try to find multiple function calls
        # Split by common delimiters
        potential_calls = re.split(r"(?<=\})\s*(?=\{)", model_output)

        for potential_call in potential_calls:
            call = FunctionCall.from_model_output(potential_call)
            if call and call.name in self.functions:
                calls.append(call)

        # If no calls found, try parsing the entire output
        if not calls:
            call = FunctionCall.from_model_output(model_output)
            if call and call.name in self.functions:
                calls.append(call)

        return calls

    def execute_function(self, call: FunctionCall) -> dict[str, Any]:
        """Execute a single function call.

        Args:
            call: Function call to execute

        Returns:
            Dictionary with result or error
        """
        result = {"name": call.name, "arguments": call.arguments, "status": "success", "result": None, "error": None}

        if call.name not in self.functions:
            result["status"] = "error"
            result["error"] = f"Function '{call.name}' not found"
            return result

        function = self.functions[call.name]

        if not function.implementation:
            result["status"] = "error"
            result["error"] = f"Function '{call.name}' has no implementation"
            return result

        try:
            # Execute the function
            func_result = function.implementation(**call.arguments)
            result["result"] = func_result
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

            if self.config.error_handling == "raise":
                raise

        # Record in history
        self.execution_history.append(result)

        return result

    def execute_function_calls(self, model_output: str, parallel: bool | None = None) -> list[dict[str, Any]]:
        """Parse and execute function calls from model output.

        Args:
            model_output: Model output potentially containing function calls
            parallel: Whether to execute calls in parallel

        Returns:
            List of execution results
        """
        calls = self.parse_function_calls(model_output)

        if not calls:
            return []

        results = []
        use_parallel = parallel if parallel is not None else self.config.parallel_calls

        if use_parallel and len(calls) > 1:
            # Execute in parallel (simplified - could use asyncio/threading)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.execute_function, call) for call in calls]
                results = [f.result() for f in futures]
        else:
            # Execute sequentially
            for call in calls:
                results.append(self.execute_function(call))

        return results

    def format_function_results(self, results: list[dict[str, Any]]) -> str:
        """Format function execution results for model consumption.

        Args:
            results: Function execution results

        Returns:
            Formatted string for model
        """
        if not results:
            return "No functions were called."

        formatted_parts = ["Function execution results:"]

        for result in results:
            if result["status"] == "success":
                formatted_parts.append(f"- {result['name']}: {json.dumps(result['result'])}")
            else:
                formatted_parts.append(f"- {result['name']}: Error - {result['error']}")

        return "\n".join(formatted_parts)
