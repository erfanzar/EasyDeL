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

"""Independent function calling handler for any inference engine.

This module provides a standalone function calling handler that works
with any text generation engine (eSurge, vSurge, vInference, etc.).

Classes:
    FunctionCallHandler: Orchestrates function calling with any engine
    OpenAIFunctionFormatter: Formatter for OpenAI-style models

Example:
    >>> from easydel.inference.esurge import eSurge
    >>> from easydel.inference.function_calling_handler import FunctionCallHandler
    >>>
    >>> # Initialize any inference engine
    >>> engine = eSurge(model="Qwen/Qwen3-0.6B")
    >>>
    >>> # Create function handler
    >>> handler = FunctionCallHandler(engine)
    >>>
    >>> # Register functions
    >>> @handler.register
    ... def get_weather(location: str):
    ...     return {"temp": 20, "condition": "sunny"}
    >>>
    >>> # Use the handler
    >>> response = handler.chat("What's the weather in Paris?")
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .function_calling import (
    Function,
    FunctionCall,
    FunctionCallingConfig,
    FunctionCallingManager,
)
from .sampling_params import SamplingParams


class InferenceEngine(Protocol):
    """Protocol for inference engines."""

    def generate(self, prompt: str | list[str], sampling_params: SamplingParams | None = None) -> list[Any]:
        """Generate text from prompt."""
        ...


class FunctionFormatter(ABC):
    """Abstract base class for formatting functions for different models."""

    @abstractmethod
    def format_system_prompt(self, functions: list[Function]) -> str:
        """Format system prompt with function definitions."""
        pass

    @abstractmethod
    def format_user_prompt(self, message: str, system_prompt: str) -> str:
        """Format user message with system prompt."""
        pass

    @abstractmethod
    def parse_function_calls(self, response: str) -> list[FunctionCall]:
        """Parse function calls from model response."""
        pass

    @abstractmethod
    def format_function_results(self, results: list[dict[str, Any]], original_query: str, original_response: str) -> str:
        """Format function results for final response generation."""
        pass


@dataclass
class FunctionCallResult:
    """Result of a function calling interaction."""

    query: str
    initial_response: str
    function_calls: list[dict[str, Any]]
    final_response: str
    success: bool = True
    error: str | None = None


class OpenAIFunctionFormatter(FunctionFormatter):
    """Formatter for OpenAI-style function calling."""

    def format_system_prompt(self, functions: list[Function]) -> str:
        """Format functions in OpenAI style."""
        if not functions:
            return ""

        function_schemas = [f.to_json_schema() for f in functions]

        return f"""You are a helpful assistant. You have access to the following functions:

{json.dumps(function_schemas, indent=2)}

When you need to use a function, respond with a JSON object in this format:
{{"function_call": {{"name": "function_name", "arguments": {{"param": "value"}}}}}}

You can call multiple functions by including multiple function_call objects."""

    def format_user_prompt(self, message: str, system_prompt: str) -> str:
        """Format using chat template."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        return messages

    def parse_function_calls(self, response: str) -> list[FunctionCall]:
        """Parse OpenAI-style function calls."""
        calls = []

        # Find function_call objects
        json_pattern = r'\{"function_call":\s*\{[^}]*"name"[^}]*"arguments"[^}]*\}\}'
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                data = json.loads(match)
                func_data = data["function_call"]
                call = FunctionCall(
                    name=func_data["name"],
                    arguments=func_data.get("arguments", {}),
                    raw_arguments=json.dumps(func_data.get("arguments", {})),
                )
                calls.append(call)
            except (json.JSONDecodeError, KeyError):
                continue

        return calls

    def format_function_results(self, results: list[dict[str, Any]], original_query: str, original_response: str) -> str:
        """Format for OpenAI to generate final response."""
        results_text = "Function Results:\n"
        for result in results:
            if result["status"] == "success":
                results_text += f"- {result['name']}: {json.dumps(result['result'])}\n"
            else:
                results_text += f"- {result['name']}: Error - {result['error']}\n"

        return f"Function results: {results_text}\nPlease provide a natural, helpful response based on these results."


class FunctionCallHandler:
    """Handles function calling with any inference engine.

    This is a standalone handler that works with any text generation engine.
    It orchestrates the function calling flow: prompt formatting, generation,
    function parsing, execution, and final response generation.

    Example:
        >>> # Works with any engine
        >>> from easydel.inference.esurge import eSurge
        >>> engine = eSurge(model="Qwen/Qwen3-0.6B")
        >>>
        >>> # Create handler
        >>> handler = FunctionCallHandler(engine, model_type="qwen")
        >>>
        >>> # Register functions
        >>> @handler.register
        ... def calculate(expression: str) -> float:
        ...     return eval(expression)
        >>>
        >>> # Use it
        >>> result = handler.chat("What is 2+2?")
        >>> print(result.final_response)
    """

    def __init__(
        self,
        engine: InferenceEngine,
        model_type: str = "auto",
        config: FunctionCallingConfig | None = None,
        formatter: FunctionFormatter | None = None,
    ):
        """Initialize function call handler.

        Args:
            engine: Any inference engine with generate method
            model_type: Model type ("qwen", "openai", "auto")
            config: Function calling configuration
            formatter: Custom formatter (overrides model_type)
        """
        self.engine = engine
        self.config = config or FunctionCallingConfig()
        self.manager = FunctionCallingManager(self.config)

        # Set up formatter
        if formatter:
            self.formatter = formatter
        else:
            self.formatter = self._get_formatter(model_type, engine)

    def _get_formatter(self, model_type: str, engine: Any) -> FunctionFormatter:
        """Get appropriate formatter based on model type."""

        return OpenAIFunctionFormatter()

    def register(self, func: Callable) -> Callable:
        """Register a function.

        Args:
            func: Function to register

        Returns:
            The original function (for decorator use)
        """
        return self.manager.register_function(func)

    def register_function(self, function: Function) -> None:
        """Register a function with explicit schema.

        Args:
            function: Function object with schema
        """
        self.manager.register(function)

    def chat(
        self,
        message: str,
        sampling_params: SamplingParams | None = None,
        max_iterations: int = 3,
        include_functions: bool = True,
    ) -> FunctionCallResult:
        """Process a chat message with function calling.

        Args:
            message: User message
            sampling_params: Generation parameters
            max_iterations: Max function calling iterations
            include_functions: Whether to include functions

        Returns:
            FunctionCallResult with the complete interaction
        """
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=512, temperature=0.7)

        # Format prompt with functions
        if include_functions and self.manager.functions:
            functions = list(self.manager.functions.values())
            system_prompt = self.formatter.format_system_prompt(functions)
        else:
            system_prompt = ""

        messages = self.formatter.format_user_prompt(message, system_prompt)

        # Apply chat template using the engine's tokenizer
        if hasattr(self.engine, "tokenizer") and hasattr(self.engine.tokenizer, "apply_chat_template"):
            prompt = self.engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback to simple concatenation if no chat template available
            prompt = self._messages_to_string(messages)

        # Track all function calls
        all_function_calls = []
        current_prompt = prompt
        initial_response = None

        for _ in range(max_iterations):
            # Generate response
            outputs = self.engine.generate(current_prompt, sampling_params)

            # Extract text from output
            if hasattr(outputs[0], "outputs"):
                response_text = outputs[0].outputs[0].text
            else:
                response_text = str(outputs[0])

            if initial_response is None:
                initial_response = response_text

            # Parse function calls
            function_calls = self.formatter.parse_function_calls(response_text)

            if not function_calls:
                # No functions called, return response
                return FunctionCallResult(
                    query=message,
                    initial_response=initial_response,
                    function_calls=all_function_calls,
                    final_response=response_text,
                )

            # Execute functions
            results = []
            for call in function_calls:
                result = self.manager.execute_function(call)
                results.append(result)
                all_function_calls.append(result)

            # Format results for next iteration
            results_text = self.formatter.format_function_results(results, message, response_text)

            # Convert back to messages format and apply chat template
            if isinstance(results_text, str):
                # Parse the formatted text back into messages
                updated_messages = [
                    *messages,
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": "Based on the function results, please provide a complete answer."},
                ]

                if hasattr(self.engine, "tokenizer") and hasattr(self.engine.tokenizer, "apply_chat_template"):
                    current_prompt = self.engine.tokenizer.apply_chat_template(
                        updated_messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    current_prompt = self._messages_to_string(updated_messages)
            else:
                current_prompt = results_text

        # Final generation with all results
        final_outputs = self.engine.generate(current_prompt, sampling_params)

        if hasattr(final_outputs[0], "outputs"):
            final_response = final_outputs[0].outputs[0].text
        else:
            final_response = str(final_outputs[0])

        return FunctionCallResult(
            query=message,
            initial_response=initial_response,
            function_calls=all_function_calls,
            final_response=final_response,
        )

    def _messages_to_string(self, messages: list[dict]) -> str:
        """Convert messages to simple string format as fallback."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nAssistant:"

    def batch_chat(
        self,
        messages: list[str],
        sampling_params: SamplingParams | None = None,
        max_iterations: int = 3,
        include_functions: bool = True,
    ) -> list[FunctionCallResult]:
        """Process multiple messages.

        Args:
            messages: List of user messages
            sampling_params: Generation parameters
            max_iterations: Max function calling iterations
            include_functions: Whether to include functions

        Returns:
            List of FunctionCallResults
        """
        results = []
        for message in messages:
            result = self.chat(message, sampling_params, max_iterations, include_functions)
            results.append(result)
        return results
