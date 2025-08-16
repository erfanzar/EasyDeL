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

"""Function calling mixin for eSurge engine.

This module provides a mixin class that adds function calling capabilities
to the eSurge inference engine.

Classes:
    FunctionCallingMixin: Adds function calling methods to eSurge

Example:
    >>> from easydel.inference.esurge import eSurge
    >>>
    >>> # eSurge automatically includes function calling
    >>> engine = eSurge(model="Qwen/Qwen3-0.6B")
    >>>
    >>> @engine.register_function
    ... def get_weather(location: str):
    ...     return {"temp": 20, "condition": "sunny"}
    >>>
    >>> response = engine.generate_with_functions(
    ...     "What's the weather in Paris?"
    ... )
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..function_calling import Function, FunctionCallingConfig, FunctionCallingManager, FunctionCallingMode
from ..sampling_params import SamplingParams


class FunctionCallingMixin:
    """Mixin to add function calling capabilities to inference engines."""

    def __init__(self, *args, **kwargs):
        """Initialize function calling support."""
        super().__init__(*args, **kwargs)

        # Initialize function calling manager
        self.function_manager = FunctionCallingManager(
            config=FunctionCallingConfig(mode=FunctionCallingMode.AUTO, max_iterations=3, include_thinking=False)
        )

    def register_function(self, func: Callable) -> Callable:
        """Register a function for the model to call.

        Args:
            func: Python function to register

        Returns:
            The original function (for decorator use)

        Example:
            >>> @engine.register_function
            ... def calculate(expression: str) -> float:
            ...     return eval(expression)
        """
        return self.function_manager.register_function(func)

    def register_function_schema(self, function: Function) -> None:
        """Register a function with explicit schema.

        Args:
            function: Function object with schema

        Example:
            >>> func = Function(
            ...     name="search",
            ...     description="Search the web",
            ...     parameters={...},
            ...     implementation=search_impl
            ... )
            >>> engine.register_function_schema(func)
        """
        self.function_manager.register(function)

    def set_function_calling_mode(self, mode: FunctionCallingMode) -> None:
        """Set the function calling mode.

        Args:
            mode: Function calling mode (AUTO, NONE, REQUIRED, SPECIFIC)
        """
        self.function_manager.config.mode = mode

    def generate_with_functions(
        self,
        prompt: str | list[str],
        sampling_params: SamplingParams | None = None,
        include_functions: bool = True,
        function_names: list[str] | None = None,
        max_iterations: int = 3,
        return_intermediate: bool = False,
    ) -> str | list[str] | dict[str, Any]:
        """Generate response with function calling support.

        This method automatically:
        1. Adds function definitions to the prompt
        2. Generates initial response
        3. Parses and executes function calls
        4. Generates final response with results

        Args:
            prompt: User prompt(s)
            sampling_params: Generation parameters
            include_functions: Whether to include function definitions
            function_names: Specific functions to include (None for all)
            max_iterations: Maximum function calling iterations
            return_intermediate: Return intermediate results

        Returns:
            Final response after function execution, or dict with details

        Example:
            >>> response = engine.generate_with_functions(
            ...     "What's 2+2 and what's the weather in NYC?"
            ... )
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=512, temperature=0.7, stop=["</function_calls>", "\n\nUser:", "\n\nSystem:"]
            )

        # Handle batch prompts
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        results = []
        for single_prompt in prompts:
            result = self._generate_single_with_functions(
                single_prompt, sampling_params, include_functions, function_names, max_iterations, return_intermediate
            )
            results.append(result)

        return results if is_batch else results[0]

    def _generate_single_with_functions(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        include_functions: bool,
        function_names: list[str] | None,
        max_iterations: int,
        return_intermediate: bool,
    ) -> str | dict[str, Any]:
        """Generate response for a single prompt with functions."""

        # Build prompt with function definitions
        if include_functions and self.function_manager.functions:
            full_prompt = self._build_function_prompt(prompt, function_names)
        else:
            full_prompt = prompt

        intermediate_results = []
        current_prompt = full_prompt

        for iteration in range(max_iterations):
            # Generate response
            outputs = self.generate(current_prompt, sampling_params=sampling_params)

            response = outputs[0].outputs[0].text
            intermediate_results.append({"iteration": iteration, "response": response})

            # Parse and execute function calls
            function_results = self.function_manager.execute_function_calls(response)

            if not function_results:
                # No function calls, return response
                if return_intermediate:
                    return {"final_response": response, "intermediate": intermediate_results, "function_calls": []}
                return response

            # Add function results to intermediate
            intermediate_results[-1]["function_calls"] = function_results

            # Format results for next iteration
            results_text = self.function_manager.format_function_results(function_results)

            # Update prompt for next iteration
            current_prompt = (
                f"{current_prompt}\n\nAssistant: {response}\n\n{results_text}\n\nBased "
                "on these function results, provide a complete answer:"
            )

        # Generate final response after all iterations
        final_outputs = self.generate(
            current_prompt,
            sampling_params=SamplingParams(
                max_tokens=sampling_params.max_tokens, temperature=sampling_params.temperature
            ),
        )

        final_response = final_outputs[0].outputs[0].text

        if return_intermediate:
            return {
                "final_response": final_response,
                "intermediate": intermediate_results,
                "function_calls": self.function_manager.execution_history[-max_iterations:],
            }

        return final_response

    def _build_function_prompt(self, user_prompt: str, function_names: list[str] | None = None) -> str:
        """Build prompt with function definitions.

        Uses model-appropriate format for function definitions.
        """
        # Get model name to determine format
        model_name = getattr(self, "model_name", "").lower()

        # Use Qwen/Hermes format by default
        if "qwen" in model_name or "hermes" in model_name:
            return self._build_qwen_function_prompt(user_prompt, function_names)
        else:
            # Generic format
            return self.function_manager.create_prompt(
                user_prompt, include_functions=True, function_names=function_names
            )

    def _build_qwen_function_prompt(self, user_prompt: str, function_names: list[str] | None = None) -> str:
        """Build Qwen-specific function calling prompt."""

        functions_to_include = function_names or list(self.function_manager.functions.keys())
        function_schemas = []

        for func_name in functions_to_include:
            if func_name in self.function_manager.functions:
                func = self.function_manager.functions[func_name]
                function_schemas.append(func.to_json_schema())

        # Qwen format
        system_message = f"""You are a helpful assistant with access to the following functions:

{json.dumps(function_schemas, indent=2)}

To use a function, respond with a JSON object in this exact format:
{{"name": "function_name", "arguments": {{"param_name": "param_value"}}}}

You may call multiple functions by including multiple JSON objects.
After receiving function results, provide a natural language response."""

        return f"{system_message}\n\nUser: {user_prompt}\nAssistant:"

    async def agenerate_with_functions(
        self,
        prompt: str | list[str],
        sampling_params: SamplingParams | None = None,
        include_functions: bool = True,
        function_names: list[str] | None = None,
        max_iterations: int = 3,
        return_intermediate: bool = False,
    ) -> str | list[str] | dict[str, Any]:
        """Async version of generate_with_functions.

        See generate_with_functions for documentation.
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=512, temperature=0.7, stop=["</function_calls>", "\n\nUser:", "\n\nSystem:"]
            )

        # Handle batch prompts
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        import asyncio

        tasks = [
            self._agenerate_single_with_functions(
                single_prompt, sampling_params, include_functions, function_names, max_iterations, return_intermediate
            )
            for single_prompt in prompts
        ]

        results = await asyncio.gather(*tasks)
        return results if is_batch else results[0]

    async def _agenerate_single_with_functions(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        include_functions: bool,
        function_names: list[str] | None,
        max_iterations: int,
        return_intermediate: bool,
    ) -> str | dict[str, Any]:
        """Async generate response for a single prompt with functions."""

        # Build prompt with function definitions
        if include_functions and self.function_manager.functions:
            full_prompt = self._build_function_prompt(prompt, function_names)
        else:
            full_prompt = prompt

        intermediate_results = []
        current_prompt = full_prompt

        for iteration in range(max_iterations):
            # Generate response
            outputs = await self.agenerate(current_prompt, sampling_params=sampling_params)

            response = outputs[0].outputs[0].text
            intermediate_results.append({"iteration": iteration, "response": response})

            # Parse and execute function calls
            function_results = self.function_manager.execute_function_calls(response)

            if not function_results:
                # No function calls, return response
                if return_intermediate:
                    return {"final_response": response, "intermediate": intermediate_results, "function_calls": []}
                return response

            # Add function results to intermediate
            intermediate_results[-1]["function_calls"] = function_results

            # Format results for next iteration
            results_text = self.function_manager.format_function_results(function_results)

            # Update prompt for next iteration
            current_prompt = (
                f"{current_prompt}\n\nAssistant: {response}\n\n{results_text}\n\nBased on these function"
                " results, provide a complete answer:"
            )

        # Generate final response after all iterations
        final_outputs = await self.agenerate(
            current_prompt,
            sampling_params=SamplingParams(
                max_tokens=sampling_params.max_tokens, temperature=sampling_params.temperature
            ),
        )

        final_response = final_outputs[0].outputs[0].text

        if return_intermediate:
            return {
                "final_response": final_response,
                "intermediate": intermediate_results,
                "function_calls": self.function_manager.execution_history[-max_iterations:],
            }

        return final_response
