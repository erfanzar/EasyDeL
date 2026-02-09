# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mixin class for tool calling functionality in inference servers.

This module provides the ToolCallingMixin class which adds tool/function calling
capabilities to inference API servers. It handles the initialization of tool parsers,
extraction of tool calls from both batch and streaming responses, and provides
utility methods for tool-related API endpoints.

The mixin is designed to be used with FastAPI-based inference servers that
implement OpenAI-compatible chat completion APIs.

Example:
    >>> from easydel.inference.tools import ToolCallingMixin
    >>>
    >>> class ChatCompletionServer(ToolCallingMixin):
    ...     def __init__(self, model_processors, tool_parser_name="hermes"):
    ...         self.tool_parsers = self.initialize_tool_parsers(
    ...             model_processors,
    ...             tool_parser_name,
    ...             enable_function_calling=True
    ...         )
    ...
    ...     def generate_response(self, request, response_text):
    ...         message, finish_reason = self.extract_tool_calls_batch(
    ...             response_text, request, "my_model"
    ...         )
    ...         return message, finish_reason

See Also:
    - easydel.inference.tools.abstract_tool.ToolParser: Base parser class
    - easydel.inference.tools.abstract_tool.ToolParserManager: Parser registry
"""

from __future__ import annotations

import typing as tp
from http import HTTPStatus

from eformer.loggings import get_logger
from fastapi.responses import JSONResponse

from ..openai_api_modules import ChatCompletionRequest, ChatMessage, DeltaMessage
from . import ToolParser, ToolParserManager

logger = get_logger("ToolCallingMixin")


class ToolCallingMixin:
    """Mixin class providing tool calling functionality for inference API servers.

    This mixin provides comprehensive tool calling support for inference servers,
    including parser initialization, tool call extraction for both batch and
    streaming responses, and utility methods for tool-related API endpoints.

    The mixin is designed to be inherited by inference server classes that need
    to support OpenAI-compatible function/tool calling APIs.

    Attributes:
        tool_parsers (dict[str, ToolParser]): Dictionary mapping model names to
            their corresponding tool parser instances. This attribute should be
            set by calling initialize_tool_parsers() during server initialization.

    Note:
        Classes using this mixin should have the following attributes available:
            - self.tool_parsers: dict[str, ToolParser] - Set via initialize_tool_parsers()
            - self.tool_parser_name: str - Name of the parser being used (optional)
            - self.enable_function_calling: bool - Whether function calling is enabled (optional)

    Example:
        >>> class MyServer(ToolCallingMixin):
        ...     def __init__(self, tokenizers, parser_name):
        ...         self.tool_parser_name = parser_name
        ...         self.enable_function_calling = True
        ...         self.tool_parsers = self.initialize_tool_parsers(
        ...             tokenizers, parser_name, True
        ...         )
        ...
        ...     def handle_completion(self, request, model_output, model_name):
        ...         return self.extract_tool_calls_batch(
        ...             model_output, request, model_name
        ...         )
    """

    tool_parsers: dict[str, ToolParser]

    def initialize_tool_parsers(
        self,
        model_processors: dict[str, tp.Any],
        tool_parser_name: str,
        enable_function_calling: bool,
    ) -> dict[str, ToolParser]:
        """Initialize tool parsers for all registered models.

        Creates tool parser instances for each model using the specified parser
        implementation. This method should be called during server initialization
        to set up tool calling capabilities.

        Args:
            model_processors (dict[str, tp.Any]): Dictionary mapping model names
                to their tokenizers/processors. Each processor should be compatible
                with the tool parser's requirements.
            tool_parser_name (str): Name of the tool parser to use. Must be a
                registered parser name (e.g., "hermes", "qwen", "mistral").
                See ToolParserManager for available parsers.
            enable_function_calling (bool): Whether to enable function calling.
                If False, returns an empty dictionary and no parsers are initialized.

        Returns:
            dict[str, ToolParser]: Dictionary mapping model names to their
                initialized tool parser instances. Returns an empty dictionary
                if enable_function_calling is False or if parser initialization
                fails for all models.

        Example:
            >>> processors = {"gpt-model": tokenizer1, "llama-model": tokenizer2}
            >>> parsers = self.initialize_tool_parsers(
            ...     processors, "hermes", enable_function_calling=True
            ... )
            >>> print(parsers.keys())
            dict_keys(['gpt-model', 'llama-model'])

        Note:
            If initialization fails for a specific model (e.g., parser not found
            or incompatible tokenizer), a warning is logged and that model is
            skipped. Other models will still be initialized.
        """
        tool_parsers = {}

        if not enable_function_calling:
            return tool_parsers

        for model_name, processor in model_processors.items():
            try:
                parser_class = ToolParserManager.get_tool_parser(tool_parser_name)
                tool_parsers[model_name] = parser_class(processor)
                logger.info(f"Initialized {tool_parser_name} tool parser for model {model_name}")
            except KeyError:
                logger.warning(f"Tool parser '{tool_parser_name}' not found, function calling disabled for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tool parser for {model_name}: {e}")

        return tool_parsers

    def extract_tool_calls_batch(
        self,
        response_text: str,
        request: ChatCompletionRequest,
        model_name: str,
    ) -> tuple[ChatMessage, str]:
        """Extract tool calls from a complete (non-streaming) response.

        Parses the complete model response to identify and extract any tool/function
        calls. This method is used for batch (non-streaming) completions where the
        entire response is available at once.

        Args:
            response_text (str): The complete generated text response from the model.
            request (ChatCompletionRequest): The original chat completion request
                containing tool definitions and other parameters.
            model_name (str): The name of the model that generated the response.
                Used to look up the appropriate parser.

        Returns:
            tuple[ChatMessage, str]: A tuple containing:
                - ChatMessage: The assistant message with role, content, and
                  optional tool_calls field populated if tools were detected.
                - str: The finish_reason - "function_call" if tools were called,
                  or "stop" for regular text completion.

        Example:
            >>> response = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
            >>> message, reason = self.extract_tool_calls_batch(
            ...     response, request, "my_model"
            ... )
            >>> print(reason)
            'function_call'
            >>> print(message.tool_calls[0].function.name)
            'get_weather'

        Note:
            If no tool parser is available for the model or if no tools are
            detected in the response, returns the original text as content
            with finish_reason "stop".
        """
        if not hasattr(self, "tool_parsers") or model_name not in self.tool_parsers:
            return ChatMessage(role="assistant", content=response_text), "stop"

        tool_parser = self.tool_parsers[model_name]
        extracted = tool_parser.extract_tool_calls(response_text, request)

        if extracted.tools_called and extracted.tool_calls:
            message = ChatMessage(
                role="assistant",
                content=extracted.content,
                tool_calls=extracted.tool_calls,
            )
            finish_reason = "function_call"
        else:
            message = ChatMessage(
                role="assistant",
                content=extracted.content if extracted.content else response_text,
            )
            finish_reason = "stop"

        return message, finish_reason

    def extract_tool_calls_streaming(
        self,
        model_name: str,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int] | None = None,
        current_token_ids: list[int] | None = None,
        delta_token_ids: list[int] | None = None,
        request: ChatCompletionRequest | None = None,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming response chunks.

        Processes incremental model output to identify partial tool calls and emit
        appropriate streaming updates. This method maintains state across calls
        (via the parser instance) to handle incomplete JSON/XML structures as
        they are being generated.

        Args:
            model_name (str): The name of the model generating the response.
                Used to look up the appropriate parser.
            previous_text (str): The accumulated text up to the previous call.
            current_text (str): The accumulated text including the current chunk.
            delta_text (str): The new text in the current streaming chunk.
            previous_token_ids (list[int] | None, optional): Token IDs accumulated
                up to the previous call. Defaults to None (treated as empty list).
            current_token_ids (list[int] | None, optional): Token IDs including
                the current chunk. Defaults to None (treated as empty list).
            delta_token_ids (list[int] | None, optional): New token IDs in the
                current chunk. Defaults to None (treated as empty list).
            request (ChatCompletionRequest | None, optional): The original request
                containing tool definitions. Defaults to None.

        Returns:
            DeltaMessage | None: A DeltaMessage containing incremental tool call
                information (tool call index, function name delta, arguments delta),
                or None if no tool call update is available for this chunk.

        Example:
            >>> # First chunk with tool call start
            >>> delta1 = self.extract_tool_calls_streaming(
            ...     "model", "", "<tool_call>", "<tool_call>"
            ... )
            >>> # Subsequent chunk with function name
            >>> delta2 = self.extract_tool_calls_streaming(
            ...     "model", "<tool_call>", '<tool_call>{"name":', '{"name":'
            ... )

        Note:
            This method is stateful - the parser instance maintains state across
            calls to track parsing progress. Make sure to use the same model_name
            consistently for a single streaming session.
        """
        if not hasattr(self, "tool_parsers") or model_name not in self.tool_parsers:
            return None

        tool_parser = self.tool_parsers[model_name]

        previous_token_ids = previous_token_ids or []
        current_token_ids = current_token_ids or []
        delta_token_ids = delta_token_ids or []

        try:
            delta = tool_parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=request,
            )
        except Exception:
            logger.debug(
                "Tool parser streaming extraction failed for model %s; falling back to raw text delta.",
                model_name,
                exc_info=True,
            )
            return DeltaMessage(content=delta_text)

        if delta is None or isinstance(delta, DeltaMessage):
            return delta
        if isinstance(delta, str):
            return DeltaMessage(content=delta)
        if isinstance(delta, dict):
            try:
                return DeltaMessage(**delta)
            except Exception:
                return DeltaMessage(content=delta_text)
        if hasattr(delta, "model_dump"):
            payload = delta.model_dump(exclude_unset=True, exclude_none=True)
            if isinstance(payload, dict):
                try:
                    return DeltaMessage(**payload)
                except Exception:
                    return DeltaMessage(content=delta_text)
        return DeltaMessage(content=delta_text)

    def get_tool_parser_for_model(self, model_name: str) -> ToolParser | None:
        """Get the tool parser instance for a specific model.

        Retrieves the tool parser that was initialized for the given model name.
        Useful for direct access to parser methods or configuration.

        Args:
            model_name (str): The name of the model to get the parser for.

        Returns:
            ToolParser | None: The ToolParser instance for the model, or None
                if no parser is available (either tool_parsers not initialized
                or model not found).

        Example:
            >>> parser = self.get_tool_parser_for_model("my_model")
            >>> if parser:
            ...     adjusted_request = parser.adjust_request(original_request)
        """
        if not hasattr(self, "tool_parsers"):
            return None
        return self.tool_parsers.get(model_name)

    def create_tools_response(self, model_names: list[str]) -> dict[str, tp.Any]:
        """Create a standardized tools response for API listing endpoints.

        Generates a response object containing tool information for each model,
        suitable for returning from a /tools or /models endpoint. Includes
        example tool definitions and supported formats.

        Args:
            model_names (list[str]): List of model names to include in the response.

        Returns:
            dict[str, tp.Any]: A dictionary containing:
                - "models": dict mapping model names to their tool configurations,
                  where each configuration includes:
                    - "tools": list of example tool definitions
                    - "tool_parser": name of the parser being used (or None)
                    - "formats_supported": list of all registered parser names
                    - "parallel_calls": bool indicating parallel call support
                - "default_format": the default tool format ("openai")

        Example:
            >>> response = self.create_tools_response(["model1", "model2"])
            >>> print(response["default_format"])
            'openai'
            >>> print(response["models"]["model1"]["formats_supported"])
            ['hermes', 'qwen', 'mistral', ...]
        """
        tools_by_model = {}

        for model_name in model_names:
            model_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "example_function",
                        "description": "An example function for demonstration",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "First parameter",
                                },
                                "param2": {
                                    "type": "number",
                                    "description": "Second parameter",
                                },
                            },
                            "required": ["param1"],
                        },
                    },
                }
            ]

            has_parser = hasattr(self, "tool_parsers") and model_name in self.tool_parsers

            tools_by_model[model_name] = {
                "tools": model_tools,
                "tool_parser": getattr(self, "tool_parser_name", None) if has_parser else None,
                "formats_supported": list(ToolParserManager.tool_parsers.keys()),
                "parallel_calls": True,
            }

        return {"models": tools_by_model, "default_format": "openai"}

    def create_tool_execution_placeholder(self) -> JSONResponse:
        """Create a placeholder response for tool execution endpoints.

        Returns a NOT_IMPLEMENTED response indicating that the tool execution
        endpoint needs to be implemented by the user. This is a placeholder
        that should be replaced with actual tool execution logic.

        Returns:
            JSONResponse: A FastAPI JSONResponse with HTTP 501 NOT_IMPLEMENTED
                status and an error message explaining that the endpoint needs
                implementation.

        Example:
            >>> @app.post("/tools/execute")
            ... def execute_tool(self, request: ToolExecuteRequest):
            ...     # Replace this with actual implementation
            ...     return self.create_tool_execution_placeholder()
        """
        error_response = {
            "error": {
                "message": "Tool execution endpoint is a placeholder. Implement based on your needs.",
                "type": HTTPStatus.NOT_IMPLEMENTED.name,
            }
        }
        return JSONResponse(content=error_response, status_code=HTTPStatus.NOT_IMPLEMENTED.value)
