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

"""Mixin class for tool calling functionality in inference servers."""

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

    This mixin provides:
    - Tool parser initialization and management
    - Tool call extraction for batch responses
    - Tool call extraction for streaming responses
    - Tool listing and metadata endpoints

    Classes using this mixin should have:
    - self.tool_parsers: dict[str, ToolParser]
    - self.tool_parser_name: str
    - self.enable_function_calling: bool
    """

    tool_parsers: dict[str, ToolParser]

    def initialize_tool_parsers(
        self,
        model_processors: dict[str, tp.Any],
        tool_parser_name: str,
        enable_function_calling: bool,
    ) -> dict[str, ToolParser]:
        """Initialize tool parsers for models.

        Args:
            model_processors: Dictionary mapping model names to their processors/tokenizers
            tool_parser_name: Name of the tool parser to use (e.g., "hermes", "qwen")
            enable_function_calling: Whether to enable function calling

        Returns:
            Dictionary mapping model names to their tool parsers
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
        """Extract tool calls from a batch response.

        Args:
            response_text: The generated text response
            request: The original chat completion request
            model_name: The model name to get the appropriate parser

        Returns:
            Tuple of (ChatMessage with potential tool calls, finish_reason)
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
        """Extract tool calls from streaming response.

        Args:
            model_name: The model name to get the appropriate parser
            previous_text: Previously accumulated text
            current_text: Current accumulated text
            delta_text: New text in this chunk
            previous_token_ids: Previous token IDs (optional)
            current_token_ids: Current token IDs (optional)
            delta_token_ids: Delta token IDs (optional)
            request: The original request (optional)

        Returns:
            DeltaMessage with tool call information or None
        """
        if not hasattr(self, "tool_parsers") or model_name not in self.tool_parsers:
            return None

        tool_parser = self.tool_parsers[model_name]

        previous_token_ids = previous_token_ids or []
        current_token_ids = current_token_ids or []
        delta_token_ids = delta_token_ids or []

        return tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,
        )

    def get_tool_parser_for_model(self, model_name: str) -> ToolParser | None:
        """Get the tool parser for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ToolParser instance or None if not available
        """
        if not hasattr(self, "tool_parsers"):
            return None
        return self.tool_parsers.get(model_name)

    def create_tools_response(self, model_names: list[str]) -> dict[str, tp.Any]:
        """Create a standardized tools response for listing endpoints.

        Args:
            model_names: List of available model names

        Returns:
            Dictionary with tools information for each model
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

        Returns:
            JSONResponse with NOT_IMPLEMENTED status
        """
        error_response = {
            "error": {
                "message": "Tool execution endpoint is a placeholder. Implement based on your needs.",
                "type": HTTPStatus.NOT_IMPLEMENTED.name,
            }
        }
        return JSONResponse(content=error_response, status_code=HTTPStatus.NOT_IMPLEMENTED.value)
