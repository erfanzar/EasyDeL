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
"""OpenAI API compatibility models and utilities.

This module provides Pydantic models and utilities for OpenAI API compatibility,
enabling EasyDeL inference engines to work with OpenAI-compatible clients and tools.

Key Components:
    - Request/Response models for chat completions and text completions
    - Function calling support with multiple format parsers
    - Token usage tracking and metrics
    - Streaming response models

Classes:
    ChatMessage: Single message in a conversation
    DeltaMessage: Incremental message for streaming
    UsageInfo: Token usage and performance metrics
    ChatCompletionRequest: Request for chat completions
    ChatCompletionResponse: Response from chat completions
    CompletionRequest: Request for text completions
    CompletionResponse: Response from text completions
    FunctionCallFormat: Supported function call formats
    FunctionCallFormatter: Formatter for function call prompts
    FunctionCallParser: Parser for extracting function calls

Example:
    >>> from easydel.inference.openai_api_modules import (
    ...     ChatCompletionRequest,
    ...     ChatMessage
    ... )
    >>> request = ChatCompletionRequest(
    ...     model="gpt-3.5-turbo",
    ...     messages=[
    ...         ChatMessage(role="user", content="Hello!")
    ...     ],
    ...     temperature=0.7
    ... )
"""

import json
import re
import time
import typing as tp
import uuid
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation.

    Attributes:
        role: Message role (system, user, assistant, function)
        content: Message content (text or structured)
        name: Optional name for the message sender
        function_call: Optional function call made by assistant
    """

    role: str
    content: str | list[tp.Mapping[str, str]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(BaseModel):
    """Represents a change (delta) in a chat message.

    Used in streaming responses to send incremental updates.

    Attributes:
        role: Optional role if starting new message
        content: Incremental content to append
        function_call: Optional function call updates
    """

    role: str | None = None
    content: str | list[tp.Mapping[str, str]] | None = None
    function_call: dict[str, tp.Any] | None = None


class UsageInfo(BaseModel):
    """Token usage and performance metrics.

    Tracks computational resources used for a request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens generated
        total_tokens: Sum of prompt and completion tokens
        tokens_per_second: Generation speed
        processing_time: Total processing time in seconds
    """

    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(BaseModel):
    """Defines a function that can be called by the model.

    Attributes:
        name: Function name
        description: Function description for the model
        parameters: JSON Schema for function parameters
        required: List of required parameter names
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)
    required: list[str] | None = None


class ToolDefinition(BaseModel):
    """Defines a tool that can be called by the model."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """
    Represents a request to the chat completion endpoint.
    Mirrors the OpenAI ChatCompletion request structure.
    """

    # Core parameters
    model: str
    messages: list[ChatMessage]

    # Sampling parameters (mirroring OpenAI)
    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    suppress_tokens: list[int] = Field(default_factory=list)
    # Added for potential EasyDeL support

    # OpenAI native parameters (some may be ignored by vInference)
    functions: list[FunctionDefinition] | None = None
    function_call: str | dict[str, tp.Any] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, tp.Any] | None = None
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] | None = None  # Ignored by EasyDeL
    user: str | None = None  # Ignored by EasyDeL
    chat_template_kwargs: dict[str, int | float | str | bool] | None = None


class ChatCompletionResponseChoice(BaseModel):
    """Represents a single choice within a non-streaming chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionResponse(BaseModel):
    """Represents a non-streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(BaseModel):
    """Represents a single choice within a streaming chat completion response chunk."""

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionStreamResponse(BaseModel):
    """Represents a single chunk in a streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo  # Usage info might be included in chunks, often zero until the end


class CountTokenRequest(BaseModel):
    """Represents a request to the token counting endpoint."""

    model: str
    conversation: str | list[ChatMessage]  # Can count tokens for a string or a list of messages


class CompletionRequest(BaseModel):
    """
    Represents a request to the completions endpoint.
    Mirrors the OpenAI Completion request structure.
    """

    model: str
    prompt: str | list[str]
    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    suppress_tokens: list[int] = Field(default_factory=list)
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class CompletionLogprobs(BaseModel):
    """Log probabilities for token generation."""

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(BaseModel):
    """Represents a single choice within a completion response."""

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionResponse(BaseModel):
    """Represents a response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


# New model for streaming completion choices (OAI compatible)
class CompletionStreamResponseChoice(BaseModel):
    """Represents a single choice within a streaming completion response chunk."""

    index: int
    text: str  # The delta text content
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionStreamResponse(BaseModel):
    """Represents a streaming response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"  # Correct object type for streaming
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]  # Use the new streaming choice model
    usage: UsageInfo | None = None
    # Usage is often None until the final chunk in OAI


class FunctionCall(BaseModel):
    """Represents a function call in the OpenAI format."""

    name: str
    arguments: str  # JSON string of arguments


class Function(BaseModel):
    """Function definition for OpenAI-compatible function calling."""

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """Tool definition supporting function calling."""

    type: str = "function"
    function: Function


class ToolCall(BaseModel):
    """Represents a tool call in responses."""

    id: str
    type: str = "function"
    function: FunctionCall


class FunctionCallFormat(str, Enum):
    """Supported function call formats.

    Different models and frameworks use different formats for function calling.

    Attributes:
        OPENAI: OpenAI's standard format
        JSON_SCHEMA: Direct JSON schema format
        HERMES: Hermes model function calling format
        GORILLA: Gorilla model function calling format
        QWEN: Qwen's special token format (✿FUNCTION✿)
        NOUS: Nous XML-style format (<tool_call>)
    """

    OPENAI = "openai"  # OpenAI's format
    JSON_SCHEMA = "json_schema"  # Direct JSON schema
    HERMES = "hermes"  # Hermes function calling format
    GORILLA = "gorilla"  # Gorilla function calling format
    QWEN = "qwen"  # Qwen's special token format
    NOUS = "nous"  # Nous XML-style format


@dataclass
class FunctionCallParser:
    """Parser for extracting function calls from generated text.

    Supports multiple function calling formats and can extract
    structured function calls from model outputs.

    Attributes:
        format: Function call format to parse
        strict: If True, require exact format matching

    Methods:
        parse: Extract function calls from text
    """

    format: FunctionCallFormat = FunctionCallFormat.OPENAI
    strict: bool = False  # If True, require exact format matching

    def parse(self, text: str) -> list[FunctionCall] | None:
        """Parse function calls from generated text.

        Args:
            text: Generated text containing function calls

        Returns:
            List of parsed FunctionCall objects, or None if no calls found

        Raises:
            ValueError: If format is unsupported
            json.JSONDecodeError: If strict mode and JSON is invalid
        """
        if self.format == FunctionCallFormat.OPENAI:
            return self._parse_openai_format(text)
        elif self.format == FunctionCallFormat.JSON_SCHEMA:
            return self._parse_json_schema_format(text)
        elif self.format == FunctionCallFormat.HERMES:
            return self._parse_hermes_format(text)
        elif self.format == FunctionCallFormat.GORILLA:
            return self._parse_gorilla_format(text)
        elif self.format == FunctionCallFormat.QWEN:
            return self._parse_qwen_format(text)
        elif self.format == FunctionCallFormat.NOUS:
            return self._parse_nous_format(text)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _parse_openai_format(self, text: str) -> list[FunctionCall] | None:
        """Parse OpenAI-style function calls."""
        function_calls = []

        # Look for function call patterns
        # Pattern 1: Direct JSON after specific markers
        patterns = [
            r"<function_call>\s*({.*?})\s*</function_call>",
            r"Function call:\s*({.*?})",
            r"```json\s*({.*?})\s*```",
            r'({.*?"name".*?"arguments".*?})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    data = json.loads(match)
                    if "name" in data and ("arguments" in data or "parameters" in data):
                        args = data.get("arguments", data.get("parameters", {}))
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        function_calls.append(FunctionCall(name=data["name"], arguments=args))
                except json.JSONDecodeError:
                    if self.strict:
                        raise
                    continue

        # Pattern 2: Natural language function calls
        if not function_calls and not self.strict:
            # Look for patterns like "call function_name with ..."
            nl_pattern = r"call\s+(\w+)\s+with\s+({.*?}|\(.*?\))"
            nl_matches = re.findall(nl_pattern, text, re.IGNORECASE | re.DOTALL)
            for name, args in nl_matches:
                try:
                    # Try to parse arguments
                    args = args.strip("()")
                    if args.startswith("{"):
                        args_dict = json.loads(args)
                    else:
                        # Simple key=value parsing
                        args_dict = {}
                        for pair in args.split(","):
                            if "=" in pair:
                                k, v = pair.split("=", 1)
                                args_dict[k.strip()] = v.strip().strip("\"'")

                    function_calls.append(FunctionCall(name=name, arguments=json.dumps(args_dict)))
                except Exception:
                    if self.strict:
                        raise
                    continue

        return function_calls if function_calls else None

    def _parse_json_schema_format(self, text: str) -> list[FunctionCall] | None:
        """Parse direct JSON schema format."""
        try:
            # Extract JSON from text
            json_match = re.search(r"{.*}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "function" in data:
                    func_data = data["function"]
                    return [
                        FunctionCall(
                            name=func_data["name"],
                            arguments=json.dumps(func_data.get("arguments", {})),
                        )
                    ]
                elif "name" in data:
                    return [
                        FunctionCall(
                            name=data["name"],
                            arguments=json.dumps(data.get("arguments", data.get("parameters", {}))),
                        )
                    ]
        except json.JSONDecodeError:
            if self.strict:
                raise
        return None

    def _parse_hermes_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Hermes-style function calls."""
        function_calls = []

        # Hermes format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                function_calls.append(
                    FunctionCall(
                        name=data["name"],
                        arguments=json.dumps(data.get("arguments", {})),
                    )
                )
            except json.JSONDecodeError:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None

    def _parse_gorilla_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Gorilla-style function calls."""
        function_calls = []

        pattern = r"<<<(\w+)\((.*?)\)>>>"
        matches = re.findall(pattern, text)

        for name, args_str in matches:
            try:
                # Parse arguments
                args_dict = {}
                if args_str:
                    for arg in args_str.split(","):
                        if "=" in arg:
                            k, v = arg.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("\"'")
                            try:
                                v = json.loads(v)
                            except Exception:
                                pass
                            args_dict[k] = v

                function_calls.append(FunctionCall(name=name, arguments=json.dumps(args_dict)))
            except Exception:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None

    def _parse_qwen_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Qwen-style function calls with special tokens."""
        function_calls = []

        # Look for Qwen special token patterns
        # Pattern: ✿FUNCTION✿: function_name\n✿ARGS✿: {...}
        pattern = r"✿FUNCTION✿:\s*(\w+)\s*\n✿ARGS✿:\s*({.*?})"
        matches = re.findall(pattern, text, re.DOTALL)

        for name, args_str in matches:
            try:
                args_dict = json.loads(args_str)
                function_calls.append(
                    FunctionCall(
                        name=name,
                        arguments=json.dumps(args_dict),
                    )
                )
            except json.JSONDecodeError:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None

    def _parse_nous_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Nous-style function calls with XML tags."""
        function_calls = []

        # Pattern: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if "name" in data:
                    args = data.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    function_calls.append(
                        FunctionCall(
                            name=data["name"],
                            arguments=args,
                        )
                    )
            except json.JSONDecodeError:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None


class FunctionCallFormatter:
    """Formats function definitions for inclusion in prompts."""

    @staticmethod
    def format_tools_for_prompt(tools: list[Tool], format: FunctionCallFormat = FunctionCallFormat.OPENAI) -> str:  # noqa
        """Format tool definitions for inclusion in prompts."""
        if format == FunctionCallFormat.OPENAI:
            return FunctionCallFormatter._format_openai_style(tools)
        elif format == FunctionCallFormat.HERMES:
            return FunctionCallFormatter._format_hermes_style(tools)
        elif format == FunctionCallFormat.GORILLA:
            return FunctionCallFormatter._format_gorilla_style(tools)
        elif format == FunctionCallFormat.QWEN:
            return FunctionCallFormatter._format_qwen_style(tools)
        elif format == FunctionCallFormat.NOUS:
            return FunctionCallFormatter._format_nous_style(tools)
        else:
            return FunctionCallFormatter._format_json_style(tools)

    @staticmethod
    def _format_openai_style(tools: list[Tool]) -> str:
        """Format tools in OpenAI style."""
        if not tools:
            return ""

        formatted = "You have access to the following functions:\n\n"

        for tool in tools:
            func = tool.function
            formatted += f"Function: {func.name}\n"
            if func.description:
                formatted += f"Description: {func.description}\n"
            if func.parameters:
                formatted += f"Parameters: {json.dumps(func.parameters, indent=2)}\n"
            formatted += "\n"

        formatted += (
            "To use a function, respond with a JSON object in this format:\n"
            '{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}\n'
        )

        return formatted

    @staticmethod
    def _format_hermes_style(tools: list[Tool]) -> str:
        """Format tools in Hermes style."""
        if not tools:
            return ""

        formatted = "You have access to these tools:\n\n"

        for tool in tools:
            func = tool.function
            tool_def = {
                "name": func.name,
                "description": func.description,
                "parameters": func.parameters,
            }
            formatted += f"<tool>{json.dumps(tool_def, indent=2)}</tool>\n\n"

        formatted += (
            "To use a tool, wrap your response in <tool_call> tags:\n"
            '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>\n'
        )

        return formatted

    @staticmethod
    def _format_gorilla_style(tools: list[Tool]) -> str:
        """Format tools in Gorilla style."""
        if not tools:
            return ""

        formatted = "Available functions:\n\n"

        for tool in tools:
            func = tool.function
            params = []
            if func.parameters.get("properties"):
                for param, schema in func.parameters["properties"].items():
                    param_type = schema.get("type", "any")
                    required = param in func.parameters.get("required", [])
                    params.append(f"{param}: {param_type}{'*' if required else ''}")

            formatted += f"- {func.name}({', '.join(params)})"
            if func.description:
                formatted += f" - {func.description}"
            formatted += "\n"

        formatted += "\nTo call a function, use: <<<function_name(param1=value1, param2=value2)>>>\n"

        return formatted

    @staticmethod
    def _format_json_style(tools: list[Tool]) -> str:
        """Format tools as JSON schema."""
        if not tools:
            return ""

        tools_json = [tool.model_dump() for tool in tools]
        return f"Available tools:\n{json.dumps(tools_json, indent=2)}"

    @staticmethod
    def _format_qwen_style(tools: list[Tool]) -> str:
        """Format tools in Qwen style with special tokens."""
        if not tools:
            return ""

        formatted = "# Tools\n\n## You have access to the following tools:\n\n"

        for tool in tools:
            func = tool.function
            formatted += f"### {func.name}\n"
            if func.description:
                formatted += f"- **Description**: {func.description}\n"
            if func.parameters:
                formatted += f"- **Parameters**: {json.dumps(func.parameters, indent=2)}\n"
            formatted += "\n"

        formatted += (
            "## When you need to call a tool, please insert the following command in your reply:\n\n"
            "✿FUNCTION✿: The tool to use, should be one of [" + ",".join([tool.function.name for tool in tools]) + "]\n"
            "✿ARGS✿: The input of the tool\n"
            "✿RESULT✿: Tool results\n"
            "✿RETURN✿: Reply based on tool results"
        )

        return formatted

    @staticmethod
    def _format_nous_style(tools: list[Tool]) -> str:
        """Format tools in Nous style with XML tags."""
        if not tools:
            return ""

        formatted = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        formatted += "You are provided with function signatures within <tools></tools> XML tags:\n"
        formatted += "<tools>\n"

        for tool in tools:
            func = tool.function
            tool_def = {
                "name": func.name,
                "description": func.description,
                "parameters": func.parameters,
            }
            formatted += f"{json.dumps(tool_def, indent=2)}\n"

        formatted += "</tools>\n\n"
        formatted += (
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

        return formatted


# Enhanced request models with function calling support
class ChatCompletionRequestWithTools(ChatCompletionRequest):
    """Extended chat completion request with tool/function support."""

    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    functions: list[Function] | None = None  # Legacy function format
    function_call: str | dict | None = None  # Legacy function call format

    # Function calling configuration
    function_call_format: FunctionCallFormat = FunctionCallFormat.OPENAI
    parallel_tool_calls: bool = True

    def get_tools(self) -> list[Tool]:
        """Get tools, converting legacy functions if needed."""
        if self.tools:
            return self.tools
        elif self.functions:
            return [Tool(type="function", function=func) for func in self.functions]
        return []


class ChatMessageWithTools(ChatMessage):
    """Chat message with tool call support."""

    tool_calls: list[ToolCall] | None = None

    @classmethod
    def from_function_calls(
        cls,
        function_calls: list[FunctionCall],
        content: str | None = None,
    ) -> "ChatMessageWithTools":
        """Create message from function calls."""
        tool_calls = [
            ToolCall(id=f"call_{i}_{fc.name}", type="function", function=fc) for i, fc in enumerate(function_calls)
        ]

        return cls(role="assistant", content=content, tool_calls=tool_calls)


class ChatCompletionResponseChoiceWithTools(ChatCompletionResponseChoice):
    """Chat completion choice with tool support."""

    message: ChatMessageWithTools


class ChatCompletionStreamResponseChoiceWithTools(ChatCompletionStreamResponseChoice):
    """Streaming chat completion choice with tool support."""

    delta: ChatMessageWithTools | DeltaMessage
