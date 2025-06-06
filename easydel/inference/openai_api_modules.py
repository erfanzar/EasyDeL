# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Defines Pydantic models for the vInference API, mimicking OpenAI's structure."""

import json
import re
import time
import typing as tp
import uuid
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""

    role: str
    content: str | list[tp.Mapping[str, str]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(BaseModel):
    """Represents a change (delta) in a chat message, used in streaming responses."""

    role: str | None = None
    content: str | list[tp.Mapping[str, str]] | None = None
    function_call: dict[str, tp.Any] | None = None


class UsageInfo(BaseModel):
    """Provides information about token usage and processing time for a request."""

    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(BaseModel):
    """Defines a function that can be called by the model."""

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
    chat_template_kwargs: dict[str, int | float | str] | None = None


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
    max_tokens: int = 16
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
    finish_reason: tp.Literal["stop", "length"] | None = None


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
    logprobs: CompletionLogprobs | None = None  # Logprobs are usually None in streaming chunks
    finish_reason: tp.Literal["stop", "length"] | None = None


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
    """Supported function call formats."""

    OPENAI = "openai"  # OpenAI's format
    JSON_SCHEMA = "json_schema"  # Direct JSON schema
    HERMES = "hermes"  # Hermes function calling format
    GORILLA = "gorilla"  # Gorilla function calling format


@dataclass
class FunctionCallParser:
    """Parser for extracting function calls from generated text."""

    format: FunctionCallFormat = FunctionCallFormat.OPENAI
    strict: bool = False  # If True, require exact format matching

    def parse(self, text: str) -> list[FunctionCall] | None:
        """Parse function calls from generated text."""
        if self.format == FunctionCallFormat.OPENAI:
            return self._parse_openai_format(text)
        elif self.format == FunctionCallFormat.JSON_SCHEMA:
            return self._parse_json_schema_format(text)
        elif self.format == FunctionCallFormat.HERMES:
            return self._parse_hermes_format(text)
        elif self.format == FunctionCallFormat.GORILLA:
            return self._parse_gorilla_format(text)
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
                    return [FunctionCall(name=func_data["name"], arguments=json.dumps(func_data.get("arguments", {})))]
                elif "name" in data:
                    return [
                        FunctionCall(
                            name=data["name"], arguments=json.dumps(data.get("arguments", data.get("parameters", {})))
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
                function_calls.append(FunctionCall(name=data["name"], arguments=json.dumps(data.get("arguments", {}))))
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
            tool_def = {"name": func.name, "description": func.description, "parameters": func.parameters}
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
        cls, function_calls: list[FunctionCall], content: str | None = None
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
