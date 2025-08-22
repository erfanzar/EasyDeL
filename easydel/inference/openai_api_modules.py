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

import time
import typing as tp
import uuid
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    field_names: tp.ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # if any(k not in field_names for k in data):
        #     warnings.warn(
        #         f"The following fields were present in the request but ignored: {data.keys() - field_names}",
        #         stacklevel=1,
        #     )
        return result


class ChatMessage(OpenAIBaseModel):
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


class DeltaMessage(OpenAIBaseModel):
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


class Function(OpenAIBaseModel):
    """Function definition for OpenAI-compatible function calling."""

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(OpenAIBaseModel):
    """Tool definition supporting function calling."""

    type: str = "function"
    function: Function


class DeltaFunctionCall(OpenAIBaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: tp.Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class UsageInfo(OpenAIBaseModel):
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


class FunctionDefinition(OpenAIBaseModel):
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


class ToolDefinition(OpenAIBaseModel):
    """Defines a tool that can be called by the model."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(OpenAIBaseModel):
    """
    Represents a request to the chat completion endpoint.
    Mirrors the OpenAI ChatCompletion request structure.
    """

    model: str
    messages: list[ChatMessage]
    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    suppress_tokens: list[int] = Field(default_factory=list)
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


class ChatCompletionResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a non-streaming chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "tool_calls", "abort"] | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    """Represents a non-streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a streaming chat completion response chunk."""

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """Represents a single chunk in a streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo  # Usage info might be included in chunks, often zero until the end


class CountTokenRequest(OpenAIBaseModel):
    """Represents a request to the token counting endpoint."""

    model: str
    conversation: str | list[ChatMessage]  # Can count tokens for a string or a list of messages


class CompletionRequest(OpenAIBaseModel):
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


class CompletionLogprobs(OpenAIBaseModel):
    """Log probabilities for token generation."""

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a completion response."""

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionResponse(OpenAIBaseModel):
    """Represents a response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a streaming completion response chunk."""

    index: int
    text: str  # The delta text content
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionStreamResponse(OpenAIBaseModel):
    """Represents a streaming response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"  # Correct object type for streaming
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]  # Use the new streaming choice model
    usage: UsageInfo | None = None
    # Usage is often None until the final chunk in OAI


class FunctionCall(OpenAIBaseModel):
    """Represents a function call in the OpenAI format."""

    name: str
    arguments: str  # JSON string of arguments


class Function(OpenAIBaseModel):
    """Function definition for OpenAI-compatible function calling."""

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolCall(OpenAIBaseModel):
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


class ExtractedToolCallInformation(OpenAIBaseModel):
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None
