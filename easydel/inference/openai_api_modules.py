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
    ChatMessage: Single message in a conversation.
    DeltaMessage: Incremental message for streaming.
    UsageInfo: Token usage and performance metrics.
    ChatCompletionRequest: Request for chat completions.
    ChatCompletionResponse: Response from chat completions.
    CompletionRequest: Request for text completions.
    CompletionResponse: Response from text completions.
    FunctionCallFormat: Supported function call formats.

Example:
    Creating a chat completion request::

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
    """Base model for OpenAI API compatibility.

    Provides common functionality for all OpenAI API models including
    support for extra fields and field name tracking.

    Attributes:
        model_config: Pydantic configuration allowing extra fields.
        field_names: Class variable caching the set of known field names.
    """

    model_config = ConfigDict(extra="allow")
    field_names: tp.ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        """Validate and track field names for the model.

        Args:
            data: Input data being validated.
            handler: The validation handler function.

        Returns:
            The validated result from the handler.
        """
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

        return result


class ChatMessage(OpenAIBaseModel):
    """Represents a single message in a chat conversation.

    Attributes:
        role: Message role (system, user, assistant, function, tool).
        content: Message content as text or structured content array.
        name: Optional name for the message sender.
        function_call: Optional function call made by the assistant.
    """

    role: str
    content: str | list[tp.Mapping[str, tp.Any]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(OpenAIBaseModel):
    """Represents a change (delta) in a chat message.

    Used in streaming responses to send incremental updates to the client.

    Attributes:
        role: Optional role if starting a new message.
        content: Incremental content to append to the message.
        function_call: Optional function call updates.
    """

    role: str | None = None
    content: str | list[tp.Mapping[str, tp.Any]] | None = None
    function_call: dict[str, tp.Any] | None = None


class Function(OpenAIBaseModel):
    """Function definition for OpenAI-compatible function calling.

    Attributes:
        name: The name of the function to call.
        description: Human-readable description of the function.
        parameters: JSON Schema defining the function parameters.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(OpenAIBaseModel):
    """Tool definition supporting function calling.

    Attributes:
        type: Type of tool, currently only "function" is supported.
        function: The function definition for this tool.
    """

    type: str = "function"
    function: Function


class DeltaFunctionCall(OpenAIBaseModel):
    """Incremental function call update for streaming.

    Attributes:
        name: Function name (typically in first chunk only).
        arguments: Partial JSON string of function arguments.
    """

    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    """Incremental tool call update for streaming.

    Attributes:
        id: Unique identifier for the tool call.
        type: Type of tool (always "function").
        index: Index of this tool call in the array.
        function: Partial function call information.
    """

    id: str | None = None
    type: tp.Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class UsageInfo(OpenAIBaseModel):
    """Token usage and performance metrics.

    Tracks computational resources used for a request.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Sum of prompt and completion tokens.
        tokens_per_second: Token generation speed (tokens/second).
        processing_time: Total processing time in seconds.
    """

    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(OpenAIBaseModel):
    """Defines a function that can be called by the model.

    Attributes:
        name: Function name used in model calls.
        description: Description helping the model understand when to use it.
        parameters: JSON Schema defining function parameters.
        required: List of required parameter names.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)
    required: list[str] | None = None


class ToolDefinition(OpenAIBaseModel):
    """Defines a tool that can be called by the model.

    Attributes:
        type: Type of tool, currently only "function" is supported.
        function: The function definition for this tool.
    """

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(OpenAIBaseModel):
    """Represents a request to the chat completion endpoint.

    Mirrors the OpenAI ChatCompletion request structure with additional
    EasyDeL-specific parameters.

    Attributes:
        model: Model identifier to use for completion.
        messages: List of messages in the conversation.
        max_tokens: Maximum tokens to generate.
        presence_penalty: Penalty for token presence (-2.0 to 2.0).
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0).
        repetition_penalty: Multiplicative repetition penalty.
        temperature: Sampling temperature (0.0 to 2.0).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter (0 disables).
        min_p: Minimum probability threshold.
        suppress_tokens: Token IDs to suppress during generation.
        functions: Legacy function definitions (deprecated).
        function_call: Legacy function call control (deprecated).
        tools: Tool definitions for function calling.
        tool_choice: Control over tool selection.
        n: Number of completions to generate.
        stream: Whether to stream responses.
        stop: Stop sequences to end generation.
        logit_bias: Bias to apply to token logits (ignored by EasyDeL).
        user: User identifier for tracking (ignored by EasyDeL).
        chat_template_kwargs: Additional kwargs for chat template.
    """

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
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
    """Represents a single choice in a non-streaming chat completion response.

    Attributes:
        index: Index of this choice in the response.
        message: The generated message content.
        finish_reason: Reason generation stopped (stop, length, function_call, etc.).
    """

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "tool_calls", "abort"] | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    """Represents a non-streaming response from the chat completion endpoint.

    Attributes:
        id: Unique identifier for the completion.
        object: Object type, always "chat.completion".
        created: Unix timestamp of when the completion was created.
        model: Model used for the completion.
        choices: List of completion choices.
        usage: Token usage information.
    """

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice in a streaming chat completion response chunk.

    Attributes:
        index: Index of this choice in the response.
        delta: Incremental message content.
        finish_reason: Reason generation stopped (if this is the final chunk).
    """

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "finished"] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """Represents a single chunk in a streaming response from chat completions.

    Attributes:
        id: Unique identifier for the completion.
        object: Object type, always "chat.completion.chunk".
        created: Unix timestamp of when the chunk was created.
        model: Model used for the completion.
        choices: List of completion choices with deltas.
        usage: Token usage information (often zero until final chunk).
    """

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo  # Usage info might be included in chunks, often zero until the end


class CountTokenRequest(OpenAIBaseModel):
    """Represents a request to the token counting endpoint.

    Attributes:
        model: Model to use for tokenization.
        conversation: Content to count tokens for (string or message list).
    """

    model: str
    conversation: str | list[ChatMessage]  # Can count tokens for a string or a list of messages


class CompletionRequest(OpenAIBaseModel):
    """Represents a request to the completions endpoint.

    Mirrors the OpenAI Completion request structure with additional
    EasyDeL-specific parameters.

    Attributes:
        model: Model identifier to use for completion.
        prompt: Input prompt(s) to complete.
        max_tokens: Maximum tokens to generate.
        presence_penalty: Penalty for token presence (-2.0 to 2.0).
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0).
        repetition_penalty: Multiplicative repetition penalty.
        temperature: Sampling temperature (0.0 to 2.0).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter (0 disables).
        min_p: Minimum probability threshold.
        suppress_tokens: Token IDs to suppress during generation.
        n: Number of completions to generate.
        stream: Whether to stream responses.
        stop: Stop sequences to end generation.
        logit_bias: Bias to apply to token logits.
        user: User identifier for tracking.
    """

    model: str
    prompt: str | list[str]
    max_tokens: int | None = None
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


class ResponsesRequest(OpenAIBaseModel):
    """Represents a request to the OpenAI Responses API (POST /v1/responses).

    This is the newer unified API that combines chat and completion functionality.

    Attributes:
        model: Model identifier to use.
        input: Input content (string, message list, or structured input).
        messages: Alternative message-based input format.
        instructions: System instructions for the model.
        store: Whether to store the response for later retrieval.
        previous_response_id: ID of previous response for continuation.
        conversation: Conversation identifier or object.
        max_output_tokens: Maximum tokens to generate.
        max_tokens: Alternative max tokens parameter.
        max_completion_tokens: Alternative max tokens parameter.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability threshold.
        presence_penalty: Penalty for token presence.
        frequency_penalty: Penalty for token frequency.
        repetition_penalty: Multiplicative repetition penalty.
        stop: Stop sequences.
        n: Number of responses to generate.
        tools: Tool definitions for function calling.
        functions: Legacy function definitions.
        tool_choice: Control over tool selection.
        parallel_tool_calls: Allow parallel tool execution.
        stream: Whether to stream the response.
        truncation: Truncation strategy.
        include: Additional data to include in response.
        metadata: Custom metadata for the request.
    """

    model: str
    input: str | list[tp.Any] | None = None
    messages: list[tp.Mapping[str, tp.Any]] | None = None
    instructions: str | None = None

    # Conversation state / storage
    store: bool | None = None
    previous_response_id: str | None = None
    conversation: str | dict[str, tp.Any] | None = None

    # Generation settings
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None

    # Tools
    tools: list[tp.Mapping[str, tp.Any]] | None = None
    functions: list[tp.Mapping[str, tp.Any]] | None = None
    tool_choice: str | dict[str, tp.Any] | None = None
    parallel_tool_calls: bool | None = None

    # Streaming / misc
    stream: bool | None = False
    truncation: str | None = None
    include: list[str] | None = None
    metadata: dict[str, tp.Any] | None = None


class CompletionLogprobs(OpenAIBaseModel):
    """Log probabilities for token generation.

    Attributes:
        tokens: List of generated tokens.
        token_logprobs: Log probability for each token.
        top_logprobs: Top log probabilities at each position.
        text_offset: Character offset for each token in the text.
    """

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a completion response.

    Attributes:
        text: The generated text.
        index: Index of this choice in the response.
        logprobs: Optional log probability information.
        finish_reason: Reason generation stopped.
    """

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call", "finished"] | None = None


class CompletionResponse(OpenAIBaseModel):
    """Represents a response from the completions endpoint.

    Attributes:
        id: Unique identifier for the completion.
        object: Object type, always "text_completion".
        created: Unix timestamp of when the completion was created.
        model: Model used for the completion.
        choices: List of completion choices.
        usage: Token usage information.
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a streaming completion response chunk.

    Attributes:
        index: Index of this choice in the response.
        text: The incremental text content.
        logprobs: Optional log probability information.
        finish_reason: Reason generation stopped (if final chunk).
    """

    index: int
    text: str  # The delta text content
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call", "finished"] | None = None


class CompletionStreamResponse(OpenAIBaseModel):
    """Represents a streaming response from the completions endpoint.

    Attributes:
        id: Unique identifier for the completion.
        object: Object type, always "text_completion.chunk".
        created: Unix timestamp of when the chunk was created.
        model: Model used for the completion.
        choices: List of completion choices with incremental text.
        usage: Token usage information (often None until final chunk).
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"  # Correct object type for streaming
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]  # Use the new streaming choice model
    usage: UsageInfo | None = None
    # Usage is often None until the final chunk in OAI


class FunctionCall(OpenAIBaseModel):
    """Represents a function call in the OpenAI format.

    Attributes:
        name: Name of the function to call.
        arguments: JSON string containing the function arguments.
    """

    name: str
    arguments: str  # JSON string of arguments


class Function(OpenAIBaseModel):
    """Function definition for OpenAI-compatible function calling.

    Attributes:
        name: The name of the function.
        description: Human-readable description of the function.
        parameters: JSON Schema defining function parameters.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolCall(OpenAIBaseModel):
    """Represents a tool call in responses.

    Attributes:
        id: Unique identifier for this tool call.
        type: Type of tool, always "function".
        function: The function call details.
    """

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    type: str = "function"
    function: FunctionCall


class FunctionCallFormat(str, Enum):
    """Supported function call formats.

    Different models and frameworks use different formats for function calling.
    This enum defines the supported formats that can be used with EasyDeL.

    Attributes:
        OPENAI: OpenAI's standard format with tools/functions.
        JSON_SCHEMA: Direct JSON schema format.
        HERMES: Hermes model function calling format.
        GORILLA: Gorilla model function calling format.
        QWEN: Qwen's special token format (uses special markers).
        NOUS: Nous XML-style format (uses <tool_call> tags).
    """

    OPENAI = "openai"  # OpenAI's format
    JSON_SCHEMA = "json_schema"  # Direct JSON schema
    HERMES = "hermes"  # Hermes function calling format
    GORILLA = "gorilla"  # Gorilla function calling format
    QWEN = "qwen"  # Qwen's special token format
    NOUS = "nous"  # Nous XML-style format


class ExtractedToolCallInformation(OpenAIBaseModel):
    """Information extracted from model output about tool calls.

    Attributes:
        tools_called: Whether any tools were called.
        tool_calls: List of extracted tool calls.
        content: Remaining content after extracting tool calls.
    """

    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None
