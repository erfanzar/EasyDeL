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

import time
import typing as tp
import uuid

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
	"""Represents a single message in a chat conversation."""

	role: str
	content: tp.Union[str, tp.List[tp.Mapping[str, str]]]
	# Supports text and multimodal content
	name: tp.Optional[str] = None
	function_call: tp.Optional[tp.Dict[str, tp.Any]] = None


class DeltaMessage(BaseModel):
	"""Represents a change (delta) in a chat message, used in streaming responses."""

	role: tp.Optional[str] = None
	content: tp.Optional[tp.Union[str, tp.List[tp.Mapping[str, str]]]] = None
	function_call: tp.Optional[tp.Dict[str, tp.Any]] = None


class UsageInfo(BaseModel):
	"""Provides information about token usage and processing time for a request."""

	prompt_tokens: int = 0
	completion_tokens: tp.Optional[int] = 0
	total_tokens: int = 0
	tokens_per_second: float = 0
	processing_time: float = 0.0


class FunctionDefinition(BaseModel):
	"""Defines a function that can be called by the model."""

	name: str
	description: tp.Optional[str] = None
	parameters: tp.Dict[str, tp.Any] = Field(default_factory=dict)
	required: tp.Optional[tp.List[str]] = None


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
	messages: tp.List[ChatMessage]

	# Sampling parameters (mirroring OpenAI)
	max_tokens: int = 16
	presence_penalty: float = 0.0
	frequency_penalty: float = 0.0
	repetition_penalty: float = 1.0
	temperature: float = 0.7
	top_p: float = 1.0
	top_k: int = 0
	min_p: float = 0.0
	suppress_tokens: tp.List[int] = Field(default_factory=list)
	# Added for potential EasyDeL support

	# OpenAI native parameters (some may be ignored by vInference)
	functions: tp.Optional[tp.List[FunctionDefinition]] = None
	function_call: tp.Optional[tp.Union[str, tp.Dict[str, tp.Any]]] = None
	tools: tp.Optional[tp.List[ToolDefinition]] = None
	tool_choice: tp.Optional[tp.Union[str, tp.Dict[str, tp.Any]]] = None
	n: tp.Optional[int] = 1  # Ignored by vInference (always returns 1 choice)
	stream: tp.Optional[bool] = False
	stop: tp.Optional[tp.Union[str, tp.List[str]]] = None
	logit_bias: tp.Optional[tp.Dict[str, float]] = None  # Ignored by EasyDeL
	user: tp.Optional[str] = None  # Ignored by EasyDeL


class ChatCompletionResponseChoice(BaseModel):
	"""Represents a single choice within a non-streaming chat completion response."""

	index: int
	message: ChatMessage
	finish_reason: tp.Optional[tp.Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
	"""Represents a non-streaming response from the chat completion endpoint."""

	id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
	object: str = "chat.completion"
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: tp.List[ChatCompletionResponseChoice]
	usage: UsageInfo


class ChatCompletionStreamResponseChoice(BaseModel):
	"""Represents a single choice within a streaming chat completion response chunk."""

	index: int
	delta: DeltaMessage
	finish_reason: tp.Optional[tp.Literal["stop", "length", "function_call"]] = None


class ChatCompletionStreamResponse(BaseModel):
	"""Represents a single chunk in a streaming response from the chat completion endpoint."""

	id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
	object: str = "chat.completion.chunk"
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: tp.List[ChatCompletionStreamResponseChoice]
	usage: UsageInfo  # Usage info might be included in chunks, often zero until the end


class CountTokenRequest(BaseModel):
	"""Represents a request to the token counting endpoint."""

	model: str
	conversation: tp.Union[
		str, tp.List[ChatMessage]
	]  # Can count tokens for a string or a list of messages


class CompletionRequest(BaseModel):
	"""
	Represents a request to the completions endpoint.
	Mirrors the OpenAI Completion request structure.
	"""

	model: str
	prompt: tp.Union[str, tp.List[str]]
	max_tokens: int = 16
	presence_penalty: float = 0.0
	frequency_penalty: float = 0.0
	repetition_penalty: float = 1.0
	temperature: float = 0.7
	top_p: float = 1.0
	top_k: int = 0
	min_p: float = 0.0
	suppress_tokens: tp.List[int] = Field(default_factory=list)
	n: tp.Optional[int] = 1
	stream: tp.Optional[bool] = False
	stop: tp.Optional[tp.Union[str, tp.List[str]]] = None
	logit_bias: tp.Optional[tp.Dict[str, float]] = None
	user: tp.Optional[str] = None


class CompletionLogprobs(BaseModel):
	"""Log probabilities for token generation."""

	tokens: tp.List[str]
	token_logprobs: tp.List[float]
	top_logprobs: tp.Optional[tp.List[tp.Dict[str, float]]] = None
	text_offset: tp.Optional[tp.List[int]] = None


class CompletionResponseChoice(BaseModel):
	"""Represents a single choice within a completion response."""

	text: str
	index: int
	logprobs: tp.Optional[CompletionLogprobs] = None
	finish_reason: tp.Optional[tp.Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
	"""Represents a response from the completions endpoint."""

	id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
	object: str = "text_completion"
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: tp.List[CompletionResponseChoice]
	usage: UsageInfo


# New model for streaming completion choices (OAI compatible)
class CompletionStreamResponseChoice(BaseModel):
	"""Represents a single choice within a streaming completion response chunk."""

	index: int
	text: str  # The delta text content
	logprobs: tp.Optional[CompletionLogprobs] = (
		None  # Logprobs are usually None in streaming chunks
	)
	finish_reason: tp.Optional[tp.Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
	"""Represents a streaming response from the completions endpoint."""

	id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
	object: str = "text_completion.chunk"  # Correct object type for streaming
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: tp.List[CompletionStreamResponseChoice]  # Use the new streaming choice model
	usage: tp.Optional[UsageInfo] = None
	# Usage is often None until the final chunk in OAI
