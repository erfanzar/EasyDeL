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

import time
import uuid
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
	role: str
	content: str


class DeltaMessage(BaseModel):
	role: Optional[str] = None
	content: Optional[str] = None


class UsageInfo(BaseModel):
	prompt_tokens: int = 0
	completion_tokens: Optional[int] = 0
	total_tokens: int = 0
	tokens_pre_second: float = 0
	processing_time: float = 0.0
	first_iter_flops: float = 0.0
	iter_flops: float = 0.0


class ChatCompletionRequest(BaseModel):
	# The openai api native parameters
	model: str
	messages: List[ChatMessage]
	function_call: Optional[str] = "none"  # Ignored by EasyDeL
	temperature: Optional[float] = 1  # Ignored by EasyDeL
	top_p: Optional[float] = 1.0  # Ignored by EasyDeL
	n: Optional[int] = 1  # Ignored by EasyDeL
	stream: Optional[bool] = False
	stop: Optional[Union[str, List[str]]] = None  # Ignored by EasyDeL
	max_tokens: Optional[int] = 16
	presence_penalty: Optional[float] = 0.0  # Ignored by EasyDeL
	frequency_penalty: Optional[float] = 0.0  # Ignored by EasyDeL
	logit_bias: Optional[Dict[str, float]] = None  # Ignored by EasyDeL
	user: Optional[str] = None  # Ignored by EasyDeL


class ChatCompletionResponseChoice(BaseModel):
	index: int
	message: ChatMessage
	finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
	id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
	object: str = "chat.completion"
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: List[ChatCompletionResponseChoice]
	usage: UsageInfo


class ChatCompletionStreamResponseChoice(BaseModel):
	index: int
	delta: DeltaMessage
	finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionStreamResponse(BaseModel):
	id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
	object: str = "chat.completion.chunk"
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: List[ChatCompletionStreamResponseChoice]
	usage: UsageInfo


class CountTokenRequest(BaseModel):
	model: str
	conversation: Union[str, ChatMessage]
