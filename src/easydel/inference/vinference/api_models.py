import time
import uuid
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
	role: str
	content: str


class UsageInfo(BaseModel):
	prompt_tokens: int = 0
	completion_tokens: Optional[int] = 0
	total_tokens: int = 0
	tps: float = 0
	processing_time: float = 0.0


class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[ChatMessage]
	stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
	response: str
	finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
	id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
	object: str = "chat.completion"
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: List[ChatCompletionResponseChoice]
	usage: UsageInfo


class ChatCompletionStreamResponseChoice(BaseModel):
	response: str


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
