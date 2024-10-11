from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict
import uuid
import time


class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[Dict[str, str]]
	stream: Optional[bool] = False


class ChatMessage(BaseModel):
	role: str
	content: str


class UsageInfo(BaseModel):
	prompt_tokens: int = 0
	completion_tokens: Optional[int] = 0
	total_tokens: int = 0


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


class DeltaMessage(BaseModel):
	role: Optional[str] = None
	content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
	index: int
	delta: DeltaMessage
	finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
	id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
	object: str = "chat.completion.chunk"
	created: int = Field(default_factory=lambda: int(time.time()))
	model: str
	choices: List[ChatCompletionStreamResponseChoice]
