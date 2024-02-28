from typing import Optional, List, Literal

from pydantic import BaseModel


class ConversationItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class GenerateAPIRequest(BaseModel):
    conversation: List[ConversationItem]
    max_new_tokens: Optional[int] = None
    greedy: Optional[bool] = False


class ModelOutput(BaseModel):
    response: str
    tokens_used: Optional[int] = None
    model_name: Optional[str] = None
    generation_time: Optional[float] = None
