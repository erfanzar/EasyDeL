from typing import Optional

from pydantic import BaseModel


class InstructRequest(BaseModel):
    instruction: str
    system: Optional[str] = None
    temperature: Optional[float] = None
    greedy: Optional[bool] = False


class ChatRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    history: Optional[list[list[str]]] = None
    temperature: Optional[float] = None
    greedy: Optional[bool] = False


class ModelOutput(BaseModel):
    response: str
    tokens_used: Optional[int] = None
    model_name: Optional[str] = None
    generation_time: Optional[float] = None

