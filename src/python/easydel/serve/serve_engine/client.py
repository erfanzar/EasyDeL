import json
from typing import Optional, Literal, Generator
from dataclasses import dataclass


@dataclass
class StreamClientResponse:
    response: str
    num_token_generated: int
    greedy: bool
    model_prompt: str
    generation_duration: float
    tokens_pre_second: float
    done: bool


class EasyClient:
    def __init__(
            self,
            host: str = "0.0.0.0",
            port: Optional[int] = None,
    ):
        host = host.replace("https://", "").replace("http://", "").replace("ws", "")
        self.host = host
        self.port = port

    def generate(
            self,
            conversation: list[dict[str, str]],
            max_new_tokens: Optional[int] = None,
            greedy: bool = False,
            version: Literal["v1"] = "v1"
    ) -> Generator[StreamClientResponse, None, None]:
        from websocket import create_connection, WebSocket
        client: WebSocket = create_connection(f"ws://{self.host}:{self.port}/stream/{version}/conversation")
        data = {
            "conversation": conversation,
            "max_new_tokens": max_new_tokens,
            "greedy": greedy
        }
        client.send(json.dumps(data))
        while True:
            data = client.recv()
            if data != "":
                data = StreamClientResponse(**json.loads(data))
                yield data
                if data.done:
                    break
