from .dantics import GenerateAPIRequest
import requests
from typing import Optional, Literal


class EasyClient:
    def __init__(
            self,
            host: str = "0.0.0.0",
            port: Optional[int] = None,
            method: Literal["https", "http"] = "http"
    ):
        self.method = method
        self.host = host
        self.port = port

    def conversation_request(
            self,
            conversation: list[dict[str, str]],
            max_new_tokens: Optional[int] = None,
            greedy: bool = False,
    ):
        port = ":" + str(self.port) if self.port is not None else ""
        requests.post(
            url=f"{self.method}://{self.host}{port}",
            data={
                "conversation": conversation,
                "max_new_tokens": max_new_tokens,
                "greedy": greedy
            },
        )
