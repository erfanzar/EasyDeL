from dataclasses import dataclass
from ._streamer import TextIteratorStreamer


@dataclass
class GenerationParams:
    max_length: int = 4096
    max_ffd_tokens: int = 64
    max_new_tokens: int = 4096
    top_p: float = 0.75
    top_k: int = 50
    temperature: float = 0.7
    do_sample: bool = True
    streamer: TextIteratorStreamer = None

    def __post_init__(self):
        assert self.max_length > self.max_ffd_tokens, (
            "You can not use max ffd tokens value higher than max length itself"
        )
