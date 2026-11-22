# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Shared helpers for the tool-parser regression tests (no array computation)."""

from __future__ import annotations

from easydel.inference.openai_api_modules import ChatCompletionRequest


class DummyTokenizer:
    """Minimal tokenizer stand-in exposing the vocab/encode/decode surface the parsers use."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = dict(vocab or {})
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        if text not in self._vocab:
            token_id = len(self._vocab) + 1
            self._vocab[text] = token_id
            self._id_to_token[token_id] = text
        return [self._vocab[text]]

    def decode(self, token_ids):
        return "".join(self._id_to_token.get(i, "") for i in token_ids)


def make_request(tools: list[dict] | None = None) -> ChatCompletionRequest:
    payload: dict = {"model": "dummy", "messages": [{"role": "user", "content": "hi"}]}
    if tools:
        payload["tools"] = tools
    return ChatCompletionRequest.model_validate(payload)


def function_tool(name: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": properties}},
    }


def stream(parser, chunks: list[str], request: ChatCompletionRequest):
    """Feed ``chunks`` through ``extract_tool_calls_streaming`` and return every non-None delta."""
    previous = ""
    deltas = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(previous, current, chunk, [], [1] * max(len(chunk), 1), [1], request)
        if delta is not None:
            deltas.append(delta)
        previous = current
    return deltas
