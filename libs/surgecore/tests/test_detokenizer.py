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

"""Host-only behavior of native batched incremental detokenization."""

import typing

import pytest
from surgecore import Detokenizer

pytestmark = pytest.mark.cpu_ok
REPLACEMENT_CHAR = "\ufffd"


def request(
    request_id,
    tokens,
    offset,
    *,
    finished=False,
    text_offset=0,
    skip_special_tokens=True,
    spaces_between_special_tokens=True,
    prompt_context=None,
):
    return {
        "request_id": request_id,
        "tokens_delta": tokens,
        "token_offset": offset,
        "finished": finished,
        "text_offset": text_offset,
        "skip_special_tokens": skip_special_tokens,
        "spaces_between_special_tokens": spaces_between_special_tokens,
        "prompt_context": prompt_context,
    }


class CharacterTokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class Utf8ByteTokenizer:
    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode("utf-8", errors="replace")


class WordPieceTokenizer:
    vocab: typing.ClassVar = {1: "Hello", 2: "##world"}

    def decode(self, tokens, **kwargs):
        return " ".join(self.vocab[token] for token in tokens).replace(" ##", "")


def test_batch_preserves_independent_stream_state_and_bounded_results():
    detokenizer = Detokenizer(CharacterTokenizer(), context_window=0)
    first = detokenizer.decode_batch([request("a", [97], 0), request("b", [120], 0)])
    assert [result.delta_text for result in first] == ["a", "x"]
    assert [result.accumulated_text for result in first] == [None, None]

    second = detokenizer.decode_batch([request("a", [98], 1, text_offset=1), request("b", [121], 1, text_offset=1)])
    assert [result.delta_text for result in second] == ["b", "y"]
    assert [result.accumulated_text for result in second] == [None, None]
    assert [result.last_decoded_index for result in second] == [2, 2]
    assert all(result.detoktook >= 0 for result in second)
    assert detokenizer.num_states == 2

    final = detokenizer.decode_batch(
        [request("a", [99], 2, finished=True, text_offset=2), request("b", [], 2, finished=True, text_offset=2)]
    )
    assert [result.delta_text for result in final] == ["c", ""]
    assert [result.accumulated_text for result in final] == ["abc", "xy"]
    assert detokenizer.num_states == 0


def test_text_offset_mismatch_returns_one_complete_resynchronization():
    detokenizer = Detokenizer(CharacterTokenizer(), context_window=0)
    detokenizer.decode_batch([request("r", [97], 0)])
    result = detokenizer.decode_batch([request("r", [98], 1, text_offset=0)])[0]
    assert result.delta_text == "b"
    assert result.accumulated_text == "ab"
    result = detokenizer.decode_batch([request("r", [99], 2, text_offset=2)])[0]
    assert result.accumulated_text is None


def test_utf8_buffering_releases_invalid_and_incomplete_sequences_like_full_decode():
    data = b"\xf0\xab\x9e" + "\u1e96 ok".encode()
    detokenizer = Detokenizer(Utf8ByteTokenizer(), context_window=0)
    deltas = []
    text = ""
    for index, byte in enumerate(data):
        result = detokenizer.decode_batch([request("r", [byte], index, text_offset=len(text))])[0]
        deltas.append(result.delta_text)
        text += result.delta_text

    assert deltas[:5] == [""] * 5
    assert deltas[5] == REPLACEMENT_CHAR + "\u1e96"
    assert text == data.decode("utf-8", errors="replace")
    final = detokenizer.decode_batch([request("r", [], len(data), finished=True, text_offset=len(text))])[0]
    assert final.accumulated_text == text
    assert final.delta_text == ""


def test_context_and_tokenizer_flags_match_huggingface_decode():
    class RecordingTokenizer(WordPieceTokenizer):
        def __init__(self):
            self.calls = []

        def decode(self, tokens, **kwargs):
            self.calls.append((list(tokens), kwargs))
            return super().decode(tokens, **kwargs)

    tokenizer = RecordingTokenizer()
    detokenizer = Detokenizer(tokenizer, context_window=1)
    result = detokenizer.decode_batch(
        [
            request(
                "r",
                [2],
                0,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                prompt_context=[1],
            )
        ]
    )[0]
    assert result.delta_text == "world"
    assert tokenizer.calls[0][0] == [1, 2]
    assert tokenizer.calls[0][1]["skip_special_tokens"] is False
    assert tokenizer.calls[0][1]["spaces_between_special_tokens"] is False
    assert tokenizer.calls[0][1]["clean_up_tokenization_spaces"] is False


def test_drain_clears_state_and_bad_offsets_are_rejected():
    detokenizer = Detokenizer(CharacterTokenizer(), context_window=0)
    detokenizer.decode_batch([request("r", [97], 0)])
    detokenizer.drain()
    assert detokenizer.num_states == 0
    with pytest.raises(ValueError, match="lost synchronization"):
        detokenizer.decode_batch([request("r", [98], 1)])
    result = detokenizer.decode_batch([request("r", [97, 98], 0)])[0]
    assert result.delta_text == "ab"


def test_terminal_decode_reconciles_suffix_without_replaying_streamed_text():
    detokenizer = Detokenizer(CharacterTokenizer(), context_window=0)
    detokenizer.decode_batch([request("r", [104, 101, 108, 108, 111], 0)])
    result = detokenizer.decode_batch([request("r", [32, 119, 111, 114, 108, 100], 5, finished=True, text_offset=5)])[0]
    assert result.accumulated_text == "hello world"
    assert result.delta_text == " world"
