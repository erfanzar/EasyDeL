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

from easydel.workers.esurge.pipeline.worker_main import FastIncrementalDecoder, _compute_suffix_delta

REPLACEMENT_CHAR = "\ufffd"


class DummyByteTokenizer:
    """Tokenizer that simulates incomplete byte sequences."""

    def decode(self, token_ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        _ = skip_special_tokens
        _ = clean_up_tokenization_spaces
        out = []
        i = 0
        while i < len(token_ids):
            token = token_ids[i]
            if token == 100:
                if i + 1 < len(token_ids) and token_ids[i + 1] == 101:
                    out.append("A")
                    i += 2
                    continue
                out.append(REPLACEMENT_CHAR)
                i += 1
                continue
            if token == 101:
                out.append("B")
                i += 1
                continue
            out.append(f"<{token}>")
            i += 1
        return "".join(out)


class DummyWordPieceTokenizer:
    """Tokenizer that emulates WordPiece joining behavior."""

    _vocab = {1: "Hello", 2: "##world"}  # noqa: RUF012

    def decode(self, token_ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        _ = skip_special_tokens
        _ = clean_up_tokenization_spaces
        tokens = [self._vocab[token] for token in token_ids]
        return " ".join(tokens).replace(" ##", "")


class RecordingSpecialTokenTokenizer:
    def __init__(self):
        self.calls = []

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        self.calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        return "".join(f"<{token}>" for token in token_ids)


class Utf8ByteTokenizer:
    """One token per byte, decoded the way byte-level BPE tokenizers decode: UTF-8 with replacement."""

    def decode(self, token_ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        _ = skip_special_tokens
        _ = clean_up_tokenization_spaces
        return bytes(token_ids).decode("utf-8", errors="replace")


def test_incremental_decoder_buffers_until_utf8_valid():
    tokenizer = DummyByteTokenizer()
    decoder = FastIncrementalDecoder(tokenizer, context_window=0)
    buffered: list[int] = []

    delta, buffered, has_buffer = decoder.decode(
        [100],
        "",
        buffered,
        skip_special_tokens=True,
        context_tokens=[],
    )
    assert delta == ""
    assert buffered == [100]
    assert has_buffer is True

    delta, buffered, has_buffer = decoder.decode(
        [101],
        "",
        buffered,
        skip_special_tokens=True,
        context_tokens=[],
    )
    assert delta == "A"
    assert buffered == []
    assert has_buffer is False


def test_incremental_decoder_streams_past_bytes_that_never_form_a_character():
    # F0 AB 9E opens a 4-byte character, and the next byte starts a new character instead of completing it.
    data = b"\xf0\xab\x9e" + "\u1e96 ok".encode()
    decoder = FastIncrementalDecoder(Utf8ByteTokenizer(), context_window=0)
    buffered: list[int] = []
    deltas = []
    for byte in data:
        delta, buffered, _ = decoder.decode([byte], "", buffered, skip_special_tokens=True, context_tokens=[])
        deltas.append(delta)

    # Held while the text ends with an incomplete character, released once U+1E96 completes.
    assert deltas[:5] == [""] * 5
    assert deltas[5] == REPLACEMENT_CHAR + "\u1e96"
    assert "".join(deltas) == data.decode("utf-8", errors="replace")
    assert buffered == []


def test_incremental_decoder_forwards_spaces_between_special_tokens_flag():
    tokenizer = RecordingSpecialTokenTokenizer()
    decoder = FastIncrementalDecoder(tokenizer, context_window=0)
    buffered: list[int] = []

    delta, buffered, has_buffer = decoder.decode(
        [42],
        "",
        buffered,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        context_tokens=[],
    )

    assert delta == "<42>"
    assert buffered == []
    assert has_buffer is False
    assert tokenizer.calls
    assert tokenizer.calls[0]["spaces_between_special_tokens"] is False
    assert tokenizer.calls[0]["skip_special_tokens"] is False


def test_incremental_decoder_uses_context_for_wordpiece():
    tokenizer = DummyWordPieceTokenizer()
    decoder = FastIncrementalDecoder(tokenizer, context_window=1)
    buffered: list[int] = []

    delta, buffered, has_buffer = decoder.decode(
        [2],
        "Hello",
        buffered,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        context_tokens=[1],
    )
    assert delta == "world"
    assert buffered == []
    assert has_buffer is False


def test_compute_suffix_delta_prefix_path():
    assert _compute_suffix_delta("hello world", "hello ") == "world"


def test_compute_suffix_delta_overlap_path():
    assert _compute_suffix_delta("world!", "hello world") == "!"


def test_compute_suffix_delta_no_overlap_avoids_replay():
    assert _compute_suffix_delta("fresh text", "old output") == ""
