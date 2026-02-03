from easydel.workers.esurge.pipeline.worker_main import FastIncrementalDecoder

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


def test_incremental_decoder_uses_context_for_wordpiece():
    tokenizer = DummyWordPieceTokenizer()
    decoder = FastIncrementalDecoder(tokenizer, context_window=1)
    buffered: list[int] = []

    delta, buffered, has_buffer = decoder.decode(
        [2],
        "Hello",
        buffered,
        skip_special_tokens=True,
        context_tokens=[1],
    )
    assert delta == "world"
    assert buffered == []
    assert has_buffer is False
