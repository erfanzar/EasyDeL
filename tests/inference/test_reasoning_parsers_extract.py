import pytest

from easydel.inference.reasoning.abstract_reasoning import ReasoningParserManager
from easydel.inference.reasoning.parsers import (
    DeepSeekR1ReasoningParser,
    DeepSeekV3ReasoningParser,
    Ernie45ReasoningParser,
    GptOssReasoningParser,
    GraniteReasoningParser,
    HunyuanA13BReasoningParser,
    IdentityReasoningParser,
    MiniMaxM2AppendThinkReasoningParser,
    MiniMaxM2ReasoningParser,
    MistralReasoningParser,
    Olmo3ReasoningParser,
    Qwen3ReasoningParser,
    SeedOSSReasoningParser,
    Step3ReasoningParser,
    Step3p5ReasoningParser,
)


class _DummyTokenizer:
    def __init__(self, vocab: dict[str, int], chat_template: str = ""):
        self._vocab = dict(vocab)
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self.chat_template = chat_template

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._vocab:
            return [self._vocab[text]]
        token_id = len(self._vocab) + 1
        self._vocab[text] = token_id
        self._id_to_token[token_id] = text
        return [token_id]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self._id_to_token.get(i, "") for i in token_ids)


@pytest.fixture()
def dummy_tokenizer():
    vocab = {
        "<think>": 1,
        "</think>": 2,
        "[THINK]": 3,
        "[/THINK]": 4,
        "<|channel|>": 5,
        "<|message|>": 6,
    }
    return _DummyTokenizer(vocab)


@pytest.fixture()
def thinking_tokenizer():
    """Tokenizer with chat_template mentioning 'thinking' for DeepSeekV3."""
    vocab = {"<think>": 1, "</think>": 2}
    return _DummyTokenizer(vocab, chat_template="some template with thinking enabled")


@pytest.fixture()
def plain_tokenizer():
    """Tokenizer WITHOUT thinking in chat_template for DeepSeekV3."""
    vocab = {"<think>": 1, "</think>": 2}
    return _DummyTokenizer(vocab, chat_template="plain template")


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_reasoning_parser_manager_includes_parsers():
    for name in [
        "deepseek_r1",
        "deepseek-r1",
        "deepseek_v3",
        "glm45",
        "holo2",
        "kimi_k2",
        "qwen3",
        "qwen3_reasoning",
        "granite",
        "mistral",
        "minimax_m2",
        "minimax_m2_append_think",
        "olmo3",
        "step3",
        "step3p5",
        "step3.5",
        "hunyuan_a13b",
        "ernie45",
        "seed_oss",
        "openai_gptoss",
        "gptoss",
        "identity",
        "none",
        "passthrough",
    ]:
        cls = ReasoningParserManager.get_reasoning_parser(name)
        assert cls is not None


def test_reasoning_parser_manager_raises_for_unknown():
    with pytest.raises(KeyError, match="not found"):
        ReasoningParserManager.get_reasoning_parser("nonexistent_parser_xyz")


# ---------------------------------------------------------------------------
# Batch extraction tests
# ---------------------------------------------------------------------------


def test_deepseek_r1_extract_reasoning(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    output = "<think>I need to figure this out</think>The answer is 42."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "I need to figure this out"
    assert content == "The answer is 42."


def test_deepseek_r1_no_tags(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    output = "The answer is 42."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "The answer is 42."


def test_deepseek_r1_empty_reasoning(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    output = "<think></think>The answer is 42."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "The answer is 42."


def test_deepseek_r1_only_start_tag(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    output = "<think>still thinking about this..."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "still thinking about this..."
    assert content is None


def test_deepseek_v3_with_thinking_template(thinking_tokenizer):
    parser = DeepSeekV3ReasoningParser(thinking_tokenizer)
    output = "<think>reasoning here</think>answer here"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "reasoning here"
    assert content == "answer here"


def test_deepseek_v3_without_thinking_template(plain_tokenizer):
    parser = DeepSeekV3ReasoningParser(plain_tokenizer)
    output = "<think>reasoning here</think>answer here"
    reasoning, content = parser.extract_reasoning(output)
    # Identity parser: no reasoning extraction
    assert reasoning is None
    assert content == output


def test_qwen3_extract_reasoning(dummy_tokenizer):
    parser = Qwen3ReasoningParser(dummy_tokenizer)
    output = "<think>step by step</think>final answer"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "step by step"
    assert content == "final answer"


def test_qwen3_strict_no_start_tag(dummy_tokenizer):
    parser = Qwen3ReasoningParser(dummy_tokenizer)
    output = "no tags here, just content"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "no tags here, just content"


def test_qwen3_strict_only_end_tag(dummy_tokenizer):
    """Qwen3 strict mode: missing start tag means all is content."""
    parser = Qwen3ReasoningParser(dummy_tokenizer)
    output = "some text</think>and more"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == output


def test_qwen3_strict_only_start_tag(dummy_tokenizer):
    """Qwen3 strict mode: missing end tag means all is content."""
    parser = Qwen3ReasoningParser(dummy_tokenizer)
    output = "<think>thinking but no end"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == output


def test_mistral_extract_reasoning(dummy_tokenizer):
    parser = MistralReasoningParser(dummy_tokenizer)
    output = "[THINK]analyzing the problem[/THINK]Here is the answer."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "analyzing the problem"
    assert content == "Here is the answer."


def test_mistral_no_tags(dummy_tokenizer):
    parser = MistralReasoningParser(dummy_tokenizer)
    output = "Just a plain answer."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "Just a plain answer."


def test_granite_extract_reasoning(dummy_tokenizer):
    parser = GraniteReasoningParser(dummy_tokenizer)
    output = "Here's my thought process:\nI think about it.\nHere's my response:\nThe result."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "I think about it."
    assert content == "The result."


def test_granite_no_delimiters(dummy_tokenizer):
    parser = GraniteReasoningParser(dummy_tokenizer)
    output = "Just plain text."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "Just plain text."


def test_granite_alternative_delimiters(dummy_tokenizer):
    parser = GraniteReasoningParser(dummy_tokenizer)
    output = "Here is my thought process:\nThinking hard.\nHere is my response:\nDone."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "Thinking hard."
    assert content == "Done."


def test_identity_extract_reasoning(dummy_tokenizer):
    parser = IdentityReasoningParser(dummy_tokenizer)
    output = "<think>some text</think>more text"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == output


def test_identity_is_reasoning_end(dummy_tokenizer):
    parser = IdentityReasoningParser(dummy_tokenizer)
    assert parser.is_reasoning_end([1, 2, 3]) is True


def test_minimax_m2_extract_reasoning(dummy_tokenizer):
    parser = MiniMaxM2ReasoningParser(dummy_tokenizer)
    output = "I think about this carefully</think>The answer is 42."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "I think about this carefully"
    assert content == "The answer is 42."


def test_minimax_m2_with_start_tag(dummy_tokenizer):
    parser = MiniMaxM2ReasoningParser(dummy_tokenizer)
    output = "<think>I think about this</think>The answer."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "I think about this"
    assert content == "The answer."


def test_minimax_m2_no_end_tag(dummy_tokenizer):
    parser = MiniMaxM2ReasoningParser(dummy_tokenizer)
    output = "Just content, no end token."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "Just content, no end token."


def test_minimax_m2_append_think(dummy_tokenizer):
    parser = MiniMaxM2AppendThinkReasoningParser(dummy_tokenizer)
    # Without start tag — parser prepends it
    output = "reasoning content</think>answer"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "reasoning content"
    assert content == "answer"


def test_step3_extract_reasoning(dummy_tokenizer):
    parser = Step3ReasoningParser(dummy_tokenizer)
    output = "thinking step by step</think>The final answer."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "thinking step by step"
    assert content == "The final answer."


def test_step3_no_end_tag(dummy_tokenizer):
    parser = Step3ReasoningParser(dummy_tokenizer)
    output = "Just content, no end."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "Just content, no end."


def test_step3_with_start_tag(dummy_tokenizer):
    parser = Step3ReasoningParser(dummy_tokenizer)
    output = "<think>thinking</think>answer"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "thinking"
    assert content == "answer"


def test_step3p5_extract_reasoning(dummy_tokenizer):
    parser = Step3p5ReasoningParser(dummy_tokenizer)
    output = "reasoning here</think>the answer"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "reasoning here"
    assert content == "the answer"


def test_hunyuan_a13b_extract_reasoning(dummy_tokenizer):
    parser = HunyuanA13BReasoningParser(dummy_tokenizer)
    output = "<think>deep thought</think>42"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "deep thought"
    assert content == "42"


def test_ernie45_extract_reasoning(dummy_tokenizer):
    parser = Ernie45ReasoningParser(dummy_tokenizer)
    output = "<think>analysis</think>result"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "analysis"
    assert content == "result"


def test_seedoss_extract_reasoning(dummy_tokenizer):
    parser = SeedOSSReasoningParser(dummy_tokenizer)
    output = "<think>seed thinking</think>seed answer"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "seed thinking"
    assert content == "seed answer"


def test_olmo3_extract_reasoning(dummy_tokenizer):
    parser = Olmo3ReasoningParser(dummy_tokenizer)
    output = "<think>olmo reasoning</think>olmo response"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "olmo reasoning"
    assert content == "olmo response"


def test_gptoss_extract_reasoning(dummy_tokenizer):
    parser = GptOssReasoningParser(dummy_tokenizer)
    output = "<|channel|>analysis of the question<|message|>the answer"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "analysis of the question"
    assert content == "the answer"


def test_gptoss_no_tags(dummy_tokenizer):
    parser = GptOssReasoningParser(dummy_tokenizer)
    output = "plain text"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == "plain text"


def test_gptoss_only_channel_tag(dummy_tokenizer):
    parser = GptOssReasoningParser(dummy_tokenizer)
    output = "<|channel|>analysis only"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "analysis only"
    assert content is None


# ---------------------------------------------------------------------------
# Streaming extraction tests
# ---------------------------------------------------------------------------


def test_deepseek_r1_streaming_reasoning_then_content(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)

    # Chunk 1: start tag + reasoning
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>thinking",
        delta_text="<think>thinking",
        previous_token_ids=[],
        current_token_ids=[1, 100],
        delta_token_ids=[1, 100],
    )
    assert delta is not None
    assert delta.reasoning_content == "thinking"
    assert delta.content is None

    # Chunk 2: more reasoning
    delta = parser.extract_reasoning_streaming(
        previous_text="<think>thinking",
        current_text="<think>thinking more",
        delta_text=" more",
        previous_token_ids=[1, 100],
        current_token_ids=[1, 100, 101],
        delta_token_ids=[101],
    )
    assert delta is not None
    assert delta.reasoning_content == " more"

    # Chunk 3: end tag + content
    delta = parser.extract_reasoning_streaming(
        previous_text="<think>thinking more",
        current_text="<think>thinking more</think>answer",
        delta_text="</think>answer",
        previous_token_ids=[1, 100, 101],
        current_token_ids=[1, 100, 101, 2, 102],
        delta_token_ids=[2, 102],
    )
    assert delta is not None
    assert delta.content == "answer"

    # Chunk 4: more content after end tag
    delta = parser.extract_reasoning_streaming(
        previous_text="<think>thinking more</think>answer",
        current_text="<think>thinking more</think>answer here",
        delta_text=" here",
        previous_token_ids=[1, 100, 101, 2, 102],
        current_token_ids=[1, 100, 101, 2, 102, 103],
        delta_token_ids=[103],
    )
    assert delta is not None
    assert delta.content == " here"
    assert delta.reasoning_content is None


def test_identity_streaming(dummy_tokenizer):
    parser = IdentityReasoningParser(dummy_tokenizer)
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="hello",
        delta_text="hello",
        previous_token_ids=[],
        current_token_ids=[100],
        delta_token_ids=[100],
    )
    assert delta is not None
    assert delta.content == "hello"
    assert delta.reasoning_content is None


def test_identity_streaming_empty_delta(dummy_tokenizer):
    parser = IdentityReasoningParser(dummy_tokenizer)
    delta = parser.extract_reasoning_streaming(
        previous_text="hello",
        current_text="hello",
        delta_text="",
        previous_token_ids=[100],
        current_token_ids=[100],
        delta_token_ids=[],
    )
    assert delta is None


def test_mistral_streaming(dummy_tokenizer):
    parser = MistralReasoningParser(dummy_tokenizer)

    # Chunk 1: start tag + reasoning text
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="[THINK]analyzing",
        delta_text="[THINK]analyzing",
        previous_token_ids=[],
        current_token_ids=[3, 100],
        delta_token_ids=[3, 100],
    )
    assert delta is not None
    assert delta.reasoning_content == "analyzing"

    # Chunk 2: end tag + content
    delta = parser.extract_reasoning_streaming(
        previous_text="[THINK]analyzing",
        current_text="[THINK]analyzing[/THINK]done",
        delta_text="[/THINK]done",
        previous_token_ids=[3, 100],
        current_token_ids=[3, 100, 4, 101],
        delta_token_ids=[4, 101],
    )
    assert delta is not None
    assert delta.content == "done"


def test_minimax_m2_streaming(dummy_tokenizer):
    parser = MiniMaxM2ReasoningParser(dummy_tokenizer)

    # Chunk 1: reasoning without start tag
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="I think so",
        delta_text="I think so",
        previous_token_ids=[],
        current_token_ids=[100],
        delta_token_ids=[100],
    )
    assert delta is not None
    assert delta.reasoning_content == "I think so"

    # Chunk 2: end tag arrives
    delta = parser.extract_reasoning_streaming(
        previous_text="I think so",
        current_text="I think so</think>The result.",
        delta_text="</think>The result.",
        previous_token_ids=[100],
        current_token_ids=[100, 2, 101],
        delta_token_ids=[2, 101],
    )
    assert delta is not None
    assert delta.content == "The result."


def test_gptoss_streaming(dummy_tokenizer):
    parser = GptOssReasoningParser(dummy_tokenizer)

    # Chunk 1: channel tag + reasoning
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<|channel|>analysis",
        delta_text="<|channel|>analysis",
        previous_token_ids=[],
        current_token_ids=[5, 100],
        delta_token_ids=[5, 100],
    )
    assert delta is not None
    assert delta.reasoning_content == "analysis"

    # Chunk 2: message tag + content
    delta = parser.extract_reasoning_streaming(
        previous_text="<|channel|>analysis",
        current_text="<|channel|>analysis<|message|>answer",
        delta_text="<|message|>answer",
        previous_token_ids=[5, 100],
        current_token_ids=[5, 100, 6, 101],
        delta_token_ids=[6, 101],
    )
    assert delta is not None
    assert delta.content == "answer"


def test_step3_streaming(dummy_tokenizer):
    parser = Step3ReasoningParser(dummy_tokenizer)

    # Chunk 1: reasoning (no start tag)
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="thinking",
        delta_text="thinking",
        previous_token_ids=[],
        current_token_ids=[100],
        delta_token_ids=[100],
    )
    assert delta is not None
    assert delta.reasoning_content == "thinking"

    # Chunk 2: end tag + content
    delta = parser.extract_reasoning_streaming(
        previous_text="thinking",
        current_text="thinking</think>answer",
        delta_text="</think>answer",
        previous_token_ids=[100],
        current_token_ids=[100, 2, 101],
        delta_token_ids=[2, 101],
    )
    assert delta is not None
    assert delta.content == "answer"


# ---------------------------------------------------------------------------
# BaseThinkingReasoningParser is_reasoning_end / extract_content_ids
# ---------------------------------------------------------------------------


def test_base_is_reasoning_end(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    # Token ID 2 = </think>
    assert parser.is_reasoning_end([1, 100, 2, 101]) is True
    assert parser.is_reasoning_end([1, 100, 101]) is False


def test_base_extract_content_ids(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    # Token ID 2 = </think>; content starts after it
    content_ids = parser.extract_content_ids([1, 100, 2, 101, 102])
    assert content_ids == [101, 102]


def test_base_extract_content_ids_no_end_token(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    content_ids = parser.extract_content_ids([1, 100, 101])
    assert content_ids == [1, 100, 101]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_both_tags_in_single_delta(dummy_tokenizer):
    """Start and end tag both appear in one delta chunk."""
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>quick thought</think>answer",
        delta_text="<think>quick thought</think>answer",
        previous_token_ids=[],
        current_token_ids=[1, 100, 2, 101],
        delta_token_ids=[1, 100, 2, 101],
    )
    assert delta is not None
    assert delta.reasoning_content == "quick thought"
    assert delta.content == "answer"


def test_no_content_after_end_tag(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    output = "<think>thinking</think>"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "thinking"
    assert content is None


def test_content_before_start_tag(dummy_tokenizer):
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    output = "prefix <think>reasoning</think>response"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "reasoning"
    assert content is not None
    assert "prefix" in content
    assert "response" in content


# ---------------------------------------------------------------------------
# assume_reasoning tests (prompt has <think>, model output is asymmetric)
# ---------------------------------------------------------------------------


def test_deepseek_r1_assume_reasoning_batch(dummy_tokenizer):
    """Batch: when assume_reasoning is set, end-only output is parsed correctly."""
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    parser.assume_reasoning = True
    output = "I need to figure this out</think>The answer is 42."
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "I need to figure this out"
    assert content == "The answer is 42."


def test_deepseek_r1_assume_reasoning_streaming(dummy_tokenizer):
    """Streaming: with assume_reasoning, text before </think> is reasoning."""
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    parser.assume_reasoning = True

    # Chunk 1: reasoning text (no <think> tag in output)
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="thinking",
        delta_text="thinking",
        previous_token_ids=[],
        current_token_ids=[100],
        delta_token_ids=[100],
    )
    assert delta is not None
    assert delta.reasoning_content == "thinking"
    assert delta.content is None

    # Chunk 2: more reasoning
    delta = parser.extract_reasoning_streaming(
        previous_text="thinking",
        current_text="thinking deeply",
        delta_text=" deeply",
        previous_token_ids=[100],
        current_token_ids=[100, 101],
        delta_token_ids=[101],
    )
    assert delta is not None
    assert delta.reasoning_content == " deeply"
    assert delta.content is None

    # Chunk 3: </think> + content
    delta = parser.extract_reasoning_streaming(
        previous_text="thinking deeply",
        current_text="thinking deeply</think>The answer.",
        delta_text="</think>The answer.",
        previous_token_ids=[100, 101],
        current_token_ids=[100, 101, 2, 102],
        delta_token_ids=[2, 102],
    )
    assert delta is not None
    assert delta.content == "The answer."

    # Chunk 4: more content after </think>
    delta = parser.extract_reasoning_streaming(
        previous_text="thinking deeply</think>The answer.",
        current_text="thinking deeply</think>The answer. Done!",
        delta_text=" Done!",
        previous_token_ids=[100, 101, 2, 102],
        current_token_ids=[100, 101, 2, 102, 103],
        delta_token_ids=[103],
    )
    assert delta is not None
    assert delta.content == " Done!"
    assert delta.reasoning_content is None


def test_deepseek_r1_no_assume_reasoning_streaming_is_content(dummy_tokenizer):
    """Without assume_reasoning, text without tags is treated as content."""
    parser = DeepSeekR1ReasoningParser(dummy_tokenizer)
    # Default: assume_reasoning = False

    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="just content",
        delta_text="just content",
        previous_token_ids=[],
        current_token_ids=[100],
        delta_token_ids=[100],
    )
    assert delta is not None
    assert delta.content == "just content"
    assert delta.reasoning_content is None


def test_qwen3_assume_reasoning_batch(dummy_tokenizer):
    """Qwen3 strict mode relaxed when assume_reasoning is set."""
    parser = Qwen3ReasoningParser(dummy_tokenizer)
    parser.assume_reasoning = True
    output = "step by step reasoning</think>final answer"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning == "step by step reasoning"
    assert content == "final answer"


def test_qwen3_assume_reasoning_streaming(dummy_tokenizer):
    """Qwen3 streaming works with assume_reasoning (no <think> in output)."""
    parser = Qwen3ReasoningParser(dummy_tokenizer)
    parser.assume_reasoning = True

    # Chunk 1: reasoning (no <think>)
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="reasoning",
        delta_text="reasoning",
        previous_token_ids=[],
        current_token_ids=[100],
        delta_token_ids=[100],
    )
    assert delta is not None
    assert delta.reasoning_content == "reasoning"
    assert delta.content is None

    # Chunk 2: end tag + content
    delta = parser.extract_reasoning_streaming(
        previous_text="reasoning",
        current_text="reasoning</think>answer",
        delta_text="</think>answer",
        previous_token_ids=[100],
        current_token_ids=[100, 2, 101],
        delta_token_ids=[2, 101],
    )
    assert delta is not None
    assert delta.content == "answer"


def test_qwen3_no_assume_strict_still_works(dummy_tokenizer):
    """Without assume_reasoning, Qwen3 strict mode still rejects end-only output."""
    parser = Qwen3ReasoningParser(dummy_tokenizer)
    output = "text</think>more"
    reasoning, content = parser.extract_reasoning(output)
    assert reasoning is None
    assert content == output


def test_granite_streaming_thought_then_response(dummy_tokenizer):
    parser = GraniteReasoningParser(dummy_tokenizer)

    # Chunk 1: thought delimiter
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="Here's my thought process:\nI think",
        delta_text="Here's my thought process:\nI think",
        previous_token_ids=[],
        current_token_ids=[100],
        delta_token_ids=[100],
    )
    assert delta is not None
    assert delta.reasoning_content is not None

    # Chunk 2: more reasoning
    delta = parser.extract_reasoning_streaming(
        previous_text="Here's my thought process:\nI think",
        current_text="Here's my thought process:\nI think deeply",
        delta_text=" deeply",
        previous_token_ids=[100],
        current_token_ids=[100, 101],
        delta_token_ids=[101],
    )
    assert delta is not None
    assert delta.reasoning_content == " deeply"

    # Chunk 3: response delimiter
    delta = parser.extract_reasoning_streaming(
        previous_text="Here's my thought process:\nI think deeply",
        current_text="Here's my thought process:\nI think deeply\nHere's my response:\nDone",
        delta_text="\nHere's my response:\nDone",
        previous_token_ids=[100, 101],
        current_token_ids=[100, 101, 102],
        delta_token_ids=[102],
    )
    assert delta is not None
    # After response delimiter, content should be present
    assert delta.content is not None
