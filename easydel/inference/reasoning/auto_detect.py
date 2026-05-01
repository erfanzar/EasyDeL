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

"""Auto-detection of reasoning parsers from model configuration.

Provides ``detect_reasoning_parser`` which inspects a model's architecture,
chat template, or tokenizer vocabulary to determine the correct reasoning
parser. This avoids requiring users to manually specify parser names when
the model type makes the choice unambiguous.

The detection hierarchy is:

1. **Explicit name** — if the caller passes a parser name, use it directly.
2. **Model type mapping** — map ``config.model_type`` to a known parser via
   ``MODEL_TYPE_TO_REASONING_PARSER``.
3. **Chat template probing** — scan the tokenizer's chat template string for
   known reasoning delimiters (``<think>``, ``[THINK]``, etc.).
4. **Vocabulary probing** — check if special reasoning tokens exist in the
   tokenizer's vocabulary.
5. **Fallback** — return ``"deepseek_r1"`` (``<think>``/``</think>``), the
   most common format across model families.

Example:
    >>> from easydel.inference.reasoning.auto_detect import detect_reasoning_parser
    >>> name = detect_reasoning_parser(model_type="qwen3")
    >>> name
    'qwen3'
    >>> name = detect_reasoning_parser(tokenizer=tokenizer)  # auto from template
"""

from __future__ import annotations

import typing as tp

from .abstract_reasoning import ReasoningParserManager, ReasoningParserName

MODEL_TYPE_TO_REASONING_PARSER: dict[str, ReasoningParserName] = {
    "qwen3_5_moe": "qwen3",
    "qwen3_5": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_next": "qwen3",
    "qwen3": "qwen3",
    "qwen2_moe": "deepseek_r1",
    "qwen2": "deepseek_r1",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v2": "deepseek_r1",
    "deepseek": "deepseek_r1",
    "llama4": "deepseek_r1",
    "llama": "deepseek_r1",
    "mistral3": "mistral",
    "mistral": "mistral",
    "mixtral": "mistral",
    "granite": "granite",
    "olmo3": "olmo3",
    "olmo2": "olmo3",
    "olmo": "olmo3",
    "phi4": "deepseek_r1",
    "phimoe": "deepseek_r1",
    "phi3": "deepseek_r1",
    "phi": "deepseek_r1",
    "glm_moe_dsa": "glm45",
    "glm4_moe_lite": "glm45",
    "glm4_moe": "glm45",
    "glm46v": "glm45",
    "glm4v_moe": "glm45",
    "glm4v": "glm45",
    "glm4": "glm45",
    "glm": "glm45",
    "internlm2": "deepseek_r1",
    "internlm": "deepseek_r1",
    "seed_oss": "seed_oss",
    "seed": "seed_oss",
    "ernie": "ernie45",
    "minimax_text_01": "minimax_m2",
    "minimax": "minimax_m2",
    "hunyuan": "hunyuan_a13b",
    "step3p5": "step3p5",
    "step3.5": "step3p5",
    "step3": "step3",
    "step": "step3",
    "gemma4_text": "gemma4",
    "gemma4": "gemma4",
    "gemma3": "deepseek_r1",
    "gemma2": "deepseek_r1",
    "gemma": "deepseek_r1",
    "cohere2": "deepseek_r1",
    "cohere": "deepseek_r1",
    "command_r": "deepseek_r1",
    "kimi_k2": "kimi_k2",
    "kimi": "kimi_k2",
    "jamba": "deepseek_r1",
    "gpt_oss": "gptoss",
    "smollm3": "deepseek_r1",
    "exaone4": "deepseek_r1",
    "exaone": "deepseek_r1",
    "falcon_h1": "deepseek_r1",
    "falcon": "deepseek_r1",
    "xerxes2": "deepseek_r1",
    "xerxes": "deepseek_r1",
    "stablelm": "deepseek_r1",
    "starcoder": "deepseek_r1",
}
"""Mapping from ``config.model_type`` to the canonical reasoning parser name.

The keys are matched as prefixes (case-insensitive) against the model's
``model_type`` string, so ``"qwen3"`` matches both ``"qwen3"`` and
``"qwen3_moe"``.  Longer prefixes are tried first to avoid false matches
(e.g. ``"deepseek_v3"`` before ``"deepseek"``).
"""

_TEMPLATE_HINTS: list[tuple[str, ReasoningParserName]] = [
    ("[THINK]", "mistral"),
    ("Here's my thought process:", "granite"),
    ("Here is my thought process:", "granite"),
    ("<|channel>", "gemma4"),
    ("<|channel|>", "gptoss"),
    ("<seed:think>", "seed_oss"),
    ("<think>", "deepseek_r1"),
]
"""Chat-template substrings mapped to reasoning parser names.

Checked in order; the first match wins.
"""

_VOCAB_HINTS: list[tuple[str, ReasoningParserName]] = [
    ("[THINK]", "mistral"),
    ("<|channel>", "gemma4"),
    ("<|channel|>", "gptoss"),
    ("<think>", "deepseek_r1"),
]
"""Special tokens to look for in the tokenizer vocabulary."""

_DEFAULT_PARSER: ReasoningParserName = "deepseek_r1"
"""Fallback when no signal is found.  ``<think>``/``</think>`` is by far the
most common reasoning format (Qwen3, DeepSeek-R1, OLMo-3, ERNIE, etc.)."""


def detect_reasoning_parser(
    *,
    parser_name: str | None = None,
    model_type: str | None = None,
    tokenizer: tp.Any | None = None,
) -> ReasoningParserName:
    """Auto-detect the appropriate reasoning parser.

    Args:
        parser_name: Explicit parser name.  If provided and valid, returned
            as-is without further detection.
        model_type: The ``model_type`` string from the model config (e.g.
            ``"qwen3"``, ``"mistral"``, ``"granite"``).
        tokenizer: A HuggingFace tokenizer.  Used for chat-template and
            vocabulary probing when ``model_type`` is not sufficient.

    Returns:
        A registered ``ReasoningParserName`` string that can be passed to
        ``ReasoningParserManager.get_reasoning_parser()``.
    """
    if parser_name is not None:
        if parser_name in ReasoningParserManager.reasoning_parsers:
            return parser_name  # type: ignore[return-value]

    if model_type is not None:
        mt = model_type.lower()
        for prefix in sorted(MODEL_TYPE_TO_REASONING_PARSER, key=len, reverse=True):
            if mt.startswith(prefix) or prefix in mt:
                return MODEL_TYPE_TO_REASONING_PARSER[prefix]

    if tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None) or ""
        if isinstance(template, str):
            for hint, name in _TEMPLATE_HINTS:
                if hint in template:
                    return name

        vocab = getattr(tokenizer, "get_vocab", lambda: {})()
        if isinstance(vocab, dict):
            for token, name in _VOCAB_HINTS:
                if token in vocab:
                    return name

    return _DEFAULT_PARSER


def get_reasoning_tags(
    *,
    parser_name: str | None = None,
    model_type: str | None = None,
    tokenizer: tp.Any | None = None,
) -> tuple[str, str]:
    """Resolve the start/end reasoning tags for a model.

    Combines ``detect_reasoning_parser`` with a lookup into the parser
    class to extract ``start_token`` and ``end_token`` attributes.

    Args:
        parser_name: Explicit parser name override.
        model_type: Model architecture string for auto-detection.
        tokenizer: Tokenizer for template/vocab probing.

    Returns:
        ``(start_tag, end_tag)`` tuple.  Defaults to
        ``("<think>", "</think>")`` if the parser has no tag attributes.
    """
    name = detect_reasoning_parser(
        parser_name=parser_name,
        model_type=model_type,
        tokenizer=tokenizer,
    )
    try:
        parser_cls = ReasoningParserManager.get_reasoning_parser(name)
        start = getattr(parser_cls, "start_token", "<think>")
        end = getattr(parser_cls, "end_token", "</think>")
        return start, end
    except KeyError:
        return "<think>", "</think>"


def make_reasoning_stripper(
    *,
    parser_name: str | None = None,
    model_type: str | None = None,
    tokenizer: tp.Any | None = None,
) -> tp.Callable[[str], str]:
    """Build a function that strips reasoning blocks from model output.

    Auto-detects the reasoning format and returns a compiled-regex
    stripping function for the correct start/end tags.  Handles both
    closed tags (``start...end``) and unclosed tags where the model
    hit the token limit before producing the end tag.

    Args:
        parser_name: Explicit parser name override.
        model_type: Model architecture string for auto-detection.
        tokenizer: Tokenizer for template/vocab probing.

    Returns:
        ``(text: str) -> str`` that removes reasoning blocks.
    """
    import re

    name = detect_reasoning_parser(
        parser_name=parser_name,
        model_type=model_type,
        tokenizer=tokenizer,
    )
    try:
        parser_cls = ReasoningParserManager.get_reasoning_parser(name)
    except KeyError:
        parser_cls = None

    if parser_cls is None:
        start_tokens = ("<think>",)
        end_token = "</think>"
    else:
        start_token = getattr(parser_cls, "start_token", "<think>")
        end_token = getattr(parser_cls, "end_token", "</think>")
        start_tokens = getattr(parser_cls, "reasoning_start_tokens", (start_token,))
        if isinstance(start_tokens, str):
            start_tokens = (start_tokens,)
        else:
            start_tokens = tuple(str(token) for token in start_tokens if token)
        if not start_tokens:
            start_tokens = (start_token,)

    escaped_start_pattern = "|".join(sorted((re.escape(token) for token in start_tokens), key=len, reverse=True))
    escaped_end = re.escape(end_token)
    closed_re = re.compile(rf"(?:{escaped_start_pattern}).*?{escaped_end}", re.DOTALL)
    unclosed_re = re.compile(rf"(?:{escaped_start_pattern}).*", re.DOTALL)

    def strip_reasoning(text: str) -> str:
        """Remove reasoning blocks from ``text`` using the resolved tags.

        Args:
            text: Source text potentially containing reasoning markup.

        Returns:
            ``text`` with closed and unclosed reasoning sections removed and
            outer whitespace stripped.
        """
        result = closed_re.sub("", text)
        result = unclosed_re.sub("", result)
        return result.strip()

    return strip_reasoning
