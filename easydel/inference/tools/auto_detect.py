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

"""Auto-detection of tool call parsers from model configuration.

Provides ``detect_tool_parser`` which inspects a model's architecture,
chat template, or tokenizer vocabulary to determine the correct tool
call parser. This avoids requiring users to manually specify parser
names when the model type makes the choice unambiguous.

The detection hierarchy is:

1. **Explicit name** — if the caller passes a parser name, use it directly.
2. **Model type mapping** — map ``config.model_type`` to a known parser via
   ``MODEL_TYPE_TO_TOOL_PARSER``.
3. **Chat template probing** — scan the tokenizer's chat template string for
   known tool-call delimiters (``<tool_call>``, ``[TOOL_CALLS]``, etc.).
4. **Vocabulary probing** — check if special tool-calling tokens exist in the
   tokenizer's vocabulary.
5. **Fallback** — return ``"hermes"`` (``<tool_call>``/``</tool_call>``), the
   most widely supported format.

Example:
    >>> from easydel.inference.tools.auto_detect import detect_tool_parser
    >>> name = detect_tool_parser(model_type="qwen3")
    >>> name
    'hermes'
    >>> name = detect_tool_parser(model_type="mistral")
    >>> name
    'mistral'
"""

from __future__ import annotations

import typing as tp

from .abstract_tool import ToolParserManager, ToolParserName

MODEL_TYPE_TO_TOOL_PARSER: dict[str, ToolParserName] = {
    "qwen3_5_moe": "hermes",
    "qwen3_5": "hermes",
    "qwen3_moe": "hermes",
    "qwen3_next": "hermes",
    "qwen3": "hermes",
    "qwen2_moe": "hermes",
    "qwen2": "hermes",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v2": "deepseek_v3",
    "deepseek": "hermes",
    "llama4": "llama4_json",
    "llama": "llama3_json",
    "mistral3": "mistral",
    "mistral": "mistral",
    "mixtral": "mistral",
    "granite": "granite",
    "olmo3": "olmo3",
    "olmo2": "olmo3",
    "olmo": "olmo3",
    "phi4": "phi4_mini_json",
    "phimoe": "hermes",
    "phi3": "hermes",
    "phi": "hermes",
    "glm_moe_dsa": "glm47",
    "glm4_moe_lite": "glm47",
    "glm4_moe": "glm47",
    "glm46v": "glm47",
    "glm4v_moe": "glm47",
    "glm4v": "glm45",
    "glm4": "glm45",
    "glm": "glm45",
    "internlm2": "internlm",
    "internlm": "internlm",
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
    "gemma3": "hermes",
    "gemma2": "functiongemma",
    "gemma": "functiongemma",
    "jamba": "jamba",
    "cohere2": "hermes",
    "cohere": "hermes",
    "command_r": "hermes",
    "kimi_k2": "kimi_k2",
    "kimi": "kimi_k2",
    "gpt_oss": "hermes",
    "smollm3": "hermes",
    "exaone4": "hermes",
    "exaone": "hermes",
    "falcon_h1": "hermes",
    "falcon": "hermes",
    "xerxes2": "hermes",
    "xerxes": "hermes",
    "stablelm": "hermes",
}
"""Mapping from ``config.model_type`` to the canonical tool parser name.

The keys are matched as prefixes (case-insensitive, longest first) against
the model's ``model_type`` string.
"""

_TEMPLATE_HINTS: list[tuple[str, ToolParserName]] = [
    ("<|tool_calls_section_begin|>", "kimi_k2"),
    ("<|tool▁calls▁begin|>", "deepseek_v3"),
    ("[TOOL_CALLS]", "mistral"),
    ("<|action_start|>", "internlm"),
    ("<seed:tool_call>", "seed_oss"),
    ("<minimax:tool_call>", "minimax_m2"),
    ("<steptml:invoke", "step3"),
    ("functools[", "phi4_mini_json"),
    ("<tool_calls>", "hunyuan_a13b"),
    ("<arg_key>", "glm45"),
    ("<tool_call>", "hermes"),
]
"""Chat-template substrings mapped to tool parser names."""

_VOCAB_HINTS: list[tuple[str, ToolParserName]] = [
    ("<|tool_calls_section_begin|>", "kimi_k2"),
    ("<|tool▁calls▁begin|>", "deepseek_v3"),
    ("[TOOL_CALLS]", "mistral"),
    ("<|action_start|>", "internlm"),
    ("<seed:tool_call>", "seed_oss"),
    ("<tool_call>", "hermes"),
]
"""Special tokens to look for in the tokenizer vocabulary."""

_DEFAULT_PARSER: ToolParserName = "hermes"
"""Fallback when no signal is found.  Hermes-style ``<tool_call>`` is the
most widely adopted format (Qwen3, Nous, many finetunes)."""


def detect_tool_parser(
    *,
    parser_name: str | None = None,
    model_type: str | None = None,
    tokenizer: tp.Any | None = None,
) -> ToolParserName:
    """Auto-detect the appropriate tool call parser.

    Args:
        parser_name: Explicit parser name.  If provided and valid, returned
            as-is without further detection.
        model_type: The ``model_type`` string from the model config (e.g.
            ``"qwen3"``, ``"mistral"``, ``"llama"``).
        tokenizer: A HuggingFace tokenizer.  Used for chat-template and
            vocabulary probing when ``model_type`` is not sufficient.

    Returns:
        A registered ``ToolParserName`` string that can be passed to
        ``ToolParserManager.get_tool_parser()``.
    """
    if parser_name is not None:
        if parser_name in ToolParserManager.tool_parsers:
            return parser_name  # type: ignore[return-value]

    if model_type is not None:
        mt = model_type.lower()
        for prefix in sorted(MODEL_TYPE_TO_TOOL_PARSER, key=len, reverse=True):
            if mt.startswith(prefix) or prefix in mt:
                return MODEL_TYPE_TO_TOOL_PARSER[prefix]

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
