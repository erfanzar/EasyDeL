# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EasyDeL Inference Reasoning Module.

Provides infrastructure for extracting thinking/reasoning content from
Large Language Model outputs. Reasoning extraction runs before tool call
parsing in the pipeline:

    Model Output -> ReasoningParser -> {reasoning, content} -> ToolParser(content) -> {tool_calls}

Note:
    Ideas and design patterns are inspired by vLLM's reasoning parser implementation.
"""

from .abstract_reasoning import ReasoningParser, ReasoningParserManager
from .basic_parsers import BaseThinkingReasoningParser
from .parsers import (
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
from .reasoning_mixin import ReasoningMixin

__all__ = (
    "BaseThinkingReasoningParser",
    "DeepSeekR1ReasoningParser",
    "DeepSeekV3ReasoningParser",
    "Ernie45ReasoningParser",
    "GptOssReasoningParser",
    "GraniteReasoningParser",
    "HunyuanA13BReasoningParser",
    "IdentityReasoningParser",
    "MiniMaxM2AppendThinkReasoningParser",
    "MiniMaxM2ReasoningParser",
    "MistralReasoningParser",
    "Olmo3ReasoningParser",
    "Qwen3ReasoningParser",
    "ReasoningMixin",
    "ReasoningParser",
    "ReasoningParserManager",
    "SeedOSSReasoningParser",
    "Step3ReasoningParser",
    "Step3p5ReasoningParser",
)
