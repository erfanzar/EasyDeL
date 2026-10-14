# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reasoning parser for Tencent Hunyuan A13B models.

Format: <think>reasoning content</think>response
"""

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["hunyuan_a13b"])  # pyright: ignore[reportUntypedClassDecorator]
class HunyuanA13BReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Tencent Hunyuan-A13B chain-of-thought outputs.

    Hunyuan-A13B follows the canonical ``<think>``…``</think>`` chain-of-thought
    grammar shared by DeepSeek-R1, Qwen3, ERNIE, OLMo-3, and Seed-OSS.
    The class plugs Hunyuan into the registry without overriding any
    behavior; all batch and streaming logic — including prompt-gated
    parsing when the chat template injects ``<think>`` via
    ``add_generation_prompt`` — comes from
    :class:`BaseThinkingReasoningParser`.

    During streaming the inherited state machine emits :class:`DeltaMessage`
    events carrying ``reasoning_content`` while inside the markers and
    switches to ``content`` once ``</think>`` is observed. Tokenization of
    the marker strings is resolved at construction time so that boundary
    detection survives both text-level and id-level scans.

    Attributes:
        start_token: Reasoning open tag ``"<think>"``.
        end_token: Reasoning close tag ``"</think>"``.
    """

    start_token = "<think>"
    end_token = "</think>"
