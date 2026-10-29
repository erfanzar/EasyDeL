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

"""Reasoning parser for DeepSeek R1 models.

Format: <think>reasoning content</think>actual response
"""

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["deepseek_r1", "deepseek-r1"])  # pyright: ignore[reportUntypedClassDecorator]
class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for DeepSeek-R1 chain-of-thought outputs.

    The DeepSeek-R1 family emits chain-of-thought wrapped in literal
    ``<think>``/``</think>`` markers; everything between the markers is
    private reasoning, everything after is the visible assistant response.
    Either marker may also be injected by the chat template (when
    ``add_generation_prompt`` is on), in which case the model output starts
    *inside* the reasoning section and only emits the closing tag.

    The class is a thin specialization of
    :class:`BaseThinkingReasoningParser` that fixes the grammar to
    ``<think>`` / ``</think>``. All state tracking — the prompt-gated
    asymmetric mode, partial start/end tag detection, streaming-event
    emission of :class:`DeltaMessage` with ``reasoning_content`` and/or
    ``content`` fields — is inherited from the base class.

    Attributes:
        start_token: The opening reasoning marker ``"<think>"`` matched in
            both batch and streaming text.
        end_token: The closing reasoning marker ``"</think>"`` that flips
            the parser from ``REASONING`` into ``CONTENT`` mode.
    """

    start_token = "<think>"
    end_token = "</think>"
