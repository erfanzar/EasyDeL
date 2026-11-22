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

"""Reasoning parser for Baidu ERNIE 4.5 models.

Format: <think>reasoning content</think>response
"""

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["ernie45"])  # pyright: ignore[reportUntypedClassDecorator]
class Ernie45ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Baidu ERNIE-4.5 chain-of-thought outputs.

    ERNIE-4.5 uses the same ``<think>``/``</think>`` grammar as DeepSeek-R1
    and most other Western open-weights reasoning models. The opening tag
    can either be emitted by the model or pre-seeded by the chat template
    when ``add_generation_prompt`` is active; in the latter case the
    prompt-gated path inherited from :class:`BaseThinkingReasoningParser`
    treats the output as already inside the reasoning section.

    The streaming state machine emits :class:`DeltaMessage` events with
    ``reasoning_content`` while inside the markers and ``content`` once the
    closing tag is consumed. All the heavy lifting (delimiter splitting,
    prompt-context handling, streaming boundary detection) lives in the
    base class — this subclass only fixes the marker strings.

    Attributes:
        start_token: Reasoning open tag ``"<think>"``.
        end_token: Reasoning close tag ``"</think>"`` whose detection flips
            the parser into the visible-content phase.
    """

    start_token = "<think>"
    end_token = "</think>"
