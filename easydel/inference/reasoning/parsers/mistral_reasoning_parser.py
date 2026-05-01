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

"""Reasoning parser for Mistral models.

Format: [THINK]reasoning content[/THINK]actual response
Uses special tokens rather than XML tags.
"""

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["mistral"])  # pyright: ignore[reportUntypedClassDecorator]
class MistralReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Mistral thinking models using bracketed delimiters.

    Mistral diverges from the ``<think>``/``</think>`` convention and uses
    bracketed special tokens ``[THINK]``/``[/THINK]`` that are part of the
    tokenizer vocabulary — they are emitted as single token IDs rather
    than as multi-character text. The base class handles both
    representations: it resolves the corresponding ``input_ids`` from the
    tokenizer at construction time and falls back to text-level scanning
    when the IDs are unavailable, so detection works the same in either
    case.

    Streaming follows the inherited state machine: while inside
    ``[THINK]``…``[/THINK]`` the parser emits :class:`DeltaMessage` events
    with ``reasoning_content``; once ``[/THINK]`` is consumed the phase
    flips and subsequent deltas surface as visible ``content``. Prompt-gated
    asymmetric parsing (chat template injecting ``[THINK]`` via
    ``add_generation_prompt``) is also handled by the base class.

    Attributes:
        start_token: Mistral reasoning open special token ``"[THINK]"``.
        end_token: Mistral reasoning close special token ``"[/THINK]"``.
    """

    start_token = "[THINK]"
    end_token = "[/THINK]"
