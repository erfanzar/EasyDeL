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

"""Reasoning parser for AI2 OLMo3 models.

Format: <think>reasoning content</think>response
"""

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["olmo3"])  # pyright: ignore[reportUntypedClassDecorator]
class Olmo3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for AI2 OLMo-3 chain-of-thought outputs.

    OLMo-3 thinking models share the de-facto-standard
    ``<think>``…``</think>`` grammar with DeepSeek-R1, Qwen3, and ERNIE.
    No OLMo-specific behavior is needed beyond fixing the marker strings;
    delimiter detection (text-level and token-id-level), prompt-context
    handling for ``add_generation_prompt``-injected start tokens, and
    streaming :class:`DeltaMessage` emission are inherited from
    :class:`BaseThinkingReasoningParser`.

    The inherited streaming state machine routes deltas into one of three
    cases on every chunk: still inside the ``<think>`` block, the
    ``</think>`` marker straddles the chunk boundary, or already past
    the boundary into visible content.

    Attributes:
        start_token: Reasoning open tag ``"<think>"``.
        end_token: Reasoning close tag ``"</think>"``.
    """

    start_token = "<think>"
    end_token = "</think>"
