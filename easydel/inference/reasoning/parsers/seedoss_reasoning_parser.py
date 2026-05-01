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

"""Reasoning parser for Seed OSS models.

Format: <think>reasoning content</think>response
"""

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["seed_oss"])  # pyright: ignore[reportUntypedClassDecorator]
class SeedOSSReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Seed-OSS chain-of-thought outputs.

    Seed-OSS reuses the literal ``<think>``…``</think>`` grammar that has
    become standard among open thinking models. This class is a marker
    subclass — all parsing semantics live in
    :class:`BaseThinkingReasoningParser`, including the streaming state
    machine that emits :class:`DeltaMessage` events with
    ``reasoning_content`` while inside the markers and ``content`` after
    the closing tag is consumed, and the prompt-gated path that handles
    chat templates which pre-emit ``<think>``.

    Attributes:
        start_token: Reasoning open tag ``"<think>"``.
        end_token: Reasoning close tag ``"</think>"``.
    """

    start_token = "<think>"
    end_token = "</think>"
