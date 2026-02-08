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

"""Mixin class for reasoning extraction functionality in inference servers."""

from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger

from ..openai_api_modules import DeltaMessage
from .abstract_reasoning import ReasoningParser, ReasoningParserManager

logger = get_logger("ReasoningMixin")


class ReasoningMixin:
    """Mixin class providing reasoning extraction for inference API servers.

    Follows the same pattern as ToolCallingMixin.
    """

    reasoning_parsers: dict[str, ReasoningParser]

    def initialize_reasoning_parsers(
        self,
        model_processors: dict[str, tp.Any],
        reasoning_parser_name: str,
        enable_reasoning: bool,
    ) -> dict[str, ReasoningParser]:
        """Initialize reasoning parsers for all registered models."""
        reasoning_parsers = {}
        if not enable_reasoning:
            return reasoning_parsers

        for model_name, processor in model_processors.items():
            try:
                parser_class = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
                reasoning_parsers[model_name] = parser_class(processor)
                logger.info(f"Initialized {reasoning_parser_name} reasoning parser for model {model_name}")
            except KeyError:
                logger.warning(
                    f"Reasoning parser '{reasoning_parser_name}' not found, " f"reasoning disabled for {model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize reasoning parser for {model_name}: {e}")

        return reasoning_parsers

    def extract_reasoning_batch(
        self,
        response_text: str,
        model_name: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from a complete response.

        Returns:
            Tuple of (reasoning_content, content_without_reasoning).
        """
        if not hasattr(self, "reasoning_parsers") or model_name not in self.reasoning_parsers:
            return None, response_text

        parser = self.reasoning_parsers[model_name]
        return parser.extract_reasoning(response_text)

    def extract_reasoning_streaming(
        self,
        model_name: str,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int] | None = None,
        current_token_ids: list[int] | None = None,
        delta_token_ids: list[int] | None = None,
        request=None,
    ) -> DeltaMessage | None:
        """Extract reasoning from streaming response chunks."""
        if not hasattr(self, "reasoning_parsers") or model_name not in self.reasoning_parsers:
            return None

        parser = self.reasoning_parsers[model_name]
        return parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids or [],
            current_token_ids=current_token_ids or [],
            delta_token_ids=delta_token_ids or [],
            request=request,
        )

    def get_reasoning_parser_for_model(self, model_name: str) -> ReasoningParser | None:
        if not hasattr(self, "reasoning_parsers"):
            return None
        return self.reasoning_parsers.get(model_name)
