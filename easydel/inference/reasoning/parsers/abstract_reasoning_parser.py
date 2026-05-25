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

"""Compatibility shim for vLLM-style import paths.

vLLM and third-party plugins import the abstract reasoning base classes from
``...parsers.abstract_reasoning_parser``. EasyDeL keeps the canonical
definitions one level up in :mod:`easydel.inference.reasoning.abstract_reasoning`;
this module re-exports them under the legacy path so existing imports continue
to work unchanged.
"""

from ..abstract_reasoning import ReasoningParser, ReasoningParserManager

__all__ = ["ReasoningParser", "ReasoningParserManager"]
