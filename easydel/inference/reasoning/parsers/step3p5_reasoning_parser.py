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

"""Reasoning parser for Step3.5 models. Same format as Step3."""

from ..abstract_reasoning import ReasoningParserManager
from .step3_reasoning_parser import Step3ReasoningParser


@ReasoningParserManager.register_module(["step3p5", "step3.5"])
class Step3p5ReasoningParser(Step3ReasoningParser):
    """Reasoning parser for Step3.5 models. Inherits Step3 logic."""

    pass
