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

"""Reasoning parser for StepFun Step-3.5 thinking models.

Step-3.5 retains the same asymmetric "no opening tag, only ``</think>`` end
marker" grammar as its Step-3 predecessor, so this module exists purely to
register an additional set of aliases (``step3p5`` / ``step3.5``) that map
to the same implementation.
"""

from ..abstract_reasoning import ReasoningParserManager
from .step3_reasoning_parser import Step3ReasoningParser


@ReasoningParserManager.register_module(["step3p5", "step3.5"])  # pyright: ignore[reportUntypedClassDecorator]
class Step3p5ReasoningParser(Step3ReasoningParser):
    """Reasoning parser for Step-3.5 outputs (same grammar as Step-3).

    Step-3.5 inherits its parent's asymmetric grammar: the model never
    emits an opening ``<think>`` marker, so the parser treats every text
    chunk before the first ``</think>`` as reasoning and everything after
    as visible content. Streaming events are produced by the parent's
    state machine — :class:`DeltaMessage` with ``reasoning_content``
    while accumulating, then a transition delta that splits the
    boundary chunk into reasoning and content halves, then ``content``
    deltas thereafter.

    The class body is intentionally empty; it exists solely to register
    the ``"step3p5"``/``"step3.5"`` aliases against the manager.
    """

    pass
