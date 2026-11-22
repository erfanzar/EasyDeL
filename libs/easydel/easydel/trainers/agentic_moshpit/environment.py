# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backward-compatible agentic environment API.

Implementations live in :mod:`easydel.trainers._shared.environment` so all
trainer algorithms share the same environment types and guarded tool wrapper.
"""

from .._shared.environment import (
    AgenticEnvironment as AgenticEnvironment,
)
from .._shared.environment import (
    ResetResult as ResetResult,
)
from .._shared.environment import (
    StepResult as StepResult,
)
from .._shared.environment import (
    ToolEnvWrapper as ToolEnvWrapper,
)
from .._shared.environment import (
    _default_tool_call_parser as _default_tool_call_parser,
)
from .._shared.environment import (
    _make_inference_tool_parser as _make_inference_tool_parser,
)
from .._shared.environment import (
    _make_regex_parser as _make_regex_parser,
)
from .._shared.environment import (
    create_tool_call_parser as create_tool_call_parser,
)
