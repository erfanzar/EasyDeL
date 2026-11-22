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

"""Backward-compatible tool registry, schemas, and built-in tools.

Implementations live in :mod:`easydel.trainers._shared.tools`. Both import
paths expose the same classes, functions, and registry, including tools
registered by downstream applications through this legacy module.
"""

from .._shared.tools import (
    _TOOL_REGISTRY as _TOOL_REGISTRY,
)
from .._shared.tools import (
    _TYPE_MAP as _TYPE_MAP,
)
from .._shared.tools import (
    BashTool as BashTool,
)
from .._shared.tools import (
    CalculatorTool as CalculatorTool,
)
from .._shared.tools import (
    FileReadTool as FileReadTool,
)
from .._shared.tools import (
    FunctionTool as FunctionTool,
)
from .._shared.tools import (
    JSONProcessorTool as JSONProcessorTool,
)
from .._shared.tools import (
    NotepadTool as NotepadTool,
)
from .._shared.tools import (
    PythonCodeTool as PythonCodeTool,
)
from .._shared.tools import (
    RegexTool as RegexTool,
)
from .._shared.tools import (
    Tool as Tool,
)
from .._shared.tools import (
    UnitConverterTool as UnitConverterTool,
)
from .._shared.tools import (
    WebFetchTool as WebFetchTool,
)
from .._shared.tools import (
    WikipediaTool as WikipediaTool,
)
from .._shared.tools import (
    function_to_json as function_to_json,
)
from .._shared.tools import (
    list_tools as list_tools,
)
from .._shared.tools import (
    make_tool as make_tool,
)
from .._shared.tools import (
    register_tool as register_tool,
)
