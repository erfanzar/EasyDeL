# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mixins composing the public :class:`eSurge` engine surface.

The eSurge engine class is built up from focused mixins so each concern
(I/O, lifecycle, monitoring, parsing, request bookkeeping, utilities) lives
in its own module. ``eSurge`` inherits from all of them.

Exports:
    EngineIOMixin: text/multimodal I/O entry points (``generate``, ``stream``).
    EngineLifecycleMixin: start/pause/terminate, AOT compile, idle resets.
    EngineMonitoringMixin: Prometheus / Grafana / console monitor lifecycle.
    EngineParsingMixin: tool-call and reasoning-block parsing helpers.
    EngineRequestsMixin: request creation / cancellation / output retrieval.
    EngineUtilsMixin: misc utilities shared across the other mixins.
"""

from .io import EngineIOMixin
from .lifecycle import EngineLifecycleMixin
from .monitoring import EngineMonitoringMixin
from .parsing import EngineParsingMixin
from .requests import EngineRequestsMixin
from .utils import EngineUtilsMixin

__all__ = [
    "EngineIOMixin",
    "EngineLifecycleMixin",
    "EngineMonitoringMixin",
    "EngineParsingMixin",
    "EngineRequestsMixin",
    "EngineUtilsMixin",
]
