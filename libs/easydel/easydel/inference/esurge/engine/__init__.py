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

"""Explicit engine components composing the eSurge facade.

Each module here owns one engine concern with constructor-injected
dependencies; :class:`~easydel.inference.esurge.eSurge` is the composition
root that wires them together.
"""

from .admission import RequestAdmission
from .bootstrap import EngineAssets, build_engine_assets
from .idle import IdleMonitor
from .loop import MAX_CONSECUTIVE_SCHEDULER_ERRORS, EngineLoop
from .monitoring_stack import MonitoringStack
from .output_pipeline import OutputPipeline
from .registry import RequestRecord, RequestRegistry

__all__ = (
    "MAX_CONSECUTIVE_SCHEDULER_ERRORS",
    "EngineAssets",
    "EngineLoop",
    "IdleMonitor",
    "MonitoringStack",
    "OutputPipeline",
    "RequestAdmission",
    "RequestRecord",
    "RequestRegistry",
    "build_engine_assets",
)
