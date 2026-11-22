# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Distributed execution engine, decorators, and remote-class machinery."""

from .decorators import (
    autoscale_execute,
    autoscale_execute_resumable,
    execute,
    execute_resumable,
)
from .executor import (
    ENV_CALL_INDEX,
    ENV_CALL_SLICE,
    MEGASCALE_DEFAULT_PORT,
    RayExecutor,
    resolve_maybe_refs,
)
from .remote import device_remote

__all__ = (
    "ENV_CALL_INDEX",
    "ENV_CALL_SLICE",
    "MEGASCALE_DEFAULT_PORT",
    "RayExecutor",
    "autoscale_execute",
    "autoscale_execute_resumable",
    "device_remote",
    "execute",
    "execute_resumable",
    "resolve_maybe_refs",
)
