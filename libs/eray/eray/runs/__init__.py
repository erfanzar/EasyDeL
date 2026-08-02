# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Task-state semantics adapted from the Iris cluster manager in
# marin-community/marin (lib/iris — Copyright The Marin Authors,
# Apache-2.0).
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

"""eray.runs — training-aware run management on Ray clusters.

The run-management layer of eray's control plane (capacity pools hold
slices, fleet turns them into Ray clusters, ``eray runs`` keeps workloads
alive on them): iris-style task states with separate failure/preemption
budgets, step-progress health with a cold-compile grace window, a futile-
repair quarantine ladder, and a single-watcher lease.
"""

from .manager import RunsManager, resolve_jobs_address
from .model import (
    TERMINAL_STATES,
    HealthPolicy,
    RunAction,
    RunObservation,
    RunRecord,
    RunSpec,
    RunState,
    plan_run,
    reset_for_retry,
)
from .store import Lease, LeaseHeldError, RunStore

__all__ = [
    "TERMINAL_STATES",
    "HealthPolicy",
    "Lease",
    "LeaseHeldError",
    "RunAction",
    "RunObservation",
    "RunRecord",
    "RunSpec",
    "RunState",
    "RunStore",
    "RunsManager",
    "plan_run",
    "reset_for_retry",
    "resolve_jobs_address",
]
