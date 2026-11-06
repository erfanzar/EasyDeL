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

"""The runner's view of one scheduling step.

eSurge schedules through surgecore's Rust ``EngineCore`` (request table, per-rank queues, token budgets, KV pages,
prefix cache, preemption, chunked prefill, speculative drafts);
:class:`~easydel.inference.esurge.engine.core_driver.EngineCoreDriver` owns it. This package keeps the step structures
the runner and the multi-host followers read: :class:`SchedulerOutput`, :class:`NewRequestData` and
:class:`CachedRequestData`.

The former Python scheduler classes (``Scheduler``, ``AsyncScheduler``, ``DPScheduler``, ``SchedulerInterface``,
the request queues and ``TokenBudgetManager``) are kept as names only: using one raises :class:`RuntimeError` that
points to ``EngineCoreDriver``.
"""

from .output import CachedRequestData, NewRequestData, SchedulerOutput


class _RemovedMeta(type):
    """Metaclass of a removed name: any attribute access raises, as instantiation does."""

    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(cls.__doc__)


def _removed(name: str) -> type:
    """A placeholder class for a scheduler name that moved into surgecore's Rust ``EngineCore``."""
    message = (
        f"easydel.inference.esurge.scheduler.{name} was removed: eSurge schedules through surgecore's Rust "
        "EngineCore. Use easydel.inference.esurge.engine.core_driver.EngineCoreDriver instead."
    )

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(message)

    return _RemovedMeta(name, (), {"__new__": __new__, "__doc__": message})


AsyncScheduler = _removed("AsyncScheduler")
DPScheduler = _removed("DPScheduler")
FCFSRequestQueue = _removed("FCFSRequestQueue")
PriorityRequestQueue = _removed("PriorityRequestQueue")
RequestQueue = _removed("RequestQueue")
Scheduler = _removed("Scheduler")
SchedulerInterface = _removed("SchedulerInterface")
TokenBudgetManager = _removed("TokenBudgetManager")

__all__ = (
    "AsyncScheduler",
    "CachedRequestData",
    "DPScheduler",
    "FCFSRequestQueue",
    "NewRequestData",
    "PriorityRequestQueue",
    "RequestQueue",
    "Scheduler",
    "SchedulerInterface",
    "SchedulerOutput",
    "TokenBudgetManager",
)
