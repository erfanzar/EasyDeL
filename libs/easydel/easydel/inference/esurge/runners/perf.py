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

"""Runner-level performance tracking state for the eSurge model runner.

Holds the per-step performance sample record and the rolling counters the
runner updates after every completed step. The accumulation and log-emission
logic itself stays in :meth:`eSurgeRunner._execute_model_impl`; this module
only owns the state that outlives a single step (EMA, last-step scalars, and
the bounded step histories consumed by benchmarks such as
an external benchmark harness).
"""

from __future__ import annotations

import typing
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RunnerPerfSample:
    """One snapshot of runner-level performance counters.

    Emitted by :class:`eSurgeRunner` after each completed step.

    Attributes:
        iteration: Monotonically increasing step counter.
        total_tokens: Tokens fed to the model this step (prompt + decode).
        num_scheduled_reqs: Number of requests in the scheduler output.
        num_new: Newly arrived requests on this step.
        num_cached: Requests served from prefix cache on this step.
        num_finished: Requests that completed this step.
        total_time: Wall-clock seconds spent in the step.
        agg_tps: Aggregate tokens/second across all sequences.
        req_tps: Average per-request tokens/second.
        ema_tps: Exponential moving average of ``agg_tps``.
    """

    iteration: int
    total_tokens: int
    num_scheduled_reqs: int
    num_new: int
    num_cached: int
    num_finished: int
    total_time: float
    agg_tps: float
    req_tps: float
    ema_tps: float


class RunnerPerfTracker:
    """Rolling performance counters for the eSurge runner.

    Kept lightweight; no allocations in the hot path. The runner exposes the
    historical ``_perf_*`` attribute names as thin passthrough properties so
    external readers (serving benchmarks) keep
    working unchanged.

    Attributes:
        iteration: Monotonically increasing step counter.
        tps_ema: Exponential moving average of aggregate tokens/second.
        alpha: EMA smoothing factor applied to ``tps_ema``.
        last_agg_tps: Aggregate tokens/second of the most recent step.
        last_req_tps: Per-request tokens/second of the most recent step.
        last_total_time: Wall-clock seconds of the most recent step.
        last_total_tokens: Tokens fed to the model on the most recent step.
        history: Bounded deque of :class:`RunnerPerfSample` step snapshots.
        phase_history: Bounded deque of per-step phase-timing dicts.
    """

    def __init__(self, *, history_maxlen: int, alpha: float = 0.2) -> None:
        """Initialize empty perf-tracking state.

        Args:
            history_maxlen: Maximum number of retained step samples in both
                ``history`` and ``phase_history``.
            alpha: EMA smoothing factor for the tokens/second average.
        """
        self.iteration = 0
        self.tps_ema: float | None = None
        self.alpha = alpha
        self.last_agg_tps: float | None = None
        self.last_req_tps: float | None = None
        self.last_total_time: float | None = None
        self.last_total_tokens: int | None = None
        self.history: deque[RunnerPerfSample] = deque(maxlen=history_maxlen)
        self.phase_history: deque[dict[str, typing.Any]] = deque(maxlen=history_maxlen)
