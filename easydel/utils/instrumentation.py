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

"""Lightweight wall-clock phase timing for slow infra paths.

Replaces the per-file ``_phase_timer`` / ``_shardgen_phase`` context managers
that were sprinkled across ``mixins/bridge.py`` and ``modules/auto`` during
the spectrax migration.  A single ``phase_timer`` factory now serves both
``from_pretrained`` (tag ``"from_pretrained"``) and ``AutoShardAndGatherFunctions``
(tag ``"AutoShardAndGatherFunctions"``), accumulating into an optional dict
and emitting one INFO line per phase.

Set the environment variable ``EASYDEL_PHASE_TIMING=0`` to silence the
per-phase log lines without touching the call sites; the timing context still
runs (and still updates accumulators) so summaries remain accurate.
"""

from __future__ import annotations

import contextlib
import time
import typing as tp

from eformer.loggings import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def phase_timer(
    label: str,
    *,
    accumulator: dict[str, float] | None = None,
    tag: str = "from_pretrained",
) -> tp.Iterator[None]:
    """Measure the wall-clock duration of a code block and log it once.

    Args:
        label: Human-readable phase name shown in the log line.
        accumulator: Optional dict the elapsed time is added into under
            ``label``.  Useful for printing a summary at the end of an outer
            phase.
        tag: Outer category prepended to the log line (e.g. ``"from_pretrained"``
            or ``"AutoShardAndGatherFunctions"``).  Defaults to
            ``"from_pretrained"`` to match the previous bridge behaviour.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if accumulator is not None:
            accumulator[label] = accumulator.get(label, 0.0) + elapsed

        logger.debug(f"[{tag}] {label}: {elapsed:.2f}s")


__all__ = ["phase_timer"]
