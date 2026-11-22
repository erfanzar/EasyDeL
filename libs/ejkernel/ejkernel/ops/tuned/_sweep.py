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

"""Sweeps that generate rows for the tuned-kernel table.

A kernel becomes tunable by registering a :class:`SweepSpec` that says which
points to measure and how to build a candidate. Everything else — timing,
picking the winner, recording the runner-up, writing rows — is shared, so
adding a kernel is a description rather than another bespoke tuning script.

The timing rule that matters: **candidates are jitted before measurement**.
Calling an ejkernel module eagerly re-enters its registry/config dispatch on
every iteration, which costs about a millisecond and swamps the kernel — a
sweep written without this reported an identical 0.93 ms for an M=192 and an
M=12288 grouped matmul.
"""

from __future__ import annotations

import time
import typing as tp
from dataclasses import dataclass

from ._store import TunedEntry, dtype_signature, shape_signature

__all__ = ["Candidate", "SweepPoint", "SweepSpec", "register_sweep", "run_sweep", "sweep_specs"]


@dataclass(frozen=True)
class Candidate:
    """One (platform, config) option to measure at a point.

    Attributes:
        platform: Backend this candidate runs on.
        config: Configuration recorded if it wins.
        build: Callable returning a zero-arg function to time. Returning a
            builder (rather than the callable itself) lets a spec jit once per
            candidate and keep that out of the measurement.
    """

    platform: str
    config: dict[str, tp.Any]
    build: tp.Callable[[], tp.Callable[[], tp.Any]]


@dataclass(frozen=True)
class SweepPoint:
    """One key to tune: a dtype signature plus a shape.

    Attributes:
        dtypes: Named dtypes for :func:`dtype_signature`.
        shape: Named dimensions for :func:`shape_signature` (bucketed).
        candidates: Options to measure at this point.
        label: Optional human-readable name for progress output.
    """

    dtypes: dict[str, tp.Any]
    shape: dict[str, int]
    candidates: list[Candidate]
    label: str = ""
    baseline: Candidate | None = None
    """What the kernel picks with NO table. Measured alongside the candidates so
    a row can say whether tuning beat not tuning, rather than only which
    candidate beat which."""


@dataclass
class SweepSpec:
    """How to sweep one kernel.

    Attributes:
        kernel: Kernel identifier written into the table.
        points: Callable producing the points to measure. Takes the parsed CLI
            options so a sweep can be narrowed without editing code.
        description: Shown by ``--list``.
    """

    kernel: str
    points: tp.Callable[..., tp.Iterable[SweepPoint]]
    description: str = ""


_SPECS: dict[str, SweepSpec] = {}


def register_sweep(spec: SweepSpec) -> SweepSpec:
    """Register a kernel sweep so the CLI can run it."""
    _SPECS[spec.kernel] = spec
    return spec


def sweep_specs() -> dict[str, SweepSpec]:
    """Registered sweeps, keyed by kernel name."""
    return dict(_SPECS)


def time_callable(fn: tp.Callable[[], tp.Any], *, reps: int = 30, rounds: int = 3) -> float:
    """Best-of-*rounds* mean milliseconds for an already-jitted callable."""
    import jax

    jax.block_until_ready(fn())  # compile
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            out = fn()
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - start) / reps)
    return best * 1e3


def run_sweep(
    spec: SweepSpec,
    *,
    device: str | None = None,
    reps: int = 30,
    min_gain: float = 1.05,
    provenance: dict[str, tp.Any] | None = None,
    on_point: tp.Callable[[str, TunedEntry], None] | None = None,
    on_skip: tp.Callable[[str, float], None] | None = None,
    **point_kwargs: tp.Any,
) -> list[TunedEntry]:
    """Measure every candidate at every point and return the winners.

    Each entry records the winner and the best *other* option, so a later
    reader can see how much the choice was worth — and, when the runner-up is
    on a different backend, whether the platform choice was close.

    Args:
        spec: Sweep to run.
        device: Device kind to record; defaults to the current device.
        reps: Timed repetitions per candidate.
        min_gain: Minimum speedup over the point's baseline before a row is
            written. Measured on v5p, ragged_page_attention_v3's best block
            sizes beat second place by a median of 1.010x with nine different
            "winners" across 35 shapes -- noise picking arbitrarily among
            equivalent configs. Persisting that is worse than persisting
            nothing, so points that cannot clear this are skipped -- as are
            points whose default could not be measured at all.
        provenance: Recorded once for the whole sweep.
        on_point: Optional progress callback ``(label, winner)``.
        **point_kwargs: Forwarded to ``spec.points``.

    Returns:
        One :class:`TunedEntry` per point that had at least one working candidate.

    Note:
        A point whose default could not be measured -- it declared none, or it
        failed to build -- is skipped rather than written. Writing anyway is how
        the first version of this sweep filled a table with 1.01x "winners":
        without a baseline the row cannot say tuning beat NOT tuning, which is
        the only claim that justifies storing it.
    """
    from ._lookup import current_device_kind

    dev = device or current_device_kind()
    out: list[TunedEntry] = []
    skipped = 0
    unverified = 0
    for point in spec.points(**point_kwargs):
        timed: list[tuple[float, Candidate]] = []
        for cand in point.candidates:
            try:
                fn = cand.build()
                timed.append((time_callable(fn, reps=reps), cand))
            except Exception:
                # A candidate that cannot compile at this shape is simply not an
                # option here; that is data, not a failure of the sweep.
                continue
        if not timed:
            continue
        timed.sort(key=lambda pair: pair[0])
        (best_ms, best), *rest = timed

        base_ms = None
        if point.baseline is not None:
            try:
                base_ms = time_callable(point.baseline.build(), reps=reps)
            except Exception:
                base_ms = None
        if base_ms is None:
            # The default may also be one of the swept candidates.
            for ms, cand in timed:
                if point.baseline is not None and cand.config == point.baseline.config:
                    base_ms = ms
                    break

        if base_ms is None:
            unverified += 1
            if on_skip is not None:
                on_skip(point.label or "", float("nan"))
            continue
        if best_ms > 0 and (base_ms / best_ms) < min_gain:
            # The default is already within `min_gain` of the best candidate, so
            # a row here would persist measurement noise as if it were a finding
            # -- worse than no row, because it reads as authoritative.
            skipped += 1
            if on_skip is not None:
                on_skip(point.label or "", base_ms / best_ms)
            continue

        entry = TunedEntry(
            kernel=spec.kernel,
            device=dev,
            dtypes=dtype_signature(**point.dtypes),
            shape_key=shape_signature(**point.shape),
            platform=best.platform,
            config=best.config,
            ms=best_ms,
            runner_up=(
                {"platform": rest[0][1].platform, "config": rest[0][1].config, "ms": rest[0][0]} if rest else None
            ),
            baseline=(
                {"config": point.baseline.config, "ms": base_ms}
                if (point.baseline is not None and base_ms is not None)
                else None
            ),
            provenance=provenance,
        )
        out.append(entry)
        if on_point is not None:
            on_point(point.label or entry.shape_key, entry)
    if skipped:
        print(f"  ({skipped} point(s) skipped: default already within {min_gain:.2f}x of best)", flush=True)
    if unverified:
        print(f"  ({unverified} point(s) skipped: no measurable default to compare against)", flush=True)
    return out
