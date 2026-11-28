# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pull a training-loss curve out of a job's printed output.

Research plans compare jobs ("lr 1e-3 against 3e-3"), and the number they are
compared on is the loss the trainer printed. EasyDeL's eLarge prints it in a
tqdm postfix -- ``training process:  50%|... 10/20 [..., loss=10, mean_loss=10.2, ...]``
-- with carriage returns between redraws and, on a gang, under whichever rank
owned the progress bar. Generic trainers print ``step=5 loss=0.31`` or a JSON
line with ``"loss": 0.31``. This module reads those shapes and nothing else:
``mean_loss``, ``eval/loss`` and ``loss_scale`` are different numbers and are
never mistaken for the loss.

Pure and bounded: no I/O, at most :data:`MAX_LOSS_POINTS` points returned.
"""

from __future__ import annotations

import math
import re
import typing as tp

#: Points kept from one job's curve; evenly thinned, the last point always kept.
MAX_LOSS_POINTS = 50
#: Characters of output examined (the tail: the final loss lives there).
MAX_SCANNED_CHARS = 4 * 1024 * 1024

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# `loss` as a whole key: not mean_loss, eval/loss, loss_scale or lossy.
_LOSS = re.compile(r"""(?<![\w/.])["']?loss["']?\s*[:=]\s*(""" + _NUMBER + r")(?![\w.])")
_TQDM_STEP = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
_STEP = re.compile(r"""(?<![\w/.])["']?(?:global_)?step["']?\s*[:=]\s*(\d+)(?![\w.])""")


class LossCurve(tp.TypedDict):
    """What :func:`extract_loss` found; ``final`` is the loss at the highest step seen."""

    final: float
    last_step: int | None
    total_steps: int | None
    count: int
    points: list[tuple[int | None, float]]


def _thin(points: list[tuple[int | None, float]], limit: int) -> list[tuple[int | None, float]]:
    if len(points) <= limit:
        return points
    picked = [points[round(index * (len(points) - 1) / (limit - 1))] for index in range(limit)]
    return list(dict.fromkeys(picked))


def extract_loss(text: str) -> LossCurve | None:
    """Return the loss curve printed in ``text``, or ``None`` when no loss was printed.

    Each line is one observation. A line's step comes from a tqdm counter
    (``10/20 [``) or a ``step=``/``"step":`` field; within one step the last
    line wins (tqdm redraws, and every rank of a gang may print the same bar).
    Lines without a step keep their order after the stepped ones. Non-finite
    values are skipped: a NaN loss is a fact worth seeing in the log, not a
    number to compare jobs on.
    """

    if len(text) > MAX_SCANNED_CHARS:
        text = text[-MAX_SCANNED_CHARS:]
    by_step: dict[int, float] = {}
    unstepped: list[float] = []
    total: int | None = None
    count = 0
    for line in text.replace("\r", "\n").split("\n"):
        if "loss" not in line:
            continue
        found = _LOSS.search(line)
        if found is None:
            continue
        value = float(found.group(1))
        if not math.isfinite(value):
            continue
        count += 1
        tqdm = _TQDM_STEP.search(line)
        if tqdm is not None:
            step: int | None = int(tqdm.group(1))
            total = int(tqdm.group(2))
        else:
            explicit = _STEP.search(line)
            step = int(explicit.group(1)) if explicit is not None else None
        if step is None:
            unstepped.append(value)
        else:
            by_step[step] = value
    if not count:
        return None
    points: list[tuple[int | None, float]] = [(step, by_step[step]) for step in sorted(by_step)]
    points.extend((None, value) for value in unstepped)
    if by_step:
        last_step: int | None = max(by_step)
        final = by_step[last_step] if not unstepped else unstepped[-1]
    else:
        last_step, final = None, unstepped[-1]
    return LossCurve(
        final=final,
        last_step=last_step,
        total_steps=total,
        count=count,
        points=_thin(points, MAX_LOSS_POINTS),
    )


__all__ = ("MAX_LOSS_POINTS", "LossCurve", "extract_loss")
