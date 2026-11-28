# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared resume frontier for episodes and rich records, across ranks and kinds."""

from __future__ import annotations

import sqlite3


def later_frontiers(db: sqlite3.Connection, scope: tuple[str, ...]) -> dict[int, int | None]:
    """Return the minimum non-null step of any strictly later attempt.

    Scalars and histograms retain their per-key overwrite semantics. Episodes,
    rollouts, completions, tables and media share this frontier: adding a rich
    stream may exclude more old episodes, but episode-only behavior is identical.
    """

    rows = db.execute(
        """SELECT s.attempt, MIN(r.step) frontier FROM (
               SELECT segment, step FROM episode_records
               UNION ALL SELECT segment, step FROM content_records
           ) r JOIN segments s USING(segment)
           WHERE s.org = ? AND s.project = ? AND s.run = ? GROUP BY s.attempt""",
        scope,
    ).fetchall()
    later: dict[int, int | None] = {}
    frontier = None
    for row in sorted(rows, key=lambda row: row["attempt"], reverse=True):
        later[row["attempt"]] = frontier
        if row["frontier"] is not None:
            frontier = min(frontier, row["frontier"]) if frontier is not None else row["frontier"]
    return later


__all__ = ("later_frontiers",)
