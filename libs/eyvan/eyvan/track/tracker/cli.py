# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line entry point for the standalone read-only tracker service."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import uvicorn

from eyvan.track._store import store_for_uri
from eyvan.track.tracker.app import create_tracker_app
from eyvan.track.tracker.index import TrackingIndex


def main() -> None:
    """Parse tracker settings and run uvicorn in the foreground."""

    parser = argparse.ArgumentParser(description="Index Eyvan run segments and serve the tracking API")
    parser.add_argument("--store", required=True, help="Tracking store root (gs://, file:// or local path)")
    parser.add_argument(
        "--db", type=Path, default=Path("~/.eyvan/tracker/index.sqlite3"), help="Rebuildable SQLite index"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--scan-interval", type=float, default=5.0, help="Store scan interval in seconds")
    parser.add_argument(
        "--log-level", default="info", choices=("critical", "error", "warning", "info", "debug", "trace")
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not math.isfinite(args.scan_interval) or args.scan_interval <= 0:
        parser.error("--scan-interval must be finite and positive")
    try:
        store = store_for_uri(args.store)
    except ValueError:
        parser.error("--store must be a valid gs://, file:// or local tracking root")
    app = create_tracker_app(
        store, TrackingIndex(args.db), token=os.environ.get("EYVAN_TRACKER_TOKEN"), scan_interval_s=args.scan_interval
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


__all__ = ("main",)
