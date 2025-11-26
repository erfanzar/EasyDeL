"""Deprecated stub for the removed eSurge web dashboard.

The custom FastAPI/JavaScript dashboard has been removed in favor of
standard Prometheus metrics that can be visualized in Grafana (or any
Prometheus-compatible UI).
"""

from __future__ import annotations

from typing import Any

_DASHBOARD_REMOVAL_MSG = (
    "The built-in eSurge dashboard was removed. "
    "Expose Prometheus metrics with engine.start_monitoring() and use Grafana "
    "or another Prometheus UI for charts."
)


class eSurgeWebDashboard:
    """Deprecated placeholder for the removed dashboard."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(_DASHBOARD_REMOVAL_MSG)


def create_dashboard(*args: Any, **kwargs: Any) -> eSurgeWebDashboard:
    """Deprecated placeholder for the removed dashboard."""
    raise RuntimeError(_DASHBOARD_REMOVAL_MSG)


def create_dashboard_fixed(*args: Any, **kwargs: Any) -> eSurgeWebDashboard:
    """Deprecated placeholder for the removed dashboard."""
    return create_dashboard(*args, **kwargs)
