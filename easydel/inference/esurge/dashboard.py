"""Deprecated stub for the removed eSurge web dashboard.

The custom FastAPI/JavaScript dashboard has been removed in favor of
standard Prometheus metrics that can be visualized in Grafana (or any
Prometheus-compatible UI).

This module provides placeholder classes and functions that raise
RuntimeError when called, guiding users to the recommended Prometheus/Grafana
approach for monitoring.

Classes:
    eSurgeWebDashboard: Deprecated placeholder that raises RuntimeError.

Functions:
    create_dashboard: Deprecated placeholder that raises RuntimeError.
    create_dashboard_fixed: Deprecated placeholder that raises RuntimeError.

Example:
    >>> # Instead of the old dashboard, use Prometheus metrics:
    >>> engine.start_monitoring()  # Starts Prometheus metrics endpoint
    >>> # Then connect Grafana to http://localhost:11184/metrics

Note:
    The built-in dashboard was removed to simplify the codebase and
    leverage industry-standard monitoring tools. Use engine.start_monitoring()
    to expose Prometheus metrics and visualize them with Grafana.
"""

from __future__ import annotations

from typing import Any

_DASHBOARD_REMOVAL_MSG = (
    "The built-in eSurge dashboard was removed. "
    "Expose Prometheus metrics with engine.start_monitoring() and use Grafana "
    "or another Prometheus UI for charts."
)


class eSurgeWebDashboard:
    """Deprecated placeholder for the removed dashboard.

    This class exists only to provide a helpful error message directing
    users to the Prometheus/Grafana monitoring approach.

    Raises:
        RuntimeError: Always raised on instantiation with migration guidance.

    Example:
        >>> # This will raise RuntimeError:
        >>> dashboard = eSurgeWebDashboard()
        RuntimeError: The built-in eSurge dashboard was removed...
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Raise RuntimeError with migration guidance.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            RuntimeError: Always raised with migration guidance message.
        """
        raise RuntimeError(_DASHBOARD_REMOVAL_MSG)


def create_dashboard(*args: Any, **kwargs: Any) -> eSurgeWebDashboard:
    """Deprecated placeholder for the removed dashboard.

    This function exists only to provide a helpful error message directing
    users to the Prometheus/Grafana monitoring approach.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        Never returns (always raises).

    Raises:
        RuntimeError: Always raised with migration guidance message.

    Example:
        >>> # This will raise RuntimeError:
        >>> dashboard = create_dashboard(engine)
        RuntimeError: The built-in eSurge dashboard was removed...
    """
    raise RuntimeError(_DASHBOARD_REMOVAL_MSG)


def create_dashboard_fixed(*args: Any, **kwargs: Any) -> eSurgeWebDashboard:
    """Deprecated placeholder for the removed dashboard.

    Alias for create_dashboard that provides the same error message.

    Args:
        *args: Passed to create_dashboard.
        **kwargs: Passed to create_dashboard.

    Returns:
        Never returns (always raises).

    Raises:
        RuntimeError: Always raised with migration guidance message.
    """
    return create_dashboard(*args, **kwargs)
