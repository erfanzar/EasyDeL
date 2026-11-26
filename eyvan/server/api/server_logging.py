# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Uvicorn logging configuration for the Eyvan control-plane process."""

from __future__ import annotations

import logging
from copy import copy
from urllib.parse import unquote

from eyvan._internal.access_logging import (
    CREDENTIAL_QUERY_PARAMETERS,
    CURSOR_QUERY_PARAMETERS,
    REDACTED_QUERY_PARAMETERS,
    REDACTED_QUERY_VALUE,
    REDACTED_URL_LOGGERS,
    QueryRedactionFilter,
    RedactingAccessFormatter,
    redact_request_target,
    uvicorn_log_config,
)
from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject

_DASHBOARD_CALLBACK_PATH = "/api/v1/dashboard/auth/callback"


def _redact_dashboard_callback_argument(value: object) -> object:
    """Hide the entire callback query, including unrecognized provider errors."""

    if value is None or isinstance(
        value, (bool, int, float, complex, bytes, bytearray, list, tuple, set, frozenset, dict)
    ):
        return value
    try:
        target = str(value)
    except Exception:
        # Match the shared query filter's handling of unrenderable arguments.
        return value
    path, separator, query = target.partition("?")
    # Uvicorn may include a mounting root_path or percent-encoded path; httpx
    # logs an absolute URL object. The suffix handles all three forms without
    # changing benign state=running on other routes. Hide the whole query so a
    # provider cannot leak private data through an unknown name or error code.
    if separator and query and unquote(path).rstrip("/").endswith(_DASHBOARD_CALLBACK_PATH):
        return f"{path}?{REDACTED_QUERY_VALUE}"
    return value


def _redact_dashboard_callback_record(record: logging.LogRecord) -> None:
    """Remove callback queries before any downstream handler sees the record."""

    if isinstance(record.args, tuple):
        record.args = tuple(_redact_dashboard_callback_argument(value) for value in record.args)
    elif isinstance(record.args, dict):
        record.args = {key: _redact_dashboard_callback_argument(value) for key, value in record.args.items()}


class DashboardCallbackRedactionFilter(QueryRedactionFilter):
    """Extend shared cursor/credential redaction for the plane's OIDC callback."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize callback queries and retain the shared query protections."""

        _redact_dashboard_callback_record(record)
        return super().filter(record)


class DashboardCallbackAccessFormatter(RedactingAccessFormatter):
    """Protect formatter-only consumers as well as the configured logger graph."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a copy with the callback query removed."""

        redacted = copy(record)
        _redact_dashboard_callback_record(redacted)
        return super().format(redacted)


class ServerLogLevel(StrEnum):
    """Log levels accepted by both Uvicorn and ``eyvan-server``."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    TRACE = "trace"


def server_log_config(level: ServerLogLevel) -> JsonObject:
    """Return one handler graph for Uvicorn and all Eyvan package loggers.

    The access formatter and each URL logger's filter extend the shared
    cursor/credential redaction with control-plane callback protection. OIDC
    callback queries are hidden in full: authorization codes, transaction state
    and provider-controlled error details must not reach any handler. Benign
    queries on other routes (including ``state=running``) remain visible.

    Cursor redaction is unconditional too: DR-0002 states that cursors are not
    logged at ordinary verbosity, and an access line is not a wire payload.

    Args:
        level: The level applied to the uvicorn and ``eyvan`` loggers.

    Returns:
        A ``logging.config.dictConfig`` document.
    """

    configured_level = level.value.upper()
    config = uvicorn_log_config(
        configured_level,
        {"eyvan": {"handlers": ["default"], "level": configured_level, "propagate": False}},
    )
    filters = config["filters"]
    formatters = config["formatters"]
    assert isinstance(filters, dict) and isinstance(formatters, dict)
    filters["redact_query"] = {"()": "eyvan.server.api.server_logging.DashboardCallbackRedactionFilter"}
    access = formatters["access"]
    assert isinstance(access, dict)
    access["()"] = "eyvan.server.api.server_logging.DashboardCallbackAccessFormatter"
    return config


__all__ = (
    "CREDENTIAL_QUERY_PARAMETERS",
    "CURSOR_QUERY_PARAMETERS",
    "REDACTED_QUERY_PARAMETERS",
    "REDACTED_QUERY_VALUE",
    "REDACTED_URL_LOGGERS",
    "QueryRedactionFilter",
    "RedactingAccessFormatter",
    "ServerLogLevel",
    "redact_request_target",
    "server_log_config",
    "uvicorn_log_config",
)
