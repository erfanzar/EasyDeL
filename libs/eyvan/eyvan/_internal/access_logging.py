# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Uvicorn access logging that keeps resumable cursors out of ordinary logs.

Uvicorn's stock ``AccessFormatter`` writes the whole request target, query
string included. Log paging sends its position as ``?after=<cursor>``, so every
page and every reconnecting follower wrote a resumable position token into the
access log of the control plane, the cell agent and the native worker at NORMAL
verbosity. DR-0002 requires the opposite: "cursors are not logged at ordinary
verbosity".

The formatter here keeps the method, path, protocol version and status -- the
parts an operator reads a request line for -- and replaces the values of an
enumerated set of query parameters with :data:`REDACTED_QUERY_VALUE`. Query
parameters outside that set are left exactly as they arrived, so a benign
``?state=running`` still identifies which request was served.

This module sits in ``eyvan._internal`` rather than beside the control plane's
own logging configuration because ``eyvan.worker`` is forbidden from importing
``eyvan.server``; the cell agent and the native worker need the same formatter.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import copy
from urllib.parse import parse_qsl, urlencode

from uvicorn.logging import AccessFormatter

from eyvan._internal.json_types import JsonObject, JsonValue

CURSOR_QUERY_PARAMETERS: frozenset[str] = frozenset({"after", "cursor"})
"""Query parameters that carry a resumable position token.

``after`` is the log-page cursor: ``GET /api/v1/workloads/{id}/log-page``,
``GET /api/v1/workloads/{id}/logs/stream``, the cell agent's
``GET /api/v1/cells/{cell}/workloads/{remote}/log-page`` and the native
worker's ``GET /api/v1/workloads/{remote}/log-page`` all bind it, and it is the
outer, nested and leaf cursor DR-0002 governs.

``cursor`` is the managed-artifact inventory position on
``GET /api/v1/artifacts/inventory``: a canonical ``eyvan://`` URI resumed as a
strict successor, which is the same class of resumable read position.
"""

CREDENTIAL_QUERY_PARAMETERS: frozenset[str] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "id_token",
        "password",
        "refresh_token",
        "secret",
        "sig",
        "signature",
        "token",
    }
)
"""Credential-shaped names that no Eyvan route binds today.

Every Eyvan capability travels in a header -- ``Authorization``,
``X-Eyvan-Artifact-Token``, ``X-Eyvan-Secret-Token``, ``X-Eyvan-Workload-Token``
-- and no route in the control plane, the cell agent or the native worker
declares any of these as a query parameter. They are redacted anyway because an
access log is a durable artifact: a caller that puts a bearer token in a query
string writes it into the log of a server that never asked for it, and nothing
downstream can un-write it. Nothing depends on their values being visible.
"""

REDACTED_QUERY_PARAMETERS: frozenset[str] = CURSOR_QUERY_PARAMETERS | CREDENTIAL_QUERY_PARAMETERS
"""The complete denylist the access formatter applies, matched case-insensitively."""

REDACTED_QUERY_VALUE = "REDACTED"
"""The fixed placeholder written in place of a sensitive query value."""

REDACTED_URL_LOGGERS: tuple[str, ...] = ("uvicorn.access", "httpx")
"""Loggers whose records are redacted before any handler sees them.

``uvicorn.access`` is the inbound request line. ``httpx`` is the outbound one:
it logs ``'HTTP Request: %s %s "%s %d %s"'`` with the whole URL at **INFO**, so
a controller-to-agent or agent-to-worker log page carries its cursor there too.
Both names were established by enumeration -- attaching a capturing handler to
the root logger at ``DEBUG`` and paging logs through the client Eyvan actually
uses -- rather than by guessing: ``httpcore.connection`` and ``httpcore.http11``
also log during that request and render the request as ``<Request [b'GET']>``,
with no URL in it, so they are deliberately not listed.
"""

#: Scalars are never request targets and are left alone so that a record's
#: ``%d`` conversions keep working.
_OPAQUE_ARGUMENT_TYPES = (bool, int, float, complex, bytes, bytearray)
#: Containers are never request targets either, and rendering one to look for a
#: query string could be arbitrarily expensive.
_CONTAINER_ARGUMENT_TYPES = (list, tuple, set, frozenset, dict)

_ACCESS_FORMAT = '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
_DEFAULT_FORMAT = "%(levelprefix)s %(message)s"
_REDACTING_ACCESS_FORMATTER = "eyvan._internal.access_logging.RedactingAccessFormatter"
_QUERY_REDACTION_FILTER = "eyvan._internal.access_logging.QueryRedactionFilter"


def redact_request_target(target: str) -> str:
    """Return one request target with every sensitive query value replaced.

    Args:
        target: The request target uvicorn logs, i.e. a percent-encoded path
            optionally followed by ``?`` and the raw query string.

    Returns:
        The target unchanged when it carries no sensitive value, otherwise the
        same path with the query re-encoded and every denied parameter's value
        replaced by :data:`REDACTED_QUERY_VALUE`. Parameter names are compared
        after percent-decoding and case-folding, so ``?%61fter=`` is redacted
        exactly like ``?after=``. Blank values are left blank: an empty
        ``after`` is the documented "read from the start" signal and carries no
        position to leak.
    """

    path, separator, query = target.partition("?")
    if not separator or not query:
        return target
    parameters = parse_qsl(query, keep_blank_values=True)
    if not any(value and name.lower() in REDACTED_QUERY_PARAMETERS for name, value in parameters):
        return target
    return (
        path
        + "?"
        + urlencode(
            [
                (name, REDACTED_QUERY_VALUE if value and name.lower() in REDACTED_QUERY_PARAMETERS else value)
                for name, value in parameters
            ]
        )
    )


def _redacted_argument(value: object) -> object:
    """Return one log argument with its sensitive query values replaced.

    Positional arguments are matched by shape rather than by position, because
    the two loggers that carry a URL disagree about where it sits: uvicorn puts
    the request target third in ``(client_addr, method, target, version,
    status)`` while httpx puts an ``httpx.URL`` second in ``(method, url,
    version, status, reason)``. An argument is only replaced when redaction
    actually changed how it renders, so an unrelated argument keeps its original
    object and type.
    """

    if isinstance(value, _OPAQUE_ARGUMENT_TYPES) or isinstance(value, _CONTAINER_ARGUMENT_TYPES) or value is None:
        return value
    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except Exception:
            # A logging filter that raises takes the log line and the request
            # with it, so an argument that cannot render is simply left alone.
            return value
    redacted = redact_request_target(text)
    return redacted if redacted != text else value


def _redacted_record(record: logging.LogRecord) -> logging.LogRecord:
    """Return the record to emit, copied and redacted when it needs it."""

    arguments = record.args
    if isinstance(arguments, tuple):
        redacted: tuple[object, ...] | dict[str, object] = tuple(_redacted_argument(item) for item in arguments)
        changed = any(new is not old for new, old in zip(redacted, arguments, strict=True))
    elif isinstance(arguments, dict):
        redacted = {key: _redacted_argument(item) for key, item in arguments.items()}
        changed = any(value is not arguments[key] for key, value in redacted.items())
    else:
        return record
    if not changed:
        return record
    duplicate = copy(record)
    duplicate.args = redacted
    return duplicate


class QueryRedactionFilter(logging.Filter):
    """Strip sensitive query values from a record before it is handled.

    A logger's filters run once, in ``Logger.handle``, ahead of every handler
    and ahead of propagation to the root logger, so this is what makes the
    redaction a property of the record rather than of one formatter: an operator
    who adds a JSON access handler, or who simply calls
    ``logging.basicConfig(level=DEBUG)`` while chasing a stuck log page, gets
    the redacted URL too. The record is edited in place because that is the
    point -- downstream of the emitting logger the raw cursor should not exist.

    A filter only runs on the logger that emitted the record, never on an
    ancestor, so this must be attached to each name in
    :data:`REDACTED_URL_LOGGERS` individually. A record carrying nothing
    sensitive is left exactly as it was.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record in place and always keep it."""

        redacted = _redacted_record(record)
        if redacted is not record:
            record.args = redacted.args
        return True


class RedactingAccessFormatter(AccessFormatter):
    """Uvicorn's access formatter with sensitive query values removed.

    The substitution happens on a copy of the record before any formatting, so
    ``%(request_line)s`` and ``%(message)s`` are both derived from the redacted
    target. It is idempotent, and deliberately duplicates
    :class:`QueryRedactionFilter`: the filter protects handlers this formatter
    never sees, and the formatter protects a logger the filter was left off.
    A record whose arguments are not uvicorn's access tuple is passed to
    uvicorn's own formatting unchanged, so this class adds no failure mode of
    its own.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format one access record with its sensitive query values replaced."""

        return super().format(_redacted_record(record))


def uvicorn_log_config(level: str = "INFO", extra_loggers: Mapping[str, JsonValue] | None = None) -> JsonObject:
    """Return uvicorn's own logging graph with the redacting access formatter.

    This is uvicorn's default handler graph, unchanged apart from the access
    formatter, so a service that adopts it keeps the log lines it had. Uvicorn
    applies its ``log_level`` argument to ``uvicorn.error`` and
    ``uvicorn.access`` after this configuration is installed, exactly as it does
    for its own default.

    Args:
        level: The level name applied to the uvicorn loggers.
        extra_loggers: Additional ``loggers`` entries merged into the graph, for
            a service that also owns package loggers of its own.

    Returns:
        A ``logging.config.dictConfig`` document.
    """

    configured_level = level.upper()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"redact_query": {"()": _QUERY_REDACTION_FILTER}},
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": _DEFAULT_FORMAT,
                "use_colors": None,
            },
            "access": {
                "()": _REDACTING_ACCESS_FORMATTER,
                "fmt": _ACCESS_FORMAT,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": configured_level, "propagate": False},
            "uvicorn.error": {"level": configured_level},
            "uvicorn.access": {
                "handlers": ["access"],
                "filters": ["redact_query"],
                "level": configured_level,
                "propagate": False,
            },
            # Filter only: httpx keeps its own level and propagation, so this
            # neither starts nor silences outbound request logging, it only
            # removes the cursor from a line something else chose to emit.
            **{name: {"filters": ["redact_query"]} for name in REDACTED_URL_LOGGERS if name != "uvicorn.access"},
            **dict(extra_loggers or {}),
        },
    }


__all__ = (
    "CREDENTIAL_QUERY_PARAMETERS",
    "CURSOR_QUERY_PARAMETERS",
    "REDACTED_QUERY_PARAMETERS",
    "REDACTED_QUERY_VALUE",
    "REDACTED_URL_LOGGERS",
    "QueryRedactionFilter",
    "RedactingAccessFormatter",
    "redact_request_target",
    "uvicorn_log_config",
)
