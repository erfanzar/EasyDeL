# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Access-log redaction for every Eyvan HTTP service (DR-0002).

Log paging sends its resumable position as ``?after=<cursor>``. Uvicorn's stock
access formatter writes the full request target, so that cursor -- outer, nested
and leaf -- was written into the access log of the control plane, the cell agent
and the native worker at NORMAL verbosity, which DR-0002 forbids.

The end-to-end coverage here runs a real uvicorn server in a subprocess for each
shipped logging configuration and reads what the configured access handler
actually wrote. The subprocess keeps ``logging.config.dictConfig`` -- which
closes every existing handler in the process -- away from pytest's own logging,
the same reason ``tests/test_api.py`` runs its logging-visibility check out of
process.
"""

from __future__ import annotations

import io
import json
import logging
import logging.config
import subprocess
import sys
import typing as tp
from urllib.parse import quote
from uuid import uuid4

import pytest
import uvicorn
from uvicorn.logging import AccessFormatter

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
from eyvan._internal.json_types import JsonObject
from eyvan.server.api.server_logging import ServerLogLevel, server_log_config

#: A DR-0002 v2 log cursor: a prefix plus a bounded base64url JSON envelope.
CURSOR = "eyvan-log-v2.eyJ2IjoyLCJraW5kIjoid29ya2xvYWQtbG9nIiwiYXR0ZW1wdCI6M30"
LOG_PAGE_PATH = "/api/v1/workloads/wl-4213/log-page"
CURSOR_TARGET = f"{LOG_PAGE_PATH}?after={CURSOR}&limit=200&state=running"

#: The longest run of cursor characters allowed to survive into a log line. Any
#: shorter window would match by chance on an alphabet this small.
_FRAGMENT = 8


def _cursor_fragments(cursor: str = CURSOR) -> tuple[str, ...]:
    """Return every contiguous ``_FRAGMENT``-character window of one cursor."""

    return tuple(cursor[index : index + _FRAGMENT] for index in range(len(cursor) - _FRAGMENT + 1))


def _assert_cursor_absent(line: str, cursor: str = CURSOR) -> None:
    """Fail when any recognisable piece of a cursor survived into a log line."""

    assert cursor not in line
    leaked = [fragment for fragment in _cursor_fragments(cursor) if fragment in line]
    assert leaked == [], f"cursor fragments leaked into the access log: {leaked}"


_PROBE = r"""
from __future__ import annotations

import io
import json
import logging
import threading
import time

import httpx
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter

from eyvan._internal.access_logging import uvicorn_log_config
from eyvan.server.api.server_logging import ServerLogLevel, server_log_config

TARGETS = json.loads(__import__("sys").argv[1])
STOCK_FORMAT = '%(client_addr)s - "%(request_line)s" %(status_code)s'


async def application(scope, receive, send):
    assert scope["type"] == "http"
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": b"ok"})


def serve(log_config):
    config = uvicorn.Config(application, host="127.0.0.1", port=0, log_config=log_config, log_level="info")
    access = logging.getLogger("uvicorn.access")
    handler = access.handlers[0]
    stream = io.StringIO()
    handler.setStream(stream)
    shapes = []

    class Shape(logging.Handler):
        def emit(self, record):
            shapes.append({"msg": record.msg, "arity": len(record.args)})

    shape = Shape()
    access.addHandler(shape)
    # A second handler with uvicorn's own formatter: whatever protects the
    # configured handler has to protect this one too.
    second = io.StringIO()
    stock_handler = logging.StreamHandler(second)
    stock_handler.setFormatter(AccessFormatter(fmt=STOCK_FORMAT))
    access.addHandler(stock_handler)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("uvicorn did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    lines = {}
    second_handler_lines = {}
    try:
        for label, target in TARGETS.items():
            stream.seek(0)
            stream.truncate(0)
            second.seek(0)
            second.truncate(0)
            response = httpx.get(f"http://127.0.0.1:{port}{target}", timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"{label} returned {response.status_code}")
            handler.flush()
            stock_handler.flush()
            lines[label] = stream.getvalue().strip()
            second_handler_lines[label] = second.getvalue().strip()
    finally:
        access.removeHandler(shape)
        access.removeHandler(stock_handler)
        server.should_exit = True
        thread.join(timeout=60)
    return {"lines": lines, "second_handler_lines": second_handler_lines, "shapes": shapes}


# Stock uvicorn goes first: ``dictConfig`` replaces a logger's handlers but
# keeps its filters, so a later stock configuration in this same process would
# inherit the redaction filter an Eyvan configuration installed.
print(
    json.dumps(
        {
            "stock_uvicorn": serve(LOGGING_CONFIG),
            "control_plane": serve(server_log_config(ServerLogLevel.INFO)),
            "cell_agent": serve(uvicorn_log_config()),
            "native_worker": serve(uvicorn_log_config()),
        }
    )
)
"""


@pytest.fixture(scope="module")
def served_access_log() -> dict[str, tp.Any]:
    """Serve every shipped logging configuration and return what it logged."""

    targets = {
        "cursor": CURSOR_TARGET,
        "empty_cursor": f"{LOG_PAGE_PATH}?after=&limit=200",
        "encoded_name": f"{LOG_PAGE_PATH}?%61fter={CURSOR}",
        "benign": "/api/v1/workloads?org_id=acme&project_id=research&state=running",
        "no_query": "/api/v1/workloads/wl-4213",
    }
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(targets)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("service", ("control_plane", "cell_agent", "native_worker"))
def test_a_served_log_page_request_never_writes_its_cursor_to_the_access_log(
    served_access_log: dict[str, tp.Any], service: str
) -> None:
    line = served_access_log[service]["lines"]["cursor"]

    assert LOG_PAGE_PATH in line
    assert "GET" in line
    assert "200" in line
    assert f"after={REDACTED_QUERY_VALUE}" in line
    _assert_cursor_absent(line)


@pytest.mark.parametrize("service", ("control_plane", "cell_agent", "native_worker"))
def test_a_served_request_keeps_its_benign_query_parameters(served_access_log: dict[str, tp.Any], service: str) -> None:
    lines = served_access_log[service]["lines"]

    assert "org_id=acme" in lines["benign"]
    assert "project_id=research" in lines["benign"]
    assert "state=running" in lines["benign"]
    # The redacted request keeps every parameter that is not on the denylist.
    assert "limit=200" in lines["cursor"]
    assert "state=running" in lines["cursor"]
    assert lines["no_query"].endswith('HTTP/1.1" 200 OK')


@pytest.mark.parametrize("service", ("control_plane", "cell_agent", "native_worker"))
def test_a_served_percent_encoded_cursor_parameter_is_still_redacted(
    served_access_log: dict[str, tp.Any], service: str
) -> None:
    # ``%61fter`` binds to ``after`` on the server, so it must redact like it.
    _assert_cursor_absent(served_access_log[service]["lines"]["encoded_name"])


@pytest.mark.parametrize("service", ("control_plane", "cell_agent", "native_worker"))
def test_a_served_empty_cursor_stays_visibly_empty(served_access_log: dict[str, tp.Any], service: str) -> None:
    # An empty ``after`` is the documented read-from-the-start signal and holds
    # no position, so a placeholder there would only hide which reads restarted.
    line = served_access_log[service]["lines"]["empty_cursor"]

    assert "after=" in line
    assert REDACTED_QUERY_VALUE not in line


@pytest.mark.parametrize("service", ("control_plane", "cell_agent", "native_worker"))
def test_a_second_access_handler_with_uvicorn_s_formatter_is_redacted_too(
    served_access_log: dict[str, tp.Any], service: str
) -> None:
    # Redaction is a property of the record, not of one formatter: an operator
    # who adds a JSON access handler must not reintroduce the leak.
    line = served_access_log[service]["second_handler_lines"]["cursor"]

    assert LOG_PAGE_PATH in line
    assert f"after={REDACTED_QUERY_VALUE}" in line
    _assert_cursor_absent(line)


def test_stock_uvicorn_access_logging_is_what_leaks_the_cursor(served_access_log: dict[str, tp.Any]) -> None:
    # Without this change the same request writes the whole cursor, so the
    # assertions above are answering a real difference rather than a tautology.
    stock = served_access_log["stock_uvicorn"]
    assert CURSOR in stock["lines"]["cursor"]
    assert LOG_PAGE_PATH in stock["lines"]["cursor"]
    assert CURSOR in stock["second_handler_lines"]["cursor"]


def test_every_served_configuration_sees_uvicorn_s_single_access_record_shape(
    served_access_log: dict[str, tp.Any],
) -> None:
    # The in-process helper below reproduces uvicorn's access record. This binds
    # that reproduction to the record uvicorn actually emits.
    shapes = {shape["msg"]: shape["arity"] for service in served_access_log.values() for shape in service["shapes"]}

    assert shapes == {_ACCESS_MESSAGE: _ACCESS_ARITY}


_ACCESS_MESSAGE = '%s - "%s %s HTTP/%s" %d'
_ACCESS_ARITY = 5
_ACCESS_FORMAT = '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'


def _access_line(log_config: JsonObject, target: str, *, status: int = 200) -> str:
    """Emit one uvicorn access record through the formatter a config ships.

    Args:
        log_config: A ``dictConfig`` document produced by Eyvan.
        target: The request target uvicorn would log.
        status: The response status code.

    Returns:
        The text the configured access formatter writes for that request.
    """

    formatters = log_config["formatters"]
    assert isinstance(formatters, dict)
    specification = dict(tp.cast("JsonObject", formatters["access"]))
    factory = logging.config.BaseConfigurator({}).resolve(str(specification.pop("()")))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(factory(**specification))
    logger = logging.getLogger(f"eyvan.tests.access.{uuid4().hex}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(_ACCESS_MESSAGE, "127.0.0.1:52000", "GET", target, "1.1", status)
    handler.flush()
    return stream.getvalue()


@pytest.mark.parametrize("parameter", sorted(REDACTED_QUERY_PARAMETERS))
def test_every_denied_parameter_is_redacted_by_every_shipped_configuration(parameter: str) -> None:
    sentinel = "Zq7-sentinel-value-Zq7"
    target = f"/api/v1/probe?{parameter}={sentinel}&limit=5"

    for log_config in (server_log_config(ServerLogLevel.INFO), uvicorn_log_config()):
        line = _access_line(log_config, target)

        assert "/api/v1/probe" in line
        assert "200" in line
        assert "limit=5" in line
        assert f"{parameter}={REDACTED_QUERY_VALUE}" in line
        assert sentinel not in line


def test_the_denylist_names_the_cursors_the_routes_actually_bind() -> None:
    # ``after`` is the log-page/log-stream cursor on all three services and
    # ``cursor`` is the artifact-inventory read position; both are resumable
    # positions, which is what DR-0002 keeps out of ordinary logs.
    assert CURSOR_QUERY_PARAMETERS == frozenset({"after", "cursor"})
    assert REDACTED_QUERY_PARAMETERS == CURSOR_QUERY_PARAMETERS | CREDENTIAL_QUERY_PARAMETERS
    assert "token" in CREDENTIAL_QUERY_PARAMETERS
    # ``state`` filters the workload list and ``name`` selects a secret mount:
    # both are ordinary request detail an operator reads the access log for.
    assert not REDACTED_QUERY_PARAMETERS & {"state", "name", "org_id", "project_id", "limit", "uri"}


@pytest.mark.parametrize(
    ("target", "expected"),
    (
        ("/api/v1/probe", "/api/v1/probe"),
        ("/api/v1/probe?", "/api/v1/probe?"),
        ("/api/v1/probe?limit=5", "/api/v1/probe?limit=5"),
        ("/api/v1/probe?after=", "/api/v1/probe?after="),
        (f"/api/v1/probe?after={CURSOR}", f"/api/v1/probe?after={REDACTED_QUERY_VALUE}"),
        (f"/api/v1/probe?AFTER={CURSOR}", f"/api/v1/probe?AFTER={REDACTED_QUERY_VALUE}"),
        (f"/api/v1/probe?a=1&after={CURSOR}&b=2", f"/api/v1/probe?a=1&after={REDACTED_QUERY_VALUE}&b=2"),
        ("/api/v1/probe?cursor=eyvan%3A%2F%2Fresults%2Fabc", f"/api/v1/probe?cursor={REDACTED_QUERY_VALUE}"),
    ),
)
def test_redaction_rewrites_only_the_values_it_has_to(target: str, expected: str) -> None:
    assert redact_request_target(target) == expected


def test_a_delimiter_bearing_cursor_is_redacted_whole() -> None:
    # DR-0002 keeps arbitrary leaf cursors inside a base64url envelope precisely
    # so delimiters survive the round trip; a percent-encoded one is one value.
    hostile = "attempt=3&rank=0&token=leaked#fragment"
    target = f"/api/v1/probe?after={quote(hostile, safe='')}&limit=5"

    line = _access_line(server_log_config(ServerLogLevel.INFO), target)

    assert f"after={REDACTED_QUERY_VALUE}" in line
    assert "rank" not in line
    assert "leaked" not in line
    assert "limit=5" in line


def test_an_unencoded_delimiter_redacts_every_denied_field_the_server_parses() -> None:
    # A raw ``&`` splits at the HTTP layer, so the server binds ``rank`` and
    # ``token`` as their own parameters. Redaction reads the same parse: the
    # cursor head and the credential go, the caller's stray field is left as the
    # ordinary parameter it became.
    target = "/api/v1/probe?after=attempt=3&rank=0&token=leaked&limit=5"

    line = _access_line(server_log_config(ServerLogLevel.INFO), target)

    assert f"after={REDACTED_QUERY_VALUE}" in line
    assert f"token={REDACTED_QUERY_VALUE}" in line
    assert "leaked" not in line
    assert "attempt" not in line
    assert "rank=0" in line
    assert "limit=5" in line


def test_a_non_access_record_behaves_exactly_as_uvicorn_s_own_formatter() -> None:
    # The redacting formatter must not invent a new failure mode for a record
    # uvicorn's access logger never emits.
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="plain lifecycle message",
        args=None,
        exc_info=None,
    )
    stock = AccessFormatter(fmt=_ACCESS_FORMAT)
    redacting = RedactingAccessFormatter(fmt=_ACCESS_FORMAT)

    with pytest.raises(Exception) as stock_failure:
        stock.format(record)
    with pytest.raises(type(stock_failure.value)):
        redacting.format(record)


def test_the_cell_agent_cli_serves_with_the_redacting_access_log(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI

    from eyvan.worker.agent import cli

    monkeypatch.delenv("EYVAN_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setattr("sys.argv", ["eyvan-agent", "--host", "127.0.0.1", "--allow-unauthenticated"])
    monkeypatch.setattr(cli, "create_agent_app", lambda **_options: FastAPI())
    served: list[dict[str, tp.Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda _application, **options: served.append(options))

    cli.main()

    (options,) = served
    line = _access_line(options["log_config"], CURSOR_TARGET)
    assert LOG_PAGE_PATH in line
    assert "200" in line
    _assert_cursor_absent(line)


def test_the_native_worker_cli_serves_with_the_redacting_access_log(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from eyvan.worker.runtime import native_worker

    monkeypatch.setenv("EYVAN_NATIVE_WORKER_SECRET", "worker-secret")
    served: list[dict[str, tp.Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda _application, **options: served.append(options))

    assert native_worker.main(("--state-dir", str(tmp_path))) == 0

    (options,) = served
    line = _access_line(options["log_config"], CURSOR_TARGET)
    assert LOG_PAGE_PATH in line
    assert "200" in line
    _assert_cursor_absent(line)


_OUTBOUND_PROBE = r"""
from __future__ import annotations

import json
import logging
import threading
import time

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from eyvan._internal.access_logging import uvicorn_log_config
from eyvan.server.api.server_logging import ServerLogLevel, server_log_config
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor

CURSOR = __import__("sys").argv[1]


async def application(scope, receive, send):
    body = b'{"records": [], "next_cursor": "", "unavailable": false, "detail": null, "truncated_ranks": []}'
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


def page(log_config):
    # ``log_level='critical'`` keeps uvicorn's own access line out of the way:
    # this measures what the outbound client logs, not what the server logs.
    config = uvicorn.Config(application, host="127.0.0.1", port=0, log_config=log_config, log_level="critical")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("uvicorn did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append({"logger": record.name, "level": record.levelname, "message": record.getMessage()})

    # Exactly what an operator does while chasing a stuck log page.
    handler = Capture()
    logging.root.addHandler(handler)
    previous = logging.root.level
    logging.root.setLevel(logging.DEBUG)
    try:
        executor = NativeRemoteExecutor(endpoint=f"http://127.0.0.1:{port}", secret="worker-secret")
        executor.log_page("remote-1", after=CURSOR, limit=200)
    finally:
        logging.root.removeHandler(handler)
        logging.root.setLevel(previous)
        server.should_exit = True
        thread.join(timeout=60)
    return seen


# Stock first, for the same filter-persistence reason as the inbound probe.
print(
    json.dumps(
        {
            "stock_uvicorn": page(LOGGING_CONFIG),
            "control_plane": page(server_log_config(ServerLogLevel.INFO)),
            "cell_agent": page(uvicorn_log_config()),
            "native_worker": page(uvicorn_log_config()),
        }
    )
)
"""


@pytest.fixture(scope="module")
def outbound_client_log() -> dict[str, tp.Any]:
    """Page logs through the real client under each configuration and capture root."""

    result = subprocess.run(
        [sys.executable, "-c", _OUTBOUND_PROBE, CURSOR],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _httpx_records(captured: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the outbound request records httpx emitted."""

    return [record for record in captured if record["logger"] == "httpx"]


@pytest.mark.parametrize("service", ("control_plane", "cell_agent", "native_worker"))
def test_the_outbound_log_page_request_never_logs_its_cursor(
    outbound_client_log: dict[str, tp.Any], service: str
) -> None:
    captured = outbound_client_log[service]
    (outbound,) = _httpx_records(captured)

    assert "/api/v1/workloads/remote-1/log-page" in outbound["message"]
    assert f"after={REDACTED_QUERY_VALUE}" in outbound["message"]
    assert "limit=200" in outbound["message"]
    _assert_cursor_absent(outbound["message"])
    # Nothing else that ran during the request may carry it either.
    for record in captured:
        _assert_cursor_absent(record["message"])


def test_the_outbound_client_leaks_the_cursor_without_the_filter(outbound_client_log: dict[str, tp.Any]) -> None:
    # The control: this is what an operator's ``logging.basicConfig(DEBUG)``
    # printed before this change, so the assertions above are a real difference.
    (outbound,) = _httpx_records(outbound_client_log["stock_uvicorn"])

    assert CURSOR in outbound["message"]


def test_the_outbound_request_line_is_logged_at_ordinary_verbosity(
    outbound_client_log: dict[str, tp.Any],
) -> None:
    # httpx logs the full URL at INFO, not DEBUG, which is why the filter and
    # not a level policy is what keeps the cursor out of it.
    (outbound,) = _httpx_records(outbound_client_log["stock_uvicorn"])

    assert outbound["level"] == "INFO"


def test_redaction_covers_every_logger_that_renders_a_url_during_a_log_page(
    outbound_client_log: dict[str, tp.Any],
) -> None:
    # The denylist of loggers is only defensible if it names every logger that
    # actually renders the URL. Anything else that logs during the request must
    # be shown not to carry the cursor at all.
    leaking = {record["logger"] for record in outbound_client_log["stock_uvicorn"] if CURSOR in record["message"]}

    assert leaking == {"httpx"}
    assert leaking <= set(REDACTED_URL_LOGGERS)
    # httpcore traces the same request and renders it as ``<Request [b'GET']>``.
    assert {record["logger"] for record in outbound_client_log["stock_uvicorn"]} > leaking


def test_the_filter_leaves_httpx_s_own_level_and_propagation_alone() -> None:
    # Filter-only entry: redaction must not quietly turn outbound request
    # logging on, off, or away from the root logger an operator configured.
    loggers = uvicorn_log_config()["loggers"]
    assert isinstance(loggers, dict)

    assert loggers["httpx"] == {"filters": ["redact_query"]}


def test_an_httpx_style_record_is_redacted_wherever_the_url_sits() -> None:
    # uvicorn puts the target third and httpx puts an URL object second, so the
    # redaction has to match on shape rather than on position.
    class Url:
        def __init__(self, value: str) -> None:
            self._value = value

        def __str__(self) -> str:
            return self._value

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(f"eyvan.tests.httpx.{uuid4().hex}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addFilter(QueryRedactionFilter())

    logger.info(
        'HTTP Request: %s %s "%s %d %s"',
        "GET",
        Url(f"http://cell-a:8090{LOG_PAGE_PATH}?after={CURSOR}&limit=200"),
        "HTTP/1.1",
        200,
        "OK",
    )
    handler.flush()
    line = stream.getvalue()

    assert LOG_PAGE_PATH in line
    assert f"after={REDACTED_QUERY_VALUE}" in line
    # The %d conversion still has an int to render.
    assert '"HTTP/1.1 200 OK"' in line
    _assert_cursor_absent(line)


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
