# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OIDC callback secrets must not reach any control-plane access handler."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_LOGGING_PROBE = r"""
import io
import json
import logging
import logging.config
import sys

import httpx
from uvicorn.logging import AccessFormatter

from eyvan.server.api.server_logging import ServerLogLevel, server_log_config

configuration = server_log_config(ServerLogLevel.INFO)
logging.config.dictConfig(configuration)
access = logging.getLogger("uvicorn.access")
configured_stream = io.StringIO()
access.handlers[0].setStream(configured_stream)
stock_stream = io.StringIO()
stock_handler = logging.StreamHandler(stock_stream)
stock_handler.setFormatter(AccessFormatter(fmt='%(client_addr)s - "%(request_line)s" %(status_code)s'))
access.addHandler(stock_handler)
records = []

class Capture(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())

access.addHandler(Capture())
target = sys.argv[1]
access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:50000", "GET", target, "1.1", 307)

# Formatter-only use must also sanitize callbacks if a logger has no filter.
formatter_spec = dict(configuration["formatters"]["access"])
factory = logging.config.BaseConfigurator({}).resolve(formatter_spec.pop("()"))
formatter = factory(**formatter_spec)
record = logging.LogRecord("probe", logging.INFO, "probe.py", 1, '%s - "%s %s HTTP/%s" %d',
                           ("127.0.0.1:50000", "GET", target, "1.1", 307), None)
formatter_line = formatter.format(record)

# httpx's URL argument goes through the same configured logger-level filter;
# no HTTP request or socket is involved.
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.INFO)
httpx_logger.addHandler(Capture())
httpx_logger.info('HTTP Request: %s %s "%s %d %s"', "GET", httpx.URL("https://plane.example" + target),
                  "HTTP/1.1", 307, "Temporary Redirect")
print(json.dumps({"configured": configured_stream.getvalue(), "stock": stock_stream.getvalue(),
                  "records": records, "formatter": formatter_line}))
"""


def _logged_target(target: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-B", "-c", _LOGGING_PROBE, target],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "target",
    (
        "/api/v1/dashboard/auth/callback?code=PRIVATE_AUTH_CODE&state=PRIVATE_LOGIN_STATE&access_token=PRIVATE_TOKEN",
        "/api/v1/dashboard/auth/callback?error=PRIVATE_UNKNOWN_ERROR&error_description=PRIVATE_PROVIDER_DETAIL"
        "&state=PRIVATE_LOGIN_STATE",
        "/api/v1/dashboard/auth/callback?%63ode=PRIVATE_AUTH_CODE&STATE=PRIVATE_LOGIN_STATE"
        "&error_description=PRIVATE_PROVIDER_DETAIL&error_description=PRIVATE_DUPLICATE_DETAIL",
        "/mounted/api/v1/dashboard/auth/callback/?code=PRIVATE_AUTH_CODE&state=PRIVATE_LOGIN_STATE",
        "/api/v1/dashboard/auth/%63allback?code=PRIVATE_AUTH_CODE&state=PRIVATE_LOGIN_STATE",
    ),
)
def test_shipped_callback_logging_redacts_before_every_handler(target: str) -> None:
    output = _logged_target(target)
    rendered = json.dumps(output)
    assert "PRIVATE_" not in rendered, rendered
    assert "REDACTED" in rendered
    assert "callback" in str(output["configured"]) or "%63allback" in str(output["configured"])
    assert "307" in str(output["configured"])
    assert len(output["records"]) == 2


def test_non_callback_workload_state_and_existing_token_redaction_are_preserved() -> None:
    output = _logged_target("/api/v1/workloads?state=running&limit=5&access_token=PRIVATE_TOKEN")
    for line in (output["configured"], output["stock"], output["formatter"], *output["records"]):
        assert "state=running" in line
        assert "limit=5" in line
        assert "PRIVATE_TOKEN" not in line
        assert "access_token=REDACTED" in line


def test_callback_without_query_keeps_the_request_path() -> None:
    output = _logged_target("/api/v1/dashboard/auth/callback")
    assert "/api/v1/dashboard/auth/callback HTTP/1.1" in str(output["configured"])
    assert "REDACTED" not in json.dumps(output)
