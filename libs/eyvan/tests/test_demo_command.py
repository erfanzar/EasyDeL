# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The one-command demo must run a real local worker and release its listeners."""

import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_a_callable_without_reading_or_changing_the_saved_context(tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "invalid-context.json"
    config.write_text("not valid json")
    result = subprocess.run(
        [sys.executable, "-m", "eyvan.sdk.cli", "demo", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            **os.environ,
            "HOME": str(home),
            "EYVAN_CONFIG_FILE": str(config),
            "EYVAN_URL": "https://not-the-demo.invalid",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(result.stdout)
    assert document["mode"] == "local-loopback"
    assert document["result"]["value"] == 45
    assert document["result"]["worker_pid"] != document["runner_pid"]
    assert document["state"] == "succeeded"
    assert document["cloud_resources_created"] == 0
    assert config.read_text() == "not valid json"
    for endpoint in (document["control_plane_url"], document["agent_url"]):
        parsed = urlsplit(endpoint)
        assert parsed.hostname == "127.0.0.1"
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex((parsed.hostname, parsed.port)) != 0


@pytest.mark.parametrize("failure", ["exception", "timeout"])
def test_a_failed_demo_callable_still_releases_both_loopback_listeners(monkeypatch, failure):
    from eyvan.sdk.client.jobs import EyvanJobError
    from eyvan.sdk.commands import demo

    endpoints = []
    original = demo._serve_loopback

    @contextmanager
    def observe(factory):
        with original(factory) as endpoint:
            endpoints.append(endpoint)
            yield endpoint

    def failing():
        if failure == "timeout":
            import time

            time.sleep(10)
        raise RuntimeError("intentional demo failure")

    monkeypatch.setattr(demo, "_serve_loopback", observe)
    monkeypatch.setattr(demo, "_sample_callable", failing)
    with pytest.raises(TimeoutError if failure == "timeout" else EyvanJobError):
        demo.run_demo(timeout=0.5 if failure == "timeout" else 20)
    assert len(endpoints) == 2
    for endpoint in endpoints:
        parsed = urlsplit(endpoint)
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex((parsed.hostname, parsed.port)) != 0
