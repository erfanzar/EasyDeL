# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Black-box regression tests for the disposable browser BFF fixture."""

from __future__ import annotations

import json
import os
import selectors
import ssl
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

from eyvan.server.api.dashboard_sessions import SESSION_COOKIE

FIXTURE = Path(__file__).with_name("browser_bff_server.py")


def _ssl_context(certificate: Path) -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=str(certificate))
    return context


def _start(
    state_dir: Path, *, fixture: Path = FIXTURE, timeout: float = 15
) -> tuple[subprocess.Popen[bytes], dict[str, str], Path]:
    process = subprocess.Popen(
        [sys.executable, str(fixture), "--state-dir", str(state_dir)],
        cwd=FIXTURE.parents[1],
        env={
            **os.environ,
            "EYVAN_DATABASE_URL": "postgresql://should-not-be-read",
            "EYVAN_REPOSITORY_BACKEND": "postgres",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = bytearray()
    errors = bytearray()
    ready_line: bytes | None = None
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            for key, _ in selector.select(max(0, deadline - time.monotonic())):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    output.extend(chunk)
                    if b"\n" in output:
                        ready_line, _, _ = output.partition(b"\n")
                        break
                else:
                    errors.extend(chunk)
            if ready_line is not None or process.poll() is not None:
                break
    except BaseException:
        _stop(process)
        raise
    finally:
        selector.close()
    if ready_line is None:
        _stop(process)
        raise AssertionError(
            f"fixture did not become ready (exit={process.returncode}): {errors.decode(errors='replace')}"
        )
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError as error:
        _stop(process)
        raise AssertionError(f"fixture printed invalid readiness JSON: {ready_line!r}") from error
    return process, ready, state_dir / "certificate.pem"


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def test_fixture_is_real_https_process_with_pkce_and_shutdown(tmp_path: Path) -> None:
    state_dir = tmp_path / "runtime"
    process, ready, certificate = _start(state_dir)
    try:
        assert process.poll() is None
        assert set(ready) == {"plane", "issuer"}
        assert urlsplit(ready["plane"]).scheme == "https"
        assert urlsplit(ready["issuer"]).scheme == "https"
        assert urlsplit(ready["plane"]).port != urlsplit(ready["issuer"]).port
        assert certificate.is_file()

        with httpx.Client(verify=_ssl_context(certificate), follow_redirects=True, trust_env=False) as client:
            login = client.get(f"{ready['plane']}/api/v1/dashboard/auth/login", follow_redirects=False)
            assert login.status_code in (302, 307)
            authorization = login.headers["location"]
            query = parse_qs(urlsplit(authorization).query)
            assert query["code_challenge_method"] == ["S256"]
            assert "code_challenge" in query
            assert query["state"]
            assert query["nonce"]
            completed = client.get(authorization)
            assert completed.status_code in (200, 303, 307)
            assert SESSION_COOKIE in client.cookies
            authenticated = client.get(f"{ready['plane']}/api/v1/session")
            assert authenticated.status_code == 200

            logout = client.post(f"{ready['plane']}/api/v1/dashboard/auth/logout", headers={"Origin": ready["plane"]})
            assert logout.status_code == 204
            assert SESSION_COOKIE not in client.cookies
            assert client.get(f"{ready['plane']}/api/v1/session").status_code == 401
            # The fixture's has_cookie monitor recognizes only the legacy name;
            # verify canonical-cookie authority on the real API instead.
            observed = client.get(f"{ready['plane']}/qualification")
            assert observed.json()["logout_attempts"][-1]["origin"] == ready["plane"]
    finally:
        _stop(process)
    assert process.returncode == 0
    assert not certificate.exists()
    assert not (state_dir / "private-key.pem").exists()


def test_fixture_provider_denial_preserves_state(tmp_path: Path) -> None:
    process, ready, certificate = _start(tmp_path / "denial")
    try:
        with httpx.Client(verify=_ssl_context(certificate), trust_env=False, follow_redirects=False) as client:
            login = client.get(f"{ready['plane']}/api/v1/dashboard/auth/login")
            authorization = login.headers["location"]
            query = parse_qs(urlsplit(authorization).query)
            denied = client.get(f"{ready['issuer']}/authorize?deny=1&{urlsplit(authorization).query}")
            assert denied.status_code in (302, 307)
            callback = parse_qs(urlsplit(denied.headers["location"]).query)
            assert callback["state"] == query["state"]
            assert callback["error"] == ["access_denied"]
    finally:
        _stop(process)


def test_start_times_out_and_cleans_up_partial_output(tmp_path: Path) -> None:
    fake = tmp_path / "silent_fixture.py"
    pid_file = tmp_path / "child.pid"
    fake.write_text(
        "import os, sys, time\nfrom pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "sys.stdout.write('partial')\nsys.stdout.flush()\ntime.sleep(30)\n"
    )
    started = time.monotonic()
    try:
        _start(tmp_path / "runtime", fixture=fake, timeout=1)
    except AssertionError as error:
        assert "did not become ready" in str(error)
    else:
        raise AssertionError("silent fixture unexpectedly became ready")
    assert time.monotonic() - started < 5
    child_pid = int(pid_file.read_text())
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError(f"timed-out fixture child {child_pid} is still alive")


def test_fixture_refuses_nonempty_state_without_overwriting_it(tmp_path: Path) -> None:
    certificate = tmp_path / "certificate.pem"
    certificate.write_text("existing material must survive")
    result = subprocess.run(
        [sys.executable, str(FIXTURE), "--state-dir", str(tmp_path)],
        cwd=FIXTURE.parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0
    assert "state directory must be empty" in result.stderr
    assert certificate.read_text() == "existing material must survive"
    assert not (tmp_path / "private-key.pem").exists()


def test_fixture_requires_state_dir() -> None:
    result = subprocess.run(
        [sys.executable, str(FIXTURE)],
        cwd=FIXTURE.parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--state-dir" in result.stderr
