# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""First fleet-admin bootstrap: host-side, refuses once accounts exist, prints only the link."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.server.api.app import create_app
from eyvan.server.api.bootstrap_admin import BootstrapError, bootstrap_admin
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError

PASSWORD = "correct horse battery"


def _local(tmp_path: Path, **overrides: object) -> EyvanSettings:
    return EyvanSettings(
        team_mode=True,
        repository_backend=RepositoryBackend.LOCAL,
        local_state_path=tmp_path / "control-plane.json",
        public_url="https://plane.example.com",
        artifact_token_key="k" * 32,
        **overrides,
    )


def test_service_bootstrap_creates_a_fleet_admin_invite_and_then_refuses() -> None:
    async def run() -> None:
        service = ControlPlaneService(InMemoryRepository())
        first, raw = await service.bootstrap_admin_invite(email="erfan@example.com", subject="erfan")
        assert [grant.role.value for grant in first.grants] == ["fleet_admin"]
        # Rerunning before acceptance replaces the lost link.
        second, raw2 = await service.bootstrap_admin_invite(email="erfan@example.com", subject="erfan")
        assert raw2 != raw
        assert await service.preview_invite(raw) is None
        assert (await service.preview_invite(raw2)).id == second.id
        await service.accept_invite(raw2, password=PASSWORD, session_ttl_s=60)
        with pytest.raises(ConflictError, match="already has team accounts"):
            await service.bootstrap_admin_invite(email="other@example.com", subject="other")

    asyncio.run(run())


def test_bootstrap_refuses_without_team_mode_or_with_memory_state(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="team mode is off"):
        asyncio.run(bootstrap_admin(EyvanSettings(), email="a@example.com", username="a"))
    with pytest.raises(BootstrapError, match="in memory"):
        asyncio.run(bootstrap_admin(EyvanSettings(team_mode=True), email="a@example.com", username="a"))


def test_bootstrap_refuses_while_a_local_plane_owns_the_state(tmp_path: Path) -> None:
    settings = _local(tmp_path)

    async def run() -> None:
        owner = LocalJsonRepository(settings.local_state_path)
        await owner.open()
        try:
            with pytest.raises(BootstrapError, match="stop it"):
                await bootstrap_admin(settings, email="a@example.com", username="a")
        finally:
            await owner.close()

    asyncio.run(run())


def test_the_bootstrap_link_signs_the_admin_in_on_the_real_plane(tmp_path: Path) -> None:
    settings = _local(tmp_path)
    result = asyncio.run(bootstrap_admin(settings, email="erfan@example.com", username="erfan"))
    assert result["url"].startswith("https://plane.example.com/#/accept-invite/eyvinv_")
    token = result["url"].rsplit("/", 1)[1]
    with TestClient(create_app(settings=settings), base_url="https://plane.example.com") as http:
        accepted = http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
        assert accepted.status_code == 200, accepted.text
        me = http.get("/api/v1/team/me")
        assert me.json()["fleet_admin"] is True and me.json()["subject"] == "erfan"


def test_eyvan_server_bootstrap_admin_runs_as_a_real_process(tmp_path: Path) -> None:
    state = tmp_path / "control-plane.json"
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("EYVAN_") and key != "PYTHONPATH"
    }
    environment.update(
        HOME=str(tmp_path / "home"),
        EYVAN_REPOSITORY_BACKEND="local",
        EYVAN_PUBLIC_URL="https://plane.example.com",
        EYVAN_PLATFORM_CONFIG_PATH=str(tmp_path / "absent-platform.yaml"),
    )
    command = [
        sys.executable,
        "-c",
        "from eyvan.server.api.app import main; main()",
        "bootstrap-admin",
        "--team",
        "--local-state-path",
        str(state),
        "--username",
        "Erfan",
    ]
    first = subprocess.run([*command, "--json"], env=environment, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stderr
    result = json.loads(first.stdout)
    assert result["username"] == "erfan" and result["email"] == "erfan@localhost.invalid"
    # Exactly one secret on stdout (the link), nothing on stderr.
    assert first.stdout.count("eyvinv_") == 1 and first.stderr == ""
    assert state.exists() and "eyvinv_" not in state.read_text()

    # Accept the invite through the real plane, then a second bootstrap is refused.
    token = result["url"].rsplit("/", 1)[1]
    settings = _local(tmp_path)
    with TestClient(create_app(settings=settings), base_url="https://plane.example.com") as http:
        assert http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD}).status_code == 200
    second = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    assert second.returncode == 1
    assert "already has team accounts" in second.stderr and "eyvinv_" not in second.stdout + second.stderr
