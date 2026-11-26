# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A fleet's bash init script: stored as one marked init command, run by real bash, kept across re-applies.

Observable outcomes only: the stored pool runtime, its generation, the audit
event (a digest, never the script), what real ``bash`` does with the rendered
entry, and the HTTP answers for admins, non-admins and attached TPUs.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from eyvan._internal import fleet_init_script as fis
from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
VIEWER = {"X-Eyvan-User": "bob", "X-Eyvan-Roles": "project_viewer", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}


def _pool(node_type: str, *, backend: str = "gcp-compute", init: tuple[str, ...] = ()) -> dict:
    pool = {
        "id": f"research/{node_type}",
        "fleet_name": "research",
        "node_type": node_type,
        "provider": "gcp",
        "capacity_backend": backend,
        "provider_project": "project-a",
        "region": "us-central1",
        "zone": "us-central1-a",
        "capacity": {"cpu": 8, "memory_gib": 32},
        "allocatable": {"cpu": 8, "memory_gib": 32},
        "accelerator_type": "cpu",
        # A pool runtime is only accepted for Eyvan-native cells.
        "executor": "eyvan-native",
        "agent_endpoint": "http://agent:8090",
        "provider_options": {"kind": "cpu", "machine_type": "n2-standard-8"},
        "runtime": {"init_commands": list(init)},
        "min_cells": 0,
        "max_cells": 2,
    }
    if backend == "gcp-attach":
        pool.update(
            capacity={"cpu": 208, "memory_gib": 448, "tpu": 4},
            allocatable={"cpu": 208, "memory_gib": 448, "tpu": 4},
            accelerator_type="v5p-8",
            provider_options={},
            max_cells=1,
        )
    return pool


def _client() -> TestClient:
    return TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True)))


def _apply(client: TestClient, key: str, *pools: dict) -> dict:
    response = client.put(
        "/api/v1/fleets/research", headers={**ADMIN, "Idempotency-Key": key}, json={"pools": list(pools)}
    )
    assert response.status_code == 200, response.text
    return {item["id"]: item for item in response.json()}


def _put(client: TestClient, script: str, key: str, headers: dict | None = None):
    return client.put(
        "/api/v1/fleets/research/init-script",
        headers={**(headers or ADMIN), "Idempotency-Key": key},
        json={"script": script},
    )


def test_render_extract_round_trip_and_refusals() -> None:
    script = 'set -e\necho "a $HOME"\ncat <<EOF\nnested\nEOF'
    entry = fis.render(fis.normalize(script))
    assert fis.is_entry(entry) and fis.extract(["echo yaml", entry]) == script
    assert fis.with_script(["echo yaml", entry], None) == ("echo yaml",)
    assert fis.with_script(["echo yaml"], "x")[-1] == fis.render("x")
    assert fis.normalize("a\r\nb  \n\n") == "a\nb"
    for bad in ("a\x00b", "x\nEND_OF_EYVAN_FLEET_INIT_SCRIPT\ny", "x" * (fis.MAX_SCRIPT_BYTES + 1)):
        with pytest.raises(fis.InitScriptError):
            fis.normalize(bad)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_rendered_entry_runs_the_script_under_real_bash_and_cannot_eat_the_bootstrap(tmp_path) -> None:
    out = tmp_path / "out"
    script = f"read -r stolen || stolen=none\necho \"ran $stolen\" > {out}\necho '$HOME stays literal' >> {out}"
    bootstrap = f"set -euo pipefail\n{fis.render(fis.normalize(script))}\necho after >> {out}\n"
    # The bootstrap itself arrives on stdin (as `sudo bash -s` does on attached hosts).
    # A trailing comment: if the script's `read` consumed bootstrap stdin it would
    # print "ran # ..." instead of "ran none".
    completed = subprocess.run(
        ["bash", "-s"],
        input=bootstrap + "# this line must not be read by the script\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert out.read_text().splitlines() == ["ran none", "$HOME stays literal", "after"]
    failing = f"set -euo pipefail\n{fis.render('exit 7')}\necho never >> {out}\n"
    assert subprocess.run(["bash", "-c", failing], check=False).returncode != 0


def test_admin_sets_reads_replaces_and_removes_the_script_across_every_pool() -> None:
    client = _client()
    before = _apply(client, "v1", _pool("cpu", init=("echo from-yaml",)), _pool("big"))
    assert client.get("/api/v1/fleets/research/init-script", headers=ADMIN).json()["script"] == ""

    first = _put(client, "pip install torch\necho ok", "s1")
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["script"] == "pip install torch\necho ok" and body["consistent"] and body["supported"]
    assert body["other_init_commands"] == ["echo from-yaml"]
    pools = {item["id"]: item for item in client.get("/api/v1/capacity-pools", headers=ADMIN).json()}
    for pool_id, pool in pools.items():
        commands = pool["runtime"]["init_commands"]
        assert fis.extract(commands) == "pip install torch\necho ok"
        assert pool["generation"] == before[pool_id]["generation"] + 1  # new cells; old ones are replaced
    assert pools["research/cpu"]["runtime"]["init_commands"][0] == "echo from-yaml"

    replay = _put(client, "pip install torch\necho ok", "s1")
    assert replay.status_code == 200 and replay.headers.get("Idempotent-Replay") == "true"
    same = _put(client, "pip install torch\necho ok", "s2")  # unchanged script: no generation bump
    assert same.status_code == 200
    again = {item["id"]: item for item in client.get("/api/v1/capacity-pools", headers=ADMIN).json()}
    assert all(again[key]["generation"] == pools[key]["generation"] for key in pools)

    removed = _put(client, "", "s3")
    assert removed.status_code == 200 and removed.json()["script"] == ""
    final = {item["id"]: item for item in client.get("/api/v1/capacity-pools", headers=ADMIN).json()}
    assert final["research/cpu"]["runtime"]["init_commands"] == ["echo from-yaml"]
    assert final["research/big"]["runtime"]["init_commands"] == []

    events = client.get("/api/v1/fleet/events", headers=ADMIN, params={"limit": 200})
    if events.status_code == 200:
        text = events.text
        assert "fleet.init_script_set" in text and "fleet.init_script_removed" in text
        assert "pip install torch" not in text  # a digest, never the script


def test_a_yaml_re_apply_keeps_the_dashboard_script_but_its_own_entry_wins() -> None:
    client = _client()
    _apply(client, "v1", _pool("cpu", init=("echo from-yaml",)))
    assert _put(client, "echo dashboard", "s1").status_code == 200
    kept = _apply(client, "v2", _pool("cpu", init=("echo from-yaml",)))
    assert fis.extract(kept["research/cpu"]["runtime"]["init_commands"]) == "echo dashboard"
    explicit = _apply(client, "v3", _pool("cpu", init=("echo from-yaml", fis.render("echo yaml-owned"))))
    assert fis.extract(explicit["research/cpu"]["runtime"]["init_commands"]) == "echo yaml-owned"


def test_non_admins_unknown_fleets_attached_tpus_and_bad_scripts_are_refused() -> None:
    client = _client()
    _apply(client, "v1", _pool("cpu"))
    assert client.get("/api/v1/fleets/research/init-script", headers=VIEWER).status_code == 403
    assert _put(client, "echo x", "n1", headers=VIEWER).status_code == 403
    assert client.get("/api/v1/fleets/nope/init-script", headers=ADMIN).status_code == 404
    bad = _put(client, "a\nEND_OF_EYVAN_FLEET_INIT_SCRIPT\nb", "b1")
    assert 400 <= bad.status_code < 500 and "END_OF_EYVAN_FLEET_INIT_SCRIPT" in bad.text

    attached = _client()
    _apply(attached, "a1", _pool("tpu", backend="gcp-attach"))
    shown = attached.get("/api/v1/fleets/research/init-script", headers=ADMIN).json()
    assert shown["supported"] is False and "eyvan gcp connect" in shown["unsupported_reasons"][0]
    refused = _put(attached, "echo x", "a2")
    assert 400 <= refused.status_code < 500 and "attached TPU" in refused.text
    assert _put(attached, "", "a3").status_code == 200  # removing nothing is always allowed
