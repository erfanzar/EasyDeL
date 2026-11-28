# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assistant providers configured by fleet admins from the dashboard/API.

Proves: only fleet admins manage providers; everyone else sees names and models
only; the API key is write-only (never in a response, an audit event, a log or a
stored document in plaintext) in every backend; a user's provider/model choice
is honoured and an unknown or disabled one refused; a key sealed under another
server secret fails readably instead of leaking or crashing.
"""

from __future__ import annotations

import json
import logging
import typing as tp

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.identity.assistant_keys import AssistantKeyError, AssistantKeySealer
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, InputError, NotFoundError

KEY = "sk-dashboard-secret-0123456789abcdef"
OTHER_KEY = "sk-second-secret-fedcba9876543210"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ALICE = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "project_submitter", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
SCOPE = {"org_id": "acme", "project_id": "r"}
ADMIN_PRINCIPAL = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
ALICE_PRINCIPAL = Principal(
    subject="alice", grants=(RoleGrant(role=Role.PROJECT_SUBMITTER, org_id="acme", project_id="r"),)
)


class Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})


def _http(recorder: Recorder | None = None) -> TestClient:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, team_mode=True, artifact_token_key="k" * 32))
    app.state.assistant_transport = httpx.MockTransport(recorder or Recorder())
    return TestClient(app, base_url="http://127.0.0.1:8080")


def _add(http: TestClient, **fields: tp.Any) -> dict[str, tp.Any]:
    body = {"name": "alpha", "base_url": "http://alpha.test/v1", "api_key": KEY, "models": ["a-small", "a-large"]}
    body.update(fields)
    response = http.post("/api/v1/assistant/providers", json=body, headers=ADMIN)
    assert response.status_code == 201, response.text
    return response.json()


def _chat(http: TestClient, **choice: tp.Any) -> httpx.Response:
    body = {"messages": [{"role": "user", "content": "hi"}], **SCOPE, **choice}
    return http.post("/api/v1/assistant/chat", json=body, headers=ALICE)


def test_only_fleet_admins_manage_providers_and_the_key_is_never_returned(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    http = _http()
    body = {"name": "alpha", "base_url": "http://alpha.test/v1", "api_key": KEY, "models": ["m"]}
    assert http.post("/api/v1/assistant/providers", json=body, headers=ALICE).status_code == 403
    created = _add(http)
    assert created["has_api_key"] is True and created["default_model"] == "a-small"
    responses = [json.dumps(created)]

    admin_view = http.get("/api/v1/assistant/providers", headers=ADMIN)
    assert admin_view.json()["can_manage"] is True
    assert admin_view.json()["providers"][0]["base_url"] == "http://alpha.test/v1"
    user_view = http.get("/api/v1/assistant/providers", headers=ALICE)
    assert user_view.json() == {
        "providers": [
            {
                "id": created["id"],
                "name": "alpha",
                "preset": None,
                "models": ["a-small", "a-large"],
                "default_model": "a-small",
            }
        ],
        "can_manage": False,
    }
    responses += [admin_view.text, user_view.text]

    pid = created["id"]
    assert http.put(f"/api/v1/assistant/providers/{pid}", json={"name": "x"}, headers=ALICE).status_code == 403
    assert http.delete(f"/api/v1/assistant/providers/{pid}", headers=ALICE).status_code == 403
    renamed = http.put(f"/api/v1/assistant/providers/{pid}", json={"name": "alpha2"}, headers=ADMIN)
    assert renamed.status_code == 200 and renamed.json()["has_api_key"] is True  # omitted key is kept
    responses.append(renamed.text)

    events = http.get("/api/v1/events/fleet", headers=ADMIN)
    if events.status_code == 200:
        responses.append(events.text)
    for text_ in responses:
        assert KEY not in text_ and "v1:" not in text_.replace('"http://alpha.test/v1"', "")
    assert KEY not in caplog.text


def test_a_validation_error_does_not_echo_a_submitted_key() -> None:
    http = _http()
    bad = http.post(
        "/api/v1/assistant/providers",
        json={"name": "x", "base_url": "ftp://nope", "api_key": KEY, "models": ["m"]},
        headers=ADMIN,
    )
    assert bad.status_code == 400 and KEY not in bad.text
    spaced = http.post(
        "/api/v1/assistant/providers",
        json={"name": "x", "base_url": "http://x/v1", "api_key": f"{KEY} extra", "models": ["m"]},
        headers=ADMIN,
    )
    assert spaced.status_code == 400 and KEY not in spaced.text
    assert (
        http.post(
            "/api/v1/assistant/providers", json={"name": "x", "base_url": "http://x/v1", "models": []}, headers=ADMIN
        ).status_code
        == 422
    )


def test_chat_uses_the_chosen_provider_and_model_with_its_own_key() -> None:
    recorder = Recorder()
    http = _http(recorder)
    first = _add(http)
    second = _add(http, name="beta", base_url="http://beta.test/v1", api_key=OTHER_KEY, models=["b-1"])

    default = _chat(http)
    assert default.status_code == 200, default.text
    assert default.json()["provider_id"] == first["id"] and default.json()["model"] == "a-small"
    assert str(recorder.requests[-1].url) == "http://alpha.test/v1/chat/completions"
    assert recorder.requests[-1].headers["authorization"] == f"Bearer {KEY}"

    chosen = _chat(http, provider_id=second["id"], model="b-1")
    assert chosen.status_code == 200, chosen.text
    assert str(recorder.requests[-1].url) == "http://beta.test/v1/chat/completions"
    assert recorder.requests[-1].headers["authorization"] == f"Bearer {OTHER_KEY}"
    assert json.loads(recorder.requests[-1].content)["model"] == "b-1"
    assert KEY not in chosen.text and OTHER_KEY not in chosen.text

    large = _chat(http, provider_id=first["id"], model="a-large")
    assert large.status_code == 200 and json.loads(recorder.requests[-1].content)["model"] == "a-large"


def test_unknown_or_disabled_providers_and_unlisted_models_are_refused() -> None:
    recorder = Recorder()
    http = _http(recorder)
    created = _add(http)
    assert _chat(http, provider_id=created["id"], model="gpt-anything").status_code == 400
    assert _chat(http, provider_id="aprv_0000000000000000").status_code == 404
    off = http.put(f"/api/v1/assistant/providers/{created['id']}", json={"enabled": False}, headers=ADMIN)
    assert off.status_code == 200 and off.json()["enabled"] is False
    assert _chat(http, provider_id=created["id"]).status_code == 404
    assert _chat(http).status_code == 404
    assert http.get("/api/v1/assistant/providers", headers=ALICE).json()["providers"] == []
    assert http.get("/api/v1/assistant/status", headers=ALICE).json() == {
        "enabled": False,
        "model": None,
        "providers": 0,
    }
    assert len(http.get("/api/v1/assistant/providers", headers=ADMIN).json()["providers"]) == 1
    assert recorder.requests == []


def test_key_can_be_replaced_and_cleared_and_a_provider_deleted() -> None:
    recorder = Recorder()
    http = _http(recorder)
    pid = _add(http)["id"]
    assert http.put(f"/api/v1/assistant/providers/{pid}", json={"api_key": OTHER_KEY}, headers=ADMIN).status_code == 200
    _chat(http)
    assert recorder.requests[-1].headers["authorization"] == f"Bearer {OTHER_KEY}"
    cleared = http.put(f"/api/v1/assistant/providers/{pid}", json={"clear_api_key": True}, headers=ADMIN)
    assert cleared.json()["has_api_key"] is False
    _chat(http)
    assert "authorization" not in recorder.requests[-1].headers
    both = http.put(f"/api/v1/assistant/providers/{pid}", json={"clear_api_key": True, "api_key": KEY}, headers=ADMIN)
    assert both.status_code == 400
    duplicate = http.post(
        "/api/v1/assistant/providers",
        json={"name": "ALPHA", "base_url": "http://x/v1", "models": ["m"]},
        headers=ADMIN,
    )
    assert duplicate.status_code == 409
    assert http.delete(f"/api/v1/assistant/providers/{pid}", headers=ADMIN).json() == {"deleted": pid}
    assert http.get("/api/v1/assistant/providers", headers=ADMIN).json()["providers"] == []
    assert http.delete(f"/api/v1/assistant/providers/{pid}", headers=ADMIN).status_code == 404


def test_a_key_sealed_under_another_server_secret_fails_readably() -> None:
    sealer = AssistantKeySealer(b"a" * 32)
    stored = sealer.seal("aprv_0123456789abcdef", KEY)
    assert KEY not in stored and sealer.open("aprv_0123456789abcdef", stored) == KEY
    with pytest.raises(AssistantKeyError) as wrong_secret:
        AssistantKeySealer(b"b" * 32).open("aprv_0123456789abcdef", stored)
    with pytest.raises(AssistantKeyError):
        sealer.open("aprv_ffffffffffffffff", stored)  # ciphertext bound to its provider id
    assert KEY not in str(wrong_secret.value) and stored not in str(wrong_secret.value)
    assert "<redacted>" in repr(sealer)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "local", "sql"])
async def test_providers_persist_with_the_key_only_as_ciphertext(backend: str, tmp_path) -> None:
    engine = None
    if backend == "memory":
        repository: tp.Any = InMemoryRepository()
    elif backend == "local":
        repository = LocalJsonRepository(tmp_path / "control-plane.json")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'plane.db'}")
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
    await repository.open()
    sealer = AssistantKeySealer(b"s" * 32)
    service = ControlPlaneService(repository)
    service.assistant_sealer = sealer
    with pytest.raises(AuthorizationError):
        await service.create_assistant_provider(ALICE_PRINCIPAL, name="x", base_url="http://x/v1", models=["m"])
    created = await service.create_assistant_provider(
        ADMIN_PRINCIPAL, name="alpha", base_url="http://alpha.test/v1/", models=["m1", " m2 ", "m1"], api_key=KEY
    )
    assert created.models == ("m1", "m2") and created.base_url == "http://alpha.test/v1"
    with pytest.raises(InputError):
        await service.update_assistant_provider(ADMIN_PRINCIPAL, created.id, default_model="not-listed")
    async with repository.transaction() as tx:
        events = await tx.list_events(entity_type="assistant_provider", entity_id=created.id, limit=100)
    assert [event.action for event in events] == ["assistant_provider.created"]
    assert events[0].payload["api_key_set"] is True
    assert KEY not in json.dumps([event.model_dump(mode="json") for event in events])
    await repository.close()

    if backend == "memory":
        reopened = repository
        raw = json.dumps(created.model_dump(mode="json"))
    elif backend == "local":
        reopened = LocalJsonRepository(tmp_path / "control-plane.json")
        raw = (tmp_path / "control-plane.json").read_text()
        assert json.loads(raw)["version"] == 9
    else:
        assert engine is not None
        reopened = SqlAlchemyRepository(engine)
        async with engine.connect() as connection:
            raw = json.dumps([row[0] for row in await connection.execute(text("SELECT data FROM assistant_providers"))])
    assert KEY not in raw and "api_key_ciphertext" in raw
    await reopened.open()
    second = ControlPlaneService(reopened)
    second.assistant_sealer = sealer
    record, model, key = await second.resolve_assistant_provider(ALICE_PRINCIPAL, provider_id=None, model=None)
    assert (record.id, model, key) == (created.id, "m1", KEY)
    with pytest.raises(NotFoundError):
        await second.resolve_assistant_provider(ALICE_PRINCIPAL, provider_id="aprv_0000000000000000", model=None)
    await reopened.close()
    if engine is not None:
        await engine.dispose()
