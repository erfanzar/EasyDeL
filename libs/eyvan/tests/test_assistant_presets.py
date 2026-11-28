# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assistant provider presets, live model discovery and OpenRouter sign-in.

Proves: an admin adds a known provider with only a preset and a key; the URL is
the preset's and cannot be redirected; models come from the provider's live
``/models`` (or the preset's suggestions when that fails, without an error
status); a stored key is reused for discovery only against its own URL; the
OpenRouter PKCE sign-in uses S256, a single-use state, and seals the exchanged
key without ever returning it; non-admins can do none of it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import typing as tp
from urllib.parse import parse_qs, urlsplit

import httpx
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Principal
from eyvan.server.api.app import create_app
from eyvan.server.api.assistant_presets import PRESETS, OpenRouterSignIns

KEY = "sk-preset-secret-0123456789abcdef"
OR_KEY = "sk-or-exchanged-fedcba9876543210"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ALICE = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "project_submitter", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}


class FakeProviders:
    """Serves /models per host, the OpenRouter key exchange, and chat completions."""

    def __init__(self, models: dict[str, list[str]] | None = None, *, models_status: int = 200) -> None:
        self.models = models or {}
        self.models_status = models_status
        self.requests: list[httpx.Request] = []
        self.exchange: dict[str, tp.Any] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            if self.models_status != 200:
                return httpx.Response(self.models_status, text="nope")
            ids = self.models.get(request.url.host, [])
            return httpx.Response(200, json={"data": [{"id": item} for item in ids]})
        if str(request.url) == "https://openrouter.ai/api/v1/auth/keys":
            self.exchange = json.loads(request.content)
            return httpx.Response(200, json={"key": OR_KEY})
        body = json.loads(request.content)
        reply = {"role": "assistant", "content": f"model={body['model']}"}
        return httpx.Response(200, json={"choices": [{"message": reply}]})


def _http(fake: FakeProviders) -> TestClient:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, team_mode=True, artifact_token_key="k" * 32))
    app.state.assistant_transport = httpx.MockTransport(fake)
    return TestClient(app, base_url="http://127.0.0.1:8080", follow_redirects=False)


def test_presets_are_listed_for_everyone_with_fixed_urls() -> None:
    http = _http(FakeProviders())
    body = http.get("/api/v1/assistant/presets", headers=ALICE).json()
    ids = [item["id"] for item in body["presets"]]
    assert {"openrouter", "openai", "zai-coding", "kimi-code", "deepseek", "ollama", "custom"} <= set(ids)
    zai = next(item for item in body["presets"] if item["id"] == "zai-coding")
    assert zai["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert zai["key_required"] is True and zai["editable_url"] is False
    assert next(item for item in body["presets"] if item["id"] == "openrouter")["sign_in"] == "openrouter"
    assert len(ids) == len(set(ids)) == len(PRESETS)


def test_an_admin_adds_a_preset_with_only_a_key_and_chat_uses_it(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    fake = FakeProviders()
    http = _http(fake)
    created = http.post("/api/v1/assistant/providers", json={"preset": "zai-coding", "api_key": KEY}, headers=ADMIN)
    assert created.status_code == 201, created.text
    doc = created.json()
    assert doc["preset"] == "zai-coding" and doc["name"] == "Z.ai GLM Coding Plan"
    assert doc["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert doc["models"] == ["glm-4.6", "glm-4.5-air"] and doc["has_api_key"] is True
    assert KEY not in created.text

    second = http.post("/api/v1/assistant/providers", json={"preset": "zai-coding", "api_key": KEY}, headers=ADMIN)
    assert second.status_code == 201 and second.json()["name"] == "Z.ai GLM Coding Plan 2"

    chat = http.post(
        "/api/v1/assistant/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "org_id": "acme", "project_id": "r"},
        headers=ALICE,
    )
    assert chat.status_code == 200, chat.text
    assert str(fake.requests[-1].url) == "https://api.z.ai/api/coding/paas/v4/chat/completions"
    assert fake.requests[-1].headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in caplog.text


def test_presets_refuse_a_redirected_url_a_missing_key_and_non_admins() -> None:
    http = _http(FakeProviders())
    moved = http.post(
        "/api/v1/assistant/providers",
        json={"preset": "openai", "api_key": KEY, "base_url": "https://attacker.test/v1"},
        headers=ADMIN,
    )
    assert moved.status_code == 400 and "fixed base URL" in moved.text and KEY not in moved.text
    keyless = http.post("/api/v1/assistant/providers", json={"preset": "openai"}, headers=ADMIN)
    assert keyless.status_code == 400 and "needs an API key" in keyless.text
    unknown = http.post("/api/v1/assistant/providers", json={"preset": "nope", "api_key": KEY}, headers=ADMIN)
    assert unknown.status_code == 400
    assert (
        http.post("/api/v1/assistant/providers", json={"preset": "openai", "api_key": KEY}, headers=ALICE).status_code
        == 403
    )
    assert http.post("/api/v1/assistant/models/discover", json={"preset": "openai"}, headers=ALICE).status_code == 403
    assert http.post("/api/v1/assistant/oauth/openrouter/start", json={}, headers=ALICE).status_code == 403
    custom = http.post("/api/v1/assistant/providers", json={"preset": "custom", "models": ["m"]}, headers=ADMIN)
    assert custom.status_code == 400 and "base_url is required" in custom.text


def test_local_presets_need_no_key_and_take_a_url() -> None:
    fake = FakeProviders({"gpu-box.lan": ["llama3.1:8b", "qwen2.5:14b"]})
    http = _http(fake)
    created = http.post(
        "/api/v1/assistant/providers",
        json={"preset": "ollama", "base_url": "http://gpu-box.lan:11434/v1"},
        headers=ADMIN,
    )
    assert created.status_code == 201, created.text
    assert created.json()["models"] == ["llama3.1:8b", "qwen2.5:14b"]  # no suggestions: discovered live
    assert created.json()["has_api_key"] is False and created.json()["name"] == "Ollama"
    assert "authorization" not in fake.requests[-1].headers


def test_discovery_is_live_falls_back_to_suggestions_and_reuses_the_stored_key() -> None:
    fake = FakeProviders({"api.deepseek.com": ["deepseek-reasoner", "deepseek-chat", "models/x", "bad id!"]})
    http = _http(fake)
    live = http.post("/api/v1/assistant/models/discover", json={"preset": "deepseek", "api_key": KEY}, headers=ADMIN)
    assert live.json() == {"models": ["deepseek-chat", "deepseek-reasoner", "x"], "source": "live", "detail": None}
    assert fake.requests[-1].headers["authorization"] == f"Bearer {KEY}" and KEY not in live.text

    created = http.post(
        "/api/v1/assistant/providers",
        json={"preset": "deepseek", "api_key": KEY, "models": ["deepseek-chat"]},
        headers=ADMIN,
    ).json()
    stored = http.post("/api/v1/assistant/models/discover", json={"provider_id": created["id"]}, headers=ADMIN)
    assert stored.json()["source"] == "live" and KEY not in stored.text
    assert str(fake.requests[-1].url) == "https://api.deepseek.com/v1/models"
    assert fake.requests[-1].headers["authorization"] == f"Bearer {KEY}"

    fake.models_status = 401
    rejected = http.post(
        "/api/v1/assistant/models/discover", json={"preset": "deepseek", "api_key": "bad"}, headers=ADMIN
    )
    assert rejected.status_code == 200
    assert rejected.json() == {
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "source": "suggested",
        "detail": "the provider rejected the API key",
    }


def _start(http: TestClient, **body: tp.Any) -> tuple[str, dict[str, list[str]]]:
    started = http.post("/api/v1/assistant/oauth/openrouter/start", json=body, headers=ADMIN)
    assert started.status_code == 200, started.text
    url = urlsplit(started.json()["authorize_url"])
    assert f"{url.scheme}://{url.netloc}{url.path}" == "https://openrouter.ai/auth"
    query = parse_qs(url.query)
    return urlsplit(query["callback_url"][0]).path, query


def test_openrouter_sign_in_is_pkce_single_use_and_seals_the_key(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    fake = FakeProviders()
    http = _http(fake)
    callback, query = _start(http, models=["z-ai/glm-4.6", "openai/gpt-4.1"], default_model="openai/gpt-4.1")
    assert query["code_challenge_method"] == ["S256"]
    assert callback.startswith("/api/v1/assistant/oauth/openrouter/callback/")

    finished = http.get(callback, params={"code": "auth-code-1"})
    assert finished.status_code == 303
    assert finished.headers["location"] == "/#/settings?assistant=openrouter-connected"
    assert fake.exchange is not None and fake.exchange["code"] == "auth-code-1"
    verifier = fake.exchange["code_verifier"]
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == [expected]

    providers = http.get("/api/v1/assistant/providers", headers=ADMIN)
    [provider] = providers.json()["providers"]
    assert provider["preset"] == "openrouter" and provider["name"] == "OpenRouter"
    assert provider["models"] == ["z-ai/glm-4.6", "openai/gpt-4.1"] and provider["default_model"] == "openai/gpt-4.1"
    assert provider["has_api_key"] is True and OR_KEY not in providers.text

    replay = http.get(callback, params={"code": "auth-code-1"})
    assert replay.headers["location"] == "/#/settings?assistant_error=expired"
    assert len(http.get("/api/v1/assistant/providers", headers=ADMIN).json()["providers"]) == 1

    http.post(
        "/api/v1/assistant/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "org_id": "acme", "project_id": "r"},
        headers=ALICE,
    )
    assert fake.requests[-1].headers["authorization"] == f"Bearer {OR_KEY}"
    assert OR_KEY not in caplog.text and verifier not in caplog.text


def test_openrouter_denial_and_unknown_state_redirect_with_an_error() -> None:
    http = _http(FakeProviders())
    callback, _ = _start(http)
    denied = http.get(callback, params={"error": "access_denied"})
    assert denied.headers["location"] == "/#/settings?assistant_error=denied"
    forged = http.get("/api/v1/assistant/oauth/openrouter/callback/forged-state", params={"code": "x"})
    assert forged.headers["location"] == "/#/settings?assistant_error=expired"
    assert http.get("/api/v1/assistant/providers", headers=ADMIN).json()["providers"] == []


def test_pending_sign_ins_expire() -> None:
    now = [0.0]
    sign_ins = OpenRouterSignIns(clock=lambda: now[0], ttl_s=60)
    principal = Principal(subject="root")
    state, _ = sign_ins.begin(principal, models=(), default_model=None, name=None)
    now[0] = 61.0
    assert sign_ins.take(state) is None
    state, _ = sign_ins.begin(principal, models=(), default_model=None, name=None)
    assert sign_ins.take(state) is not None and sign_ins.take(state) is None
    assert "verifier" not in repr(sign_ins)
