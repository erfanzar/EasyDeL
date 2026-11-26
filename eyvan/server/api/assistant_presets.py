# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Built-in assistant provider presets, live model discovery and OpenRouter sign-in.

A fleet admin should not have to know a provider's base URL or type model ids
by hand. A preset fixes the endpoint; the admin pastes a key (or signs in with
OpenRouter) and picks from the models the provider itself lists.

Model discovery calls the provider's OpenAI-compatible ``GET /models`` from the
plane. A key used for discovery is either the one just typed or the provider's
stored key; a stored key is only ever sent to that provider's own stored URL.

OpenRouter sign-in is OAuth PKCE (S256): the plane keeps the verifier and a
single-use, expiring state in memory, OpenRouter redirects the browser back to
the plane with a code, and the plane exchanges it for a user-controlled API key
that is sealed like any typed key. The key never reaches the browser.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import time
import typing as tp
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from eyvan.config import EyvanSettings
from eyvan.records import ASSISTANT_MODEL_PATTERN, CLAUDE_CODE_BASE_URL, Principal, utcnow
from eyvan.server.api.auth import authenticated_principal
from eyvan.server.service.errors import EyvanServiceError

if tp.TYPE_CHECKING:
    from eyvan.server.service import ControlPlaneService

logger = logging.getLogger(__name__)

_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]

#: How long one ``GET /models`` may take before the suggestions are offered instead.
DISCOVERY_TIMEOUT_S = 15.0
#: Upper bound on model ids returned to the dashboard from one discovery.
MAX_DISCOVERED_MODELS = 1000
#: A provider record offers at most this many models.
MAX_PROVIDER_MODELS = 32

OPENROUTER_AUTHORIZE_URL = "https://openrouter.ai/auth"
OPENROUTER_KEY_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
#: A sign-in the admin did not finish within this window must be started again.
SIGN_IN_TTL_S = 600
MAX_PENDING_SIGN_INS = 64
SETTINGS_REDIRECT = "/#/settings"
#: The preset answered by the plane host's own ``claude`` CLI rather than over HTTP.
CLAUDE_CODE_PRESET = "claude-code"


@dataclass(frozen=True, slots=True)
class AssistantPreset:
    """One known OpenAI-compatible provider."""

    id: str
    label: str
    base_url: str
    key_required: bool = True
    editable_url: bool = False
    key_url: str | None = None
    sign_in: str | None = None
    suggested_models: tuple[str, ...] = ()

    def document(self) -> dict[str, tp.Any]:
        return {
            "id": self.id,
            "label": self.label,
            "base_url": self.base_url,
            "editable_url": self.editable_url,
            "key_required": self.key_required,
            "key_url": self.key_url,
            "sign_in": self.sign_in,
            "suggested_models": list(self.suggested_models),
        }


#: Endpoints as each provider documents its OpenAI-compatible API. Suggested
#: models only pre-select a sensible few; the list the admin picks from comes
#: from the provider's live ``/models``.
PRESETS: tuple[AssistantPreset, ...] = (
    AssistantPreset(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        key_url="https://openrouter.ai/keys",
        sign_in="openrouter",
        suggested_models=("anthropic/claude-sonnet-4", "openai/gpt-4.1", "z-ai/glm-4.6", "moonshotai/kimi-k2"),
    ),
    # The plane host's own signed-in `claude` CLI: no URL, no key (see assistant_claude_code).
    AssistantPreset(
        "claude-code",
        "Claude Code",
        "",
        key_required=False,
        editable_url=False,
        key_url="https://claude.com/product/claude-code",
        sign_in="claude-code",
        suggested_models=("sonnet", "opus", "haiku"),
    ),
    AssistantPreset(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        key_url="https://platform.openai.com/api-keys",
        suggested_models=("gpt-4.1", "gpt-4.1-mini"),
    ),
    AssistantPreset(
        "zai-coding",
        "Z.ai GLM Coding Plan",
        "https://api.z.ai/api/coding/paas/v4",
        key_url="https://z.ai/manage-apikey/apikey-list",
        suggested_models=("glm-4.6", "glm-4.5-air"),
    ),
    AssistantPreset(
        "zai",
        "Z.ai API",
        "https://api.z.ai/api/paas/v4",
        key_url="https://z.ai/manage-apikey/apikey-list",
        suggested_models=("glm-4.6", "glm-4.5-air"),
    ),
    AssistantPreset(
        "bigmodel-coding",
        "BigModel GLM Coding Plan (China)",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        key_url="https://open.bigmodel.cn/usercenter/apikeys",
        suggested_models=("glm-4.6", "glm-4.5-air"),
    ),
    AssistantPreset(
        "kimi-code",
        "Kimi Code",
        "https://api.kimi.com/coding/v1",
        key_url="https://www.kimi.com/coding",
        suggested_models=("kimi-for-coding",),
    ),
    AssistantPreset(
        "moonshot",
        "Moonshot (Kimi API)",
        "https://api.moonshot.ai/v1",
        key_url="https://platform.moonshot.ai/console/api-keys",
    ),
    AssistantPreset(
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com/v1",
        key_url="https://platform.deepseek.com/api_keys",
        suggested_models=("deepseek-chat", "deepseek-reasoner"),
    ),
    AssistantPreset("groq", "Groq", "https://api.groq.com/openai/v1", key_url="https://console.groq.com/keys"),
    AssistantPreset("xai", "xAI (Grok)", "https://api.x.ai/v1", key_url="https://console.x.ai"),
    AssistantPreset(
        "gemini",
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        key_url="https://aistudio.google.com/apikey",
        suggested_models=("gemini-2.5-pro", "gemini-2.5-flash"),
    ),
    AssistantPreset(
        "mistral",
        "Mistral",
        "https://api.mistral.ai/v1",
        key_url="https://console.mistral.ai/api-keys",
        suggested_models=("mistral-large-latest",),
    ),
    AssistantPreset(
        "together", "Together AI", "https://api.together.xyz/v1", key_url="https://api.together.ai/settings/api-keys"
    ),
    AssistantPreset(
        "fireworks",
        "Fireworks",
        "https://api.fireworks.ai/inference/v1",
        key_url="https://fireworks.ai/account/api-keys",
    ),
    AssistantPreset("cerebras", "Cerebras", "https://api.cerebras.ai/v1", key_url="https://cloud.cerebras.ai"),
    AssistantPreset(
        "qwen",
        "Qwen (DashScope)",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        key_url="https://dashscope.console.aliyun.com/apiKey",
        suggested_models=("qwen-plus", "qwen-max"),
    ),
    AssistantPreset("minimax", "MiniMax", "https://api.minimax.io/v1", key_url="https://www.minimax.io"),
    AssistantPreset("ollama", "Ollama", "http://localhost:11434/v1", key_required=False, editable_url=True),
    AssistantPreset("lmstudio", "LM Studio", "http://localhost:1234/v1", key_required=False, editable_url=True),
    AssistantPreset("custom", "Custom (OpenAI-compatible)", "", key_required=False, editable_url=True),
)
PRESETS_BY_ID: dict[str, AssistantPreset] = {preset.id: preset for preset in PRESETS}


def get_preset(preset_id: str) -> AssistantPreset:
    preset = PRESETS_BY_ID.get(preset_id)
    if preset is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"unknown assistant preset {preset_id!r}")
    return preset


def resolve_base_url(preset: AssistantPreset | None, base_url: str | None) -> str:
    """The URL a provider uses: fixed by its preset, or given when the preset allows it."""

    given = (base_url or "").strip().rstrip("/")
    if preset is not None and preset.id == CLAUDE_CODE_PRESET:
        if given and given != CLAUDE_CODE_BASE_URL:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Claude Code runs as the plane host's own CLI and takes no base URL",
            )
        return CLAUDE_CODE_BASE_URL
    if preset is not None and not preset.editable_url:
        if given and given != preset.base_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{preset.label} has a fixed base URL; choose the custom preset for another endpoint",
            )
        return preset.base_url
    url = given or (preset.base_url if preset is not None else "")
    if not url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="base_url is required for this provider")
    return url


class ModelDiscoveryError(RuntimeError):
    """The provider's model list could not be read; the message never carries a key."""


async def discover_models(
    base_url: str, api_key: str | None, *, transport: httpx.AsyncBaseTransport | None = None
) -> list[str]:
    """Return the model ids one OpenAI-compatible endpoint lists, sorted and validated."""

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=DISCOVERY_TIMEOUT_S, transport=transport) as client:
            response = await client.get(base_url.rstrip("/") + "/models", headers=headers)
    except httpx.HTTPError as exc:
        raise ModelDiscoveryError(f"the provider is unreachable ({type(exc).__name__})") from None
    if response.status_code in {401, 403}:
        raise ModelDiscoveryError("the provider rejected the API key")
    if response.status_code != 200:
        raise ModelDiscoveryError(f"the provider answered HTTP {response.status_code} for its model list")
    try:
        payload = response.json()
    except ValueError:
        raise ModelDiscoveryError("the provider returned an unreadable model list") from None
    # OpenAI shape is {"data": [{"id": ...}]}; some providers return the bare list.
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ModelDiscoveryError("the provider returned an unreadable model list")
    found: dict[str, None] = {}
    for item in items:
        model_id = item.get("id") if isinstance(item, dict) else item
        if not isinstance(model_id, str):
            continue
        model_id = model_id.removeprefix("models/")  # Gemini prefixes its ids
        if re.fullmatch(ASSISTANT_MODEL_PATTERN, model_id):
            found[model_id] = None
    return sorted(found)[:MAX_DISCOVERED_MODELS]


async def _claude_code_model_list(executable: str | None) -> tuple[list[str], str, str | None]:
    """(models, source, detail) from the plane host's Claude Code CLI, never over HTTP.

    Falls back to the preset's suggestions when the CLI's list cannot be read.
    """

    # Imported here: assistant_claude_code imports ModelDiscoveryError from this module.
    from eyvan.server.api.assistant_claude_code import claude_code_models, find_claude_executable

    try:
        models = await claude_code_models(executable or find_claude_executable())
    except ModelDiscoveryError as exc:
        return list(PRESETS_BY_ID[CLAUDE_CODE_PRESET].suggested_models), "suggested", str(exc)
    return models[:MAX_DISCOVERED_MODELS], "live", None


def _unique_name(base: str, taken: set[str]) -> str:
    base = base[:60]
    if base.lower() not in taken:
        return base
    for index in range(2, 100):
        candidate = f"{base} {index}"
        if candidate.lower() not in taken:
            return candidate
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="choose a name for this provider")


async def provider_creation_fields(
    service: ControlPlaneService,
    principal: Principal,
    *,
    preset_id: str | None,
    name: str | None,
    base_url: str | None,
    models: tp.Sequence[str] | None,
    api_key: str | None,
    transport: httpx.AsyncBaseTransport | None,
    claude_code_executable: str | None = None,
) -> dict[str, tp.Any]:
    """Fill a new provider's name, URL and models from its preset where the admin left them out.

    A ``claude-code`` provider takes no key; without chosen models it offers the
    plane host's live Claude Code model list (``claude_code_executable``, else the
    discovered ``claude``), or the preset's suggestions when that cannot be read.
    """

    preset = get_preset(preset_id) if preset_id else None
    if preset is None and not (name and base_url and models):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="choose a preset, or give name, base_url and models"
        )
    url = resolve_base_url(preset, base_url)
    if preset is not None and preset.key_required and not (api_key or "").strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{preset.label} needs an API key")
    chosen = [model.strip() for model in models or () if model.strip()]
    if preset is not None and preset.id == CLAUDE_CODE_PRESET:
        if (api_key or "").strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Claude Code uses the plane host's own sign-in and takes no API key",
            )
        if not chosen:
            chosen = (await _claude_code_model_list(claude_code_executable))[0][:MAX_PROVIDER_MODELS]
    if not chosen and preset is not None:
        chosen = list(preset.suggested_models)
        if not chosen:
            try:
                chosen = (await discover_models(url, api_key, transport=transport))[:MAX_PROVIDER_MODELS]
            except ModelDiscoveryError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"choose the models to offer; {exc}"
                ) from None
    if not chosen:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="choose at least one model")
    if not name:
        taken = {item.name.lower() for item in await service.list_assistant_providers(principal)}
        label = preset.label if preset is not None and preset.id != "custom" else (urlsplit(url).hostname or "custom")
        name = _unique_name(label, taken)
    return {"name": name, "base_url": url, "models": chosen, "preset": preset.id if preset is not None else None}


# ----------------------------------------------------------- OpenRouter PKCE


@dataclass(slots=True)
class PendingSignIn:
    verifier: str = field(repr=False)
    principal: Principal
    models: tuple[str, ...]
    default_model: str | None
    name: str | None
    deadline: float


class OpenRouterSignIns:
    """Single-use, expiring PKCE sign-ins held in this process only."""

    def __init__(self, *, clock: tp.Callable[[], float] = time.monotonic, ttl_s: float = SIGN_IN_TTL_S) -> None:
        self._clock = clock
        self._ttl_s = ttl_s
        self._pending: dict[str, PendingSignIn] = {}

    def __repr__(self) -> str:
        return f"OpenRouterSignIns(pending={len(self._pending)})"

    def _prune(self) -> None:
        now = self._clock()
        for state in [state for state, item in self._pending.items() if item.deadline <= now]:
            del self._pending[state]

    def begin(
        self, principal: Principal, *, models: tuple[str, ...], default_model: str | None, name: str | None
    ) -> tuple[str, str]:
        """Return (state, S256 code challenge) for one new sign-in."""

        self._prune()
        if len(self._pending) >= MAX_PENDING_SIGN_INS:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="too many sign-ins in progress")
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
        self._pending[state] = PendingSignIn(
            verifier=verifier,
            principal=principal,
            models=models,
            default_model=default_model,
            name=name,
            deadline=self._clock() + self._ttl_s,
        )
        return state, challenge

    def take(self, state: str) -> PendingSignIn | None:
        """Consume one sign-in; ``None`` when unknown, already used or expired."""

        self._prune()
        return self._pending.pop(state, None)


class OpenRouterExchangeError(RuntimeError):
    """The code could not be exchanged; the message carries no code, verifier or key."""


async def exchange_openrouter_code(
    code: str, verifier: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    """Exchange an authorization code for the user's OpenRouter API key."""

    try:
        async with httpx.AsyncClient(timeout=DISCOVERY_TIMEOUT_S, transport=transport) as client:
            response = await client.post(
                OPENROUTER_KEY_EXCHANGE_URL,
                json={"code": code, "code_verifier": verifier, "code_challenge_method": "S256"},
            )
    except httpx.HTTPError as exc:
        raise OpenRouterExchangeError(f"OpenRouter is unreachable ({type(exc).__name__})") from None
    if response.status_code != 200:
        raise OpenRouterExchangeError(f"OpenRouter answered HTTP {response.status_code}")
    try:
        key = response.json().get("key")
    except (ValueError, AttributeError):
        key = None
    if not isinstance(key, str) or not key.strip():
        raise OpenRouterExchangeError("OpenRouter returned no key")
    return key.strip()


# ---------------------------------------------------------------- routes


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiscoverBody(_Body):
    preset: str | None = Field(default=None, max_length=64)
    base_url: str | None = Field(default=None, max_length=512)
    #: Write-only; used for this one request and never stored or returned.
    api_key: str | None = Field(default=None, max_length=4096)
    #: Use this provider's stored URL and key (so editing needs no re-typed key).
    provider_id: str | None = Field(default=None, max_length=32)


class SignInBody(_Body):
    models: list[str] | None = Field(default=None, max_length=MAX_PROVIDER_MODELS)
    default_model: str | None = Field(default=None, max_length=128)
    name: str | None = Field(default=None, min_length=1, max_length=64)


def _settings_redirect(query: str) -> RedirectResponse:
    return RedirectResponse(f"{SETTINGS_REDIRECT}?{query}", status_code=status.HTTP_303_SEE_OTHER)


def register_assistant_preset_routes(app: FastAPI, *, settings: EyvanSettings, service: ControlPlaneService) -> None:
    """Register preset listing, model discovery, OpenRouter sign-in and Claude Code host routes."""

    sign_ins = OpenRouterSignIns()
    app.state.openrouter_sign_ins = sign_ins

    def require_admin(principal: Principal) -> None:
        if not service.can_manage_assistant_providers(principal):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="only fleet admins configure assistant providers"
            )

    def transport(request: Request) -> httpx.AsyncBaseTransport | None:
        # A test/embedding seam, not configuration: a fake provider transport.
        return getattr(request.app.state, "assistant_transport", None)

    def claude_code_executable(request: Request) -> str | None:
        # A test/embedding seam, not configuration: the `claude` executable to use.
        override = getattr(request.app.state, "claude_code_executable", None)
        return str(override) if override else None

    # Imported here: assistant_claude_code imports ModelDiscoveryError from this module.
    from eyvan.server.api.assistant_claude_code import register_claude_code_routes

    register_claude_code_routes(app, service=service)

    @app.get("/api/v1/assistant/presets", tags=["assistant"])
    async def list_assistant_presets(principal: _Principal) -> dict[str, tp.Any]:
        """Known providers an admin can add by pasting a key (or signing in)."""

        del principal
        return {"presets": [preset.document() for preset in PRESETS]}

    @app.post("/api/v1/assistant/models/discover", tags=["assistant"])
    async def discover_assistant_models(
        body: DiscoverBody, request: Request, principal: _Principal
    ) -> dict[str, tp.Any]:
        """Ask the provider which models it serves; fall back to the preset's suggestions."""

        require_admin(principal)
        if body.provider_id is not None:
            record, stored_key = await service.assistant_provider_key(principal, body.provider_id)
            # A stored key goes only to its own stored URL, never to a caller-chosen one.
            url, key = record.base_url, body.api_key or stored_key
            preset = PRESETS_BY_ID.get(record.preset or "")
        else:
            preset = get_preset(body.preset) if body.preset else None
            url, key = resolve_base_url(preset, body.base_url), body.api_key
        if (preset is not None and preset.id == CLAUDE_CODE_PRESET) or url == CLAUDE_CODE_BASE_URL:
            # The plane host's own CLI lists its models; nothing goes over HTTP and no key is used.
            models, source, detail = await _claude_code_model_list(claude_code_executable(request))
            return {"models": models, "source": source, "detail": detail}
        suggestions = list(preset.suggested_models) if preset is not None else []
        try:
            models = await discover_models(url, key, transport=transport(request))
        except ModelDiscoveryError as exc:
            return {"models": suggestions, "source": "suggested", "detail": str(exc)}
        return {"models": models, "source": "live", "detail": None if models else "the provider listed no models"}

    @app.post("/api/v1/assistant/oauth/openrouter/start", tags=["assistant"])
    async def start_openrouter_sign_in(body: SignInBody, principal: _Principal) -> dict[str, tp.Any]:
        """Begin OpenRouter OAuth PKCE; the browser goes to ``authorize_url``."""

        require_admin(principal)
        state, challenge = sign_ins.begin(
            principal,
            models=tuple(body.models or ()),
            default_model=body.default_model,
            name=body.name,
        )
        callback = f"{settings.public_url.rstrip('/')}/api/v1/assistant/oauth/openrouter/callback/{state}"
        query = urlencode({"callback_url": callback, "code_challenge": challenge, "code_challenge_method": "S256"})
        return {
            "authorize_url": f"{OPENROUTER_AUTHORIZE_URL}?{query}",
            "expires_at": (utcnow() + timedelta(seconds=SIGN_IN_TTL_S)).isoformat(),
        }

    @app.get("/api/v1/assistant/oauth/openrouter/callback/{state}", include_in_schema=False)
    async def finish_openrouter_sign_in(
        state: str, request: Request, code: str | None = None, error: str | None = None
    ) -> RedirectResponse:
        """OpenRouter's redirect target; authorized only by the single-use state it carries."""

        pending = sign_ins.take(state)
        if pending is None:
            return _settings_redirect("assistant_error=expired")
        if error or not code:
            return _settings_redirect("assistant_error=denied")
        try:
            key = await exchange_openrouter_code(code, pending.verifier, transport=transport(request))
        except OpenRouterExchangeError as exc:
            logger.warning("OpenRouter sign-in failed: %s", exc)
            return _settings_redirect("assistant_error=exchange_failed")
        try:
            fields = await provider_creation_fields(
                service,
                pending.principal,
                preset_id="openrouter",
                name=pending.name,
                base_url=None,
                models=pending.models,
                api_key=key,
                transport=transport(request),
            )
            await service.create_assistant_provider(
                pending.principal, api_key=key, default_model=pending.default_model, **fields
            )
        except (HTTPException, EyvanServiceError) as exc:
            logger.warning("OpenRouter sign-in could not save the provider: %s", type(exc).__name__)
            return _settings_redirect("assistant_error=save_failed")
        return _settings_redirect("assistant=openrouter-connected")


__all__ = (
    "CLAUDE_CODE_PRESET",
    "PRESETS",
    "PRESETS_BY_ID",
    "AssistantPreset",
    "ModelDiscoveryError",
    "OpenRouterSignIns",
    "discover_models",
    "exchange_openrouter_code",
    "provider_creation_fields",
    "register_assistant_preset_routes",
    "resolve_base_url",
)
