# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assistant LLM providers that fleet admins configure from the dashboard.

Fleet admins (``MANAGE_FLEET``) add, change and remove providers; every
authenticated user may list the enabled ones (names and models only) and pick
one for a chat turn. The API key is write-only: it is sealed before it is
stored, never returned, never logged and never put in an audit event, which
records only whether a key is set.
"""

from __future__ import annotations

import secrets
import typing as tp

from eyvan.identity.assistant_keys import AssistantKeyError, AssistantKeySealer, validate_api_key
from eyvan.records import AssistantProviderRecord
from eyvan.server.service._control_plane.shared import (
    AuditEvent,
    AuthorizationError,
    ConflictError,
    InputError,
    NotFoundError,
    Permission,
    Principal,
    RepositorySession,
    UnavailableError,
    _new_id,
    utcnow,
)

#: Upper bound on configured providers, so the picker and the chat resolver stay small.
MAX_ASSISTANT_PROVIDERS = 32

_UNSET: tp.Any = object()


def _read_view(repository: tp.Any) -> tp.AsyncContextManager[tp.Any]:
    """A read that never queues behind a commit: the committed snapshot where the store offers one."""

    snapshot = getattr(repository, "snapshot", None)
    return tp.cast(tp.AsyncContextManager[tp.Any], snapshot() if snapshot is not None else repository.transaction())


def _event(actor: str, provider: AssistantProviderRecord, action: str) -> AuditEvent:
    return AuditEvent(
        id=_new_id("evt"),
        actor=actor,
        entity_type="assistant_provider",
        entity_id=provider.id,
        action=action,
        payload={
            "name": provider.name,
            "base_url": provider.base_url,
            "models": list(provider.models),
            "default_model": provider.default_model,
            "enabled": provider.enabled,
            # Whether one is set -- never the key, never its ciphertext.
            "api_key_set": provider.api_key_ciphertext is not None,
        },
    )


def _clean_models(models: tp.Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name.strip() for name in models if name.strip()))


class AssistantProvidersMixin:
    """Transactional assistant-provider operations."""

    assistant_sealer: AssistantKeySealer | None = None

    def _require_provider_admin(self, principal: Principal) -> None:
        if not self.authorizer.allows(principal, Permission.MANAGE_FLEET):  # type: ignore[attr-defined]
            raise AuthorizationError(f"{principal.subject!r} may not configure assistant providers")

    def can_manage_assistant_providers(self, principal: Principal) -> bool:
        return bool(self.authorizer.allows(principal, Permission.MANAGE_FLEET))  # type: ignore[attr-defined]

    def _sealer(self) -> AssistantKeySealer:
        if self.assistant_sealer is None:
            raise UnavailableError("this plane has no server secret to protect assistant provider keys")
        return self.assistant_sealer

    @staticmethod
    async def _refuse_duplicate_name(tx: RepositorySession, name: str, *, keep: str | None = None) -> None:
        for existing in await tx.list_assistant_providers():
            if existing.id != keep and existing.name.lower() == name.lower():
                raise ConflictError(f"an assistant provider named {name!r} already exists")

    async def list_assistant_providers(self, principal: Principal) -> list[AssistantProviderRecord]:
        """Every provider for a fleet admin; only enabled ones for everyone else."""

        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            providers = await tx.list_assistant_providers()
        if self.can_manage_assistant_providers(principal):
            return providers
        return [provider for provider in providers if provider.enabled]

    async def create_assistant_provider(
        self,
        principal: Principal,
        *,
        name: str,
        base_url: str,
        models: tp.Sequence[str],
        api_key: str | None = None,
        default_model: str | None = None,
        max_steps: int = 6,
        enabled: bool = True,
        preset: str | None = None,
    ) -> AssistantProviderRecord:
        """Add one provider (fleet admins). ``api_key`` is sealed before anything is stored."""

        self._require_provider_admin(principal)
        provider_id = f"aprv_{secrets.token_hex(8)}"
        cleaned = _clean_models(models)
        try:
            ciphertext = None if not api_key else self._sealer().seal(provider_id, validate_api_key(api_key))
            provider = AssistantProviderRecord(
                id=provider_id,
                name=name.strip(),
                preset=preset,
                base_url=base_url.strip().rstrip("/"),
                models=cleaned,
                default_model=(default_model or "").strip() or (cleaned[0] if cleaned else ""),
                max_steps=max_steps,
                enabled=enabled,
                api_key_ciphertext=ciphertext,
                created_by=principal.subject,
            )
        except ValueError as exc:
            raise InputError(str(exc)) from None
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            if len(await tx.list_assistant_providers()) >= MAX_ASSISTANT_PROVIDERS:
                raise ConflictError(f"at most {MAX_ASSISTANT_PROVIDERS} assistant providers may be configured")
            await self._refuse_duplicate_name(tx, provider.name)
            await tx.put_assistant_provider(provider)
            await tx.append_event(_event(principal.subject, provider, "assistant_provider.created"))
        return provider

    async def update_assistant_provider(
        self,
        principal: Principal,
        provider_id: str,
        *,
        name: str | None = None,
        base_url: str | None = None,
        models: tp.Sequence[str] | None = None,
        default_model: str | None = None,
        max_steps: int | None = None,
        enabled: bool | None = None,
        api_key: str | None = None,
        clear_api_key: bool = False,
    ) -> AssistantProviderRecord:
        """Change some fields of one provider (fleet admins); an omitted key keeps the stored one."""

        self._require_provider_admin(principal)
        if api_key and clear_api_key:
            raise InputError("send either a new api_key or clear_api_key, not both")
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            current = await tx.get_assistant_provider(provider_id)
            if current is None:
                raise NotFoundError(f"assistant provider {provider_id!r} not found")
            new_models = _clean_models(models) if models is not None else current.models
            new_default = (default_model or "").strip() or current.default_model
            if new_default not in new_models and new_models:
                # A model list that dropped the old default falls back to its first entry
                # unless the caller named a default explicitly.
                new_default = new_models[0] if not default_model else new_default
            update: dict[str, tp.Any] = {
                "name": name.strip() if name is not None else current.name,
                "base_url": base_url.strip().rstrip("/") if base_url is not None else current.base_url,
                "models": new_models,
                "default_model": new_default,
                "max_steps": max_steps if max_steps is not None else current.max_steps,
                "enabled": enabled if enabled is not None else current.enabled,
                "updated_at": utcnow(),
            }
            try:
                if clear_api_key:
                    update["api_key_ciphertext"] = None
                elif api_key:
                    update["api_key_ciphertext"] = self._sealer().seal(provider_id, validate_api_key(api_key))
                updated = AssistantProviderRecord.model_validate({**current.model_dump(), **update})
            except ValueError as exc:
                raise InputError(str(exc)) from None
            await self._refuse_duplicate_name(tx, updated.name, keep=provider_id)
            await tx.put_assistant_provider(updated)
            await tx.append_event(_event(principal.subject, updated, "assistant_provider.updated"))
            return updated

    async def delete_assistant_provider(self, principal: Principal, provider_id: str) -> None:
        self._require_provider_admin(principal)
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            current = await tx.get_assistant_provider(provider_id)
            if current is None:
                raise NotFoundError(f"assistant provider {provider_id!r} not found")
            await tx.delete_assistant_provider(provider_id)
            await tx.append_event(_event(principal.subject, current, "assistant_provider.deleted"))

    async def assistant_provider_key(
        self, principal: Principal, provider_id: str
    ) -> tuple[AssistantProviderRecord, str | None]:
        """Return one provider and its plaintext key for a fleet admin's model discovery.

        The key never leaves the plane: callers use it only as the provider's
        ``Authorization`` header.
        """

        self._require_provider_admin(principal)
        async with _read_view(self.repository) as tx:  # type: ignore[attr-defined]
            current = await tx.get_assistant_provider(provider_id)
        if current is None:
            raise NotFoundError(f"assistant provider {provider_id!r} not found")
        if current.api_key_ciphertext is None:
            return current, None
        try:
            return current, self._sealer().open(current.id, current.api_key_ciphertext)
        except AssistantKeyError as exc:
            raise UnavailableError(str(exc)) from None

    async def resolve_assistant_provider(
        self, principal: Principal, *, provider_id: str | None, model: str | None
    ) -> tuple[AssistantProviderRecord, str, str | None]:
        """Return (provider, model, plaintext key) for one chat turn.

        No choice means the first enabled provider by name and its default
        model. A disabled or unknown provider, or a model it does not list, is
        refused rather than silently substituted.
        """

        providers = [item for item in await self.list_assistant_providers(principal) if item.enabled]
        if provider_id is None:
            if not providers:
                raise NotFoundError("no assistant provider is configured; a fleet admin can add one")
            chosen = providers[0]
        else:
            matching = [item for item in providers if item.id == provider_id]
            if not matching:
                raise NotFoundError(f"assistant provider {provider_id!r} is not available")
            chosen = matching[0]
        selected = model or chosen.default_model
        if selected not in chosen.models:
            raise InputError(f"model {selected!r} is not offered by assistant provider {chosen.name!r}")
        key = None
        if chosen.api_key_ciphertext is not None:
            try:
                key = self._sealer().open(chosen.id, chosen.api_key_ciphertext)
            except AssistantKeyError as exc:
                raise UnavailableError(str(exc)) from None
        return chosen, selected, key


__all__ = ("MAX_ASSISTANT_PROVIDERS", "AssistantProvidersMixin")
