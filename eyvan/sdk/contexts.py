# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent, non-secret Eyvan execution contexts."""

from __future__ import annotations

import json
import os
import re
import tempfile
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from eyvan._internal.json_types import JsonValue, is_json_value
from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.fleet_config import validate_fleet_name

CONTEXT_CONFIG_VERSION = 1
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_ORG_ID = "default"
DEFAULT_PROJECT_ID = "default"
DEFAULT_CONFIG_PATH = Path("~/.config/eyvan/config.json")


class ContextConfigError(ValueError):
    """Raised when the persistent context file cannot be read or validated."""


class EyvanContext(BaseModel):
    """One named set of non-secret Eyvan client defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: None for a plane saved before any fleet was chosen (``eyvan login`` on a
    #: plane with no fleet, or several): later commands still target the plane
    #: and infer or ask for the fleet, exactly as with no context at all.
    fleet: str | None
    org_id: str
    project_id: str
    base_url: str = DEFAULT_BASE_URL
    autoscale_cache_dir: str | None = None

    @field_validator("fleet")
    @classmethod
    def _valid_fleet(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_fleet_name(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("org_id", "project_id")
    @classmethod
    def _non_empty_scope(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("scope values must be non-empty strings")
        return normalized

    @field_validator("base_url")
    @classmethod
    def _valid_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, a query, or a fragment")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("base_url has an invalid port") from exc
        return normalized

    @field_validator("autoscale_cache_dir")
    @classmethod
    def _valid_autoscale_cache_dir(cls, value: str | None) -> str | None:
        """Normalize an optional user-owned autoscale cache directory."""

        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("autoscale_cache_dir must be a non-empty path when set")
        return str(Path(normalized).expanduser().resolve())


class EyvanContextConfig(BaseModel):
    """Versioned collection of named contexts with one active selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: tp.Literal[CONTEXT_CONFIG_VERSION] = CONTEXT_CONFIG_VERSION
    active_context: str | None = None
    contexts: dict[str, EyvanContext] = Field(default_factory=dict)

    @field_validator("contexts")
    @classmethod
    def _valid_context_names(cls, value: dict[str, EyvanContext]) -> dict[str, EyvanContext]:
        for name in value:
            try:
                validate_fleet_name(name)
            except ValueError as exc:
                raise ValueError(f"invalid context name {name!r}: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _active_context_exists(self) -> EyvanContextConfig:
        if self.active_context is not None and self.active_context not in self.contexts:
            raise ValueError(f"active context {self.active_context!r} is not present in contexts")
        return self

    @property
    def active(self) -> EyvanContext | None:
        """Return the active context, if one has been selected."""

        if self.active_context is None:
            return None
        return self.contexts[self.active_context]


def plane_context_name(base_url: str) -> str:
    """Return a stable context name for a plane saved without a fleet: ``plane-<host>-<port>``."""

    split = urlsplit(base_url)
    raw = f"plane-{split.hostname or 'unknown'}-{split.port or ''}"
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:63].rstrip("-")
    return validate_fleet_name(slug)


def context_config_path(environ: tp.Mapping[str, str] | None = None) -> Path:
    """Resolve the context file path from ``EYVAN_CONFIG_FILE`` or the user default."""

    environment = os.environ if environ is None else environ
    configured = environment.get("EYVAN_CONFIG_FILE")
    return Path(configured or DEFAULT_CONFIG_PATH).expanduser()


def _decode_config(path: Path, content: str) -> JsonValue:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(content)
    else:
        payload = json.loads(content)
    if not is_json_value(payload):
        raise ContextConfigError("the document contains a non-JSON value")
    return payload


def _encoded_config(config: EyvanContextConfig, path: Path) -> str:
    payload = config.model_dump(mode="json")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_dump(payload, sort_keys=True)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class FileContextStore:
    """Owner-only, atomic persistence for non-secret Eyvan contexts."""

    path: Path | None = None

    @property
    def resolved_path(self) -> Path:
        """Return the expanded path used by this store."""

        return (self.path or context_config_path()).expanduser()

    def load(self) -> EyvanContextConfig:
        """Load and validate the context document, or return an empty config."""

        target = self.resolved_path
        try:
            if not target.exists():
                return EyvanContextConfig()
            payload = _decode_config(target, target.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ContextConfigError("the document root must be an object")
            return EyvanContextConfig.model_validate(payload)
        except ContextConfigError:
            raise
        except (OSError, json.JSONDecodeError, yaml.YAMLError, ValidationError) as exc:
            raise ContextConfigError(f"invalid Eyvan context configuration {target}: {exc}") from exc

    def save(self, config: EyvanContextConfig) -> Path:
        """Atomically write ``config`` with owner-only permissions."""

        target = self.resolved_path
        create_private_directory(target.parent)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                handle.write(_encoded_config(config, target))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o600)
            return target
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def activate(self, context: EyvanContext, *, name: str | None = None) -> EyvanContextConfig:
        """Create or replace a named context and make it active."""

        chosen = name or context.fleet
        if chosen is None:
            raise ValueError("a context without a fleet needs an explicit name")
        context_name = validate_fleet_name(chosen)
        current = self.load()
        contexts = dict(current.contexts)
        contexts[context_name] = context
        updated = EyvanContextConfig(active_context=context_name, contexts=contexts)
        self.save(updated)
        return updated

    def active(self) -> EyvanContext | None:
        """Return the active context without applying environment overrides."""

        return self.load().active


def load_context_config(path: Path | None = None) -> EyvanContextConfig:
    """Compatibility wrapper for :meth:`FileContextStore.load`."""

    return FileContextStore(path).load()


def save_context_config(config: EyvanContextConfig, path: Path | None = None) -> Path:
    """Compatibility wrapper for :meth:`FileContextStore.save`."""

    return FileContextStore(path).save(config)


def set_active_context(
    context: EyvanContext,
    *,
    name: str | None = None,
    path: Path | None = None,
) -> EyvanContextConfig:
    """Compatibility wrapper for :meth:`FileContextStore.activate`."""

    return FileContextStore(path).activate(context, name=name)


def get_active_context(path: Path | None = None) -> EyvanContext | None:
    """Compatibility wrapper for :meth:`FileContextStore.active`."""

    return FileContextStore(path).active()
