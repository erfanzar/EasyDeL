# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Platform-owned Eyvan routing configuration.

This file intentionally owns infrastructure routing such as the default cell
agent. User fleet and autoscale YAML files do not need to know internal service
URLs.
"""

from __future__ import annotations

import os
import typing as tp
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from eyvan._internal.json_types import JsonValue, is_json_object, is_json_value

if tp.TYPE_CHECKING:
    from eyvan.config import EyvanSettings

DEFAULT_PLATFORM_CONFIG_PATH = Path("/etc/eyvan/platform.yaml")


@dataclass(frozen=True, slots=True)
class EyvanPlatformConfig:
    """Resolved platform routing defaults available to the control plane."""

    default_agent_endpoint: str | None = None
    #: ``team: true`` turns on team accounts for every server reading this file.
    #: The only other switch is ``eyvan-server --team``; there is no environment variable.
    team: bool = False

    @classmethod
    def from_document(cls, value: JsonValue) -> EyvanPlatformConfig:
        """Parse one strict platform YAML mapping."""

        if not is_json_object(value):
            raise ValueError("Eyvan platform YAML must be a mapping")
        allowed = {"version", "agents", "team"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown Eyvan platform field {unknown[0]!r}")
        version = value.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError("Eyvan platform version must be 1")
        team = value.get("team", False)
        # A real YAML boolean only: "yes" in quotes or 1 is a typo, and guessing
        # which way a typo was meant would decide who can sign in to the plane.
        if not isinstance(team, bool):
            raise ValueError("Eyvan platform team must be true or false")
        agents = value.get("agents", {})
        if not isinstance(agents, dict):
            raise ValueError("Eyvan platform agents must be a mapping")
        unknown_agents = sorted(set(agents) - {"default"})
        if unknown_agents:
            raise ValueError(f"unknown Eyvan platform agent {unknown_agents[0]!r}")
        default = agents.get("default")
        if default is None:
            return cls(team=team)
        if not isinstance(default, dict) or set(default) != {"endpoint"}:
            raise ValueError("Eyvan platform default agent must contain only endpoint")
        endpoint = default.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip() or endpoint != endpoint.strip():
            raise ValueError("Eyvan platform default agent endpoint must be a non-empty trimmed string")
        return cls(default_agent_endpoint=endpoint, team=team)


def load_platform_config(path: Path | str = DEFAULT_PLATFORM_CONFIG_PATH) -> EyvanPlatformConfig:
    """Load a platform YAML file, treating an absent optional file as no default."""

    target = Path(path).expanduser()
    if not target.exists():
        return EyvanPlatformConfig()
    try:
        value = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid Eyvan platform configuration {target}: {exc}") from exc
    try:
        if not is_json_value(value):
            raise ValueError("Eyvan platform YAML must contain only JSON-compatible values")
        return EyvanPlatformConfig.from_document(value)
    except ValueError as exc:
        raise ValueError(f"invalid Eyvan platform configuration {target}: {exc}") from exc


def resolve_team_mode(settings: EyvanSettings, *, flag: bool, platform: EyvanPlatformConfig) -> EyvanSettings:
    """Return ``settings`` with team mode on when the flag, the platform file, or the settings ask for it.

    The single resolver every surface uses (the server, ``bootstrap-admin`` and
    ``eyvan doctor``), so the three cannot disagree about whether a plane runs
    team accounts. Team mode is never read from the environment.
    """

    return replace(settings, team_mode=settings.team_mode or flag or platform.team)


#: The environment switch team mode used until 0.0.3. It is no longer a way to
#: turn anything on; it is read only so a plane still configured with it refuses
#: to start instead of silently coming back without team accounts (personal
#: tokens rejected, and caps and approval holds off for every principal that can
#: still authenticate).
REMOVED_TEAM_MODE_VARIABLE = "EYVAN_TEAM_MODE"

REMOVED_TEAM_MODE_MESSAGE = (
    f"{REMOVED_TEAM_MODE_VARIABLE} is set, but it no longer turns on team mode. Remove it and use "
    "`eyvan-server --team` or `team: true` in the platform file (/etc/eyvan/platform.yaml, or the path in "
    "EYVAN_PLATFORM_CONFIG_PATH). Starting anyway would bring a team plane back without team accounts."
)


def refuse_removed_team_variable(environ: tp.Mapping[str, str] | None = None) -> None:
    """Raise ``ValueError`` when the removed team-mode environment variable is present, whatever its value."""

    environment = os.environ if environ is None else environ
    if REMOVED_TEAM_MODE_VARIABLE in environment:
        raise ValueError(REMOVED_TEAM_MODE_MESSAGE)


__all__ = (
    "DEFAULT_PLATFORM_CONFIG_PATH",
    "REMOVED_TEAM_MODE_MESSAGE",
    "REMOVED_TEAM_MODE_VARIABLE",
    "EyvanPlatformConfig",
    "load_platform_config",
    "refuse_removed_team_variable",
    "resolve_team_mode",
)
