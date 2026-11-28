# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small deterministic helpers shared by Eyvan CLI commands."""

from __future__ import annotations

import json
import shlex
import warnings
from pathlib import Path

import click
import yaml
from pydantic import ValidationError

from eyvan._internal.json_types import JsonValue
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.sdk.tables import describe


class JsonRenderer:
    """Stable JSON output used by machine-readable CLI paths."""

    def emit(self, value: JsonValue) -> None:
        """Write one deterministic JSON document to stdout."""

        click.echo(json.dumps(value, indent=2, sort_keys=True))


def emit_result(
    renderer: JsonRenderer,
    response: JsonValue,
    headline: str,
    *,
    as_json: bool,
    fields: tuple[str, ...] | None = None,
) -> None:
    """Report what a mutation did, or its record when asked for the record.

    Mutating commands answered with the stored document -- creating an
    organization printed thirty keys, and whether it worked was somewhere
    inside. The caller knows what it just did and the record does not, so the
    sentence comes from here and the identity comes from the response.
    """

    if as_json:
        renderer.emit(response)
        return
    for line in describe(headline, response, fields=fields):
        click.echo(line)


class FleetConfigLoader:
    """Load and validate public fleet YAML documents."""

    def load(self, path: Path) -> EyvanFleetConfig:
        """Return a validated config or a Click-facing actionable error.

        Any warning the schema raises is re-rendered rather than left to
        Python's default handler, which prints the line inside pydantic that
        happened to call ``warn`` -- a path under ``site-packages`` that says
        nothing about which file the reader should edit. The advice is worth
        keeping; the frame it arrived in is not.
        """

        try:
            with warnings.catch_warnings(record=True) as raised:
                warnings.simplefilter("always")
                config = EyvanFleetConfig.from_yaml(path)
        except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
            raise click.ClickException(f"invalid fleet configuration {path}: {exc}") from exc
        # One line per distinct message, not per pool. The schema warns as it
        # validates, so a twenty-pool fleet raised the same sentence twenty
        # times -- which buries the file name the line exists to supply.
        for message in dict.fromkeys(str(warning.message) for warning in raised):
            click.echo(f"{path}: {message}", err=True)
        return config


def parse_environment(values: tuple[str, ...], *, option: str = "--env") -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` options without accepting ambiguous input.

    ``option`` names the flag in the error, because more than one flag now has
    this shape and an error naming the wrong one sends a user to the wrong line
    of their command.
    """

    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or key != key.strip() or not item or item != item.strip():
            raise click.ClickException(f"{option} values must use KEY=VALUE with non-empty trimmed strings")
        result[key] = item
    return result


def shell_command(values: tuple[str, ...]) -> str:
    """Return one safely quoted remote shell command."""

    return shlex.join(values)
