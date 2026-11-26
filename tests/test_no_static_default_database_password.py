# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""No shipped surface may carry a static default database password.

The settings default, the Alembic configuration, the migration entry point, the
deployment manifests, and the docs all used to name
``postgresql+asyncpg://eyvan:eyvan@127.0.0.1:5432/eyvan``. A password that
ships in the tree is a password every installation shares, and "change the
default" is the step everyone skips. The loopback-development default is now
password-less (it relies on the host's trust/peer auth), and every real
deployment supplies ``EYVAN_DATABASE_URL`` or a secret reference.

Test fixtures may still spell ``eyvan:eyvan@some-host`` — those are inert
strings pointing at hosts that do not exist. This scan covers only surfaces a
user receives or runs.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Everything a user receives or runs. Tests are deliberately excluded: their
#: fixture URLs point at nonexistent hosts and are not configuration.
SHIPPED_PATHS = (
    "alembic.ini",
    "docker-compose.yml",
    "Dockerfile",
    "README.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "eyvan",
    "deploy",
    "docs",
    "examples",
    "oci",
    "scripts",
)

STATIC_DEFAULT = "eyvan:eyvan@"


def _shipped_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for relative in SHIPPED_PATHS:
        path = ROOT / relative
        candidates = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        # Bytecode caches carry whatever the source said at any past moment;
        # scanning them would flag history, not the tree.
        kept = (item for item in candidates if "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"})
        files.extend(kept)
    return files


def test_no_shipped_surface_carries_the_static_default_password() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in _shipped_files()
        if STATIC_DEFAULT in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert not offenders, (
        "static default database password found in shipped surfaces: "
        + ", ".join(offenders)
        + "; use the password-less loopback default or require an explicit EYVAN_DATABASE_URL"
    )


def test_the_loopback_default_is_password_less() -> None:
    from eyvan.config import EyvanSettings
    from eyvan.server.persistence import migrate

    for name, url in (
        # EyvanSettings is a slotted dataclass: the class attribute is a
        # member_descriptor, so the default is read from an instance.
        ("EyvanSettings.database_url", EyvanSettings().database_url),
        ("migrate.DEFAULT_DATABASE_URL", migrate.DEFAULT_DATABASE_URL),
    ):
        remainder = url.split("://", 1)[1]
        userinfo = remainder.split("@", 1)[0] if "@" in remainder else ""
        assert ":" not in userinfo, f"{name} embeds a password: {url!r}"
        assert url == "postgresql+asyncpg://eyvan@127.0.0.1:5432/eyvan", name


def test_the_alembic_default_matches_the_migration_default() -> None:
    from eyvan.server.persistence import migrate

    ini = (ROOT / "alembic.ini").read_text(encoding="utf-8")

    assert f"sqlalchemy.url = {migrate.DEFAULT_DATABASE_URL}" in ini
