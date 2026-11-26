# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The typing ratchet must not depend on editable-install error suppression."""

import tomllib
from pathlib import Path


def test_mypy_import_reporting_is_explicit_without_skipping_type_analysis():
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["tool"]["mypy"]
    assert config["follow_imports"] == "silent"
    assert config["files"] == [
        "eyvan/domain",
        "eyvan/config.py",
        "eyvan/records.py",
        "eyvan/server/api/dashboard_sessions.py",
        "eyvan/server/api/server_bind.py",
        "eyvan/server/api/server_startup.py",
        "eyvan/sdk/commands/team.py",
        "eyvan/sdk/commands/assistant.py",
        "eyvan/server/api/assistant.py",
        "eyvan/server/api/team_routes.py",
        "eyvan/server/api/bootstrap_admin.py",
        "eyvan/server/service/_control_plane/accounts.py",
        "eyvan/server/service/_control_plane/team_requests.py",
    ]
    assert not config.get("ignore_errors", False)
