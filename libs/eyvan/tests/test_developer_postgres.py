# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Developer full-suite setup fails closed and uses a disposable database."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_developer_database_is_separate_loopback_only_and_ephemeral():
    config = yaml.safe_load((ROOT / "docker-compose.test.yml").read_text())
    service = config["services"]["postgres-test"]
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yaml").read_text())
    assert service["image"] == ci["jobs"]["python"]["services"]["postgres"]["image"]
    assert service["ports"] == ["127.0.0.1:55432:5432"]
    assert service["tmpfs"] == ["/var/lib/postgresql/data"]
    assert "volumes" not in service
    assert service["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_TEST_PASSWORD:?")
    assert "POSTGRES_HOST_AUTH_METHOD" not in service["environment"]


def test_documented_full_suite_requires_postgres():
    text = (ROOT / "CONTRIBUTING.md").read_text()
    assert "EYVAN_REQUIRE_POSTGRES=1 uv run pytest tests" in text
    assert "docker compose -f docker-compose.test.yml up -d --wait" in text
    assert "Never use a production" in text
