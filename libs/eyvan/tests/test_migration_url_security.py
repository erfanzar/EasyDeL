# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Migration URLs round-trip exactly and never put credentials in diagnostics."""

from __future__ import annotations

import io
import socket
import typing as tp
from pathlib import Path

import pytest
from alembic import command, context
from alembic.config import Config
from sqlalchemy import make_url
from sqlalchemy.dialects.postgresql.asyncpg import dialect

from eyvan.server.persistence import migrate

ROOT = Path(__file__).resolve().parents[1]
ENCODED_URL = "postgresql+asyncpg://FAKE_USER:proof%40secret%25%2F@db.invalid:5433/eyvan?application_name=proof%25"


@pytest.fixture(autouse=True)
def _no_engines_or_network(monkeypatch: pytest.MonkeyPatch) -> tp.Iterator[None]:
    """Every real engine or socket connection is forbidden, even on failing tests."""

    import sqlalchemy
    import sqlalchemy.ext.asyncio

    monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)
    attempts: list[str] = []

    def forbidden(*_args: tp.Any, **_kwargs: tp.Any) -> tp.NoReturn:
        attempts.append("engine or network connection")
        pytest.fail("migration URL regression attempted real engine/network I/O")

    monkeypatch.setattr(migrate, "create_async_engine", forbidden)
    monkeypatch.setattr(sqlalchemy, "create_engine", forbidden)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "create_async_engine", forbidden)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "async_engine_from_config", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    yield
    assert attempts == [], "no engine/network attempts are permitted"


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_percent_encoded_database_url_round_trips_config(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    assert make_url(ENCODED_URL).password == "proof@secret%/"
    if source == "environment":
        monkeypatch.setenv("EYVAN_DATABASE_URL", ENCODED_URL)
    config = migrate.build_alembic_config(ENCODED_URL if source == "explicit" else None)

    assert config.get_main_option("sqlalchemy.url") == ENCODED_URL
    assert config.get_section(config.config_ini_section)["sqlalchemy.url"] == ENCODED_URL


@pytest.mark.parametrize("source", ["source-environment", "source-ini", "packaged-environment", "packaged-explicit"])
def test_offline_env_preserves_percent_encoded_url_without_an_engine(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    output = io.StringIO()
    if source.startswith("source"):
        config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
        if source == "source-ini":
            config.set_main_option("sqlalchemy.url", ENCODED_URL.replace("%", "%%"))
        else:
            monkeypatch.setenv("EYVAN_DATABASE_URL", ENCODED_URL)
    else:
        if source == "packaged-environment":
            monkeypatch.setenv("EYVAN_DATABASE_URL", ENCODED_URL)
            config = migrate.build_alembic_config()
        else:
            config = migrate.build_alembic_config(ENCODED_URL)
            monkeypatch.setenv("EYVAN_DATABASE_URL", "postgresql+asyncpg://ambient.invalid/wrong")
        config.output_buffer = output
    urls: list[str] = []
    original_configure = context.configure

    def capture_url(**kwargs: tp.Any) -> tp.Any:
        urls.append(kwargs["url"])
        return original_configure(**kwargs)

    monkeypatch.setattr(context, "configure", capture_url)
    command.upgrade(config, "head", sql=True)

    assert urls == [ENCODED_URL]
    assert "CREATE TABLE workloads (" in output.getvalue()
    assert "proof" not in output.getvalue()


@pytest.mark.parametrize("arguments", [["--check"], []], ids=["check", "upgrade"])
def test_percent_url_reaches_cli_inspection_and_upgrade_unchanged(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    inspected: list[str] = []
    upgraded: list[tuple[str, str]] = []

    def inspect(config: Config) -> migrate.MigrationDatabaseState:
        inspected.append(config.get_main_option("sqlalchemy.url"))
        return migrate.MigrationDatabaseState((), (), ("head",), migrate.MigrationClassification.FRESH)

    def upgrade(config: Config, revision: str) -> None:
        upgraded.append((config.get_main_option("sqlalchemy.url"), revision))

    monkeypatch.setattr(migrate, "inspect_database_state", inspect)
    monkeypatch.setattr(migrate.command, "upgrade", upgrade)

    assert migrate.main([*arguments, "--database-url", ENCODED_URL]) == 0
    assert inspected == [ENCODED_URL]
    assert upgraded == ([] if arguments else [(ENCODED_URL, "head")])


def test_percent_url_reaches_inspection_engine_boundary_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []
    disposed: list[bool] = []

    class RefusedConnection:
        async def __aenter__(self) -> tp.NoReturn:
            raise OSError("synthetic credential-free refusal")

        async def __aexit__(self, *_args: tp.Any) -> None:
            pass

    class FakeEngine:
        def connect(self) -> RefusedConnection:
            return RefusedConnection()

        async def dispose(self) -> None:
            disposed.append(True)

    def fake_engine(url: str, **_kwargs: tp.Any) -> FakeEngine:
        urls.append(url)
        return FakeEngine()

    monkeypatch.setattr(migrate, "create_async_engine", fake_engine)
    config = migrate.build_alembic_config(ENCODED_URL)
    with pytest.raises(OSError, match="synthetic credential-free refusal"):
        migrate.inspect_database_state(config)
    assert urls == [ENCODED_URL]
    assert disposed == [True]


def test_source_online_env_delivers_exact_percent_url_to_engine_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlalchemy.ext.asyncio

    class EngineBoundaryReached(Exception):
        pass

    urls: list[str] = []

    def fake_engine(options: dict[str, tp.Any], **_kwargs: tp.Any) -> tp.NoReturn:
        urls.append(options["sqlalchemy.url"])
        raise EngineBoundaryReached

    monkeypatch.setattr(sqlalchemy.ext.asyncio, "async_engine_from_config", fake_engine)
    monkeypatch.setenv("EYVAN_DATABASE_URL", ENCODED_URL)
    with pytest.raises(EngineBoundaryReached):
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    assert urls == [ENCODED_URL]


@pytest.mark.parametrize(
    ("url", "target"),
    [
        (
            "postgresql+asyncpg://db.invalid:5433/eyvan?password=FAKE_QUERY_SECRET&sslkey=FAKE_KEY_PATH",
            "postgresql+asyncpg://db.invalid:5433/eyvan",
        ),
        (
            "postgresql+asyncpg://FAKE_USER:FAKE_AUTH_SECRET@db.invalid:5433/eyvan"
            "?password=FAKE_QUERY_SECRET&application_name=FAKE_QUERY_METADATA",
            "postgresql+asyncpg://db.invalid:5433/eyvan",
        ),
        (
            "postgresql+asyncpg://FAKE_USER:FAKE_AUTH_SECRET@[::1]:5433/eyvan?password=FAKE_QUERY_SECRET",
            "postgresql+asyncpg://[::1]:5433/eyvan",
        ),
        (
            "postgresql+asyncpg://FAKE_USER:FAKE_AUTH_SECRET@db.invalid:bad/eyvan?password=FAKE_QUERY_SECRET",
            None,
        ),
        ("malformed FAKE_AUTH_SECRET ?password=FAKE_QUERY_SECRET", None),
    ],
    ids=["query-password", "authority-and-query", "ipv6", "invalid-port", "malformed"],
)
def test_refused_connection_diagnostics_omit_user_and_entire_query(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], url: str, target: str | None
) -> None:
    inspected: list[str] = []
    if "FAKE_AUTH_SECRET" not in url:
        _, options = dialect().create_connect_args(make_url(url))
        assert options["password"] == "FAKE_QUERY_SECRET"

    def refuse(config: Config) -> tp.NoReturn:
        inspected.append(config.get_main_option("sqlalchemy.url"))
        raise OSError("synthetic credential-free refusal")

    monkeypatch.setattr(migrate, "inspect_database_state", refuse)
    with pytest.raises(migrate.MigrationConnectionError) as caught:
        migrate.main(["--check", "--database-url", url])
    assert migrate.console_main(["--check", "--database-url", url]) == 1
    captured = capsys.readouterr()
    assert inspected == [url, url]
    for message in (str(caught.value), captured.err):
        assert "could not reach the control-plane database" in message
        assert "synthetic credential-free refusal" in message
        assert "--local-daemon" in message
        assert "FAKE_" not in message
        assert "password=" not in message
        assert "?" not in message
        assert "Traceback" not in message
        if target is not None:
            assert target in message
