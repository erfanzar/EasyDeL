# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An unreachable database is an answer, not a traceback.

A control plane started with ``--local-daemon`` keeps its state in a versioned
local document and has no PostgreSQL at all, so the first thing an operator of
one sees from this command is a refused connection. It answered with sixty lines
of asyncpg stack and no statement of which database it tried, where that URL came
from, or that their deployment needs no migration.
"""

from __future__ import annotations

import pytest

from eyvan.server.persistence import migrate


@pytest.fixture(autouse=True)
def _no_ambient_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)


def _refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_config: object) -> object:
        raise OSError(61, "Connect call failed ('127.0.0.1', 5432)")

    monkeypatch.setattr(migrate, "inspect_database_state", explode)


def test_a_refused_connection_is_a_sentence_not_a_stack(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operator path prints one line and exits non-zero."""

    _refuse(monkeypatch)

    code = migrate.console_main(["--check"])

    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "could not reach the control-plane database" in captured.err


def test_the_message_names_the_url_and_where_it_came_from(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ "Connection refused" without a target is not actionable."""

    _refuse(monkeypatch)

    migrate.console_main(["--check"])

    error = capsys.readouterr().err
    assert "127.0.0.1:5432" in error
    assert "the local-development default" in error


def test_the_message_names_the_configuration_that_makes_this_expected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A local-daemon deployment has nothing to migrate, and should be told so."""

    _refuse(monkeypatch)

    migrate.console_main(["--check"])

    assert "--local-daemon" in capsys.readouterr().err


def test_an_explicit_url_is_attributed_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Which of the two sources produced the URL changes what the operator fixes."""

    monkeypatch.setenv("EYVAN_DATABASE_URL", "postgresql+asyncpg://someone@db.internal:5432/eyvan")
    _refuse(monkeypatch)

    migrate.console_main(["--check"])

    assert "EYVAN_DATABASE_URL" in capsys.readouterr().err


def test_a_password_never_reaches_the_error_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An error message is the most-copied string a tool emits."""

    monkeypatch.setenv("EYVAN_DATABASE_URL", "postgresql+asyncpg://eyvan:hunter2@db.internal:5432/eyvan")
    _refuse(monkeypatch)

    migrate.console_main(["--check"])

    error = capsys.readouterr().err
    assert "hunter2" not in error
    assert "postgresql+asyncpg://db.internal:5432/eyvan" in error
    assert "eyvan:***@" not in error


def test_the_programmatic_entry_point_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` keeps its typed error; only the console wrapper prettifies."""

    _refuse(monkeypatch)

    with pytest.raises(migrate.MigrationConnectionError):
        migrate.main(["--check"])
