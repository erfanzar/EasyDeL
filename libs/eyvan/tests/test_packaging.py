# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Packaging invariants: pinned image requirements and the packaged migration entrypoint."""

from __future__ import annotations

import re
import sqlite3
import tomllib
import typing as tp
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory

import eyvan.server.persistence.migrate as migrate

EYVAN_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = EYVAN_ROOT / "requirements-docker.txt"
BUILD_REQUIREMENTS_PATH = EYVAN_ROOT / "requirements-build.txt"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pyproject() -> dict[str, tp.Any]:
    with (EYVAN_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_lines(path: Path) -> list[str]:
    """Return logical requirement lines with backslash continuations joined."""

    logical_lines: list[str] = []
    buffer = ""
    for raw_line in path.read_text().splitlines():
        line = buffer + " " + raw_line.strip() if buffer else raw_line.strip()
        if line.endswith("\\"):
            buffer = line[:-1].rstrip()
            continue
        buffer = ""
        if line:
            logical_lines.append(line)
    if buffer:
        logical_lines.append(buffer)
    return logical_lines


def _pinned_requirements(path: Path) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for line in _requirement_lines(path):
        if line.startswith("#"):
            continue
        token = line.split()[0]
        name, _, _version = token.partition("==")
        pinned[_normalize(name)] = token
    return pinned


def test_requirements_docker_exists_and_every_line_is_pinned() -> None:
    assert REQUIREMENTS_PATH.is_file(), "run: uv export --package eyvan --no-dev --no-emit-workspace --locked"
    requirement_lines = [line for line in _requirement_lines(REQUIREMENTS_PATH) if not line.startswith("#")]
    assert requirement_lines, "requirements-docker.txt has no requirement lines"
    for line in requirement_lines:
        token = line.split()[0]
        assert "==" in token, f"requirement is not fully pinned: {token!r}"
        assert "--hash=sha256:" in line, f"requirement is missing hashes: {token!r}"


def test_build_backend_is_exactly_pinned_and_hashed() -> None:
    project = _pyproject()
    assert project["build-system"]["requires"] == ["uv_build==0.7.19"]
    assert project["dependency-groups"]["eyvan-build"] == ["uv-build==0.7.19"]
    assert BUILD_REQUIREMENTS_PATH.is_file()
    requirements = [line for line in _requirement_lines(BUILD_REQUIREMENTS_PATH) if not line.startswith("#")]
    assert len(requirements) == 1
    assert requirements[0].split()[0] == "uv-build==0.7.19"
    assert "--hash=sha256:" in requirements[0]


def test_requirements_cover_every_mandatory_pyproject_dependency() -> None:
    dependencies = _pyproject()["project"]["dependencies"]
    mandatory = {_normalize(re.match(r"[A-Za-z0-9_.-]+", dependency).group(0)) for dependency in dependencies}
    assert mandatory, "pyproject declares no mandatory dependencies"
    pinned = _pinned_requirements(REQUIREMENTS_PATH)
    missing = mandatory - pinned.keys()
    assert not missing, f"mandatory dependencies missing from requirements-docker.txt: {sorted(missing)}"


def test_published_dependency_contract_excludes_known_vulnerable_floors() -> None:
    """Eyvan's own mandatory dependencies may not admit a known-vulnerable version.

    Optional extras are deliberately excluded: ``ray`` is pinned by other
    workspace packages too, so its floor is not Eyvan's to raise here.
    """

    dependencies = set(_pyproject()["project"]["dependencies"])
    assert {
        "click>=8.4.2",
        "cryptography>=50.0.0",
        "fastapi>=0.141.1",
        "idna>=3.18",
        "pygments>=2.20.0",
        "starlette>=1.5.0",
    } <= dependencies


def test_eyvan_declares_no_ray_dependency_in_any_extra() -> None:
    """Eyvan replaces Ray rather than dispatching to it.

    An unused extra is not free: it keeps a competitor in the published
    dependency contract, in every dependency audit, and in every reader's model
    of what Eyvan needs to run.
    """

    project = _pyproject()["project"]
    assert "ray" not in project["optional-dependencies"]
    declared = " ".join(project["dependencies"]) + " " + str(project["optional-dependencies"])
    assert "ray" not in declared.lower().replace("array", "")


def test_pyproject_registers_migrate_entrypoint() -> None:
    """The script must point at the wrapper, not the raising entry point.

    ``main`` raises typed errors because a Python caller wants them; an operator
    at a terminal wants a sentence. Pointing the console script back at ``main``
    would restore the sixty-line asyncpg traceback that a --local-daemon
    deployment -- which has no PostgreSQL and needs no migration at all -- sees
    the very first time it runs this command.
    """

    scripts = _pyproject()["project"]["scripts"]
    assert scripts.get("eyvan-migrate") == "eyvan.server.persistence.migrate:console_main"


def test_alembic_config_points_at_packaged_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)
    config = migrate.build_alembic_config("postgresql+asyncpg://eyvan:eyvan@db:5432/eyvan")
    script_location = Path(config.get_main_option("script_location"))
    assert script_location == migrate.MIGRATIONS_DIR
    assert (script_location / "env.py").is_file()
    assert (script_location / "versions").is_dir()
    assert config.get_main_option("sqlalchemy.url") == "postgresql+asyncpg://eyvan:eyvan@db:5432/eyvan"


def test_alembic_config_honors_database_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_DATABASE_URL", "postgresql+asyncpg://eyvan:eyvan@db.internal:5432/eyvan")
    config = migrate.build_alembic_config()
    assert config.get_main_option("sqlalchemy.url") == "postgresql+asyncpg://eyvan:eyvan@db.internal:5432/eyvan"


def test_packaged_migrations_are_discoverable_by_alembic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)
    script = ScriptDirectory.from_config(migrate.build_alembic_config())
    discovered = {revision.revision for revision in script.walk_revisions()}
    version_files = {path.stem for path in migrate.MIGRATIONS_DIR.glob("versions/*.py") if path.stem != "__init__"}
    assert version_files, "no migration scripts are packaged"
    assert discovered == version_files


def _packaged_migration_state(
    *,
    tables: tuple[str, ...],
    current_revisions: tuple[str, ...],
) -> migrate.MigrationDatabaseState:
    script = ScriptDirectory.from_config(migrate.build_alembic_config())
    return migrate.classify_database_state(
        tables=tables,
        current_revisions=current_revisions,
        script_directory=script,
    )


def _packaged_head_and_parent() -> tuple[str, str]:
    script = ScriptDirectory.from_config(migrate.build_alembic_config())
    heads = script.get_heads()
    assert len(heads) == 1
    head = script.get_revision(heads[0])
    down_revision = head.down_revision
    if isinstance(down_revision, tuple):
        assert down_revision
        parent = sorted(down_revision)[0]
    else:
        assert isinstance(down_revision, str)
        parent = down_revision
    return head.revision, parent


def _offline_script_directory(
    root: Path,
    revisions: dict[str, str | tuple[str, ...] | None],
) -> ScriptDirectory:
    versions = root / "versions"
    versions.mkdir(parents=True)
    for index, (revision, down_revision) in enumerate(revisions.items()):
        (versions / f"{index:02d}_{revision}.py").write_text(
            f"revision = {revision!r}\ndown_revision = {down_revision!r}\nbranch_labels = None\ndepends_on = None\n"
        )
    return ScriptDirectory(str(root))


def _unsafe_migration_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, migrate.MigrationDatabaseState]:
    tables = ("alembic_version", "workloads")

    packaged = _offline_script_directory(
        tmp_path / "unknown",
        {"base": None, "packaged_head": "base"},
    )
    unknown = migrate.classify_database_state(
        tables=tables,
        current_revisions=("not_in_the_package",),
        script_directory=packaged,
    )

    ahead_graph = _offline_script_directory(
        tmp_path / "ahead",
        {"base": None, "packaged_head": "base", "future_head": "packaged_head"},
    )
    monkeypatch.setattr(ahead_graph, "get_heads", lambda: ["packaged_head"])
    ahead = migrate.classify_database_state(
        tables=tables,
        current_revisions=("future_head",),
        script_directory=ahead_graph,
    )

    diverged_graph = _offline_script_directory(
        tmp_path / "diverged",
        {"base": None, "packaged_head": "base", "other_branch": "base"},
    )
    monkeypatch.setattr(diverged_graph, "get_heads", lambda: ["packaged_head"])
    diverged = migrate.classify_database_state(
        tables=tables,
        current_revisions=("other_branch",),
        script_directory=diverged_graph,
    )

    merged_graph = _offline_script_directory(
        tmp_path / "multiple-current",
        {
            "base": None,
            "left": "base",
            "right": "base",
            "merge_head": ("left", "right"),
        },
    )
    multiple_current = migrate.classify_database_state(
        tables=tables,
        current_revisions=("left", "right"),
        script_directory=merged_graph,
    )

    branched_graph = _offline_script_directory(
        tmp_path / "multiple-heads",
        {"base": None, "left_head": "base", "right_head": "base"},
    )
    multiple_heads = migrate.classify_database_state(
        tables=tables,
        current_revisions=("left_head",),
        script_directory=branched_graph,
    )
    return {
        "unknown": unknown,
        "ahead": ahead,
        "diverged": diverged,
        "multiple-current": multiple_current,
        "multiple-heads": multiple_heads,
    }


def test_migration_database_inspection_distinguishes_fresh_and_unknown_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "migration-policy.db"
    config = migrate.build_alembic_config(f"sqlite+aiosqlite:///{database}")

    fresh = migrate.inspect_database_state(config)
    assert fresh.classification is migrate.MigrationClassification.FRESH
    assert fresh.current_revisions == ()
    assert len(fresh.expected_heads) == 1

    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE existing_control_plane_data (id INTEGER PRIMARY KEY)")

    existing = migrate.inspect_database_state(config)
    assert existing.classification is migrate.MigrationClassification.UNKNOWN
    assert existing.requires_maintenance is False
    with pytest.raises(migrate.UnsafeAutomaticMigrationError, match="authorizes only"):
        migrate.require_safe_migration(existing, maintenance_acknowledged=True)


def test_packaged_graph_classifies_exact_head_and_known_single_ancestor() -> None:
    head, parent = _packaged_head_and_parent()
    exact = _packaged_migration_state(
        tables=("alembic_version", "workloads"),
        current_revisions=(head,),
    )
    behind = _packaged_migration_state(
        tables=("alembic_version", "workloads"),
        current_revisions=(parent,),
    )

    assert exact.classification is migrate.MigrationClassification.EXACT_HEAD
    assert behind.classification is migrate.MigrationClassification.BEHIND
    assert behind.requires_maintenance is True


def test_migrate_main_upgrades_fresh_database_with_env_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tp.Any, str]] = []

    def fake_upgrade(config: tp.Any, revision: str, **_kwargs: tp.Any) -> None:
        calls.append((config, revision))

    monkeypatch.setattr(migrate.command, "upgrade", fake_upgrade)
    monkeypatch.setattr(
        migrate,
        "inspect_database_state",
        lambda _config: _packaged_migration_state(tables=(), current_revisions=()),
    )
    monkeypatch.setenv("EYVAN_DATABASE_URL", "postgresql+asyncpg://eyvan:eyvan@db.internal:5432/eyvan")

    assert migrate.main([]) == 0
    assert len(calls) == 1
    config, revision = calls[0]
    assert revision == "head"
    assert config.get_main_option("sqlalchemy.url") == "postgresql+asyncpg://eyvan:eyvan@db.internal:5432/eyvan"
    assert Path(config.get_main_option("script_location")) == migrate.MIGRATIONS_DIR


def test_migrate_help_never_opens_or_changes_a_database(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    called = False

    def fail_upgrade(*_args: tp.Any, **_kwargs: tp.Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(migrate.command, "upgrade", fail_upgrade)
    monkeypatch.setattr(
        migrate,
        "inspect_database_state",
        lambda _config: pytest.fail("--help must not inspect a database"),
    )
    with pytest.raises(SystemExit, match="0"):
        migrate.main(["--help"])
    assert called is False
    assert "Upgrade the Eyvan control-plane database" in capsys.readouterr().out


def test_migrate_can_render_offline_sql_without_inspecting_or_applying(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool, str]] = []

    def fake_upgrade(config: tp.Any, revision: str, *, sql: bool) -> None:
        calls.append((revision, sql, config.get_main_option("sqlalchemy.url")))

    monkeypatch.setattr(migrate.command, "upgrade", fake_upgrade)
    monkeypatch.setattr(
        migrate,
        "inspect_database_state",
        lambda _config: pytest.fail("offline SQL must not inspect a database"),
    )
    assert migrate.main(["--database-url", "postgresql+asyncpg://example/db", "--sql", "0011_quota_grants"]) == 0
    assert calls == [("0011_quota_grants", True, "postgresql+asyncpg://example/db")]


def test_migrate_rejects_an_online_non_head_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        migrate,
        "inspect_database_state",
        lambda _config: pytest.fail("an invalid online target must be rejected before database I/O"),
    )
    with pytest.raises(migrate.UnsafeAutomaticMigrationError, match="fixed to packaged head"):
        migrate.main(["0011_quota_grants"])


def test_migrate_main_keeps_same_head_rollouts_usable_without_acknowledgement(monkeypatch: pytest.MonkeyPatch) -> None:
    head, _parent = _packaged_head_and_parent()
    calls: list[str] = []
    monkeypatch.setattr(
        migrate,
        "inspect_database_state",
        lambda _config: _packaged_migration_state(
            tables=("alembic_version", "workloads"),
            current_revisions=(head,),
        ),
    )
    monkeypatch.setattr(migrate.command, "upgrade", lambda _config, revision: calls.append(revision))

    assert migrate.main([]) == 0
    assert calls == ["head"]


def test_one_use_acknowledgement_is_rejected_for_fresh_and_exact_head() -> None:
    head, _parent = _packaged_head_and_parent()
    states = (
        _packaged_migration_state(tables=(), current_revisions=()),
        _packaged_migration_state(
            tables=("alembic_version", "workloads"),
            current_revisions=(head,),
        ),
    )

    for state in states:
        with pytest.raises(migrate.UnsafeAutomaticMigrationError, match="unnecessary"):
            migrate.require_safe_migration(state, maintenance_acknowledged=True)


def test_migrate_refuses_a_behind_schema_without_maintenance_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _head, parent = _packaged_head_and_parent()
    monkeypatch.setattr(
        migrate,
        "inspect_database_state",
        lambda _config: _packaged_migration_state(
            tables=("alembic_version", "workloads"),
            current_revisions=(parent,),
        ),
    )
    monkeypatch.setattr(
        migrate.command,
        "upgrade",
        lambda *_args, **_kwargs: pytest.fail("an unacknowledged migration must not run"),
    )

    with pytest.raises(migrate.UnsafeAutomaticMigrationError, match="Quiesce every writer"):
        migrate.main([])


def test_migrate_acknowledgement_allows_only_the_known_behind_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    _head, parent = _packaged_head_and_parent()
    calls: list[str] = []
    monkeypatch.setattr(
        migrate,
        "inspect_database_state",
        lambda _config: _packaged_migration_state(
            tables=("alembic_version", "workloads"),
            current_revisions=(parent,),
        ),
    )
    monkeypatch.setattr(migrate.command, "upgrade", lambda _config, revision: calls.append(revision))

    assert migrate.main(["--maintenance-acknowledged"]) == 0
    assert calls == ["head"]


def test_migrate_check_is_non_mutating_for_safe_states(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    head, parent = _packaged_head_and_parent()
    states_and_arguments = (
        (_packaged_migration_state(tables=(), current_revisions=()), ["--check"]),
        (
            _packaged_migration_state(
                tables=("alembic_version", "workloads"),
                current_revisions=(head,),
            ),
            ["--check"],
        ),
        (
            _packaged_migration_state(
                tables=("alembic_version", "workloads"),
                current_revisions=(parent,),
            ),
            ["--check", "--maintenance-acknowledged"],
        ),
    )
    monkeypatch.setattr(
        migrate.command,
        "upgrade",
        lambda *_args, **_kwargs: pytest.fail("--check must never mutate the database"),
    )

    for state, arguments in states_and_arguments:
        monkeypatch.setattr(migrate, "inspect_database_state", lambda _config, state=state: state)
        assert migrate.main(arguments) == 0
    output = capsys.readouterr().out
    assert "fresh" in output
    assert "at packaged head" in output
    assert "known behind; maintenance acknowledged" in output


def test_unknown_ahead_diverged_and_multi_revision_states_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _unsafe_migration_states(tmp_path, monkeypatch)
    assert {name: state.classification for name, state in states.items()} == {
        "unknown": migrate.MigrationClassification.UNKNOWN,
        "ahead": migrate.MigrationClassification.AHEAD,
        "diverged": migrate.MigrationClassification.DIVERGED,
        "multiple-current": migrate.MigrationClassification.MULTIPLE_CURRENT_REVISIONS,
        "multiple-heads": migrate.MigrationClassification.MULTIPLE_PACKAGED_HEADS,
    }

    for state in states.values():
        for acknowledged in (False, True):
            with pytest.raises(migrate.UnsafeAutomaticMigrationError, match="unsafe Alembic"):
                migrate.require_safe_migration(state, maintenance_acknowledged=acknowledged)


def test_check_refuses_every_unsafe_graph_state_even_with_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _unsafe_migration_states(tmp_path, monkeypatch)
    monkeypatch.setattr(
        migrate.command,
        "upgrade",
        lambda *_args, **_kwargs: pytest.fail("--check must never mutate the database"),
    )

    for state in states.values():
        monkeypatch.setattr(migrate, "inspect_database_state", lambda _config, state=state: state)
        with pytest.raises(migrate.UnsafeAutomaticMigrationError, match="authorizes only"):
            migrate.main(["--check", "--maintenance-acknowledged"])


def test_dockerfiles_install_from_pinned_requirements() -> None:
    for name in ("Dockerfile", "Dockerfile.agent", "Dockerfile.worker"):
        content = (EYVAN_ROOT / name).read_text()
        assert "uv sync" not in content, f"{name} still re-resolves dependencies at image-build time"
        assert "requirements-docker.txt" in content, f"{name} does not install the pinned runtime requirements"
        assert "requirements-build.txt" in content, f"{name} does not install the pinned build backend"
        assert "--no-build-isolation" in content, f"{name} still resolves an isolated build backend"
        assert "uv pip uninstall uv-build" in content, f"{name} leaves the build backend in the runtime image"


def test_control_plane_agent_and_worker_images_have_distinct_entrypoints_and_oci_labels() -> None:
    control = (EYVAN_ROOT / "Dockerfile").read_text()
    agent = (EYVAN_ROOT / "Dockerfile.agent").read_text()
    worker = (EYVAN_ROOT / "Dockerfile.worker").read_text()

    assert 'ENTRYPOINT ["eyvan-server"]' in control
    assert 'ENTRYPOINT ["eyvan-agent"]' in agent
    assert 'ENTRYPOINT ["eyvan-native-worker"]' in worker
    assert 'ENTRYPOINT ["eyvan-agent"]' not in control
    assert "eyvan:latest" not in control
    assert "eyvan-agent:latest" not in agent
    assert ":latest" not in worker
    for dockerfile in (control, agent, worker):
        assert "org.opencontainers.image.version" in dockerfile
        assert "org.opencontainers.image.revision" in dockerfile
        assert "org.opencontainers.image.licenses" in dockerfile


def test_only_the_control_plane_image_bundles_the_hosted_dashboard() -> None:
    control = (EYVAN_ROOT / "Dockerfile").read_text()
    agent = (EYVAN_ROOT / "Dockerfile.agent").read_text()
    worker = (EYVAN_ROOT / "Dockerfile.worker").read_text()

    assert "EYVAN_DASHBOARD_DIR=/app/dashboard/dist" in control
    assert "COPY --from=dashboard /dashboard/dist /app/dashboard/dist" in control
    assert "EYVAN_DASHBOARD_DIR" not in agent
    assert "COPY --from=dashboard" not in agent
    assert "EYVAN_DASHBOARD_DIR" not in worker
    assert "COPY --from=dashboard" not in worker


def test_package_smoke_builds_and_installs_both_artifact_types_outside_the_source_tree() -> None:
    script_path = EYVAN_ROOT / "scripts" / "smoke-package-artifacts.sh"
    script = script_path.read_text()

    assert script_path.stat().st_mode & 0o111
    assert "build --no-build-isolation --wheel --sdist" in script
    assert "dist/*.whl" in script
    assert "dist/*.tar.gz" in script
    assert 'cd "${tmp}/outside-source"' in script
    assert '"${UV_BIN}" pip install' in script
    assert '"${ROOT}/requirements-build.txt"' in script
    assert "--no-build-isolation" in script
    assert '"${venv}/bin/${entrypoint}" --help' in script
    assert "python -I" not in script  # invocation is the isolated venv path, not ambient Python
    assert "uv run" not in script


def test_migration_job_uses_wheel_entrypoint() -> None:
    content = (EYVAN_ROOT / "deploy" / "migration-job.yaml").read_text()
    assert 'command: ["eyvan-migrate"]' in content
    assert "alembic" not in content


def test_lazy_create_app_reexport_matches_the_real_factory_signature() -> None:
    """``eyvan.server.api.create_app`` must expose every parameter of the real factory.

    The lazy re-export exists so ``import eyvan.server.api`` does not pull in FastAPI.
    It re-declares the signature by hand, so any parameter added to
    ``eyvan.server.api.app.create_app`` and not mirrored here becomes silently
    unreachable through the public entry point.
    """

    import inspect

    import eyvan.server.api as lazy_api
    from eyvan.server.api.app import create_app as real_create_app

    lazy = inspect.signature(lazy_api.create_app)
    real = inspect.signature(real_create_app)

    assert list(lazy.parameters) == list(real.parameters)
    for name, real_parameter in real.parameters.items():
        assert lazy.parameters[name].kind is real_parameter.kind, f"{name} changed parameter kind"
        assert lazy.parameters[name].default == real_parameter.default, f"{name} default drifted"


def test_the_worker_image_stages_the_multi_host_initialization_hook() -> None:
    """Both slice executors put this path on a multi-host workload's PYTHONPATH.

    Only the GCP VM startup script ever created it, so on Kubernetes the
    directory did not exist, ``sitecustomize`` never ran, and
    ``jax.distributed.initialize`` was never called on the workload's behalf.
    Nothing failed -- the ranks ran as independent single-process jobs, which is
    the wrong-results failure rather than the loud one.
    """

    from eyvan.worker.runtime.native_gang import _MULTIHOST_SITE_PATH as GANG_PATH
    from eyvan.worker.runtime.native_slice import _MULTIHOST_SITE_PATH as SLICE_PATH

    worker = (EYVAN_ROOT / "Dockerfile.worker").read_text()

    assert GANG_PATH == SLICE_PATH, "the two executors must stage the same hook directory"
    assert f"install -d -m 0755 {GANG_PATH}" in worker
    assert f"{GANG_PATH}/eyvan_multihost.py" in worker
    assert f"{GANG_PATH}/eyvan_distributed_environment.py" in worker
    assert f"{GANG_PATH}/sitecustomize.py" in worker
    assert "initialize_worker_runtime()" in worker


def test_every_image_pins_an_interpreter_inside_the_supported_range() -> None:
    """`uv venv` alone takes the newest interpreter uv knows about.

    That became 3.14 -- outside this package's own ``>=3.10,<3.14`` -- and the
    hashed exports carry wheels for supported versions only, so the first real
    image build fell through to compiling httptools from source in a base image
    with no compiler. Local wheel and sdist smoke never saw it: they run in the
    workspace's own interpreter.
    """

    import re

    from packaging.specifiers import SpecifierSet

    supported = SpecifierSet(_pyproject()["project"]["requires-python"])
    for name in ("Dockerfile", "Dockerfile.agent", "Dockerfile.worker"):
        content = (EYVAN_ROOT / name).read_text()
        pinned = re.findall(r"uv venv --python (\S+)", content)
        assert pinned, f"{name} does not pin an interpreter for uv venv"
        for version in pinned:
            assert f"{version}.0" in supported, f"{name} pins {version}, outside {supported}"
