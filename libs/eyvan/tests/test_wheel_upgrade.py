# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A development-baseline wheel can be replaced without losing PostgreSQL data.

There are no published tags yet. The immutable baseline is explicitly a source
commit, not a claimed previous release; the candidate gets a scratch-only local
version suffix. Real DDL evolution is tested separately by test_postgres_semantics.
"""

import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

BASELINE = "0e82bebcd8d96f08bd7e1df460208f73a2e948fa"  # pragma: allowlist secret
ROOT = Path(__file__).resolve().parents[1]


def _run(command, *, cwd, environment):
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _extract_baseline(target):
    archive = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            BASELINE,
            "eyvan",
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "requirements-docker.txt",
            "requirements-build.txt",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert archive.returncode == 0, archive.stderr.decode()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as source:
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and ".." not in name.parts
            destination = target.joinpath(*name.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                assert member.isfile(), "baseline archives must contain no links or special files"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, destination.open("xb") as output:
                    shutil.copyfileobj(stream, output)


# This child is invoked with -I from a scratch directory, so an import cannot
# accidentally resolve to the checkout instead of the installed wheel.
CHILD = r"""
import asyncio, json, os, sys
from pathlib import Path
from importlib.metadata import version
import eyvan
from sqlalchemy.ext.asyncio import create_async_engine
from eyvan.records import CommandExecution, Principal, Role, RoleGrant, WorkloadDefinition, WorkloadState
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
assert 'site-packages' in Path(eyvan.__file__).parts
async def run():
    engine = create_async_engine(os.environ['EYVAN_DATABASE_URL'])
    if sys.argv[1] == 'seed':
        await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        service = ControlPlaneService(repository)
        admin = Principal(subject='upgrade-owner', grants=(RoleGrant(role=Role.FLEET_ADMIN),))
        if sys.argv[1] == 'seed':
            row, replayed = await service.submit_workload(admin, org_id='upgrade', project_id='test',
                spec=WorkloadDefinition(name='preserved-work', execution=CommandExecution(command='true'),
                    metadata={'upgrade-proof': 'baseline-data'}), idempotency_key='baseline-submit')
            assert not replayed
            print(json.dumps({'id': row.id, 'version': version('eyvan')}))
        else:
            identifier = sys.argv[2]
            async with repository.transaction() as tx:
                row = await tx.get_workload(identifier)
                assert row.spec.name == 'preserved-work'
                assert row.spec.metadata['upgrade-proof'] == 'baseline-data'
                assert row.state is WorkloadState.PENDING
            await service.cancel_workload(admin, identifier, idempotency_key='candidate-cancel')
            async with repository.transaction() as tx:
                assert (await tx.get_workload(identifier)).state is WorkloadState.CANCELLED
            print(json.dumps({'id': identifier, 'version': version('eyvan'), 'state': 'cancelled'}))
    finally:
        await repository.close()
        await engine.dispose()
asyncio.run(run())
"""


def _empty_public_schema(url: str) -> None:
    """Start from an empty schema, as the sibling PostgreSQL fixtures do.

    ``postgres_url`` is one database for the whole session, and other tests
    reset it on entry rather than on exit. Without this, the baseline wheel
    seeds a database a previous test already stamped at a newer revision than
    the baseline knows, and fails resolving a revision it never shipped.
    """

    async def reset() -> None:
        engine = create_async_engine(url, poolclass=None)
        try:
            async with engine.begin() as connection:
                await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
                await connection.execute(text("CREATE SCHEMA public"))
        finally:
            await engine.dispose()

    asyncio.run(reset())


def test_a_candidate_wheel_migrates_and_preserves_baseline_data(postgres_url, tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    _empty_public_schema(postgres_url)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EYVAN_") and key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    environment["EYVAN_DATABASE_URL"] = postgres_url
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _extract_baseline(baseline)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    assert not any(path.is_symlink() for path in (ROOT / "eyvan").rglob("*"))
    shutil.copytree(ROOT / "eyvan", candidate / "eyvan", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for filename in ("README.md", "LICENSE", "pyproject.toml"):
        shutil.copy2(ROOT / filename, candidate / filename)
    manifest = (candidate / "pyproject.toml").read_text()
    current_version = re.search(r'^version = "([^"]+)"', manifest, re.M).group(1)
    candidate_version = current_version.split("+", 1)[0] + "+upgrade.test"
    (candidate / "pyproject.toml").write_text(
        manifest.replace(
            f'version = "{current_version}"',
            f'version = "{candidate_version}"',
            1,
        )
    )
    venv = tmp_path / "venv"
    _run([uv, "venv", "--python", sys.executable, str(venv)], cwd=tmp_path, environment=environment)
    python = str(venv / "bin/python")
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--require-hashes",
            "-r",
            str(baseline / "requirements-docker.txt"),
            "-r",
            str(baseline / "requirements-build.txt"),
        ],
        cwd=tmp_path,
        environment=environment,
    )
    wheels = []
    for source in (baseline, candidate):
        output = source / "wheel"
        _run(
            [uv, "build", "--python", python, "--no-build-isolation", "--wheel", "--out-dir", str(output)],
            cwd=source,
            environment=environment,
        )
        built = list(output.glob("*.whl"))
        assert len(built) == 1
        wheels.append(built[0])
    _run([uv, "pip", "install", "--python", python, "--no-deps", str(wheels[0])], cwd=tmp_path, environment=environment)
    seeded = json.loads(_run([python, "-I", "-c", CHILD, "seed"], cwd=tmp_path, environment=environment))
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--require-hashes",
            "-r",
            str(ROOT / "requirements-docker.txt"),
            "-r",
            str(ROOT / "requirements-build.txt"),
        ],
        cwd=tmp_path,
        environment=environment,
    )
    _run([uv, "pip", "install", "--python", python, "--no-deps", str(wheels[1])], cwd=tmp_path, environment=environment)
    inspection = (
        "from eyvan.server.persistence.migrate import build_alembic_config, inspect_database_state; "
        "print(inspect_database_state(build_alembic_config()).classification.value)"
    )
    state = _run([python, "-I", "-c", inspection], cwd=tmp_path, environment=environment).strip()
    acknowledgement = ["--maintenance-acknowledged"] if state == "behind" else []
    _run(
        [python, "-I", "-m", "eyvan.server.persistence.migrate", *acknowledgement], cwd=tmp_path, environment=environment
    )
    upgraded = json.loads(
        _run([python, "-I", "-c", CHILD, "verify", seeded["id"]], cwd=tmp_path, environment=environment)
    )
    assert upgraded["version"] == candidate_version
    assert upgraded["version"] != seeded["version"]
    assert upgraded["id"] == seeded["id"] and upgraded["state"] == "cancelled"
