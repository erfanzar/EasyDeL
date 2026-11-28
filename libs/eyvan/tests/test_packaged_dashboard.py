# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A wheel or sdist carries the built dashboard, so ``pip install eyvan`` serves a UI.

The plane used to find its dashboard only beside a source checkout, so every
installed wheel answered "/" with the no-dashboard page. Release builds now copy
``dashboard/dist`` into ``eyvan/_dashboard`` before ``uv build`` and refuse an
artifact that lacks it; the plane and ``eyvan head`` fall back to that copy.
"""

from __future__ import annotations

import importlib.util
import io
import pathlib
import subprocess
import tarfile
import zipfile

import pytest

from eyvan.server.api import app as app_module

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _stage_module():
    spec = importlib.util.spec_from_file_location("stage_dashboard", ROOT / "scripts" / "stage_dashboard.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle(directory: pathlib.Path, marker: str = "a1") -> None:
    (directory / "assets").mkdir(parents=True)
    (directory / "index.html").write_text(
        f'<script type="module" crossorigin src="/assets/index-{marker}.js"></script>'
        f'<link rel="stylesheet" crossorigin href="/assets/index-{marker}.css">',
        encoding="utf-8",
    )
    (directory / "assets" / f"index-{marker}.js").write_text("export {};", encoding="utf-8")
    (directory / "assets" / f"index-{marker}.css").write_text("body{}", encoding="utf-8")


def test_an_installed_package_serves_the_bundle_it_carries(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EYVAN_DASHBOARD_DIR", raising=False)
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "site-packages/eyvan/server/api/app.py"))
    packaged = tmp_path / "site-packages/eyvan/_dashboard"
    _write_bundle(packaged)

    assert app_module.dashboard_bundle() == packaged


def test_a_checkout_build_wins_over_a_stale_packaged_copy(monkeypatch, tmp_path) -> None:
    """A developer who rebuilt the dashboard must see that build, not the copy staged for the last release."""

    monkeypatch.delenv("EYVAN_DASHBOARD_DIR", raising=False)
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "eyvan/server/api/app.py"))
    _write_bundle(tmp_path / "eyvan/_dashboard", "old")
    _write_bundle(tmp_path / "dashboard/dist", "new")

    assert app_module.dashboard_bundle() == tmp_path / "dashboard/dist"


def test_a_packaged_copy_without_an_index_is_not_served(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EYVAN_DASHBOARD_DIR", raising=False)
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "site-packages/eyvan/server/api/app.py"))
    (tmp_path / "site-packages/eyvan/_dashboard/assets").mkdir(parents=True)

    assert app_module.dashboard_bundle() is None


def test_a_head_installed_from_a_wheel_ships_the_packaged_dashboard(monkeypatch, tmp_path) -> None:
    from eyvan.sdk.commands import head

    package = tmp_path / "site-packages/eyvan"
    _write_bundle(package / "_dashboard")
    monkeypatch.setattr(head, "eyvan_package_root", lambda: package)

    bundle = head._local_dashboard_bundle()

    assert bundle is not None
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = {name.removeprefix("./") for name in archive.getnames()}
    assert "index.html" in names
    assert "assets/index-a1.js" in names


def test_staging_copies_the_build_and_drops_files_from_an_older_one(tmp_path) -> None:
    stage = _stage_module()
    source, target = tmp_path / "dist", tmp_path / "eyvan/_dashboard"
    _write_bundle(source)
    (target / "assets").mkdir(parents=True)
    (target / "assets/index-stale.js").write_text("old", encoding="utf-8")

    copied = stage.stage(source, target)

    assert copied == 3
    assert (target / "index.html").read_bytes() == (source / "index.html").read_bytes()
    assert not (target / "assets/index-stale.js").exists()


def test_staging_refuses_a_dashboard_that_was_never_built(tmp_path) -> None:
    stage = _stage_module()
    (tmp_path / "dist").mkdir()

    with pytest.raises(SystemExit, match="npm --prefix dashboard run build"):
        stage.stage(tmp_path / "dist", tmp_path / "eyvan/_dashboard")


def _wheel(path: pathlib.Path, members: dict[str, str]) -> pathlib.Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path


def _sdist(path: pathlib.Path, members: dict[str, str]) -> pathlib.Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(f"eyvan-0.0.1/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _members(tmp_path: pathlib.Path) -> dict[str, str]:
    _write_bundle(tmp_path / "bundle")
    return {
        f"eyvan/_dashboard/{path.relative_to(tmp_path / 'bundle').as_posix()}": path.read_text(encoding="utf-8")
        for path in (tmp_path / "bundle").rglob("*")
        if path.is_file()
    }


def test_the_release_check_accepts_artifacts_that_carry_the_whole_bundle(tmp_path) -> None:
    stage = _stage_module()
    members = _members(tmp_path)

    assert stage.missing(_wheel(tmp_path / "eyvan-0.0.1-py3-none-any.whl", members)) == []
    assert stage.missing(_sdist(tmp_path / "eyvan-0.0.1.tar.gz", members)) == []


def test_the_release_check_refuses_a_wheel_without_the_dashboard(tmp_path) -> None:
    stage = _stage_module()
    wheel = _wheel(tmp_path / "eyvan-0.0.1-py3-none-any.whl", {"eyvan/__init__.py": ""})

    assert stage.missing(wheel) == ["eyvan/_dashboard/index.html"]


def test_the_release_check_refuses_an_index_whose_assets_are_absent(tmp_path) -> None:
    stage = _stage_module()
    members = _members(tmp_path)
    members.pop("eyvan/_dashboard/assets/index-a1.js")

    assert stage.missing(_wheel(tmp_path / "eyvan-0.0.1-py3-none-any.whl", members)) == [
        "eyvan/_dashboard/assets/index-a1.js"
    ]


@pytest.mark.parametrize(
    ("workflow", "build"),
    [(".github/workflows/ci.yaml", "uv build --no-build-isolation"), (".github/workflows/publish.yaml", "uv build")],
)
def test_release_builds_stage_the_dashboard_and_check_what_they_built(workflow: str, build: str) -> None:
    text = (ROOT / workflow).read_text(encoding="utf-8")
    if workflow.endswith("ci.yaml"):
        text = text[text.index("  package:") :]
    built = text.index(build)

    assert text.index("npm --prefix dashboard run build") < text.index("scripts/stage_dashboard.py") < built
    assert text.index("scripts/stage_dashboard.py --check", built) > built


def test_the_staged_copy_is_never_committed() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "eyvan/_dashboard/index.html"], cwd=ROOT, check=False, capture_output=True
    )

    assert result.returncode == 0, "eyvan/_dashboard/ must be gitignored; it is a build output"
