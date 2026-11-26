# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan runtime init`` from an installed wheel, with no source tree at all.

This is the defect the file exists for. ``uv build`` needs a directory holding
``pyproject.toml``; a wheel installs into ``site-packages``, which has none. So
the distribution this project publishes -- and every OCI image built from it --
could not provision a GCP fleet with itself:

    could not initialize native GCP runtime: uv build failed:
    .../site-packages does not appear to be a Python project

An installed wheel needs no project directory, because the native-worker wheel
*is* this distribution: the installer already wrote every file and its hash into
``RECORD``. Repacking those bytes reproduces the wheel that was installed rather
than building a second one that may differ from it.

The proof at the bottom is deliberately expensive. It builds the real wheel,
installs it into a throwaway virtual environment, and runs the real console
script with ``uv`` removed from ``PATH``, so a regression that quietly falls
back to building from source fails here instead of on somebody's first GCE box.
Only Eyvan comes from the wheel; its dependencies are shared from the running
environment, because the claim under test is about Eyvan's own installation and
resolving twenty packages over a network would make this test a weather report.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import sysconfig
import typing as tp
import zipfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pytest

import eyvan
from eyvan.sdk.commands.runtime import (
    InstalledDistributionWheelBuilder,
    NativeWorkerWheelBuildError,
    UvNativeWorkerWheelBuilder,
    _default_package_root,
    installed_eyvan_distribution,
    native_worker_wheel_builder,
)

EYVAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT = "example-project-1"
ARTIFACT_ROOT = "gs://eyvan-runtime-artifacts/eyvan"
MODULE_SOURCE = b"VERSION = '1.2.3'\n"
EDITABLE_PTH = b"/src\n"


def _urlsafe_sha256(payload: bytes) -> str:
    """Encode a digest the way ``RECORD`` does."""

    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")


def _install_distribution(root: Path, *, files: dict[str, bytes], extra_record: tp.Sequence[str] = ()) -> Path:
    """Write one PEP 376 installation into ``root`` and return its dist-info."""

    dist_info = root / "acme-1.2.3.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "WHEEL").write_bytes(b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    (dist_info / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: acme\nVersion: 1.2.3\n")
    (dist_info / "INSTALLER").write_bytes(b"uv\n")
    (dist_info / "entry_points.txt").write_bytes(b"[console_scripts]\neyvan = eyvan.sdk.cli:main\n")
    payloads = {
        f"acme-1.2.3.dist-info/{name}": (dist_info / name).read_bytes()
        for name in ("WHEEL", "METADATA", "INSTALLER", "entry_points.txt")
    }
    for name, payload in files.items():
        target = root.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        payloads[name] = payload
    lines = [f"{name},sha256={_urlsafe_sha256(payload)},{len(payload)}" for name, payload in payloads.items()]
    lines.extend(extra_record)
    lines.append("acme-1.2.3.dist-info/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dist_info


def _members(wheel: Path) -> dict[str, bytes]:
    """Return the archive's file members, ignoring directory entries."""

    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def test_a_source_checkout_still_builds_the_wheel_from_source() -> None:
    """Today's behaviour is unchanged where it was already correct.

    ``runtime deploy`` promises to deploy *this tree*; resolving a checkout to
    the installed distribution would quietly deploy the last thing installed.
    """

    package_root = _default_package_root()

    assert package_root == EYVAN_ROOT
    source = native_worker_wheel_builder()
    try:
        assert isinstance(source, UvNativeWorkerWheelBuilder)
        assert source.package_root == EYVAN_ROOT
    finally:
        source.close()


def test_uv_build_is_refused_rather_than_pointed_at_a_directory_with_no_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The old default returned ``site-packages`` and let ``uv build`` fail there.

    That produced an error naming a directory the operator never chose, with no
    flag and no next step. Refusing where the choice is made keeps the
    diagnosis with the code that knows why.
    """

    package = tmp_path / "site-packages" / "eyvan"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(eyvan, "__file__", str(package / "__init__.py"))

    assert _default_package_root() is None
    with pytest.raises(NativeWorkerWheelBuildError, match="not running from a source checkout"):
        UvNativeWorkerWheelBuilder()


def test_a_distribution_other_than_the_running_one_is_never_staged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Staging code this process is not running is the drift, not the fix.

    Every worker in the fleet would install it and every workload would succeed
    on it, which is exactly how a fix that "was deployed" turns out not to be.
    """

    elsewhere = tmp_path / "elsewhere" / "eyvan" / "__init__.py"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("", encoding="utf-8")
    monkeypatch.setattr(eyvan, "__file__", str(elsewhere))

    with pytest.raises(NativeWorkerWheelBuildError, match="this process imported"):
        installed_eyvan_distribution()


def test_repacking_carries_the_installed_files_and_drops_installer_bookkeeping(tmp_path: Path) -> None:
    """A wheel holds packaged files plus wheel metadata -- nothing the installer added.

    Console launchers live outside the install root and are regenerated from
    ``entry_points.txt``; bytecode is never shipped; ``INSTALLER`` and
    ``direct_url.json`` describe this machine's install, so carrying them would
    make the staged bytes depend on which tool ran ``pip``.
    """

    root = tmp_path / "site-packages"
    dist_info = _install_distribution(
        root,
        files={"eyvan/__init__.py": MODULE_SOURCE, "eyvan/capacity/driver.py": b"DRIVER = True\n"},
        extra_record=[
            f"../../../bin/eyvan,sha256={_urlsafe_sha256(b'launcher')},8",
            "eyvan/__pycache__/__init__.cpython-311.pyc,,",
            f"acme-1.2.3.dist-info/REQUESTED,sha256={_urlsafe_sha256(b'')},0",
            f"acme-1.2.3.dist-info/direct_url.json,sha256={_urlsafe_sha256(b'{}')},2",
        ],
    )
    builder = InstalledDistributionWheelBuilder(metadata.Distribution.at(dist_info))
    try:
        wheel = builder.build()
        members = _members(wheel)

        assert wheel.name == "acme-1.2.3-py3-none-any.whl"
        assert set(members) == {
            "eyvan/__init__.py",
            "eyvan/capacity/driver.py",
            "acme-1.2.3.dist-info/METADATA",
            "acme-1.2.3.dist-info/RECORD",
            "acme-1.2.3.dist-info/WHEEL",
            "acme-1.2.3.dist-info/entry_points.txt",
        }
        assert members["eyvan/__init__.py"] == MODULE_SOURCE
        assert members["eyvan/capacity/driver.py"] == b"DRIVER = True\n"
        record = members["acme-1.2.3.dist-info/RECORD"].decode("utf-8").splitlines()
        expected = f"eyvan/__init__.py,sha256={_urlsafe_sha256(MODULE_SOURCE)},{len(MODULE_SOURCE)}"
        assert expected in record
        assert record[-1] == "acme-1.2.3.dist-info/RECORD,,"
    finally:
        builder.close()
    assert not wheel.parent.exists()


def test_a_file_edited_after_installation_is_refused_instead_of_staged(tmp_path: Path) -> None:
    """RECORD is the installer's claim about the bytes; a mismatch is not that wheel.

    Staging it would publish one host's local edit to every worker in the fleet
    under a content hash that looks exactly like a release.
    """

    root = tmp_path / "site-packages"
    dist_info = _install_distribution(root, files={"eyvan/__init__.py": MODULE_SOURCE})
    (root / "eyvan" / "__init__.py").write_bytes(b"VERSION = 'patched'\n")
    builder = InstalledDistributionWheelBuilder(metadata.Distribution.at(dist_info))

    try:
        with pytest.raises(NativeWorkerWheelBuildError, match="does not match the hash RECORD holds"):
            builder.build()
    finally:
        builder.close()


def test_hex_encoded_record_hashes_are_verified_too(tmp_path: Path) -> None:
    """uv copies the wheel's own hex digests into RECORD; pip writes base64.

    Reading only one of the two encodings would silently stop verifying most of
    the files in an installation, which is worse than verifying none of them.
    """

    root = tmp_path / "site-packages"
    dist_info = _install_distribution(root, files={"eyvan/__init__.py": MODULE_SOURCE})
    hex_digest = hashlib.sha256(b"a different module\n").hexdigest()
    lines = [
        f"eyvan/__init__.py,sha256={hex_digest},{len(MODULE_SOURCE)}" if line.startswith("eyvan/__init__.py,") else line
        for line in (dist_info / "RECORD").read_text(encoding="utf-8").splitlines()
    ]
    (dist_info / "RECORD").write_text("\n".join(lines) + "\n", encoding="utf-8")
    builder = InstalledDistributionWheelBuilder(metadata.Distribution.at(dist_info))

    try:
        with pytest.raises(NativeWorkerWheelBuildError, match="does not match the hash RECORD holds"):
            builder.build()
    finally:
        builder.close()


def test_a_wheel_that_installed_files_outside_site_packages_is_refused(tmp_path: Path) -> None:
    """A wheel ``.data`` directory would be dropped silently, so it is refused loudly.

    Console launchers are the legitimate exception: an installer regenerates
    them from ``entry_points.txt``, so leaving them out of the repack is what
    reproduces the original wheel rather than what loses part of it.
    """

    root = tmp_path / "site-packages"
    dist_info = _install_distribution(
        root,
        files={"eyvan/__init__.py": MODULE_SOURCE},
        extra_record=[f"../../../bin/eyvan,sha256={_urlsafe_sha256(b'launcher')},8"],
    )
    launchers_only = InstalledDistributionWheelBuilder(metadata.Distribution.at(dist_info))
    try:
        assert "eyvan/__init__.py" in _members(launchers_only.build())
    finally:
        launchers_only.close()

    record = (dist_info / "RECORD").read_text(encoding="utf-8")
    stray = f"../../../share/eyvan/policy.yaml,sha256={_urlsafe_sha256(b'policy')},6\n"
    (dist_info / "RECORD").write_text(record + stray, encoding="utf-8")
    with_data = InstalledDistributionWheelBuilder(metadata.Distribution.at(dist_info))

    try:
        with pytest.raises(NativeWorkerWheelBuildError, match="outside its package directory"):
            with_data.build()
    finally:
        with_data.close()


def test_a_source_linked_install_says_to_use_its_checkout(tmp_path: Path) -> None:
    """An editable install records no module files, so there is nothing to repack."""

    root = tmp_path / "site-packages"
    (root).mkdir(parents=True, exist_ok=True)
    (root / "_eyvan_editable.pth").write_bytes(EDITABLE_PTH)
    dist_info = _install_distribution(
        root,
        files={},
        extra_record=[f"_eyvan_editable.pth,sha256={_urlsafe_sha256(EDITABLE_PTH)},{len(EDITABLE_PTH)}"],
    )
    builder = InstalledDistributionWheelBuilder(metadata.Distribution.at(dist_info))

    try:
        with pytest.raises(NativeWorkerWheelBuildError, match="source checkout"):
            builder.build()
    finally:
        builder.close()


@dataclass(frozen=True, slots=True)
class WheelInstallation:
    """One throwaway environment holding Eyvan installed from its own wheel."""

    wheel: Path
    venv: Path
    site_packages: Path

    @property
    def console_script(self) -> Path:
        """The ``eyvan`` entry point the installer generated in this environment."""

        return self.venv / "bin" / "eyvan"


def _run(command: tp.Sequence[str], **kwargs: tp.Any) -> subprocess.CompletedProcess[str]:
    """Run one bounded setup command and surface its output on failure."""

    completed = subprocess.run(command, capture_output=True, text=True, timeout=900, **kwargs)
    if completed.returncode != 0:
        raise AssertionError(f"{list(command)} failed ({completed.returncode}):\n{completed.stderr}{completed.stdout}")
    return completed


@pytest.fixture(scope="session")
def wheel_installation(tmp_path_factory: pytest.TempPathFactory) -> WheelInstallation:
    """Build Eyvan's wheel and install it where no source tree can be found."""

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("building and installing Eyvan's own wheel requires uv")
    root = tmp_path_factory.mktemp("installed-wheel")
    _run((uv, "build", "--wheel", "--out-dir", str(root / "dist"), str(EYVAN_ROOT)))
    wheels = sorted((root / "dist").glob("*.whl"))
    assert len(wheels) == 1, wheels
    venv = root / "venv"
    _run((uv, "venv", "--python", sys.executable, str(venv)))
    _run((uv, "pip", "install", "--no-deps", "--python", str(venv / "bin" / "python"), str(wheels[0])))
    site_packages = next(iter(venv.glob("lib/python*/site-packages")))
    # Eyvan is the installed wheel; only its dependencies are shared from the
    # running environment. A .pth line is appended to sys.path *after* this
    # venv's own site-packages, and .pth files are not executed for a plain path
    # entry, so a source-linked Eyvan over there can never shadow this one.
    (site_packages / "zz-eyvan-runtime-dependencies.pth").write_text(
        sysconfig.get_paths()["purelib"] + "\n",
        encoding="utf-8",
    )
    return WheelInstallation(wheel=wheels[0], venv=venv, site_packages=site_packages)


_GCLOUD_STUB = r"""import json
import os
import shutil
import sys

argv = sys.argv[1:]
with open(os.environ["GCLOUD_STUB_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(argv) + "\n")
joined = " ".join(argv)
if joined.startswith("storage objects describe"):
    sys.stderr.write("ERROR: (gcloud.storage.objects.describe) 404 not found\n")
    raise SystemExit(1)
if joined.startswith("storage cp"):
    sources = [value for value in argv if os.path.isfile(value)]
    if len(sources) != 1:
        sys.stderr.write("stub cp needs exactly one local source\n")
        raise SystemExit(2)
    shutil.copyfile(sources[0], os.environ["GCLOUD_STUB_BUCKET"])
    raise SystemExit(0)
if joined.startswith(("projects describe", "storage buckets describe", "compute tpus locations list")):
    raise SystemExit(0)
sys.stderr.write("stub gcloud refuses " + joined + "\n")
raise SystemExit(3)
"""


@dataclass(frozen=True, slots=True)
class StagedRuntime:
    """What one real ``eyvan runtime init`` produced against a stubbed gcloud."""

    document: dict[str, tp.Any]
    staged_wheel: Path
    gcloud_commands: tuple[tuple[str, ...], ...]
    search_path: str


@pytest.fixture(scope="session")
def staged_runtime(
    wheel_installation: WheelInstallation,
    tmp_path_factory: pytest.TempPathFactory,
) -> StagedRuntime:
    """Run the installed console script end to end with uv removed from PATH."""

    root = tmp_path_factory.mktemp("runtime-init")
    stub_directory = root / "bin"
    stub_directory.mkdir()
    (stub_directory / "gcloud.py").write_text(_GCLOUD_STUB, encoding="utf-8")
    stub = stub_directory / "gcloud"
    stub.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{stub_directory / "gcloud.py"}" "$@"\n',
        encoding="utf-8",
    )
    stub.chmod(0o700)
    profile_directory = root / "private"
    profile_directory.mkdir(mode=0o700)
    log = root / "gcloud.log"
    bucket = root / "staged.whl"
    search_path = str(stub_directory)
    completed = subprocess.run(
        (
            str(wheel_installation.console_script),
            "runtime",
            "init",
            "--project",
            PROJECT,
            "--artifact-root",
            ARTIFACT_ROOT,
            "--profile",
            str(profile_directory / "runtime.json"),
            "--yes",
            "--json",
        ),
        capture_output=True,
        text=True,
        timeout=900,
        cwd=root,
        env={
            "PATH": search_path,
            "HOME": str(root / "home"),
            "GCLOUD_STUB_LOG": str(log),
            "GCLOUD_STUB_BUCKET": str(bucket),
        },
    )

    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    return StagedRuntime(
        document=json.loads(completed.stdout),
        staged_wheel=bucket,
        gcloud_commands=tuple(tuple(json.loads(line)) for line in log.read_text(encoding="utf-8").splitlines()),
        search_path=search_path,
    )


def test_runtime_init_succeeds_from_an_installed_wheel_with_no_source_tree(
    wheel_installation: WheelInstallation,
    staged_runtime: StagedRuntime,
) -> None:
    """The headline: the published wheel can provision a fleet with itself.

    ``uv`` is absent from the subprocess ``PATH``, so a fallback to ``uv build``
    could not have produced this -- it would have failed the fixture instead.
    """

    staged = staged_runtime.staged_wheel.read_bytes()

    assert not (wheel_installation.site_packages / "pyproject.toml").exists()
    assert shutil.which("uv", path=staged_runtime.search_path) is None
    assert staged_runtime.document["project_id"] == PROJECT
    assert staged_runtime.document["checks"] == ["project_access", "storage_access", "tpu_access"]
    assert staged_runtime.document["native_worker_wheel_uri"] == (
        f"{ARTIFACT_ROOT}/runtime/wheels/{hashlib.sha256(staged).hexdigest()}/{wheel_installation.wheel.name}"
    )


def test_the_staged_object_is_the_wheel_that_was_installed(
    wheel_installation: WheelInstallation,
    staged_runtime: StagedRuntime,
) -> None:
    """A native worker pip-installs this object, so it must be the same wheel.

    Every member is compared against the wheel that was installed. RECORD is the
    one exception: it is regenerated, because installers write its digests in
    two different encodings and a repack must publish its own.
    """

    staged = _members(staged_runtime.staged_wheel)
    built = _members(wheel_installation.wheel)
    record_name = next(name for name in built if name.endswith(".dist-info/RECORD"))

    assert set(staged) == set(built)
    assert {name: payload for name, payload in staged.items() if name != record_name} == {
        name: payload for name, payload in built.items() if name != record_name
    }
    recorded = dict(
        line.split(",")[:2] for line in staged[record_name].decode("utf-8").splitlines() if line.count(",") == 2
    )
    assert recorded["eyvan/__init__.py"] == f"sha256={_urlsafe_sha256(staged['eyvan/__init__.py'])}"
    assert recorded[record_name] == ""


def test_the_initialized_runtime_pinned_the_object_gcloud_was_asked_to_write(
    staged_runtime: StagedRuntime,
) -> None:
    """The profile must name the object that now exists, not one staging intended."""

    copies = [command for command in staged_runtime.gcloud_commands if command[:2] == ("storage", "cp")]

    assert len(copies) == 1
    assert staged_runtime.document["native_worker_wheel_uri"] in copies[0]
    assert "--no-clobber" in copies[0]


def test_repacking_one_installation_twice_produces_identical_bytes(
    wheel_installation: WheelInstallation,
    tmp_path: Path,
) -> None:
    """The staged object is content-addressed, so an unstable repack re-uploads forever."""

    script = tmp_path / "repack.py"
    script.write_text(
        "import hashlib\n"
        "from eyvan.sdk.commands.runtime import native_worker_wheel_builder\n"
        "digests = []\n"
        "for _ in range(2):\n"
        "    source = native_worker_wheel_builder()\n"
        "    try:\n"
        "        wheel = source.build()\n"
        "        digests.append(hashlib.sha256(wheel.read_bytes()).hexdigest())\n"
        "        name = wheel.name\n"
        "    finally:\n"
        "        source.close()\n"
        "print(type(source).__name__, name, *digests)\n",
        encoding="utf-8",
    )

    completed = _run((str(wheel_installation.venv / "bin" / "python"), str(script)), cwd=tmp_path)

    kind, name, first, second = completed.stdout.split()
    assert kind == "InstalledDistributionWheelBuilder"
    assert name == wheel_installation.wheel.name
    assert first == second


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
