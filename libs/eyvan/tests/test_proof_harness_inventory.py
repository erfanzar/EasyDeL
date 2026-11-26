# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The monorepo proof inputs are tracked templates, not operator-local state."""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import tarfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from eyvan.capacity.fleet_config import EyvanFleetConfig

_MONOREPO = Path(__file__).resolve().parents[3]
_CONFIGS = _MONOREPO / "configs"
_SCRIPTS = _MONOREPO / "scripts"
_VALUES = {
    "GCP_PROJECT_ID": "example-eyvan-project",
    "GCS_BUCKET": "example-eyvan-artifacts",
    "OCI_IMAGE": ("us-east5-docker.pkg.dev/example-eyvan-project/eyvan/easydel-oci-smoke@sha256:" + "0" * 64),
}


def _monorepo_file(path: Path) -> Path:
    if not path.exists():
        pytest.skip("proof inputs live in the EasyDeL monorepo, not the standalone Eyvan checkout")
    return path


def _renderer() -> ModuleType:
    path = _monorepo_file(_SCRIPTS / "render_eyvan_config.py")
    spec = importlib.util.spec_from_file_location("render_eyvan_config_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_gcp_proof_config_is_a_sanitized_renderable_template(tmp_path: Path) -> None:
    templates = sorted(_monorepo_file(_CONFIGS).glob("eyvan-*.yaml.in"))
    assert templates, "the source-of-truth checkout contains no reproducible Eyvan templates"
    renderer = _renderer()

    for template in templates:
        source = template.read_text(encoding="utf-8")
        parsed_template = yaml.safe_load(source)
        assert parsed_template["provider"]["project"] == "{{GCP_PROJECT_ID}}"
        assert "/Users/" not in source
        assert all(bucket == "{{GCS_BUCKET}}" for bucket in re.findall(r"gs://([^/]+)", source))
        rendered = renderer.render_template(template, overrides=_VALUES)
        destination = tmp_path / template.name.removesuffix(".in")
        destination.write_text(rendered, encoding="utf-8")
        loaded = EyvanFleetConfig.from_yaml(destination)
        assert loaded.provider.project == _VALUES["GCP_PROJECT_ID"]


def test_local_proof_config_needs_no_operator_identity() -> None:
    path = _monorepo_file(_CONFIGS / "eyvan-proof-local.yaml")
    source = path.read_text(encoding="utf-8")
    assert "{{" not in source
    assert EyvanFleetConfig.from_yaml(path).provider.type == "local"


def test_mutating_live_harnesses_require_an_explicit_acknowledgement() -> None:
    harnesses = (
        "eyvan_easydel_acceptance.py",
        "eyvan_easydel_v5p16_acceptance.py",
        "eyvan_live_v6e16_oci.py",
        "eyvan_native_v5p16_live.py",
    )
    for name in harnesses:
        source = _monorepo_file(_SCRIPTS / name).read_text(encoding="utf-8")
        assert "confirm-provider-mutations" in source or "EYVAN_LIVE_CONFIRM_PROVIDER_MUTATIONS" in source


def test_live_controller_bootstrap_uses_only_locked_downloads() -> None:
    path = _monorepo_file(_SCRIPTS / "eyvan_live_controller_bootstrap.sh")
    source = path.read_text(encoding="utf-8")

    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0
    assert "astral.sh/uv/install.sh" not in source
    assert re.search(r"curl[^\n]*\|", source) is None
    assert "python3 -I" in source
    assert "HttpsOnlyRedirect" in source
    assert "signal.alarm(600)" in source
    assert 'response.headers.get("Content-Length")' in source
    assert "remaining + 1" in source
    assert "locked download exceeded its byte cap" in source
    assert 'GOOGLE_CLOUD_CLI_VERSION="579.0.0-0"' in source
    assert "google-cloud-cli_579.0.0-0_amd64_7df6ff976357178feae6af1049593fde.deb" in source
    assert "google-cloud-cli_579.0.0-0_arm64_abd121e390555efe93be4f4c2d626210.deb" in source
    assert 'gcloud_size="95806534"' in source
    assert 'gcloud_size="61296360"' in source
    assert "683cea1c789986f2c5f147b5bdb634f45c2727327084c6a89862bd544fd8b297" in source
    assert "e319048d00c5b9547231c1d57a27efe13ff4160557b5191fc13c7a300898e88d" in source
    assert 'UV_VERSION="0.9.21"' in source
    assert "0a1ab27383c28ef1c041f85cbbc609d8e3752dfb4b238d2ad97b208a52232baf" in source
    assert "416984484783a357170c43f98e7d2d203f1fb595d6b3b95131513c53e50986ef" in source
    assert "actual_members" in source and "expected_members" in source
    assert "printf '%s/\\n%s/uv\\n%s/uvx'" in source
    assert '&& ! -L "${scratch}/uv/${uv_bundle}/${member}"' in source
    assert 'install_locked_deb google-cloud-cli "${GOOGLE_CLOUD_CLI_VERSION}" "${gcloud_deb}"' in source
    assert 'apt-get install -y --reinstall --no-install-recommends "${deb}"' in source
    assert "apt-key.gpg" not in source


def test_fresh_archive_smoke_is_vcs_free_locked_and_distribution_complete() -> None:
    path = _monorepo_file(_SCRIPTS / "smoke_eyvan_fresh_archive.sh")
    source = path.read_text(encoding="utf-8")

    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0
    assert 'git -C "${repository}" archive --format=tar "${candidate}"' in source
    assert 'find "${archive}" -name .git' in source
    assert "python3 scripts/verify_eyvan_release.py" in source
    assert "uv sync --locked --package eyvan --no-dev --group eyvan-build --no-install-project" in source
    assert "uv pip install --python .venv/bin/python --no-build-isolation --no-deps libs/eyvan" in source
    assert "eyvan-native-worker" in source and "eyvan-migrate" in source
    assert "uv build --no-build-isolation --package eyvan" in source
    assert 'scripts/smoke_eyvan_distribution.sh "${version}"' in source
    assert "--locked-runtime libs/eyvan/requirements-docker.txt" in source


@pytest.mark.parametrize(
    ("installed", "expected_fragment", "succeeds"),
    (
        ("", "install -y --no-install-recommends /tmp/locked.deb", True),
        ("578.0.0-0", "install -y --no-install-recommends /tmp/locked.deb", True),
        ("579.0.0-0", "install -y --reinstall --no-install-recommends /tmp/locked.deb", True),
        ("580.0.0-0", "", False),
    ),
)
def test_live_controller_bootstrap_forces_exact_deb_bytes(
    tmp_path: Path,
    installed: str,
    expected_fragment: str,
    succeeds: bool,
) -> None:
    path = _monorepo_file(_SCRIPTS / "eyvan_live_controller_bootstrap.sh")
    log = tmp_path / "apt.log"
    probe = r"""source "${BOOTSTRAP_PATH}"
dpkg-query() {
    [[ -n "${INSTALLED}" ]] || return 1
    printf '%s' "${INSTALLED}"
}
dpkg() {
    [[ "$1" == "--compare-versions" ]]
    case "$3" in
        gt) [[ "${INSTALLED}" == "580.0.0-0" ]] ;;
        eq) [[ "${INSTALLED}" == "$4" ]] ;;
        *) return 2 ;;
    esac
}
apt-get() {
    printf '%s\n' "$*" >>"${APT_LOG}"
}
install_locked_deb google-cloud-cli 579.0.0-0 /tmp/locked.deb
"""
    result = subprocess.run(
        ["bash", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env={
            "APT_LOG": str(log),
            "BOOTSTRAP_PATH": str(path),
            "INSTALLED": installed,
        },
    )
    assert (result.returncode == 0) is succeeds, result.stderr
    observed = log.read_text(encoding="utf-8").strip() if log.exists() else ""
    assert expected_fragment in observed
    if installed == "579.0.0-0":
        assert "--reinstall" in observed
    if installed == "580.0.0-0":
        assert observed == ""
        assert "refusing to downgrade" in result.stderr


def test_live_controller_bootstrap_rejects_a_shadowed_locked_tool(tmp_path: Path) -> None:
    path = _monorepo_file(_SCRIPTS / "eyvan_live_controller_bootstrap.sh")
    exact = tmp_path / "exact"
    shadow = tmp_path / "shadow"
    exact.mkdir()
    shadow.mkdir()
    for directory in (exact, shadow):
        tool = directory / "uv"
        tool.write_text("#!/usr/bin/env bash\nprintf 'uv 0.9.21\n'\n", encoding="utf-8")
        tool.chmod(0o755)
    probe = r"""source "${BOOTSTRAP_PATH}"
verify_locked_command uv "${EXPECTED_UV}" 0.9.21
"""
    environment = {
        "BOOTSTRAP_PATH": str(path),
        "EXPECTED_UV": str(exact / "uv"),
    }
    rejected = subprocess.run(
        ["bash", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env={**environment, "PATH": f"{shadow}:{exact}:/usr/bin:/bin"},
    )
    assert rejected.returncode != 0
    assert "uv is shadowed" in rejected.stderr

    accepted = subprocess.run(
        ["bash", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env={**environment, "PATH": f"{exact}:{shadow}:/usr/bin:/bin"},
    )
    assert accepted.returncode == 0, accepted.stderr


def test_locked_uv_archive_accepts_the_pinned_release_member_shape(tmp_path: Path) -> None:
    path = _monorepo_file(_SCRIPTS / "eyvan_live_controller_bootstrap.sh")
    bundle = "uv-x86_64-unknown-linux-gnu"
    archive = tmp_path / "uv.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        directory = tarfile.TarInfo(f"{bundle}/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        handle.addfile(directory)
        for name in ("uv", "uvx"):
            payload = f"{name} pinned payload".encode()
            member = tarfile.TarInfo(f"{bundle}/{name}")
            member.size = len(payload)
            member.mode = 0o755
            handle.addfile(member, io.BytesIO(payload))

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "${BOOTSTRAP_PATH}"; validate_uv_archive_members "${ARCHIVE}" "${BUNDLE}"',
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            "ARCHIVE": str(archive),
            "BOOTSTRAP_PATH": str(path),
            "BUNDLE": bundle,
            "PATH": "/usr/bin:/bin",
        },
    )
    assert result.returncode == 0, result.stderr
