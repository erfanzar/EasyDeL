# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The standalone repository ships its own CI, and it must not drift.

`.github/` lives at the monorepo root and is not part of the `libs/eyvan`
subtree, so the published repository had no CI at all: every check that guarded
this package ran only in a repository its users never see. These workflows live
inside the package so the subtree split carries them.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
import yaml

_EYVAN = pathlib.Path(__file__).resolve().parents[1]
_WORKFLOWS = _EYVAN / ".github" / "workflows"


def _workflow(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["ci.yaml", "image.yaml", "publish.yaml"])
def test_every_action_is_pinned_to_a_full_commit(name: str) -> None:
    """A moving action tag is an unreviewed dependency with write access."""

    text = (_WORKFLOWS / name).read_text(encoding="utf-8")
    uses = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", text, flags=re.MULTILINE)

    assert uses
    for reference in uses:
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference), reference


@pytest.mark.parametrize("name", ["ci.yaml", "image.yaml", "publish.yaml"])
def test_checkout_actions_match_the_release_lock(name: str) -> None:
    lock = json.loads((_EYVAN / "oci" / "release-lock.json").read_text(encoding="utf-8"))
    text = (_WORKFLOWS / name).read_text(encoding="utf-8")
    pins = re.findall(r"uses:\s*actions/checkout@([^\s#]+)", text)
    assert pins
    assert set(pins) == {lock["actions"]["checkout"]}


def test_the_image_workflow_reads_its_bases_from_the_release_lock() -> None:
    """A base image must only change where the policy verifier looks.

    Writing the digests into the workflow puts them somewhere `validate-policy`
    does not read, so they could drift from the lock with nothing objecting.
    """

    text = (_WORKFLOWS / "image.yaml").read_text(encoding="utf-8")

    assert "oci/release-lock.json" in text
    lock = json.loads((_EYVAN / "oci" / "release-lock.json").read_text(encoding="utf-8"))
    for role in ("control-plane", "agent", "worker"):
        assert role in text, f"the image matrix does not build {role}"
        for value in lock["images"][role]["buildArgs"].values():
            assert value not in text, "a base image digest is hard-coded instead of read from the lock"


def test_no_workflow_publishes_a_moving_tag() -> None:
    """`latest` makes "which image is running" unanswerable after the fact."""

    text = (_WORKFLOWS / "image.yaml").read_text(encoding="utf-8")

    assert ":latest" not in text
    assert "sha-${{ github.sha }}" in text


def test_publishing_carries_no_long_lived_credential() -> None:
    """Trusted publishing means a compromised workflow has nothing to steal."""

    text = (_WORKFLOWS / "publish.yaml").read_text(encoding="utf-8")
    publish = _workflow("publish.yaml")["jobs"]["publish"]

    assert "--trusted-publishing always" in text
    assert publish["permissions"]["id-token"] == "write"
    assert "PYPI" not in text and "password" not in text


def test_the_suite_runs_against_the_production_backend() -> None:
    """A green suite on SQLite says nothing about SERIALIZABLE or asyncpg."""

    python = _workflow("ci.yaml")["jobs"]["python"]

    assert re.fullmatch(r"postgres:17-alpine@sha256:[0-9a-f]{64}", python["services"]["postgres"]["image"])
    steps = json.dumps(python["steps"])
    assert "EYVAN_REQUIRE_POSTGRES" in steps


def test_every_install_is_hash_pinned() -> None:
    """The standalone repository has no lock; the hashed exports are the lock."""

    text = (_WORKFLOWS / "ci.yaml").read_text(encoding="utf-8")

    installs = [line for line in text.splitlines() if "uv pip install" in line and "-r " in line]
    assert installs
    for line in installs:
        assert "--require-hashes" in line, line


@pytest.mark.parametrize("name", ["requirements-docker.txt", "requirements-dev.txt", "requirements-build.txt"])
def test_the_exports_the_workflows_install_from_are_shipped(name: str) -> None:
    """A workflow that installs a file the subtree does not carry cannot run."""

    export = _EYVAN / name

    assert export.is_file(), f"{name} is missing"
    lines = [line for line in export.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert any("--hash=sha256:" in line for line in lines), f"{name} carries no hashes"


def test_the_full_suite_is_scheduled_every_night_without_a_subset_filter() -> None:
    workflow = _workflow("ci.yaml")
    # PyYAML's YAML 1.1 loader interprets the unquoted `on` key as True.
    events = workflow.get("on", workflow.get(True))
    assert any(item["cron"].split()[2:] == ["*", "*", "*"] for item in events["schedule"])
    python = workflow["jobs"]["python"]
    assert "if" not in python
    full_suite = next(step for step in python["steps"] if step.get("name") == "Suite, against the production backend")
    assert full_suite["run"] == ".venv/bin/python -m pytest -q -rs tests"
    assert "if" not in full_suite


def test_python_publishing_excludes_the_sbom_from_upload_arguments() -> None:
    steps = _workflow("publish.yaml")["jobs"]["publish"]["steps"]
    publish = next(step for step in steps if step.get("name") == "Publish")
    assert publish["run"] == "uv publish --trusted-publishing always dist/*.whl dist/*.tar.gz"


def test_release_provenance_covers_packages_and_sbom_before_publication() -> None:
    job = _workflow("publish.yaml")["jobs"]["publish"]
    assert job["permissions"].get("attestations") == "write"
    steps = job["steps"]
    attestation = next(step for step in steps if step.get("uses", "").startswith("actions/attest-build-provenance@"))
    assert set(attestation["with"]["subject-path"].splitlines()) == {
        "dist/*.whl",
        "dist/*.tar.gz",
        "dist/sbom-cyclonedx.json",
    }
    assert steps.index(attestation) < next(index for index, step in enumerate(steps) if step.get("name") == "Publish")


def test_dispatch_version_is_data_not_interpolated_shell_source() -> None:
    steps = _workflow("publish.yaml")["jobs"]["publish"]["steps"]
    agreement = next(step for step in steps if step.get("name") == "The tag and the source must agree")
    assert "${{ github.event.inputs.version }}" not in agreement["run"]
    assert agreement["env"]["REQUESTED_VERSION"] == "${{ github.event.inputs.version }}"
    assert 'CLAIMED="$REQUESTED_VERSION"' in agreement["run"]


def test_nightly_repository_soak_requires_postgres_and_an_extended_run() -> None:
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    soak = next((step for step in steps if step.get("name") == "Nightly repository soak"), None)
    assert soak is not None
    assert "github.event_name == 'schedule'" in soak["if"]
    assert soak["env"]["EYVAN_REQUIRE_POSTGRES"] == "1"
    assert "test_repository_soak.py" in soak["run"]
    assert "--soak-iterations=1000" in soak["run"]


def test_quarantine_diagnostics_do_not_mask_the_full_suite() -> None:
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    step = next((step for step in steps if step.get("name") == "Nightly quarantine diagnostics"), None)
    assert step is not None
    assert "github.event_name == 'schedule'" in step["if"]
    assert "!cancelled()" in step["if"]
    assert "-m quarantine" in step["run"]
    assert "--strict-markers" in step["run"]
    assert "continue-on-error" not in step


def test_precommit_ruff_is_installed_by_the_documented_dev_setup() -> None:
    pins = set(re.findall(r"--from (ruff==[0-9.]+)", (_WORKFLOWS / "ci.yaml").read_text()))
    assert len(pins) == 1
    pin = pins.pop()
    assert f'"{pin}"' in (_EYVAN / "pyproject.toml").read_text()
    assert pin in (_EYVAN / "requirements-dev.txt").read_text()


def test_ci_enforces_import_boundaries_with_the_locked_dev_tool() -> None:
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    assert any(step.get("run") == ".venv/bin/lint-imports" for step in steps)
    assert "import-linter==" in (_EYVAN / "requirements-dev.txt").read_text()


def test_ci_checks_scheduler_budgets_and_retains_nightly_ten_thousand_cell_samples() -> None:
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    regular = next((step for step in steps if step.get("name") == "Scheduler regression budget"), None)
    nightly = next((step for step in steps if step.get("name") == "Nightly 10k-cell simulation"), None)
    assert regular is not None and nightly is not None
    assert "--budget-file scripts/scheduler-budgets.json" in regular["run"]
    assert "--cells 10000" in nightly["run"]
    assert "--iterations 100" in nightly["run"]
    assert "github.event_name == 'schedule'" in nightly["if"]
    assert any(step.get("with", {}).get("path") == "benchmark-results/*.json" for step in steps)


def test_nightly_seeded_fuzz_is_reproducible_and_blocking() -> None:
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    fuzz = next(step for step in steps if step.get("name") == "Nightly seeded invariant fuzz")
    assert "github.event_name == 'schedule'" in fuzz["if"]
    assert fuzz["env"]["EYVAN_REQUIRE_POSTGRES"] == "1"
    assert fuzz["env"]["FUZZ_SEED"] == "${{ github.run_number }}"
    assert '--hypothesis-seed="$FUZZ_SEED"' in fuzz["run"]
    for name in (
        "test_scheduler_properties.py",
        "test_scheduler_budget_properties.py",
        "test_chaos_on_postgres.py",
        "test_settlement_properties.py",
        "test_orphan_settlement_properties.py",
        "test_unstarted_retry_properties.py",
        "test_running_retry_properties.py",
        "test_preemption_settlement_properties.py",
        "test_controller_retry_properties.py",
        "test_dispatch_delivery_properties.py",
        "test_priority_preemption_properties.py",
        "test_preemption_phase_recovery.py",
        "test_dispatch_response_loss.py",
        "test_dispatch_claim_takeover.py",
    ):
        assert name in fuzz["run"]
    assert "continue-on-error" not in fuzz
    receipt = next(step for step in steps if step.get("name") == "Retain invariant fuzz receipts")
    assert "!cancelled()" in receipt["if"]
    assert receipt["with"]["path"] == "fuzz-results/"


def test_ci_refuses_stale_generated_api_docs() -> None:
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    assert any(step.get("run") == ".venv/bin/python scripts/generate_api_reference.py --check" for step in steps)


def test_python_support_starts_at_311_and_the_ci_matrix_matches() -> None:
    import tomllib

    from packaging.specifiers import SpecifierSet

    project = tomllib.loads((_EYVAN / "pyproject.toml").read_text())
    supported = SpecifierSet(project["project"]["requires-python"])
    assert "3.10" not in supported
    assert all(version in supported for version in ("3.11", "3.12", "3.13"))
    assert "3.14" not in supported
    assert project["tool"]["ruff"]["target-version"] == "py311"
    assert _workflow("ci.yaml")["jobs"]["python"]["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
    ]
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    setup = next(step for step in steps if step.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert setup["with"].get("python-version") == "${{ matrix.python-version }}"


def test_python_dependency_audit_uses_the_complete_hashed_export_without_pip_resolution() -> None:
    steps = _workflow("ci.yaml")["jobs"]["python"]["steps"]
    audit = next((step for step in steps if step.get("name") == "Python dependency audit"), None)
    assert audit is not None
    assert "--disable-pip" in audit["run"] and "--require-hashes" in audit["run"]
    assert "--requirement requirements-docker.txt" in audit["run"]
    assert "continue-on-error" not in audit
    assert "pip-audit==2.10.1" in (_EYVAN / "requirements-dev.txt").read_text()


def test_ci_fetches_the_immutable_wheel_upgrade_baseline() -> None:
    checkout = _workflow("ci.yaml")["jobs"]["python"]["steps"][0]
    assert checkout.get("with", {}).get("fetch-depth") == 0
    assert checkout["with"]["persist-credentials"] is False
