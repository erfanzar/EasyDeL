# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Passing a credential to a workload without persisting it in the spec.

The documented advice is that secrets are never persisted in specs. The only
mechanism for passing one was ``WorkloadDefinition.environment`` -- a persisted
``dict[str, str]`` validated for identifier names and NUL bytes -- so the advice
was contradicted by the only available means of following it.

Two pieces here. A reference side: values live in a store the operator already
runs, and the spec carries a reference resolved at dispatch by something holding
the agent's identity. And the gate that makes the reference safe.

**The gate is the security-critical piece.** Without it, any submitter in a
project can name any secret the *agent's* identity can read -- and that identity
is broad by construction, because it provisions capacity. It turns "may submit
jobs" into "may read every secret in the project", available to anyone who can
write a spec.
"""

from __future__ import annotations

import base64
import os
import subprocess

import pytest

from eyvan.records import (
    CommandExecution,
    PolicyRecord,
    Principal,
    Role,
    RoleGrant,
    WorkloadDefinition,
)
from eyvan.secrets import (
    FileSecretStore,
    GcpSecretManagerStore,
    SecretUnavailable,
    read_secret,
    secret_store_from_reference,
)
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError
from eyvan.server.service.scheduling import CoreSchedulingService

OPERATOR = Principal(
    subject="alice",
    grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
)
ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _service() -> ControlPlaneService:
    return ControlPlaneService(InMemoryRepository(), scheduler=CoreSchedulingService())


def _spec(**mounts: str) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="train",
        execution=CommandExecution(command="python train.py"),
        secret_mounts=dict(mounts),
    )


async def _policy(service: ControlPlaneService, *prefixes: str, project: str = "research") -> None:
    await service.put_policy(
        ADMIN,
        PolicyRecord(
            org_id="acme",
            project_id=project,
            allowed_secret_references=prefixes,
            updated_by="root",
        ),
        idempotency_key=f"policy-{project}",
    )


# --------------------------------------------------------------------------- the gate


async def test_an_unlisted_secret_reference_is_refused_at_submission() -> None:
    """The escalation this exists to prevent.

    The agent's identity can read every secret in the project because it
    provisions capacity. Without this check, naming one in a spec is how a
    submitter borrows that identity.
    """

    service = _service()
    await _policy(service, "gcp-secret://my-project/training-hf-token")

    with pytest.raises(AuthorizationError, match="not permitted in this project"):
        await service.submit_workload(
            OPERATOR,
            org_id="acme",
            project_id="research",
            spec=_spec(DB_PASSWORD="gcp-secret://my-project/production-database"),
            idempotency_key="w1",
        )


async def test_a_listed_segment_root_is_admitted() -> None:
    """The gate must not block the configuration it exists to permit."""

    service = _service()
    await _policy(service, "gcp-secret://my-project/training-hf-token")

    workload, _ = await service.submit_workload(
        OPERATOR,
        org_id="acme",
        project_id="research",
        spec=_spec(HF_TOKEN="gcp-secret://my-project/training-hf-token"),
        idempotency_key="w1",
    )

    assert workload.spec.secret_mounts == {"HF_TOKEN": "gcp-secret://my-project/training-hf-token"}


async def test_a_project_with_no_policy_may_mount_nothing() -> None:
    """The empty allow-list is "none", not "any".

    A permissive default would mean every project that never configured a policy
    silently grants every submitter the agent's read access.
    """

    service = _service()

    with pytest.raises(AuthorizationError):
        await service.submit_workload(
            OPERATOR,
            org_id="acme",
            project_id="research",
            spec=_spec(TOKEN="gcp-secret://my-project/anything"),
            idempotency_key="w1",
        )


async def test_a_workload_with_no_mounts_is_unaffected() -> None:
    """Every existing workload must keep submitting."""

    service = _service()

    workload, _ = await service.submit_workload(
        OPERATOR,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        idempotency_key="w1",
    )

    assert workload.spec.secret_mounts == {}


async def test_every_mount_is_checked_not_just_the_first() -> None:
    """A spec that mixes a permitted reference with a forbidden one is forbidden.

    The permitted name sorts *before* the forbidden one deliberately: with the
    forbidden mount first, an implementation that inspects only one entry still
    refuses, and the test would not distinguish the two. Checking one would make
    the escalation available to anyone who also names something legitimate.
    """

    service = _service()
    await _policy(service, "gcp-secret://my-project/training-hf-token")

    with pytest.raises(AuthorizationError, match="production-database"):
        await service.submit_workload(
            OPERATOR,
            org_id="acme",
            project_id="research",
            spec=_spec(
                HF_TOKEN="gcp-secret://my-project/training-hf-token",
                WANDB_KEY="gcp-secret://my-project/production-database",
            ),
            idempotency_key="w1",
        )


async def test_a_root_does_not_leak_across_projects() -> None:
    """One project's allow-list must not admit another's submissions.

    Both projects have a policy, so this distinguishes "read *this* project's
    allow-list" from "read *an* allow-list" -- which a project with no policy at
    all cannot.
    """

    service = _service()
    await _policy(service, "gcp-secret://my-project/training-hf-token")
    await _policy(service, "gcp-secret://my-project/platform-hf-token", project="platform")

    other = Principal(
        subject="mallory",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="platform"),),
    )

    with pytest.raises(AuthorizationError):
        await service.submit_workload(
            other,
            org_id="acme",
            project_id="platform",
            spec=_spec(HF_TOKEN="gcp-secret://my-project/training-hf-token"),
            idempotency_key="w1",
        )


@pytest.mark.parametrize(
    "name",
    [
        "BAD-NAME",
        "PATH",
        "JAX_PROCESS_ID",
        "EYVAN_SECRET_0_TOKEN",
        "EYVAN_CONTAINER_RUNTIME",
        "EYVAN_WORKLOAD_TOKEN",
        "EYVAN_CHILDREN_URL",
        "EYVAN_MANAGED_CALLABLE_URL",
        "EYVAN_MANAGED_CALLABLE_TOKEN",
        "EYVAN_MANAGED_WORKSPACE_URL",
        "EYVAN_MANAGED_WORKSPACE_TOKEN",
        "EYVAN_MANAGED_RESULT_URL",
        "EYVAN_MANAGED_RESULT_TOKEN",
        "EYVAN_PROGRESS_URL",
        "EYVAN_CHECKPOINT_RANK",
        "EYVAN_CHECKPOINT_URI",
        "EYVAN_CHECKPOINT_BASE_URI",
        "EYVAN_PROFILE_URI",
        "EYVAN_SOURCE_JOB_ID",
        "EYVAN_SOURCE_CONTROL_GROUP",
        "EYVAN_CALL_ARGS_JSON",
        "EYVAN_CALL_KWARGS_JSON",
        "EYVAN_RESOURCE_BUNDLES",
        "EYVAN_PLACEMENT_STRATEGY",
        "EYVAN_FUTURE_RUNTIME_FIELD",
    ],
)
def test_secret_mount_names_follow_the_workload_environment_contract(name: str) -> None:
    with pytest.raises(ValueError):
        _spec(**{name: "gcp-secret://my-project/training-hf-token"})


def test_secret_mount_cannot_collide_with_an_ordinary_environment_value() -> None:
    with pytest.raises(ValueError, match="collides"):
        WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            environment={"HF_TOKEN": "ordinary"},
            secret_mounts={"HF_TOKEN": "gcp-secret://my-project/training-hf-token"},
        )


def test_pool_api_schema_rejects_secret_mount_name_and_line_injection() -> None:
    from pydantic import ValidationError

    from eyvan.server.api.schemas import PoolWorkloadSubmitRequest

    with pytest.raises(ValidationError):
        PoolWorkloadSubmitRequest.model_validate(
            {
                "org_id": "acme",
                "project_id": "research",
                "command": "python train.py",
                "secret_mounts": {"PATH": "gcp-secret://my-project/training-hf-token"},
            }
        )
    with pytest.raises(ValidationError):
        PoolWorkloadSubmitRequest.model_validate(
            {
                "org_id": "acme",
                "project_id": "research",
                "command": "python train.py",
                "secret_mounts": {"HF_TOKEN": "gcp-secret://my-project/training-hf-token\nINJECTED=1"},
            }
        )


# --------------------------------------------------------------------------- the store


def test_a_scheme_selects_its_backend() -> None:
    """Adding a backend is adding a scheme, not editing a dispatch table."""

    assert isinstance(secret_store_from_reference("gcp-secret://p/name"), GcpSecretManagerStore)
    assert isinstance(secret_store_from_reference("file:///run/secrets/token"), FileSecretStore)


def test_an_unknown_scheme_is_refused_rather_than_read_as_a_path() -> None:
    """A typo in ``gcp-secret`` would otherwise become a missing-file error.

    That error names the wrong problem, and an operator would go looking for a
    file nobody meant to create.
    """

    with pytest.raises(SecretUnavailable, match="unknown scheme"):
        secret_store_from_reference("gcpsecret://p/name")


def test_a_file_secret_is_read(tmp_path) -> None:
    """The local-development and projected-volume case share one reader."""

    path = tmp_path / "token"
    path.write_bytes(b"s3cret")

    assert read_secret(f"file://{path}", allowed_roots=(f"file://{tmp_path}",)) == b"s3cret"


@pytest.mark.parametrize(
    "spelling",
    ["short", "extra-slash", "upper-scheme", "empty-query", "trailing-slash", "root-slash"],
)
def test_file_uri_spelling_is_canonical_at_policy_and_value_boundaries(tmp_path, spelling: str) -> None:
    token = tmp_path / "token"
    if spelling == "short":
        reference = f"file:{token}"
    elif spelling == "extra-slash":
        reference = f"file:///{token}"
    elif spelling == "upper-scheme":
        reference = f"FILE://{token}"
    elif spelling == "empty-query":
        reference = f"file://{token}?"
    elif spelling == "trailing-slash":
        reference = f"file://{token}/"
    else:
        reference = "file:///"

    with pytest.raises(ValueError, match="canonical file"):
        _spec(HF_TOKEN=reference)
    with pytest.raises(ValueError, match="canonical file"):
        PolicyRecord(
            org_id="acme",
            project_id="research",
            allowed_secret_references=(reference,),
            updated_by="root",
        )


def test_an_unreadable_file_returns_only_a_stable_category(tmp_path) -> None:
    reference = f"file://{tmp_path / 'sensitive-absent-name'}"

    with pytest.raises(SecretUnavailable, match="could not be read safely") as captured:
        read_secret(reference, allowed_roots=(f"file://{tmp_path}",))

    assert reference not in str(captured.value)
    assert captured.value.__cause__ is None


def test_a_gcp_reference_is_resolved_through_gcloud() -> None:
    """The agent already authenticates gcloud.

    A second credential path would be a second thing to get wrong on the host
    whose identity is the entire security boundary.
    """

    seen: list[tuple[str, ...]] = []

    value = b"\x00binary\xff\nvalue"

    def runner(command):
        seen.append(tuple(command))
        wire = base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii") + "\n"
        return subprocess.CompletedProcess(command, 0, wire, "")

    assert GcpSecretManagerStore(runner=runner).read("gcp-secret://my-project/hf-token") == value
    joined = " ".join(seen[0])
    assert "--secret=hf-token" in joined
    assert "--project=my-project" in joined
    assert "access latest" in joined


@pytest.mark.parametrize(
    ("value", "wire"),
    [
        (b"", ""),
        (b"f", "Zg"),
        (b"f", "Zg=="),
        (b"fo", "Zm8"),
        (b"fo", "Zm8="),
        (b"foo", "Zm9v"),
        (b"\xfb\xff", "-_8"),
        (b"\xfb\xff", "-_8="),
    ],
)
def test_gcloud_accepts_unpadded_or_exactly_padded_canonical_base64url(value: bytes, wire: str) -> None:
    store = GcpSecretManagerStore(runner=lambda command: subprocess.CompletedProcess(command, 0, f"{wire}\n", ""))

    assert store.read("gcp-secret://my-project/hf-token") == value


@pytest.mark.parametrize(
    "wire",
    [
        "Zg=",  # partial: one byte requires two padding characters
        "Zg===",  # excess
        "Zm8==",  # excess: two bytes require one
        "Zm9v=",  # needless: three bytes require none
        "=",  # padding without data
        "Z=g=",  # misplaced
        "Zh==",  # non-zero discarded bits, not a canonical encoding of b'f'
    ],
)
def test_gcloud_rejects_noncanonical_base64url_padding(wire: str) -> None:
    store = GcpSecretManagerStore(runner=lambda command: subprocess.CompletedProcess(command, 0, f"{wire}\n", ""))

    with pytest.raises(SecretUnavailable):
        store.read("gcp-secret://my-project/hf-token")


def test_a_pinned_version_is_honored() -> None:
    """Pinning is how a rotation is rolled out deliberately rather than at once."""

    seen: list[tuple[str, ...]] = []

    def runner(command):
        seen.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "dmFsdWU=\n", "")

    GcpSecretManagerStore(runner=runner).read("gcp-secret://my-project/hf-token/7")

    assert "access 7" in " ".join(seen[0])


def test_a_denied_read_does_not_reflect_provider_output() -> None:
    marker = "provider-reflected-secret-value"

    def runner(command):
        return subprocess.CompletedProcess(command, 1, marker, f"PERMISSION_DENIED: {marker}")

    with pytest.raises(SecretUnavailable, match="secret provider could not resolve") as captured:
        GcpSecretManagerStore(runner=runner).read("gcp-secret://my-project/hf-token")

    assert marker not in str(captured.value)
    assert captured.value.__cause__ is None


def test_a_malformed_gcp_reference_is_refused() -> None:
    """A reference missing its project would resolve against the agent's default.

    That is a different secret in a different project, read successfully, with
    nothing to indicate the mistake.
    """

    with pytest.raises(SecretUnavailable, match="secret provider could not resolve"):
        GcpSecretManagerStore(runner=lambda command: subprocess.CompletedProcess(command, 0, "", "")).read(
            "gcp-secret:///name"
        )


def test_gcloud_malformed_wire_output_is_refused_without_reflection() -> None:
    marker = "wire-secret-marker"
    malformed = (
        f"{marker}!\n",
        "dmFs\ndWU=\n",
        "A\n",
        "dmFsdWU=\n\n",
        "dmFsdWU===\n",
    )
    for output in malformed:
        store = GcpSecretManagerStore(
            runner=lambda command, output=output: subprocess.CompletedProcess(command, 0, output, "")
        )
        with pytest.raises(SecretUnavailable) as captured:
            store.read("gcp-secret://my-project/hf-token")
        assert marker not in str(captured.value)
        assert captured.value.__cause__ is None


def test_gcloud_runner_exceptions_are_normalized() -> None:
    marker = "subprocess-reflected-secret"

    def runner(_command):
        raise RuntimeError(marker)

    with pytest.raises(SecretUnavailable, match="secret provider could not resolve") as captured:
        GcpSecretManagerStore(runner=runner).read("gcp-secret://my-project/hf-token")

    assert marker not in str(captured.value)
    assert captured.value.__cause__ is None


async def test_raw_prefixes_do_not_cross_a_path_segment_boundary() -> None:
    service = _service()
    await _policy(service, "gcp-secret://my-project/training")

    with pytest.raises(AuthorizationError):
        await service.submit_workload(
            OPERATOR,
            org_id="acme",
            project_id="research",
            spec=_spec(HF_TOKEN="gcp-secret://my-project/training-prod"),
            idempotency_key="segment-boundary",
        )


def test_file_store_rejects_dot_encoded_nul_and_authority_escapes(tmp_path) -> None:
    root = f"file://{tmp_path}"
    references = (
        f"file://{tmp_path}/../outside",
        f"file://{tmp_path}/%2e%2e/outside",
        f"file://{tmp_path}/%2Foutside",
        f"file://{tmp_path}/token\x00suffix",
        f"file://attacker{tmp_path}/token",
    )
    store = FileSecretStore(allowed_roots=(root,))

    for reference in references:
        with pytest.raises(SecretUnavailable):
            store.read(reference)


def test_file_store_rejects_inside_outside_and_root_symlinks(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "token").write_bytes(b"outside")
    (root / "inside-link").symlink_to(outside / "token")
    (root / "outside-directory").symlink_to(outside, target_is_directory=True)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(root, target_is_directory=True)

    store = FileSecretStore(allowed_roots=(f"file://{root}",))
    with pytest.raises(SecretUnavailable):
        store.read(f"file://{root / 'inside-link'}")
    with pytest.raises(SecretUnavailable):
        store.read(f"file://{root / 'outside-directory' / 'token'}")
    with pytest.raises(SecretUnavailable):
        FileSecretStore(allowed_roots=(f"file://{root_link}",)).read(f"file://{root_link / 'token'}")


def test_file_store_reads_from_the_open_descriptor_across_a_swap(tmp_path, monkeypatch) -> None:
    """Replacing the pathname after open cannot replace the bytes being read."""

    root = tmp_path / "root"
    root.mkdir()
    token = root / "token"
    displaced = root / "opened-token"
    token.write_bytes(b"authorized-value")
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "token" and dir_fd is not None and not swapped:
            swapped = True
            token.rename(displaced)
            token.write_bytes(b"attacker-replacement")
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)

    assert FileSecretStore(allowed_roots=(f"file://{root}",)).read(f"file://{token}") == b"authorized-value"
    assert token.read_bytes() == b"attacker-replacement"


# --------------------------------------------------------------------------- the surfaces


def _cli(monkeypatch, *args):
    """Run one CLI command against the recording client the other CLI tests use."""

    from click.testing import CliRunner
    from test_cli import FakeClient, _json_argument

    from eyvan.sdk import cli

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    result = CliRunner().invoke(
        cli.main,
        ["--user", "alice", "--org", "acme", "--project", "research", *args],
    )
    payloads = [_json_argument(call) for call in fake.calls]
    return result, payloads


def test_a_submitter_can_declare_a_mount_from_the_cli(monkeypatch) -> None:
    """A spec field with no way to set it is a feature nobody has."""

    result, payloads = _cli(
        monkeypatch,
        "workload",
        "submit",
        "--name",
        "train",
        "--command",
        "python train.py",
        "--secret",
        "HF_TOKEN=gcp-secret://my-project/training-hf-token",
    )

    assert result.exit_code == 0, result.output
    assert payloads[-1]["spec"]["secret_mounts"] == {"HF_TOKEN": "gcp-secret://my-project/training-hf-token"}


def test_a_malformed_secret_option_names_its_own_flag(monkeypatch) -> None:
    """An error naming --env sends a user to the wrong line of their command."""

    result, _payloads = _cli(monkeypatch, "workload", "submit", "--name", "t", "--command", "c", "--secret", "HF_TOKEN")

    assert result.exit_code != 0
    assert "--secret" in result.output


def test_an_operator_can_open_the_allow_list_from_the_cli(monkeypatch) -> None:
    """The gate is unusable if the only way to configure it is a database write."""

    from test_governance_cli import FakeClient, _invoke, _json_argument

    fake = FakeClient()

    result = _invoke(
        fake, monkeypatch, ["policy", "set", "--allow-secret-reference", "gcp-secret://my-project/training-hf-token"]
    )

    assert result.exit_code == 0, result.output
    assert _json_argument(fake.calls[0])["allowed_secret_references"] == ["gcp-secret://my-project/training-hf-token"]


def test_an_omitted_secret_allow_list_forbids_rather_than_permits(monkeypatch) -> None:
    """The inversion, asserted rather than only documented.

    Its neighbours -- regions, providers, capacity classes -- all mean "no
    restriction" when empty, and one of them has a test saying so. This one has
    to mean the opposite, and the two conventions living in one command is
    exactly the kind of thing that gets "fixed" into a security hole.
    """

    from test_governance_cli import FakeClient, _invoke, _json_argument

    fake = FakeClient()

    assert _invoke(fake, monkeypatch, ["policy", "set"]).exit_code == 0
    payload = _json_argument(fake.calls[0])

    assert payload["allowed_secret_references"] == []
    assert payload["allowed_regions"] == []


def test_the_inverted_default_is_documented_where_an_operator_reads_it() -> None:
    """Every other allow-list here means "no restriction" when empty; this one does not.

    An operator who assumes the usual convention would read a populated list as
    narrowing an otherwise-open dimension, when it is in fact the only thing
    opening a closed one.
    """

    from click.testing import CliRunner

    from eyvan.sdk import cli

    output = CliRunner().invoke(cli.main, ["policy", "set", "--help"]).output

    assert "--allow-secret-reference" in output
    assert "NOTHING" in output
