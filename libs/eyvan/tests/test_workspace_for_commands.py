# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A workspace belongs to the workload, not to one execution kind.

The three workspace fields first appeared on ``CallableExecution``, because
callables needed them first. That made a modelling accident look like a design:
``CommandExecution`` carries only a command string, so ``eyvan run`` structurally
could not ship a working tree, and putting your own code on an accelerator meant
baking a container image or using the Python decorator.

These tests pin the corrected shape -- ``WorkloadDefinition.workspace``, honoured
for every execution kind -- and the compatibility bridge that keeps specs written
against the old shape working, because a submission is a durable record.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CallableExecution,
    CommandExecution,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkspaceRef,
)

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64


def _spec(**overrides) -> WorkloadDefinition:
    base = {
        "name": "train",
        "execution": CommandExecution(command="python train.py"),
        "gang": ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
    }
    return WorkloadDefinition(**{**base, **overrides})


def test_a_command_workload_can_carry_a_workspace() -> None:
    """The whole point: `eyvan run` ships your code without an image build."""

    spec = _spec(workspace=WorkspaceRef(uri="eyvan://workspaces/" + _DIGEST, sha256=_DIGEST))

    resolved = spec.resolved_workspace()

    assert resolved is not None
    assert resolved.sha256 == _DIGEST


def test_a_command_workload_without_one_resolves_to_nothing() -> None:
    """Shipping a tree must stay opt-in, never a surprise on every submission."""

    assert _spec().resolved_workspace() is None


def test_a_callable_spec_written_against_the_old_shape_still_resolves() -> None:
    """A submission is a durable record; an older client must keep working."""

    spec = _spec(
        execution=CallableExecution(
            artifact_uri="eyvan://callables/" + _DIGEST,
            sha256=_DIGEST,
            python="3.13",
            workspace_uri="eyvan://workspaces/" + _OTHER_DIGEST,
            workspace_sha256=_OTHER_DIGEST,
        )
    )

    resolved = spec.resolved_workspace()

    assert resolved is not None
    assert resolved.uri == "eyvan://workspaces/" + _OTHER_DIGEST
    assert resolved.sha256 == _OTHER_DIGEST


def test_the_workload_level_workspace_wins_over_the_legacy_one() -> None:
    """A newer client fills the spec-level field deliberately; honour that."""

    spec = _spec(
        execution=CallableExecution(
            artifact_uri="eyvan://callables/" + _DIGEST,
            sha256=_DIGEST,
            python="3.13",
            workspace_uri="eyvan://workspaces/" + _OTHER_DIGEST,
            workspace_sha256=_OTHER_DIGEST,
        ),
        workspace=WorkspaceRef(uri="eyvan://workspaces/" + _DIGEST, sha256=_DIGEST),
    )

    assert spec.resolved_workspace().sha256 == _DIGEST


def test_a_reference_without_a_verifiable_digest_is_refused() -> None:
    """A bundle that cannot be verified must never reach a leased host."""

    with pytest.raises(ValueError):
        WorkspaceRef(uri="eyvan://workspaces/x", sha256="not-a-digest")


def test_an_empty_uri_is_refused() -> None:
    with pytest.raises(ValueError):
        WorkspaceRef(uri="   ", sha256=_DIGEST)
