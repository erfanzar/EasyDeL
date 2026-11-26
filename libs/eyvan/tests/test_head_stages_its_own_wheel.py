# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A head must not need an index to install Eyvan.

The default package was the name ``eyvan``, which pip resolves from an index.
Eyvan is published to none, so a real GCE bootstrap died with "No matching
distribution found for eyvan" after the VM, firewall, secrets and PostgreSQL had
all already been created. The distribution running the command is the one the
head should run, so it is sent rather than looked up.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from eyvan.sdk.commands.head import (
    HEAD_WHEEL_DIR,
    STAGED_WHEEL_PACKAGE,
    HeadSpec,
    HeadStep,
    render_head_bootstrap_script,
)

_FLEET = {"fleet": "demo-fleet", "project": "my-demo-project", "region": "us-central1", "zone": "us-central1-b"}


def test_a_head_installs_a_staged_wheel_by_default() -> None:
    """No index, no --package, no name for pip to fail to resolve."""

    spec = HeadSpec(**_FLEET)

    assert spec.stages_wheel
    installs = [line for line in render_head_bootstrap_script(spec).splitlines() if "pip install" in line]
    assert installs and all(HEAD_WHEEL_DIR in line for line in installs)
    assert not any(line.rstrip().endswith(" eyvan") for line in installs)
    assert all("/wheels/" in line for line in installs), "Eyvan comes from the staged file, never from an index"
    assert all(line.rstrip().endswith(".whl") for line in installs), "pip refuses a non-PEP-427 wheel filename"


def test_an_explicit_package_still_wins_and_skips_staging() -> None:
    """A pinned version or an already-present path must keep working."""

    spec = HeadSpec(**_FLEET, package="eyvan==0.0.1")

    assert not spec.stages_wheel
    assert any(
        "eyvan==0.0.1" in line for line in render_head_bootstrap_script(spec).splitlines() if "pip install" in line
    )


def test_the_staged_wheel_is_owner_only_and_never_in_argv() -> None:
    """``ps`` shows argv to every user on the host; a wheel is delivered on stdin."""

    from eyvan.sdk.commands.head import HeadProvisioner

    action = HeadProvisioner(spec=HeadSpec(**_FLEET), runner=_forbidden)._stage_wheel_action()

    assert action.sends_stdin
    assert action.step is HeadStep.STAGE_WHEEL
    joined = " ".join(action.args)
    assert "umask 077" in joined
    assert "base64 -d" in joined
    assert "sha256sum" in joined, "the host must report what it wrote so the sender can verify it"


def test_a_truncated_upload_is_refused_instead_of_installed() -> None:
    """Installing a wheel whose bytes were altered in flight is worse than failing."""

    from eyvan.sdk.commands.head import HeadCommandError, HeadProvisioner

    provisioner = HeadProvisioner(spec=HeadSpec(**_FLEET), runner=_forbidden)
    payload = b"not really a wheel"
    sent = hashlib.sha256(payload).hexdigest()
    observed = hashlib.sha256(payload[:-1]).hexdigest()

    assert sent != observed
    with pytest.raises(HeadCommandError) as caught:
        provisioner._verify_staged_wheel(sent=sent, observed=observed)
    assert observed in str(caught.value) and sent in str(caught.value)


def test_the_payload_round_trips_through_the_text_boundary() -> None:
    """The stdin seam carries text; a wheel is binary and must survive base64."""

    payload = bytes(range(256)) * 8

    assert base64.b64decode(base64.b64encode(payload).decode("ascii")) == payload


def _forbidden(*args, **kwargs):  # pragma: no cover - a real call is the failure
    raise AssertionError("tests must never reach the real command boundary")


def test_the_staging_step_is_planned_only_when_it_stages() -> None:
    """--dry-run must show the step, and must not show it when it will not run."""

    from eyvan.sdk.commands.head import HeadProvisioner

    staged = HeadProvisioner(spec=HeadSpec(**_FLEET), runner=_forbidden).plan()
    named = HeadProvisioner(spec=HeadSpec(**_FLEET, package="eyvan==0.0.1"), runner=_forbidden).plan()

    assert any(action.step is HeadStep.STAGE_WHEEL for action in staged.actions)
    assert not any(action.step is HeadStep.STAGE_WHEEL for action in named.actions)
    assert STAGED_WHEEL_PACKAGE not in " ".join(argument for action in named.actions for argument in action.args)
