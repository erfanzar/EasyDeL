# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A head must be able to accept the fleet it was created for.

Every capacity pool needs an agent endpoint. A loopback development plane
supplies one automatically, so a fleet that applied on a laptop was refused on a
real head with "no default cell agent is configured by the Eyvan platform" --
after the head had been provisioned, authenticated and migrated.
"""

from __future__ import annotations

from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.sdk.commands.head import (
    HEAD_AGENT_PORT,
    HEAD_AGENT_SERVICE_NAME,
    PLATFORM_CONFIG_PATH,
    HeadSpec,
    render_head_bootstrap_script,
)

_FLEET = {"fleet": "demo-fleet", "project": "my-demo-project", "region": "us-central1", "zone": "us-central1-b"}


def test_the_head_installs_and_starts_a_cell_agent() -> None:
    script = render_head_bootstrap_script(HeadSpec(**_FLEET))

    assert HEAD_AGENT_SERVICE_NAME in script
    # Enable *and* restart: `enable --now` does nothing to an already-active
    # unit, so a re-run would leave the agent on the wheel and environment it
    # started with while reporting success.
    assert f"systemctl enable {HEAD_AGENT_SERVICE_NAME}" in script
    assert f"systemctl restart {HEAD_AGENT_SERVICE_NAME}" in script
    assert f"--port {HEAD_AGENT_PORT}" in script


def test_the_platform_default_is_written_and_parses() -> None:
    """A file the control plane cannot parse is the same as no file at all."""

    import yaml

    script = render_head_bootstrap_script(HeadSpec(**_FLEET))

    assert PLATFORM_CONFIG_PATH in script
    body = script.split("<<'EYVAN_PLATFORM'\n", 1)[1].split("\nEYVAN_PLATFORM", 1)[0]
    config = EyvanPlatformConfig.from_document(yaml.safe_load(body))
    assert config.default_agent_endpoint == f"http://127.0.0.1:{HEAD_AGENT_PORT}"


def test_the_agent_is_not_exposed_beyond_the_head() -> None:
    """The control plane reaches it over loopback; nothing else should."""

    script = render_head_bootstrap_script(HeadSpec(**_FLEET))
    body = script.split("<<'EYVAN_PLATFORM'\n", 1)[1].split("\nEYVAN_PLATFORM", 1)[0]

    assert "127.0.0.1" in body
    assert "0.0.0.0" not in body
