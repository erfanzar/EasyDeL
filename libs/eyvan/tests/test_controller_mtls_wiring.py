# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Controller wiring for the control-plane-to-agent mutual-TLS client."""

from unittest.mock import Mock, patch

import pytest

from eyvan.config import EyvanSettings
from eyvan.identity.mtls import MutualTlsConfigurationError, MutualTlsFiles
from eyvan.records import CellRecord
from eyvan.server.controller.lifecycle import _ControllerLifecycle


def _cell(endpoint: str = "https://agent.example") -> CellRecord:
    return CellRecord.model_construct(id="cell-1", endpoint=endpoint)


def _lifecycle(settings: EyvanSettings, *, agent_factory=None) -> _ControllerLifecycle:
    return _ControllerLifecycle(repository=Mock(), service=Mock(), settings=settings, agent_factory=agent_factory)


def test_default_agent_preserves_constructor_kwargs_when_mtls_is_disabled() -> None:
    lifecycle = _lifecycle(EyvanSettings(agent_shared_secret="secret"))
    with patch("eyvan.server.controller.lifecycle.CellAgentClient") as client:
        lifecycle._default_agent(_cell())

    client.assert_called_once_with("https://agent.example", token="secret")


def test_default_agent_passes_configured_mtls_files_when_enabled() -> None:
    files = MutualTlsFiles("ca.pem", "client.pem", "client.key")
    lifecycle = _lifecycle(
        EyvanSettings(
            agent_shared_secret="secret",
            mutual_tls_client_enabled=True,
            mutual_tls_trust_bundle=files.trust_bundle,
            mutual_tls_certificate=files.certificate,
            mutual_tls_private_key=files.private_key,
        )
    )
    with patch("eyvan.server.controller.lifecycle.CellAgentClient") as client:
        lifecycle._default_agent(_cell())

    client.assert_called_once_with("https://agent.example", token="secret", mtls=files)


def test_enabled_mtls_requires_a_complete_client_identity() -> None:
    lifecycle = _lifecycle(EyvanSettings(mutual_tls_client_enabled=True, mutual_tls_trust_bundle="ca.pem"))
    with patch("eyvan.server.controller.lifecycle.CellAgentClient") as client:
        with pytest.raises(MutualTlsConfigurationError, match="certificate and private key"):
            lifecycle._default_agent(_cell())

    client.assert_not_called()


def test_explicit_agent_factory_is_preserved() -> None:
    factory = Mock(return_value=Mock())
    lifecycle = _lifecycle(EyvanSettings(mutual_tls_client_enabled=True), agent_factory=factory)
    cell = _cell()

    assert lifecycle._agent_factory(cell) is factory.return_value
    factory.assert_called_once_with(cell)


__all__ = ()
