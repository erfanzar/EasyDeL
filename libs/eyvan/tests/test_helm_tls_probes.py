# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kubelet probes must speak the configured listener protocol."""

import pytest
from test_helm_chart import HELM, _render_chart, _resource


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
@pytest.mark.parametrize("tls_secret, scheme", [("eyvan-tls", "HTTPS"), ("", "HTTP")])
def test_health_probes_follow_the_listener_protocol(tls_secret, scheme) -> None:
    documents = _render_chart("--set-string", f"controlPlane.tls.existingSecret={tls_secret}")
    deployment = _resource(documents, "Deployment", "eyvan-control-plane")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    for probe in ("readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"].get("scheme", "HTTP") == scheme
