# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verified ServiceMonitor trust contracts; real rendering requires Helm."""

from __future__ import annotations

import subprocess

import pytest
import yaml
from test_helm_chart import CHART, HELM, _render_chart, _resource

TRUST = {
    "caSecretName": "eyvan-metrics-ca",
    "caSecretKey": "ca.crt",
    "serverName": "eyvan-control-plane.release-ns.svc.cluster.local",
}


def test_metrics_trust_defaults_are_absent_and_tls_references_are_required() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["serviceMonitor"].get("tls") == dict.fromkeys(TRUST, "")
    policies = (CHART / "templates" / "policies.yaml").read_text(encoding="utf-8")
    monitor = policies.split("kind: ServiceMonitor", 1)[1]
    assert "{{- if .Values.controlPlane.tls.existingSecret }}" in monitor
    assert "      tlsConfig:\n        ca:\n          secret:\n" in monitor
    for name in TRUST:
        assert f"serviceMonitor.tls.{name} is required when the monitored listener uses TLS" in monitor
        assert f"(.Values.serviceMonitor.tls.{name} | trim)" in monitor
    assert "insecureSkipVerify" not in monitor


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
@pytest.mark.parametrize("tls_secret", ["eyvan-tls", ""])
def test_real_helm_monitor_verifies_tls_or_uses_plaintext(tls_secret: str) -> None:
    arguments = [
        "--set",
        "serviceMonitor.enabled=true",
        "--set",
        "controlPlane.insecurePlaintextBind=true",
        "--set-string",
        f"controlPlane.tls.existingSecret={tls_secret}",
    ]
    if tls_secret:
        for name, value in TRUST.items():
            arguments.extend(["--set-string", f"serviceMonitor.tls.{name}={value}"])
    documents = _render_chart(*arguments)
    endpoint = _resource(documents, "ServiceMonitor", "eyvan-control-plane")["spec"]["endpoints"][0]
    assert endpoint["scheme"] == ("https" if tls_secret else "http")
    assert endpoint["authorization"]["credentials"] == {"name": "eyvan-secrets", "key": "metrics-token"}
    if tls_secret:
        assert endpoint["tlsConfig"] == {
            "ca": {"secret": {"name": TRUST["caSecretName"], "key": TRUST["caSecretKey"]}},
            "serverName": TRUST["serverName"],
        }
    else:
        assert "tlsConfig" not in endpoint


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
@pytest.mark.parametrize("missing", list(TRUST))
@pytest.mark.parametrize("empty", ["", "   "])
def test_real_helm_tls_monitor_refuses_missing_trust(missing: str, empty: str) -> None:
    arguments = [
        "--set",
        "serviceMonitor.enabled=true",
        "--set-string",
        "controlPlane.tls.existingSecret=eyvan-tls",
    ]
    for name, value in TRUST.items():
        arguments.extend(["--set-string", f"serviceMonitor.tls.{name}={empty if name == missing else value}"])
    result = subprocess.run(
        [HELM, "template", "eyvan", str(CHART), "--namespace", "release-ns", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"serviceMonitor.tls.{missing} is required when the monitored listener uses TLS" in result.stderr
    assert "kind: ServiceMonitor" not in result.stdout


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
def test_real_helm_disabled_monitor_does_not_require_scraper_trust() -> None:
    documents = _render_chart("--set-string", "controlPlane.tls.existingSecret=eyvan-tls")
    assert not any(document["kind"] == "ServiceMonitor" for document in documents)
