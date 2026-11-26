# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Source contracts and optional real Helm regressions for the chart's policies and ServiceMonitor.

Source contracts do not emulate Go templating or claim cluster/CNI/scrape proof.
"""

from __future__ import annotations

import pathlib

import pytest
from test_helm_chart import HELM, _render_chart, _resource

from eyvan.capacity.providers.kubernetes.workers import API_CLIENT_LABEL, job_manifest

CHART = pathlib.Path(__file__).resolve().parent.parent / "deploy" / "helm" / "eyvan"
POLICIES = CHART / "templates" / "policies.yaml"


def test_service_monitor_follows_tls_listener_source_contract() -> None:
    policies = POLICIES.read_text(encoding="utf-8")
    monitor = policies.split("kind: ServiceMonitor", 1)[1]
    control = (CHART / "templates" / "control-plane.yaml").read_text(encoding="utf-8")
    assert 'ternary "HTTPS" "HTTP" (ne .Values.controlPlane.tls.existingSecret "")' in control
    assert 'scheme: {{ ternary "https" "http" (ne .Values.controlPlane.tls.existingSecret "") }}' in monitor
    assert "tlsConfig:" in monitor
    assert "insecureSkipVerify: true" not in monitor


def test_canonical_worker_ingress_is_independent_of_external_client_label() -> None:
    job = job_manifest(
        cell_id="cell-proof",
        namespace="worker-ns",
        image="registry.example/eyvan-worker@sha256:" + "a" * 64,
        worker_count=1,
        provider_generation=1,
        fencing_token="writer-014-proof",
        resources={"cpu": "1", "memory": "1Gi"},
    )
    labels = job["spec"]["template"]["metadata"]["labels"]
    assert labels[API_CLIENT_LABEL] == "true"
    assert "example.com/eyvan-client" not in labels
    policies = POLICIES.read_text(encoding="utf-8")
    # Check the actual source peer, rather than a hand-written template evaluator.
    canonical_peer = (
        "        - namespaceSelector:\n"
        "            matchLabels:\n"
        '              kubernetes.io/metadata.name: {{ include "eyvan.workerNamespace" . | quote }}\n'
        "          podSelector:\n"
        "            matchLabels:\n"
        f'              {API_CLIENT_LABEL}: "true"'
    )
    assert canonical_peer in policies
    assert (
        '{{- if and .Values.cellAgent.enabled (eq (include "eyvan.capacityBackend" .) "kubernetes") '
        '(ne .Values.networkPolicy.apiClientLabel "eyvan.dev/api-client") }}'
    ) in policies
    # Retain both existing configurable external-client peers.
    assert policies.count('{{ .Values.networkPolicy.apiClientLabel }}: "true"') == 2


def _worker_documents(api_label: str, worker_namespace: str) -> list[dict]:
    return _render_chart(
        "--set",
        "cellAgent.enabled=true",
        "--set-string",
        "cellAgent.fleetStateUri=gs://state-bucket/eyvan/state.json",
        "--set-string",
        "cellAgent.capacityBackend=kubernetes",
        "--set-string",
        f"cellAgent.kubernetes.workerNamespace={worker_namespace}",
        "--set-string",
        "workerImage.digest=sha256:" + "a" * 64,
        "--set-string",
        f"networkPolicy.apiClientLabel={api_label}",
    )


def _allows(policy: dict, pod_labels: dict, namespace: str) -> bool:
    for rule in policy["spec"]["ingress"]:
        if not any(port.get("protocol", "TCP") == "TCP" and port["port"] == 8080 for port in rule["ports"]):
            continue
        for peer in rule["from"]:
            namespace_labels = {"kubernetes.io/metadata.name": namespace}
            namespace_selector = peer.get("namespaceSelector")
            if namespace_selector is None:
                if namespace != policy["metadata"]["namespace"]:
                    continue
            elif not all(namespace_labels.get(key) == value for key, value in namespace_selector["matchLabels"].items()):
                continue
            selector = peer["podSelector"]["matchLabels"]
            if all(pod_labels.get(key) == value for key, value in selector.items()):
                return True
    return False


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
@pytest.mark.parametrize("api_label", ["eyvan.dev/api-client", "example.com/eyvan-client"])
@pytest.mark.parametrize("worker_namespace", ["release-ns", "worker-ns"])
def test_real_helm_custom_client_label_preserves_narrow_worker_access(api_label: str, worker_namespace: str) -> None:
    policy = _resource(_worker_documents(api_label, worker_namespace), "NetworkPolicy", "eyvan-control-plane")
    job = job_manifest(
        cell_id="cell-proof",
        namespace=worker_namespace,
        image="registry.example/eyvan-worker@sha256:" + "a" * 64,
        worker_count=1,
        provider_generation=1,
        fencing_token="writer-014-proof",
        resources={"cpu": "1", "memory": "1Gi"},
    )
    labels = job["spec"]["template"]["metadata"]["labels"]
    assert _allows(policy, labels, worker_namespace)
    assert not _allows(policy, {}, worker_namespace)
    assert not _allows(policy, labels, "unrelated-ns")
    assert _allows(policy, {api_label: "true"}, "release-ns")
    assert _allows(policy, {api_label: "true"}, worker_namespace)


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
@pytest.mark.parametrize("tls_secret, scheme", [("eyvan-tls", "https"), ("", "http")])
def test_real_helm_service_monitor_protocol_matches_listener(tls_secret: str, scheme: str) -> None:
    documents = _render_chart(
        "--set",
        "serviceMonitor.enabled=true",
        "--set",
        "controlPlane.insecurePlaintextBind=true",
        "--set-string",
        f"controlPlane.tls.existingSecret={tls_secret}",
        "--set-string",
        "serviceMonitor.tls.caSecretName=eyvan-metrics-ca",
        "--set-string",
        "serviceMonitor.tls.caSecretKey=ca.crt",
        "--set-string",
        "serviceMonitor.tls.serverName=eyvan-control-plane.release-ns.svc.cluster.local",
    )
    monitor = _resource(documents, "ServiceMonitor", "eyvan-control-plane")
    assert monitor["spec"]["endpoints"][0].get("scheme", "http") == scheme
