# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The properties the shipped manifests have to keep.

Manifests are the part of a system nobody tests and everybody edits. A YAML file
has no compiler, no type checker and no test suite of its own, so a hardening
decision made once survives exactly until the next person needs a quick change
and does not know why the line was there.

These assert the decisions, not the file. Each one names the consequence of
losing it, because that is the information a future editor needs and the diff
will not carry.

Deliberately not a lint of every field. A test that pinned the whole document
would fail on every legitimate edit and be deleted within a month, which would
take the load-bearing assertions with it.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

DEPLOY = pathlib.Path(__file__).resolve().parent.parent / "deploy"


def _documents(name: str) -> list[dict]:
    return [document for document in yaml.safe_load_all((DEPLOY / name).read_text(encoding="utf-8")) if document]


def _kind(name: str, kind: str) -> dict:
    matches = [document for document in _documents(name) if document.get("kind") == kind]
    assert len(matches) == 1, f"{name} should contain exactly one {kind}, found {len(matches)}"
    return matches[0]


def _container(name: str) -> dict:
    return _kind(name, "Deployment")["spec"]["template"]["spec"]["containers"][0]


def _pod(name: str) -> dict:
    return _kind(name, "Deployment")["spec"]["template"]["spec"]


# --------------------------------------------------------------------------- probes


def test_readiness_and_liveness_probe_different_endpoints() -> None:
    """The whole reason the endpoints were split.

    Pointing liveness at the database-checking probe restarts every replica the
    moment the database blips, turning a recoverable outage into a total one and
    then preventing recovery by stampeding reconnections.
    """

    container = _container("control-plane.yaml")

    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"


def test_the_cell_agent_has_a_liveness_probe() -> None:
    """It had readiness alone, so a wedged agent stayed out of the Service and running.

    A cell whose agent never answers is capacity the control plane cannot
    observe, cannot release, and keeps paying for.
    """

    assert _container("cell-agent.yaml")["livenessProbe"]["httpGet"]["path"] == "/healthz"


def test_the_agent_liveness_threshold_tolerates_a_long_provider_call() -> None:
    """A restart mid-call loses the fence the agent was holding.

    So the threshold is generous on purpose, and a future edit that tightens it
    to look more responsive should have to change this line and read why.
    """

    probe = _container("cell-agent.yaml")["livenessProbe"]

    assert probe["failureThreshold"] * probe["periodSeconds"] >= 60


def test_the_cell_agent_has_one_nonoverlapping_fleet_owner() -> None:
    deployment = _kind("cell-agent.yaml", "Deployment")

    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}


# --------------------------------------------------------------------------- hardening


def test_the_control_plane_does_not_run_as_root() -> None:
    """It holds a database URL, an artifact signing key and an agent shared secret.

    Root inside that container is root over all three, and the process needs
    none of it: it binds an unprivileged port and writes nothing outside /tmp.
    """

    security = _pod("control-plane.yaml")["securityContext"]

    assert security["runAsNonRoot"] is True
    assert security["runAsUser"] != 0


def test_the_control_plane_container_cannot_escalate_or_write_its_image() -> None:
    """The two settings that decide whether a compromise can persist."""

    security = _container("control-plane.yaml")["securityContext"]

    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]


def test_a_read_only_root_still_leaves_a_writable_tmp() -> None:
    """Python needs one for anything that spools to disk.

    An emptyDir rather than relaxing the root filesystem: the point is that a
    compromise cannot persist, and a volume that vanishes with the pod keeps it.
    """

    mounts = {mount["mountPath"] for mount in _container("control-plane.yaml")["volumeMounts"]}
    volumes = {volume["name"] for volume in _pod("control-plane.yaml")["volumes"]}

    assert "/tmp" in mounts
    assert "tmp" in volumes


# --------------------------------------------------------------------------- exposure


def test_the_network_policy_does_not_blanket_allow_the_api_port() -> None:
    """``/metrics`` exports every tenant's spend, labeled by org and project.

    It is authenticated by a required scrape token, and the NetworkPolicy limits
    which workloads can present that credential. A rule with an empty ``from``
    allows the entire cluster and would silently undo that defense in depth.
    """

    policy = _kind("control-plane.yaml", "NetworkPolicy")

    assert policy["spec"]["policyTypes"] == ["Ingress"]
    for rule in policy["spec"]["ingress"]:
        assert rule.get("from"), "an ingress rule with no `from` allows every pod in the cluster"


def test_reaching_the_api_requires_an_explicit_label() -> None:
    """More friction than allowing the namespace, and the friction is the feature.

    An unlabelled workload reaching the control plane by accident is how a
    cluster-wide compromise becomes a fleet-wide one.
    """

    policy = _kind("control-plane.yaml", "NetworkPolicy")
    selectors = [
        selector
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
        for key, selector in peer.items()
        if key in {"podSelector", "namespaceSelector"}
    ]

    assert selectors, "the policy admits peers without selecting them"
    assert all(selector.get("matchLabels") for selector in selectors), (
        "an empty selector matches everything, which is the blanket allow this policy exists to avoid"
    )


def test_api_ingress_ands_namespace_and_pod_for_cross_namespace_clients() -> None:
    policy = _kind("control-plane.yaml", "NetworkPolicy")
    api_rule = next(
        rule
        for rule in policy["spec"]["ingress"]
        if any(
            peer.get("podSelector", {}).get("matchLabels", {}).get("eyvan.dev/api-client") == "true"
            for peer in rule["from"]
        )
    )

    assert api_rule["from"] == [
        {"podSelector": {"matchLabels": {"eyvan.dev/api-client": "true"}}},
        {
            "namespaceSelector": {"matchLabels": {"eyvan.dev/api-client": "true"}},
            "podSelector": {"matchLabels": {"eyvan.dev/api-client": "true"}},
        },
    ]
    assert not any(set(peer) == {"namespaceSelector"} for rule in policy["spec"]["ingress"] for peer in rule["from"])


def test_dynamic_kubernetes_workers_receive_the_api_client_grant() -> None:
    from eyvan.capacity.providers.kubernetes.workers import API_CLIENT_LABEL, job_manifest

    manifest = job_manifest(
        cell_id="cell-a",
        namespace="worker-ns",
        image="registry.example/worker@sha256:" + "a" * 64,
        worker_count=1,
        provider_generation=1,
        fencing_token="fence",
        resources={"cpu": "1"},
    )

    pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
    assert pod_labels[API_CLIENT_LABEL] == "true"


def test_scraping_and_api_paths_keep_distinct_network_selectors() -> None:
    """The selectors remain defense in depth even though both paths use TCP 8080.

    NetworkPolicy cannot authorize an HTTP path; API and metrics credentials do
    that. Distinct pod labels still avoid granting the socket to unrelated pods.
    """

    policy = _kind("control-plane.yaml", "NetworkPolicy")
    labels = {
        frozenset(selector["matchLabels"].items())
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
        for selector in peer.values()
    }

    assert frozenset({("eyvan.dev/api-client", "true")}) in labels
    assert frozenset({("eyvan.dev/metrics-scraper", "true")}) in labels


# --------------------------------------------------------------------------- startup prerequisites


def test_the_raw_control_plane_receives_the_required_metrics_token() -> None:
    """A network-bound server refuses to start without this exact setting."""

    environment = {item["name"]: item for item in _container("control-plane.yaml")["env"]}

    assert environment["EYVAN_METRICS_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "eyvan-secrets",
        "key": "metrics-token",
    }


def test_raw_dashboard_client_id_is_non_secret_runtime_configuration() -> None:
    environment = {item["name"]: item for item in _container("control-plane.yaml")["env"]}

    client_id = environment["EYVAN_OIDC_DASHBOARD_CLIENT_ID"]
    assert client_id["value"] == "eyvan-dashboard"
    assert "valueFrom" not in client_id


def test_the_raw_agent_is_a_distinct_image_and_entrypoint_with_its_root_secret() -> None:
    """The control-plane entrypoint starts the wrong process and cannot bootstrap workers."""

    control = _container("control-plane.yaml")
    agent = _container("cell-agent.yaml")
    environment = {item["name"]: item for item in agent["env"]}

    assert agent["image"] != control["image"]
    assert agent["command"] == ["eyvan-agent"]
    assert environment["EYVAN_NATIVE_WORKER_ROOT_SECRET"]["valueFrom"]["secretKeyRef"] == {
        "name": "eyvan-agent-secrets",
        "key": "native-worker-root-secret",
    }
    assert environment["EYVAN_CAPACITY_BACKEND"]["value"] == "gcp"
    for artifact in ("PACKAGE", "WHEELHOUSE"):
        assert environment[f"EYVAN_NATIVE_WORKER_{artifact}_SHA256"]["value"] == ("REPLACE_WITH_64_LOWERCASE_HEX_SHA256")
        assert environment[f"EYVAN_NATIVE_WORKER_{artifact}_GENERATION"]["value"] == ("REPLACE_WITH_GCS_GENERATION")


def test_raw_image_markers_require_registry_digests_without_fabricating_one() -> None:
    """Qualification must replace an obvious marker, never accidentally deploy `latest`."""

    control = _container("control-plane.yaml")["image"]
    agent = _container("cell-agent.yaml")["image"]
    migration = _kind("migration-job.yaml", "Job")["spec"]["template"]["spec"]["containers"][0]["image"]

    assert control == migration
    assert agent != control
    for image in (control, agent, migration):
        assert ":latest" not in image
        assert image.endswith("@sha256:REPLACE_WITH_PUBLISHED_DIGEST")


def test_raw_prerequisites_exist_before_the_migration_and_deployments() -> None:
    prerequisites = _documents("prerequisites.yaml")
    prerequisite_kinds = [document["kind"] for document in prerequisites]

    assert prerequisite_kinds == ["Namespace", "ConfigMap", "ServiceAccount", "ServiceAccount"]
    assert {document["metadata"]["name"] for document in prerequisites if document["kind"] == "ServiceAccount"} == {
        "eyvan-control-plane",
        "eyvan-cell-agent",
    }
    assert not any(
        document["kind"] in {"Namespace", "ConfigMap", "ServiceAccount"} for document in _documents("control-plane.yaml")
    )
    assert not any(document["kind"] == "ServiceAccount" for document in _documents("cell-agent.yaml"))
    assert _kind("migration-job.yaml", "Job")["spec"]["template"]["spec"]["serviceAccountName"] == (
        "eyvan-control-plane"
    )


def test_raw_control_plane_ksa_documents_its_gcs_workload_identity() -> None:
    accounts = {
        document["metadata"]["name"]: document
        for document in _documents("prerequisites.yaml")
        if document["kind"] == "ServiceAccount"
    }

    assert accounts["eyvan-control-plane"]["metadata"]["annotations"] == {
        "iam.gke.io/gcp-service-account": "eyvan-control-plane@PROJECT_ID.iam.gserviceaccount.com"
    }
    assert accounts["eyvan-cell-agent"]["metadata"]["annotations"] == {
        "iam.gke.io/gcp-service-account": "eyvan-cell-agent@PROJECT_ID.iam.gserviceaccount.com"
    }


def test_raw_platform_config_is_mounted_read_only_at_the_server_default() -> None:
    config = _kind("prerequisites.yaml", "ConfigMap")
    container = _container("control-plane.yaml")
    mounts = {mount["name"]: mount for mount in container["volumeMounts"]}
    volumes = {volume["name"]: volume for volume in _pod("control-plane.yaml")["volumes"]}

    assert "platform.yaml" in config["data"]
    assert mounts["platform-config"] == {
        "name": "platform-config",
        "mountPath": "/etc/eyvan/platform.yaml",
        "subPath": "platform.yaml",
        "readOnly": True,
    }
    assert volumes["platform-config"]["configMap"]["name"] == "eyvan-platform"


def test_install_document_requires_a_rollout_after_raw_identity_or_config_changes() -> None:
    install = (DEPLOY.parent / "docs" / "install.md").read_text(encoding="utf-8")

    assert "platform ConfigMap does not restart" in install
    assert "ServiceAccount annotation" in install
    assert "changing the `eyvan-control-plane` WIF annotation" in install
    assert "rollout restart deployment/eyvan-control-plane deployment/eyvan-cell-agent" in install


def test_install_document_waits_for_raw_migration_before_deployments() -> None:
    install = (DEPLOY.parent / "docs" / "install.md").read_text(encoding="utf-8")

    prerequisites = install.index("kubectl apply -f deploy/prerequisites.yaml")
    migration = install.index("kubectl apply -f deploy/migration-job.yaml")
    wait = install.index("--for=condition=complete")
    agent = install.index("kubectl apply -f deploy/cell-agent.yaml")
    control = install.index("kubectl apply -f deploy/control-plane.yaml")
    assert prerequisites < migration < wait < agent < control


# --------------------------------------------------------------------------- availability


def test_a_drain_cannot_take_the_last_control_plane_replica() -> None:
    """A node drain with no budget evicts both replicas at once.

    The control plane is what releases capacity; while it is down, every cell in
    the fleet keeps billing and nothing reclaims them.
    """

    budget = _kind("control-plane.yaml", "PodDisruptionBudget")

    assert budget["spec"]["minAvailable"] == 1
    assert budget["spec"]["selector"]["matchLabels"] == {"app": "eyvan-control-plane"}


def test_the_disruption_budget_selects_the_deployment_it_protects() -> None:
    """A budget whose selector matches nothing is a budget that protects nothing.

    It applies cleanly, reports healthy, and evicts everything.
    """

    budget = _kind("control-plane.yaml", "PodDisruptionBudget")
    pods = _kind("control-plane.yaml", "Deployment")["spec"]["template"]["metadata"]["labels"]

    assert budget["spec"]["selector"]["matchLabels"].items() <= pods.items()


# --------------------------------------------------------------------------- shape


@pytest.mark.parametrize("name", ["prerequisites.yaml", "control-plane.yaml", "cell-agent.yaml", "migration-job.yaml"])
def test_every_shipped_manifest_parses(name: str) -> None:
    """A manifest that does not parse fails at ``kubectl apply``, mid-deploy.

    Cheap to check here, expensive to discover there.
    """

    documents = _documents(name)

    assert documents
    assert all(document.get("kind") and document.get("apiVersion") for document in documents)


# --------------------------------------------------------------------------- identity


def test_the_dashboard_identity_is_configurable_without_rebuilding() -> None:
    """One image, two identity providers.

    The issuer and client id were compiled in by ``VITE_`` build arguments, so an
    operator with a staging and a production realm had to build two images that
    differed in nothing else. The control plane now serves them, and the
    dashboard prefers the baked values when they exist so every existing image
    behaves identically.
    """

    from eyvan.config import EyvanSettings

    assert hasattr(EyvanSettings(), "oidc_dashboard_client_id")

    oidc = (pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "src" / "auth" / "oidc.ts").read_text(
        encoding="utf-8"
    )
    assert "/api/v1/dashboard/config" in oidc


def test_dashboard_identity_is_runtime_only_in_the_reusable_image() -> None:
    """The release image must not bake one deployment's OIDC identity.

    Older development images declared an unset ``VITE_EYVAN_OIDC_ISSUER``
    argument and relied on the browser's empty-string fallback. The immutable
    release image has a stronger boundary: it accepts no identity build
    arguments at all and resolves public OIDC configuration from the server.
    """

    dockerfile = (pathlib.Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")
    oidc = (pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "src" / "auth" / "oidc.ts").read_text(
        encoding="utf-8"
    )

    assert "ARG VITE_EYVAN_OIDC_ISSUER" not in dockerfile
    assert "ARG VITE_EYVAN_OIDC_CLIENT_ID" not in dockerfile
    assert "/api/v1/dashboard/config" in oidc
    # Development builds still treat an explicitly empty Vite value as unset.
    assert "import.meta.env.VITE_EYVAN_OIDC_ISSUER || undefined" in oidc


# --------------------------------------------------------------------------- found on hardware


def test_a_slow_ingress_call_is_retryable_rather_than_a_cell_failure() -> None:
    """Found by running P6, and only findable that way.

    A `firewall-rules update` on a rule with one source range exceeded the
    hardcoded 60-second timeout on a degraded link. The raw ``TimeoutExpired``
    reached the agent as an untyped 500, the control plane recorded
    ``provider_failure`` with no detail, and the Compute instance that call had
    already created kept running and billing.

    Slow is not failed. The ingress rule is reconciled on every pass, so a slow
    call should cost one cycle rather than a cell.
    """

    import subprocess

    from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode
    from eyvan.capacity.providers.gcp import native_worker_ingress

    def slow(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["gcloud", "compute", "firewall-rules", "update"], timeout=1.0)

    original = native_worker_ingress.subprocess.run
    native_worker_ingress.subprocess.run = slow  # type: ignore[assignment]
    try:
        with pytest.raises(CapacityProviderError) as raised:
            native_worker_ingress._gcloud_runner(["compute", "firewall-rules", "update", "rule"])
    finally:
        native_worker_ingress.subprocess.run = original  # type: ignore[assignment]

    assert raised.value.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
    assert raised.value.retryable is True
    # The detail an operator was missing entirely: what timed out, and after how
    # long. It lived only in an agent log file before this.
    assert "timed out" in str(raised.value)
    assert "firewall-rules" in str(raised.value)


def test_the_ingress_timeout_is_configurable_and_not_sixty_seconds() -> None:
    """Sixty was proven too tight by a real call on a real link.

    Configurable for the same reason the TPU path is: nobody can pick one number
    that suits every network between a laptop and Google.
    """

    from eyvan.capacity.providers.gcp.native_worker_ingress import DEFAULT_INGRESS_TIMEOUT_S

    assert DEFAULT_INGRESS_TIMEOUT_S > 60.0
