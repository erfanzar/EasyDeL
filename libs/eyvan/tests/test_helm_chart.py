# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Static chart contracts plus real Helm lint/render when the binary exists.

The static checks cover mistakes Go templating also lets through. A conditional
test invokes the real ``helm lint`` and ``helm template`` commands when Helm is
installed; neither path claims Kubernetes admission or runtime proof.

The static checks target the class of mistake an operator otherwise discovers at
install time:

* a ``.Values`` reference to a key that does not exist in ``values.yaml``,
  which renders as an empty string and produces a deployment missing an
  environment variable rather than an error;
* a default that is a credential or a permissive setting;
* a migration hook that is not actually a hook;
* the hardening decisions the raw manifests carry, silently absent from the
  templated copies.

Deliberately not a re-implementation of Go templating. Testing a hand-rolled
evaluator would test the evaluator.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

CHART = pathlib.Path(__file__).resolve().parent.parent / "deploy" / "helm" / "eyvan"
ROOT = CHART.parents[2]


def _tool(name: str) -> str | None:
    explicit = os.getenv(f"{name.upper()}_BIN")
    if explicit:
        return explicit
    discovered = shutil.which(name)
    bundled = pathlib.Path("/tmp/eyvan-tools") / name
    return discovered or (str(bundled) if bundled.is_file() else None)


HELM = _tool("helm")


def _values() -> dict:
    return yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))


def _templates() -> list[pathlib.Path]:
    """Every template, including ``_helpers.tpl``.

    The helper file is where ``nameOverride`` and the namespace are read, so a
    scan restricted to ``*.yaml`` reports them as configurable-but-unused --
    which is the kind of false positive that gets a guard deleted.
    """

    return sorted(path for path in (CHART / "templates").iterdir() if path.suffix in {".yaml", ".tpl"})


def _render_chart(*arguments: str) -> list[dict]:
    assert HELM is not None
    rendered = subprocess.run(
        [HELM, "template", "eyvan", str(CHART), "--namespace", "release-ns", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rendered.returncode == 0, rendered.stdout + rendered.stderr
    return [document for document in yaml.safe_load_all(rendered.stdout) if document]


def _resource(documents: list[dict], kind: str, name: str) -> dict:
    matches = [item for item in documents if item.get("kind") == kind and item.get("metadata", {}).get("name") == name]
    assert len(matches) == 1, (kind, name, matches)
    return matches[0]


def _resolve(values: dict, path: str) -> bool:
    """Whether ``a.b.c`` names something in ``values.yaml``."""

    node: object = values
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _references(text: str) -> set[str]:
    """Every ``.Values.a.b`` path a template reads."""

    return {match.group(1) for match in re.finditer(r"\.Values\.([A-Za-z0-9_.]+)", text)}


# --------------------------------------------------------------------------- shape


def test_the_chart_metadata_parses_and_names_a_version() -> None:
    """A malformed Chart.yaml fails at install, after the operator has committed."""

    chart = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))

    assert chart["apiVersion"] == "v2"
    assert chart["name"] == "eyvan"
    assert chart["version"]


def test_the_values_file_parses() -> None:
    """Everything else here depends on it, so it is asserted first."""

    assert isinstance(_values(), dict)


@pytest.mark.parametrize("template", _templates(), ids=lambda path: path.name)
def test_every_values_reference_resolves(template: pathlib.Path) -> None:
    """The mistake Go templating does not catch.

    A reference to a key that is not in ``values.yaml`` renders as the empty
    string. The release succeeds, the Deployment applies, and the container
    starts without the environment variable it needed -- which surfaces as a
    runtime failure with no connection to the typo that caused it.
    """

    values = _values()
    dangling = sorted(path for path in _references(template.read_text(encoding="utf-8")) if not _resolve(values, path))

    assert not dangling, f"{template.name} reads values that do not exist: {', '.join(dangling)}"


def test_no_value_is_referenced_only_by_its_own_default() -> None:
    """A values key nothing reads is configuration an operator sets with no effect.

    Worse than a missing key, because it looks like it worked.
    """

    referenced = {path for template in _templates() for path in _references(template.read_text(encoding="utf-8"))}

    def leaves(node: object, prefix: str = "") -> list[str]:
        if not isinstance(node, dict):
            return [prefix]
        return [leaf for key, value in node.items() for leaf in leaves(value, f"{prefix}.{key}".lstrip("."))]

    def is_read(leaf: str) -> bool:
        # A whole subtree can be consumed at once -- `toYaml .Values.x.resources`
        # renders every key beneath it -- so an ancestor reference counts.
        parts = leaf.split(".")
        return any(".".join(parts[: index + 1]) in referenced for index in range(len(parts)))

    unread = sorted(leaf for leaf in leaves(_values()) if not is_read(leaf))

    assert not unread, f"these values are configurable and read by nothing: {', '.join(unread)}"


# --------------------------------------------------------------------------- defaults


def test_no_default_is_a_credential() -> None:
    """A chart that ships a working secret ships it to everyone who installs it."""

    values = (CHART / "values.yaml").read_text(encoding="utf-8")

    for suspicious in ("password", "secretKey", "token:", "-----BEGIN"):
        assert suspicious not in values, f"values.yaml appears to carry a credential: {suspicious}"


def test_the_insecure_options_default_off() -> None:
    """An operator who installs with no values gets a plane that refuses to serve.

    Which is the correct direction: an empty trusted-callable list refuses
    pickled callables, no issuer means every request is 401, and the schema is
    touched only by the migration hook.
    """

    values = _values()

    assert values["controlPlane"]["autoCreateSchema"] is False
    assert values["controlPlane"]["trustedCallableProjects"] == ""
    assert values["controlPlane"]["oidc"]["issuer"] == ""
    assert values["controlPlane"]["repositoryBackend"] == "postgres"
    assert values["image"]["tag"] != "latest"
    assert values["agentImage"]["tag"] != "latest"
    assert "tag" not in values["workerImage"]
    assert values["workerImage"]["digest"] == ""
    assert values["cellAgent"]["enabled"] is False
    assert values["cellAgent"]["replicas"] == 1
    assert values["cellAgent"]["capacityBackend"] == "gcp"
    assert values["migration"]["maintenanceAcknowledged"] is False


def test_the_protective_options_default_on() -> None:
    """The reverse of the rule above, and just as load-bearing.

    A NetworkPolicy or disruption budget an operator has to remember to enable
    is one most deployments will not have.
    """

    values = _values()

    assert values["networkPolicy"]["enabled"] is True
    assert values["podDisruptionBudget"]["enabled"] is True
    assert values["migration"]["enabled"] is True


def test_the_service_monitor_defaults_off_because_its_crd_may_not_exist() -> None:
    """A chart that installs an object of an unknown kind fails the whole release."""

    assert _values()["serviceMonitor"]["enabled"] is False


# --------------------------------------------------------------------------- ordering


def test_the_migration_is_a_pre_install_hook() -> None:
    """Otherwise an operator orders the Job against the rollout by hand.

    Getting that wrong starts a control plane against a schema it refuses, which
    is correct behaviour and a confusing first impression.
    """

    migration = (CHART / "templates" / "migration-job.yaml").read_text(encoding="utf-8")

    assert '"helm.sh/hook": pre-install,pre-upgrade' in migration
    assert "helm.sh/hook-weight" in migration


def test_migration_hook_requires_an_explicit_maintenance_acknowledgement_for_schema_changes() -> None:
    migration = (CHART / "templates" / "migration-job.yaml").read_text(encoding="utf-8")
    values = _values()

    assert values["migration"]["maintenanceAcknowledged"] is False
    assert ".Values.migration.maintenanceAcknowledged" in migration
    assert 'args: ["--maintenance-acknowledged"]' in migration


def test_release_namespace_is_authoritative_and_worker_namespace_defaults_to_it() -> None:
    """The migration namespace cannot diverge; an empty worker namespace resolves safely."""

    templates = "\n".join(template.read_text(encoding="utf-8") for template in _templates())
    helper = (CHART / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    readme = (CHART / "README.md").read_text(encoding="utf-8")

    assert ".Values.namespace" not in templates
    assert ".Release.Namespace" in helper
    assert ".Values.cellAgent.kubernetes.workerNamespace" in helper
    assert "exact Kubernetes DNS-label namespace name" in helper
    assert _values()["cellAgent"]["kubernetes"]["workerNamespace"] == ""
    assert not (CHART / "templates" / "namespace.yaml").exists()
    assert "--create-namespace" in readme


def test_migration_hook_uses_only_preexisting_namespace_resources() -> None:
    """Normal chart ServiceAccounts do not exist when a pre-install hook runs."""

    migration = (CHART / "templates" / "migration-job.yaml").read_text(encoding="utf-8")
    readme = (CHART / "README.md").read_text(encoding="utf-8")

    assert "serviceAccountName:" not in migration
    assert "automountServiceAccountToken: false" in migration
    assert "Create the namespace and Secret before the first install" in readme


# --------------------------------------------------------------------------- runtime wiring


def test_the_control_plane_and_service_monitor_share_metrics_authentication() -> None:
    control = (CHART / "templates" / "control-plane.yaml").read_text(encoding="utf-8")
    policies = (CHART / "templates" / "policies.yaml").read_text(encoding="utf-8")

    assert "EYVAN_METRICS_TOKEN" in control
    assert "key: metrics-token" in control
    assert "authorization:" in policies
    assert "type: Bearer" in policies
    assert "key: metrics-token" in policies


def test_the_agent_has_its_own_image_entrypoint_and_worker_root_secret() -> None:
    helper = (CHART / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    agent = (CHART / "templates" / "cell-agent.yaml").read_text(encoding="utf-8")

    assert ".Values.agentImage.repository" in helper
    assert 'include "eyvan.cellAgentImage"' in agent
    assert 'command: ["eyvan-agent"]' in agent
    assert "EYVAN_NATIVE_WORKER_ROOT_SECRET" in agent
    assert "key: native-worker-root-secret" in agent


def test_enabled_cell_agent_has_one_nonoverlapping_owner_and_durable_state() -> None:
    helper = (CHART / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    agent = (CHART / "templates" / "cell-agent.yaml").read_text(encoding="utf-8")

    assert 'include "eyvan.validateCellAgent"' in agent
    assert "cellAgent.replicas must be exactly 1" in helper
    assert "canonical gs://bucket/non-empty-object" in helper
    assert "replicas: 1" in agent
    assert "type: Recreate" in agent


def test_kubernetes_capacity_wires_an_immutable_worker_and_namespaced_least_privilege_rbac() -> None:
    helper = (CHART / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    agent = (CHART / "templates" / "cell-agent.yaml").read_text(encoding="utf-8")
    accounts = (CHART / "templates" / "serviceaccount.yaml").read_text(encoding="utf-8")
    rbac = (CHART / "templates" / "kubernetes-worker-rbac.yaml").read_text(encoding="utf-8")

    assert "EYVAN_CAPACITY_BACKEND" in agent
    assert "EYVAN_KUBERNETES_NAMESPACE" in agent
    assert "EYVAN_KUBERNETES_WORKER_IMAGE" in agent
    assert 'printf "%s@%s" $repository $digest' in helper
    assert "workerImage.digest is required for Kubernetes capacity" in helper
    assert "name: eyvan-native-worker" in accounts
    assert ".Values.serviceAccounts.cellAgent.annotations" in accounts
    assert ".Values.serviceAccounts.worker.annotations" in accounts
    assert 'resources: ["jobs"]' in rbac
    assert rbac.count('verbs: ["create", "get", "delete"]') == 2
    assert 'resources: ["pods", "events"]' in rbac
    assert 'resources: ["services", "secrets"]' in rbac
    assert "kind: RoleBinding" in rbac
    assert "kind: ClusterRole" not in rbac
    assert "kind: ClusterRoleBinding" not in rbac


def test_the_worker_dockerfile_is_a_distinct_native_worker_artifact() -> None:
    worker = (ROOT / "Dockerfile.worker").read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["eyvan-native-worker"]' in worker
    assert "requirements-docker.txt" in worker
    assert "--require-hashes" in worker
    assert "EYVAN_WORKER_BASE_IMAGE" in worker
    assert "EYVAN_UV_IMAGE" in worker
    assert "org.opencontainers.image.version" in worker
    assert ":latest" not in worker


def test_the_chart_mounts_its_platform_config_at_the_server_default() -> None:
    platform = (CHART / "templates" / "platform-config.yaml").read_text(encoding="utf-8")
    control = (CHART / "templates" / "control-plane.yaml").read_text(encoding="utf-8")

    assert "kind: ConfigMap" in platform
    assert "platform.yaml: |" in platform
    assert ".Release.Namespace" in platform
    assert "mountPath: /etc/eyvan/platform.yaml" in control
    assert "readOnly: true" in control
    assert "name: eyvan-platform" in control
    assert "checksum/platform-config" in control
    assert 'Template.BasePath "/platform-config.yaml"' in control


def test_qualification_uses_registry_digests_without_an_example_digest() -> None:
    helper = (CHART / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    readme = (CHART / "README.md").read_text(encoding="utf-8")

    assert 'printf "%s@%s" .Values.image.repository .Values.image.digest' in helper
    assert 'printf "%s@%s" .Values.agentImage.repository .Values.agentImage.digest' in helper
    assert 'printf "%s@%s" $repository $digest' in helper
    assert 'regexMatch "^sha256:[0-9a-f]{64}$"' in helper
    assert '--set-string image.digest="$EYVAN_CONTROL_PLANE_DIGEST"' in readme
    assert '--set-string agentImage.digest="$EYVAN_AGENT_DIGEST"' in readme
    assert '--set-string workerImage.digest="$EYVAN_WORKER_DIGEST"' in readme
    assert not re.search(r"sha256:[0-9a-f]{64}", readme)


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
def test_real_helm_renders_gcp_and_kubernetes_capacity_modes() -> None:
    default_documents = _render_chart()
    assert not any(
        item["kind"] == "Deployment" and item["metadata"]["name"] == "eyvan-cell-agent" for item in default_documents
    )
    assert not any(item["kind"] == "Role" for item in default_documents)
    assert not any(
        item["kind"] == "ServiceAccount" and item["metadata"]["name"] == "eyvan-native-worker"
        for item in default_documents
    )

    gcp_documents = _render_chart(
        "--set",
        "cellAgent.enabled=true",
        "--set-string",
        "cellAgent.fleetStateUri=gs://state-bucket/eyvan/gcp-state.json",
    )
    gcp_agent = _resource(gcp_documents, "Deployment", "eyvan-cell-agent")
    gcp_env = {item["name"]: item for item in gcp_agent["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert gcp_agent["spec"]["replicas"] == 1
    assert gcp_agent["spec"]["strategy"] == {"type": "Recreate"}
    assert gcp_env["EYVAN_CAPACITY_BACKEND"]["value"] == "gcp"
    assert gcp_env["EYVAN_FLEET_STATE_URI"]["value"] == "gs://state-bucket/eyvan/gcp-state.json"

    digest = "sha256:" + "a" * 64
    kubernetes_documents = _render_chart(
        "--set",
        "cellAgent.enabled=true",
        "--set-string",
        "cellAgent.fleetStateUri=gs://state-bucket/eyvan/kubernetes-state.json",
        "--set-string",
        "cellAgent.capacityBackend=kubernetes",
        "--set-string",
        "cellAgent.kubernetes.workerNamespace=worker-ns",
        "--set-string",
        "workerImage.repository=registry.example/eyvan-native-worker",
        "--set-string",
        f"workerImage.digest={digest}",
        "--set-string",
        r"serviceAccounts.cellAgent.annotations.iam\.gke\.io/gcp-service-account=agent@example.iam.gserviceaccount.com",
        "--set-string",
        r"serviceAccounts.worker.annotations.iam\.gke\.io/gcp-service-account=worker@example.iam.gserviceaccount.com",
    )
    agent = _resource(kubernetes_documents, "Deployment", "eyvan-cell-agent")
    environment = {
        item["name"]: item["value"]
        for item in agent["spec"]["template"]["spec"]["containers"][0]["env"]
        if "value" in item
    }
    assert environment["EYVAN_CAPACITY_BACKEND"] == "kubernetes"
    assert environment["EYVAN_KUBERNETES_NAMESPACE"] == "worker-ns"
    assert environment["EYVAN_KUBERNETES_WORKER_IMAGE"] == f"registry.example/eyvan-native-worker@{digest}"

    role = _resource(kubernetes_documents, "Role", "eyvan-cell-agent-capacity")
    binding = _resource(kubernetes_documents, "RoleBinding", "eyvan-cell-agent-capacity")
    worker_account = _resource(kubernetes_documents, "ServiceAccount", "eyvan-native-worker")
    agent_account = _resource(kubernetes_documents, "ServiceAccount", "eyvan-cell-agent")
    assert role["metadata"]["namespace"] == "worker-ns"
    service_secret_rule = next(rule for rule in role["rules"] if rule["resources"] == ["services", "secrets"])
    assert service_secret_rule["verbs"] == ["create", "get", "delete"]
    assert binding["metadata"]["namespace"] == "worker-ns"
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "eyvan-cell-agent", "namespace": "release-ns"}]
    assert worker_account["metadata"]["namespace"] == "worker-ns"
    assert worker_account["automountServiceAccountToken"] is False
    assert worker_account["metadata"]["annotations"]["iam.gke.io/gcp-service-account"].startswith("worker@")
    assert agent_account["metadata"]["annotations"]["iam.gke.io/gcp-service-account"].startswith("agent@")

    policy = _resource(kubernetes_documents, "NetworkPolicy", "eyvan-control-plane")
    api_peers = policy["spec"]["ingress"][0]["from"]
    assert api_peers == [
        {"podSelector": {"matchLabels": {"eyvan.dev/api-client": "true"}}},
        {
            "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "worker-ns"}},
            "podSelector": {"matchLabels": {"eyvan.dev/api-client": "true"}},
        },
    ]
    assert not any(set(peer) == {"namespaceSelector"} for rule in policy["spec"]["ingress"] for peer in rule["from"])


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
def test_real_helm_rejects_a_mutable_or_missing_kubernetes_worker_image() -> None:
    assert HELM is not None
    for extra in ((), ("--set-string", "workerImage.digest=sha256:not-a-digest")):
        rendered = subprocess.run(
            [
                HELM,
                "template",
                "eyvan",
                str(CHART),
                "--namespace",
                "release-ns",
                "--set",
                "cellAgent.enabled=true",
                "--set-string",
                "cellAgent.fleetStateUri=gs://state-bucket/eyvan/state.json",
                "--set-string",
                "cellAgent.capacityBackend=kubernetes",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rendered.returncode != 0
        assert "workerImage.digest" in rendered.stderr


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--set", "cellAgent.enabled=true"), "cellAgent.fleetStateUri"),
        (
            (
                "--set",
                "cellAgent.enabled=true",
                "--set-string",
                "cellAgent.fleetStateUri=gs://state-bucket/",
            ),
            "canonical gs://bucket/non-empty-object",
        ),
        (
            (
                "--set",
                "cellAgent.enabled=true",
                "--set-string",
                "cellAgent.fleetStateUri=gs://State-Bucket/eyvan/state.json",
            ),
            "canonical gs://bucket/non-empty-object",
        ),
        (
            (
                "--set",
                "cellAgent.enabled=true",
                "--set-string",
                "cellAgent.fleetStateUri=gs://state-bucket/eyvan/state.json?generation=1",
            ),
            "canonical gs://bucket/non-empty-object",
        ),
        (
            (
                "--set",
                "cellAgent.enabled=true",
                "--set-string",
                "cellAgent.fleetStateUri=gs://state-bucket/eyvan/state.json",
                "--set",
                "cellAgent.replicas=2",
            ),
            "cellAgent.replicas must be exactly 1",
        ),
    ],
)
def test_real_helm_rejects_unsafe_cell_agent_ownership(arguments: tuple[str, ...], message: str) -> None:
    assert HELM is not None
    rendered = subprocess.run(
        [HELM, "template", "eyvan", str(CHART), "--namespace", "release-ns", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )

    assert rendered.returncode != 0
    assert message in rendered.stderr


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
def test_real_helm_rejects_a_worker_namespace_that_cannot_be_its_exact_namespace_label() -> None:
    assert HELM is not None
    rendered = subprocess.run(
        [
            HELM,
            "template",
            "eyvan",
            str(CHART),
            "--namespace",
            "release-ns",
            "--set-string",
            "cellAgent.kubernetes.workerNamespace=Not_A_Namespace",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert rendered.returncode != 0
    assert "exact Kubernetes DNS-label namespace name" in rendered.stderr


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
def test_real_helm_refuses_unverified_gcs_bootstrap_artifacts() -> None:
    assert HELM is not None
    rendered = subprocess.run(
        [
            HELM,
            "template",
            "eyvan",
            str(CHART),
            "--namespace",
            "release-ns",
            "--set",
            "cellAgent.enabled=true",
            "--set-string",
            "cellAgent.fleetStateUri=gs://state-bucket/eyvan/state.json",
            "--set-string",
            "cellAgent.nativeWorkerPackage=gs://artifacts.invalid/eyvan.whl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rendered.returncode != 0
    assert "nativeWorkerPackageSha256" in rendered.stderr


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
def test_platform_checksum_changes_and_migration_acknowledgement_is_not_sticky() -> None:
    default_documents = _render_chart()
    changed_documents = _render_chart("--set-string", "platform.defaultAgentEndpoint=https://agent.example.invalid")
    acknowledged_documents = _render_chart("--set", "migration.maintenanceAcknowledged=true")

    default_control = _resource(default_documents, "Deployment", "eyvan-control-plane")
    changed_control = _resource(changed_documents, "Deployment", "eyvan-control-plane")
    default_checksum = default_control["spec"]["template"]["metadata"]["annotations"]["checksum/platform-config"]
    changed_checksum = changed_control["spec"]["template"]["metadata"]["annotations"]["checksum/platform-config"]
    assert default_checksum != changed_checksum

    default_migration = _resource(default_documents, "Job", "eyvan-schema-migration")
    acknowledged_migration = _resource(acknowledged_documents, "Job", "eyvan-schema-migration")
    assert "args" not in default_migration["spec"]["template"]["spec"]["containers"][0]
    assert acknowledged_migration["spec"]["template"]["spec"]["containers"][0]["args"] == ["--maintenance-acknowledged"]


# --------------------------------------------------------------------------- hardening


def test_the_templated_control_plane_keeps_the_manifest_hardening() -> None:
    """The templates are a second copy, and a second copy is where a setting is lost.

    Each of these was a deliberate decision in ``deploy/control-plane.yaml``, and
    a chart that quietly dropped one would hand an operator a *less* safe
    deployment than the raw manifest they were replacing.
    """

    template = (CHART / "templates" / "control-plane.yaml").read_text(encoding="utf-8")

    for required in (
        "runAsNonRoot: true",
        "allowPrivilegeEscalation: false",
        "readOnlyRootFilesystem: true",
        'drop: ["ALL"]',
        "path: /readyz",
        "path: /healthz",
    ):
        assert required in template, f"the chart drops {required!r} from the control plane"


def test_the_templated_probes_are_not_the_same_endpoint() -> None:
    """Pointing liveness at the database check restarts every replica on a blip."""

    template = (CHART / "templates" / "control-plane.yaml").read_text(encoding="utf-8")
    readiness = template.index("readinessProbe")
    liveness = template.index("livenessProbe")

    assert "/readyz" in template[readiness:liveness]
    assert "/healthz" in template[liveness:]


def test_the_templated_agent_has_a_liveness_probe() -> None:
    """A wedged agent leaves capacity nothing can observe, release, or stop paying for."""

    assert "livenessProbe" in (CHART / "templates" / "cell-agent.yaml").read_text(encoding="utf-8")


def test_the_chart_documents_rendering_without_claiming_cluster_proof() -> None:
    readme = (CHART / "README.md").read_text(encoding="utf-8")

    assert "helm lint" in readme
    assert "helm template" in readme
    assert "do not prove Kubernetes" in readme


def test_ci_ready_validation_script_pins_and_requires_helm_and_kubeconform() -> None:
    script_path = ROOT / "scripts" / "validate-deployment.sh"
    script = script_path.read_text(encoding="utf-8")

    assert script_path.stat().st_mode & 0o111
    assert 'HELM_VERSION="v3.18.6"' in script
    assert 'KUBECONFORM_VERSION="v0.7.0"' in script
    assert '"${HELM_BIN}" lint' in script
    assert '"${HELM_BIN}" template' in script
    assert script.count("cellAgent.enabled=true") == 2
    assert "cellAgent.fleetStateUri=gs://" in script
    assert "cellAgent.capacityBackend=gcp" in script
    assert "cellAgent.capacityBackend=kubernetes" in script
    assert '"${KUBECONFORM_BIN}" -strict -summary' in script
    assert "ignore-missing-schemas" not in script
    assert "missing required" in script


@pytest.mark.skipif(HELM is None, reason="helm binary is not installed")
def test_helm_binary_lints_and_renders_the_chart() -> None:
    """Use the real Go template engine when it is available; this is not cluster proof."""

    assert HELM is not None
    lint = subprocess.run([HELM, "lint", str(CHART)], check=False, capture_output=True, text=True)
    assert lint.returncode == 0, lint.stdout + lint.stderr

    rendered = subprocess.run(
        [
            HELM,
            "template",
            "eyvan",
            str(CHART),
            "--namespace",
            "eyvan-system",
            "--set",
            "serviceMonitor.enabled=true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rendered.returncode == 0, rendered.stdout + rendered.stderr
    documents = [document for document in yaml.safe_load_all(rendered.stdout) if document]
    kinds = {document["kind"] for document in documents}
    assert {"ConfigMap", "Deployment", "Job", "NetworkPolicy", "PodDisruptionBudget", "ServiceMonitor"} <= kinds
