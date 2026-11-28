# The Eyvan Helm chart

The chart installs the control plane, platform routing ConfigMap, schema-policy
migration hook, and, when explicitly enabled, a separately built cell agent. In
Kubernetes capacity mode it also installs namespaced worker RBAC. Release
resources use `.Release.Namespace`; an optional worker namespace defaults to it.

## Prerequisites and Secret contract

The `pre-install` hook runs before ordinary chart resources. Create the namespace and Secret before the first install; specifically, create the release
namespace and `existingSecret`, then keep `--create-namespace` on install
commands. `existingSecret` defaults to `eyvan-secrets` and must contain:

| key | consumer |
|---|---|
| `database-url` | migration and control plane |
| `artifact-token-key` | control plane; identical on every replica |
| `agent-shared-secret` | control plane and cell agent |
| `metrics-token` | control plane and optional `ServiceMonitor` |
| `native-worker-root-secret` | cell agent; distinct from the agent secret |

The chart creates neither credentials nor PostgreSQL. Regenerating
`artifact-token-key` invalidates issued capabilities, and tying a database to a
release makes `helm uninstall` a data-loss operation.

## Build and qualify immutable images

The only release path is the manual `Eyvan immutable OCI release` workflow at
an exact full candidate SHA already reachable from protected `main`. Its job is
guarded by the `eyvan-oci-release` GitHub environment. A pull request gets a
separate read-only lane: it never receives an OIDC token, registry login, or
write permission, builds with `push: false`/`load: true`, runs every entrypoint's
`--help`, emits SPDX, and applies the same High/Critical scan policy.

[`oci/release-lock.json`](../../../oci/release-lock.json) is canonical JSON and
pins the sole `linux/amd64` base manifests, host-tool downloads and GitHub Action
commits. [`oci/high-critical-allowlist.json`](../../../oci/high-critical-allowlist.json)
is initially empty. Validate both and the Dockerfiles before using a value:

```bash
python3 scripts/verify_eyvan_oci_release.py validate-policy \
  --lock oci/release-lock.json \
  --allowlist oci/high-critical-allowlist.json \
  --repository-root .
```

Every Dockerfile expects the repository root (the package root) as its build
context and has no base-image, source, revision, or version default. The
examples below are local mechanics, **not qualification**: only the protected
workflow records the registry-reported index/platform digests, scanner and DB
snapshot, SBOM attestations, keyless signatures/Rekor inclusion, provenance,
chart/render, wheel/sdist, and signed canonical receipt. Copy base arguments
exactly from the validated release lock; do not substitute a tag or digest-only
reference.

```bash
export EYVAN_SOURCE=https://github.com/erfanzar/eyvan
export EYVAN_REVISION="$(git rev-parse HEAD)" # must be the full 40-character SHA
export EYVAN_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
IFS=$'\t' read -r EYVAN_NODE_IMAGE EYVAN_RUNTIME_IMAGE EYVAN_UV_IMAGE EYVAN_WORKER_BASE_IMAGE \
  < <(python3 - <<'PY'
import json

with open("oci/release-lock.json", encoding="utf-8") as handle:
    images = json.load(handle)["images"]
values = [
    images["control-plane"]["buildArgs"]["EYVAN_NODE_IMAGE"],
    images["control-plane"]["buildArgs"]["EYVAN_RUNTIME_IMAGE"],
    images["control-plane"]["buildArgs"]["EYVAN_UV_IMAGE"],
    images["worker"]["buildArgs"]["EYVAN_WORKER_BASE_IMAGE"],
]
print("\t".join(values))
PY
)
export EYVAN_NODE_IMAGE EYVAN_RUNTIME_IMAGE EYVAN_UV_IMAGE EYVAN_WORKER_BASE_IMAGE

# From the repository root (use the same validated values):
docker build --platform linux/amd64 -f Dockerfile \
  --build-arg EYVAN_NODE_IMAGE --build-arg EYVAN_RUNTIME_IMAGE --build-arg EYVAN_UV_IMAGE \
  --build-arg EYVAN_SOURCE --build-arg EYVAN_REVISION --build-arg EYVAN_VERSION \
  -t "$EYVAN_CONTROL_PLANE_REPOSITORY:$BUILD_ID" .
docker build --platform linux/amd64 -f Dockerfile.agent \
  --build-arg EYVAN_RUNTIME_IMAGE --build-arg EYVAN_UV_IMAGE \
  --build-arg EYVAN_SOURCE --build-arg EYVAN_REVISION --build-arg EYVAN_VERSION \
  -t "$EYVAN_AGENT_REPOSITORY:$BUILD_ID" .
docker build --platform linux/amd64 -f Dockerfile.worker \
  --build-arg EYVAN_WORKER_BASE_IMAGE --build-arg EYVAN_UV_IMAGE \
  --build-arg EYVAN_SOURCE --build-arg EYVAN_REVISION --build-arg EYVAN_VERSION \
  -t "$EYVAN_WORKER_REPOSITORY:$BUILD_ID" .
```

The workflow pushes each `sha-$FULL_SHA` artifact once, scans the exact digest
(including unfixed findings), signs and verifies it with the exact workflow
identity and GitHub OIDC issuer, then copies that same index digest to the
one-use `v$VERSION` tag. It never rebuilds during promotion and never creates
`latest`. The Helm overlay preserves three independent values:
`image`, `agentImage`, and `workerImage`; the rendered release must contain the
three corresponding `repository@sha256:...` references. Consume a release only
after verifying its signed receipt, signature/Rekor inclusion, and all image,
chart, render, package, and attestation digests. Copy those exact references;
do not rebuild, retag, resolve a version tag at deploy time, or use `latest`.

The chart has no `imagePullSecrets` value and does not create registry
credentials. For a private registry, pre-create a pull Secret in the release
namespace and arrange an approved image-pull credential on each workload's
ServiceAccount (or apply an equivalent reviewed admission policy) before
installing; otherwise pods may remain Pending with image-pull errors. Repeat
this for the separately configured worker namespace and worker ServiceAccount
when Kubernetes capacity mode is used. Validate pullability in the target
cluster; local render success does not test registry authorization.

A receipt deliberately records `deploymentApplied=false`, hardware evidence as
`not-run`, and `linux/arm64`, `linux/ppc64le`, and `linux/s390x` as unverified.
Static tests and hosted `linux/amd64` image builds do not prove Kubernetes
admission, image pullability from a target cluster, database migration,
provider behavior, accelerator compatibility, TPU/GPU runtime behavior, or
performance. Qualify those separately in the exact target environment without
editing the release receipt.

## Listener TLS prerequisite

Before either install example, create a TLS Secret in the release namespace
containing `tls.crt` and `tls.key` for the control-plane listener. The chart
references it through `controlPlane.tls.existingSecret`; it does not create
certificates. A network-bound plane refuses startup without listener TLS or
an explicit `controlPlane.insecurePlaintextBind=true` acknowledgement. External
TLS termination alone does not bypass that refusal; protect any plaintext hop
when using the reviewed exception.

## GCP capacity mode

`cellAgent.enabled` defaults to `false`; set an external mTLS agent endpoint or
enable the in-cluster owner deliberately. When enabled, rendering fails unless
`cellAgent.replicas` is exactly `1` and `cellAgent.fleetStateUri` is a canonical
`gs://bucket/non-empty-object` URI. Its Deployment uses `Recreate`, so an upgrade
cannot overlap two owners of the same fleet-state object.

`cellAgent.capacityBackend` is explicit and defaults to `gcp`. Native VM
bootstrap downloads the wheel and wheelhouse only from the configured GCS object
generations and verifies each file's SHA-256 before extraction or installation.
Record both values after upload (for example from your release metadata or an
object describe operation); a content-addressed object name alone does not
replace the check.

```bash
helm upgrade --install eyvan ./deploy/helm/eyvan \
  --namespace eyvan-system --create-namespace --wait --wait-for-jobs \
  --set-string image.repository="$EYVAN_CONTROL_PLANE_REPOSITORY" \
  --set-string image.digest="$EYVAN_CONTROL_PLANE_DIGEST" \
  --set-string controlPlane.tls.existingSecret=eyvan-listener-tls \
  --set-string agentImage.repository="$EYVAN_AGENT_REPOSITORY" \
  --set-string agentImage.digest="$EYVAN_AGENT_DIGEST" \
  --set-string controlPlane.publicUrl=https://eyvan.example.com \
  --set-string controlPlane.artifactStoreUri=gs://your-bucket/eyvan \
  --set-string controlPlane.oidc.issuer=https://idp.example.com \
  --set-string controlPlane.oidc.dashboardClientId=eyvan-dashboard \
  --set cellAgent.enabled=true \
  --set-string cellAgent.capacityBackend=gcp \
  --set-string cellAgent.fleetStateUri=gs://your-state-bucket/eyvan/fleet-state.json \
  --set-string cellAgent.nativeWorkerPackage="$EYVAN_WORKER_WHEEL_URI" \
  --set-string cellAgent.nativeWorkerPackageSha256="$EYVAN_WORKER_WHEEL_SHA256" \
  --set-string cellAgent.nativeWorkerPackageGeneration="$EYVAN_WORKER_WHEEL_GENERATION" \
  --set-string cellAgent.nativeWorkerWheelhouse="$EYVAN_WORKER_WHEELHOUSE_URI" \
  --set-string cellAgent.nativeWorkerWheelhouseSha256="$EYVAN_WORKER_WHEELHOUSE_SHA256" \
  --set-string cellAgent.nativeWorkerWheelhouseGeneration="$EYVAN_WORKER_WHEELHOUSE_GENERATION"
```

For GKE Workload Identity, set annotations as values rather than embedding cloud
credentials. For example:

```yaml
serviceAccounts:
  controlPlane:
    annotations:
      iam.gke.io/gcp-service-account: eyvan-control-plane@PROJECT_ID.iam.gserviceaccount.com
  cellAgent:
    annotations:
      iam.gke.io/gcp-service-account: eyvan-agent@PROJECT_ID.iam.gserviceaccount.com
```

## Kubernetes capacity mode

Kubernetes mode runs one immutable `Dockerfile.worker` artifact per worker in an
Indexed Job. The worker namespace defaults to the release namespace. If
`cellAgent.kubernetes.workerNamespace` differs, create that namespace before the
release; the chart deliberately does not own its lifecycle. Configure a durable,
agent-unique `cellAgent.fleetStateUri`; the example uses GCS and the cell-agent
Workload Identity annotation supplies its cloud identity.

```bash
helm upgrade --install eyvan ./deploy/helm/eyvan \
  --namespace eyvan-system --create-namespace --wait --wait-for-jobs \
  --set-string image.repository="$EYVAN_CONTROL_PLANE_REPOSITORY" \
  --set-string image.digest="$EYVAN_CONTROL_PLANE_DIGEST" \
  --set-string controlPlane.tls.existingSecret=eyvan-listener-tls \
  --set-string agentImage.repository="$EYVAN_AGENT_REPOSITORY" \
  --set-string agentImage.digest="$EYVAN_AGENT_DIGEST" \
  --set cellAgent.enabled=true \
  --set-string cellAgent.capacityBackend=kubernetes \
  --set-string cellAgent.fleetStateUri=gs://your-state-bucket/eyvan/kubernetes-fleet-state.json \
  --set-string cellAgent.kubernetes.workerNamespace=eyvan-workers \
  --set-string workerImage.repository="$EYVAN_WORKER_REPOSITORY" \
  --set-string workerImage.digest="$EYVAN_WORKER_DIGEST"
```

The chart creates `eyvan-native-worker` in the worker namespace and a Role plus
RoleBinding for the release namespace's `eyvan-cell-agent`. The Role can create,
get, and delete Jobs, Services, and Secrets, and read Pods/events. It grants no
cluster-wide role, pod exec, pod logs, or arbitrary mutation. Optional Kueue
ClusterQueue inspection is cluster scoped and therefore is not granted by the
stock Role; opt-in Kueue needs a separately reviewed read grant. Worker
service-account annotations are configurable at
`serviceAccounts.worker.annotations`.

Every generated worker pod carries `eyvan.dev/api-client=true`. For a separate
worker namespace, the API ingress peer ANDs that pod selector with Kubernetes'
automatic exact `kubernetes.io/metadata.name=<worker-namespace>` namespace label;
the chart validates the configured namespace as the exact DNS-label value. A
same-release-namespace client still needs the narrow pod label, and no bare
namespace selector admits unrelated pods.

Every worker image is rendered as `repository@sha256:...`; Kubernetes mode fails
template rendering when `workerImage.digest` is absent or malformed. There is no
worker tag fallback.

## Schema changes are maintenance, not rolling rollout

The hook inspects the database before calling Alembic:

* a database with no tables may be initialized;
* a database already at the packaged head proceeds as a no-op, so ordinary
  same-head rollouts remain usable;
* a non-empty schema at another revision is refused by default.

`--atomic` cannot reverse a committed database migration. Automatic rollback may
instead restart the old image against the new schema. Do not enable maintenance
acknowledgement as a standing value, and do not infer application/schema
N/N-1 compatibility; no such compatibility contract is implemented.

For a release that actually changes the schema:

1. Quiesce every control-plane replica and any other database writer.
2. Create and verify a backup, then rehearse restoring it to an isolated database.
3. Run `eyvan-migrate --check` from the candidate image. A refusal on the
   non-empty old schema is expected until maintenance is acknowledged.
4. Set `migration.maintenanceAcknowledged=true` for this one release and run the
   migration/upgrade without relying on `--atomic` as database rollback.
5. Verify the new revision and application health, then reset the value to false
   and resume writers.

If application rollout fails after migration, hold writers quiesced and repair
forward with a corrected image or migration. Never run `alembic downgrade` (or
any schema downgrade) as Helm rollback. Backup restore is an operator-controlled
disaster-recovery action, not an automatic release rollback.

## Platform config and metrics

When `cellAgent.enabled=true`, the default route is
`http://eyvan-cell-agent.<release-namespace>.svc.cluster.local:8090`. With the
safe disabled default, set `platform.defaultAgentEndpoint` to the external mTLS
agent. The control-plane pod template includes a checksum of the ConfigMap
content, so Helm changes to `platform.*` trigger a rollout.

External Secret contents and ServiceAccount annotations are not part of the pod
template checksum. After changing either, explicitly restart and wait for every
affected control-plane or cell-agent Deployment so no pod keeps the old identity
or credential.

`serviceMonitor.enabled` remains false because its CRD may not exist. When
explicitly enabled, it reads the required `metrics-token` key. A scraper in the
release namespace needs `eyvan.dev/metrics-scraper=true` on its pod; a
cross-namespace scraper needs that label on both namespace and pod. `/metrics`
and the API share TCP 8080, so NetworkPolicy cannot distinguish their paths;
the metrics bearer and API authentication remain the authorization boundary.

The monitor targets the control-plane Service directly, not the production
Ingress. When `controlPlane.tls.existingSecret` is set, it uses HTTPS and
rendering fails unless all three `serviceMonitor.tls` values below are non-empty.
The CA Secret must already exist in the release namespace and its selected key
must contain only the public CA certificate bundle, **never a private key**.
`serverName` must match a SAN in the listener's serving certificate and identify
the intended control plane; configure that certificate and name together. A
public ingress hostname is not automatically the correct identity for the
Service target. The chart never disables certificate verification.

```yaml
serviceMonitor:
  enabled: true
  tls:
    caSecretName: eyvan-metrics-ca
    caSecretKey: ca.crt
    serverName: eyvan-control-plane.eyvan-system.svc.cluster.local
```

This example assumes that exact Service DNS name is in the serving certificate's
SANs. The chart does not issue certificates or create the CA Secret. With the
explicit plaintext listener (`controlPlane.insecurePlaintextBind=true` and no
TLS Secret), the monitor uses HTTP without `tlsConfig`; ingress TLS termination
does not encrypt this direct Service scrape. A disabled monitor requires no
scraper trust values, even when the API listener uses TLS.

## Post-deployment checks

After `helm upgrade --install` (and after any migration), do not treat
`--wait` as the complete qualification. Verify the control-plane rollout and,
when enabled, the single-owner cell-agent rollout:

```bash
kubectl -n eyvan-system rollout status deployment/eyvan-control-plane
kubectl -n eyvan-system rollout status deployment/eyvan-cell-agent
kubectl -n eyvan-system get pods -o wide
```

Through the production ingress, check `/healthz` and `/readyz`; the latter
must be ready against the intended database. Make an authenticated API request
through the configured OIDC path, confirm an unauthenticated request is
rejected, and scrape `/metrics` with the `metrics-token` bearer. Confirm the
scrape is accepted and current, and inspect control-plane and agent logs for
401s, database errors, image-pull failures, and registration/heartbeat errors.
Finally confirm the agent registers and execute a harmless capacity/agent
operation. In Kubernetes mode, verify a worker Job and pod start with the
expected worker digest. Record these checks alongside the signed receipt.

## Mandatory local render/schema validation

The CI-ready validation command fails when either tool is absent or has the
wrong version. It runs pinned `helm lint`, `helm template`, and kubeconform over
the safe disabled default, qualified GCP, and Kubernetes namespace/RBAC modes,
then runs strict kubeconform over every render and the raw
manifests:

```bash
scripts/validate-deployment.sh
```

It pins Helm v3.18.6 and kubeconform v0.7.0. It does not silently skip missing
schema tools. The separate artifact smoke builds and installs both distributions
in clean virtual environments outside the source tree:

```bash
scripts/smoke-package-artifacts.sh
```

These checks do not prove Kubernetes admission, image pullability, migration
completion, readiness, agent registration, an authenticated scrape, or any
provider/cluster behavior. Those require the exact qualified artifacts and a
disposable environment.
