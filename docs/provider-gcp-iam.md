# GCP IAM: what Eyvan needs, and why

Eyvan's GCP driver runs as **whatever principal `gcloud` is authenticated as**
(`gcloud auth print-access-token` decides). This page lists every GCP
permission Eyvan exercises, so a project owner can grant the least privilege
that actually works — and so a missing permission fails fast with an
actionable message instead of a raw 403 half way through bootstrap.

## The one grant that bites: Secret Manager binding

Every `eyvan-native` cell gets a **cell-scoped Secret Manager secret** holding
its worker token. Eyvan creates the secret, rotates a version per generation,
and binds `roles/secretmanager.secretAccessor` for exactly the cell's service
account. Creating secrets is covered by `roles/editor`; **binding IAM is
not**.

If the active account lacks `secretmanager.secrets.setIamPolicy`, bootstrap
now fails **before anything is created**, with a message naming the missing
permission. The fix is a one-time project-level grant from an owner:

```bash
gcloud projects add-iam-policy-binding <project> \
  --member=user:<your-account> \
  --role=roles/secretmanager.admin
```

`roles/secretmanager.admin` is a standard role containing the required secret
administration permissions; a purpose-built custom role may be narrower.
A role label such as Editor is not proof of effective permission: conditional
grants, deny policies and other IAM bindings also matter.

Check effective permissions rather than inferring them from historical
bootstrap success.

Eyvan checks both `secretmanager.secrets.create` and
`secretmanager.secrets.setIamPolicy` up front via the read-only
Cloud Resource Manager project `testIamPermissions` API (cached per project
per process). Secret Manager's corresponding method accepts individual secret
resources, not a project path. This distinction is regression-tested so an
unsupported endpoint cannot silently turn every preflight into "unknown".

## Full permission inventory

| GCP surface | What Eyvan calls | Permission needed | Covered by |
|---|---|---|---|
| Secret Manager | `secrets create/describe/list/delete`, `versions add`, `add-iam-policy-binding` | `secretmanager.secrets.create`, `secretmanager.secrets.setIamPolicy` | editor + **secretmanager.admin** |
| Compute Engine | `instances create/describe/delete`, `firewall-rules describe/create` (native worker ingress) | `compute.instances.*`, `compute.firewalls.*` | editor |
| Cloud TPU | `compute tpus queued-resources create/describe/delete` | `tpu.queuedResources.*`, `compute.networks.*` | editor |
| Cloud Storage | bucket-level `add-iam-policy-binding` granting `roles/storage.objectViewer` + `roles/storage.objectCreator` to the cell service account | `storage.buckets.getIamPolicy` / `setIamPolicy` | editor |
| Resource Manager | `projects describe` (resolve the project number into the default Compute service-account email) | `resourcemanager.projects.get` | editor |

Notes:

- The artifact bucket itself (e.g. `gs://eyvan-<project>`) must exist and be
  writable by the account; `eyvan runtime init` documents this.
- **Spot is not a permission**: preemptible/Spot capacity needs quota in the
  target zone, which is a quota request, not an IAM change. Eyvan never falls
  back to on-demand silently.
- Nothing above requires `resourcemanager.projects.setIamPolicy` — Eyvan never
  changes project-level IAM policy; only per-secret and per-bucket bindings on
  resources it created or was pointed at.
- The preflight check treats an unreachable project IAM API as *unknown*,
  not *denied*: bootstrap proceeds and fails closed with the provider's own
  diagnostic, per the "unknown is a state" rule.
