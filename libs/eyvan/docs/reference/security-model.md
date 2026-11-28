# Eyvan security model

This is a source-backed reference to controls implemented in the current tree. It
is not a threat-model substitute: deployment topology, credential custody, and
network policy remain operator responsibilities.

## 1. Authentication and authorization

HTTP authentication is implemented by `eyvan/server/api/auth.py`.

* **OIDC.** `OIDCTokenVerifier` discovers JWKS from the configured issuer (or
  uses `oidc_jwks_url`) and verifies only `RS256`/`ES256` tokens with the
  configured issuer and audience and a required `exp` claim. The normalized
  identity is the token `sub`, optional `name`, and `eyvan_grants`.
* **Bootstrap.** A configured bootstrap bearer secret authenticates exactly the
  fixed subject `eyvan-bootstrap` and grants global `fleet_admin`;
  `bootstrap_principal` does not let the bearer choose a subject or role.
* **Development auth.** When `dev_auth_enabled` is on, requests use the
  `X-Eyvan-User`, `X-Eyvan-Org`, `X-Eyvan-Project`, and `X-Eyvan-Roles` headers,
  but a generated local-development credential is still required once issued.
  Automatic dashboard identity is used only for browser same-origin requests
  with a loopback-literal `Host` and a non-navigation fetch. These checks help
  against cross-site requests and DNS rebinding; headers are not proof of
  identity for arbitrary clients. Development auth is therefore not a network
  authentication scheme.
* **Durable grants.** `authenticated_principal` unions the OIDC/bootstrap
  grants with grants resolved from the control plane on every request. This
  makes a durable grant revocation effective on the next request; claim grants
  remain honored.

`eyvan/server/service/authz.py:Authorizer` evaluates explicit grants; there is
no implicit project inheritance. Roles and effective permissions are:

| Role | Scope | Permissions |
|---|---|---|
| `fleet_admin` | global | all permissions |
| `org_admin` | organization | read, submit, operate, manage project, manage own capacity |
| `project_admin` | project | read, submit, operate, manage project |
| `project_operator` | project | read, submit, operate |
| `project_submitter` | project | submit only |
| `project_viewer` | project | read only |

`MANAGE_FLEET` is distinct from an organization’s `MANAGE_OWN_CAPACITY`.
Route and service enforcement calls `Authorizer.require`; failures are
authorization errors rather than authentication successes.

## 2. Capability tokens

`eyvan/artifacts/_capability_tokens.py:sign_claims` and `verify_signature` use
HMAC-SHA-256 over a canonical, sorted JSON payload. The signed wire form is
`version.payload.signature`; the version prefix is itself signed and verification
requires the expected prefix, separating audiences/capability families.
Signing material must be at least 32 bytes (`require_secret`).

Managed artifact tokens (`eyvan/artifacts/managed.py`) bind an action, artifact
kind/object, organization, project, issuance time, and optionally workload ID.
They expire by default (`artifact_token_ttl_s`, seven days in settings), with
bounded caller-selected TTLs. A non-expiring token is permitted only for a
workload-bound write. Secret-read capabilities use the separate one-hour
`secret_token_ttl_s` default: the attempt/state binding is the revocation
mechanism because a value cannot be unread once disclosed. Capability validity
is also checked against the relevant artifact or workload state; possession of
a correctly signed token is not a general API grant.

## 3. Trust boundaries

The control plane authenticates API callers and authorizes tenant-scoped
operations. It then crosses separate boundaries to:

* the **cell agent**, which receives control-plane commands and interacts with
  provider capacity;
* the **native worker**, which executes workload code and receives only the
  capabilities/material needed for its attempt; and
* **external webhook receivers**, to which the plane makes outbound POSTs.

A worker is an execution boundary, not a trusted caller. Artifact transfer
tokens are scoped and signed; secret references are resolved at launch under
project policy. Container validation rejects protected Eyvan credential mount
sources (`eyvan/records.py:NativeContainerMount`). Operators must protect the
control-plane database, provider identities, agent/worker hosts, artifact
stores, and TLS/private-key files; source controls do not make a compromised
worker or host trustworthy.

## 4. Secret lifecycle

`eyvan/secrets.py` persists and transports **references**, not secret values.
Accepted references are canonical `file:///absolute/path` and
`gcp-secret://project/name[/version]`; ambiguous encodings, traversal segments,
queries/fragments, unknown schemes, and malformed authorities are rejected.
Project policy matches scheme, authority, and complete path segments rather
than raw string prefixes.

`FileSecretStore` opens each component with directory descriptors and
`O_NOFOLLOW`, requires a regular file, and bounds reads to the configured
maximum (one MiB by default). `GcpSecretManagerStore` invokes the fixed
`gcloud secrets versions access` command, validates canonical base64url output,
and normalizes failures. `redact_secret_material` removes known values and
secret capability-looking strings from text, but redaction is not a promise
that a value intentionally printed by workload code can be recovered or made
safe. Operators must set narrow file roots and protect provider credentials.

## 5. mTLS staged rollout

`eyvan/identity/mtls.py` models two client-to-server edges:
control plane → agent → worker. Every surface defaults to `off`; mTLS is not
implicitly enabled. Modes are `off`, `permissive` (verify a certificate when
present but accept clients without one), and `required` (reject clients without
a trusted certificate). TLS contexts require a trust bundle and use TLS 1.2 or
newer; server contexts also require a certificate and private key. Eyvan passes
key paths to OpenSSL and does not hold key bytes.

Use `enablement_plan` / `validate_rollout_step`: distribute trust anchors and
key pairs first, make servers permissive from the inside out, enable clients,
verify connectivity, then make servers required (worker, then agent). The
validator rejects requiring before the client presents and rejects changing
both in the same restart step. Reversal is servers permissive first, then
clients off. This does not configure TLS for the main external HTTP listener.

## 6. Credential rotation

`eyvan/identity/rotation.py:RotatingCredential` rotates signing material under
a caller-held fence. Only one rotation may be pending; a committed rotation
makes the new generation sign while the old generation verifies until an
absolute `not_after`, after which it verifies nothing regardless of token
claims. Material and diagnostic representations are deliberately not logged;
all accepted material is at least 32 bytes.

If the outcome cannot be observed, `mark_inconclusive` leaves old verification
working but blocks new signing and further rotation. There is no automatic
retry: a human must resolve the same fence as committed or abandoned. Operators
must perform the provider installation/distribution step out of band and
resolve inconclusive outcomes deliberately.

## 7. Webhooks and SSRF

`eyvan/server/service/webhooks.py:validate_target` accepts only HTTP(S) URLs.
Unless `webhook_allow_private_targets` is explicitly enabled, literal or
resolved loopback, private, link-local, and reserved addresses are refused;
all DNS answers are checked, not just the first. Delivery re-checks at send
time and pins the vetted DNS address (`eyvan/server/controller/webhook_delivery.py`),
with SNI/Host preserved for HTTPS and redirects disabled. An unresolvable name
is accepted at creation and checked again at delivery, so DNS health at
subscription time is not treated as proof of safety.

Each body carries `X-Eyvan-Signature: sha256=<hex>` (HMAC-SHA-256 using the
subscription secret) and `X-Eyvan-Sequence`. Delivery is bounded to ten
seconds, at-least-once, and disabled after ten consecutive failures. **The
private-target exception is real:** enabling `webhook_allow_private_targets`
deliberately removes this SSRF address policy and is suitable only where the
operator controls the network and receiver.

## 8. Callable execution

Callable payloads use `cloudpickle` (`eyvan/worker/runtime/callable_protocol.py`).
`load_callable` verifies format, Python version, SHA-256, and—only when a key is
provided—an HMAC signature, before deserializing. Deserialization executes
pickle opcodes, so the control plane requires an explicit project trust policy
via `CallableProjectTrustPolicy`; trusted entries may be scoped as `org/project`
or (less safely) bare project IDs. Untrusted projects are rejected. This is a
code-execution trust decision, not sandboxing: trusted callable code and its
imports must be treated as able to act with the worker/container’s authority.

Standalone SDK/API callables are unsigned: the submission boundary rejects
caller signing keys and artifact signatures. Optional HMAC support belongs to
the low-level callable protocol for custom integrations; the controller does
not distribute a producer/verifier key. A digest detects corrupt bytes but is
not caller authentication and does not make an untrusted callable safe.
Operators must restrict trusted projects and protect the worker/container's
authority.

## 9. Supply-chain controls

The OCI release workflow (`.github/workflows/eyvan-oci-release.yaml`) has a
read-only pull-request lane that builds locally, smoke-tests entrypoints,
generates SPDX SBOMs, and gates Trivy High/Critical findings. Its manual release
lane requires protected `main`, immutable releases, and recorded environment
approval before registry writes; it builds and qualifies exact digests, signs
with keyless cosign identity, and records receipts. Actions and release tools
are pinned by commit/checksums in `oci/release-lock.json`; no image is tagged
`latest`. The Python publish workflow (`.github/workflows/publish.yaml`) is tag
/version checked, builds with hash-pinned requirements, smoke-tests the wheel,
and uses PyPI trusted publishing rather than a stored PyPI token.

These are workflow gates, not a guarantee that an operator deploys only the
qualified digest. Verify image provenance, receipts, registry policy, and
runtime configuration when deploying.

## 10. Known limitations and operator duties

* The control-plane listener supports built-in TLS via `EYVAN_TLS_CERT_FILE`
  and `EYVAN_TLS_KEY_FILE`, configured together. Separately, native agent/worker mTLS
  uses the trust bundle, certificate, key, and rollout modes described in section 5;
  these internal modes default to `off` and do not configure the external
  control-plane listener. A network-bound control plane refuses plaintext HTTP
  unless explicitly acknowledged with `EYVAN_INSECURE_PLAINTEXT_BIND=1` or
  `eyvan-server --insecure-plaintext`, including behind external TLS termination.
  External TLS termination and network access controls remain a deployment
  alternative; protect any plaintext hop and keep development auth local.
* Artifact capability transfer over plain HTTP is refused unless the operator
  explicitly enables `allow_insecure_artifact_http`. Enabling it exposes
  bearer-like capability tokens to the transport and is an operator risk, not
  encryption.
* Webhooks may use `http://`; signing authenticates the body but does not
  provide confidentiality or transport integrity against an observer. Prefer
  HTTPS and protect webhook secrets.
* Optional callable signing is not equivalent to mandatory provenance or
  sandboxing, as described above.
* Protect OIDC issuer/JWKS configuration, bootstrap and signing material,
  secret-provider credentials, databases, logs, and host/container runtime
  permissions. Rotate credentials using the fenced state machine, review
  durable grants, keep private-target webhook allowance off unless required,
  and deploy only reviewed, digest-qualified artifacts.

## Source index

Primary implementation references: `eyvan/server/api/auth.py`,
`eyvan/server/api/server_bind.py`, `eyvan/server/api/server_startup.py`,
`eyvan/worker/agent/cli.py`, `eyvan/worker/runtime/native_worker.py`,
`eyvan/records.py`, `eyvan/server/service/authz.py`,
`eyvan/artifacts/_capability_tokens.py`, `eyvan/artifacts/managed.py`,
`eyvan/secrets.py`, `eyvan/identity/mtls.py`, `eyvan/identity/rotation.py`,
`eyvan/server/service/webhooks.py`,
`eyvan/server/controller/webhook_delivery.py`,
`eyvan/server/service/callable_trust.py`,
`eyvan/worker/runtime/callable_protocol.py`,
`.github/workflows/eyvan-oci-release.yaml`,
`.github/workflows/publish.yaml`, and `oci/release-lock.json`.

The documented behavior is intentionally limited to those sources; absence of
a control from this page should not be read as evidence that an unlisted
control exists.
