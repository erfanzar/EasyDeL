# Security model

The [working threat model and release checklist](design/0005-threat-model.md)
separates implemented controls from remaining qualification gaps. It is not an
independent security-review approval.

## Production boundary decisions

These accepted design records have distinct implementation and qualification
states and must not be read as provider claims:

- [artifact scope and generation-fenced deletion](design/0001-scoped-artifact-lifecycle.md)
  aligns the metadata key with the tenant-scoped byte store and is
  implemented at the host/PostgreSQL boundary; live migration, object-store,
  and destructive collection evidence is still required;
- [log rank and cursor compatibility](design/0002-global-log-ranks-and-cursors.md)
  treats an opaque cursor as untrusted state, not as a credential; the
  narrower M1.3 global-rank/cursor/archive surface is implemented, but strict
  topology-bound rejection and the full archive manifest remain pending along
  with multi-host hardware evidence;
- [service health and metrics ownership](design/0003-service-health-and-metrics.md)
  remains implementation-pending and keeps requests, responses, and raw
  application telemetry out of Eyvan; and
- [federation trust and witnessed ownership](design/0004-federation-consistency.md)
  remains implementation-pending and requires external PKI and an external
  quorum-backed lease/CAS service rather than an Eyvan-built CA or consensus
  protocol.

## The authority boundary

The plane is the authority for placement, leasing, and reap. Agents own
machines and execute what the plane leases to them. Your callable runs on
worker hosts under the native runtime — it is workload code, not trusted
code. Standalone callable execution requires explicitly trusted projects
and digest-verified unsigned inputs; least privilege on the worker; no plane
credentials in the workload environment. Digests do not authenticate a submitter
or make an untrusted pickle safe.

## Who a request is

- **SDK → plane:** standalone submissions contain an unsigned callable artifact
  plus a workload record. The standalone SDK/API rejects caller signing keys
  and artifact signatures, including on loopback planes. Artifact integrity is
  verified by content digest; admission requires an explicitly trusted project.
  Optional low-level HMAC/custom-submitter helpers do not establish controller
  key distribution or end-to-end standalone signature authentication: the
  standalone controller does not distribute a producer/verifier key.
- **Plane → agent:** bearer authentication with
  `EYVAN_AGENT_SHARED_SECRET`. A credential mismatch is rejected with HTTP 401.
  Restore the matching shared secret before restarting the plane.
  `--allow-unauthenticated` on the agent exists for loopback local development
  only.
- **Agent → native workers:** a separate root secret
  (`EYVAN_NATIVE_WORKER_ROOT_SECRET`) fences worker initialization.

## Capability tokens and secrets

| secret | who reads it | what it protects |
|---|---|---|
| `EYVAN_AGENT_SHARED_SECRET` | plane, agent | plane↔agent API |
| `EYVAN_NATIVE_WORKER_ROOT_SECRET` | agent | native worker bring-up |
| `EYVAN_ARTIFACT_TOKEN_KEY` | plane | artifact store tokens |
| `EYVAN_METRICS_TOKEN` | metrics clients | metrics scrape auth |
| `EYVAN_BOOTSTRAP_TOKEN` | plane and admin clients | authenticates a global fleet-admin principal on every request |

Treat the bootstrap token as a full administrator credential, not a one-time
provisioning token. It remains usable for authenticated requests while configured.

Secrets enter via environment or the runtime profile
(`~/.eyvan/gcp-native-runtime.json`, written `0600`), never via fleet configs
or the repository. The full surface is tabulated in
[env-surface](env-surface.md).

## What jobs can do

A callable may: use its leased hardware, read its inputs from the artifact
store by digest, and write its result manifest. It may not: name other
tenants' artifacts or hold plane credentials. The fleet's idle timeout
governs eligible unleased surplus capacity, subject to demand, protection,
and minimum-capacity policy; it is not a workload runtime or cost deadline.
Active lease expiry/quarantine and fenced retirement are separate lifecycle
controls. Use explicit workload cancellation to request termination; an idle
capacity timer does not cancel a valid leased workload. Destructive artifact
collection is disabled unless explicitly enabled (see [artifact-safety](artifact-safety.md)).

## Exposure

The loopback plane binds `127.0.0.1` and requires none of the off-loopback
secrets; binding wider is refused at startup until
`EYVAN_TLS_CERT_FILE` and `EYVAN_TLS_KEY_FILE` are set,
`EYVAN_REPOSITORY_BACKEND=postgres`, `EYVAN_METRICS_TOKEN`, and
`EYVAN_ARTIFACT_TOKEN_KEY` are set (`eyvan doctor --server-host 0.0.0.0`
enumerates the refusals before you start). The TLS refusal can only be
overridden by a reviewed, explicit decision — `EYVAN_INSECURE_PLAINTEXT_BIND=1`
or `eyvan-server --insecure-plaintext` — never by default; plaintext on a
network port is how bearer tokens, bootstrap credentials, and capability
tokens get stolen. The listener TLS terminates the browser/API edge.

Cell-agent and native-worker listeners support explicit mutual TLS through
`EYVAN_MUTUAL_TLS_MODE`, `EYVAN_MUTUAL_TLS_TRUST_BUNDLE`,
`EYVAN_MUTUAL_TLS_CERTIFICATE`, and `EYVAN_MUTUAL_TLS_PRIVATE_KEY`.
`EYVAN_MUTUAL_TLS_CLIENT_ENABLED` configures controller-created cell-agent
clients with the supplied trust and identity files. Required native-worker
mTLS is covered by a local handshake test that rejects clients without a
certificate. These paths remain opt-in: default internal listener enforcement,
automated bootstrap certificate handoff, and all cross-machine client paths
are not yet qualified. Use HTTPS endpoints and an explicitly configured
mutually authenticated transport across machines; do not infer encryption
from a bearer token or private network alone. Provider credentials (`EYVAN_AWS_ACCESS_KEY_ID`, `EYVAN_AWS_SECRET_ACCESS_KEY`,
`EYVAN_AWS_SESSION_TOKEN`, `EYVAN_AWS_REGION`, `EYVAN_AZURE_ACCESS_TOKEN`,
`EYVAN_AZURE_SUBSCRIPTION_ID`, `EYVAN_AZURE_RESOURCE_GROUP`,
`EYVAN_AZURE_LOCATION`, and gcloud's own) stay scoped to the agent or plane
process that talks to that cloud.

### Preparatory native-worker certificate bootstrap

The low-level `build_native_worker_startup_script()` renderer accepts an optional
`worker_tls_secret_resource`: one immutable `projects/.../secrets/.../versions/N`
resource, with a positive numeric version (not `latest`). The secret contains a
JSON object with exactly `trust_bundle`, `certificate`, and `private_key`, each
holding PEM text. Use a dedicated worker key, never an agent/controller private
key. Provision and grant access to this bundle separately; Eyvan does not issue
certificates or change its IAM policy through this option.

The pre-start `eyvan.worker.runtime.native_tls_bootstrap` module fetches the
bundle without echoing provider output, rejects malformed/inconsistent material
and encrypted keys, and stages files with mode 0600 in an owner-only directory.
Only after TLS validation does it publish `/run/eyvan-native/tls`; the existing
root-run systemd unit owns that directory and requires client certificates.
An existing destination is refused rather than overwritten. Systemd's private
runtime-directory lifecycle removes these files on stop; prepare all clients
before enabling a requiring listener. A new version changes the rendered script.
The helper checks CA parsing and key-pair consistency, not certificate expiry,
remote trust, endpoint SANs or per-cell issuance policy.

**This is not a GCP fleet configuration option yet.** Current provider endpoints
use provider-selected IP addresses; pre-issued certificates require a stable
DNS or reserved-IP/SAN contract, including distinct multi-host identities.
Automatic fleet wiring, replacement/rotation and deployed access remain open.
Do not disable hostname verification to bridge that gap. When omitted, the
renderer retains its previous output exactly; networking defaults are unchanged.

## Dashboard sessions

The dashboard never holds an OIDC token. Sign-in is a backend-for-frontend
flow (`eyvan/server/api/dashboard_sessions.py`): the server mints the PKCE
challenge and state, exchanges the authorization code at the identity
provider, and opens an in-process session. The browser holds a
`__Host-eyvan_session` cookie — HttpOnly, Secure, SameSite=Lax — and a temporary
`__Host-eyvan_oidc_tx` cookie binds the login transaction. Both use `Path=/` and
no `Domain`; the `__Host-` prefix prevents a sibling subdomain from planting a
parent-domain authentication cookie. The legacy unprefixed cookies are not accepted;
existing browsers must sign in again. OIDC tokens stay server-side, and these
cookies are inaccessible to JavaScript. The API authenticates the session cookie
by verifying the stored access token exactly as it would a presented bearer;
a presented bearer still wins, so SDK clients are unaffected. The one credential
that still lives in the tab is the loopback development credential, whose model
is this-machine-only by design.

Before creating a session, the callback binds the ID token's issuer, audience,
authorized party, subject, nonce and expiry to the configured provider/client
and login transaction. Discovery issuer checks also apply on cache hits.
The ID token's authenticity relies on the direct authenticated TLS token
exchange, not a separate ID-token JWKS/signature verification step; API access
tokens are independently verified. Provider-controlled error descriptions and
unknown error codes are replaced with fixed messages rather than copied into
redirect URLs. These request-level protections are tested; real-browser and
multi-replica deployment qualification remain separate requirements.

Limits that shape deployment, stated rather than discovered: the session
store is in-process (more than one API replica needs sticky sessions until
sessions move into the repository). Each process retains at most 1,024 pending
login transactions and 1,024 sessions. Full stores reject new admissions with
HTTP 503 and Retry-After rather than evicting a live session. Expiry and removal
release slots; deployment-level rate limiting is still needed for login floods.
Cookie-authenticated writes, including
logout, require an Origin matching the configured public URL. SameSite=Lax
is defense in depth, not the sole CSRF boundary: hostile sibling subdomains
can be same-site without being same-origin. Missing or opaque origins are
refused for these writes. Server-side session lifetime is bounded by the
configured session TTL and any shorter provider lifetime, independently of
browser cookie expiry. The unconditional `Secure` flag means a plane serving
plain HTTP over a network cannot hold OIDC sessions at all.

For TLS termination at a reverse proxy, the browser must still use HTTPS.
Set `EYVAN_PUBLIC_URL` to that browser-facing HTTPS URL: the callback URI and
cookie-write Origin checks use this configuration, not Host or forwarded headers.
The proxy must preserve the browser's Origin and cookies and route authentication
and API paths to the same session-owning process. Proxy headers do not establish
a different trusted public origin. If the proxy-to-API listener is network-bound
plaintext, the separate explicit acknowledgement described under Exposure is
still required; external HTTPS does not waive the listener's startup refusal.

The dashboard's HTML is served with a strict Content-Security-Policy
(`script-src 'self'`, `object-src 'none'`, `frame-ancestors 'none'`, and
companions such as `X-Content-Type-Options: nosniff`), so an injected script
has no external origin to load from and no sibling origin to frame.

## Authentication and policy audit

Set `EYVAN_AUTH_AUDIT_ENABLED=1` to retain protected-API authentication
rejections as structured `authentication_failed` audit events. The default is
off to preserve existing event-stream behavior. Events use the route template,
HTTP method and rejection status; they do not retain credentials, cookies,
query strings, raw path identifiers or unverified tenant claims. The actor is
`anonymous` because a rejected credential establishes no trustworthy identity.

These unscoped security events are visible through the fleet-administrator
audit view, not a caller-selected project's history. Grant issuance/revocation
and policy edits already produce transactional `grant.issued`, `grant.revoked`
and `policy.updated` events carrying the verified actor and scope.

An audit-store failure does not turn a rejected request into an authenticated
one or replace its denial. It emits an error naming only the exception type,
not exception text that could include sensitive database parameters. Retention
and deployment-level rate limiting remain operator responsibilities: enabling
this audit on an attacked endpoint creates durable writes. This coverage is
for protected API authentication; IdP-side events and browser-login callback
outcomes are not claimed as covered by it.
