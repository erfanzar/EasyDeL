# Part 9 — The REST API and authentication

Source: `eyvan/server/api/` — `app.py` (routes), `auth.py`, `schemas.py`,
`sse.py`, `server_startup.py`, `server_bind.py`. Wire reference:
`docs/reference/rest-api.md`. The running server's `/docs` (OpenAPI) is
authoritative for exact schemas.

## Conventions

- Versioned API: `/api/v1/...`; `/v1/...` is accepted as a compatibility alias.
- `GET /healthz` is liveness (no database); `GET /readyz` is readiness (proves
  database readiness; not in OpenAPI).
- `GET /metrics` is Prometheus, authenticated by `EYVAN_METRICS_TOKEN` on network
  binds.
- Errors are JSON with a `detail`; 400 input/service, 401 auth, 403 authorization,
  404 missing, 409 conflict, 422 validation, 503 unavailable (may carry
  `Retry-After`). Some boundaries add a stable `code` (e.g. `client_too_old`).
- Mutations accept a non-empty `Idempotency-Key` (route-declared); a replay may be
  marked `Idempotent-Replay: true`.
- SDK requests carry `X-Eyvan-Client-Version`; a server with
  `EYVAN_MIN_CLIENT_VERSION` rejects older clients with `426` and
  `code: "client_too_old"`.

## Route families

All under `/api/v1` unless noted:

- **system**: `/health`, `/build`, `/session`, `/query`, `/process/stacks`,
  `/process/allocations`, `/dashboard/config`, `/system/invariants`,
  `/system/soak-evidence`, `/system/reconcile`, `/secrets/rotate`;
  Prometheus metrics are at `/metrics`, outside `/api/v1`.
- **workloads**: submit/estimate/list/get/children/progress/endpoints/exec/
  log-page/process-status/cancel/kick/checkpoint/retry, plus
  `/workloads/{id}/logs/stream`.
- **capacity & fleets**: `/capacity` plus `queue-trend`, `carbon`,
  `launch-latency`, `warmth`, `billing`, `quota`, `orphans`, `builds`;
  `/capacity-pools` CRUD; `/fleets` CRUD; `/cells` + drain/delete; `/usage`.
- **artifacts**: list, `collect`, `holds`, `usage`, content-addressed
  `PUT /artifacts/callables/{digest}` and `/artifacts/workspaces/{digest}`,
  `GET /artifacts/results/{result_id}`, and unlisted internal content-transfer
  routes.
- **governance / tenancy / federation**: policies, grants, organizations,
  projects, budgets, quotas, webhooks, leases, peers, cross-plane grants.
- **events**: `/events`, `/fleet/events`, and SSE `/events/stream`.

## Authentication paths (`auth.py`)

A presented bearer token takes precedence over cookies and development auth:

- **Team tokens** — with team mode enabled (`eyvan-server --team`, or
  `team: true` in the platform file), `eyv_...` bearer tokens resolve
  against team accounts. Invalid, expired, or revoked tokens return 401; they do
  not fall through to OIDC.
- **Bootstrap token** — a head started with bootstrap identity accepts its
  generated secret for the fixed `eyvan-bootstrap` principal (`Role.FLEET_ADMIN`).
  The caller cannot self-assign roles. Comparison uses `secrets.compare_digest`.
- **OIDC bearer** — `OIDCTokenVerifier` validates issuer, audience
  (`EYVAN_OIDC_AUDIENCE`), expiry, and signature via JWKS (RS256/ES256). An
  `eyvan_grants` claim supplies role grants.

Without a bearer token, the server checks the team-session cookie when team mode
is enabled, then the OIDC BFF session cookie. Cookie-authenticated writes require
an Origin matching `EYVAN_PUBLIC_URL`. The OIDC BFF keeps provider tokens
server-side; the browser holds an HttpOnly cookie (Part 13).

**Development auth** is the final fallback and is refused on a network bind.
Explicit `X-Eyvan-User` / `X-Eyvan-Roles` / `X-Eyvan-Org` /
`X-Eyvan-Project` headers supply the development identity. The
`is_own_dashboard_request()` predicate controls only the default identity for the
plane's own dashboard; explicit dev headers do not require that predicate.

On the automatic loopback development-auth path, an issued
`LocalDevelopmentCredential` (0600 file) must be presented as
`X-Eyvan-Local-Credential`, including by the plane's dashboard. The local-credential
check is separate from the dashboard predicate. Bearer and authenticated cookie
paths return before this check and do not require the local credential.

Authorization: `authenticated_principal()` unions OIDC claims with plane-issued
durable grants, resolved per request (no cache), so `eyvan grant revoke` takes
effect on the next request. Roles (`eyvan/records.py`): `fleet_admin`,
`org_admin`, `project_admin`, `project_operator`, `project_submitter`,
`project_viewer`; permissions: `read`, `submit`, `operate`, `manage_project`,
`manage_own_capacity`, `manage_fleet`.

## SSE streams (`sse.py`)

Two streams, two different contracts:

- `GET /api/v1/events/stream` — the scoped audit stream. `after_sequence`,
  `follow`, `Last-Event-ID` (integer sequence cursor); frames carry JSON data, an
  action in `event`, the audit sequence as `id`; idle streams emit `: keep-alive`.
- `GET /api/v1/workloads/{id}/logs/stream` — one workload's structured logs.
  `after`/`follow`/`limit`/`Last-Event-ID`; the cursor is a joined
  per-attempt/rank cursor (not the audit sequence); frames use `log.record` and
  `log.attempt_changed`; reconnection is at-least-once at page boundaries.

Implementation invariants worth copying: authorization happens **before** the
generator is built (otherwise unauthorized callers see 200-then-nothing); streams
set `Cache-Control: no-cache` and `X-Accel-Buffering: no` to defeat proxy
buffering; the audit stream uses bounded lookback + dedupe.

## Static dashboard serving

`dashboard_bundle()` in `app.py`: if `EYVAN_DASHBOARD_DIR` names an existing
directory it is served; otherwise a checkout's `dashboard/dist` is used when it
contains `index.html`, then the copy a release build packaged as `eyvan/_dashboard`
(`scripts/stage_dashboard.py`). The bundle is mounted at `/` with `StaticFiles(...,
html=True)`; API routes stay under `/api/...`. Because the dashboard uses hash
routes, deep navigation needs no history-mode rewrite rules. With no bundle at all
(a wheel built without staging), `/` returns a self-contained page explaining
exactly that, with pointers to `eyvan dashboard-dev`, `/healthz`, and `/docs`.
