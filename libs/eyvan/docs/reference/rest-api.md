# Eyvan REST API Reference — Partial Working Draft

> **Status: partial working draft.** This page is intentionally incomplete and is
> not a promise that every route, field, or response has been documented or
> verified. Use the running server's OpenAPI document and the implementation in
> `eyvan/server/api/app.py` (plus `auth.py`, `schemas.py`, and `sse.py`) as the
> source of truth. The notes below record high-value, source-backed behavior;
> they are not an exhaustive API specification.

For schema-derived documentation, use the [generated route index](api-generated.md)
and [OpenAPI snapshot](openapi.json). Regenerate with
`uv run python scripts/generate_api_reference.py`; CI checks for drift with
`--check`. Routes intentionally excluded from OpenAPI remain outside that index.

- [Conventions](#conventions)
- [Authentication & Authorization](#authentication--authorization)
- [Error model](#error-model)
- [Idempotency](#idempotency)
- [Compatibility](#compatibility)
- [Tags](#tags)
  - [system](#system)
  - [workloads](#workloads)
  - [artifacts](#artifacts)
  - [capacity](#capacity)
  - [fleets & cells](#fleets--cells)
  - [governance & tenancy](#governance--tenancy)
  - [federation](#federation)
  - [events](#events)
- [Streaming (SSE)](#streaming-sse)
- [Internal and operational routes](#internal-and-operational-routes)
- [Schemas](#schemas)

## Conventions

The versioned public API uses `/api/v1/...`. The server also rewrites the
compatibility spelling `/v1/...` to that prefix. Unless noted otherwise, route
parameters and query parameters are described by the route's OpenAPI schema;
this draft intentionally does not duplicate those schemas. `GET /healthz` is a
liveness endpoint and does not touch the database.

## Authentication & Authorization

Authenticated API routes accept an OIDC bearer token in the standard header:

```http
Authorization: Bearer <token>
```

The configured OIDC verifier validates the token's issuer, audience, expiry, and
signature using the issuer's JWKS. A head started with bootstrap identity may
instead accept its generated bootstrap secret as a bearer token. In local
development, explicit `X-Eyvan-User`, `X-Eyvan-Roles`, `X-Eyvan-Org`, and
`X-Eyvan-Project` headers are supported when development authentication is
enabled. When a loopback development credential exists, callers must also send
`X-Eyvan-Local-Credential`; possession of the port or identity headers alone is
not authentication. Authorization is then evaluated against the principal's
role grants and route scope.

Health/liveness and other routes without an authentication dependency may be
called without credentials. Do not infer access requirements from this draft;
check the route implementation and OpenAPI definition for a particular path.

## Error model

Errors are JSON objects with a `detail` member. Common service failures map to
`400` (input/service error), `403` (authorization), `404` (not found), `409`
(conflict), and `503` (unavailable). Request and model validation errors use
`422` with `detail` as a list of validation entries. Authentication failures
normally use `401`.

Some boundary errors include a stable `code` and structured `context`; for
example, an unsupported SDK can receive `code: "client_too_old"`. Clients
should use the HTTP status and documented code when available, rather than
matching human-readable detail text. A `503` may include `Retry-After` when the
caller can retry safely.

## Idempotency

Mutation routes that accept an idempotency key require a non-empty
`Idempotency-Key` header (currently bounded by the route validation to 200
characters). The key is passed to the service layer to make retries/replays
safe for that operation; a replay may be marked with the response header
`Idempotent-Replay: true`. The exact set of accepting routes is defined by the
OpenAPI schema and route signatures, so callers should not assume every POST
supports the header. Read-only estimates and overwrite-style operations may
have different semantics and do not automatically gain idempotency from this
header.

## Compatibility

The server accepts `/v1/` as a compatibility alias for `/api/v1/`. SDK requests
send `X-Eyvan-Client-Version`. If `EYVAN_MIN_CLIENT_VERSION` is configured and
the reported, parseable client version is below the floor, the server returns
`426 Upgrade Required` with `code: "client_too_old"` and a context containing
`client_version` and `minimum_client_version`. Missing or unparseable version
headers are accepted by this one-sided gate, including requests from browsers
and curl. A legacy scoped-artifact request missing required `org_id` and
`project_id` can likewise receive `426` and
`X-Eyvan-Required-Client-Feature: scoped-artifact-identity`.

## Tags

This is a route-family index, not a complete endpoint listing. For exact paths,
methods, parameters, and response models, consult the running OpenAPI schema.

### system

Health/build and selected system, query, dashboard, and credential-rotation
operations live under `/api/v1`. Build information is authenticated; health is
intended for liveness. Some system operations are operational controls and are
not general application APIs.

### workloads

Workload submission and inspection are centered on `/api/v1/workloads`, with
related batch, group, DAG, estimate, lifecycle, progress, service, execution,
and log routes. Submission mutations use idempotency keys where declared by the
route signature.

### artifacts

Managed artifact inventory, holds, collection, content transfer, and usage
routes live under `/api/v1/artifacts`. Content-transfer routes are internal or
not included in the public schema unless explicitly exposed by OpenAPI.

### capacity

Capacity reports and operations include `/api/v1/capacity` and related report
families such as quota, warmth, carbon, launch latency, usage, and queue trend.
The exact authorization and query requirements vary by route.

### fleets & cells

Fleet and cell resources include `/api/v1/fleets`, `/api/v1/cells`, and capacity
pool routes. These are administrative surfaces with scope- and role-dependent
authorization.

### governance & tenancy

Policy, webhook, quota, lease, and related governance/tenancy resources are
under `/api/v1`. Their schemas and required scope fields are authoritative.

### federation

Federation operations are exposed under the versioned API where enabled. This
working draft does not attempt to describe provider-specific request or
response details.

### events

Audit event polling is available at `/api/v1/events` and fleet-scoped events at
`/api/v1/fleet/events`. Streaming variants are described below.

## Streaming (SSE)

`GET /api/v1/events/stream` returns `text/event-stream` for an authenticated,
org/project-scoped audit stream. It accepts `after_sequence`, `follow`, and the
`Last-Event-ID` header; the latter is an integer sequence cursor. Frames carry
JSON data, an action in the SSE `event` field, and the audit sequence as `id`.
Idle follow streams emit the SSE comment `: keep-alive`.

`GET /api/v1/workloads/{workload_id}/logs/stream` follows one workload's
structured logs. It accepts `after`, `follow`, `limit`, and `Last-Event-ID`.
Its cursor is a joined per-attempt/rank cursor, not the audit sequence. Frames
use events such as `log.record` and `log.attempt_changed`; reconnecting is
at-least-once at page boundaries. Both streams set `Cache-Control: no-cache`
and `X-Accel-Buffering: no`, and authorization/cursor validation happens before
the streaming response is committed.

## Internal and operational routes

The following are deliberately separated from the public route-family index:

- `GET /readyz` (not in the OpenAPI schema) is readiness, not liveness.
- `GET /metrics` (not in the OpenAPI schema) exposes Prometheus metrics.
- Artifact and secret content-transfer routes, selected progress-report and
  child-workload routes, and system reconciliation/proof/soak controls may be
  hidden from OpenAPI or intended for trusted internal workers/operators.
- The dashboard root and static assets are UI delivery, not REST API contracts.

Hidden from OpenAPI does not mean unauthenticated or safe for arbitrary callers.
Treat these routes as implementation details unless a deployment explicitly
supports them.

## Schemas

Response and request models are defined in `eyvan/server/api/schemas.py` and
in route signatures. This section is intentionally left as a pointer rather
than an incomplete schema transcription; use the server's OpenAPI document for
machine-readable models.
