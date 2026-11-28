# Eyvan SDK & CLI Reference — Partial Working Draft

> **Status: partial working draft.** This page is intentionally incomplete and
> does not claim exhaustive symbol, option, default, or behavior coverage. Use
> the installed package/source and the relevant implementation and tests under
> `eyvan/sdk/`, `eyvan/__init__.py`, and `eyvan/config.py` as the source of
> truth. The notes below are a concise, source-backed orientation rather than a
> complete SDK or CLI specification.

## Table of Contents

1. [Public API (`import eyvan`)](#1-public-api-import-eyvan)
2. [Configuration objects: `TpuConfig`, `GpuConfig`, `CpuConfig`](#2-configuration-objects)
3. [`execute` / `init` contracts](#3-execute--init-contracts)
4. [`EyvanJob` handle lifecycle](#4-eyvanjob-handle-lifecycle)
5. [Error taxonomy](#5-error-taxonomy)
6. [Client architecture (`eyvan.sdk.client`)](#6-client-architecture-eyvansdkclient)
   - [Endpoint resolution](#61-endpoint-resolution)
   - [Transport](#62-transport)
   - [Retry policy](#63-retry-policy)
   - [Idempotency](#64-idempotency)
   - [Requests & types](#65-requests--types)
   - [Identity](#66-identity)
   - [Sync & async clients](#67-sync--async-clients)
   - [Jobs & callable submission](#68-jobs--callable-submission)
7. [Supporting SDK modules](#7-supporting-sdk-modules)
8. [CLI reference (`eyvan ...`)](#8-cli-reference)
9. [Appendix: shared CLI helpers](#9-appendix-shared-cli-helpers)
10. [Cross-cutting HTTP behavior](#10-cross-cutting-http-behavior)
   - [Authentication](#101-authentication)
   - [Errors and retries](#102-errors-and-retries)
   - [Idempotency](#103-idempotency)
   - [SSE helpers](#104-sse-helpers)
   - [Compatibility](#105-compatibility)

---

## 1. Public API (`import eyvan`)

_(partial; consult `eyvan/__init__.py` and its tests for the currently exported
surface.)_

## 2. Configuration objects

_(partial; configuration class fields and defaults are defined by source, not
this page.)_

## 3. `execute` / `init` contracts

_(partial.)

## 4. `EyvanJob` handle lifecycle

_(partial.)

## 5. Error taxonomy

_(partial; the transport behavior below is the most stable cross-cutting part.)

## 6. Client architecture (`eyvan.sdk.client`)

_(partial; module and class details remain source-defined.)_

### 6.1 Endpoint resolution

`ControlPlaneEndpoint` accepts an absolute `http` or `https` URL, removes a
trailing slash, and rejects credentials, queries, fragments, and invalid ports.
The default control-plane URL is `http://127.0.0.1:8080`.

### 6.2 Transport

The sync and async transports use HTTPX. They attach the SDK client-version
header and resolved identity headers, expose checked requests that raise on
failure, and translate transport-level HTTPX failures into
`EyvanUnavailableError`. Explicitly injected HTTPX clients are not owned by the
transport.

### 6.3 Retry policy

Mutation retries are bounded by `RetryPolicy` (default `max_attempts=3`). The
retryable statuses are 429, 502, 503, and 504; a 503 without `Retry-After` is
not retried by the default policy. Retry delays honor numeric or HTTP-date
`Retry-After` values (capped by policy) and otherwise use capped exponential
backoff. Callers should still avoid retrying non-idempotent operations without a
stable key.

### 6.4 Idempotency

`IdempotencyKeyFactory` preserves a caller-supplied non-empty key. Otherwise it
creates a fresh key in the form `<prefix>[-<operation>]-<uuid>`. Transport
mutation retries reuse the same generated or supplied `Idempotency-Key` on
all attempts; this is what prevents a transient retry from becoming a second
mutation where the server supports idempotency.

### 6.5 Requests & types

_(partial; request and response types are defined in
`eyvan/sdk/client/requests.py` and `types.py`.)_

### 6.6 Identity

`ClientIdentity` emits either `Authorization: Bearer <token>` (which takes
precedence) or development headers: `X-Eyvan-User`, `X-Eyvan-Roles`, and
optional `X-Eyvan-Org`/`X-Eyvan-Project`. Loopback resolution can synthesize a
local subject and load the owner-only credential from
`~/.eyvan/local-development-credential`, sending it as
`X-Eyvan-Local-Credential`.

### 6.7 Sync & async clients

_(partial; see `sync.py` and `async_client.py` for constructors and lifecycle.)_

### 6.8 Jobs & callable submission

_(partial; see `jobs.py`, `callables.py`, and the workload command modules.)_

## 7. Supporting SDK modules

_(partial.)

## 8. CLI reference

_(partial; this page is not a complete option reference.
Run `eyvan <command> --help` against
the installed version for authoritative options and defaults.)_

## 9. Appendix: shared CLI helpers

_(partial.)

## 10. Cross-cutting HTTP behavior

The following behavior is shared by the SDK's HTTP boundary and is documented
here because it affects every command or client that uses that boundary.

### 10.1 Authentication

Use an OIDC/bootstrap bearer token through the client's token identity, or use
explicit development identity only for a development-authenticated plane. Do
not treat loopback location as authorization: when a local credential exists,
the SDK loads and sends it, and the server requires it. Roles and scope are
server-issued/validated authorization data, not a substitute for a bearer
credential on a production plane.

### 10.2 Errors and retries

`EyvanHTTPError` retains the response status and extracted structured `detail`.
JSON errors with a string `detail` preserve that string; non-string details are
compact JSON. `EyvanUnavailableError` indicates that the configured endpoint
could not be reached. A 422 response containing `extra_forbidden` is annotated
by the SDK as likely client/control-plane version skew; this is diagnostic, not
a replacement for inspecting the server's response.

### 10.3 Idempotency

For mutation calls, the SDK puts a stable `Idempotency-Key` into the request
headers and reuses it through retries. A supplied key must be non-empty. The
server may identify a replay with `Idempotent-Replay: true`; behavior remains
route-specific, so read the REST route schema before assuming a mutation is
replayable.

### 10.4 SSE helpers

The control plane provides audit and workload-log endpoints as
`text/event-stream`, supports `Last-Event-ID` cursors, and emits keep-alive
comments; see the REST reference for the wire-level framing. The current
`WorkloadFollower` in `eyvan/sdk/follow.py` is a CLI-oriented polling follower
that reads ordinary resumable log pages, rather than a generic SSE client. Do
not assume it provides SSE parsing or reconnection semantics. Callers that need
SSE should use an HTTP client against the documented stream routes and treat
exact helper availability as source-defined.

### 10.5 Compatibility

SDK requests include `X-Eyvan-Client-Version`. A server configured with a
minimum version can reject an older parseable SDK at the boundary with `426`
and `code: "client_too_old"`; upgrade the client or align the control plane.
The REST server also accepts `/v1/` as an alias for `/api/v1/`. These are
compatibility mechanisms, not a guarantee that arbitrary SDK/server versions
are interchangeable. The server's OpenAPI document and the version-skew
response are authoritative.
