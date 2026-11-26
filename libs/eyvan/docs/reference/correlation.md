# Operation correlation and tracing

Set `EYVAN_CORRELATION_ENABLED=1` on the control plane after upgrading the agents
and native workers that receive its mutation envelopes. The default is off;
unset correlation fields are omitted from serialized outbox and mutation data,
so old envelopes do not acquire a new `null` key.

## Identity and ownership

Correlation is observability, not authorization or a fencing token.

- Workload commands use the server-issued workload ID.
- Workload-driven capacity commands use the cell's recorded
  `autoscale_request_workload_id`.
- Background capacity operations without a workload use their outbox command
  ID. A warm cell's creation is not falsely attributed to a later workload.
- Legacy IDs that do not fit the bounded, header-safe alphabet are mapped to a
  deterministic SHA-256-based ID; underlying persisted IDs are not rewritten.
- Caller-provided correlation hints in command payloads are not trusted.

The identifier is retained in the outbox JSON record and propagated through
agent/native-worker mutation adapters, including gang dispatch, cancellation,
preemption and predecessor retirement. It does not change generation checks,
fencing tokens, or idempotency keys.

Relevant single-workload API responses expose `X-Eyvan-Correlation-ID` when
enabled. Lease reads use the lease's workload ID. Cell reads expose an origin
only while a workload origin is recorded; they do not invent one from the cell
ID. Batch/group/DAG responses use their parent/coordinator identity; children
remain distinct workloads with their own IDs.

## Logs and spans

Outbox delivery and mutating provider boundaries emit operation start/completion
or failure logs carrying the ID. Context is task-scoped, propagates through
`asyncio.to_thread`, and resets on failure. An explicitly uncorrelated nested
operation clears the outer correlation for its duration.

The boundaries also create OpenTelemetry spans with `eyvan.correlation_id` when
an SDK is configured. The API dependency alone uses no-op tracing; this change
does not install or configure a production exporter.

Controller-to-agent HTTP calls now generate W3C `traceparent` during a correlated
operation with a valid active span. Ordinary uncorrelated calls keep their old
headers, even inside an unrelated ambient span. Admission's correlation setting
controls the persisted operation ID; turning it off does not erase IDs from
already-admitted outbox records. Set `EYVAN_CORRELATION_ENABLED=1` on receiving
agents as well to join this remote parent, after normal credential validation.
Missing, malformed or oversized parents start an independent context. Context
is restored on success or failure and copied into synchronous route handlers.

Automatic generation/extraction excludes `baggage` and `tracestate`; it does not
rewrite operator-supplied custom-client headers or external instrumentation.
Incoming trace IDs are untrusted telemetry, never workload identity, admission
policy or authentication.

The native remote client also generates this parent during correlated calls,
including parallel multi-host staging. Each staging task receives its own copy
of the calling context. Native workers opt in separately with
`EYVAN_CORRELATION_ENABLED=1` (or `create_native_worker_app`'s
`correlation_enabled=True`). After bearer authentication they create a request
span using the route template, never the raw URL/query. CLI configuration is
validated before consuming a one-use secret. Disabled calls preserve bearer
headers and existing retry behavior: only the existing GET protocol-error
retry is allowed; mutations are not automatically retried.

Local SDK/ASGI tests join controller, agent and native-worker span IDs and
verify thread fan-out, rejection and exception cleanup. Bootstrap configuration,
a trace parent persisted across durable queueing, and a deployed collector
remain separate work; no automatic cloud rollout is implied.

Span failures retain error status and exception type, not raw provider exception
messages or stack traces that may include credentials. A real in-memory SDK
exporter regression verifies that behavior. Other application logging remains
subject to the existing redaction rules.

## Evidence and limits

Regression coverage includes flag-off wire shape, invalid IDs, legacy IDs,
PostgreSQL persistence, concurrent task/thread scope, provider-boundary context,
API headers and mutation forwarding. It is not a claim that a live cloud
provider or a deployed telemetry collector has been exercised. Fresh Spot
validation remains subject to the requested project's IAM permissions.
