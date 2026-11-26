# Worker and Runtime Reference

This document describes Eyvan's worker-side code: the **worker agent**
(`eyvan/worker/agent/`), the **runtime** (`eyvan/worker/runtime/`), and the
worker-level support modules `eyvan/worker/distributed_environment.py` and
`eyvan/worker/identity.py`. It is grounded in the actual code; every
behavioral claim names the symbol that implements it.

Module map (all paths relative to the repository root):

| Module | Role |
|---|---|
| `eyvan/worker/agent/app.py` | FastAPI application exposing the worker agent HTTP surface |
| `eyvan/worker/agent/client.py` | HTTP client the control plane uses to talk to agents |
| `eyvan/worker/agent/cli.py` | CLI entry point that runs the agent |
| `eyvan/worker/agent/models.py` | Pydantic request/response models for the agent API |
| `eyvan/worker/agent/service.py` | `CellAgentService`: business logic behind the routes |
| `eyvan/worker/distributed_environment.py` | Multi-host distributed environment construction (JAX coordination) |
| `eyvan/worker/identity.py` | Worker identity and attestation helpers |
| `eyvan/worker/runtime/native_worker.py` | Native (bare-metal) single-host worker execution |
| `eyvan/worker/runtime/native_gang.py` | Native gang execution across hosts |
| `eyvan/worker/runtime/native_slice.py` | TPU slice execution model |
| `eyvan/worker/runtime/native_remote.py` | HTTP(S) execution client for a remote native worker |
| `eyvan/worker/runtime/native_bootstrap.py` | Bootstrap payload executed on native hosts |
| `eyvan/worker/runtime/container.py` | Container (OCI) worker execution |
| `eyvan/worker/runtime/local_process.py` | Local subprocess execution |
| `eyvan/worker/runtime/local_worker.py` | Local worker wrapper |
| `eyvan/worker/runtime/callable_protocol.py` | Callable wire protocol, sha256/HMAC signing and verification |
| `eyvan/worker/runtime/callable_worker.py` | In-process callable worker entry |
| `eyvan/worker/runtime/callables.py` | Public re-exports for callable runtime |
| `eyvan/worker/runtime/oci_callable_runner.py` | Runs a staged callable inside an OCI container |
| `eyvan/worker/runtime/oci_callable_stager.py` | Stages callable artifacts into an OCI image/workspace |
| `eyvan/worker/runtime/stage_executor.py` | Executes a workload's ordered stages |
| `eyvan/worker/runtime/systemd_transient.py` | systemd transient-unit execution backend |
| `eyvan/worker/runtime/host_accelerators.py` | Host accelerator discovery |
| `eyvan/worker/runtime/node_health.py` | Node health probes and reporting |
| `eyvan/worker/runtime/progress.py` | Progress record emission |
| `eyvan/worker/runtime/registry.py` | Runtime registry (workload → runtime handle) |
| `eyvan/worker/runtime/results.py` | Result re-exports |
| `eyvan/worker/runtime/secret_materialization.py` | Materializing workload secrets on the worker |
| `eyvan/worker/runtime/workspace_entry.py` | Preparing the workload workspace before entry |
| `eyvan/worker/runtime/log_records.py` | Structured log record capture and shipping |
| `eyvan/worker/runtime/http_retry.py` | Small HTTP retry helper |
| `eyvan/worker/runtime/_multihost_site.py` | `sitecustomize` payload injected for multi-host bootstrap |
| `eyvan/worker/runtime/dependency_entry.py` | Installing/validating dependency entry points |
| `eyvan/worker/runtime/in_job.py` | Helpers available to code running inside a job |

## 1. The worker agent HTTP surface

The **cell agent** is a FastAPI application built by `create_agent_app`
(`eyvan/worker/agent/app.py`). One agent process fronts the capacity and
execution backends for the cells it owns; the control plane is its only
intended caller. It is served by uvicorn from `eyvan/worker/agent/cli.py`
(`main()`, default `--host 0.0.0.0 --port 8090`), which loads
`EyvanSettings` — either from the environment or, on GCP native workers, from
a `GcpNativeRuntimeProfile` file via the suppressed `--runtime-profile`
option — logs the agent's workload identity (a principal, never a token) on
every start, and mounts mutual-TLS server options from
`eyvan.identity.mtls.server_ssl_options` when the operator has moved the
server mode off `off`.

### 1.1 Authentication and hard failure rules

- Every `/api/v1/**` route depends on `authenticate`, which compares the
  `Authorization` header to `Bearer <settings.agent_shared_secret>` with
  `hmac.compare_digest`. The comparison is wrapped in `try/except TypeError`
  because `compare_digest` raises for non-ASCII strings and the header is
  caller-controlled: garbage must answer 401, never 500. `/healthz` is the
  only unauthenticated route.
- If `agent_shared_secret` is unset, startup **fails** with `ValueError`
  naming the exact remedy unless `--allow-unauthenticated` /
  `EYVAN_AGENT_ALLOW_UNAUTHENTICATED=1` is passed, in which case the agent
  logs a loud warning that every endpoint is reachable without credentials.
- OpenAPI docs are disabled (`docs_url=None, redoc_url=None`).

### 1.2 Error taxonomy (exception handlers in `app.py`)

The agent's deliberate answers are status codes with a machine-readable
`code` field, and the client is built to distinguish every one of them:

| Status | Raised by | Body | Client error (`client.py`) |
|---|---|---|---|
| 409 | `StaleMutationError` | `{"detail", "code": "stale_mutation"}` | `CellAgentStaleMutationError` |
| 409 | `DriverBusyError` | `{"detail", "code": "driver_busy"}` | `CellAgentBusyError` |
| 400 | `ValueError` (rejected provider configuration) | `{"detail", "code": "invalid_configuration"}` | `CellAgentInvalidConfigurationError` (terminal) |
| 501 | `UnsupportedCapabilityError` | `{"detail", "code": "unsupported_capability"}` | `CellAgentUnsupportedCapabilityError` (terminal) |
| 502 | `CapacityProviderError` | `CapacityProviderFailure` (`detail`, `code: CapacityProviderFailureCode`) | `CellAgentCapacityProviderError` (provider failure, operator-safe) |
| any other 5xx, transport error | agent itself failing | unstructured | `CellAgentUnavailableError` — *nothing was learned* |

Agent unavailability is not provider failure. A transport error or bare 5xx
means the agent could not answer; it is not evidence that capacity was lost.
Typed configuration and provider failures remain distinct so an operator can
act on the cause without retiring capacity by inference.

### 1.3 Route inventory

All routes are created inside `create_agent_app` and delegate to
`CellAgentService` (`service.py`). Cell-scoped routes carry `{cell_id}`;
gang routes are additionally keyed by `{workload_id}/{remote_id}`.

Discovery and introspection:

- `GET /api/v1/capabilities` → `AgentCapabilities`: provider, executor,
  accelerator types, capacity classes, `capacity_backends` (forwarded, not
  dropped — an agent whose declaration was stripped looks like one that
  predates the field), `supports_checkpoint`, `supports_callables`,
  `supports_exec`, `max_gang_replicas`, `resource_enforcement`.
- `GET /api/v1/build` → `AgentBuild`: the running code's version, origin,
  source digest, file count, Python and platform, from
  `build_identity()`. A separate route rather than a capabilities field
  because capabilities is `extra="forbid"`: an added field would make an
  older control plane unable to decode any capabilities response at all,
  while an old caller never asks for this route and a new caller meeting an
  old agent gets 404, recorded as *unknown*.
- `GET /api/v1/cells/{cell_id}/worker-builds` → per-native-worker build
  list. Three distinct answers: a list from drivers that probed, `[]` from
  backends that run the workload in the agent process itself, `None` from a
  driver that cannot be asked. Conflating any two "reports agreement nobody
  established".
- `GET /api/v1/cells` → durable cell inventory (name, state, desired_state,
  generation, project, zone, accelerator type, queued-resource ID), sorted
  explicitly rather than inherited from the storage layer. Read-only: it
  exists because a deleted control plane leaves cells with
  `desired_state: up` that the agent keeps re-ensuring — observed as a
  v5p-128 queued resource resurrected from a fleet deleted days earlier.

Capacity lifecycle (all return `AgentObservation`):

- `GET /api/v1/cells/{cell_id}` — `observe`
- `POST /api/v1/cells/{cell_id}/reconcile` — `reconcile`
- `PUT /api/v1/cells/{cell_id}/capacity` — `apply_capacity`
- `POST /api/v1/cells/{cell_id}/drain` — `drain`
- `DELETE /api/v1/cells/{cell_id}` — `release` (also closes cached execution
  drivers for the cell via `CellClosableResolver`)

Workload lifecycle (all return `WorkloadObservation`):

- `POST /api/v1/cells/{cell_id}/workloads` — `submit`
- `POST /api/v1/cells/{cell_id}/workload-gangs` — `submit_gang`
- `GET .../workloads/{remote_id}` and `GET
  .../workload-gangs/{workload_id}/{remote_id}` — status reads
- `POST .../workloads/{remote_id}/stop` / `/retire`, and the gang
  equivalents — fenced mutations
- `POST .../workloads/{remote_id}/checkpoint` — launch a checkpoint hook
  (`CheckpointRequest.hook`, optional `output_uri`)

Observability and diagnostics:

- `GET .../workloads/{remote_id}/logs` — plain text (`PlainTextResponse`);
  the gang variant returns rank-prefixed logs from every member.
- `GET .../workloads/{remote_id}/log-page?after=&limit=` — one resumable
  page of structured `LogRecord` JSON. Deliberately a separate route from
  the plain-text one, whose byte contract is frozen for existing consumers.
- `POST .../workloads/{remote_id}/exec` — run one diagnostic command beside
  a live workload. `ExecRequest.command` is an argv tuple, **never a shell
  string** ("a string would be split by a shell on the host, which turns a
  diagnostic into an injection surface"), and `timeout_s` (≤600) is enforced
  by systemd as the unit's `RuntimeMaxSec`, so the bound survives the agent
  dying.
- `GET .../workloads/{remote_id}/process-status` — live process detail for
  one workload attempt.

Staging:

- `POST /api/v1/cells/{cell_id}/stage-actions` — one wire method with a
  named `CellStageAction` (`measure` / `stage` / `evict`) rather than three
  routes, so an agent that predates staging answers 404 for all three at
  once and a control plane learns "no staging executor" from a single probe.
  Returns `CellStageActionResult` with a `CellStageVerdict` (`ok` /
  `refused` / `inconclusive`) carried in a 200 body rather than a status
  code, because the two failures mean opposite things (terminal vs.
  transient) and every meaningful status on this wire is already spoken
  for. `size_bytes` and `digest` (`sha256:<64 lowercase hex>`, enforced by a
  model validator) describe bytes **read back off that host's disk** —
  never the size the asset declared, because the control plane charges its
  budget and scores placement against this number.

### 1.4 The service layer (`CellAgentService`)

`CellAgentService` (`service.py`) composes three independent backends:

- a `CapacityDriver` (`DEFAULT_CAPACITY_BACKENDS.create(settings.capacity_backend, ...)`),
- a `FleetStateStore` (durable, fenced state; `StaleFenceError` on lease loss),
- an `ExecutionDriverResolver` (`eyvan.worker.runtime.registry.ConfiguredExecutionResolver`)
  keyed by the capacity capabilities' `executor` name.

Key behaviors:

- **Readiness probe on observation.** `_with_execution_readiness` runs the
  resolver's `readiness(cell_id, observation)` for cells in
  `READY_CAPACITY_STATUSES`; a failed or negative probe downgrades the
  reported state to `CapacityStatus.EXECUTION_CONNECTING` (never a 500, so
  the controller keeps fencing scheduling without recovering a workload
  that may still be running on the VM).
- **Fenced submission.** `submit` requires
  `spec.metadata.eyvan_workload_id` and uses
  `spec.submission_id` (or the idempotency key) as the remote ID. The whole
  dance — retire predecessors, clear predecessor fences, launch, re-check
  the lease with `state.require_lease()` between steps — runs inside
  `state.workload_submission(...)`. If the lease is lost mid-flight
  (`StaleFenceError`) after a launch, the launched execution is retired
  before `StaleMutationError` propagates; if retirement also fails, the
  error says so explicitly.
- **Invalid workload requests** raise `InvalidWorkloadRequestError`, which
  subclasses both `HTTPException` (422) and `ValueError`, with both
  initializers called explicitly, so HTTP and in-process callers each keep
  their established contract (asserted by `test_native_gang_agent`).
- **Remote retirement.** `_retire_predecessor` retires a superseded
  generation on its own cell; when that cell lives behind another agent,
  `_retire_on_source_agent` dials it with `CellAgentClient`. A cell whose
  capacity is definitively lost (reclaimed Spot) cannot be running anything,
  so the fence is already satisfied and retirement is skipped — raising
  there used to block the *successor's* placement forever.
- **Staging gate.** `stage_asset` fences the request's generation against
  the agent's own durable record with **equality** — a stale generation can
  neither stage nor evict — before touching any disk. Host-local backends
  (`_HOST_LOCAL_EXECUTION_BACKENDS = {"native-local"}`) get a
  `LocalAssetStageExecutor`; an empty source allowlist is the off switch and
  reads as *no executor*, not as a refusal (a refusal is terminal and would
  poison the staging ledger fleet-wide). Remote backends
  (`_REMOTE_STAGING_EXECUTION_BACKENDS = {"eyvan-native"}`) resolve the
  cell's execution driver and either take it as an `AssetStageDriver`
  directly or, for multi-host cells, ask for a
  `MultiHostStageCapableDriver.asset_stage_driver(generation=..., fence=...)`
  view that fans each action to every machine, re-checking the fence
  (`_staging_fence`) between hosts — "a fan-out that began under a
  generation this agent no longer owns must stop rather than finish writing
  to somebody else's machines." Staging never advances the fence.
- **`exec_in_workload` two-gate check**: `capabilities().supports_exec` is
  the operator's admission decision ("may anyone do this on this fleet"),
  `ExecCapableExecutionDriver` is whether the backend can do it at all, and
  the two misconfigurations get different messages.
- **`log_page`** reports a backend without cursor support as 501 rather
  than an empty page: "this backend has no structured logs" and "this
  workload produced no output" are different answers.

## 2. Plane ↔ agent protocol

### 2.1 The wire contract

The protocol is versioned by the models in `eyvan/worker/agent/models.py`,
all rooted at `AgentModel` (`extra="forbid"`, `allow_inf_nan=False`).
Forbidden extras are a deliberate rolling-upgrade decision: a rolled-back
agent must never be handed a field it would persist and then fail to read
back. The compensating rule lives on the client: `CellAgentClient._wire`
serializes every request with `exclude_none=True`, so an omitted optional
field parses to exactly the same object on a matching agent, while an agent
that predates the field accepts the payload. Sending explicit `null` for an
unknown optional field can instead trigger `extra_forbidden` before any
provider call; that protocol refusal is not provider rejection.

New information therefore travels on **new routes** (which old agents
answer 404 and old clients never call) rather than new fields on
`extra="forbid"` models. `AgentBuild`, `worker-builds`, and
`stage-actions` all follow this pattern; `AgentCapabilities` documents the
rule explicitly for `capacity_backends` ("empty means the agent predates
the field and must be skipped, not rejected").

### 2.2 Fencing

Every mutating request carries a `MutationModel`:

```python
class MutationModel(AgentModel):
    idempotency_key: str  # 1..256 chars
    generation: int       # >= 0
    fencing_token: str    # 1..512 chars
```

The service projects it onto `MutationContext` (`_context`) and the durable
store enforces it: `StaleFenceError` / `StaleMutationError` surface as 409
`stale_mutation`, and a driver that cannot take its mutation lease answers
409 `driver_busy`. Generations fence *capacity*; the
`state.workload_submission` / `state.gang_workload_submission` context
managers fence *workload identity*, recording predecessors so a superseded
generation is retired before its successor launches — the mechanism that
guarantees two generations of one workload never run at once.

### 2.3 Gang protocol

A gang submission (`GangWorkloadRequest`) names `members` (≥2), each a
`GangWorkloadMember` with its own independently fenced `MutationModel`,
plus a shared `spec`. Model validators require: trimmed, unique cell IDs;
every member using the **same** `agent_endpoint`; and matching generations
across members. The coordinator must be the lexicographically first member
cell (`submit_gang` rejects otherwise). Gang mutations
(`GangWorkloadMutationRequest`) re-supply the member list, and
`_validate_gang_mutation_members` compares `(cell_id, agent_endpoint,
generation, fencing_token)` tuples — without idempotency keys — against
the stored fence: "a gang holds one lease per cell, so a request that does
not own every member fence checkpoints none of them."

### 2.4 Client timeouts

`CellAgentClient` (`client.py`) sizes timeouts by what the operation does,
with reasons documented per call:

- default 30 s for ordinary reads;
- 1800 s for `reconcile`, `apply_capacity`, `submit`, `submit_gang` —
  submission holds the agent's durable fence through remote launch, which
  on GCS is several CAS calls before the native worker responds, so a 30 s
  timeout can expire *after a job was accepted but before its remote ID was
  persisted*;
- 600–700 s for `retire` / `release`;
- staging per action: `_STAGE_TIMEOUTS = {measure: 60, stage: 1800, evict:
  300}` — measuring and evicting are a stat and an unlink and are how a
  hung agent is discovered, so they must not inherit a fetch's patience;
- exec derives its HTTP timeout from the command budget plus a 30 s margin,
  capped at 660 s (just past the wire's `le=600`), so a long diagnostic is
  not recorded as a rank error while still running.

The client optionally presents a mutual-TLS client certificate
(`MutualTlsFiles` → `client_ssl_context`); per the rollout documented in
`eyvan.identity.mtls`, clients present certificates at step 4, strictly
before any agent is moved to `required` at step 5 — reversing those two
locks a control plane out of the agents holding its running workloads.

## 3. Worker execution models

### 3.1 Native worker

`create_native_worker_app` (`eyvan/worker/runtime/native_worker.py`) is the
HTTP entry point on a native host. It requires a non-empty bearer secret;
`/healthz` is liveness-only and the execution routes are authenticated. The
worker launches the command with the operator-provided runtime environment,
workload identity, and Eyvan-owned topology variables. Caller-supplied
JAX/TPU topology overrides are removed by
`without_distributed_topology_overrides` before dispatch. A native worker can
serve a local executor, or the production systemd executor described below.

### 3.2 Container worker

`DockerContainerExecutor` (`runtime/container.py`) runs an OCI image through
Docker, but retains systemd's durable records and cgroup supervision. The
container name is deterministic (`docker_container_name`); stop/reconcile also
inspects Docker because the daemon does not put the container in the transient
service's cgroup. Images use `always`, `if-not-present`, or `never` pull
policies. Mounts must be absolute, non-root, non-dot paths with unique targets;
staged artifact mounts are read-only. The callable runner is staged into the
image and rejects shell composition instead of accepting an arbitrary shell
entrypoint.

### 3.3 Local process / local worker

`LocalProcessExecutor` (`runtime/local_process.py`) is the development and
in-process-test backend. It persists fenced JSON records and logs under
`EYVAN_LOCAL_RUNTIME_DIR`, `EYVAN_AGENT_STATE_DIR/runtime`, or
`~/.local/state/eyvan/runtime` (in that order). It records process identity
including a start marker, so a recycled PID is never enough authority to
signal a process. It is single-host and does not provide the isolation or
cgroup guarantees of systemd.

### 3.4 systemd transient units

`SystemdTransientServiceExecutor` (`runtime/systemd_transient.py`) creates one
transient service per submission. systemd owns the process cgroup, so stopping
a workload also stops background children; durable records distinguish a
missing unit from success. Journald is the log body store. Runtime failures
may be `FAILED`, `LOST`, or `INDETERMINATE`; an observation failure is not
silently converted into success, and recoverable indeterminacy can be resolved
by a later observation. Secret-bearing records suppress diagnostic output.

### 3.5 Choosing a runtime: the registry

`ExecutionRuntimeRegistry` registers `native-local`, `eyvan-native`, and the
registered-but-retired `ray-jobs` name. `ExecutionDriverResolver` selects and
caches by backend, cell, endpoint, and token fingerprint, and closes stale
cell drivers. `native-local` rejects remote endpoints. `eyvan-native` requires
a worker endpoint and bearer secret; multiple native endpoints resolve to a
`NativeSliceExecutor`. Unknown names fail explicitly, while `ray-jobs` tells
operators to resubmit natively. The cell's reported capabilities and
readiness, not a workload's suggestion, determine whether a backend is usable.

## 4. Callable lifecycle

### 4.1 Shipping callables (cloudpickle)

`dump_callable` serializes a callable with cloudpickle and records format
version 1, the producing Python major/minor version, and a SHA-256 digest.
`execute_callable` obtains the artifact through the configured URI transfer,
loads it, invokes it, and writes a self-validating result artifact. Supported
transfers include local paths/`file://`, HTTP(S) artifact URLs, and `gs://`
through the native worker's GCS identity. Result ownership is rank zero only;
`ReplicaContext` validates rank metadata and prevents every rank from writing
the same durable result.

### 4.2 sha256 / optional low-level HMAC verification

`load_callable` checks format and Python compatibility, then verifies the
payload digest before deserializing. `load_result` verifies the result envelope.
Workspace bundles are verified before extraction by
`workspace_entry.materialize`. These digest checks provide integrity, not
submitter identity.

The standalone SDK/API callables are unsigned: `EyvanCallableSubmitter`
(`eyvan/sdk/client/callables.py`) rejects caller signing keys and artifact
signatures, and the control-plane service independently rejects signed callable
submissions and caller-provided signing-key environments. This applies to
loopback as well as remote standalone planes.

The low-level `dump_callable` / `load_callable` and result helpers support
optional HMAC-SHA256 for a custom submitter that owns its producer/verifier
protocol. When a verifier key is configured, a matching signature is required;
setting `EYVAN_CALLABLE_SIGNING_KEY` on a worker consequently refuses ordinary
unsigned standalone artifacts. The standalone controller does not distribute a
producer/verifier key, so this is not a standalone worker-setup instruction or
an end-to-end signing guarantee. HMAC authenticates bytes relative to the shared
key; it does not sandbox code or identify an individual submitter.

### 4.3 Trusted-project policy

Cloudpickle deserialization executes pickle opcodes. It is therefore only for
trusted Eyvan projects and trusted artifact locations; a digest alone does not
make an untrusted callable safe. Standalone callable admission requires an
explicitly trusted project (`EYVAN_TRUSTED_CALLABLE_PROJECTS`); trust admission
and digest verification are not signature authentication. The runtime validates
the artifact before execution but does not sandbox Python code. In OCI mode the
staged callable bridge imports the dependency-light protocol rather than the
host `eyvan` package.

## 5. JAX multi-host bootstrap

### 5.1 The `_multihost_site` sitecustomize payload

`_multihost_site.py` is injected at `/opt/eyvan-native/multihost` and loaded as
`sitecustomize` before user code. It establishes the rank/process environment,
then invokes JAX's coordination initialization. The default
`GangInitializationPolicy` waits 600 seconds for formation and permits 900
seconds without a heartbeat; `fatal=True` makes a rank that never joins end
its own workload. These values can be changed only through the validated
policy/environment settings. The payload emits a structured initialization
outcome rather than claiming a cluster formed when it did not.

### 5.2 Distributed environment construction

`distributed_environment.py` defines the owned topology namespace. Eyvan
strips caller topology variables and writes authoritative process/rank values.
Native TPU cells use GCE metadata discovery (`GCE_TPU`) when TPU metadata is
present, including a multi-host single slice; other multi-host backends use an
explicit coordinator. The native coordinator port is 12345 and GPU rendezvous
is 12346; both must be reachable between gang hosts. GPU rank layout defaults
to `HOST_LAUNCHER` (opt-in `RANK_PROCESS`), and invalid layout configuration
is refused rather than silently defaulted.

## 6. Gangs and slices (TPU)

`NativeGangExecutor` coordinates independently fenced cells; members are
ordered deterministically and each endpoint retains its cell-scoped bearer
secret and mutation context. `NativeSliceExecutor` treats several hosts in one
cell as one lease, fence, and stage row while fan-out runs per host. Gang and
slice observations preserve rank, endpoint, remote ID, status, and checkpoint
information. Formation or observation failures are represented as
`INDETERMINATE` where the runtime cannot prove a terminal result. Managed
staging on a slice is unanimous: partial fetch, disagreement, unreachable hosts,
or a fence change yields an unproven result, never a false cache hit.

## 7. Secret materialization

`materialize_secrets` runs in the native worker's execution backend before
launch (`systemd_transient.py:_resolved_launch_environment`). Dispatch supplies
one short-lived capability per declared mount; the worker redeems it and places
the resulting text value in the private environment file. Capability scaffolding
(`EYVAN_SECRET_*`) is removed before the job starts and secrets are not put in
argv, unit definitions, or durable records. Names must be valid environment
identifiers, must not collide with existing variables, and may not use the
Eyvan-owned namespace. Fetches use a 30-second request timeout and bounded
retries; non-text, NUL/CR/LF-containing, or oversized values are refused.

## 8. Artifact and result flow

Managed staging is implemented by `LocalAssetStageExecutor` and exposed by
`stage-actions`. The source allowlist (`EYVAN_CELL_STAGE_SOURCES`) is empty by
default; an empty allowlist means no staging executor, not a terminal refusal.
Fetches are bounded and resumable, with progress fsynced before failure;
reported size and digest are read back from the worker disk. A digest mismatch
is quarantined. For a slice, every host must agree and complete before the
asset is considered present; the fence is checked before and during fan-out.
Callable/workspace transfers are content-addressed and reuse completed
extractions. Workspace extraction publishes a completion marker atomically;
partial extraction is discarded. Callable result artifacts are written through
the configured local, HTTP, or GCS transfer and validated on read.

## 9. Progress and log reporting

`report_progress` (`runtime/progress.py`) is an advisory in-job channel. It is
off when `EYVAN_PROGRESS_URL` and its managed-result token are absent, returns
`True` only when accepted, and swallows transport/control-plane failures so a
training loop cannot fail because monitoring is slow. It reports rank, attempt,
step, optional total, a message, and finite numeric metrics; invalid arguments
raise locally. The server truncates messages at 512 characters.

Live logs are backend-native: plain text is supplied by `logs`, while
`StructuredLogExecutionDriver.log_page` supplies resumable `LogRecord` pages.
Journald provides the body and opaque cursors; the control plane retains an
index and archives terminal-attempt logs so cell release does not destroy
history. Malformed, mismatched or oversized cursors produce typed refusals.
An attempt change reports `attempt_changed` and restarts at empty positions;
a legacy cursor with unknown topology may also need a reset. Bound cursors
reject a different topology digest; legacy cursors do not carry that full
binding. Fully bound v2 emission is opt-in through
`EYVAN_LOG_CURSOR_V2_EMISSION_ENABLED` (off by default). A backend without
structured logs is reported as 501, distinct from a workload with no output.

## 10. Node health

`node_health.py` recognizes only host-attributable accelerator faults:
IOMMU-group or VFIO-device unavailability, TPU system/initialization failure,
and an absent accelerator. It returns a bounded evidence excerpt and stable
`HostFaultCode`; the control plane decides whether to quarantine/replace the
host. Resource-busy, OOM, coordinator/network, generic socket, and generic
`XlaRuntimeError` messages are deliberately not host faults because they can
be workload or transport failures. Execution readiness is separately probed by
`HealthCheckedExecutionDriver.health`; a negative probe reports
`EXECUTION_CONNECTING` rather than recovering a possibly running workload.

## 11. Supporting utilities

### 11.1 HTTP retry

`http_retry.py` is shared by callable transfers, secret fetches, progress, and
in-job child submission. It makes three total attempts, retries only 408, 425,
429, 500, 502, 503, and 504, honors `Retry-After` up to 30 seconds, and uses
exponential delays otherwise (0.25, 0.5, then capped at 2 seconds). It does
not retry authorization, validation, conflict, or not-found responses.

### 11.2 Dependency entry

`dependency_entry.prepare` content-addresses one environment per normalized
requirement set and interpreter (cache-format version, resolved interpreter
path, Python version and the parent site-packages it layers over, so an entry
built by another interpreter or an older format is never reused), locks
concurrent installers, layers a virtualenv over host
site-packages (preserving accelerator libraries), and writes a completion
marker only after installation succeeds. A failed install removes the partial
environment and fails the job. The default cache is `/var/lib/eyvan/runtime-envs`,
falling back to `./.eyvan-runtime-envs` when unwritable.

### 11.3 In-job helpers

`in_job.py` exposes the workload capability and `submit_child`. It is absent
outside managed dispatch; rank zero is the default submitter, while
`all_ranks=True` explicitly opts every rank in. Child submission uses a
workload-scoped token and idempotency key, and unlike progress, refusal or
transport failure raises `ChildSubmissionError`.

### 11.4 Workspace entry

`workspace_entry.py` verifies the digest of a content-addressed bundle and,
when a low-level verifier key is configured, requires its matching signature
before extraction. Standalone SDK bundles are unsigned. It then changes
directory and `exec`s the user's command so PID, signals, exit status, cgroup
accounting, and process introspection describe the workload.
Only a directory carrying `.eyvan-extraction-complete` is reused; a partial
extraction is rebuilt.

## 12. Operational limits and invariants

The public agent model bounds diagnostic commands to 600 seconds (the client
allows a 30-second HTTP margin), mutation idempotency keys to 256 characters,
fencing tokens to 512, gang membership to at least two members, and reported
runtime process count to 10,000. Secret values are bounded by
`eyvan.secrets.MAX_SECRET_BYTES`; log cursors are bounded to 16 MiB and 10,000
positions, with individual cursor values bounded to 8 MiB. Stage requests carry
non-negative byte budgets and bounded asset/detail/path fields. Treat these as
wire contracts: raise or refuse invalid values rather than truncating security,
identity, or cursor fields.
