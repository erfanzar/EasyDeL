# Part 10 — Workers: the cell agent and the native runtime

Source: `eyvan/worker/` — `agent/` (the cell agent API and client) and `runtime/`
(the native worker and its support modules). Deep reference:
`docs/reference/worker-runtime.md`.

## The cell agent (`eyvan-agent`)

One agent serves one **backend family** through a single `CapacityDriver`
(`worker/agent/service.py`). The GCP router can serve `gcp-tpu`, `gcp-compute`, and
`gcp-attach` pools behind the same endpoint. Fleet validation compares backend
families, not just backend names. The plane and agent are separate processes:
either can restart independently.

- Transport: FastAPI app from `create_agent_app()` (`worker/agent/app.py`).
- Auth: bearer `EYVAN_AGENT_SHARED_SECRET`, compared with `hmac.compare_digest`.
  Starting without the secret refuses unless `--allow-unauthenticated`, which logs
  a loud warning that every endpoint is reachable without credentials — local
  development only.
- OpenAPI docs are disabled on the agent.
- Runtime records (pid/lock/log) live under `~/.eyvan/cell-agent-*`.

### The error taxonomy is the point

The agent's deliberate answers are status codes with machine-readable `code`s, and
the plane's client (`worker/agent/client.py`) distinguishes every one:

| status | raised by | client error | meaning |
|---|---|---|---|
| 409 | `StaleMutationError` | `CellAgentStaleMutationError` | stale generation/fence — retry with fresh state |
| 409 | `DriverBusyError` | `CellAgentDriverBusyError` | driver is mid-mutation |
| 400 | `ValueError` | `CellAgentInvalidConfigurationError` | rejected provider configuration — **terminal** |
| 501 | `UnsupportedCapabilityError` | `CellAgentUnsupportedCapabilityError` | backend does not have that capability — **terminal** |
| 502 | `CapacityProviderError` | `CellAgentCapacityProviderError` | provider failure, operator-safe detail |
| other 5xx / transport error | agent itself failing | `CellAgentUnavailableError` | **nothing was learned** |

Agent unavailability is inconclusive: it does not establish that provider
capacity is absent or failed. `CellAgentUnavailableError` preserves that boundary.
`CellAgentClient` sizes timeouts by the operation.

The agent also probes execution readiness: `_with_execution_readiness` runs the
resolver's readiness probe for ready cells, and a failed probe **downgrades** the
reported state to `EXECUTION_CONNECTING` — never a 500, so the controller keeps
fencing scheduling without recovering a workload.

## The native worker

`worker/runtime/native_worker.py` is the authenticated HTTP transport for one
execution worker, including `POST /api/v1/cells/{cell_id}/stage-actions` — the
second hop of managed asset staging. Deployment order is **native workers → agents
→ controller**; a 404 from an old worker propagates as "no staging executor" —
absence of a capability, never a failure.

The staging executor (`stage_executor.py`) does bounded fetches with fsynced
resume points, verifies size and digest from disk, quarantines on mismatch, and
enforces a source allowlist (`EYVAN_CELL_STAGE_SOURCES`, empty by default —
otherwise the route would be an arbitrary-read primitive on accelerator hosts).

Sibling modules and their jobs:

| module | job |
|---|---|
| `local_worker.py` | detached worker for the `native-local` executor — flock-guarded subprocess, SIGTERM→SIGKILL with 30 s grace |
| `native_gang.py` / `native_slice.py` | multi-cell (gang) and slice execution |
| `native_remote.py` / `native_bootstrap.py` | remote worker bring-up |
| `container.py` / `oci_callable_stager.py` / `oci_callable_runner.py` | the OCI model: run callables inside the fleet's container image |
| `secret_materialization.py` | resolve secret *references* into attempt env vars |
| `results.py` | write result manifests back through the artifact store |
| `in_job.py` | `in_job_context`, `submit_child` — in-job capability surfaces |
| `progress.py` | `report_progress` — advisory progress via `EYVAN_PROGRESS_URL` + token; stdlib-only, silent when nothing listens; rank from `EYVAN_DISTRIBUTED_PROCESS_ID` / `JAX_PROCESS_ID` |
| `node_health.py` / `systemd_transient.py` / `host_accelerators.py` / `http_retry.py` | node health, systemd units, host accelerator discovery, bounded HTTP retries |

## Two provisioning models, two allocation modes

**Runtime models** (`docs/runtimes.md`):

| | venv model | OCI model |
|---|---|---|
| Python | the fleet, via `runtime.python` | your image |
| libraries | `runtime.init_commands` + per-job `dependencies` | image packages plus per-job dependencies (`--with`); workspaces (`--workspace`) are supported |
| container | none — native venv at `/opt/eyvan-native/venv` | `runtime.container.image` |
| typical use | fast iteration | pinned, reproducible training runtime |

The interpreter contract is checked **before provisioning** when `runtime.python`
is declared: a callable pickled on a different `major.minor` is refused at
submission, not on the accelerator.

**Allocation modes** (independent of runtime):

- **elastic** (default) — `pods` is a worker count satisfied from any combination
  of same-type cells: warm slices pack first (biggest first — fewest hardware
  worlds), only the deficit provisions. Resizing 2→4 keeps the two warm slices.
  One lease group, one hardware world per cell — framework-level parallelism.
- **atomic** (`TpuConfig(..., atomic=True)`) — one exact-shape provider resource,
  one hardware world spanning every pod; required when the math needs a single
  JAX world. Resizing rebuilds from zero — a provider topology cannot grow — and
  the plane drains now-unusable leftovers instead of billing them idle.

All-or-nothing holds in both: nothing is leased unless the full worker count is
covered, so a deficit that never grants holds no warm cell hostage.

## Logs

Multi-host logs are flattened into a global rank order with a stable cursor
(design record `docs/design/0002-global-log-ranks-and-cursors.md`). `job.logs()`
and `eyvan workload logs <id>` render the flattened stream; a rank lost mid-stream
is reported unavailable on the next page rather than silently truncating history
(proof P2). Logs from live attempts are archived incrementally by the controller's
`archive-live-logs` phase; for kinds like logs, the managed copy is the only
durable copy once journald dies with the cell.
