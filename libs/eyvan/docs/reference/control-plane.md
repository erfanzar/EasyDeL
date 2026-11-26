# Control-plane internals

This is a source-level reference for the control plane implemented under
`eyvan/server/`. It describes durable contracts and orchestration visible in the
current source; it is not a claim that every provider or deployment mode has
been exercised in a live environment.

## Architecture overview

Eyvan separates **decisions**, **durable state**, and **side effects**:

- `ControlPlaneService` (`server/service/control_plane.py`) is the application
  boundary. It authorizes requests, validates lifecycle transitions, applies
  idempotency, and commits records, audit events, and durable outbox commands
  in repository transactions. `schedule_once()` reads a snapshot and returns a
  scheduling plan; it does not itself provision or dispatch anything.
- `Repository` and `RepositorySession` (`server/persistence/base.py`) are the
  persistence contract. A session is an atomic read/write view; the service
  uses it to update workloads, leases, cells, pools, budgets, artifacts, and
  outbox records. Implementations include SQLAlchemy, the owner-only local JSON
  store, and an in-memory test store. The local store copy-on-writes a versioned
  document and validates aggregate identities; SQL schema compatibility is
  rejected when the revision does not match the packaged schema.
- `EyvanController` (`server/controller/__init__.py`) is a composition of
  concern-specific mixins. It turns service decisions into durable mutations,
  claims outbox work, and performs provider/cell-agent I/O. External I/O is
  generally outside database transactions; the resulting observation or
  command outcome is then fenced and persisted.
- Capacity is represented by pools, cells, and leases. Workloads are placed by
  the scheduling service, while provider capacity and cell-agent operations are
  delivered asynchronously through the outbox. Commands are idempotent and
  carry aggregate generation and fencing information, so a stale command is
  discarded or recovered rather than applied to a replacement.

The controller also owns housekeeping: observation, settlement and billing,
service health/lifecycle, log archival, artifact staging/collection, retention,
metrics, budget-grant expiry, and webhook delivery. These concerns share the
same durable state but have independent phase failure and cadence policies.

## Leader election and the declared-phase leader loop

`_ControllerLoops.run()` starts two independent loops:

1. `_run_scheduler_loop()` runs the global phase schedule only while this
   replica holds leadership.
2. `_run_outbox_loop()` dispatches durable commands independently of global
   reconciliation. Outbox delivery therefore does not wait behind a slow
   scheduling/provider phase.

The schedule is declared by `_leader_phases()` in
`server/controller/loops.py`. In order, its phases are:

1. unmanaged-capacity recovery and stale-build recycling;
2. capacity-pool reconciliation;
3. cell observation, expired-lease quarantine, workload observation, and
   orphan/coordinator settlement;
4. DAG-node release, group placement, admitted-group release, and expansion-wave
   release;
5. service health probing, replica lifecycle reconciliation, autoscaling, and
   ordinary service reconciliation;
6. workload scheduling;
7. cell billing and live-log archival;
8. retention, superseded-result marking, artifact collection, metrics export,
   grant expiry, asset staging/eviction, and webhook delivery.

The exact order is part of the behavior: observation precedes decisions that
commit against observed state; settlement and release precede scheduling; health
probing precedes replica retirement; lifecycle precedes autoscaling; and
staging precedes eviction. Per-phase intervals are tracked by name. A phase
with no interval runs each cycle; phases marked `zero_interval_disables` are
skipped when configured with zero.

Before every phase the controller re-acquires/verifies leadership. Losing the
lock mid-cycle returns without running later phases and is treated as an
ordinary demotion. Phase exceptions are contained so unrelated reclamation and
settlement continue. A phase can declare `requires` (currently the placement,
release, and scheduling phases require fresh `observe-cells` and
`observe-workloads`); if a required observation failed in this cycle, the
dependent phase is skipped instead of committing against stale inventory.

`PostgresAdvisoryLeader` (`server/persistence/leader.py`) holds a
session-scoped PostgreSQL advisory lock on a dedicated autocommit connection.
Renewal checks `pg_locks` for ownership, not merely connection health; loss or
failure drops the connection and demotes the replica. Direct connections or
session pooling are supported. Transaction-pooling proxies are a deployment
constraint because session-scoped lock ownership cannot safely follow a
transaction between backend sessions. `SingleProcessLeader` is the local
non-distributed implementation.

**Runtime note:** the phase semantics above are read from the implementation.
Provider latency, proxy behavior, and failover timing are not independently
verified by this document.

## Controller mixins

`EyvanController` has one mixin per operational concern (the MRO is declared in
`server/controller/__init__.py`):

| Mixin | Responsibility |
| --- | --- |
| `_ControllerLifecycle` | Construction, shutdown, agent-client caching, per-cell locks, fleet capacity gates, and common mutation/fence helpers. |
| `_ControllerFleets` | Fleet/pool/cell administration, capacity-pool reconciliation, autoscale cell recovery, and manual leader-gated reconciliation. |
| `_ControllerScheduling` | Commit placement and preemption plans returned by the scheduling service. |
| `_ControllerOutbox` | Claim, heartbeat, ordered-per-aggregate, concurrent-across-aggregate delivery and recovery of durable commands. |
| `_ControllerCapacityCommands` | Provider apply, drain, and release operations, including superseded-apply recovery. |
| `_ControllerWorkloadCommands` | Render and dispatch workload attempts (including gang membership), cancellation, checkpoint, fencing, and stale-build refusal. |
| `_ControllerPreemption` | Durable checkpoint/stop/finalize/rollback workflow for preemption victims. |
| `_ControllerSettlement` | Terminal release, orphan cleanup, billing, retention, service reconciliation, metrics, and artifact/grant housekeeping. |
| `_ControllerServices` | Health probes, fenced unhealthy-replica retirement, and bounded autoscaling decisions. |
| `_ControllerObservation` | Cell/workload polling, fault quarantine, DAG/group release, and log/status reads. |
| `_ControllerLogArchival` | Incremental archival of logs from live attempts. |
| `_ControllerStaging` | Managed-asset stage, measure, confirm, restore, and eviction workflows. |
| `_ControllerExpansion` | Admission and placement of bounded expansion waves and groups. |
| `_ControllerWebhookDelivery` | At-least-once delivery of durable webhook events with a per-subscription cursor. |

The split is organizational, not a second runtime object model: the mixins are
cooperative methods on one controller and use the shared repository, service,
principal, locks, settings, and capability clients initialized by lifecycle.

## Service layer and boundaries

`ControlPlaneService` is deliberately the narrow boundary between API/commands
and durable control-plane state. Public operations authorize against the
principal and organization/project scope, validate inputs and expected
versions, then perform the state transition and append audit/idempotency/outbox
records atomically where appropriate. It exposes workload submission, batch/DAG/
group expansion, cancellation/checkpoint/retry, fleet/pool/cell operations,
lease and webhook APIs, policy/quota/budget administration, service operations,
read-only queries, usage/cost reports, artifact/secret/progress operations, and
settlement helpers.

Pure policy components stay below that facade: `CoreSchedulingService` produces
placement/preemption decisions; `service_policy()` and `plan_service()` turn a
service parent and its child records into replica intent; service autoscaling
and health functions produce decisions from durable observations. The facade
commits those decisions and enforces authorization and generation checks. The
controller then performs side effects through provider drivers, cell agents,
health probes, artifact/stage executors, and webhook transports.

The durable outbox is the side-effect boundary. Submission or a controller
commit records intent first; the outbox loop claims it, renews long operations,
executes the topic, and settles the command as success, retryable failure, or
terminal/stale. Commands for one aggregate remain serialized while unrelated
aggregates may run concurrently. This is at-least-once delivery, not exactly-once
execution; idempotency keys and generation/fencing checks provide the safety
boundary.

## State machines

### Workloads

The transitions in `server/service/state.py` are:

```text
PENDING -> ADMITTED | CANCELLED | FAILED
ADMITTED -> LEASED | CANCELLING | FAILED
LEASED -> DISPATCHING | CANCELLING | FAILED
DISPATCHING -> RUNNING | CANCELLING | FAILED
RUNNING -> CHECKPOINTING | SUCCEEDED | CANCELLING | FAILED
CHECKPOINTING -> RUNNING | SUCCEEDED | CANCELLING | FAILED
CANCELLING -> SUCCEEDED | CANCELLED | FAILED
FAILED, CANCELLED -> PENDING      (retry)
SUCCEEDED -> no outgoing transition
```

`TERMINAL_WORKLOAD_STATES` is `{SUCCEEDED, FAILED, CANCELLED}`. A service
coordinator is intentionally a `PENDING` parent; its ordinary child workloads
are the independently terminal attempts, so service restart does not require a
terminal-to-pending transition on the same workload.

### Leases

```text
PENDING -> ACTIVE | RELEASED | EXPIRED | QUARANTINED
ACTIVE -> REVOKING | RELEASED | EXPIRED | QUARANTINED
REVOKING -> RELEASED | EXPIRED | QUARANTINED
QUARANTINED -> ACTIVE | RELEASED | EXPIRED
RELEASED, EXPIRED -> terminal
```

`ACTIVE_LEASE_STATES` includes `PENDING`, `ACTIVE`, `REVOKING`, and
`QUARANTINED`; quarantine is recoverable and therefore is not treated as death.
Terminal workload settlement archives logs best-effort, then releases and bills
the lease. Orphan sweeps are deliberately separate from ordinary observation
so a finished workload or leftover lease cannot retain capacity indefinitely.

### Managed services

A service policy is metadata on a non-schedulable parent: desired replicas,
revision, roll availability/surge bounds, restart backoff, optional health
policy, and optional autoscaling bounds/cooldown/step. `plan_service()` either
cancels on scale-to-zero, rolls stale revisions within the availability bounds,
starts missing current-revision children, scales in excess children, or waits
for restart backoff. The controller retires a provably unhealthy/stale replica
once, fenced by child generation; normal service reconciliation starts the
successor after the old child is terminal.

## Named invariants

`server/invariants.py` gives stable names used by point-in-time fleet checks and
longitudinal snapshot comparisons:

| Name | Contract |
| --- | --- |
| **I1 `NO_STRANDED_CELLS`** | A live cell (`REGISTERING`, `READY`, `DRAINING`, or `UNAVAILABLE`) has an existing capacity pool explaining it. |
| **I2 `NO_LEASES_OUTLIVING_WORKLOADS`** | No active lease (`PENDING`, `ACTIVE`, `REVOKING`, or `QUARANTINED`) belongs to a missing or terminal workload. |
| **I3 `NO_STUCK_OUTBOX`** | No pending, processing, or failed outbox command remains undelivered past the 900-second stuck threshold. |
| **I4 `NO_RUNNING_WORKLOAD_WITHOUT_A_LEASE`** | A workload in a lease-holding state has an active lease. The checked states are `LEASED`, `DISPATCHING`, `RUNNING`, `CHECKPOINTING`, and `CANCELLING`. |
| **I5 `BUDGET_MATCHES_SETTLED_COST`** | Budget spend matches the sum of settled lease costs, within the implementation tolerance. |
| **I6 `BUDGET_SPEND_IS_MONOTONE`** | Within one budget period spend and revision do not regress; a period change must be proved as a forward revision/period transition. |
| **I7 `ARTIFACT_STORAGE_IS_BOUNDED`** | Longitudinal managed-artifact bytes remain under the supplied ceiling. |
| **I8 `AUDIT_RETENTION_PRUNES`** | When retention checking is enabled, the oldest audit event is not beyond the retention window plus the implementation's two-window slack. |
| **I9 `NO_LEASE_ON_A_MISSING_CELL`** | No active lease references a cell that no longer exists (non-`DELETED`). |

`check_fleet()` reads the relevant state in a consistent transaction and
returns subject-specific violations. `compare_snapshots()` is required for I6
and I7 because monotonicity and bounded growth are properties of observations
over time, not of one read. These checks are source-backed contracts; their
ability to detect a violation in a particular production deployment depends on
the configured repository and metrics/soak observer, which is not verified here.

## Source map

- `eyvan/server/controller/loops.py` — phase declaration and loop semantics
- `eyvan/server/controller/__init__.py` — controller composition
- `eyvan/server/controller/_support.py` — `LeaderPhase`, gates, and shared fences
- `eyvan/server/service/control_plane.py` — application facade
- `eyvan/server/service/state.py` — workload and lease transitions
- `eyvan/server/persistence/base.py` — repository transaction contract
- `eyvan/server/persistence/leader.py` — PostgreSQL leader ownership
- `eyvan/server/persistence/local.py` — local durable document store
- `eyvan/server/invariants.py` — named fleet invariants and checks
