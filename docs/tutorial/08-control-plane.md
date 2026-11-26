# Part 8 — The control plane

`eyvan-server` is the plane: a FastAPI transport (`server/api/app.py`), an
application facade (`server/service/control_plane.py`), a leader-elected
controller (`server/controller/`), and durable state (`server/persistence/`).
Deep reference: `docs/reference/control-plane.md` and `docs/reference/persistence.md`.

## The separation that organizes everything

Eyvan separates **decisions**, **durable state**, and **side effects**:

- `ControlPlaneService` — the application boundary. It authorizes requests against
  the principal and org/project scope, validates lifecycle transitions and expected
  versions, applies idempotency, and commits records + audit events + durable
  outbox commands atomically. `schedule_once()` reads a snapshot and returns a
  plan; it provisions nothing itself.
- `Repository` / `RepositorySession` (`persistence/base.py`) — the persistence
  contract: an atomic read/write view with typed accessors per record family.
  Implementations: SQLAlchemy (PostgreSQL), owner-only local JSON, in-memory.
- `EyvanController` (`controller/__init__.py`) — a composition of 15 concern
  mixins (Fleets, Scheduling, Outbox, CapacityCommands, WorkloadCommands,
  Preemption, Settlement, Services, Observation, LogArchival, Staging, Expansion,
  WebhookDelivery, …). It turns service decisions into durable mutations, claims
  outbox work, and performs provider/cell-agent I/O — generally **outside**
  database transactions, with results fenced and persisted afterward.

## The leader loop and its phases

`_ControllerLoops.run()` runs two loops: the scheduler loop (leader-only) and the
outbox loop (durable command delivery, independent — it never waits behind a slow
scheduling phase).

The leader phase order is declared in `_leader_phases()`
(`server/controller/loops.py`):

1. unmanaged-capacity recovery, stale-build recycling, capacity-pool reconciliation
2. cell observation, expired-lease quarantine, workload observation
3. orphan settlement (workloads, leases, coordinators) — *settlement precedes
   scheduling so a finished workload cannot hold capacity*
4. DAG-node/group/expansion-wave release and placement (`requires` fresh
   observation)
5. service health probing, replica lifecycle, autoscaling, service reconciliation
6. workload scheduling
7. cell billing and live-log archival
8. retention, superseded-result marking, artifact collection, metrics export,
   grant expiry, asset staging/eviction, webhook delivery — *staging precedes
   eviction*

Structural rules: leadership is re-acquired/verified before **every** phase
(losing the lock mid-cycle is an ordinary demotion, not an error); phase failures
are contained so unrelated phases continue; a phase can declare `requires` — the
placement/release/scheduling phases require this cycle's `observe-cells` and
`observe-workloads` to have succeeded, and are skipped rather than committing
against stale inventory. Per-phase intervals are advanced before the call so a
persistently failing phase cannot pin the tick.

Leadership is a session-scoped PostgreSQL advisory lock
(`PostgresAdvisoryLeader`, `persistence/leader.py`) on a dedicated autocommit
connection; renewal checks `pg_locks` for actual ownership. Use a direct
connection or session pooling — PgBouncer in transaction-pooling mode cannot
safely carry a session-scoped lock. `SingleProcessLeader` is the local
implementation.

## State machines

From `server/service/state.py` (enforced by `ensure_workload_transition` /
`ensure_lease_transition`):

```text
Workloads:
PENDING -> ADMITTED | CANCELLED | FAILED
ADMITTED -> LEASED | CANCELLING | FAILED
LEASED -> DISPATCHING | CANCELLING | FAILED
DISPATCHING -> RUNNING | CANCELLING | FAILED
RUNNING -> CHECKPOINTING | SUCCEEDED | CANCELLING | FAILED
CHECKPOINTING -> RUNNING | SUCCEEDED | CANCELLING | FAILED
CANCELLING -> SUCCEEDED | CANCELLED | FAILED
FAILED, CANCELLED -> PENDING   (explicit retry)
SUCCEEDED                     no outgoing transition

Leases:
PENDING -> ACTIVE | RELEASED | EXPIRED | QUARANTINED
ACTIVE -> REVOKING | RELEASED | EXPIRED | QUARANTINED
REVOKING -> RELEASED | EXPIRED | QUARANTINED
QUARANTINED -> ACTIVE | RELEASED | EXPIRED     (quarantine is recoverable)
RELEASED, EXPIRED                              terminal
```

Terminal workload settlement archives logs best-effort, then releases and bills
the lease. Orphan sweeps are separate from ordinary observation so a finished
workload or leftover lease cannot retain capacity indefinitely. A service
coordinator is intentionally a `PENDING` parent whose children are the
independently terminal attempts — service restart needs no terminal-to-pending
transition.

## The durable outbox

The outbox is the side-effect boundary. A service commit records intent first;
the outbox loop claims commands (`READ COMMITTED` + `FOR UPDATE SKIP LOCKED`), renews
long operations, executes the topic, and settles each as success, retryable
failure, or terminal/stale. Commands for one aggregate serialize; unrelated
aggregates run concurrently. This is **at-least-once** delivery — idempotency keys
and generation/fencing checks are the safety boundary, not exactly-once execution.

## Persistence

Three repository backends (`persistence/`):

| backend | use | notes |
|---|---|---|
| `InMemoryRepository` | tests/dev | copy-on-write, same transaction machinery, no durability |
| `LocalJsonRepository` | zero-config single server | owner-only JSON doc (`~/.eyvan/control-plane.json`); tmpfile + fsync + atomic replace + dir fsync; an unconfirmable replace is refused, never silently treated as a failed write; a `.lock` flock means a second server gets `LocalControlPlaneInUseError` |
| `SqlAlchemyRepository` | production multi-replica | PostgreSQL; `SERIALIZABLE` service transactions; rows carry the full Pydantic record in `data` with typed indexed columns for hot filters |

Concurrency details worth knowing: serialization failures (SQLSTATE 40001) and
deadlocks (40P01) retry at session acquisition (up to six attempts, bounded
backoff); commit-time conflicts surface as `TransientRepositoryError` because a
caller body cannot safely be replayed. Table families: core execution
(`workloads`, `cells`, `capacity_pools`, `leases`), governance (`organizations`,
`projects`, `policies`, `quotas`, `budgets`, `role_grants`), delivery/history
(`idempotency_keys`, `outbox`, `audit_events`), execution evidence
(`progress_reports`, `workload_topologies`, `cell_stages`,
`credential_rotations`, `expansion_admissions`, `service_replica_health`,
`service_decisions`), artifacts/federation (`managed_artifacts`,
`artifact_references`, `plane_peers`, budget/quota grants).

Migrations: packaged Alembic scripts, current revision `0024_research_plans`
(`eyvan-migrate` entry point). The repository rejects a stamped revision that does
not equal the build's head (`SchemaRevisionMismatch`); upgrades run during
explicitly acknowledged maintenance with writers quiesced; **a failed rollout is
repaired forward — Alembic downgrade is not the rollback procedure**.

## Named invariants

`server/invariants.py` gives stable names to fleet-wide contracts, checked by
`check_fleet()` in one consistent transaction:

| name | contract |
|---|---|
| I1 `NO_STRANDED_CELLS` | every live cell has a capacity pool explaining it |
| I2 `NO_LEASES_OUTLIVING_WORKLOADS` | no active lease belongs to a missing/terminal workload |
| I3 `NO_STUCK_OUTBOX` | no command undelivered past the 900 s stuck threshold |
| I4 `NO_RUNNING_WORKLOAD_WITHOUT_A_LEASE` | lease-holding states always have an active lease |
| I5 `BUDGET_MATCHES_SETTLED_COST` | budget spend equals settled lease cost within tolerance |
| I6 `BUDGET_SPEND_IS_MONOTONE` | spend/revision never regress within a period (checked via `compare_snapshots`) |
| I7 `ARTIFACT_STORAGE_IS_BOUNDED` | managed artifact bytes stay under the ceiling |
| I8 `AUDIT_RETENTION_PRUNES` | oldest audit event stays inside the retention window |
| I9 `NO_LEASE_ON_A_MISSING_CELL` | no active lease references a deleted cell |

## Managed services

A service policy is metadata on a non-schedulable parent: desired replicas,
revision, roll availability/surge bounds, restart backoff, optional health policy
and autoscaling bounds. `plan_service()` cancels on scale-to-zero, rolls stale
revisions within bounds, starts missing children, scales in excess, or waits for
backoff. The controller retires a provably unhealthy replica once, fenced by child
generation; the successor starts after the old child is terminal.
