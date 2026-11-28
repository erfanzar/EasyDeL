# Persistence & schema reference

Eyvan separates **control-plane state** (the scheduler's durable view of tenants,
workloads, capacity, policy, and delivery) from **agent fleet state** (the
fenced, provider-facing document that cell agents mutate).  The control-plane
repository is defined in `eyvan/server/persistence/base.py`; its record types
are in `eyvan/records.py`.  This page describes the shipped implementations and
schema policy as implemented by the source.

## Contents

1. [Repository contract](#repository-contract)
2. [Backends](#backends)
3. [Transactions and concurrency](#transactions-and-concurrency)
4. [Tables and model families](#tables-and-model-families)
5. [Migrations and compatibility](#migrations-and-compatibility)
6. [Leadership](#leadership)
7. [Control-plane versus agent fleet state](#control-plane-versus-agent-fleet-state)

## Repository contract

`Repository` is an async lifecycle and transaction surface.  A
`transaction()` yields a `RepositorySession`, an atomic read/write view.  The
session provides typed `get_*`, `list_*`, `put_*`, and the explicitly supported
delete/aggregate operations for every persisted record family; callers do not
write SQL or manipulate backend-specific rows.  Lists have stable ordering and
support the documented scope/state filters.  A missing record is a real
`None`/empty result, not permission to reconstruct durable evidence (notably
attempt topologies and staged assets).

The repository also exposes `open`, `close`, `ping`, event waiting, and outbox
claiming.  `wait_for_events` observes committed `AuditEvent`s.  Outbox claiming
is atomic and returns fenced, at-least-once commands; terminal succeeded or
discarded commands are the only entries eligible for pruning.  SQL-backed
repositories additionally implement bounded, read-only ad-hoc queries;
state-file repositories raise `ReadOnlyQueryUnsupported` rather than pretending
that an unreadable store is empty.  Typed model decoding is strict; PostgreSQL
list decoders log and skip an unreadable/forward-versioned row rather than
bringing down the whole fleet scan.

## Backends

### In-memory (`InMemoryRepository`)

`InMemoryRepository` (`persistence/memory.py`) is a copy-on-write repository for
focused tests and development.  It has no external durability or inter-process
coordination.  It nonetheless uses the same serializable transaction machinery
as the local repository and supports event waits, so service behavior can be
tested without PostgreSQL.  State disappears when the process exits.

### Local JSON (`LocalJsonRepository`)

`LocalJsonRepository` (`persistence/local.py`) is the zero-configuration,
single-server control-plane store.  It serializes the complete typed state as
an owner-only JSON document (default path is supplied by the local profile),
validates schema and duplicate identities on load, and keeps deterministic
ordering.  Each changed transaction is written through a temporary file,
`fsync`ed, atomically replaced, and followed by a parent-directory `fsync`.
An unconfirmable replace is reported as an ambiguous commit and the store is
refused until reloaded; it is never silently treated as a failed write.

A companion `.lock` file is held with a non-blocking `fcntl.flock`.  Thus only
one local server process may own a state document; a second server receives
`LocalControlPlaneInUseError`.  The document has its own versioned format (the
supported versions are 1 through 11, selected to preserve all collections)
and is not the PostgreSQL schema.

### PostgreSQL (`SqlAlchemyRepository`)

`SqlAlchemyRepository` (`persistence/sqlalchemy.py`) uses SQLAlchemy async
sessions and JSON `data` columns, with frequently filtered/accounted fields
materialized as typed indexed columns.  Ordinary service transactions use
`SERIALIZABLE` isolation.  Outbox claim transactions intentionally use
`READ COMMITTED` with `FOR UPDATE SKIP LOCKED`, so active-active delivery
workers cannot claim the same command while avoiding unnecessary serialization
failures.  PostgreSQL is the production multi-replica backend and the only
backend that can safely serve the arbitrary read-only query surface.

## Transactions and concurrency

The in-memory/local implementation takes an async process lock and makes a
structural copy of the state's containers, including nested containers; record
models themselves are shared with committed state.  It yields the transaction
view and publishes a changed snapshot only after `_persist` succeeds.  A changed
snapshot increments its revision; audit-event waiters are notified only after
commit.

Treat every fetched record and all of its nested values as read-only.  Frozen
model attributes do not freeze dictionaries such as `labels`, `metadata`,
`spec.environment`, or outbox payloads.  Mutating them in place changes the
shared committed record, escapes rollback, and can evade change detection.
Instead, construct new nested values and replace the record with
`model_copy(update=...)` and the session's matching `put_*` method.  For example,
inside a transaction with an existing `cell-1`:

```python
cell = await transaction.get_cell("cell-1")
assert cell is not None
labels = {**cell.labels, "phase": "updated"}
updated = cell.model_copy(update={"labels": labels})
await transaction.put_cell(updated)
```

For nested models, replace the changed nested model as well as its parent; do
not mutate a mapping that the replacement still shares with the original.
Provided callers follow this replacement-only contract, an exception from the
caller aborts the transaction and leaves the prior snapshot unchanged.
`StateRepository.verify_record_immutability` is a test-only diagnostic guard,
disabled in production: it detects in-place edits before persistence or adds a
diagnostic note to a caller exception, but does not restore already mutated
shared records or provide production isolation.

PostgreSQL commits one service operation in one database transaction.  Row reads
used for read/modify/write are acquired with `FOR UPDATE`; uniqueness and
record identity are enforced by primary keys, unique constraints, and scoped
composite keys.  Serialization failures
(SQLSTATE `40001`) and deadlocks (`40P01`) are retryable at session acquisition
(up to six attempts with bounded backoff), while a commit-time conflict is
translated to `TransientRepositoryError` because the caller body cannot safely
be replayed by the context manager.  Unique violations are likewise not
blindly retried.  Callers must retry a transient operation from a fresh
snapshot.

## Tables and model families

The SQLAlchemy metadata in `persistence/sqlalchemy.py` is the source for this
inventory.  Most rows retain the complete Pydantic record in `data`; selected
identity, state, scope, time, and accounting fields are real columns for keys,
indexes, and bounded sweeps.

- **Core execution:** `workloads` (`WorkloadRecord`), `cells`
  (`CellRecord`), `capacity_pools` (`CapacityPoolRecord`), and `leases`
  (`LeaseRecord`).  These represent submitted work, observed execution cells,
  declarative pools, and fenced assignments.
- **Governance and tenancy:** `organizations` (`OrganizationRecord`),
  `projects` (`ProjectRecord`), `policies` (`PolicyRecord`), `quotas`
  (`QuotaRecord`), `budgets` (`BudgetRecord`), and `role_grants`
  (`RoleGrantRecord`).  Scope is part of identity; a project name or URI is not
  globally unique across organizations.
- **Delivery and history:** `idempotency_keys` (`IdempotencyRecord`),
  `outbox` (`OutboxRecord`), and `audit_events` (`AuditEvent`).  Idempotency is
  keyed by scope/operation/key; the outbox couples backend commands to domain
  commits; audit events have a monotonic sequence and are immutable. `webhooks`
  stores scoped subscriptions and delivery cursors.
- **Team accounts and requests:** `users`, `invites`, `api_tokens`, and
  `team_requests` store accounts, invitations, revocable tokens, and
  capacity/approval requests.
- **Assistant and research:** `assistant_providers`,
  `assistant_conversations`, and `research_plans` store provider configuration,
  user-owned conversations, and budgeted research plans.
- **Execution evidence and operations:** `progress_reports` (latest report
  per workload/rank), `workload_topologies` (frozen rank map per workload/
  attempt), `cell_stages` (one asset per cell), `credential_rotations` (one
  generation per credential), `expansion_admissions` (one plan per expansion),
  `service_replica_health` (one row per service/replica), and
  `service_decisions` (bounded, per-service explained decisions).
- **Artifacts and federation:** `managed_artifacts` (scoped lifecycle and
  byte-accounting ledger), `artifact_references` (workload reverse index),
  `plane_peers` (`PlanePeerRecord`), `budget_grants` (`BudgetGrantRecord`), and
  `quota_grants` (`QuotaGrantRecord`).
- **Schema bookkeeping:** `alembic_version` is Alembic's revision stamp;
  `schema_revisions` is also understood for development schemas.

## Migrations and compatibility

Packaged Alembic scripts live in
`eyvan/server/persistence/migrations/versions/`.  The current SQL revision is
`SCHEMA_REVISION = "0024_research_plans"`; the migration sequence includes the
transactional outbox, capacity pools and query indexes, retention columns,
artifact/federation/tenant and grant support, progress and audit history,
scoped artifact identity/lifecycle, attempt topology, cell staging, credential
rotation, expansion admissions, service operations, and webhooks. Revisions
0020–0024 add team accounts, team requests, assistant providers, assistant
conversations, and research plans.

Use the packaged `eyvan-migrate` entry point (or `persistence/migrate.py`), which
uses `EYVAN_DATABASE_URL` or the documented local-development default
`postgresql+asyncpg://eyvan@127.0.0.1:5432/eyvan`.  The default carries no
password on purpose — it only works against a loopback Postgres configured for
trust/peer auth — and every real deployment sets `EYVAN_DATABASE_URL` (or a
secret reference) with its own credentials.  Online upgrades target
`head`; the migration tool classifies the database against the packaged graph
and fails closed for unknown, diverged, ahead, multiple-head, or multiple-current
revision states.  A known-behind schema may be upgraded only during explicitly
acknowledged maintenance after writers are quiesced and a backup/restore plan is
verified.  A failed rollout is repaired forward; Alembic downgrade is not the
rollback procedure.  On startup the repository rejects a stamped revision that
does not equal this build's head (`SchemaRevisionMismatch`).

The local JSON document has an independent version/migration-compatibility
policy.  It accepts supported historical versions and refuses unknown fields or
versions rather than dropping durable evidence; new collections cause a higher
document version when necessary.

## Leadership

`PostgresAdvisoryLeader` uses a session-scoped PostgreSQL advisory lock (the
stable default lock ID is defined in `leader.py`) on a dedicated connection.
`pg_try_advisory_lock` makes acquisition non-blocking: exactly one connected
replica can lead the global scheduler.  Renewal verifies actual ownership in
`pg_locks` using the current backend PID; it does not re-call the re-entrant
acquire function.  Loss of the connection or lock demotes the replica, and
`release` unlocks and closes the dedicated connection.

Use a direct PostgreSQL connection or a session-pooling proxy.  PgBouncer in
transaction-pooling mode is unsupported for this lock: it can move statements
to another backend session, invalidating session-scoped ownership.  Leadership
is separate from ordinary repository transactions and does not make agent
fleet mutations leader-only.

## Control-plane versus agent fleet state

The repository described above is **control-plane state**: workloads, leases,
cells, pools, policy, quota/budget, artifacts, audit, outbox, and controller
coordination records.  It is what the API and scheduler need to survive a
restart and coordinate replicas.

Agent fleet state is a separate `FleetStateStore` in
`eyvan/capacity/fleet_state.py`.  It is an atomic JSON document held in a
`LocalBackend` or a GCS object through `GcsBackend`; GCS object generations
provide compare-and-swap.  It records provider-facing cell-agent state and
uses fencing/durable leases so two agents cannot mutate one fleet concurrently.
A local fleet document uses an advisory file lock.  Fleet-state unavailability
or a compare-and-swap conflict is not interpreted as an empty fleet.  Do not
put fleet-agent state in the control-plane tables, or treat a control-plane
cell row as proof that an agent's provider mutation completed.

## Source references

- `eyvan/server/persistence/base.py` — repository/session contract and typed errors
- `eyvan/server/persistence/state.py` — shared state model and copy-on-write semantics
- `eyvan/server/persistence/memory.py`, `local.py` — development and local backends
- `eyvan/server/persistence/sqlalchemy.py` — PostgreSQL rows, sessions, indexes, and transactions
- `eyvan/server/persistence/migrate.py` and `migrations/versions/` — migration policy and graph
- `eyvan/server/persistence/leader.py` — advisory-lock leadership
- `eyvan/capacity/fleet_state.py` — separate agent fleet-state store
- `eyvan/records.py` — persisted Pydantic record models
