# Multiple replicas and failover

## Supported shape

Eyvan API instances share **one PostgreSQL primary and one schema**. Controller
enabled instances contend for a session-scoped advisory lock; API-only instances
can serve reads without running reconciliation. This is not a multi-primary,
cross-region consensus system.

```text
                     load balancer
                      /          \
           API + controller      API-only instance
                      \          /
                   PostgreSQL primary
                         |
               durable outbox / leases
                         |
                capacity and execution agents
```

Set `EYVAN_CONTROLLER_ENABLED=0` on an API-only instance. Keep the database URL,
OIDC configuration, public URL, artifact signing key, metrics credential and
shared artifact store consistent across replicas. Network-bound processes also
need their listener TLS configuration or an explicit reviewed plaintext opt-in.
A local JSON repository is not a shared deployment backend.

Shared artifact storage matters independently of the database: a capability
signed by one replica is useless on another if it points at a file only the
first replica can read. Do not use per-pod local artifact directories for a
multi-replica managed-callable deployment.

## Leadership and in-flight work

`eyvan/server/persistence/leader.py` holds leadership on a dedicated database
session. Renewal checks actual ownership in `pg_locks`, not just connection
liveness and not a re-entrant lock acquisition. Losing the backend session
releases its advisory lock; a contender can acquire leadership afterwards.

Use a direct PostgreSQL connection or a **session-pooling** proxy for leadership.
Transaction pooling does not preserve the backend session that owns the lock and
is not the supported deployment. Do not point competing replicas at independent
primaries and expect advisory locks to coordinate them.

Leadership is not an exactly-once cloud transaction. Outbox claims, lease and
cell fences, provider idempotency and reconciliation govern work already in
flight. A database disconnect cannot undo a provider request already sent.
Failures must still be recovered through the recorded command and provider
state rather than inferred from a healthy new leader alone.

## Reads and browser sessions

API-only instances read committed data from the shared primary. The regression
in `tests/test_api_replica_reads.py` uses independent repository/connection-pool
instances and verifies that an API-only application sees another instance's
submission and subsequent cancellation. This demonstrates shared-primary
coherence, not throughput scaling or asynchronous read-replica consistency.

Dashboard BFF sessions are currently process-local. Multiple API instances need
session affinity for browser sign-in and subsequent requests, or a future shared
session store. Losing the instance holding a session requires signing in again.
SDK bearer authentication does not depend on that store. This is a real limit,
not a zero-interruption failover claim.

## Evidence and operational checks

Local checks ran against PostgreSQL **17.11** built from the official
checksum-verified source archive into an isolated workspace cache. The existing
test fixture created and stopped disposable loopback-only clusters. Tests cover:

- Two contending leaders and transfer after explicit release.
- `pg_terminate_backend` against the lock holder, followed by ownership transfer.
- Rechecking lock ownership rather than merely connection liveness.
- Concurrent outbox claims and `FOR UPDATE SKIP LOCKED` behavior.
- Duplicate/concurrent settlement and submission invariants.
- Independent API-only reads of committed updates.

Run the same database-backed checks with a disposable test database or the
fixture's local binary directory:

```bash
EYVAN_REQUIRE_POSTGRES=1 EYVAN_TEST_POSTGRES_BINDIR=/path/to/postgres/bin \
  uv run pytest tests/test_postgres_semantics.py tests/test_chaos_on_postgres.py tests/test_api_replica_reads.py
```

**Never point these fixtures at production data.** Some tests reset the test
schema. CI uses its dedicated PostgreSQL service.

This evidence does not establish Kubernetes rescheduling behavior, a database
primary promotion procedure, network-partition safety across database clusters,
zero downtime, or a production availability SLA. Deployment qualification still
needs restore exercises, load-balancer/session-affinity checks, and provider
receipts for the actual environment.
