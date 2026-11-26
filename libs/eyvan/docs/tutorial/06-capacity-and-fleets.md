# Part 6 — Capacity, fleets, and autoscaling

Source: `eyvan/capacity/` — `fleet_config.py`, `planning.py`, `fleet_state.py`,
`hardware_catalog.py`, `stage_policy.py`, `audit.py`, and the `autoscale/` package.
The release reference is `docs/reference/capacity-and-providers.md`; this part is
the guided tour.

## A fleet config becomes pools

`EyvanFleetConfig` (`fleet_config.py`) is a Pydantic model loaded with
`.from_yaml()`. The pipeline:

1. **Normalize** — `normalize_pool_schema` accepts `pools` (or legacy
   `node_types`) and expands multi-zone pools into one pool per zone
   (`v6e-8` → `v6e-8-us-east5-a`), so the scheduler can race zone alternates for
   spot. `_apply_catalog_shape` fills omitted CPU/memory/TPU admission resources
   from the hardware catalog.
2. **Type** — pools become `CpuNodeType` / `GpuNodeType` / `TpuNodeType` with
   `min_cells`/`max_cells`/`initial_cells` (legacy `min_workers` aliases work),
   quota groups, warm windows, and timeouts. `MAX_POOL_WORKERS = 10_000`;
   `max_workers: 0` means the package-wide bound, not infinite capacity.
3. **Validate** — fleet/pool name patterns, cell-ID prefix collisions, GCP
   project+zone presence, `capacity_backend` required for non-GCP, and the
   **one-backend-family-per-agent-endpoint** rule: one agent constructs one
   driver. The GCP router can serve TPU, Compute, and attached-TPU pools at the
   same endpoint.
4. **Compile** — `pool_payloads()` emits one `CapacityPoolRecord` per node type
   (`id = f"{fleet}-{node_type}"`, labels like `eyvan_pool`, `eyvan_fleet`);
   `desired_cells()` compiles deterministic initial cell ids
   (`{prefix}-{index:04d}`). TPU pools default to `SPOT` unless an operator sets a
   class explicitly.

A worked example lives in `examples/fleet.yaml` (GCP, three node types, per-node
runtime blocks with artifacts/init commands/container); the minimal one is
`examples/laptop.yaml` (local backend, one CPU pool). Two provisioning models
apply to every fleet — the venv model (fleet provides Python via
`runtime_python` + `runtime_init_commands`) and the OCI model (your image via
`runtime.container.image`) — described in `docs/runtimes.md` and Part 10.

## How a workload triggers capacity acquisition

The reconcile loop is `FleetService.reconcile_capacity_pools_once()`
(`server/controller/fleets.py`):

1. Load pools, workloads, cells, leases, budgets, quotas, policies.
2. **Repair drifted cells** — `cell_requires_replacement` (`planning.py`);
   mismatched live cells are re-upserted idempotently.
3. **Simulate supply** — `CapacitySimulation.build(...)` counts free cells by
   `(pool, worker_count)`, respecting cooldown/consecutive-failure belief,
   disabled pools, and drained cells. The acquisition bound also reads a pool's
   recent *replaced* incarnations (`_recent_failed_acquisition_streak`, one-hour
   memory), and a pool that denied a workload queue admission is dropped from
   that workload's candidates.
4. **Compute demand** — per-pool pending demand with priority bands and ages;
   `_race_alternates` implements cross-pool racing via the `eyvan_pool_alternates`
   label so a shape can be raced across zones. At the scheduler boundary
   (`to_core_workload`) a cross-zone workload's pool, template and region pins
   become "one of its candidates" rather than exact matches, using the
   `eyvan_pool_alternate_templates` recorded at submission.
5. **Plan** — `CapacityPlanner.plan(demand)` produces launch counts honoring
   min/max cells, warm windows, defragmentation hysteresis
   (`_DEFRAG_MIN_PENDING_SECONDS = 180`), and shared-quota allocation across pools.
6. **Launch** — a monotonic cell index (over all cells ever, including
   tombstoned ones) plus a per-pool-record **epoch** produce collision-free cell
   ids; the upsert carries an idempotency key
   `autoscale:create:{cell_id}:g{pool.generation}`.

Cell lifecycle: `registering → ready → leased → draining → deleting → deleted`,
with `unavailable` for provider trouble. Cells are what cost money; the controller
reaps idle ones on the fleet's timeout.

## The defaults profile and generated fleets

`~/.eyvan/autoscale/defaults.yaml` is the one place an operator pins what every
generated zone fleet derives:

```yaml
version: 1
project: my-gcp-project
fleet_prefix: research
bootstrap_egress: public-internet
runtime_python: '3.13'
runtime_init_commands:
- uv pip install --python /opt/eyvan-native/venv/bin/python --prerelease=allow 'easydel[tpu,profile,torch]'
# runtime_container_image: us-east5-docker.pkg.dev/my-project/eyvan/runtime@sha256:...
min_workers: 0
max_workers: 0
```

- `runtime_python` — the interpreter every worker runs; arms the
  pre-provisioning mismatch refusal (a callable pickled on a different
  `major.minor` is refused at submission, not on the accelerator).
- `runtime_init_commands` — run on every host after the worker environment exists
  and before the worker unit starts.
- `runtime_container_image` — pin one blessed OCI image (prefer an immutable
  digest) and every generated fleet declares it.

The workflow:

```bash
eyvan autoscale init                            # edit the profile
eyvan autoscale generate --zone us-central1-a   # cache configs/research-<zone>.yaml
eyvan up --select research-us-central1-a --preemptible
eyvan down ~/.eyvan/autoscale/configs/research-us-central1-a.yaml -y   # drain
```

Generation (`autoscale/service.py`, `compiler.py`, `profiles.py`) discovers GCP
TPU quota, classifies each type with typed skip reasons, and compiles one
validated fleet per eligible zone — preserving finite shared generation quota and
caching queue-admission denials (`queue_admission.py`) so future generation skips
targets that refused admission (exact project/zone/type/Spot keys, 24-hour lazy
expiry).

## Planning and the hardware catalog

- `planning.py` is pure worker-unit planning: monotonic provider-safe cell ids,
  active-lease accounting, candidate-pool ranking, acquisition backoff, and
  workload-owned vs shared cell separation.
- `hardware_catalog.py` is the authority for what a named shape *is*: kind, chips,
  quota units, per-host CPU and memory, topology where grounded, and the HTTPS
  source/version. It is **not** quota discovery — quota metrics say what is
  available and join on type. GPU rows describe one VM; TPU quota units differ by
  generation (e.g. v4/v5p = 2 per schedulable TPU). Unknown hardware remains
  usable only with an explicit complete resource vector.

## Fleet state: the agent-side durable document

`fleet_state.py` implements `FleetStateStore` — a separate durable document from
the control-plane repository (do not confuse the two; see Part 8). Schema
`eyvan.fleet-state` v1 holds cells, cell tombstones, workload fences, and a
mutation lease.

- Backends: `LocalBackend` (atomic tmpfile + `os.replace`, `fcntl` lock) or
  `GcsBackend` (generation-precondition compare-and-swap via `gcs_rest.py`;
  the gcloud subprocess path is the fallback).
- A durable, TTL'd **mutation lease** (`acquire_lease`/`exclusive()`) ensures one
  cell agent mutates at a time; dead same-host holders are broken automatically.
- Every mutation validates `(generation, fencing_token)` against the record or
  tombstone (`_validate_fence`); a tombstone permanently fences a cell id so a
  released generation cannot be resurrected.
- Conflicts retry; unavailable state is never interpreted as an empty fleet.

## Stage policy and capacity audit

- `stage_policy.py` declares the per-cell staging budget (a hard charged-byte
  ceiling), max single staged asset, eviction order (`least_recently_used` or
  `largest_first`), and retry limit. Zero disables staging.
- `audit.py` emits transition-only structured lifecycle events (planned,
  requesting, provisioning, provider transition, ready, preempted, lost,
  replacement, removed) with fleet/pool/cell/provider/resource context. A sink
  failure can never break a lifecycle transition.
