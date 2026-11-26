# Domain Logic and Scheduling Reference

This document describes Eyvan's pure domain layer (`eyvan/domain/`) and the
server-side adapter that feeds it (`eyvan/server/service/scheduling.py`). It is
grounded in the actual code; every algorithmic claim below names the symbol
that implements it. Where the code makes a decision legibly, a worked example
follows.

Module map (all paths relative to the repository root):

| Module | Lines | Role |
|---|---|---|
| `eyvan/domain/models.py` | 779 | Immutable domain objects: `ResourceVector`, `WorkloadSpec`, `CapacityCell`, `CapacityLease`, `PlacementScore`, result types |
| `eyvan/domain/scheduler.py` | 1246 | The pure `Scheduler`: admission, fairness, placement, preemption |
| `eyvan/domain/pricing.py` | 158 | `PricingCatalog`: versioned price samples with specificity-ranked lookup |
| `eyvan/domain/costing.py` | 78 | Exact `Decimal` money and resource-hour arithmetic |
| `eyvan/domain/spend.py` | 116 | `RunSpend` / `spend_of`: settled vs. reserved cost accounting over lease documents |
| `eyvan/domain/availability.py` | 189 | `CapacityBelief`: decayed empirical availability evidence, backoff, probe policy |
| `eyvan/domain/carbon.py` | 197 | Carbon-intensity contract types (reporting only, never placement) |
| `eyvan/domain/prediction.py` | 161 | `predict_demand`: four-gate predictive pre-warming (ships disabled) |
| `eyvan/domain/accelerators.py` | 28 | `sort_accelerator_types`: family/size ordering helper |
| `eyvan/server/service/scheduling.py` | 588 | `CoreSchedulingService`: persisted-record → domain-object adapter |

---

## 1. The pure-functional scheduler contract

### 1.1 The one public entry point

`Scheduler.schedule` (`eyvan/domain/scheduler.py`, class `Scheduler`) is the
only policy entry point:

```python
def schedule(
    self,
    workloads: tp.Iterable[WorkloadSpec],
    cells: tp.Iterable[CapacityCell],
    *,
    leases: tp.Iterable[CapacityLease] = (),
    quotas: tp.Iterable[Quota] = (),
    budgets: tp.Iterable[Budget] = (),
) -> SchedulingResult:
```

Inputs are five **immutable snapshots**: pending workload specs, capacity
cells, active leases, quotas, and budgets. The constructor takes only
`currency: str = "USD"`, which is uppercased and used to reject budgets in any
other currency during snapshot validation.

The docstring states the contract directly: *"The scheduler is deliberately
pure: it reads immutable snapshots and returns proposed leases and preemption
plans. Callers persist decisions and realize them through fenced
reconciliation."* The scheduler:

- holds no clock, no store, no provider handle;
- never mutates its inputs (all domain objects are `@dataclass(frozen=True,
  slots=True)`);
- performs no I/O and no randomness;
- returns a `SchedulingResult` describing *proposals*, and the caller decides
  whether to persist and execute them.

### 1.2 Output: `SchedulingResult`

```python
@dataclass(frozen=True, slots=True)
class SchedulingResult:
    placements: tuple[PlacementDecision, ...]
    preemptions: tuple[PreemptionPlan, ...]
    infeasibilities: tuple[Infeasibility, ...]
    considered_order: tuple[str, ...]
```

Every workload considered in a pass lands in **exactly one** of the three
outcome tuples, and `considered_order` records the order workloads were popped
from the fairness queue — the audit trail for *why* a workload was or was not
reached.

`PlacementDecision` carries the leader `lease: CapacityLease`, the winning
`score: PlacementScore`, and `member_leases` for multi-cell placements
(cell-group and elastic). Invariants enforced in `__post_init__`: no member may
repeat the leader's `lease_id`, all lease ids are unique, and every member
belongs to the placement's workload. `all_leases` returns `(leader,
*members)`.

`PreemptionPlan` carries `cell_id`, an ordered `victims: tuple[PreemptionVictim,
...]`, the `replacement_lease`, and its `score`. Each `PreemptionVictim` names
the action — `CHECKPOINT_AND_STOP` or `TERMINATE` — and the victim's
`grace_period_seconds`.

`Infeasibility` carries one of twelve stable `InfeasibilityCode` values
(`NO_CELLS`, `HARD_CONSTRAINT`, `TOPOLOGY_MISMATCH`,
`CAPACITY_CLASS_MISMATCH`, `RUNTIME_UNSUPPORTED`, `REPLICA_COUNT_UNSUPPORTED`,
`REPLICA_TOO_LARGE`, `INSUFFICIENT_CAPACITY`, `QUOTA_EXCEEDED`,
`BUDGET_EXCEEDED`, `UNPRICED_CELL`, `PREEMPTION_UNSAFE`) plus per-cell
`CellInfeasibility` detail records.

### 1.3 Determinism guarantees

Determinism is structural, not aspirational:

- **No clock, no RNG.** The only hash use is lease-id derivation
  (`_make_lease`), which is `sha256(workload_id \0 attempt \0 generation \0
  cell_id)` truncated to 32 hex chars — a deterministic function of the input
  snapshot. Two passes over the same snapshot propose identical lease ids,
  which is what makes replay and fenced validation possible downstream.
- **Decimal everywhere.** `ResourceVector` stores every quantity as
  `Decimal` (parsed through `str`, never raw float), so quota, budget, and
  fairness arithmetic never depends on binary floating-point rounding.
- **Canonical ordering.** All maps are iterated in sorted order: cells are
  scanned `sorted(cells.values(), key=cell_id)`, label pairs are stored as
  sorted tuples (`_canonical_pairs`), and every tie in scoring ends in a
  string key (`cell_id`, `lease_id`).
- **Validation before decision.** `_validate_snapshot` rejects ambiguous
  snapshots up front with `ValueError`: duplicate `workload_id` / `lease_id` /
  `cell_id` / quota or budget scopes; a pending workload that already holds an
  active lease on the same attempt; a lease referencing an unknown cell;
  leases that overcommit a cell's capacity; a budget in the wrong currency.

### 1.4 Why side-effect-free

Purity is what allows the same function to serve three different call sites:
the live admission path, the "explain" path that asks *what would
happen to this workload* without committing anything, and the fenced
re-validation a preemption controller runs inside a transaction before
executing a plan. Because the function cannot write, re-running it is always
safe, and a stale plan is detected by simply re-running `schedule` and
comparing the resulting `PreemptionPlan` (victim list and cell) with the
proposed one.

---

## 2. Core domain objects (`models.py`)

### 2.1 `ResourceVector`

A canonical, non-negative, `Decimal`-valued vector of named resources.
Construction validates: names must be non-empty trimmed strings, quantities
must be finite and non-negative, duplicates are rejected, and items are stored
sorted by name. Key operations used by the scheduler:

- `fits_within(capacity)` — every named quantity ≤ capacity; **a missing name
  in `capacity` reads as zero** (hard placement constraint).
- `within_limits(limits)` — every quantity in `limits` is respected; **a
  missing limit means unconstrained** (quota/budget semantics — deliberately
  the opposite of `fits_within`).
- `dominant_share(denominator)` — max over dimensions of `quantity /
  capacity`; a dimension with positive demand against zero capacity yields
  `Decimal("Infinity")`. This is the DRF share function.
- `dot(prices)` — scalar product with a per-resource price vector; the basis
  of `CapacityCell.hourly_cost`.
- `scale(factor)` — multiply every quantity (factor must be non-negative).
- `__sub__` refuses to go negative, which is what makes `remaining_after =
  remaining_before - requested` in scoring safe.

### 2.2 Identity, quota, and budget

`WorkloadIdentity` is the fully qualified `(tenant_id, project_id,
principal_id)` triple charged for a workload. `IdentityScope` is any prefix of
it (tenant depth 1, project depth 2, principal depth 3);
`WorkloadIdentity.scopes` returns all three prefixes. Every accounting gate —
quota, budget, fairness — walks `identity.scopes`, so a limit declared at any
level of the hierarchy binds every workload beneath it.

`Quota` pairs a scope with a `limit: ResourceVector` and a positive `weight`
(default 1) used by weighted DRF. `Budget` carries a monetary `limit`,
durable `spent`, a currency (uppercased; must match the scheduler's), optional
`accelerator_hour_limits` / `accelerator_hours_spent` vectors keyed by
resource names (e.g. `tpu`, `gpu_h100`), and `hard_limit: bool = True`.
`Budget.remaining` is `limit - spent` clamped at zero.

### 2.3 `WorkloadSpec`

The immutable policy input for one workload. Notable fields and their
semantics in the scheduler:

- `gangs: tuple[ResourceGang, ...]` — all-or-nothing groups of identical
  replicas; `gang_ids` must be unique; `total_resources` sums
  `resources_per_replica × replicas` over gangs.
- `priority: PriorityBand` — strict bands `BATCH(0) < INTERACTIVE(1) <
  PRODUCTION(2)`. Bands never interleave: the scheduler drains
  `PRODUCTION` fully before `INTERACTIVE` before `BATCH`.
- `attempt`, `generation`, `queue_position` — attempt/generation feed the
  deterministic lease id; `queue_position` is the intra-identity heap key
  (the adapter maps it from `1000 - intra_band_priority`, so a larger
  user-facing priority pops first).
- `required_labels`, `allowed_label_values`, `denied_label_values` — hard
  constraints; `preferred_labels`, `preferred_cells` — soft (scoring only).
- `allowed_capacity_classes` / `capacity_class_preference` — the constructor
  normalizes the preference: entries not in the allowed set are dropped, and
  missing allowed classes are appended in value order, so
  `capacity_class_preference.index(...)` in scoring always terminates.
  Default preference is `RESERVED, ON_DEMAND, SPOT`.
- `worker_resource_scope` — `ATOMIC_CELL` (a gang encodes one whole cell) or
  `PER_WORKER` (a gang encodes one worker). `allocation_resources` multiplies
  `total_resources` by `worker_count` only for `PER_WORKER`; that product is
  what one physical cell must hold and what quota/cost accounting charges.
- `placement` — `ATOMIC` (exact-shape: one cell whose `worker_count` equals
  the workload's) or `ELASTIC` (pack cells of any matching accelerator type,
  up to `worker_count` total). `ELASTIC` requires `PER_WORKER` scope.
- `cell_group_count > 1` — the pod-group path: `N` distinct full-capacity
  cells placed atomically.
- `checkpoint_policy` — `NONE` / `ON_PREEMPTION` / `PERIODIC` plus
  `grace_period_seconds` and `allow_forced_termination`. `checkpointable` is
  `mode is not NONE`. This decides preemption eligibility and action.
- `estimated_duration_seconds` — feeds the reservation-cost estimate
  (`hourly_cost × duration / 3600`).

### 2.4 `CapacityCell`

A failure-isolated placement cell with one allocatable pool. Scheduler-relevant
fields:

- `capacity`, `capacity_class` (parsed through `CapacityClass.parse`, which
  accepts `on-demand`/`on demand`/`ondemand` aliases and serializes
  `on_demand`), `labels`, `topology`, `runtimes`, `worker_count`.
- `unit_prices: ResourceVector` — the operator's **complete** price statement
  for the cell. Two derived predicates drive the cost gates:
  - `declares_prices` — the vector has *any* entries. An empty vector means
    "pricing unknown", **not** "free". Declared zeros count as knowledge.
  - `prices_cover(resources)` — every requested dimension is named in
    `unit_prices`. A partial vector is by contract complete about what the
    runtime charges (unlisted resources are uncharged), so a partial price is
    deterministic but incomplete — surfaced as `cost_unknown` in scoring so
    fully priced cells win cost tie-breaks.
- `route_priority` (lower is better; the adapter derives it from availability
  evidence) and `startup_latency_seconds` (adapter derives from p95 provision
  latency) — scoring keys.
- `max_replica_resources` (must fit within `capacity`), `max_gang_replicas` —
  per-replica shape limits checked as static constraints.
- `exclusive_workloads` — at most one active workload on the cell (set by the
  adapter for `native-local` / `eyvan-native` executors).
- `autoscale_request_workload_id` — reserves a freshly requested cell for the
  workload that caused its creation; any other workload gets a
  `HARD_CONSTRAINT` failure.
- `enabled` — the adapter maps it from `CellState.READY`.

### 2.5 `CapacityLease`

The reservation unit. Carries `resources`, `priority`, `hourly_cost`,
`reserved_cost` (both validated finite and non-negative), and
`resource_hours_reserved` (the per-dimension hour vector used by the
accelerator-hour budget gate). It identifies one workload attempt and cell,
with the charged identity, checkpoint policy, revocability and a non-negative
`fencing_epoch`. Construction requires non-empty IDs and resources, a positive
attempt, and finite non-negative costs.
