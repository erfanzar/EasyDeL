# Part 5 — The domain layer and scheduling

`eyvan/domain/` holds provider-neutral policy. Scheduler decisions take immutable
snapshots and return proposals without I/O, a clock, or randomness. Availability
helpers can use the current time when no timestamp is supplied. The server-side
adapter (`eyvan/server/service/scheduling.py`, `CoreSchedulingService`) converts
persisted records into domain objects, runs the scheduler, and commits the
proposals; the controller realizes them through fenced reconciliation.

Module map:

| module | role |
|---|---|
| `models.py` | `ResourceVector`, `WorkloadSpec`, `CapacityCell`, `CapacityLease`, decision types |
| `scheduler.py` | the pure `Scheduler`: admission, fairness, placement, preemption |
| `pricing.py` | `PricingCatalog`: versioned price samples, specificity-ranked lookup |
| `costing.py` | exact `Decimal` money and resource-hour arithmetic |
| `spend.py` | `RunSpend` / `spend_of`: settled vs reserved cost over lease documents |
| `availability.py` | `CapacityBelief`: decayed empirical availability evidence, backoff |
| `carbon.py` | carbon-intensity contract types (reporting only, never placement) |
| `prediction.py` | `predict_demand`: four-gate predictive pre-warming (ships disabled) |
| `accelerators.py` | family/size ordering helper |

## The contract

`Scheduler.schedule(workloads, cells, *, leases=(), quotas=(), budgets=())` is the
one public entry point. Inputs are five immutable snapshots; the output is a
`SchedulingResult`:

```python
@dataclass(frozen=True, slots=True)
class SchedulingResult:
    placements: tuple[PlacementDecision, ...]
    preemptions: tuple[PreemptionPlan, ...]
    infeasibilities: tuple[Infeasibility, ...]
    considered_order: tuple[str, ...]
```

Every workload considered lands in exactly one of the three outcome tuples, and
`considered_order` records the fairness queue's pop order — the audit trail for why
a workload was or was not reached.

Why purity matters: the same function serves the live admission path, the
"explain" path (what would happen to this workload? — without committing), and the
fenced re-validation a preemption controller runs inside a transaction. Because it
cannot write, re-running it is always safe, and a stale plan is detected by
re-running and comparing.

## Determinism is structural

- **No clock, no RNG.** Lease ids are `sha256(workload_id \0 attempt \0 generation
  \0 cell_id)` truncated to 32 hex chars — a deterministic function of the
  snapshot. Two passes propose identical lease ids; that is what makes replay and
  fenced validation possible.
- **Decimal everywhere.** `ResourceVector` parses every quantity through `str`,
  never raw float, so quota/budget/fairness arithmetic never depends on binary
  floating-point rounding.
- **Canonical ordering.** Cells are scanned sorted by id; label pairs are stored
  sorted; every tie ends in a string key.
- **Validation before decision.** Duplicate workload/lease/cell ids, a pending
  workload already holding an active lease on the same attempt, leases
  overcommitting a cell, a budget in the wrong currency — all rejected up front
  with `ValueError`.

## `ResourceVector` semantics worth memorizing

- `fits_within(capacity)` — a missing name in `capacity` reads as **zero** (hard
  placement constraint).
- `within_limits(limits)` — a missing limit means **unconstrained** (quota/budget
  semantics — deliberately the opposite).
- `dominant_share(denominator)` — max over dimensions of `quantity / capacity`;
  the DRF share function. Positive demand against zero capacity is `Infinity`.
- `dot(prices)` — the basis of a cell's hourly cost.

## Identity, quota, budget

`WorkloadIdentity` is the `(tenant_id, project_id, principal_id)` triple charged
for a workload; `IdentityScope` is any prefix of it. Every accounting gate — quota,
budget, fairness — walks `identity.scopes`, so a limit declared at any hierarchy
level binds every workload beneath it.

`Budget` carries a monetary `limit`, durable `spent`, a currency (must match the
scheduler's), optional accelerator-hour limits (keyed by resource names like `tpu`
or `gpu_h100`), and `hard_limit` (default `True`).

## `WorkloadSpec` — the policy input

Key fields and their semantics:

- `gangs` — all-or-nothing groups of identical replicas.
- `priority` — strict bands `BATCH(0) < INTERACTIVE(1) < PRODUCTION(2)`. Bands
  never interleave: PRODUCTION drains fully before INTERACTIVE before BATCH.
- `queue_position` — the intra-identity heap key (the adapter maps it from
  `1000 - intra_band_priority`, so larger user-facing priority pops first).
- `required_labels` / `allowed_label_values` / `denied_label_values` — hard
  constraints; `preferred_labels` / `preferred_cells` — soft (scoring only).
- `placement: ATOMIC | ELASTIC` — ATOMIC demands one exact-shape cell; ELASTIC
  packs cells of any matching accelerator type up to the worker count (and
  requires per-worker scope).
- `checkpoint_policy` — `NONE` / `ON_PREEMPTION` / `PERIODIC` with a grace period;
  decides preemption eligibility and action.
- `estimated_duration_seconds` — feeds the reservation-cost estimate.

## `CapacityCell` and pricing truth

`CapacityCell.unit_prices` is the operator's **complete** price statement. Two
predicates drive the cost gates:

- `declares_prices` — any entries at all. An empty vector means "pricing unknown",
  **not** "free"; declared zeros count as knowledge.
- `prices_cover(resources)` — every requested dimension is named. A partial vector
  is by contract complete about what the runtime charges, but it surfaces as
  `cost_unknown` in scoring so fully priced cells win cost tie-breaks.

## Fairness: hierarchical DRF

Within a priority band, a `_HierarchicalDRFQueue` picks tenant → project →
principal by weighted dominant resource share, so no tenant can starve another
within the same band. Placements and preemptions update provisional lease
accounting immediately, so consecutive decisions in one pass see each other's
effects.

## Placement scoring and preemption

A placement scores candidate cells (fit, cost, availability belief via
`route_priority`, startup latency, soft preferences) and the winner receives a
proposed lease. When nothing fits but preemption could make room, the scheduler
emits a `PreemptionPlan`: the cell, ordered victims (`CHECKPOINT_AND_STOP` or
`TERMINATE`, each with a grace period), and the replacement lease. Checkpointable
victims with a real hook are preferred for checkpoint-and-stop; a cell is only
proposed for preemption when it is safe (`PREEMPTION_UNSAFE` is one of the twelve
stable `InfeasibilityCode` values, alongside `NO_CELLS`, `QUOTA_EXCEEDED`,
`BUDGET_EXCEEDED`, …).

## Money: pricing, costing, spend

- `PricingCatalog.quote()` picks the most **specific** sample (which of
  project/region/zone are populated) then the freshest; ties break on the scope
  tuple, never insertion order.
- `costing.py` does exact `Decimal` arithmetic: `money()`, `cost_for_hours()`.
- `spend.py` distinguishes **reserved** cost (a lease is holding capacity) from
  **settled** cost (the lease finished); `settled_at` is the boundary. An unpriced
  pool settles to zero, which is explicitly *not the same as free* — the `priced`
  flag says so.

## Availability belief

`availability.py` keeps per-pool success/failure evidence with time decay.
`ProvisionOutcome` treats `HOST_FAULT` as its own outcome — "a bad host says
something about the physical pool" — with the harshest backoff base (300 s vs 60 s
for stockout). This belief is what `route_priority` in scoring consumes, and it is
why a pool that just failed you is deprioritized without anyone editing config.
