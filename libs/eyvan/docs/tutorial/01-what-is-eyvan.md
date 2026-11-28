# Part 1 — What Eyvan is

## The one-paragraph version

Eyvan is a **capacity plane**: software that sits between "I need this much
accelerator, shaped this way" and machines actually running your code. You submit
Python callables with a requested *shape* (for example `TpuConfig(type="v5p-16",
pods=2, capacity_class=CapacityClass.SPOT)`). The plane asks providers for machines
matching that shape, boots them into **cells**, dispatches your callable, and
returns its result. Workloads, leases, and audit events record the lifecycle.

It is not a job queue with human babysitting: the plane retries refused capacity on
its own, reaps idle cells, and reconciles its entire world from durable state after
its own restart.

## The five ideas everything else follows from

1. **Shape, not pool.** A workload names what it needs, not where it should run.
   Which pool satisfies it is the scheduler's decision, made against live cell
   state and provider quota.
2. **Receipts or it did not happen.** "Spot v6e-16 gets reclaimed before boot" is a
   claim with a recorded run behind it or it is nothing. `eyvan proof` makes pass
   criteria executable (exit 0 pass, 1 fail, 2 *did not exercise the behaviour*,
   3 refusal) — see `docs/proofs.md` and Part 14.
3. **Failure narrates itself.** A refused capacity attempt renders as
   `blocked | … | the plane is retrying (attempt N)` with the provider's own
   sentence attached — never a bare `FAILED`.
4. **Nothing that decides names a provider.** The scheduler, the planner, and the
   durable model may not special-case any cloud. Backends declare capabilities; the
   core reads them. `tests/test_provider_boundary.py` enforces this.
5. **Recover from durable state.** Workloads, cells, leases, and pending commands
   live in the plane's store. After restart, the controller reconciles those
   records with agent observations. See `docs/proofs.md` for recovery checks.

## Architecture

```
        you (SDK: init + @execute)
                │  unsigned callable artifact + workload record
                ▼
        eyvan-server  ── the plane: API, scheduler, capacity planner,
                │        controller, durable state
                ▼
        eyvan-agent   ── owns machines per backend family; reconciles cells
                │
        ┌───────┴────────┐
     local backend     GCP backend (queued resources: spot / on-demand)
        laptop-cpu      v5p / v6e pools per zone
```

Three programs do the work (all are console scripts in `pyproject.toml`):

| Program | Entry point | Role |
|---|---|---|
| `eyvan` | `eyvan.sdk.cli:main` | The CLI and the SDK's command layer |
| `eyvan-server` | `eyvan.server.api.app:main` | The control plane: FastAPI app + leader-elected controller |
| `eyvan-agent` | `eyvan.worker.agent.cli:main` | The cell agent: owns machines for one backend family |

Plus two support entry points: `eyvan-native-worker` (the worker runtime on an
acquired machine) and `eyvan-migrate` (schema migrations).

The Python package splits the same way:

```
eyvan/
├── sdk/          client: execution.py (init/execute), client/ stack, commands/ (CLI)
├── server/       control plane: api/ (FastAPI), controller/, service/, persistence/
├── worker/       agent/ (cell agent), runtime/ (native worker, progress, results)
├── domain/       pure policy: models, scheduler, pricing, costing, spend, availability
├── capacity/     fleet_config, planning, fleet_state, hardware_catalog,
│                 drivers/ (one per backend), providers/ (cloud API seams),
│                 autoscale/ (zone-fleet generation)
├── artifacts/    content-addressed stores, capability tokens, managed lifecycle
├── identity/     mTLS, credential rotation
├── records.py    the durable Pydantic record models and state enums
└── _internal/    shared utilities (network, platform, observability, …)
```

## The vocabulary

| term | meaning |
|---|---|
| **plane** | the control plane (`eyvan-server`) — scheduler, planner, state |
| **agent** | a cell agent (`eyvan-agent`) serving one backend family |
| **fleet** | a named set of capacity pools you can submit into |
| **pool** | one (backend, accelerator, capacity class, zone) combination |
| **cell** | one acquired allocation (e.g. one spot queued resource) — the unit that costs money |
| **workload** | your callable + its requested shape, moving through durable states |
| **lease** | binds a ready cell to a workload for dispatch |
| **shape** | what you asked for: type, worker count, capacity class (`TpuConfig`/`GpuConfig`/`CpuConfig`) |

## The registered backends

From `eyvan/capacity/drivers/registry.py`: `local` (loopback laptop fleets),
`gcp-tpu` (generated zone fleets driving TPU queued resources), `gcp-compute`,
`gcp-attach` (existing TPU slices), `gcp` (router), `aws-ec2` (+ `aws`),
`azure-vm` (+ `azure`), `kubernetes`/`k8s`, `manual`, and `onprem-ssh`.

A registered backend is not evidence that a particular account, network, image,
or accelerator works. Use `docs/proofs.md` to select checks for the backend and
build you intend to deploy.

## A workload's full journey

1. `init(fleet=…)` resolves the plane and credentials; `@execute(TpuConfig)` wraps
   your function.
2. Calling it dumps the bound callable and uploads its unsigned bytes to the
   content-addressed artifact store (`sha256`-named object). Standalone callable
   execution requires an explicitly trusted project; a digest checks integrity,
   not submitter identity or pickle safety.
3. The workload record lands `pending`; the scheduler looks for a ready cell.
4. With none, the capacity planner asks the provider — one queued resource per
   worker, atomically. Refusals render as `blocked … retrying (attempt N)` and the
   plane retries within the attempt budget; queue-admission denials (no permission
   for that accelerator in that location) fail the workload with the provider's
   own words.
5. The cell boots, the agent registers it, the scheduler leases it, dispatch runs
   your callable on the worker, and the result comes back through the artifact
   store.
6. Idle cells drain after the fleet's timeout and delete their provider resources;
   `eyvan cells` shows every stage.

Continue to [Part 2 — Getting started](02-getting-started.md) to run all of this
on your laptop.
