# Eyvan, explained

## 1. What Eyvan is

A capacity plane: software between "I need this much accelerator, shaped this
way" and machines running your code. You submit Python callables with a
requested shape; Eyvan asks providers for machines that shape, boots them into
cells, dispatches your callable, and records a receipt for every step. It is
not a job queue with human babysitting — the plane retries refused capacity,
reaps idle cells, and reconciles its state after its own restart.

## 2. The five ideas

1. **Shape, not pool.** A workload names `TpuConfig(type="v5p-16", pods=2,
   capacity_class=SPOT)` and a fleet. Which pool satisfies it is the
   scheduler's decision, made against live cell state and provider quota.
2. **Receipts.** "Spot v6e-16 gets reclaimed before boot" is a claim with a
   recorded run behind it or it is nothing. [`eyvan proof`](proofs.md) makes
   pass criteria executable — exit 0 pass, 1 fail, 2 *did not exercise the
   behaviour*, 3 refusal (the run never happened and must not wear a verdict).
3. **Honest narration.** The SDK's wait renders each state transition with the
   operative diagnostic: `blocked | … | the plane is retrying (attempt N)`,
   the provider's queue-admission refusal verbatim, `0/2 workers ready`.
4. **Provider-agnostic core.** Nothing that decides — scheduler, planner,
   durable model — may name a provider (`tests/test_provider_boundary.py`).
   Backends declare capabilities; the core reads them.
5. **Durable everything.** Workloads, cells, leases, and fleet state live in
   the plane's store (`EYVAN_REPOSITORY_BACKEND`, `EYVAN_DATABASE_URL`; the
   loopback plane persists in `~/.eyvan/control-plane.json`). A plane that
   dies mid-dispatch and comes back reconciles the whole world from it.

## 3. Architecture

```
        you (SDK: init + @execute)
                │  signed callable artifact + workload record
                ▼
        eyvan-server  ── the plane: API, scheduler, capacity planner,
                │        controller, durable state
                ▼
        eyvan-agent   ── owns machines per backend; reconciles cells
                │
        ┌───────┴────────┐
     local backend     GCP backend (queued resources: spot / on-demand)
        laptop-cpu      v5p / v6e pools per zone
```

The plane talks to agents with a shared bearer secret
(`EYVAN_AGENT_SHARED_SECRET`); agents run native workers under a separate root
secret. Both live in `~/.eyvan/gcp-native-runtime.json` for a profiled setup.

## 4. The object model

- **Fleet** — a name you submit into; owns pools. Generated zone fleets land
  in `~/.eyvan/autoscale/configs/research-<zone>.yaml`.
- **Pool** — (backend, accelerator type, capacity class, zone, max). A fleet
  like `research-us-central1-a` has one pool per accelerator type its quota
  covers.
- **Cell** — one allocation attempt: `registering → ready → leased → draining
  → deleting → deleted`, with `unavailable` for provider trouble. Cells are
  what cost money; the controller reaps idle ones on the fleet's timeout.
- **Workload** — your callable + shape, durable: `pending → leased →
  running → completed/failed/cancelled`. Submissions are idempotent per
  callable digest: identical callables share one stored artifact.
- **Lease** — binds a ready cell to a workload for dispatch. A crash
  mid-dispatch can quarantine one (see [troubleshoot](troubleshoot.md)).

## 5. How a workload actually runs

1. `init(fleet=…)` resolves the plane and credentials; `@execute(TpuConfig)`
   wraps your function.
2. Calling it dumps the bound callable, signs it, and uploads it to the
  content-addressed artifact store (`sha256`-named object; a concurrent
  identical upload that loses the race verifies the winner and continues).
3. The workload record lands `pending`; the scheduler looks for a ready cell.
4. With none, the capacity planner asks the provider — a queued resource per
   worker, atomically. Refusals render as `blocked … retrying (attempt N)`
   and the plane retries; queue-admission denials (no permission for that
   accelerator in that location) fail the workload with the provider's words.
5. The cell boots, the agent registers it, the scheduler leases it, dispatch
   runs your callable on the worker, and the result manifests back through
   the artifact store.
6. Idle cells drain after the fleet's timeout and delete their provider
   resources; `eyvan cells` shows every stage.

## 6. Using it

- Submit and wait: call the decorated function; it blocks with live progress.
- Detach: the workload is durable — `eyvan workload status|logs|cancel ID`.
- Inspect placement: `eyvan workload explain ID` (placeable, candidate cells,
  provisioning cell ids, delivery state).
- Measure a zone: `eyvan capacity launch-latency` — per-pool handover
  seconds the plane itself recorded.

Continue with [install](install.md) for a real deployment, or
[operate](operate.md) for the runbook.
