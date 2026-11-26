# Receipts and elastic packing: design and comparison protocol

Status: implemented mechanisms described below; cross-scheduler experiments
remain unperformed. This is not a claim that Eyvan outperforms another scheduler.

## The problem being separated

A research job has several different outcomes that are easy to conflate:

1. A plane accepted durable work.
2. A scheduler found an admissible placement.
3. A provider accepted a capacity request.
4. Workers started and produced a result.
5. A particular experiment established a particular property of a particular build.

Neither a successful submission nor a zero exit status establishes all five.
Eyvan separates durable workload identity, fenced backend commands, runtime
observations and experimental evidence rather than treating a successful CLI
invocation as a universal receipt.

## Three records with different meanings

### Submission receipt

`eyvan/sdk/job_receipt.py:JobSubmissionReceipt` renders the workload ID,
allocation mode and status/log/cancel commands. It is a control handle: the
caller can disconnect and later address the durable workload. It is not proof
that capacity exists, that a worker executed, or that an accelerator was used.
The rendered receipt is not itself the persistence mechanism.

### Transactional intent and observation

`eyvan/records.py:OutboxRecord` describes a fenced, **at-least-once** backend
command committed with domain state. It carries an idempotency key, aggregate
generation, claim owner/expiry, attempts and a delivery state. The controller
executes external work outside the state transaction and reconciles what it
observes. A claim expiry or controller failure can lead to another delivery.

This is not a blanket exactly-once guarantee over arbitrary provider APIs or
user code. Idempotency, generation/fencing checks and observation handling must
hold at the specific adapter boundary. A durable intent is not provider success;
a provider acceptance is not worker readiness. Unknown observations must remain
unknown rather than becoming zero utilization or healthy capacity.

The public service facade and subsystem boundary are described in
[the control-plane architecture](../reference/control-plane-architecture.md).
The public provider extension contract is in [providers](../providers.md).

### Experimental proof receipt

`eyvan/sdk/commands/receipts.py` implements `eyvan.proof-receipt/1`. Its important
properties are attribution and explicit limits, not merely JSON output:

- The evaluator's own source digest identifies the criterion. Loosening the
  evaluator changes the identity of the experiment even if its name is unchanged.
- Client and plane build identities distinguish the candidate from a shared
  package version string. Unavailable identity is reported, not reconstructed
  from a convenient current checkout.
- Evidence files can be identified by digest. Finite JSON and atomic replacement
  prevent a partially written receipt from looking like a completed result.
- Unverified dimensions are derived from observations, including missing build
  identity and the limits of a dirty checkout.
- A receipt digest detects alteration relative to that digest. **It is not a
  signature**: someone who can rewrite the whole document can recompute it.
  Hosted artifact attestation is a separate trust mechanism.

The proof CLI distinguishes pass, fail, not-exercised and pre-run refusal. A
Spot run with no reclamation must not pass a reclamation experiment merely
because its training process exited successfully. See [proofs](../proofs.md)
and `tests/test_proof_receipts.py` for criteria, exit semantics and regressions.
The lightweight scheduler and browser JSON reports are useful local test
receipts, but are not automatically instances of this proof-receipt schema.

## Elastic packing without claiming a single hardware world

`eyvan/domain/scheduler.py:Scheduler._schedule_elastic` accepts a per-worker
resource request and an exact worker total. Matching cells may have different
physical widths. Whole cells remain the allocation unit; a larger cell is not
silently split to fit a smaller residual demand.

For example, a four-worker job with available cell widths `[3, 2, 2]` is
feasible as `2 + 2`. Greedily choosing `3` first would falsely report a deficit.
The implementation orders candidates deterministically, then retains a
fewest-cell subset for each reachable worker total. Each pass uses a snapshot
of the reachable totals so the same cell cannot be selected twice. Equal-size
answers retain the earlier deterministic candidate sequence.

Static compatibility, remaining capacity, exclusivity, quotas and known-price
requirements filter or constrain the decision. If the preferred exact pack
violates a hard budget, a second dynamic program preserves the cheapest pack
for each `(worker total, cell count)` and selects a fewest-cell affordable
answer. This relies on equal worker totals having the same requested
resource-hours for the workload's per-worker shape. Budget failures are not
permission to choose an unpriced cell.

The admission remains all-or-nothing for the requested total. A partial warm
subset is not leased while waiting indefinitely for the missing workers.
Scheduling consumes a snapshot; the service must still revalidate before
committing leases because the snapshot can be stale.

Packing fewer cells reduces the number of hardware worlds. It does **not**
magically join independent cells into a single JAX world: native execution runs
per cell of an elastic pack. Atomic/exact-shape requests and elastic requests
therefore have different semantics. Comparing their utilization without
preserving the application's distributed-execution requirements is misleading.

Regression evidence includes `tests/test_elastic_packing_preserves_feasible_subsets.py`
and `tests/test_scheduler_budget_properties.py`: bounded generated inputs are
compared with exhaustive subset feasibility, including affordable alternatives.
These checks do not establish correctness for every controller history or live
provider failure.

## Current measurements and their boundary

The reproducible local 1k/10k-cell simulations, raw sample format and coarse CI
budgets are documented in [scheduler benchmarks](../reference/scheduler-benchmarks.md).
They measure snapshot construction, ordinary scheduling and elastic decisions.
They do not measure database transactions, cloud provisioning, training
throughput or another scheduler. They are not a measured full reconciliation
loop p99 or a production tail-latency commitment.

## Protocol for Slurm / Volcano / kube-batch comparisons

No measurements against these systems have been collected for this document.
The following protocol defines the missing experiment rather than substituting
an Eyvan heuristic for a named external system.

1. **Pin candidates.** Record each scheduler's immutable source/image identity,
   configuration, plugins, runtime, test harness and evaluator digest. If a
   candidate cannot run in the selected environment, report that limitation;
   do not silently substitute a different scheduler or version.
2. **Define semantic equivalence.** Use CPU-only whole-cell placement cases
   first. State whether the job needs one distributed world or independent
   worker worlds, its exact worker count, shape compatibility, exclusivity,
   quota and budget semantics. Report unsupported semantics explicitly instead
   of pretending a translated request is equivalent.
3. **Separate decision and system tests.** A pure decision microbenchmark can
   compare equivalent algorithms over frozen inputs. A system benchmark must
   drive real scheduler instances and include their admission/control paths.
   Never compare an in-process Eyvan function call with a competitor's complete
   API-to-execution latency under the same metric name.
4. **Publish workload inputs.** Include exact-fit, `[3, 2, 2]` fragmentation,
   mixed-width, budget-constrained and deliberately infeasible cases. Preserve
   seeds, arrival order, initial leases, prices and resource dimensions. Use
   the same affordable-feasibility oracle where semantics actually match.
5. **Measure observable outcomes.** Record accepted/rejected requests, exact
   worker coverage, overcommit violations, number of cells/worlds, reserved
   cost and time to first execution. Measure elapsed time at a common observer
   rather than subtracting unsynchronized timestamps from different machines.
6. **Retain distributions and failures.** Publish warm-up policy, raw samples,
   sample count, p50/p99 definition, process/resource limits and failed cases.
   Repeat in randomized order. Do not remove slow runs or tune budgets after
   observing results without explicitly reporting that change.
7. **Keep hardware evidence distinct.** Any live capacity used for this
   roadmap must be acquired through Eyvan and be Spot-only, with no on-demand
   fallback. Record actual provider/shape/zone, acquisition and cleanup,
   reclamations and exact candidate identity. A local simulation is not a
   hardware receipt, and an IAM refusal is a not-run result.

Completion requires runnable candidate adapters and retained real comparison
receipts. This design write-up alone does not close the Innovation benchmark
item, the Spot checkpoint/restore demonstration or the production pilot.
