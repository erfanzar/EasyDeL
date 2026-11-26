# Scheduler regression benchmarks

Behavioral sensitivity is measured separately by
[scheduler mutation testing](scheduler-mutations.md). A timing budget passing
is not evidence that tests reject incorrect scheduling decisions.

## Reproduce

Run the deterministic CPU simulation used by CI:

```bash
uv run python scripts/benchmark_scheduler.py --cells 1000 --iterations 20 \
  --budget-file scripts/scheduler-budgets.json --output benchmark-results/scheduler-1000.json
uv run python scripts/benchmark_scheduler.py --cells 10000 --iterations 100 \
  --budget-file scripts/scheduler-budgets.json --output benchmark-results/scheduler-10000.json
```

## Scenarios and budgets

Each independent iteration constructs mixed-width domain cells, places four
ordinary workloads, then packs one elastic workload with up to 32 workers.
One warm-up iteration is excluded. The harness checks successful placement and
exact elastic worker totals before accepting a sample. It retains raw samples,
nearest-rank p50/p99, interpreter/platform identity and budget violations.

The phases are domain snapshot construction, ordinary `Scheduler.schedule`
and elastic packing. Budgets in `scripts/scheduler-budgets.json` are:

| Cells | Snapshot p99 ms | Schedule p99 ms | Elastic p99 ms |
|---:|---:|---:|---:|
| 1,000 | 25 | 75 | 75 |
| 10,000 | 300 | 900 | 750 |

These are coarse regression guards, not confidence intervals or production
latency commitments. Budget changes need an explained performance change and
retained before/after results, not automatic fitting to the latest sample.
Missing, malformed, incomplete, negative or non-finite budgets are errors.
The script writes its result before exiting 1 on a budget violation.

## CI and failure diagnosis

The Python 3.12 CI entry runs the 1,000-cell guard. The nightly schedule adds
the 10,000-cell simulation with 100 samples. Artifacts retain raw JSON even on
budget failure; an earlier failed step can prevent measurement entirely.

For diagnosis, repeat with `--diagnostics` and a different output path. This
adds process CPU-time samples and per-generation GC collection-count deltas.
It does not force collection, disable GC, change scenarios or budgets, discard
slow samples, or replace the original result. CI retains both results when its
failure-only diagnostic rerun executes; a passing rerun does not waive the
original failure.

With 20 samples, nearest-rank p99 is the maximum. Inspect the distribution,
repeat in controlled reference/candidate blocks, and record process load and
instrumentation. GC counts show overlap, not time spent collecting; CPU versus
wall time is a signal, not proof of contention or a causal explanation.

## Evidence limits

This measures pure scheduling decisions, not database snapshot latency, the
full reconciliation loop, cloud provisioning, training throughput or another
scheduler. Spot is modeled data; the script acquires no cloud capacity.
Controller/database scaling, the [repository soak](../testing-policy.md) and
hardware qualification require separate evidence. Re-run on the candidate and
target runner rather than carrying forward local optimization timings.
