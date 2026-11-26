# Proofs and receipts

Host, static, and real-PostgreSQL suites exercise this package. None of that is
evidence about a TPU, GPU, provider lifecycle, or the current release image.
**No historical run qualifies the current candidate without an immutable
exact-candidate receipt.** Hardware runs do not substitute for CPU-establishable
software gates, and a host-suite pass never proves device execution.

Each proof defines an acceptance criterion and shape before capacity is
acquired. Record exact client, plane, agent and worker identities, observations,
unverified dimensions and owned-resource cleanup. A proof receipt's digest is
not a signature or a release attestation.

`eyvan proof catalog | children | logs | preemption | warm-window | secret |
kubernetes` fetches observations and evaluates criteria in
`eyvan/sdk/commands/proof.py`. Exit status is 0 for pass, 1 for failure, 2 for
inconclusive/not exercised, and 3 for pre-run refusal. No reclamation, no cell,
or an observation outside the required window is not a pass.

## P0 — a callable runs on a multislice allocation

Establishes the decorator path end to end: submission, atomic capacity,
dispatch and a real multi-host device result.

**Passes when** the callable returns the whole allocation's device count and
the workload completes without manual intervention. For the shape below,
expect 16 devices across four hosts, four chips per host—not 16 per host.

**Shape** v5p-16 × 2, Spot, one multislice queued resource. The catalog defines
each v5p-16 slice as eight chips on two hosts. Budget for provisioning plus a
short device-query workload; acquisition latency is not guaranteed.

## P1 — the hardware catalog is authoritative for hosts per slice

Establishes that generated pools use the sourced catalog rather than guessed
host counts or admission resources.

**Passes when** the pool's positive CPU, memory and TPU resources match the
catalog row and observed host count agrees per slice. For multislice cells,
compare hosts per slice rather than the cell's total with a single-slice row.

**Shape** v5p-16, first one slice, then a multislice allocation.

## P2 — multi-host structured logs, with a rank lost mid-stream

Establishes flattened global-rank logging and pagination under a real missing
writer, not only a simulated transport.

**Passes when** every rank appears, records carry timestamps, streams page
consistently, and a rank lost mid-stream is reported unavailable on the next
page rather than silently truncating history. Rank ordering is not causal
ordering between hosts.

**Shape** v5p-16 multi-host. Any deliberate failure injection requires its own
authorization and an owned-resource cleanup plan.

## P3 — in-job submission and exec

Establishes child-submission authority and exec under the owner's identity.

**Passes when** every child's `created_by` is the submitting human rather than
a service identity, and exec answers on every rank while the trainer remains
RUNNING.

**Shape** CPU fleet first, then v5p-16. CPU results do not qualify accelerators.

## P4 — secret delivery, per attempt

Establishes that declared `secret_mounts` resolve at dispatch for every attempt.

**Passes when** the job reads the value and exits 0 across at least two
attempts. `eyvan proof secret` requires exactly one `secret.issued` and one
`secret.read` per attempt, no read without issuance, and no value in logs,
records or audit payloads. A multi-attempt run also checks redelivery.

**Shape** Any cell; CPU is sufficient. Use a disposable secret and never put
its value in the evidence.

## P5 — Spot reclamation is charged to the preemption budget

Establishes restart under `max_preemption_retries`, without charging the job's
ordinary failure budget.

**Passes when** reclamation is classified as preemption, only the preemption
retry budget increments, the job resumes (from a checkpoint where declared),
and `--max-retries 0` does not fail it.

**Shape** v5p-16 Spot, opportunistically run alongside real work until
reclaimed. Same-host priority preemption and injected capacity loss do not
establish natural provider reclamation.

## P6 — a warm window holds and releases across its boundary

Establishes that the declared floor follows the warm-window policy without
operator intervention.

**Passes when** the floor is held for the window's duration and released within
one controller pass of its end, with events recording both boundaries.

**Shape** CPU fleet; observe the entire window and a reconciliation margin.

## P7 — Kubernetes against a real API server

Establishes the capacity adapter against a real API server, not injected clients.

**Passes when** provider lifecycle failures map to the documented
`CapacityProviderFailureCode` set and node-pool claims round-trip. Check Kueue's
exclusive-queue/no-preemption prerequisites when queue admission is configured.

**Shape** Kubernetes CPU node pool first, then a GPU node pool. Record API,
image and device/runtime identities separately.

## Qualification limits

Keep software checks, provider lifecycle, device execution, injected loss,
priority checkpoint/restore and natural reclamation as distinct results.
Re-run applicable proofs against the intended candidate and deployment. Publish
only sanitized evidence; local state paths and incident transcripts are not a
support matrix or current-candidate proof.
