# A bounded Spot TPU priority/checkpoint example

`examples/spot_priority_demo.py` is a dry-run-first operator example, not a
provisioning shortcut. It uses an explicitly selected, already configured Eyvan
control plane and cell agent. `examples/spot_training_payload.py` is the remote
payload. Neither file contains a runtime profile, account credential, or project
selection for the developer's environment.

## Prerequisites and cost boundary

- Use Python 3.11–3.13 on a Linux or macOS host. Paid execution requires POSIX
  interval timers and the main thread so a slow HTTP response cannot evade the
  operation deadline. Planning does not require those timers.
- Configure an authenticated Eyvan plane and GCP-capable cell agent using the
  [runtime guide](../runtimes.md). The running components and staged worker wheel
  must have matching builds. If you selected a separate local provider-state
  path, retain it on every deploy/restart; do not silently switch profiles.
- Select a GCP project and zone with supported `v5p-8` Spot capacity, an existing
  GCS bucket/prefix, and the worker service account's required permissions. See
  [provider IAM](../provider-gcp-iam.md). This example does not grant IAM, create
  a bucket, configure an issuer, or change the active gcloud account/project.
- The controller must be able to reach the explicit agent URL. HTTP is accepted
  only for loopback endpoints; other endpoints require HTTPS. A remote
  authenticated plane can use the existing `EYVAN_TOKEN` environment setting;
  never place a token in an endpoint URL or receipt.
- Keep the local host awake. The fleet permits **one Spot cell at a time**, has
  zero initial/minimum cells, and has no on-demand fallback. `v5p-8` exposes four
  TPU chips; the example derives workload resources from the compiled fleet.
  Stockouts may cause the run to time out without executing anything.

## Plan, then explicitly execute

Run from the repository root. Replace the project, zone and bucket with your
explicitly authorized targets. Planning compiles the requests locally without
reading runtime profiles, resolving credentials or contacting a service:

```bash
uv run python examples/spot_priority_demo.py \
  --url http://127.0.0.1:8080 \
  --org default --project default \
  --gcp-project your-gcp-project \
  --zone us-east5-a \
  --agent-url http://127.0.0.1:8090 \
  --artifact-prefix gs://your-bucket/spot-priority-demo
```

Repeat the same command with `--execute --receipt-dir ./spot-demo-results`
only after reviewing the plan and accepting paid Spot allocation. Every
invocation generates a new run/fleet identity; a dry-run identity is not reused.
Existing fleets are refused, not overwritten. The default workflow deadline is
2,400 seconds; `--timeout` accepts 60–3,600 seconds. Each complete SDK operation
has at most 15 seconds, including retries/body consumption and clipped to the
remaining phase deadline. Cleanup has its own 120-second budget.

The run performs these steps:

1. Apply its uniquely labelled, zero-initial-cell Spot fleet through Eyvan.
2. Submit a checkpointable batch workload restricted to that fleet. The payload
   checks its metadata project, Spot/preemptible status and four-device TPU
   backend before computation.
3. Run deterministic data-parallel training to step 20 and create a JSON GCS
   checkpoint with a create-only generation precondition. The trainer waits;
   its checkpoint hook validates the durable object **without initializing
   JAX**, because the trainer still owns the TPU while its hook runs.
4. Submit an interactive workload. Eyvan performs priority checkpoint/stop,
   runs the interactive TPU computation, and resumes the batch workload.
5. Require batch attempt 2, no execution-failure/provider-preemption/capacity-loss
   increments, the expected checkpoint URI, step-40 weights matching an
   uninterrupted all-replica control, and the same checkpoint/restored VM name.
   Log cursors advance and matching markers are retained across pages.

This is **same-host scheduler priority preemption**, not natural Spot reclamation
or replacement-host recovery. It contains no provider VM-deletion command or
fault-injection option. Earlier hardware runs are not automatic qualification
of a later example revision.

## Receipts, failures and cleanup

The private receipt records the generated fleet, admission identities, observed
workload states, payload SHA-256 and bounded verification/cleanup results. It
stores exception classes rather than credential-bearing exception messages.
Checkpoint objects and receipts are retained for inspection, not deleted by the
example. Keep this directory out of Git.

Cleanup first disables only the matching run-labelled fleet, then cancels owned
workload identities. Cancellation retries cannot consume the budget before the
fleet-disable attempt. If an admission response was lost, it reconciles the exact
original request/idempotency key rather than inventing a new identity. A receipt
write failure before sending the POST does **not** trigger a new submission
from cleanup. Ownership reads and cleanup mutations have bounded retries;
ownership verification is never bypassed to force deletion.

Ctrl-C and SIGTERM request cleanup; during cleanup they are recorded and deferred
without disabling the operation deadlines. SIGKILL, host failure, or an unavailable
control plane can still prevent cleanup. Inspect the retained fleet
and workload identifiers and perform operator cleanup rather than abandoning
resources. `api_cells_deleted` means the Eyvan API reported all owned cells as
deleted. **It is not an independent provider-absence check**:
`provider_absence_verified` remains false. Inspect GCP for the exact owned cell
labels and pending operations before declaring cloud teardown complete. Missing
cell-status data remains unknown, not an empty/deleted fleet.

## Local tests

```bash
uv run pytest tests/test_spot_priority_demo.py tests/test_spot_training_payload.py -q
```

These cover planning isolation, typed requests, malformed checkpoints, metadata
and hook boundaries, marker pagination, API-double orchestration, ambiguous
admission, interruption/deadline paths, and cleanup failures. A real loopback
slow-drip HTTP test exercises the total-operation deadline. They do not execute
JAX on TPUs or prove deployed controller/provider behavior.
