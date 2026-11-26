# Operating Eyvan

The runbook. Assumes the concepts from [eyvan-explained](eyvan-explained.md);
installation lives in [install](install.md). The
[receipts and elastic-packing design](design/0006-receipts-and-elastic-packing.md)
separates workload admission from experimental proof and defines the still-open
cross-scheduler comparison protocol.

## Production design contracts

- **Durable state wins.** Cells, workloads, fleets, and leases live in the
  plane's store. Restart with the intended durable state configuration and
  matching credentials so the controller can reconcile its records. Inspect
  quarantined leases before intervening; see [troubleshoot](troubleshoot.md).
- **Restart with credentials, or don't restart.** The plane needs the agent's
  shared secret to authenticate its requests; without it every call to a
  still-running cell agent is rejected with 401. For a profiled local runtime,
  restore it before starting the plane:

  ```bash
  if EYVAN_AGENT_SHARED_SECRET=$(python3 -c '
  import json
  from pathlib import Path
  secret = json.loads((Path.home() / ".eyvan/gcp-native-runtime.json").read_text())["agent_shared_secret"]
  if not isinstance(secret, str) or not secret:
      raise SystemExit("Runtime profile has no nonempty agent_shared_secret")
  print(secret)
  '); then
    export EYVAN_AGENT_SHARED_SECRET
    eyvan-server --log-level info
  else
    printf '%s\n' 'Credential extraction failed; server was not started.' >&2
    false
  fi
  ```

  The profile must contain a nonempty string secret. Missing or malformed
  profiles and invalid secrets stop this command block before server startup;
  do not bypass the check with a bare server command.

- **Cells cost money; the plane reaps them.** Idle cells drain after the
  fleet's idle timeout. If you tear the plane down for the night, drain its
  fleets first: `eyvan down <config> -y`.

## Consume a release immutably

Before changing a production plane, select one candidate full source SHA and
one signed release receipt. Download only the receipt, its signature/bundle,
and the receipt-named assets; verify the receipt's canonical bytes, keyless
signature and Rekor inclusion, then verify every referenced image, package,
chart, render, and attestation digest. Confirm the receipt binds the candidate
source SHA and the exact workflow identity. Deploy the three image references
from the receipt (`control-plane`, `agent`, and, for Kubernetes capacity,
`worker`) as `repository@sha256:...`; do not resolve a tag, rebuild locally, or
substitute `latest`. Keep the verified receipt with the deployment record.

## Every day

```bash
eyvan capacity fleets            # what exists, what is drained
eyvan capacity list              # pools this project can submit into
eyvan cells                      # every cell and its stage
eyvan capacity quota --fits v5p-16:2   # zone-by-zone fit for a shape
eyvan capacity launch-latency    # per-pool handover seconds, plane-recorded
```

`quota --fits` prints free slices per zone and flags zones whose usage is
*unknown* — treat those as unproven, not free. Quota, queue-admission permission,
and current supply are separate checks.

## Running a job on accelerators

```python
from eyvan import TpuConfig, execute, init
from eyvan.records import CapacityClass

init(fleet="research-us-central1-a")

@execute(TpuConfig(type="v5p-16", pods=2, capacity_class=CapacityClass.SPOT))
def run():
    ...
```

- The blocking call narrates state. `blocked | … | the plane is retrying
  (attempt N)` means the provider refused an attempt and the plane is
  retrying — watch, don't intervene, until the attempt budget is spent.
- A queue-admission denial ("User does not have permission to submit requests
  … in location X") removes that zone from the workload's candidates. A
  workload with cross-zone alternates keeps going in the zones that have not
  refused it; once every candidate has refused, it fails with the provider's
  words. A single-zone workload fails on the first denial. Pick another zone.
- `blocked | … | every zone this job accepts has failed to grant capacity five
  times running` means the planner has stopped asking (the acquisition bound,
  counted across replaced cells for an hour). It asks again once that hour
  passes, or at once if any candidate zone grants capacity.
- Racing: with `EYVAN_CAPACITY_RACE_WIDTH=N` the planner asks up to N of a
  workload's candidate zones at once, across regions, never a zone that has
  already denied it. Candidates come from same-shape pools declared in one
  fleet (`zones: [...]`, or one node type per zone).
- Follow up detached: `eyvan workload status|logs|cancel <id>`, and
  `eyvan workload explain <id>` for placement truth — candidate cells,
  provisioning cell ids, delivery state and error.

## Choosing a zone: race it, don't guess

Compare `eyvan capacity launch-latency` samples with a small probe workload
in each candidate zone. Check quota, queue-admission permission, readiness,
and the callable's result separately. Historical launch timings do not
predict current Spot supply. See [proofs](proofs.md) for proof criteria.

## When something is queued and not running

1. `eyvan workload explain <id>` — is it placeable? provisioning? delivery?
2. If `hard_constraint: no cell satisfies all hard workload constraints`
   with a provisioning cell id, capacity is being built — wait.
3. If the provider queue is `WAITING_FOR_RESOURCES` for tens of minutes, the
   zone is congested. Cancel and race another zone rather than waiting
   blind.

## Fleets up and down

```bash
eyvan up --select research-us-central1-a --preemptible   # re-enable a generated zone fleet
eyvan down ~/.eyvan/autoscale/configs/research-us-central1-a.yaml -y   # drain it back
```

`eyvan up --region <region> --preemptible` discovers zones and generates
Spot TPU fleets from live quota, then applies the first one. To select a zone,
use `eyvan up --quota-based --zone <zone> --preemptible` instead; `--region`
and `--zone` cannot be combined. `down` takes the **config path**, not a fleet
name, and drains every cell the fleet manages. An empty fleet holds no cell
and costs nothing, but drains restore the state you found.

## Release, migration, and rollback sequence

1. Freeze new submissions and database writers, and take a verified backup
   (including an isolated restore rehearsal) before a schema-changing release.
2. Verify the immutable receipt and render, create/update the release with the
   exact digest references, and run `eyvan-migrate --check` from that candidate.
3. If the schema is changing, acknowledge maintenance for this release only,
   run the migration while writers remain quiesced, and wait for the migration
   job and control-plane rollout to complete. Reset the acknowledgement before
   reopening writes.
4. Perform the post-deployment checks below, then resume submissions and
   writers. Record the receipt, migration revision, and check results.

A Helm rollback can restore workloads to an older image, but cannot undo a
committed database migration. If the candidate fails after migration, keep
writers stopped and repair forward; restore a backup only as a deliberate
disaster-recovery action. Never use an Alembic downgrade as release rollback.

## Post-deployment checks

- `kubectl rollout status` succeeds for the control-plane and (if enabled) the
  single-owner cell-agent Deployment, and `kubectl get pods` shows all pods
  ready; confirm the rendered pods still use the three verified digests.
- Query `/healthz` and `/readyz` through the production ingress. The latter
  proves database readiness; authenticate an API request with the production
  OIDC path (not development auth) and verify an unauthorized request is
  rejected.
- Scrape `/metrics` with the configured bearer, confirm the scrape is
  accepted and contains current control-plane/service metrics, and check logs
  for authentication or database errors.
- Confirm the cell agent registers and remains healthy, then run a harmless
  capacity/agent operation and inspect its status/logs. In Kubernetes capacity
  mode, verify worker Jobs and their pods start from the expected digest.

## One agent, one backend

The cell agent (`eyvan-agent`, pid/lock/log under `~/.eyvan/cell-agent-*`)
owns machines for its backend and authenticates the plane with
`EYVAN_AGENT_SHARED_SECRET`. The plane and the agent are separate processes:
either can restart independently, and a credentials mismatch shows up as 401
loops on `/api/v1/cells/<id>/…` in both logs. The fix is on the
[troubleshoot](troubleshoot.md) page.
