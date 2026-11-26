# Part 14 — Operations

The runbook: `docs/operate.md`, `docs/troubleshoot.md`, `docs/proofs.md`. This
part compresses them into the working loop.

## Every day

```bash
eyvan capacity fleets            # what exists, what is drained
eyvan capacity list              # pools this project can submit into
eyvan cells                      # every cell and its stage
eyvan capacity quota --fits v5p-16:2   # zone-by-zone fit for a shape
eyvan capacity launch-latency    # per-pool handover seconds, plane-recorded
eyvan events --org my-org --project my-project   # scoped audit feed
```

`quota --fits` flags zones whose usage is *unknown*; treat them as unproven, not
free. A quota row does not establish queue admission or available supply. Verify
all three before relying on a zone.

## Production design contracts

- **Recover from durable state.** Restart the plane against the same store and
  configuration. Check existing workload IDs, leases, and pending command delivery
  with `eyvan workload explain <id>` and the scoped audit feed. A listening API
  alone does not establish that reconciliation has completed.
- **Restart with credentials, or don't restart.** A bare `eyvan-server` comes up
  without the runtime profile's secrets and 401-loops against a still-running
  agent (cells flip `unavailable` in a recreate loop). Restore the agent secret
  from the profile, refusing startup if extraction fails:
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
  This restores only the agent bearer secret, not the complete runtime profile.
  Native-worker credentials, TLS material and callable trust still require their
  configured restart path; do not bypass a failed extraction with a bare server
  command.
- **Cells cost money; the plane reaps them.** Idle cells drain after the fleet's
  idle timeout. Tearing down for the night: drain fleets first
  (`eyvan down <config> -y`).

## Running a job on accelerators

- `blocked … the plane is retrying (attempt N)` = the provider refused an attempt
  and the plane is retrying within its budget — **watch, don't intervene**, until
  the attempt budget is spent.
- A queue-admission denial ("no permission … in location X") is **terminal** —
  pick another zone.
- Follow detached: `eyvan workload status|logs|cancel <id>`, and
  `eyvan workload explain <id>` for placement truth.

## Choosing a zone

1. Run `eyvan capacity quota --fits v5p-16:2` for the required shape. Keep unknown
   usage separate from known quota headroom; check recorded admission refusals.
2. Inspect `eyvan capacity launch-latency`. Compare handover times and sample
   counts for relevant pools. No samples means unmeasured, not fast.
3. If live validation is needed, explicitly authorize a bounded probe with the
   same shape and purchase class. Check queue admission, worker readiness, and
   an actual workload result separately.
4. Cancel owned probe workloads, drain their capacity, and verify provider
   cleanup. Repeat when shape, zone, or runtime changes.

Quota, queue admission, and available supply are separate gates. Previous launch
times do not guarantee current Spot availability.

## When something is queued and not running

1. `eyvan workload explain <id>` — placeable? provisioning? delivery state?
2. `hard_constraint: no cell satisfies all hard workload constraints` **with** a
   provisioning cell id → capacity is being built; wait.
3. With **no** provisioning cell, check matching pools, pool bounds, shared quota,
   and acquisition backoff using `eyvan capacity fleets`, `eyvan capacity list`,
   and workload explanation. A matching pool alone does not guarantee a launch.
4. Provider queue `WAITING_FOR_RESOURCES` for tens of minutes → the zone is
   congested. Cancel and race another zone rather than waiting blind.

## Troubleshooting quick table

| symptom | cause | action |
|---|---|---|
| `EyvanUnavailableError: control plane is unavailable` | plane process died | restart `eyvan-server`; state reconciles |
| 401 loops on `/api/v1/cells/<id>/…` after a plane restart | bare restart without the agent secret | restart with `EYVAN_AGENT_SHARED_SECRET` from the runtime profile |
| agent exits: shared secret required | agent started bare | pass the secret / use the profile; `--allow-unauthenticated` is dev-only |
| cells `unavailable` + recreate loop | plane↔agent auth mismatch or provider `PROVIDER_ERROR` | read both logs |
| `leased · attempt N` stuck; `explain` shows quarantined lease | plane crashed mid-dispatch | cancel and resubmit (known rough edge) |
| workload stuck in CANCELLING | cancel raced a dispatch/reap pass | give the controller a pass; `eyvan workload-force-finalize` for the stuck case |
| `GcsCommandError … HTTPError 412` on submit | an object appeared during a digest-addressed upload | current code re-hashes the stored object and continues only if it matches; report a regression if 412 still escapes |
| `eyvan down/up` says config missing | given a fleet name, not a path | use `~/.eyvan/autoscale/configs/<fleet>.yaml` |

## Proof runs

Hardware claims go through `eyvan proof` — executable pass criteria with four exit
codes: **0** pass, **1** fail, **2** did not exercise the behaviour, **3**
refusal (the run never happened and must not wear a verdict). Verdict 2 is the one
that matters: a spot slice never reclaimed, a pool that never got a cell, an
observation outside the warm window — each produces a clean-looking run that
establishes nothing. Subcommands: `catalog`, `children`, `logs`, `preemption`,
`warm-window`, `secret`, `kubernetes`, `verify`.

`docs/proofs.md` defines the contracts: catalog shape, child identity, ranked logs,
secret handling, preemption accounting, warm windows, and Kubernetes transport.
Choose the contract before running, retain the build identity and verdict, and
state which dimensions were not exercised. Historical results do not qualify a
changed build or deployment.

## Watchdogging

Alert on: plane-down (health surface `/healthz` + controller status), 401 loops
between plane and agent, and cells sitting in `unavailable` or `draining` past a
threshold. Each of these is distinguishable in `eyvan events` before it costs a
night.
