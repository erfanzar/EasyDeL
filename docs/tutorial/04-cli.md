# Part 4 — The CLI

`eyvan` is a Click app composed in `eyvan/sdk/cli.py`. Global options: `--url`
(`EYVAN_URL`), `--token` (`EYVAN_TOKEN`), `--user` (`EYVAN_DEV_USER`), `--org`,
`--project`, `--role` (default `operator`). Against a loopback plane with no
explicit identity it upgrades to `fleet_admin` and prints one dim line saying so —
it is your own development plane, and it tells you who you are acting as.

The endpoint resolution is: flag → saved context (`eyvan use`) → this node's own
plane. Run `eyvan <command> --help` for authoritative options; this part maps the
territory.

## Connection and orientation

| command | what it does |
|---|---|
| `eyvan start` | discover what exists: a local plane, or heads in your project |
| `eyvan connect` | connect to / inspect a control plane |
| `eyvan use` / `eyvan context` | save / show the active context (URL, org, project, fleet) |
| `eyvan doctor` | what is missing, named together; `--server-host 0.0.0.0` previews network-bind refusals |

## Fleets and capacity

| command | what it does |
|---|---|
| `eyvan up [CONFIG]` | apply a fleet config (`examples/laptop.yaml`, a generated zone fleet, `--region/--zone --preemptible`); `--select <fleet>` re-enables one; on a loopback URL it also starts and supervises the plane |
| `eyvan down <config-path> -y` | drain every cell the fleet manages — takes the **config path**, not a fleet name |
| `eyvan status` | fleet status |
| `eyvan config validate <file>` | validate a config without applying it |
| `eyvan capacity fleets` | what exists, what is drained |
| `eyvan capacity list` | pools this project can submit into |
| `eyvan capacity quota --fits v5p-16:2` | zone-by-zone fit for a shape (quota **and** admission **and** supply are three different gates) |
| `eyvan capacity launch-latency` | per-pool handover seconds the plane itself recorded |
| `eyvan capacity warmth` / `queue-trend` / `billing` / `usage` / `carbon` / `builds` / `orphans` / `invariants` | the capacity reports |
| `eyvan capacity pool list/get`, `eyvan capacity cell get`, `eyvan capacity ingress list/release` | pool/cell/ingress detail |
| `eyvan gcp browse [--project P] [--json]` | list existing TPU VMs and unreachable locations; a terminal can pick and confirm attachment, while JSON/non-TTY use only lists |
| `eyvan gcp connect --tpu NAME [--project P] [--zone Z] --dry-run` | preview attaching an existing TPU without creating, stopping or deleting it |
| `eyvan gcp connect --ssh user@host --dry-run` | discover a TPU slice or preview a prepared CPU native-worker host; CPU/memory overrides are optional |
| `eyvan race tpu TYPE --project P [--zone Z] [--keep-first] [--account EMAIL] [--json]` | request one TPU shape in every zone that offers it, report who grants it, then delete everything |
| `eyvan race sweep RUN_ID [--project P]` | delete whatever a race left behind and confirm its zones are clean |

### Race for a TPU shape

Quota is not queue access: a zone can show a large Spot quota and still refuse
the request (code 7), leave it waiting, or suspend it after a few minutes. `eyvan
race tpu` asks every zone at once:

```bash
eyvan race tpu v6e-8 --project my-project --account me@example.com --yes
eyvan race tpu v6e-8 --project my-project --zone us-east5-b --keep-first --yes
eyvan race sweep 0a1b2c3d
```

Without `--zone`, every zone of the project whose catalog offers the shape is
raced. Spot is the default (`--on-demand` switches). By default it only
measures: each grant is deleted as soon as it is seen. `--keep-first` keeps the
first grant, stops the others, and prints the `eyvan gcp connect` and delete
commands; the kept TPU bills until deleted. `--account` mints the token as that
account for this process only; gcloud's configured account is not changed.
`--timeout` (minutes, default 30) bounds the wait per zone.

Every request is named `eyvan-race-<run id>-<zone>` and labelled `eyvan-race`.
Cleanup waits a request out of `PROVISIONING` (where the API refuses deletes),
and a final sweep lists every raced zone again; a zone that cannot be listed is
reported as unknown, not clean. Ctrl-C stops the race but not the cleanup. The
receipt is written to `~/.eyvan/races/<run id>.json` (or `--receipt`), and
`eyvan race sweep` reads it. Without `--yes` the command asks before creating
anything and refuses to run unattended.

### Attach existing hardware

```bash
eyvan gcp browse --project my-project --json
eyvan gcp connect --tpu my-tpu --project my-project --zone us-east5-a --dry-run
eyvan gcp connect --tpu my-tpu --project my-project --zone us-east5-a
# Once ready, use the printed fleet name in init(fleet="...").
```

TPUs must be READY and have a shipped catalog shape and an address on every
host. Their actual Spot/on-demand/reserved class is retained. A missing zone is
resolved from project inventory; ambiguous names or unreachable locations require
an explicit `--zone`. For `--tpu`, `--endpoint-mode public` is the default: the CLI
reuses the profile-owned laptop IPv4 `/32` ingress rule and adds its tag.
`internal` uses internal addresses for workers and SSH/SCP. For a TPU discovered
through `--ssh`, the CLI probes worker reachability and may select an SSH tunnel;
`--endpoint-mode tunnel` explicitly selects that route and requires `--ssh`.

Execution builds the local worker wheel, copies it to every host, delivers a
per-incarnation token over SSH stdin into a root-owned `0600` file, and reuses the
native-worker Python/venv/systemd bootstrap. It requires the local native GCP
runtime profile (`eyvan runtime init`) and passwordless sudo on the TPU hosts.
Wheel copies and stdin-bearing SSH calls select each worker separately. Parallel
SSH children cannot consume one another's private input. No IAM or Secret Manager
changes are made.

After every worker passes health checks, the command writes
`~/.eyvan/attached/<fleet>.yaml`, applies it through the same service as `eyvan up`,
activates its context and waits for readiness. `--fleet NAME` overrides the stable
name; `--timeout SECONDS` bounds worker-health and fleet-readiness waits.
`--dry-run` reads inventory and prints every step and the YAML without builds or
mutations. It may run read-only SSH probes. Connect never prompts, including in a
non-TTY session; omit `--dry-run` only when ready to attach. The hidden `--yes`
option remains for compatibility but is not required. `--json` keeps stdout to
one plan/result document. A failure leaves installed workers/ingress intact,
names its step, and never destroys hardware as rollback.

With `--ssh`, the CLI first discovers whether the address belongs to a TPU through
the TPU API or host-side discovery. A TPU is attached as its **whole slice**, with
shape, hosts, and purchase class discovered rather than supplied by the caller.
Otherwise it reads CPU and memory over SSH for a plain host. That path requires
an already-prepared native worker, an onprem-ssh agent with matching authentication,
and an absolute worker state directory writable by the SSH user. It does not
install software or create firewall rules on a plain host. `--cpus` and
`--memory-gib` are optional resource overrides for plain hosts only; their capacity
class defaults to `reserved` (`--capacity-class on_demand` is also supported,
never Spot), and `--region` defaults to `onprem`.

A successful plan does not establish worker reachability or TPU execution.
Verify readiness and run a workload before relying on an attached fleet.

## Autoscale (generated zone fleets)

```bash
eyvan autoscale init                      # once: edit ~/.eyvan/autoscale/defaults.yaml interactively
eyvan autoscale generate --zone us-central1-a   # cache configs/<prefix>-<zone>.yaml from live quota
eyvan up --select research-us-central1-a --preemptible
```

Generation discovers GCP TPU quota, resolves typed profiles, filters documented
exclusions and cached queue-admission denials, and compiles one validated fleet per
eligible zone. Generated files are derived — never hand-edit them; regenerate after
changing defaults.

## Heads (a GCP control-plane VM)

```bash
eyvan head up fleet.yaml --dry-run   # print every action and resource name; issues nothing
eyvan head up fleet.yaml --yes       # create/adopt the VM, generate credentials, run migrations, systemd
eyvan head status fleet.yaml
eyvan head rollback fleet.yaml --yes
eyvan head down fleet.yaml --yes     # requires --yes; never infers confirmation
```

`head down` demands `--yes` because the head keeps the control plane's durable
state on its own disk.

## Workloads

Top-level convenience: `eyvan run` (run a callable/command workload),
`eyvan workload exec` (exec beside a running workload).

The `eyvan workload …` family:

- submission: `submit`, `submit-batch`, `submit-dag`, `estimate`
- inspection: `list`, `get`, `status`, `timeline`, `cost`, `explain`,
  `process-status`, `logs`, `log-page`, `children`, `endpoints`
- lifecycle: `cancel`, `checkpoint`, `retry`, `kick`

`explain` is the diagnostic workhorse: it prints whether the workload is placeable,
candidate cells, provisioning cell ids, and delivery state/error.

## Operations

| command | what it does |
|---|---|
| `eyvan cells` | every cell and its stage |
| `eyvan cell-drain` / `cell-delete` | drain/delete a cell by id |
| `eyvan leases` | active leases |
| `eyvan events --org ORG --project PROJECT` | the scoped audit event feed; both scope flags are required |
| `eyvan preemption-force-finalize` / `workload-force-finalize` | the stuck-case escape hatches (a cancel that raced a dispatch pass) |

## Governance, tenancy, federation

- `eyvan quota list/get/set/delete`, `eyvan budget list/get/status/set/roll`,
  `eyvan policy list/get/set`, `eyvan webhook list/add/remove`,
  `eyvan system-reconcile` — policy surfaces
- `eyvan org list/get/create`, `eyvan project list/create`,
  `eyvan grant list/add/revoke` — tenancy and role grants
- `eyvan federation peer list/add`, `eyvan federation grant issue/revoke` (federation budget grants),
  `eyvan federation quota-grant list/issue/revoke` — plane-to-plane federation

## Proof, soak, services, query, resources

- `eyvan proof catalog | children | logs | preemption | warm-window | secret |
  kubernetes | verify` — executable pass criteria; exit 0 pass, 1 fail,
  2 did-not-exercise, 3 refusal (see Part 14)
- `eyvan soak` / `eyvan soak-monitor` — longitudinal fleet checks
- `eyvan service health` / `service decisions` — managed-service replica health
  and explained decisions
- `eyvan query` — read-only query console (SQL-backed repositories only);
  `eyvan process stacks/allocations` — process diagnostics
- `eyvan list <kind>` — list `workloads`, `cells`, `pools`, `leases`, `events`,
  `fleets`, `quotas`, or `budgets`
- `eyvan get <kind> <id>` — get a single `workloads`, `cells`, `pools`, `leases`,
  or `fleets` record; events, quotas, and budgets are list-only on this surface

## Artifacts

`eyvan artifacts list | usage | inventory | collectable | references | collect |
hold | release-hold` — the managed artifact ledger (Part 11). A listing failure is
never rendered as an empty scope; `collectable` output is report-only and is not a
deletion claim.

## Runtime and dashboard

- `eyvan runtime init` — stage the local native runtime profile
  (`~/.eyvan/gcp-native-runtime.json`: agent shared secret, native worker root
  secret, artifact root, wheel URIs)
- `eyvan runtime restart` / `runtime deploy` — rebuild and redeploy the local
  plane and agent from current source
- `eyvan dashboard` — open the console already hosted by the configured plane;
  it does not start a server. `eyvan dashboard-dev` starts the source-checkout
  Vite development server against the plane (Part 13).
