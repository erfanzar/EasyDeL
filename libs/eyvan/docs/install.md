# Installing Eyvan

A network-bound deployment needs explicit identity, storage, and listener
configuration. Local `eyvan up examples/laptop.yaml` starts a loopback CPU
fleet without cloud credentials; it is not a remote installation recipe.
This page covers both paths. Eyvan requires Python `>=3.11,<3.14`,
as declared in `pyproject.toml`. Python 3.10 is no longer supported. Upgrade
existing worker/submitter environments before deploying this version; explicit
unsupported interpreter declarations are refused. Native bootstrap selects a
supported host interpreter, or reports that `runtime.python` must be set when
none is available.

## Ask before you start

Ask `doctor` for all applicable startup refusals before starting the server:

```bash
uv run eyvan doctor --server-host 0.0.0.0     # the address eyvan-server will bind
```

The check reads this machine's `EYVAN_*` environment and reports all
applicable refusals, including listener TLS, PostgreSQL, `EYVAN_METRICS_TOKEN`,
and `EYVAN_ARTIFACT_TOKEN_KEY`, with the reason for each. Authentication-mode
conflicts and team-mode public URL requirements also depend on configuration.
It exits non-zero when any requirement is unsatisfied. Without `--server-host`
it checks the server default, `127.0.0.1`. Loopback binds skip the network-bind
requirements, not every configuration check.

## The fifteen configuration areas

These areas depend on the deployment; they are not fifteen mandatory overrides.

| # | What | Purpose or default | Missing looks like |
|---|------|--------------------|--------------------|
| 1 | PostgreSQL and `EYVAN_DATABASE_URL` | The local repository has no cross-replica leader election. | Startup refuses a network bind/backend mismatch, naming `EYVAN_REPOSITORY_BACKEND`. |
| 2 | The schema, managed by `eyvan-migrate` | Production disables auto-create. Existing-schema changes require maintenance acknowledgement. | Startup reports `SchemaRevisionMismatch`, or the migration safety gate refuses. |
| 3 | An authentication mode: OIDC (`EYVAN_OIDC_ISSUER`), bootstrap bearer (`EYVAN_BOOTSTRAP_TOKEN`), or team accounts (`eyvan-server --team` or `team: true` in `platform.yaml`) | Development auth is refused on a network bind. Bootstrap tokens grant global fleet-admin authority; bootstrap and OIDC cannot be configured together on a network bind. | Requests cannot authenticate without a usable mode and credential. |
| 4 | `EYVAN_OIDC_AUDIENCE` when using OIDC | Defaults to `eyvan`; set it to the audience issued for this API. | Tokens with a different audience are rejected. |
| 5 | The non-secret `EYVAN_OIDC_DASHBOARD_CLIENT_ID` for browser OIDC login | The server-side BFF runs authorization-code + PKCE; the browser holds only a session cookie. | OIDC dashboard login cannot start without its client ID. |
| 6 | An IdP `eyvan_grants` claim or durable grants from `eyvan grant add` | Roles need an authoritative source. | Authenticated requests are 403. |
| 7 | One artifact signing key of at least 32 bytes, identical on every replica, as `EYVAN_ARTIFACT_TOKEN_KEY` | Capability tokens are HMACs. | Startup refuses a network bind naming the setting; an accepted per-process key would give intermittent artifact/secret 403s tracking replica count. |
| 8 | `EYVAN_ARTIFACT_STORE_URI` (`gs://...`) | A pod filesystem is not shared. | One replica cannot read another's result. |
| 9 | `EYVAN_PUBLIC_URL` over HTTPS | Jobs receive capability URLs built from it. | Jobs cannot fetch callables or secrets. |
| 10 | `EYVAN_TRUSTED_CALLABLE_PROJECTS` | A pickled callable executes arbitrary code; empty trusts nobody. | Callable submissions are refused; command workloads still work. |
| 11 | A high-entropy `EYVAN_METRICS_TOKEN` | `/metrics` exposes spend labels and a network server requires scrape authentication. | The server exits naming the setting. |
| 12 | One agent shared secret | The control plane authenticates cell-agent mutations. | Agent operations are refused. |
| 13 | A distinct, stable native-worker root secret | It derives per-cell credentials; reusing the agent secret collapses capability boundaries. | Workers fail authentication or rotation breaks live cells. |
| 14 | `platform.yaml` and one agent per capacity boundary, each with its own `EYVAN_FLEET_STATE_URI` | Shared fleet-state ownership fences competing agents. | Pool upserts lack a route, or cells flap on fence conflicts. |
| 15 | A native worker artifact pinned by digest (Kubernetes) or wheel/wheelhouse SHA-256 plus GCS generation (GCP) | Tags and object names alone can be replaced; private-egress VMs cannot reach PyPI. | Workers pull different code, fail verification, or never register. |

TLS termination, secret-manager integration, and mTLS in front of every external
agent are additional platform prerequisites.

## One command, for a GCP head

`eyvan head up` does the whole sequence for one small CPU VM on GCP: it creates
or adopts a labelled instance in the fleet's own project and zone, restricts
ingress on the control-plane port to the VPC's internal ranges, generates the
credentials the server refuses to start without, points the server at
PostgreSQL, runs the migration, and starts the control plane under systemd so it
survives both ssh exit and reboot.

**The create path has run against real GCP** -- see P0 in
[proofs.md](proofs.md): a head was provisioned, adopted on a re-run, and used to
dispatch a multislice workload that returned its device count. What that receipt
does not cover is every option this command accepts. A flag no live run has
passed -- another egress mode, an external database, an OIDC issuer -- is tested
only against the injected command boundary, where the sequence, the ownership
refusals, the idempotent re-run, the ingress scope and the mid-sequence failure
path all are. Read the plan first:

```bash
eyvan head up fleet.yaml --dry-run
```

`--dry-run` issues no command at all, not merely no mutating command. It prints
every action and every resource name it would use, so the whole thing is
reviewable before any of it is affordable.

```bash
eyvan head up fleet.yaml --yes
eyvan head status fleet.yaml
eyvan head down fleet.yaml --yes
```

`head down` requires `--yes` and never infers confirmation from a terminal: the
head keeps the control plane's durable state on its own disk.

### What it decides for you, and what it refuses to

* **Credentials.** `EYVAN_METRICS_TOKEN`, `EYVAN_ARTIFACT_TOKEN_KEY` and
  `EYVAN_AGENT_SHARED_SECRET` are generated from the system CSPRNG and delivered
  on stdin, never in a command argument where `ps` would show them. They are
  written to `/etc/eyvan/head.env`, mode 0640, root-owned and readable only by
  the service user. A re-run **keeps** them; rotating a live head's artifact
  signing key invalidates every capability token already minted, so that needs
  `--rotate-secrets`.
* **Persistence.** `EYVAN_REPOSITORY_BACKEND=postgres` and `EYVAN_DATABASE_URL`
  are set for you. With no `--database-url`, PostgreSQL is installed on the head
  and a password is generated. With one, the head uses that database and no
  local server is installed. `eyvan-migrate` runs before the service starts.
* **Ingress.** Exactly one rule opens the control-plane port, and only to the
  internal subnet ranges of the VPC in that region -- RFC 1918 or the shared
  100.64.0.0/10 space, nothing else. A discovered range outside those, or a
  region with no subnet at all, is a refusal with the reason stated, never a
  wider rule. `0.0.0.0/0` is not reachable through any flag. `--source-range`
  narrows it further and is validated the same way. With `--iap-ssh` (the
  default) a second rule admits Google's fixed IAP forwarding range on port 22
  so the installer can reach a VM that has no external address; `--no-iap-ssh`
  removes it and assumes you already have connectivity.
* **Egress** for the head itself is one of `cloud-nat`,
  `private-google-access`, or `public-internet`; the value is validated, not
  inferred from what was found lying around.

## The laptop path

```bash
uv sync --extra dev
uv run eyvan up examples/laptop.yaml    # loopback CPU fleet, no cloud credentials
uv run eyvan start                      # discovers: local plane reachable? heads in the project?
```

Run `eyvan up examples/laptop.yaml` before calling the SDK's `init()` for the
local fleet. Bare `eyvan up` selects a cached generated autoscale fleet instead.
State lands under `~/.eyvan/`: `control-plane.json` (the loopback plane's
durable world), `autoscale/defaults.yaml` plus `autoscale/configs/` (generated
zone fleets), and `cell-agent-*` records.

For accelerator fleets, configure `EYVAN_AGENT_SHARED_SECRET` on the plane and
agent and `EYVAN_NATIVE_WORKER_ROOT_SECRET` on the native agent before starting
them as separate processes:

```bash
uv run eyvan-server --log-level info       # the plane
uv run eyvan-agent --port 8090 --log-level info  # requires the configured secrets
```

For a profiled GCP setup both secrets are stored in
`~/.eyvan/gcp-native-runtime.json`. Restore matching credentials before
restarting the plane and agent. Local components rebuild from source with
`uv run eyvan runtime deploy`.

### Dashboard verification

From the repository root, install and verify the dashboard with:

```bash
npm --prefix dashboard ci
npm --prefix dashboard test
npm --prefix dashboard run build
```

See [Contributing — Dashboard](../CONTRIBUTING.md#running-the-tests) for the
same checks and their purpose.

## Raw manifest order

1. Check the image digests: the control plane and agent need artifact-store
   access and fleet-state/bootstrap access. On a cluster without GKE Workload
   Identity, remove the example annotations. The migration and control-plane
   references must be the same registry-reported `repository@sha256:...`
   digest; the cell-agent image is a separate digest from `Dockerfile.agent`.
2. Apply prerequisites, then provision both Secrets.
3. Recreate and wait for the safe migration Job.
4. Only after it completes, apply the agent and control-plane Deployments.

```bash
kubectl apply -f deploy/prerequisites.yaml
# Provision eyvan-secrets and eyvan-agent-secrets with your secret manager here.
kubectl -n eyvan-system get secret eyvan-secrets eyvan-agent-secrets

kubectl -n eyvan-system delete job eyvan-schema-migration --ignore-not-found
kubectl apply -f deploy/migration-job.yaml
kubectl -n eyvan-system wait \
  --for=condition=complete --timeout=10m job/eyvan-schema-migration

kubectl apply -f deploy/cell-agent.yaml
kubectl apply -f deploy/control-plane.yaml
kubectl -n eyvan-system rollout status deployment/eyvan-cell-agent --timeout=10m
kubectl -n eyvan-system rollout status deployment/eyvan-control-plane --timeout=10m
```

Do not use `kubectl apply -f deploy/`. Repeat the delete/apply/wait sequence
before every raw upgrade.

The raw platform ConfigMap does not restart or refresh the existing `subPath`
mount when it changes. ServiceAccount annotation and externally provisioned
Secret changes also do not alter a Deployment pod template. After changing any
of them, apply the object and explicitly restart and wait for every affected
Deployment; in particular, changing the `eyvan-control-plane` WIF annotation
requires a control-plane rollout:

```bash
kubectl apply -f deploy/prerequisites.yaml
kubectl -n eyvan-system rollout restart deployment/eyvan-control-plane deployment/eyvan-cell-agent
kubectl -n eyvan-system rollout status deployment/eyvan-control-plane --timeout=10m
kubectl -n eyvan-system rollout status deployment/eyvan-cell-agent --timeout=10m
```

The Helm chart places a content checksum in the pod template, so a `platform.*`
value change causes a rollout automatically. External Secret content and
ServiceAccount annotation changes are not checksummed there either; restart the
affected Helm-managed Deployments after either change.

## Secret and non-secret configuration

Secrets enter via environment or the runtime profile
(`~/.eyvan/gcp-native-runtime.json`, written `0600`), never via fleet configs
or the repository. Everything else (timeouts, batch sizes, backend selection)
is [environment surface](env-surface.md) with safe defaults; fleet-level
runtime provisioning (interpreter, init commands, default container image)
lives in the [autoscale defaults profile](runtimes.md#the-defaults-profile) so
every generated zone fleet derives it.

## Safe schema upgrades

The server does not run Alembic migrations at startup. It refuses a stamped
schema revision that does not match the build; `EYVAN_AUTO_CREATE_SCHEMA` is
for development schema creation, not production upgrades.

1. Quiesce submissions and every database writer. Verify a backup and rehearse
   an isolated restore.
2. Install the candidate and run `eyvan-migrate --check` to inspect its migration
   policy. A known-behind schema requires a one-use maintenance acknowledgement.
3. With writers still stopped, run `eyvan-migrate --maintenance-acknowledged`
   for that schema upgrade. Omit the acknowledgement for a fresh or exact-head
   database; the migration gate refuses an unnecessary acknowledgement.
4. Start the matching server build, verify it, then reopen writes. Repair failed
   upgrades forward; do not use an Alembic downgrade as rollback.

Keep the previous release's artifacts until the new one has served real work.
The [artifact-safety](artifact-safety.md) page defines what may be cleaned and
when.

## After install

- `eyvan doctor` again, on the head, bound address included.
- `eyvan capacity fleets` -- what exists; generate zone fleets with
  `eyvan autoscale generate --zone <zone>` and apply with
  `eyvan up --select <fleet> --preemptible`.
- Smoke-test with a trivial callable before spending (the zone-race probe in
  [operate](operate.md) is a good template).
