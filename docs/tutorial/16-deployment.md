# Part 16 — Deployment

Sources: `docs/install.md`, `docs/operate.md`, `deploy/` (`prerequisites.yaml`,
`migration-job.yaml`, `cell-agent.yaml`, `control-plane.yaml`,
`helm/eyvan/`), `Dockerfile*`, `docker-compose.yml`, `oci/`.

## Network deployment prerequisites

Configure these for a network-bound deployment. The loopback development path has
different defaults; see Part 2. Full setup details are in `docs/install.md`:

1. **PostgreSQL** + `EYVAN_DATABASE_URL` — the local store has no cross-replica
   leader election. A network bind with the local backend is refused at startup.
2. **Schema via `eyvan-migrate`** — production disables auto-create
   (`EYVAN_AUTO_CREATE_SCHEMA`); existing-schema changes need maintenance
   acknowledgement. A revision mismatch fails closed (`SchemaRevisionMismatch`).
3. **Authentication** — OIDC, bootstrap bearer authentication, or team accounts.
   For OIDC, configure `EYVAN_OIDC_ISSUER`, `EYVAN_OIDC_AUDIENCE`, the non-secret
   `EYVAN_OIDC_DASHBOARD_CLIENT_ID`, and IdP `eyvan_grants` or durable grants from
   `eyvan grant add`. The dashboard uses a BFF: the server completes the
   authorization-code/PKCE exchange and keeps provider tokens server-side.
   Bootstrap identity uses the generated bearer token instead of an OIDC issuer;
   team accounts are turned on by `eyvan-server --team` or `team: true` in the
   platform file (`/etc/eyvan/platform.yaml`); there is no environment variable
   for it. Development headers are not a network authentication option.
4. **Listener TLS** — `EYVAN_TLS_CERT_FILE` and `EYVAN_TLS_KEY_FILE`, or the
   explicit `EYVAN_INSECURE_PLAINTEXT_BIND=1` acknowledgement. TLS termination at a
   proxy does not waive this listener requirement.
5. **`EYVAN_ARTIFACT_TOKEN_KEY`** — one signing key ≥32 bytes, identical on every
   replica (capability tokens are HMACs).
6. **`EYVAN_ARTIFACT_STORE_URI`** (`gs://...`) — a pod filesystem is not shared.
7. **`EYVAN_PUBLIC_URL`** over HTTPS — jobs receive capability URLs built from it.
8. **`EYVAN_TRUSTED_CALLABLE_PROJECTS`** — a pickled callable executes arbitrary
   code; network deployments must explicitly trust the submitting projects.
9. **`EYVAN_METRICS_TOKEN`** — `/metrics` requires scrape auth on a network bind.
10. **Agent shared secret** (`EYVAN_AGENT_SHARED_SECRET`).
11. **A distinct native-worker root secret** — reusing the agent secret collapses
    capability boundaries.
12. **`platform.yaml` + one agent per capacity boundary**, each with its own
    `EYVAN_FLEET_STATE_URI` — shared fleet-state ownership fences competing agents.
13. **A worker artifact pinned by digest** (Kubernetes) or wheel/wheelhouse
    SHA-256 + GCS generation (GCP) — tags can be replaced, and private-egress VMs
    cannot reach PyPI.

Configure secret-manager integration and authenticated TLS for external agent
connections separately. `eyvan doctor --server-host <address>` previews server
startup refusals; it does not validate every deployment dependency.

## Path A: `eyvan head up` — one GCP VM

```bash
eyvan head up fleet.yaml --dry-run   # prints every action + resource name; issues nothing
eyvan head up fleet.yaml --yes
eyvan head status fleet.yaml
eyvan head down fleet.yaml --yes     # requires --yes; never infers confirmation
```

What it decides for you: creates or adopts a labelled instance in the fleet's
project/zone; restricts control-plane ingress to the VPC's internal ranges (RFC
1918 or 100.64.0.0/10 — `0.0.0.0/0` is reachable through **no** flag); generates
the credentials the server refuses to start without (CSPRNG, delivered on stdin,
written to `/etc/eyvan/head.env` mode 0640); installs PostgreSQL and a generated
password (or uses `--database-url`); runs `eyvan-migrate`; starts the plane under
systemd. A re-run keeps secrets (rotating them live needs `--rotate-secrets`).
Egress is explicitly chosen: `cloud-nat`, `private-google-access`, or
`public-internet` — validated, not inferred.

## Path B: Helm

`deploy/helm/eyvan/` (see its README and `values.yaml`). The chart places a
content checksum in the pod template, so a `platform.*` value change causes a
rollout automatically. **External Secret content and ServiceAccount annotation
changes are not checksummed** — after either, restart the affected Deployments:

```bash
kubectl -n eyvan-system rollout restart deployment/eyvan-control-plane deployment/eyvan-cell-agent
kubectl -n eyvan-system rollout status deployment/eyvan-control-plane --timeout=10m
```

## Path C: raw manifests

`deploy/prerequisites.yaml` → provision `eyvan-secrets` and `eyvan-agent-secrets`
with your secret manager → the migration Job → only after it completes, the
Deployments:

```bash
kubectl apply -f deploy/prerequisites.yaml
# provision secrets here
kubectl -n eyvan-system delete job eyvan-schema-migration --ignore-not-found
kubectl apply -f deploy/migration-job.yaml
kubectl -n eyvan-system wait --for=condition=complete --timeout=10m job/eyvan-schema-migration
kubectl apply -f deploy/cell-agent.yaml
kubectl apply -f deploy/control-plane.yaml
kubectl -n eyvan-system rollout status deployment/eyvan-cell-agent --timeout=10m
kubectl -n eyvan-system rollout status deployment/eyvan-control-plane --timeout=10m
```

Do **not** `kubectl apply -f deploy/`. Repeat the delete/apply/wait sequence before
every raw upgrade. The raw ConfigMap does not refresh an existing `subPath` mount;
a ServiceAccount WIF annotation change requires an explicit control-plane rollout.

## Images

`Dockerfile` (control plane), `Dockerfile.agent` (cell agent),
`Dockerfile.worker` (native worker for Kubernetes capacity), `oci/` (release
tooling), `docker-compose.yml` for local composition. Deploy images as
`repository@sha256:...` from a verified release receipt — never a tag, never
`latest` (see "Consume a release immutably" in `docs/operate.md`).

## Release, migration, rollback

1. Freeze submissions and database writers; take a verified backup (with an
   isolated restore rehearsal).
2. Verify the immutable receipt; run `eyvan-migrate --check` from the candidate.
3. If the schema changes: acknowledge maintenance for this release only, migrate
   while writers are quiesced, wait for the migration Job and rollout, reset the
   acknowledgement before reopening writes.
4. Post-deployment checks (below), then resume.

**A Helm rollback can restore workloads to an older image but cannot undo a
committed migration.** If the candidate fails after migration: keep writers
stopped and repair forward. Never use an Alembic downgrade as release rollback;
restore a backup only as deliberate disaster recovery.

## Post-deployment checks

- `kubectl rollout status` green for control-plane and cell-agent; pods use the
  three verified digests.
- `/healthz` and `/readyz` through the production ingress (readyz proves database
  readiness); authenticate a real request via the configured authentication path
  and verify an unauthorized request is rejected.
- Scrape `/metrics` with the configured bearer; confirm current metrics; check
  logs for auth/database errors.
- The cell agent registers and stays healthy; run one harmless capacity operation.
  In Kubernetes capacity mode, verify worker Jobs start from the expected digest.

## After install

- `eyvan doctor` on the head, bound address included.
- `eyvan capacity fleets`; generate zone fleets (`eyvan autoscale generate --zone
  …`) and apply (`eyvan up --select <fleet> --preemptible`).
- Smoke-test with a trivial callable before spending (Part 14's zone-race probe is
  the template).
