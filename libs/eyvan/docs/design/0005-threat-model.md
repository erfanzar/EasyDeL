# DR-0005: Eyvan threat model and release review

**Status:** Working security record. It documents implemented boundaries and
controls, and names gaps that remain open. It is not an independent security
review, a production-pilot report, or authorization to expose a deployment.

## System and trust boundaries

Eyvan has these principal actors and boundaries:

- **Submitter / SDK to the control plane.** A caller submits a workload,
  signed or digest-addressed callable artifact, and workload metadata. The API
  authenticates the caller with OIDC bearer verification, a bootstrap token, or
  loopback-only development credentials, then applies grants and policy.
- **Browser to the dashboard/API.** The dashboard is served by the plane. The
  OIDC authorization-code exchange, PKCE verifier, nonce, access token, and
  session live server-side; the browser receives an HttpOnly, Secure,
  SameSite=Lax session cookie. Cookie-authenticated writes require the configured
  public origin.
- **Control plane to repository and outbox.** Durable workload, lease, policy,
  audit, artifact metadata, and capacity state are an authority boundary. The
  scheduler produces decisions from a snapshot; application revalidates them in
  a transaction. External calls are delivered through durable outbox commands,
  not from inside database transactions.
- **Control plane to cell agent.** The plane leases and fences capacity; the
  agent owns its machine and delegates execution. The normal shared secret is
  `EYVAN_AGENT_SHARED_SECRET`. Mutual TLS is available but defaults off and
  requires an ordered rollout; the agent endpoint is not itself a TLS
  terminator by default.
- **Agent to native worker.** Worker bring-up is fenced by
  `EYVAN_NATIVE_WORKER_ROOT_SECRET`. The native-worker service is a trusted
  platform component holding execution authority; the workload code it runs
  is untrusted and must not inherit that authority.
- **Plane/agent to cloud or infrastructure provider.** Provider credentials
  belong to the process that talks to that provider. Provider responses and
  capacity observations are external input; provider failure or uncertainty
  must not be converted into successful placement or deletion.
- **Artifact and secret capability boundaries.** Artifact, workload-submit, and
  secret capabilities use distinct signed version prefixes and claim checks.
  Secret capabilities are bound to workload, project, attempt, generation,
  dispatch fence, and mount, and expire. Artifact identity is scoped by
  `(org_id, project_id, uri)` and lifecycle operations are generation-fenced.
- **Operator, deployment, and identity-provider boundaries.** Operators control
  host files, environment, provider projects, TLS material, OIDC configuration,
  databases, and object stores. Eyvan does not claim to secure a compromised
  operator environment, an identity provider, or provider account. Peer
  registration and budget/quota grant federation are implemented. Shared-fleet
  ownership federation and automatic failover remain open pending external PKI
  and a quorum-backed ownership witness.

## Assets and security objectives

The primary assets are workload code and results, artifact bytes and metadata,
secrets and capability tokens, bearer/bootstrap/session credentials, tenant and
project identity, lease/fence state, provider credentials, audit records,
spend/metrics data, and release artifacts. The objectives are:

1. **Confidentiality:** one tenant, project, workload, or operator should not
   read another's artifacts, secrets, tokens, logs, or spend data without an
   authorized scope.
2. **Integrity and authority:** only an authenticated, authorized actor or a
   current fenced controller/agent may mutate workload, lease, policy, artifact,
   and capacity state; stale commands must not win races.
3. **Availability and safe recovery:** failures should retry or remain unknown,
   not fabricate success; a stuck or compromised component must not strand all
   unrelated work or cause unbounded destructive action.
4. **Accountability and release integrity:** security-relevant denials and
   policy/grant changes are auditable where enabled, and published artifacts are
   reproducible or hash-verifiable to the extent the release process claims.

## Threats and current controls

| Threat | Current control and evidence | Remaining gap / qualification limit |
| --- | --- | --- |
| Forged, expired, or cross-scope API identity | OIDC issuer, audience, issuer, expiry, and algorithm checks; bootstrap identity is a fixed fleet administrator; authorization matrix and authentication-audit tests | Identity-provider compromise and deployment rate limiting remain operator concerns; independent review pending |
| Browser token theft, login CSRF, or session replay | Server-side PKCE/state/nonce flow, single-use transactions, HttpOnly/Secure cookie, origin check, bounded session store, CSP and security-header tests | Provider-response and real-browser verification remain pending; sessions are in-process and need affinity or an external store across replicas |
| Network interception of bearer, bootstrap, or capability tokens | The control-plane listener refuses non-loopback plaintext by default and requires TLS, Postgres, metrics token, and artifact signing key; TLS startup and manifest tests | Native-worker explicit mTLS and controller-client wiring are locally tested, but default internal listener enforcement and bootstrap certificate handoff remain incomplete; explicit insecure control-plane mode still exists |
| Cross-tenant artifact or secret access | Scoped artifact keys, lifecycle/generation fencing, exact capability claim sets, distinct token audiences, live workload authorization, tenant-isolation and capability tests | Live object-store migration/destructive collection evidence and broader deployment qualification remain pending |
| Replay or stale capacity/workload mutation | Durable outbox, idempotency and generation/fencing fields, aggregate ordering, stale-command recovery, scheduler transaction revalidation, controller tests | Multi-host provider and production failover evidence are not implied by local tests |
| Malicious or accidental workload escape | Workloads are treated as untrusted; worker credentials are not placed in the workload environment; leases and cell cleanup bound ownership | Host/container isolation and provider IAM are deployment responsibilities; no claim of sandboxing arbitrary code |
| Secret leakage through logs, diagnostics, or audit | Secret-bearing output is suppressed; authentication audit omits credentials, cookies, query strings, raw identifiers, and unverified claims; redaction tests exist | Retention, log access, IdP-side audit, and browser callback audit remain deployment/provider responsibilities |
| Metrics or dashboard data exposure | Network binds require a metrics token; dashboard CSP and authentication controls protect operator views | Deployment network policy and rate limiting are still required; metrics token management is operator-owned |
| Dependency, source, or release substitution | Hash-pinned exports, lock consistency checks, detect-secrets baseline, dependency audit, SBOM and pinned provenance workflow inputs | No local/public provenance attestation has been issued here; external review and real release verification are pending |
| Split-brain federation or unsafe failover | Peer registration and budget/quota grants exist; shared-fleet ownership and automatic failover are not implemented | External PKI, quorum-backed witnessed ownership, reconciliation, and failover rehearsal remain open |
| Denial of service through login, outbox, or provider failure | Dashboard stores cap admissions rather than evicting live sessions; outbox retries/backoff and per-aggregate ordering; provider errors remain distinct from agent unavailability | Edge rate limiting, capacity planning, and production SLO evidence remain pending |

## Security evidence boundaries

The repository contains focused tests for authentication and authorization,
CSRF/session behavior, TLS startup and probes, capability tokens, artifact tenant
isolation and lifecycle, fencing, audit redaction, dependency policy, and
release-manifest rules. `CONTRIBUTING.md` defines the Python, dashboard,
PostgreSQL, lock, format, and package checks. Those checks establish repository
behavior only. They do not establish a secure cloud account, a trusted OIDC
provider, a production deployment, hardware isolation, external penetration
testing, or an independent security review.

Known roadmap gaps include internal-edge HTTPS/mTLS defaults, independent
security review, real release provenance issuance, provider-response/browser
verification, live object-store qualification, and a production pilot. See
[docs/security.md](../security.md), [docs/roadmap.md](../roadmap.md), and the
four earlier design records for the current implementation status. In
particular, this record must not be read as changing those statuses.

## Per-release security review checklist

The maintainer should record each item as **pass**, **fail**, **not applicable**,
or **pending**, with a link to evidence. A pending item is not a pass.

### Scope and code review

- [ ] Review the full diff for authentication, authorization, tenant scope,
      token, secret, TLS, logging, persistence, provider, and deployment changes.
- [ ] Re-read this threat model and update boundaries, assets, threats, and gaps
      for any changed trust assumption.
- [ ] Confirm public wire/persisted behavior changes are flagged or migrated,
      and that rollback does not widen authority or delete by inference.
- [ ] Obtain human review of security-sensitive changes. Assignment of an
      independent subsystem reviewer is currently **pending**.

### Automated evidence

- [ ] Run focused security and authorization tests plus the full applicable
      Python suite; run the dashboard tests and type-checking build when the
      dashboard is affected.
- [ ] Run docs grounding/link checks, ruff checks/format checks for changed
      Python, and dashboard lint/format checks when applicable.
- [ ] Run the real PostgreSQL path for persistence or transaction changes; do
      not substitute SQLite evidence for deployed-backend evidence.
- [ ] Run lock/export consistency, hash-pinned install/package smoke checks,
      secret scanning, dependency audit, SBOM generation, and manifest checks.

### Deployment and release integrity

- [ ] Verify non-loopback startup still requires TLS, shared durable state,
      metrics authentication, and stable artifact-token signing material.
- [ ] Verify secrets are supplied through the documented operator boundary,
      are absent from artifacts/logs, and have a rotation or rollback plan.
- [ ] Verify agent/worker transport mode and rollout order; record explicitly
      if mTLS remains opt-in.
- [ ] Inspect wheel, sdist, OCI inputs, SBOM, hashes, and generated references;
      publish only selected intended files.
- [ ] Verify release provenance in the actual publishing environment before
      claiming attestation. Local workflow tests are not issuance evidence.
- [ ] Record source identity, artifact digests, test results, exceptions,
      external-review status, and production-qualification status in release
      notes. Human sign-off and independent external review are **pending**
      unless separately recorded.

## Decision and maintenance rule

This record is maintained with the implementation. A threat or control may be
marked implemented only when source behavior and a reproducible test or
operational receipt support it. A design decision can constrain future work
without proving that work exists. The maintainer owns the current record;
additional human reviewers and external security review remain pending.
