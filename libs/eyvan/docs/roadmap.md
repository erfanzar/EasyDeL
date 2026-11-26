# Roadmap

Status: **[done]** implemented repository capability · **[next]** remaining
implementation or qualification · **[ongoing]** a continuing practice.
Implemented does not mean production-qualified. Tests, hosted CI, release
provenance, deployment rehearsals and hardware proofs are separate evidence.

## Architecture

- **[done]** A stable `ControlPlaneService` facade composes private transactional
  subsystems. Import-linter enforces subsystem and layer boundaries; the
  [architecture tour](reference/control-plane-architecture.md) maps entry points.
- **[next]** Preserve public imports, transaction boundaries and the module-size
  guard as new capabilities are added.

## Correctness

- **[done]** Scheduler properties cover whole-cell allocation, worker
  conservation, contention, deterministic placement and affordable alternatives.
- **[done]** Bounded service/repository histories cover terminal and orphan
  settlement, retries, provider/capacity loss, budget accounting and fenced replay.
- **[done]** Controller tests cover failed delivery, admission-response loss,
  expired-claim takeover, renewal refusal and durable priority-preemption phase
  recovery. Agent doubles and explicit fault injection are not deployed failover.
- **[next]** Extend generated histories to concurrent failover and renewal
  failures; establish conservation across broader combinations of transitions.
- **[next]** Verify the seeded nightly invariant lane on hosted runners and
  retain replay inputs. Deterministic chaos cases are not randomized histories.
- **[next]** Complete [domain mutation qualification](reference/scheduler-mutations.md).
  The aggregate at `4c2d8d5` is a historical baseline: 1,582 of 1,959 generated
  mutants detected, 377 surviving. New domain code, survivor classification and
  skipped dataclass/property bodies require fresh qualification. Tool exit 0
  with survivors is not a passing mutation gate.

## Testing and developer experience

- **[done]** CI defines Python 3.11–3.13, coverage ratchets, Ruff, mypy, import
  contracts, secret/dependency checks and dashboard test/build/lint/format gates.
- **[done]** An explicit `fast` subset supports local iteration without replacing
  the full suite. [Quarantine](testing-policy.md) tracks investigation and never
  suppresses a failing test.
- **[done]** `eyvan demo` runs a disposable local callable; doctor preflights and
  provider diagnostics expose configuration failures before expensive work.
- **[done]** The versioned provider extension ABI validates explicitly trusted
  plugins, compatibility and backend-name conflicts.
- **[next]** Qualify the disposable Compose PostgreSQL service at runtime and
  retain full required-PostgreSQL evidence separately from memory/SQLite tests.
- **[next]** Verify nightly full-suite, extended repository-soak and scheduler
  lanes on their actual hosted runners. Configuration tests alone do not prove
  execution.

## Security

- **[done]** Network-bound control-plane startup requires TLS or an explicit
  plaintext exception. Deployment probes follow the listener protocol; secrets
  and database credentials are operator-supplied rather than static defaults.
- **[done]** Dashboard OIDC uses server-side PKCE and cookie sessions with origin
  checks and bounded admission. Browser and session regressions cover the local
  boundary; target IdP, certificates and replica routing still need qualification.
- **[done]** The [threat model](design/0005-threat-model.md) records controls,
  unresolved boundaries and a per-release review checklist.
- **[next]** Complete internal-edge HTTPS/mTLS defaults, stable endpoint/SAN and
  multi-host identity, issuance and rotation. The optional TLS bootstrap helper
  is preparatory, not a complete fleet rollout.
- **[next]** Obtain maintainer threat-checklist approval and independent external
  security review. Agent review does not supply either human approval.
- **[next]** Qualify shared-fleet federation ownership and automatic failover;
  peer registration and budget/quota grants do not provide witnessed ownership.

## Documentation and release integrity

- **[done]** Generated REST/OpenAPI references have deterministic regeneration
  and stale-output checks. Operator guides cover provider permissions and
  deployment boundaries.
- **[done]** Publishing configuration includes hash-pinned inputs, SBOM generation
  and provenance steps; OCI qualification addresses immutable artifacts.
- **[next]** Verify provenance issuance, signatures, published artifacts and
  target-platform coverage in an authorized real release. Local policy checks
  are not a release receipt.
- **[next]** Rehearse migration, backup/restore and recovery on the intended
  deployment before enabling destructive artifact collection. See
  [artifact safety](artifact-safety.md) and the
  [release checklist](release-readiness.md).

## Observability

- **[done]** Monitoring rules, opt-in credential-free authentication audit,
  correlation IDs and tracing hooks are implemented.
- **[next]** Qualify deployed correlation and OpenTelemetry export through the
  controller, agent and native worker. Durable-queue trace continuity and a
  deployed collector remain separate acceptance criteria; see
  [correlation](reference/correlation.md).
- **[next]** Add controller-progress readiness and define an external
  autoscaling-signal contract. Current readiness proves repository reachability,
  not loop progress.

## Performance and availability

- **[done]** [Scheduler benchmarks](reference/scheduler-benchmarks.md) define
  deterministic scenarios, raw samples and unchanged regression budgets.
- **[next]** Verify the hosted 10k-cell lane and measure full controller/database
  reconciliation latency separately from pure scheduler timings.
- **[done]** The [HA guide](reference/high-availability.md) documents shared-primary
  advisory leadership, outbox claims and API-only replicas.
- **[next]** Qualify database-primary promotion, target proxy/session behavior
  and deployed failover. Shared-primary tests do not establish production SLAs.

## Hardware and experiments

- **[done]** A reusable dry-run-first Spot priority-checkpoint example and
  portable payload are documented in [tutorial 17](tutorial/17-spot-priority-demo.md).
- **[next]** Reproduce checkpoint/yield/restore on the intended candidate with
  exact device and cleanup evidence. Distinguish same-host priority restoration,
  injected replacement-host recovery and natural Spot reclamation.
- **[next]** Qualify natural reclamation and longer-running training under the
  [proof criteria](proofs.md). No local test substitutes for a hardware result.
- **[next]** Run Slurm / Volcano / kube-batch comparisons using the
  [comparison protocol](design/0006-receipts-and-elastic-packing.md). Preserve
  semantic equivalence, candidate identities, raw distributions and failures;
  no performance superiority is inferred from the design.

## Governance and maturity

- **[done]** Security policy, Code of Conduct, contribution templates, this
  roadmap and [governance document](../GOVERNANCE.md) exist.
- **[next]** Obtain approval of the proposed governance process, appoint a second
  maintainer and subsystem owners, establish a release cadence and groom
  contributor issues. Documents alone do not establish those practices.
- **[ongoing]** Keep changelog and compatibility notes aligned with reviewed
  releases.
- **[next]** Qualify upgrades from an actual published release, separately from
  the installed-wheel development-baseline test.
- **[next]** Complete a 4–6 week production pilot with published SLOs, operational
  review and human sign-off. Software checks alone do not close this item.
