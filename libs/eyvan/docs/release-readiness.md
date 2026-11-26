# Release readiness checklist

Use this checklist for an immutable release candidate. It is not a release
approval or a record of checks run. Record results, skips, exceptions and human
sign-off with the candidate's release evidence, not as a rolling session log.

## Repository gates

- [ ] Run the Python matrix supported by `pyproject.toml` and CI (3.11–3.13).
      Build the dashboard before the Python suite; run the full suite with a
      disposable PostgreSQL database and `EYVAN_REQUIRE_POSTGRES=1`.
- [ ] Run Ruff lint/format, mypy, import contracts, lock/export consistency,
      generated API/environment reference checks and license-header checks.
- [ ] Run dashboard tests, the type-checking production build, lint, formatting
      and the applicable browser flows.
- [ ] Run secret scanning and dependency audits with the pinned CI inputs.
      A past clean scan is not a current advisory result.
- [ ] Confirm documentation commands, versions and deployment examples agree
      with the packaged source. Review the changelog and known limitations.

See [CONTRIBUTING.md](../CONTRIBUTING.md) and
[testing policy](testing-policy.md) for commands and evidence boundaries.

## Package and release integrity

- [ ] Check package, dashboard and chart version metadata and the release tag
      against the publishing workflow's requirements.
- [ ] Build wheel and sdist with the pinned backend and hash-pinned build
      requirements; inspect contents and smoke-test an isolated installation.
- [ ] Verify the intended SBOM, hashes, provenance and signatures from the
      actual release workflow. Workflow configuration is not issued provenance.
- [ ] Qualify exact OCI digests and required platforms; do not infer target
      cluster pullability or accelerator compatibility from a local build.

## Deployment and recovery

- [ ] Review TLS, identity, metrics authentication, stable capability-signing
      material, provider permissions and secret delivery for the target.
- [ ] Rehearse database maintenance with quiesced writers, matching
      database/object-store backups and a restore plan. Do not use downgrade
      or an automatic image rollback as a schema rollback strategy.
- [ ] Rehearse upgrade and recovery using installed artifacts. Distinguish a
      development baseline from an actual published-release upgrade.
- [ ] Qualify artifact inventory and migration before enabling destructive
      collection; see [artifact safety](artifact-safety.md).
- [ ] Run only the authorized provider/hardware proofs applicable to the
      advertised support, against the exact candidate. Retain cleanup evidence
      and unmet dimensions; see [proofs](proofs.md).

## Human approval and scope

- [ ] Complete the [security review checklist](design/0005-threat-model.md)
      and record maintainer approval and independent-review status.
- [ ] Record outstanding [roadmap](roadmap.md) items, operational limits and
      production-qualification status. Local tests do not establish a hosted
      CI result, secure deployment, hardware result or production SLO.
- [ ] Obtain explicit release-owner authorization before tagging or publishing.
