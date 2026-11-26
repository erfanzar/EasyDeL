# Eyvan governance

## Status and scope

This document records existing maintainership and proposes a review and release
process for maintainer approval. It is not evidence that those new processes
have been approved, or that a second maintainer, an external reviewer, a
production pilot, or a public release exists. Human assignments and approval
of newly proposed process remain **pending**.

Eyvan is an AGPL-3.0-or-later project mirrored from an upstream monorepo. A
change in this repository may be applied upstream and mirrored back; contributors
should read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing substantial
work. The [Code of Conduct](CODE_OF_CONDUCT.md) applies to project spaces.

## Maintainership

- The current project maintainer is the person identified by the contact address
  in [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- The maintainer is responsible for the canonical project direction, repository
  administration, security response coordination, release decisions, and keeping
  this document truthful.
- There is currently no second maintainer or subsystem owner recorded here.
  Appointment of additional maintainers and per-subsystem ownership is **human
  assignment pending**; this document does not imply that either has happened.
- Contributors may propose changes, review code, report regressions, and raise
  security issues without being maintainers. Authority to merge, publish, or
  disclose a vulnerability remains with the current maintainer until a future
  written change to this policy says otherwise.

## Proposals and review

1. Small changes should include a focused description, tests or an explanation
   of why tests do not apply, and documentation updates when behavior or
   operator action changes.
2. Large changes should start with an issue or design record. Security,
   persistence, wire-format, provider, and release changes should identify
   trust-boundary and rollback implications. Design records state their real
   status; acceptance is not the same as implementation or qualification.
3. The author must not be the sole source of evidence for a security claim.
   Review should inspect the relevant implementation, tests, deployment
   surfaces, and failure behavior. A human reviewer assignment for each
   subsystem is **pending**; until then, the maintainer performs or explicitly
   records the review.
4. CI and repository checks are required evidence, not a substitute for human
   judgment. A green check does not establish cloud, hardware, production, or
   external security-review evidence.
5. Review comments and unresolved risks should be recorded in the change or its
   design record. Do not hide an uncertainty by rendering it as healthy, empty,
   or complete.

Security reports must follow [SECURITY.md](SECURITY.md), not a public issue.
The stated acknowledgment target is 72 hours; it is a target, not a guarantee.

## Release process

A release is a deliberate maintainer action, not an implication of a passing
local test run. Before publishing a version, the maintainer should complete the
checklist in [docs/design/0005-threat-model.md](docs/design/0005-threat-model.md)
and record exceptions and their evidence.

The release process is:

1. Confirm the intended version and changelog entry, and inspect the complete
   diff for accidental credentials, generated-file drift, and unreviewed
   security-sensitive behavior.
2. Run the applicable Python, dashboard, documentation, formatting, packaging,
   dependency-lock, and secret-scan checks. Persistence changes require the
   deployed backend path as described in `CONTRIBUTING.md`.
3. Confirm that `uv.lock` and hash-pinned requirement exports agree, and that
   the package and deployment manifests describe the same security defaults.
4. Build the wheel, sdist, OCI inputs, and SBOM through the repository's pinned
   release workflow. Verify artifact contents and hashes before publication.
5. Publish only after the maintainer has reviewed the evidence and any remaining
   risks. Record the source identity and artifact digests in the release notes.
6. After publication, verify the public artifacts and document any failed or
   unperformed verification. A release must not claim provenance attestation,
   production qualification, live-provider evidence, or an external review
   unless that evidence actually exists.

The repository currently records release-provenance issuance and a production
pilot as pending work; this policy does not promote either to completed status.
An independent external security review is also **pending** and is a release
readiness item rather than an implied approval gate that has already occurred.

## Changes to this policy

The current maintainer may propose governance changes in a reviewed change.
The change must explain which authority or process is altered and must not
retroactively claim approvals or evidence. Adding maintainers, assigning
subsystem owners, or delegating release authority requires an explicit human
record in a later change.
