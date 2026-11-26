# Eyvan immutable OCI policy

This directory is the checked-in, fail-closed input to
`.github/workflows/eyvan-oci-release.yaml`. It contains no registry credential,
environment-specific OIDC issuer/client, release digest, or exception hidden in
prose.

## Canonical policy files

`release-lock.json` and `high-critical-allowlist.json` are strict canonical
JSON: UTF-8, lexicographically sorted object keys, compact separators, and one
trailing newline. Duplicate keys, unknown fields, non-canonical bytes, moving
image/tool references, placeholder action SHAs, or a Dockerfile that diverges
from the lock fail `validate-policy`.

The release lock makes four explicit supply-chain decisions:

* the only built platform is `linux/amd64`; other named platforms remain
  unverified;
* every base stage is passed as a required `fixed-tag@sha256:<amd64-manifest>`
  build argument;
* each downloaded host tool has a fixed version URL, exact byte size, and
  SHA-256; streaming stops at the locked size and checks `Content-Length` when
  present, and every used GitHub Action is addressed by a full commit SHA;
* control plane, cell agent, and native worker have distinct repositories,
  entrypoints, labels, and Helm digest values.

The base pins were independently resolved as both multi-platform index and
`linux/amd64` leaf manifests on 2026-08-08. `baseImages` records both digests;
each Docker build reference intentionally uses the leaf manifest. Re-resolving
a source tag is evidence for a reviewed lock update; it must never silently
change a release build.

Run from the repository root:

```bash
python3 scripts/verify_eyvan_oci_release.py validate-policy \
  --lock oci/release-lock.json \
  --allowlist oci/high-critical-allowlist.json \
  --repository-root .
```

## High/Critical exceptions

The allowlist starts empty. The scan gate fails on every unmatched High or
Critical Trivy finding, including findings with no fixed version. An exception
is an exact tuple of:

* image `role`;
* CVE/GHSA `vulnerability`;
* package name, exact PURL, and installed `version`;
* accountable `owner`, concrete HTTPS `issue`, specific `reason`; and
* second-precision UTC `expiresAt`, later than gate time and no more than 30
  days away.

Wildcards, placeholders, stale/long expirations, duplicate entries, near
matches, and unused entries for the scanned role fail. Entries remain sorted by
role, vulnerability, PURL, and version. An exception is not proof of safety; it
is a short, reviewable release decision.

## Verifier surfaces

The verifier is standard-library-only and exposes:

* `validate-policy` for the lock, allowlist, and Dockerfiles;
* `capture-scanner-metadata` and `capture-db-metadata` for exact tool/DB
  evidence;
* `gate-scan` for raw Trivy JSON and a canonical zero-unmatched disposition
  whose `evaluatedAt` is the exact second-precision policy decision clock used
  for receipt replay;
* `verify-image-metadata` for `linux/amd64`, entrypoint, labels, and OIDC
  neutrality;
* `verify-rendered-digest-refs` for the exact three-way Helm digest split;
* `extract-rekor` for one inclusion-proved entry in a Cosign bundle; and
* `emit-receipt` / `verify-receipt` for the closed receipt-v1 schema and all
  referenced artifact hashes.

`download_release_assets.py` first permits only the fixed receipt and detached
bundle (1 MiB/4 MiB caps). After their signature and closed schema are verified
with logic checked out from the protected workflow commit, it accepts exactly
34 unique receipt-named assets. GitHub release metadata must contain exactly 36
assets, each no larger than 256 MiB and no more than 1 GiB in aggregate; reported
and streamed byte counts plus every signed hash are checked. No wildcard release
download is used.

The receipt binds full source SHA/ref/version and environment-gated workflow
run, three role/index/platform digests, locked bases and exact labels, SPDX and
SBOM attestation bundles, raw scans/tool/DB/allowlist/dispositions, Cosign
issuer/workflow/bundle/Rekor facts, signed SLSA-v1 predicates and materials,
wheel/sdist, chart/overlay/render and deployment references. No arbitrary
`diagnostics` object is accepted. It explicitly says no deployment or hardware
proof was run and names every unverified platform.

The signed receipt cannot recursively hash its own signature bundle. The
workflow therefore signs the canonical receipt as a blob, strictly verifies it
against the same issuer and exact workflow identity, then promotes the already
qualified index digests to immutable version tags without rebuilding or adding a
`latest` tag. The PyPI workflow is the standalone repository's `publish.yaml`:
tag-driven on `v*` only, it refuses a tag that disagrees with the version
declared in `pyproject.toml`, builds with the hash-pinned backend, and publishes
from the reviewer-gated `pypi` environment with OIDC trusted publishing and
`persist-credentials: false` -- no PyPI token exists in this repository.

## Consume a release

A version tag is only a locator, not release evidence. Treat the signed receipt
and its detached Cosign bundle as the trust root: download the fixed, named
release assets, verify the canonical receipt and its bundle with the locked
workflow identity and OIDC issuer, and verify Rekor inclusion. Then verify the
receipt's three image index/platform digests, package/chart/render digests, and
attestation hashes before copying the exact `repository@sha256:...` references
to deployment configuration. For example, the verifier step is:

```bash
python3 scripts/verify_eyvan_oci_release.py verify-receipt \
  --lock oci/release-lock.json \
  --receipt evidence/eyvan-oci-release-receipt-v1.json
cosign verify-blob --new-bundle-format \
  --bundle evidence/eyvan-oci-release-receipt-v1.bundle.json \
  --certificate-identity \
  'https://github.com/erfanzar/eyvan/.github/workflows/eyvan-oci-release.yaml@refs/heads/main' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  evidence/eyvan-oci-release-receipt-v1.json
```

Use `verify-rendered-digest-refs` against the target Helm render and verify each
image signature against the same identity; do not rebuild, resolve a tag during
installation, or substitute `latest`. If any receipt, signature, Rekor,
attestation, or digest check fails, stop rather than treating the artifact as a
release candidate. The receipt's `deploymentApplied=false` and hardware
`not-run` fields are intentional: deployment and accelerator qualification
remain operator checks in the exact target environment.

## Release governance prerequisites

The release job is intentionally unusable until an operator configures all of
these repository controls outside the workflow:

* `main` is protected by branch protection or a qualifying ruleset;
* the `eyvan-oci-release` environment requires an explicit reviewer approval
  and restricts deployment branches to protected `main` only;
* immutable GitHub releases are enabled; and
* that environment exposes `EYVAN_GITHUB_GOVERNANCE_TOKEN`, a fine-grained,
  read-only token with repository Administration read access, solely so the
  workflow can call the immutable-releases status endpoint.

Before executing candidate verifier scripts or writing a registry, the workflow
queries the branch, immutable-release, and run-approval APIs with bounded reads
and fails closed unless all three controls are proved. It never enables or edits
those controls.

## Qualification boundary

Pull-request qualification is static/hosted and read-only. Release
qualification is a GitHub-hosted `linux/amd64` build plus registry scan,
attestation, and digest/render verification. Neither lane applies a deployment
or exercises CPU/GPU/TPU workload hardware. A canonical receipt with
`deploymentApplied=false`, hardware `not-run`, and non-amd64 platforms listed as
unverified is accurate evidence, not a readiness claim for a live cluster.
