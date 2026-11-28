# Test gates and flaky-test quarantine

## No silent passes

The complete Python suite remains blocking on pull requests and on the nightly
schedule. A flaky failure is still a failure: automatic retries must not turn a
failed PR into a green one, and `skip`/`xfail` is not a quarantine mechanism.
The fast smoke subset described in CONTRIBUTING.md is for local iteration only.

## Quarantine is a tracked investigation queue

A maintainer may label a reproducibly flaky test with
`pytest.mark.quarantine(issue=..., owner=..., until=...)`. Every marker requires:

- A GitHub issue URL documenting the failing command, seed where applicable,
  environment, failure output, and known reproduction conditions.
- A named owner responsible for fixing or removing the cause.
- An ISO date in UTC, no more than 14 days ahead. Expired, malformed, or missing
  metadata fails collection rather than making the test disappear.

The collection hook in `tests/conftest.py` enforces the metadata and **does not
add skip or xfail**. Marked tests remain in the normal full-suite gate. Nightly
CI also selects `-m quarantine --strict-markers` for up to three separate runs,
which helps distinguish order dependence from isolated failure. Any failing
run fails the job; exit code 5 is accepted only because an empty investigation
queue is normal. Cancellation stops this diagnostic lane.

There is no permission here to quarantine an unfixed security, tenant-isolation,
data-loss, or ownership-fencing bug merely to make a release green. A release
with a failing test remains blocked. An exception requires an explicit release
policy change and review, not a marker added to a test.

Close the tracking issue and remove the marker only after recording the fix's
regression test and successful repeated verification. Extending an expiry
requires a reviewed explanation in the issue; the date must not be refreshed
automatically.

## Nightly repository soak

`tests/test_repository_soak.py` retains one repository while repeatedly
submitting, replaying, reconstructing the service, cancelling, and checking
invariants. Ordinary suite runs exercise ten cycles per backend. The nightly
lane uses `--soak-iterations=1000` and requires PostgreSQL; memory and SQLite
remain included as separate backend cases.

This verifies repository/service behavior under growing history. It does not
prove provider behavior, accelerator execution, leadership failover, or a
multi-day production soak. Those need separate evidence. Benchmark regression
budgets and hardware receipts are tracked separately in docs/roadmap.md.

## CI evidence limits

Local workflow-contract tests validate configuration, action pins and gate
structure. They do not establish that a GitHub-hosted job ran. Release
provenance requires the publishing workflow's GitHub OIDC identity: local tests
cannot issue that attestation, and a configured step is not an issued receipt.
