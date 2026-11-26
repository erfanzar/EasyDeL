# DR-0002: Flattened global log ranks and opaque cursors

**Status:** Global ranks, typed cursor outcomes, topology-bound cursor decoding
and schema-v2 terminal archive manifests are implemented. Bound v2 cursor
emission is opt-in and defaults off. Multi-host deployment qualification
remains separate from host tests.

## Ordering and identity

Log writers share a flattened global rank space. The attempt topology is frozen
at dispatch so later topology changes do not renumber history. Merged records
are ordered by timestamp, then rank (`eyvan/worker/runtime/log_records.py`);
rank is a deterministic tie-breaker, not a causal timeline. Clock skew can
change apparent cross-host order.

## Cursor contract

Cursors are opaque, untrusted positions, not credentials. The decoder validates
structure and reports typed invalid/incompatible outcomes. A bound cursor for
a different workload, topology, kind or mode is rejected. Topology checks need
a bound cursor; legacy cursor compatibility is not full topology binding.

The controller reports `attempt_changed` and restarts a new attempt at empty
positions. A legacy cursor without a recoverable rank witness may also reset
when topology is unknown; it does not gain a fabricated topology.

`EYVAN_LOG_CURSOR_V2_EMISSION_ENABLED` enables fully bound v2 emission. Decode
support must reach all serving and rollback images before enabling emission.
See `decode_log_cursor()` and the controller's `_decoded_cursor()` in
`eyvan/server/controller/observation.py`.

## Terminal archives

Settlement writes an immutable schema-v2 manifest last for each terminal
attempt. It records topology, per-rank outcomes, limits, gaps and completeness.
Archive replay validates the manifest and each object's recorded digest and
size. An unreadable manifest or mismatching object remains an explicit gap;
it must not silently fall back to a less complete view.

## Qualification still required

Exercise concurrent writers, clock skew, missing ranks, retries, archive
retention and decode-before-emit upgrades on the target multi-host deployment.
Local tests establish protocol behavior, not hardware or production timing.
