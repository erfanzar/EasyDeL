# DR-0004: Federation consistency, trust, and witnessed ownership

**Status:** Accepted architecture; implementation pending. No external PKI
or ownership-witness provider is qualified, so secure federation and
automatic failover remain disabled.

## Context

One plane serves one project well and serves it honestly. Two planes
serving the same fleets — for locality, burst, or surviving a regional
failure — introduce the hard problem: two authorities that can each say "I
own this cell" with no third party to arbitrate. Split-brain capacity is
not an availability risk; it is a billing event and a data-integrity event
at once.

## Decision

1. **Eyvan builds no CA and no consensus protocol.** Trust comes from an
   external PKI the planes already trust, and ownership arbitration comes
   from an external, quorum-backed lease/CAS service. The plane implements
   the *client* side of both, never the authority side.
2. **Ownership is witnessed, not asserted.** A federated claim on a fleet or
   cell is valid only with a witness record from the external service —
   a lease with an epoch, renewable, and revocable by the quorum. A plane
   that cannot reach the witness cannot claim; it can only release.
3. **Consistency follows the witness.** A plane reconciles its durable state
   against witness epochs at the boundary: a lost epoch means foreign state
   is read-only until re-witnessed, never merged, never overwritten.
4. **Failover is manual until it is boring.** Automatic failover stays
   disabled until a qualified witness provider plus evidence of clean
   failover under partition exists. Until then, failover is an operator
   action with the same receipts discipline as everything else.

## Invariants

- No plane may create or lease provider resources in a fleet whose witness
  epoch it does not hold.
- A partitioned plane must degrade to read-only for foreign-scoped state
  rather than act on stale ownership.
- Revocation propagates at most one lease interval; resources created under
  a revoked lease are reaped by the witness holder, and the plane treats
  reap notices as authoritative.

## Open

Everything implementation-shaped: the PKI integration surface, the witness
client, epoch reconciliation, and the failover runbook with its rehearsal
matrix. Until then the honest statement — the one this package ships — is:
one plane per fleet boundary, and if you need two, you need the external
witness first.
