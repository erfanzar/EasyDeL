# Part 7 — Providers: writing a capacity backend

A provider is **one driver, one capability entry, and a transport** — with no edit
to the scheduler, the planner, or the durable model. `aws-ec2` is a built-in
example of that contract. Contract doc: `docs/providers.md`; implementation
reference: `docs/reference/capacity-and-providers.md`.

## The shape

```text
eyvan_acme_provider/
├── __init__.py        # driver + capability entry + transport registration
└── pyproject.toml     # entry point registering the package with eyvan
```

The driver implements the `CapacityDriver` protocol (`eyvan/capacity/drivers/base.py`):
`capabilities()`, `observe()`, `reconcile()`, `apply_capacity()`, `drain()`,
`release()` — every mutation takes a `MutationContext(idempotency_key, generation,
fencing_token)`. Execution (submit/status/stop/checkpoint/logs) is the separate
`ExecutionDriver` axis.

Out-of-tree providers are opt-in through the `eyvan.capacity_providers` entry
point and `EYVAN_CAPACITY_PROVIDER_PLUGINS`. Plugin ABI is `1.0` and strict: a
major mismatch or newer minor is refused; import, contract, ABI, and name
conflicts are reported rather than swallowed.

## Built-in drivers

From `drivers/registry.py` (lazy imports — selecting one backend never imports
another's dependencies):

| Registered names | Driver | Cells are … |
|---|---|---|
| `gcp` | GCP router | TPU queued resources, Compute VMs, or attached TPUs, selected by pool backend |
| `gcp-tpu` | GCP TPU queued resources | queued resource per cell (multislice capable) |
| `gcp-compute` | GCP Compute Engine | one VM |
| `gcp-attach` | attached GCP TPU | existing TPU slice; never provisioned or deleted by the driver |
| `kubernetes`, `k8s` | Kubernetes | one Indexed Job; Kueue is the capacity-admission boundary |
| `aws-ec2`, `aws` | AWS EC2 | one instance |
| `azure-vm`, `azure` | Azure VM | one VM |
| `local` | local development | a logical, unisolated record on the agent host |
| `manual` | operator-declared hosts | adopted, never provisioned/deleted |
| `onprem-ssh` | SSH-claimed hosts | a fenced POSIX marker on the host |

## GCP Compute networking

New `gcp-compute` configurations default to a public IP
(`provider_options.internal_ip_only: false`). This permits direct public-internet
bootstrap; it does not bypass Google networking policies or Eyvan's authenticated
worker ingress checks. Existing configurations explicitly recording
`internal_ip_only: true` remain private.

For private-only instances, explicitly set `internal_ip_only: true` and select
`bootstrap_egress: cloud-nat` (with NAT configured on the subnet), or
`bootstrap_egress: private-google-access` (with the required worker dependencies
staged). Fleet validation rejects an explicitly private address combined with
`bootstrap_egress: public-internet` before applying the fleet. Spot versus
on-demand capacity selection is independent of these networking settings.

## Capabilities are declarative

`BackendCapabilities` (data, in `drivers/base.py`) declares whether a backend can
build a **multi-worker cell** and its maximum width. Current multi-worker backends:
GCP TPU queued resources (256), Kubernetes indexed Jobs (512), manual hosts (512),
on-prem SSH (512). Declaring the capability is what makes multi-worker atomic
shapes schedulable on your cloud; not declaring it means every cell you build is
one machine — the safe reading of a silent driver. These limits are provider
ceilings, not hardware qualification.

## The durability rules every driver must keep

All drivers implement observe/apply/release/reconcile through the same durable
cell and mutation context:

- A stale generation or fencing token must **not** mutate anything.
- Replayed creates are idempotent (the same client token / idempotency key).
- Provider state belonging to another incarnation is a **leak**: recreate it, never
  adopt it.
- An unreachable provider is **unknown/inconclusive**, never "absent" — ambiguity
  is surfaced, never converted into a prettier verdict.
- Deletion is best-effort-and-reported: an uncertain delete result is retryable
  state, not success.

Concrete examples from the built-ins:

- **GCP TPU** uses a provider management UID + generation; Compute uses a VM
  identity + startup config. TPU placement distinguishes stockout, quota,
  queue-admission, preemption, and provider-unavailable failures.
- **Kubernetes** encodes Eyvan generation labels and a fencing fingerprint
  annotation on Jobs (durable fencing is not substituted by `resourceVersion`).
  Eyvan delegates *capacity* admission to Kueue but keeps workload admission and
  preemption policy; the ClusterQueue must be exclusively Eyvan's with preemption
  disabled. Unreadable cluster capacity is UNKNOWN, never full.
- **AWS EC2** launches one instance per cell with ownership in
  cell/generation/fence tags; the same client token makes launch replay idempotent;
  spot reclamation maps to `PREEMPTED`, not a generic loss. The hand-rolled SigV4
  signer and Query API live in `capacity/providers/aws/`. It currently requires an
  explicit region + access keys (instance roles are not read by the registry
  factory).
- **Azure VM** polls ARM long-running operations; ownership fenced in the driver.
- **On-prem SSH** claims hosts with a marker containing cell, management UID,
  provider generation, and fence; a foreign/old marker yields `NEEDS_RECREATE`; an
  unreachable host is `UNOBSERVED`, not gone.

## The conformance suite

`capacity/conformance.py` is a provider-ABI suite, deliberately harsher than unit
tests. It verifies: routable capabilities; unknown-cell `ABSENT`; stale-fence
non-mutation; exactly-once create under replay; leaked-incarnation detection
without adoption; and the difference between inconclusive and conclusive provider
failures. Implement `ProviderConformanceHarness` and run
`CapacityProviderConformanceTests` / `run_conformance`.

Passing conformance proves the contract against the harness — it does **not**
prove a real cloud account, network, image, quota, or accelerator works.
Qualification = conformance green **plus** a recorded spend on real infrastructure
(an immutable exact-candidate receipt; see `docs/proofs.md`).

## The two-directional guarantee

Eyvan guarantees you:

- Nothing that decides (scheduler, planner, durable model) names or special-cases
  your provider — enforced by `tests/test_provider_boundary.py`.
- Your driver decides machine-level truth and the core trusts your answers,
  including "unavailable/inconclusive".

You guarantee back:

- Every machine you create is trackable to a cell: if you created it, the plane
  can name and reap it. Ownership bugs are the worst kind; the conformance suite
  exists to catch them.
- Refusals carry their reason — your diagnostic renders verbatim under
  `blocked … the plane is retrying`.
- Deletion uncertainty is retryable state, never success.

## Turning a provider on

Install the provider package into the **cell agent's environment**. Allowlist it
with `EYVAN_CAPACITY_PROVIDER_PLUGINS` there, then select its backend with the
agent's `EYVAN_CAPACITY_BACKEND`. Point the fleet's pools at that agent and name
the matching capacity backend in the fleet configuration. The agent constructs
the driver; installing the package only in the plane does not enable it.
`eyvan capacity list` shows the pools available to the project. Qualify the
installed provider version before relying on it.
