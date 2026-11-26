# Capacity and providers — release reference

This is the implementation reference for Eyvan's capacity plane. It describes the
release code, not a promise that a cloud account or a particular accelerator has
been qualified. The primary sources are linked inline; user setup remains in
[`docs/providers.md`](../providers.md).

## 1. Architecture

A fleet is declarative configuration compiled into capacity pools. The control
plane plans cells, leases cells to workloads, and sends fenced mutations to a
cell agent. The agent constructs exactly one capacity **driver** and reconciles
one cell at a time. A driver owns the provider-side lifecycle (or records an
operator-owned host), while the scheduler, durable state, idempotency, fencing,
and lifecycle audit are backend-neutral. See [`fleet_config.py`](../../eyvan/capacity/fleet_config.py),
[`planning.py`](../../eyvan/capacity/planning.py), and the
[`CapacityDriver` contract](../../eyvan/capacity/drivers/base.py).

A **driver/backend** is the lifecycle adapter selected by `capacity_backend`.
A **provider client** is the narrower API implementation used by a driver (for
example EC2 Query, ARM, Cloud TPU/Compute Engine, or Kubernetes HTTP). Provider
clients do not replace the driver contract or durable fencing.

## 2. Drivers, capabilities, and registry

`BackendCapabilities` is data: it declares whether a backend can build a
multi-worker cell and its maximum width. Current multi-worker backends are GCP
TPU queued resources (256), Kubernetes indexed Jobs (512), manual hosts (512),
and on-prem SSH hosts (512). AWS EC2, Azure VM, GCP Compute Engine, and local
are one machine per cell today. These limits are implementation/provider
ceilings, not hardware qualification. [`base.py`](../../eyvan/capacity/drivers/base.py)
contains the canonical backend aliases, agent-family mapping, failure codes,
and capability table.

The registry imports driver/provider modules lazily: selecting one backend must
not import another backend's dependencies. Built-ins are registered in
[`registry.py`](../../eyvan/capacity/drivers/registry.py):

| Registered names | Driver and substrate | Notable behavior |
|---|---|---|
| `gcp`, `gcp-tpu` | GCP router / TPU queued-resource driver | TPU queued resources; native or configured execution |
| `gcp-compute` | GCP Compute Engine driver | One VM per cell; CPU/GPU/TPU machine shapes |
| `gcp-attach` | Existing GCP TPU slice | Observes an operator-owned slice; release does not delete it |
| `kubernetes`, `k8s` | Kubernetes driver | One Indexed Job per cell; Kueue admission only with `queue_name` configured |
| `aws-ec2`, `aws` | AWS EC2 driver | One instance per cell |
| `azure-vm`, `azure` | Azure VM driver | One VM per cell |
| `local` | Local development driver | Logical, unisolated cell on the agent host |
| `manual` | Fixed operator-declared hosts | Provisions/deletes nothing |
| `onprem-ssh` | SSH-claimed operator-owned hosts | Fenced remote marker; optional start/stop commands |

The registry normalizes case and underscores to hyphens. `tpu`, `compute` and
`ec2` are agent-family aliases in `base.py`, not registered backend names.
Execution names are a separate vocabulary: `local`/`local-process` canonicalize
to `native-local`, and
`native`, `native-remote`, `native-worker`, `eyvan` to `eyvan-native`.
Retired `ray`/`rayjobs` names canonicalize to `ray-jobs` and remain a refused
backend rather than an unknown typo. Out-of-tree providers are opt-in through
`eyvan.capacity_providers` and `EYVAN_CAPACITY_PROVIDER_PLUGINS`; ABI `1.0`
is strict (major mismatch or newer minor is refused), and import, contract,
ABI, and name conflicts are reported rather than swallowed.

## 3. Built-in driver distinctions

All drivers implement observe/apply/release/reconcile through the same durable
cell and mutation context. A stale generation or fencing token must not mutate;
replayed creates use idempotency; provider state belonging to another
incarnation is a leak and is recreated, not adopted; an unreachable provider is
unknown/inconclusive, not absent.

* **GCP.** The router shares one factory for TPU queued-resource, Compute and
  attached-TPU modes. TPU queued resources use a provider management UID and
  generation; Compute uses a VM identity and startup configuration.
  `gcp-attach` observes an existing operator-owned TPU slice: drain retains an
  unschedulable local claim, and release tombstones the claim. Neither operation
  deletes or stops the TPU or its workers. Native-worker endpoint, secret, artifact,
  service-account, ingress, bootstrap, and runtime options are persisted with the
  incarnation. TPU placement distinguishes stockout, quota, queue admission,
  preemption, and provider-unavailable failures.
* **Kubernetes.** The cell is an Indexed Job. Durable Eyvan fencing is not
  substituted by `resourceVersion`; Jobs carry generation labels and a fencing
  fingerprint annotation for leak detection. When `queue_name` is configured,
  Eyvan delegates *capacity* admission to Kueue but retains workload admission
  and preemption policy. That ClusterQueue must be exclusively Eyvan's with
  preemption disabled; without `queue_name`, this Kueue admission path is unused. Unreadable cluster capacity is UNKNOWN, never full.
* **AWS EC2.** `RunInstances` is one instance per cell, ownership is encoded in
  cell/generation/fence tags, and the same client token makes launch replay
  idempotent. Spot reclamation is `PREEMPTED`, not a generic loss. The hand-rolled
  SigV4 signer and Query API are in [`providers/aws`](../../eyvan/capacity/providers/aws/).
  The driver currently requires an explicit region and access-key/secret-key
  pair; instance-role credentials are not read by the registry factory.
* **Azure VM.** ARM requests are polled as long-running operations and VM
  ownership is fenced in the driver. Shapes and ARM transport are in
  [`providers/azure`](../../eyvan/capacity/providers/azure/). It is one VM per
  cell, with CPU/GPU accelerator vocabulary and on-demand/spot/reserved classes.
* **Local.** This development adapter materializes a logical record only. It
  makes no CPU, memory, accelerator, cgroup, or container-isolation claim.
* **Manual.** Existing hosts and native-worker endpoints are adopted by
  declaration; Eyvan does not start, stop, or delete them. The driver reports
  operator-declared shape and provides native-worker fencing/cgroups where the
  host runtime does so. Spot is deliberately not advertised.
* **On-prem SSH.** Hosts are claimed with a POSIX marker containing cell,
  management UID, provider generation, and fence. A foreign/old marker yields
  `NEEDS_RECREATE`; an unreachable host is `UNOBSERVED`, not gone. Eyvan owns
  the claim and marker, while the operator owns software installation and
  normally worker startup.

## 4. Provider clients

The provider packages are transport and provider-domain seams, intentionally
smaller than drivers:

* AWS signing (`signing.py`) produces SigV4 requests; `api.py` and `instances.py`
  validate and model EC2 launch/describe/terminate behavior.
* Azure `api.py` performs ARM requests and LRO polling; `machines.py` carries
  Azure VM-shape data.
* GCP queued resources implement TPU create/observe/delete, TPU type parsing and
  zone policy; Compute Engine implements VM operations. Quota catalog,
  discovery, and usage expose quota observations, not host topology. The native
  worker ingress and runtime (`bootstrap`, `profile`, `supervisor`) establish
  the worker endpoint and configured runtime.
* Kubernetes `api.py` models Jobs/services/secrets; `workers.py` renders the
  indexed Job and endpoint manifests; `kueue.py` resolves LocalQueue to
  ClusterQueue and reads capacity explanations. Kueue reads are diagnostics and
  optional autoscale input, not a second durable fence.

## 5. Conformance

[`conformance.py`](../../eyvan/capacity/conformance.py) is a provider-ABI suite,
not merely an interface/type check. It verifies routable capabilities and
unknown-cell `ABSENT`, then checks stale-fence non-mutation, exactly-once
create under replay, leaked-incarnation detection without adoption, and the
difference between inconclusive and conclusive provider failures. A provider
should implement `ProviderConformanceHarness` and run
`CapacityProviderConformanceTests` or `run_conformance`. Passing conformance
proves the contract against the harness; it does **not** prove a real cloud
account, network, image, quota, or accelerator works.

## 6. Autoscaling

The autoscale service is discovery-backed and currently generates GCP TPU Spot
fleets. It discovers quota rows, filters documented zone exclusions, unsupported
TPU shapes, unavailable live usage, insufficient shared quota, and exact cached
queue-admission denials, resolves typed profiles, then compiles one validated
fleet per eligible zone. Generated pools preserve finite shared generation
quota; `max_workers: 0` means the package-wide bound (10,000), not infinite
capacity. Defaults and generated configurations are cached locally. See
[`autoscale/service.py`](../../eyvan/capacity/autoscale/service.py),
[`compiler.py`](../../eyvan/capacity/autoscale/compiler.py), and
[`models.py`](../../eyvan/capacity/autoscale/models.py).

The controller reconciles desired min/max cells. Planning prefers workload-owned
free cells, then exact-width shared cells, then availability belief; it can use
pool alternates, cooldown/failure belief, and delayed defragmentation. Kubernetes
pool capacity observation is opt-in and UNKNOWN never becomes a full/empty
conclusion. Queue-admission denials are exact project/zone/type/Spot keys with
a 24-hour lazy-expiring cache; explicit workload requests are still sent.

## 7. Fleet configuration and state

`EyvanFleetConfig` validates names, pool resources, scaling bounds, executor and
backend compatibility, provider/agent endpoint families, and quota groups.
CPU/GPU/TPU/memory resource aliases are normalized; known catalog shapes fill
omitted admission resources, while unknown shapes require a complete explicit
positive vector. A fleet agent endpoint serves one driver family, although GCP
TPU and Compute modes intentionally share the GCP factory. Pool `zones` expand
into same-shape zone siblings that planning may treat as alternates.

`FleetStateStore` persists schema `eyvan.fleet-state` version 1: cells,
cell tombstones, workload fences, and a mutation lease. A cell record carries
provider identity, desired state, provider generation, execution endpoint,
Eyvan generation, fencing token, idempotency keys, and metadata. Local state is
atomic JSON guarded by a file lock. GCS state is a JSON object updated with
compare-and-swap: the REST path reads the object generation and writes with
`ifGenerationMatch`; the gcloud subprocess path remains a fallback. Conflicts
retry; unavailable state is not interpreted as an empty fleet. Tombstones keep
released cell generations from being resurrected.

Sources: [`fleet_state.py`](../../eyvan/capacity/fleet_state.py) and
[`gcs_rest.py`](../../eyvan/capacity/gcs_rest.py).

## 8. Planning, catalog, and quota boundaries

[`planning.py`](../../eyvan/capacity/planning.py) is pure worker-unit planning:
it mints monotonic, provider-safe cell IDs, accounts for active leases, ranks
candidate pools, enforces acquisition backoff and defragmentation delay, and
keeps cells owned by an autoscale request distinct from shared cells.

[`hardware_catalog.py`](../../eyvan/capacity/hardware_catalog.py) is the
authority for what a named shape *is*: kind, chips, quota units, per-host CPU
and memory, backend, topology where grounded, and HTTPS source/version. It is
not quota discovery: quota metrics say what is available and join on type.
GPU rows describe one VM (`hosts=1`, exact device count, machine type and GPU
model); TPU quota units differ by generation. Unknown hardware remains usable
only with an explicit complete resource vector. Catalog data is sourced
provider documentation, not test execution.

## 9. Audit and stage policy

[`audit.py`](../../eyvan/capacity/audit.py) emits transition-only structured
lifecycle events (planned/requesting/provisioning/provider transition/ready,
preempted/unhealthy/lost/retry/replacement/removed), including fleet, pool,
cell, provider, location, resource, worker count, capacity class, states, and
optional replacement reason. It logs every event and can append best-effort
JSONL; a sink failure cannot break a lifecycle transition.

[`stage_policy.py`](../../eyvan/capacity/stage_policy.py) declares the per-cell
stage budget, maximum single staged asset, explicit `least_recently_used` or
`largest_first` eviction, and retry limit. The budget is a hard charged-byte
ceiling; zero disables staging and `max_stage_bytes` cannot exceed it. The
policy declares bounds; staging decisions are in the server stage service.

## 10. Qualification limits (explicit)

This release has source-backed adapters, unit/integration/conformance coverage,
and provider-specific fakes, but that is not cloud or hardware qualification.
The repository does **not** establish that Eyvan has successfully provisioned
an EC2 instance, Azure VM, GCP TPU/VM, Kubernetes cluster, or any particular
GPU/TPU model in a real account. It does not establish credentials, IAM,
network/firewall/SSH reachability, quotas, images, drivers, Kueue setup,
accelerator topology, performance, pricing, or interruption behavior for an
operator's environment. Treat catalog rows as reviewable documentation and run
an operator-owned smoke test and qualification matrix before production claims.

## Glossary

* **Cell:** one schedulable capacity unit; it may contain one host or an atomic
  multi-worker allocation where the backend declares that capability.
* **Driver/backend:** lifecycle implementation selected by a normalized backend
  name.
* **Provider client:** API transport/domain seam used by a driver.
* **Generation/fence:** durable ownership version and token; stale mutations and
  foreign provider resources are rejected.
* **Pool:** homogeneous declarative capacity template and scaling bounds.
* **Quota vs catalog:** current provider allowance/usage versus static shape
  definition.
* **UNKNOWN/UNOBSERVED:** no conclusion because the substrate could not be
  read; neither means absent.
