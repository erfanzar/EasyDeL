# Eyvan (ایوان)

Eyvan is a capacity plane for research compute. You state the shape of work you
need — `2 × v5p-16, spot, this fleet` — as a digest-verified unsigned Python
callable submitted to an explicitly trusted project. The plane acquires the
machines, boots them, dispatches your callable, and returns its result.
Capacity records show what was requested, what a provider accepted or refused,
and how long the handover took.

## The sixty-second version

```bash
uv sync --extra dev                      # workspace install + development dependencies
uv run eyvan up examples/laptop.yaml     # start the local loopback CPU fleet before SDK init
uv run eyvan start                       # find or point at a control plane
uv run eyvan doctor                      # what is missing, named together
```

Eyvan supports Python `>=3.11,<3.14` (the range declared in `pyproject.toml`).

```python
from eyvan import CpuConfig, execute, init

init(fleet="laptop")


@execute(CpuConfig())
def train():
    return sum(range(10))
```

Calling `train()` submits the callable and waits, narrating progress:
`queued`, `provisioning`, `blocked … the plane is retrying (attempt N)`,
`dispatching`, then your result. The job is durable — it keeps running if your
client exits (`eyvan workload status <id>` to follow).

### With a TPU fleet

First configure a fleet with matching TPU pools and a runtime that provides
JAX. The laptop fleet above has only CPU capacity.

```python
from eyvan import TpuConfig, execute, init
from eyvan.records import CapacityClass

init(fleet="research-us-central1-a")


@execute(TpuConfig(type="v5p-16", pods=2, capacity_class=CapacityClass.SPOT))
def device_count():
    import jax

    return jax.device_count()
```

## The SDK in one page

### Connecting: `init`

```python
init(address=None, /, *, fleet=None, org_id=None, project_id=None,
     artifact_uri=None, artifact_store=None, ...)
```

With no arguments the SDK discovers the plane the CLI uses (`eyvan start`
shows you what exists: a local plane, or heads in your project). Point it at
an explicit plane by address — `eyvan.init("http://10.0.0.5:8080")` — or
bind a default fleet with `eyvan.init(fleet="research-us-central1-a")`.
Everything else is scope (`org_id`, `project_id`), where artifacts go (a
`gs://` root or your own `ArtifactStore` implementation), and fleet-state
location (`EYVAN_FLEET_STATE_URI`) when agents own distinct capacity
boundaries.

### The shape: `TpuConfig`, `GpuConfig`, `CpuConfig`

One decorator takes all three; everything under it is shape-agnostic. The
TPU config's full surface (`eyvan/sdk/client/jobs.py`):

| field | default | meaning |
|---|---|---|
| `type` | — | accelerator type, e.g. `"v5p-16"`, `"v6e-8"` |
| `pods` | `1` | workers — how many slices of `type` the job asks for |
| `atomic` | `False` | `False` (**elastic**, node-pool style): pack warm cells of the same type and provision only the deficit — going `pods` 2→4 keeps the two warm slices and asks for two more, one lease group, one hardware world per cell. `True`: one exact-shape provider resource, one hardware world spanning every pod — what single-mesh training needs |
| `capacity_class` | Spot preference | `SPOT`, `ON_DEMAND`, `RESERVED`, or `FLEX_START`; an explicit value is exact. If omitted, a reserved-only fleet may substitute `RESERVED` |
| `max_retries` | `DEFAULT` | retries for **lost capacity** attempts |
| `max_preemption_retries` | `10` | restarts after the provider reclaims spot — deliberately generous; reclamation is the cheap failure |
| `max_failure_retries` | `0` | retries of the job's *own* failure — deliberately stingy; a non-zero exit usually repeats, and re-running it costs an accelerator |
| `secret_mounts` | `{}` | secret **references** resolved at dispatch into env vars on the cell — a reference is what gets persisted, never the value |
| `checkpointable` / `checkpoint_hook` / `checkpoint_output_uri` | `False`/— | the scheduler grants preemption-checkpoint time only when a real hook is named; otherwise a reclaimed job restarts from zero |
| `can_submit_children` | `False` | lets the job submit children under your identity — off by default; a credential nobody asked for is one nobody is watching |
| `dependencies` | `()` | packages installed on the cell before your callable loads — adding a library should not mean rebuilding an image |

### The decorator: `execute`

```python
@execute(TpuConfig(type="v6e-8", pods=4), workspace="src/")
def run(epochs: int):
    ...
```

`workspace=` accepts a directory path and ships that tree to the worker's
working directory before the callable imports. Omit it to derive the workspace
from the decorated function's source-file directory; interactive callables
without a source file need an explicit path. Set `workspace=False` to disable
workspace shipping.

### Calling: blocking wait, or detached handle

```python
result = run(epochs=3)          # blocks; live progress lines; returns your value
```

```python
job = run.submit(epochs=3)      # EyvanJob — detach, follow, intervene
job.status()                    # WorkloadState
job.logs()                      # the flattened, globally-ranked log stream
job.wait(timeout=600, progress=True)  # blocking wait, same narration
job.result(timeout=3600)        # your value, or EyvanJobError with the reason
job.cancel()                    # durable cancel
```

Your callable's *return value* comes back through the artifact store; its
imports resolve on the cell (image packages + `dependencies` + whatever
`workspace` shipped). Identical callables deduplicate by content digest;
a concurrent upload verifies the existing object's digest before using it.
See [artifact-safety](docs/artifact-safety.md).

One more client-side knob worth knowing: the returned job exposes its
transport (`.base_url`) so tooling built on the SDK can follow the same
plane the SDK followed, without re-deriving discovery.

Failure narration is part of the API contract: a refused capacity attempt
renders as `blocked | … | the plane is retrying (attempt N)` with the
provider's diagnostic, and a terminal failure arrives as an exception whose
message contains the provider's own sentence — not a bare exit code.

## Developing on top of Eyvan

Four seams, in order of how often you need them:

1. **A different shape** — no code: a `TpuConfig`/`GpuConfig`/`CpuConfig`
   value. Check the shape is real for you first:
   `eyvan capacity quota --fits v5p-16:2` (quota **and** admission
   permission **and** supply are three different gates).
2. **A different fleet** — a YAML config: `eyvan up <config>` from a
   hand-written file (see `examples/fleet.yaml`, `examples/laptop.yaml`) or
   a generated fleet (`eyvan up --region <region> --preemptible`, or
   `eyvan up --quota-based --zone <zone> --preemptible`). Node types, runtime
   env, and autoscaling knobs live in the config, not in your callable.
3. **A different cloud** — a provider package with a driver and transport.
   External plugins default to single-machine cells; multi-worker planning
   also needs an in-tree capability entry and a supporting pool factory.
   See [providers](docs/providers.md) for the ABI and conformance contract.
4. **A different workflow** — composition: `can_submit_children=True` grants
   your job a submission capability under your identity (fan-out workers,
   coordinator patterns), and checkpointing hooks let long spot jobs survive
   reclamation by design rather than by luck.

### Developing Eyvan itself

```bash
git clone <this repo> && cd eyvan
uv sync --extra dev                # editable install + dev deps (pytest, pytest-asyncio)
npm --prefix dashboard ci
npm --prefix dashboard run build # build the SPA before Python tests
uv run pytest tests -m fast -q   # smoke subset for iteration, not a release gate
uv run pytest tests -q -rs       # full suite; real PostgreSQL requires the test fixture setup
uv run ruff check eyvan tests    # lint (line 121, py3.11)
uv run eyvan-server --log-level info   # run the plane from source
uv run eyvan-agent --host 127.0.0.1 --port 8090 --allow-unauthenticated  # local dev only
```

Verify the dashboard from a fresh checkout with the commands in
[Contributing — Dashboard](CONTRIBUTING.md#running-the-tests):

```bash
npm --prefix dashboard ci
npm --prefix dashboard test
npm --prefix dashboard run build
```

The rules that keep the package honest — enforce them in review, not in
memory:

- **Provider boundary:** nothing that decides (scheduler, planner, durable
  model) may name a provider; `tests/test_provider_boundary.py` enforces it.
- **Claims need receipts:** a behavior claim cites a recorded run; hardware
  claims go through `eyvan proof` (exit 0 pass, 1 fail, 2 *did not exercise
  the behaviour*, 3 refusal) — see [proofs](docs/proofs.md).
- **Tests assert public behavior:** state transitions, API outputs, parity —
  not private helper calls or log strings.
- **Diagnostics travel with failures:** a CLI or provider failure raises with
  its own words attached (`GcsCommandError` carries command + stderr).
- Architecture decisions are design records under
  [docs/design/](docs/design/0001-scoped-artifact-lifecycle.md), each stating
  its real implementation status.

## The ideas everything else follows from

1. **You state the shape, not the pool.** The scheduler and capacity planner
   decide where that shape can exist.
2. **Receipts or it did not happen.** Capacity claims cite recorded runs.
3. **Failure narrates itself.** Diagnostics render, never `FAILED`.
4. **Nothing that decides names a provider.** Clouds plug in; the core stays
   agnostic.
5. **Durable state survives the plane.** A restarted `eyvan-server`
   reconciles the whole world from its durable store.

## Concepts

| term    | meaning                                                            |
|---------|--------------------------------------------------------------------|
| plane   | the control plane (`eyvan-server`) — scheduler, planner, state     |
| agent   | a cell agent (`eyvan-agent`) owning machines on one backend        |
| fleet   | a named set of capacity pools you can submit into                  |
| pool    | one (backend, accelerator, capacity class, zone) combination       |
| cell    | one acquired allocation (e.g. one spot queued resource)            |
| workload| your callable + its requested shape, moving through durable states |

## Where it runs

Registered capacity backends: `local` (loopback laptop fleets), `gcp-tpu`
(generated zone fleets driving TPU queued resources, spot or on-demand),
`gcp-compute`, the `gcp` umbrella, `aws-ec2` (the worked provider example)
and the `aws` umbrella, `azure-vm` and the `azure` umbrella, `kubernetes`
(`k8s`) against a real API server, `manual`, `onprem-ssh`, and `gcp-attach`
(an existing Cloud TPU VM attached with `eyvan gcp connect`; observed, never
created or deleted).

Backend availability is not proof of qualification on your infrastructure.
GCP TPU is the backend with live hardware runs; the on-premises SSH driver is
code with no run against a real SSH host yet. See [proofs](docs/proofs.md) for
evidence requirements and backend-specific limits.

## The docs map

New to the project? Start with the [tutorial](docs/tutorial/README.md) —
seventeen source-grounded parts that walk from "what is Eyvan" through the SDK,
the control plane, providers, operations, deployment, and an explicit Spot TPU example.

| page | what it answers |
|---|---|
| [tutorial](docs/tutorial/README.md) | a part-per-subsystem tutorial, grounded in the source |
| [eyvan-explained](docs/eyvan-explained.md) | the mental model, one workload's full journey |
| [install](docs/install.md) | laptop path, the fifteen, GCP head, secrets |
| [operate](docs/operate.md) | the runbook: fleets, workloads, capacity, zone racing |
| [runtimes](docs/runtimes.md) | the two provisioning models, the defaults profile, custom images |
| [proofs](docs/proofs.md) | what is proven, with receipts; known rough edges |
| [providers](docs/providers.md) | writing a capacity provider; the conformance contract |
| [security](docs/security.md) | trust boundaries, tokens, what jobs can do |
| [env-surface](docs/env-surface.md) | every `EYVAN_*` variable |
| [artifact-safety](docs/artifact-safety.md) | tenancy, lifecycle, the digest-addressed store |
| [troubleshoot](docs/troubleshoot.md) | symptom → cause → action tables |
| [design records](docs/design/0001-scoped-artifact-lifecycle.md) | the ADRs and their real status |

## What is proven

See [proofs](docs/proofs.md) for proof criteria and evidence limits. After a
restart, use the intended durable state and matching credentials; inspect
quarantined leases before intervening. See [operate](docs/operate.md) and
[troubleshoot](docs/troubleshoot.md) for recovery procedures.
