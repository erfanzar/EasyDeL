# Part 3 — The SDK

The SDK's job: make "run this function on that shape" one import and one decorator.
Source: `eyvan/sdk/execution.py` (init/execute), `eyvan/sdk/client/jobs.py` (shape
configs and `EyvanJob`), `eyvan/records.py` (durable records), `eyvan/sdk/client/`
(transport stack).

## The public surface

`import eyvan` lazily exports (`eyvan/__init__.py`): `init`, `execute`,
`TpuConfig`, `GpuConfig`, `CpuConfig`, `EyvanJob`, `EyvanJobError`,
`EyvanPlacementError`, `EyvanUnavailableError`, `EyvanClient`,
`AsyncEyvanClient`, `ClientIdentity`, `EyvanCallableSubmitter`, `EyvanSettings`,
`EyvanFleetConfig`, and the in-job helpers `in_job_context`, `submit_child`,
`report_progress`. Everything is loaded on first attribute access and cached —
the package root stays light so native worker subprocesses can import
`eyvan.worker.runtime.local_worker` without dragging in the client stack.

## `init()` — connect to a plane

```python
eyvan.init(address=None, /, *, fleet=None, org_id=None, project_id=None,
           artifact_uri=None, artifact_store=None, token=None, dev_user=None,
           signing_key=None, base_url=None)
```

Called for you the first time a decorated callable is submitted, so a script that
is happy with the defaults never has to name it. Resolution order for the plane
(verified in `eyvan/sdk/execution.py`):

1. explicit `address`/`base_url` argument
2. `EYVAN_URL` environment variable
3. the saved CLI context (`eyvan use` / `eyvan connect`)
4. this node's own plane, if it serves one
5. the loopback default `http://127.0.0.1:8080`

The returned `ExecutionContext` carries the winning `base_url`, so you always know
which plane you are submitting to. Passing both `address` and `base_url` is an
error only when their values differ; `artifact_uri` and `artifact_store` are
mutually exclusive. Two safety rules worth knowing:

- On a **loopback** plane with no explicit identity, the SDK/CLI default to the
  `fleet_admin` role (it is your own dev plane). Anything naming a token or remote
  plane keeps the `operator` role.
- A `signing_key` is refused by the default standalone SDK/API submitter,
  including on loopback (`reject_remote_callable_signing_key`). An explicit
  custom submitter may implement its own producer/verifier protocol; the
  standalone controller does not distribute a callable signing key.

On loopback with no running plane, `init` starts one for you
(`ensure_loopback_control_plane`), and auto-applies a cached generated fleet
config if the named fleet is not yet serving — submission is meant to be one step.

## The shapes: `TpuConfig`, `GpuConfig`, `CpuConfig`

One decorator takes all three; everything under it is shape-agnostic. Fields
verified in `eyvan/sdk/client/jobs.py`:

| field | TPU default | meaning |
|---|---|---|
| `type` | — | accelerator type: `"v5p-16"`, `"v6e-8"`, `"nvidia-tesla-a100"`, `"cpu"` |
| `pods` | `1` | workers. Only `TpuConfig` allows `pods > 1` |
| `atomic` | `False` | `False` (**elastic**): pack warm cells of the same type, provision only the deficit; one lease group, one hardware world per cell. `True`: one exact-shape provider resource, one hardware world spanning every pod — single-mesh training |
| `chips` / `chips_indexes` | `None` | TPU only: run on part of one host instead of the whole slice. `chips=1` takes any free chip; `chips_indexes=[1, 2]` takes exactly those (global index; host = index // chips per host). Needs a pool with `chip_sharing: true`; see [runtimes](../runtimes.md#chip-shares) |
| `capacity_class` | `SPOT` (TPU) / `ON_DEMAND` | `reserved`, `on_demand`, `spot`, `flex_start` (`eyvan/records.py`) |
| `max_retries` | `3` | retries for **lost capacity** attempts |
| `max_preemption_retries` | `10` | restarts after spot reclamation — deliberately generous; reclamation is the cheap failure |
| `max_failure_retries` | `0` | retries of the job's *own* failure — deliberately stingy; a non-zero exit usually repeats |
| `secret_mounts` | `{}` | secret **references** resolved at dispatch into env vars; the reference is persisted, never the value |
| `checkpointable` / `checkpoint_hook` / `checkpoint_output_uri` | `False` / — | the scheduler grants preemption-checkpoint time only when a real hook is named |
| `can_submit_children` | `False` | lets the job submit children under your identity (fan-out, coordinators) |
| `dependencies` | `()` | packages installed on the cell before your callable loads |

Note what `type` means for GPUs: it names the pool's accelerator (e.g.
`nvidia-tesla-a100`), not a count of GPUs — one GPU VM is one pod.

## `@execute` — the decorator

```python
@execute(TpuConfig(type="v6e-8", pods=4), workspace="src/")
def run(epochs: int):
    ...

result = run(epochs=3)      # blocks; live progress lines; returns your value
```

- `config` must be an *instance* (a bare class raises `TypeError`).
- `workspace=` ships a directory tree to the worker's working directory, extracted
  before your callable imports — a research script can import its own modules with
  no image build. By default the decorated function's own directory ships;
  `workspace=False` opts out entirely.
- The standalone callable is pickled (cloudpickle) and uploaded unsigned by
  content digest. Only explicitly trusted projects may submit callables; digest
  verification is not signature authentication or a sandbox.
  Identical callables deduplicate to one stored artifact; concurrent identical
  submissions are safe (the loser of the upload race verifies the winner's digest
  and continues — pinned by tests in `tests/test_callable_execution.py`).
- Your callable's imports resolve **on the cell** (image packages +
  `dependencies` + whatever `workspace` shipped). The return value comes back
  through the artifact store.
- A `PYTHONPATH` set in your client process does **not** travel to workers — the
  SDK warns.

## `EyvanJob` — the detached handle

```python
job = run.submit(epochs=3)            # alias: run.remote(...)
job.id                                # workload id
job.status()                          # WorkloadState
job.logs()                            # currently available logs as a string
job.wait(timeout=600, progress=True)  # blocking wait, same narration
job.result(timeout=3600)              # your value or EyvanJobError
job.cancel()                          # durable cancel
```

Errors: `EyvanJobError` (terminal failure — the message contains the provider's
own sentence), `EyvanPlacementError` (actionable: names the types/fleets that
*are* placeable and suggests `eyvan use FLEET`), `EyvanUnavailableError` (the
plane could not be reached). `EyvanJob` has no `.base_url` attribute. Keep the
`ExecutionContext` returned by `init()` and read its `.base_url` when tooling
needs the resolved plane URL.

## In-job helpers

Inside a running callable:

- `report_progress(...)` (`eyvan/worker/runtime/progress.py`) — advisory
  self-reported progress to the plane via `EYVAN_PROGRESS_URL` +
  `EYVAN_MANAGED_RESULT_TOKEN`; stdlib-only, silent when nothing is listening.
- `submit_child(...)` / `in_job_context()` — when the shape declared
  `can_submit_children=True`, the job holds a submission capability under *your*
  identity: children's `created_by` is the submitting human, not a service
  identity (proof P3).

## The client stack underneath

All HTTP behavior lives in `eyvan/sdk/client/`:

| module | what it does |
|---|---|
| `endpoint.py` | `ControlPlaneEndpoint` — validated absolute URL; `is_loopback` is the dev-plane signal; default `http://127.0.0.1:8080` |
| `identity.py` | `Authorization: Bearer` takes precedence; else dev headers `X-Eyvan-User` / `X-Eyvan-Roles` / `X-Eyvan-Org` / `X-Eyvan-Project` (+ loopback local credential) |
| `transport.py` | httpx sync/async; `EyvanHTTPError` keeps status + structured `detail`; 422 `extra_forbidden` is annotated as likely version skew; `EyvanUnavailableError` tells you to start `eyvan-server` |
| `retry.py` | `RetryPolicy(max_attempts=3)`; retries 429/502/503/504; a 503 without `Retry-After` is not retried; honors capped `Retry-After` |
| `idempotency.py` | stable `Idempotency-Key` reused across retry attempts — a transient retry cannot become a second mutation |
| `sync.py` / `async_client.py` | `EyvanClient` and its async twin |
| `callables.py` | `EyvanCallableSubmitter` — the submission boundary |

Compatibility: requests carry `X-Eyvan-Client-Version`; a server with
`EYVAN_MIN_CLIENT_VERSION` set answers an old SDK with `426` and
`code: "client_too_old"`.

## One more knob: checkpointing

```python
TpuConfig(
    type="v5p-16", pods=2,
    checkpointable=True,
    checkpoint_hook="python -m my_module.save_checkpoint",  # shell command
    checkpoint_output_uri="gs://my-bucket/checkpoints/",
)
```

The hook is a shell command, not a Python import path. The runtime passes its
output location in `EYVAN_CHECKPOINT_URI` (empty when no output URI was supplied).
The scheduler grants preemption-checkpoint time (`ON_PREEMPTION` policy) only when
a real hook is named. The workload must save and restore its checkpoint; merely
setting `checkpointable=True` does not implement recovery.
