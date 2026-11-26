# Runtimes and images

How a cell gets its Python, your libraries, and — optionally — your Docker
image. Grounded companion to [operate](operate.md); installation
prerequisites live in [install](install.md).

## Two provisioning models

| | venv model | OCI model |
|---|---|---|
| who provides Python | the fleet, via `runtime.python` | your image |
| who provides libraries | `runtime.init_commands` (and per-job `dependencies`) | the image (`runtime.artifacts` adds code objects) |
| who provides the container | none — native venv at `/opt/eyvan-native/venv` | `runtime.container.image` |
| generated fleets | default; driven by the [defaults profile](#the-defaults-profile) | yes, when `runtime_container_image` is set |
| typical use | fast iteration, `uv pip install` per fleet | pinned, reproducible training runtime |

The interpreter contract is checked **before provisioning** when
`runtime.python` is declared: a callable pickled on a different
`major.minor` is refused at submission.

## Two allocation modes

Independent of the runtime: **how a job's `pods` workers are allocated**.

- **elastic (default)** — `pods` is a worker count, satisfied from any
  combination of same-type cells: warm slices are packed first (biggest
  first, so the fewest hardware worlds) and only the deficit is
  provisioned. Resizing `pods` 2→4 keeps the two warm slices and asks for
  two more. The job runs as one lease group, one hardware world per cell —
  framework-level parallelism across worlds.
- **atomic** — `TpuConfig(..., atomic=True)`: one exact-shape provider
  resource, one hardware world spanning every pod. Required when the job's
  math needs the whole allocation in one JAX world (single-mesh training,
  an all-reduce across every chip). Resizing rebuilds from zero — a
  provider topology cannot grow — and the plane drains now-unusable
  smaller leftovers instead of billing them to the idle timeout.

All-or-nothing holds in both: nothing is leased unless the full worker
count is covered, so a deficit that never grants holds no warm cell
hostage.

## Chip shares

A pool that sets `chip_sharing: true` lets several jobs share one TPU
slice, each holding some chips of one host and seeing only those:

```python
@execute(TpuConfig("v5p-128", chips=1))               # any free chip
def probe(): ...

@execute(TpuConfig("v5p-128", chips_indexes=[1, 2]))  # exactly chips 1 and 2
def train(): ...
```

The CLI equivalent is `eyvan run --tpu v5p-128 --chips 1 -- python probe.py`
or `--chip-index 1 --chip-index 2`.

- Chip indexes are global within the slice: host `index // chips_per_host`
  owns chip `index` (a v5p host has 4 chips, so a v5p-128 has chips 0-63).
- One share lives on one host. On a 4-chip host it is one chip, a
  neighbouring pair, or the whole host. The chips form a ring 0-1-2-3-0;
  the diagonal pairs {0,2} and {1,3} are refused because libtpu aborts on
  them (measured on a v5p host). Other host widths offer single chips only.
- A share reserves its fraction of the host's CPU and memory.
- Pinned chips wait while busy; a share never preempts anything. Whole-slice
  jobs and shares never run on the same cell at once.
- The job gets its own libtpu environment (`TPU_VISIBLE_CHIPS`, process
  bounds, a private `TPU_PROCESS_PORT`) and runs only on the owning host.
  Separate shares are separate JAX runtimes: there are no collectives
  between them.
- Turning `chip_sharing` on or off updates live cells in place; it does
  not replace them.

Pools without `chip_sharing` refuse chip requests and keep one job per cell.

## The defaults profile

`~/.eyvan/autoscale/defaults.yaml` is the one place an operator pins what
every generated zone fleet derives:

```yaml
version: 1
project: my-gcp-project
fleet_prefix: research
bootstrap_egress: public-internet
runtime_python: '3.13'
runtime_init_commands:
- uv pip install --python /opt/eyvan-native/venv/bin/python --prerelease=allow 'easydel[tpu,profile,torch]'
# runtime_container_image: us-east5-docker.pkg.dev/my-gcp-project/eyvan/runtime@sha256:...
min_workers: 0
max_workers: 0
```

- `runtime_python` — the interpreter every worker runs; arms the
  pre-provisioning mismatch refusal.
- `runtime_init_commands` — run on every host after the worker environment
  exists and before the worker unit starts. The window to install what the
  interpreter needs.
- `runtime_container_image` — pin one blessed OCI image (prefer an
  immutable digest) and every generated fleet declares it as its container.
  The generator states the image but never the host preparation: docker
  installation and `gcloud auth configure-docker <registry-host>` belong in
  `runtime_init_commands`, where the one command that must match the
  image's registry stays visible instead of derived.

`eyvan autoscale init` edits this profile interactively; the file is
owner-only (`0600`) and machine round-trippable.

The native worker host requires **Python 3.11 through 3.13**, including when
workloads run in an OCI container. Declare `runtime_python` in the defaults
profile, or `runtime.python` in a hand-written fleet, when the VM image does
not supply a supported interpreter. Built-in Ubuntu 22 TPU images can expose
Python 3.10 as `python3`; leaving the declaration unset can therefore fail
bootstrap before any workload starts. `examples/fleet.yaml` explicitly selects
`runtime.python: "3.11"`; node-level declarations can override it.

An explicit declaration builds a managed interpreter environment rather than
inheriting the image's system-site packages. Install matching host accelerator
libraries through `runtime_init_commands`, or supply workload dependencies in
your OCI image. Undeclared custom-image behavior and global generator defaults
remain unchanged; do not assume Eyvan silently upgrades an image's Python.

## A zone fleet from scratch

```bash
eyvan autoscale init                     # once: the profile above
eyvan autoscale generate --zone us-central1-a   # caches configs/<prefix>-<zone>.yaml from live quota
eyvan up --select research-us-central1-a --preemptible   # apply
```

With `fleet_prefix: a1` the fleet is `a1-us-central1-a` — your "A1 up".
Regenerate after changing defaults; generated files are derived, so never
hand-edit them (a regeneration will not preserve it). Teardown drains every
cell the fleet manages:

```bash
eyvan down ~/.eyvan/autoscale/configs/research-us-central1-a.yaml -y
```

## Customizing the Docker image (OCI model)

The complete worked shape is
[examples/easydel-v5p16.yaml](../examples/easydel-v5p16.yaml):

1. Build the image with JAX/EasyDeL baked in; push to Artifact Registry;
   reference it by **immutable digest**.
2. Declare it per pool (or once, via `runtime_container_image`):
   `image`, `pull_policy: if-not-present`,
   `working_dir: /var/lib/eyvan/workload`, `privileged: true` — native OCI
   TPU jobs reject launch unprivileged.
3. Make hosts container-ready in `init_commands` (docker installed,
   `gcloud auth configure-docker` for the exact registry host).
4. Ship application code as `runtime.artifacts` (for example
   `gs://…/training-code/` → `/opt/eyvan-runtime/training`) with
   `env: PYTHONPATH` pointing at it — the image build does not implicitly
   publish application code; upload the object before `eyvan up`.
5. `bootstrap_egress: public-internet` for the docker install and pull, or
   a prebaked engine plus wheelhouse with `private-google-access`.

### Attached TPUs (existing hardware)

A TPU Eyvan did not create gets its image at attach time:

```bash
eyvan gcp connect --ssh you@HOST --python 3.13 \
  --container-image us-east5-docker.pkg.dev/PROJECT/REPO/runtime@sha256:... \
  --init-command "gcloud auth configure-docker us-east5-docker.pkg.dev --quiet"
```

The attach path states the same container block a generated fleet does
(`privileged: true`, `pull_policy: if-not-present`,
`working_dir: /var/lib/eyvan/workload`); without `privileged` every TPU job
would be refused at launch. Docker must already run on every host (Cloud TPU
VM images ship it). To change the image later, re-run `connect`; the
dashboard's fleet init-script editor refuses attached TPUs because Eyvan never
creates their hosts.

### Per-job packages and workspaces inside an image

`eyvan run --with PKG` and `--workspace DIR` also work in a container whose
image has no Eyvan installed. Their wrappers are staged, standard-library
only, into the read-only bridge mount
(`/opt/eyvan-native/oci-callable-runner`) beside the callable runner, and the
container executor rewrites `python -m eyvan.worker.runtime.<wrapper>` to
those files. The image still needs a Python 3.11+ on `PATH` as `python`, and
`--with` needs `pip` (or `uv`) in that interpreter.

## Per-job additions

`TpuConfig(dependencies=("flax",))` installs packages before the callable loads.
The job gets its own environment, built from the interpreter running the
wrapper and inheriting that interpreter's packages.

For native venv workers, this includes `/opt/eyvan-native/venv` (Eyvan, JAX,
libtpu) and the host's system packages. Environments are cached per requirement
set and interpreter in `/var/lib/eyvan/runtime-envs`, which the bootstrap
creates for the workload user. A second job or rank using the same interpreter
can reuse the install. The cache identity includes the format version, resolved
interpreter path, Python version, and inherited parent site-packages paths.
Older cache entries remain untouched for jobs that may still use them.

In OCI mode, the environment inherits the image's Python packages, not the
host venv. Containers are removed after execution; cross-job cache reuse needs
an explicit persistent mount. The native host cache is not mounted
automatically.

`@execute(..., workspace="src/")` ships a source tree to the worker's
working directory for the callable to import.

### Per-job setup step

`setup=` runs a bash script on every host the job lands on, after its
`dependencies` are installed and before the callable (or command) starts:

```python
from eyvan import SetupScript, TpuConfig

# Inline bash.
TpuConfig("v6e-8", setup="$EYVAN_PYTHON -m pip install 'git+https://github.com/me/lib@abc123'")

# A script fetched on each host, with arguments and an optional pin.
fetched = SetupScript("https://example.com/eyvan_setup.sh", args=["--ref", "abc123"], sha256=None)
TpuConfig("v6e-8", setup=fetched)

# A local file, read at submission and shipped inline.
local = SetupScript("scripts/eyvan_setup.sh", args=["--ref", "abc123"])
TpuConfig("v6e-8", setup=local)
```

`GpuConfig` and `CpuConfig` take the same `setup=`.

For EasyDeL there is a built-in step, `EasyDeLSetup`:

```python
from eyvan import EasyDeLSetup, TpuConfig

easydel = EasyDeLSetup(branch="vnext-into-main", repo="erfanzar/EasyDeL")
TpuConfig("v6e-8", setup=easydel)
```

`branch` defaults to `vnext-into-main` and `repo` to `erfanzar/EasyDeL`; `repo`
also accepts an `https://` git URL. `branch` may be a branch, a tag or a full
commit. It is resolved to one commit when the job is submitted, not when the
decorator runs, so every host installs the same code and a host's cached
install is reused only for that commit; pushing to the branch makes the next
job install again. For EasyDeL's uv workspace (a root `pyproject.toml` with
`[tool.uv.workspace]` and a `uv.lock`), the repository's own lock is exported
for every member and installed into the job's interpreter: every package under
`libs/` comes from that checkout, and every other dependency at the version the
lock pins. A plain `pip install .` of that root would take the members from
PyPI, and `uv sync` would build a separate `.venv`, so neither is used. Without
a workspace lock, every `libs/*` package is installed from the checkout in one
resolver call; a single package at the repository root also works. `accelerator` picks the extras (`tpu`, the
default, `cuda`, or `cpu` for none), and `torch_cpu=True` also installs CPU-only
torch, which EasyDeL uses to read Hugging Face checkpoints. The hosts need
`git`.

- The step runs as the job user, not root, inside the job's own environment:
  `VIRTUAL_ENV` and `PATH` point at it, and `EYVAN_PYTHON` names its
  interpreter. Install into that interpreter. System changes such as apt
  packages or kernel settings belong in the fleet's `runtime.init_commands`.
- URLs must be `https://`. A redirect that leaves https is refused, and so is a
  script larger than 1 MiB. With `sha256`, a fetched script whose bytes differ
  fails the job. Without it, the bytes fetched at run time are what the cache
  is keyed by, so a script that changed behind its URL runs again.
- The result is cached per host with the job's environment, keyed by the exact
  script content plus its arguments. A second job or rank with the same step
  reuses it; changing either builds a new environment. Jobs without `setup=`
  keep their existing cache keys.
- Output is not captured: it goes to the job's log (`eyvan workload logs`). A
  non-zero exit fails the job before it starts, discards the half-built
  environment, and counts against `max_failure_retries`.
- On a multi-host job the step does not see the multi-host JAX join marker, so
  any Python it runs cannot try to join the gang. The job itself still joins.
- Unset, `setup` is omitted from the request and the stored workload, so the
  payload is unchanged. Inline scripts are capped at 32 KiB and take at most 64
  arguments.

## Local components

`eyvan runtime init` stages the local native runtime profile
(`~/.eyvan/gcp-native-runtime.json`: agent shared secret, native worker
root secret, artifact root, wheel URIs); `eyvan runtime deploy` rebuilds
and redeploys the local plane and agent from current source. A plane
restarted without the profile's secrets 401-loops against a running agent
— [troubleshoot](troubleshoot.md) has the table.
