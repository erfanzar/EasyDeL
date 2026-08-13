---
name: debug-tpu-setup
description: Diagnose EasyDeL TPU VM setup, local editable installs from libs, Ray startup, libtpu lock, bad-node recovery, or scripts/tpu_setup.sh failures. Use when TPU jobs fail before model code runs, imports resolve to wrong package versions, Ray workers cannot start, or one TPU VM behaves differently from the slice.
---

# Skill: Debug TPU Setup

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first, then read the `TPU Setup And Bad-Node
Recovery` section in `.claude/ops/OPS.md`.

## First Reads

Read these before running recovery commands:

- `WORKSPACE.md`
- `.claude/ops/OPS.md`
- `scripts/tpu_setup.sh`
- `libs/eformer/pyproject.toml`
- `libs/spectrax/pyproject.toml`
- `libs/ejkernel/pyproject.toml`
- `libs/easydel/pyproject.toml`

## Setup Entry Point

Use the repo setup script:

```bash
scripts/tpu_setup.sh --branch vnext
```

The script supports `--branch <ref>` and `--branch=<ref>`. It uses
`EASYDEL_REPO_URL`, `EASYDEL_SRC_DIR`, and `RAY_EXECUTABLE_PATH`, installs the workspace packages from `libs/`, and
verifies package versions for EasyDeL, eformer, ejkernel, spectrax, and JAX.

## Triage Order

1. Find the first remote failure in the setup log: clone, install, Ray, or health check. Later import errors are often
   fallout.
2. Confirm editable installs resolve from the workspace `libs/` paths.
3. Check whether libtpu is already owned by another process.
4. Check whether exactly one TPU VM is unhealthy or failing accelerator init.
5. Only perform node deletion or broad recovery when the user approves it.

## libtpu Lock Rule

TPU is single-process for libtpu. If a job owns the TPU, do not run TPU validation in parallel. CPU probes are only for
unrelated host-side checks such as import paths or package versions; they do not validate TPU kernels, eSurge runtime
behavior, or performance.

Useful read-only checks from `.claude/ops/OPS.md`:

```bash
fuser /dev/vfio/0 || true
ls -l /tmp/libtpu_lockfile || true
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" \
  --format="table(name.basename(),state,health,acceleratorType)"
```

## Version And Import Checks

When the symptom is "it installed but imports are wrong", verify package versions and module locations from the
environment used by the TPU job:

```bash
python - <<'PY'
import importlib.metadata as md
for name in ["easydel", "eformer", "ejkernel", "spectrax", "jax"]:
    print(name, md.version(name))
PY
```

If imports work on the driver but fail remotely, inspect the Ray environment and setup log before changing repo code.

## Recovery Boundary

Do not delete TPU VMs, restart slices, or clear shared scratch/checkpoint paths without explicit user approval. If
recovery is approved, follow the concrete
`gcloud compute tpus tpu-vm ...` commands in `.claude/ops/OPS.md`.
