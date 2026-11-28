---
name: debug-tpu-setup
description: Diagnose EasyDeL TPU VM setup, local editable installs from libs, Ray startup, libtpu lock, bad-node recovery, or scripts/tpu_setup.sh failures. Use when TPU jobs fail before model code runs, imports resolve to wrong package versions, Ray workers cannot start, or one TPU VM behaves differently from the slice.
---

# Skill: Debug TPU Setup

Specialization of `.agents/skills/run-research/SKILL.md`. Load `run-research` first, then read the TPU setup and
bad-node recovery runbooks in `.agents/ops/OPS.md`.

## First Reads

- `WORKSPACE.md` and `.agents/ops/OPS.md`
- `scripts/tpu_setup.sh`
- Root `pyproject.toml`
- `libs/{easydel,spectrax,ejkernel,eformer,eray,surgecore,eyvan}/pyproject.toml`
- `libs/surgecore/Cargo.toml`

## Setup Entry Point

```bash
scripts/tpu_setup.sh [--branch <branch-or-tag-or-sha>]
```

Key behaviors of `scripts/tpu_setup.sh`:

- `--branch <ref>` / `--branch=<ref>` defaults to `main` and accepts branches, tags, or raw commit SHAs (falls back to
  `git fetch` + detached `FETCH_HEAD`).
- **Two-venv layout** (both Python 3.13.5): `~/orchestrator-venv` holds local `eopod`; `~/easy-venv` is the remote TPU
  runtime venv.
- Clones/updates `$EASYDEL_REPO_URL` (default `https://github.com/erfanzar/EasyDeL`) into `$EASYDEL_SRC_DIR` (default
  `~/easydel-src`) on every host. **Dirty-checkout guard**: if `$EASYDEL_SRC_DIR` already exists with uncommitted local
  edits and is on a different commit, the script refuses to switch refs instead of clobbering work.
- **Rust / `surgecore` bootstrap**: `libs/surgecore` is a PyO3/maturin Rust crate built from source by `uv sync`. If
  `cargo` is absent on a host, the script installs a minimal stable toolchain via `rustup` into `~/.cargo/bin`.
- Installs all 7 workspace packages editable via
  `uv sync --all-packages --active --inexact --extra tpu --extra torch --extra lm-eval --extra profile --group dev`,
  verifies all 7 packages plus profile/dev tools (`xprof`, `tb-nightly`, `xprof-nightly`, `ruff`, `pytest`,
  `pre-commit`, `basedpyright`, `import-linter`) on worker 0, installs `ray[default]==2.54.0`, configures Ray head and
  workers, and runs a Ray + JAX `eray` health check.
- **Input env vars honored**: `EASYDEL_REPO_URL`, `EASYDEL_SRC_DIR`, `EOPOD_COMMAND_TIMEOUT` (1800),
  `EOPOD_CLONE_TIMEOUT` (300), `EOPOD_INSTALL_TIMEOUT` (2400), `REMOTE_PYTHON_TIMEOUT` (300), `EASYDEL_VERBOSE` (0).
  (`RAY_EXECUTABLE_PATH` is set internally to `${REMOTE_VENV_PATH}/bin/ray`, not read as an input.)

## Triage Order And Remote Logs

1. Inspect the **first** failure on the affected host before chasing downstream import tracebacks:
   - Clone / dirty-checkout errors: `/tmp/easydel-clone.log`
   - Rustup / `cargo` / maturin `surgecore` build failures: `/tmp/easydel-rustup.log`
   - Ray stop / start failures: `/tmp/easydel-ray-stop.log`, `/tmp/easydel-ray-start.log`
2. Verify all 7 workspace packages resolve editably from `$EASYDEL_SRC_DIR/libs/` (snippet below).
3. Check whether `libtpu` is held by another process.
4. Check whether a single TPU VM in the slice is unhealthy or failing accelerator init.
5. Only recreate/delete TPU VMs or wipe shared state after explicit user approval.

## libtpu Lock Rule

TPU hosts hold a single `libtpu` process lock — never run TPU validation in parallel with an active TPU job. Host-side
probes (versions, import paths) may inspect Python without touching the accelerator, but they do not validate TPU
runtime behavior.

Read-only checks from `.agents/ops/OPS.md`:

```bash
fuser /dev/vfio/0 || true
ls -l /tmp/libtpu_lockfile || true
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" \
  --format="table(name.basename(),state,health,acceleratorType)"
```

## Version And Editable-Path Check

Run from the TPU job's Python (e.g. `~/easy-venv/bin/python`) to verify distribution versions AND editable `libs/`
locations (note: SpectraX's distribution name is `spectrax-lib`, not `spectrax`):

```bash
python - <<'PY'
import importlib
import importlib.metadata as md
from pathlib import Path

for dist, mod_name in [
    ("easydel", "easydel"),
    ("eformer", "eformer"),
    ("ejkernel", "ejkernel"),
    ("spectrax-lib", "spectrax"),
    ("eray", "eray"),
    ("surgecore", "surgecore"),
    ("eyvan", "eyvan"),
    ("jax", "jax"),
]:
    mod = importlib.import_module(mod_name)
    print(f"{dist:14s} {md.version(dist):10s} {Path(mod.__file__).resolve()}")
PY
```

If imports succeed on the driver but fail on workers, inspect the Ray worker runtime env and `/tmp/easydel-ray-start.log`
before changing repo code (see `.agents/skills/eray-orchestration/SKILL.md` for Ray/pod orchestration).

## Recovery Boundary

Never delete TPU VMs, restart slices, or clear shared scratch/checkpoint paths without explicit user approval. When
approved, use the `gcloud compute tpus tpu-vm ...` procedures in `.agents/ops/OPS.md`.
