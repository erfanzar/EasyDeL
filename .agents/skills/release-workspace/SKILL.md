---
name: release-workspace
description: Release or publish one of the six EasyDeL uv workspace packages. Use for version bumps, dry-run releases, publish tags, standalone repo subtree mirror sync, package dependency pins, surgecore Rust/maturin releases, or pre-push release hygiene for easydel, spectrax, ejkernel, eformer, eray, or surgecore.
---

# Skill: Release A Workspace Package

Use this when asked for a package release, version bump, PyPI publish, or subtree mirror sync. Releasing is two
explicit steps: `scripts/release.sh` (local bump + commit) and `scripts/publish.sh` (pushes tag -> CI publishes).

## First Reads

- `WORKSPACE.md` and `.pre-commit-config.yaml`
- `scripts/release.sh`, `scripts/publish.sh`, `scripts/subtree-sync.sh`
- `.github/workflows/publish.yaml` and `.github/workflows/sync-subtrees.yaml`
- Target `libs/<package>/pyproject.toml` (plus `libs/surgecore/Cargo.toml` when releasing `surgecore`)

## Version Bump Rule (Local Only)

Never hand-edit package versions, `Cargo.toml`/`Cargo.lock` crate versions, or sibling dependency pins:

```bash
scripts/release.sh <easydel|spectrax|ejkernel|eformer|eray|surgecore> <new-version> --dry-run
scripts/release.sh <easydel|spectrax|ejkernel|eformer|eray|surgecore> <new-version>
```

Operational invariants of `scripts/release.sh`:

- **Clean tree required**: fails if `git status --porcelain` is non-empty (commit or stash first).
- Maps `spectrax` to PyPI distribution `spectrax-lib` (the other 5 match their lib name).
- On sibling releases (`!= easydel`), syncs exact `==` pins in `libs/easydel/pyproject.toml` while preserving
  environment markers (e.g. `; sys_platform == 'linux'`).
- On `easydel` releases, refreshes `libs/easydel/README.md` from root `README.md`.
- On Rust crates (`libs/surgecore/Cargo.toml`), bumps `[package] version` in `Cargo.toml` and the crate's own entry in
  `Cargo.lock` via `sed` (no `cargo` required for the bump).
- Runs `uv lock`, stages release files, and creates a local commit `release: <dist> v<version>` — nothing leaves the
  machine.

## Publish Order And `surgecore` Wheels

- **Mandatory publish order**: `libs/easydel/pyproject.toml` pins exact sibling versions (`eformer==`, `ejkernel==`,
  `eray==`, `spectrax-lib==`, `surgecore==`). Any bumped sibling MUST be published and available on PyPI **before**
  publishing an `easydel` release that pins it.
- **`surgecore` CI pipeline** (`.github/workflows/publish.yaml`): pushing `surgecore-v*` skips the Python `uv build`
  job and runs `surgecore-wheels` (maturin `manylinux_2_28` for `x86_64` and `aarch64` plus macOS `arm64`, `abi3-py311`
  covering Python 3.11–3.13), `surgecore-sdist`, and `surgecore-publish`.

## Publish Rule (Explicit User Approval Required)

Only publish when the user explicitly asks:

```bash
scripts/publish.sh <easydel|spectrax|ejkernel|eformer|eray|surgecore> --dry-run
scripts/publish.sh <easydel|spectrax|ejkernel|eformer|eray|surgecore>
```

- Creates `<lib>-v<version>` and pushes **only the tag** to `origin` (triggers `.github/workflows/publish.yaml` via
  PyPI trusted publishing or `PYPI_API_TOKEN`). Push the git branch separately with `git push`. Do not invent manual
  tags.

## Mirror Sync (All 5 Standalone Mirrors)

Mirrors: `spectrax` (`erfanzar/Spectrax`), `ejkernel` (`erfanzar/ejkernel`), `eformer` (`erfanzar/eformer`), `eray`
(`erfanzar/eray`), and `surgecore` (`erfanzar/esurgecore`).

```bash
DRY_RUN=1 scripts/subtree-sync.sh auto   # preview mirror updates
scripts/subtree-sync.sh auto             # push out-of-date mirrors (also runs in pre-push hook)
scripts/subtree-sync.sh pull <lib>       # reconcile a diverged mirror
```

`SUBTREE_SYNC_SKIP=1 git push ...` bypasses the pre-push hook when knowingly requested. `.github/workflows/sync-subtrees.yaml`
is the server-side backstop (requires `SUBTREE_SYNC_TOKEN`).

## Verification

Before a release commit or publish, run:

```bash
uv run lint-imports
uv lock --check
uv run pre-commit run --all-files
```

Run affected tests via `.agents/skills/test-workspace/SKILL.md` when releasing code changes beyond version metadata.
For `surgecore` releases, also run:

```bash
(cd libs/surgecore && cargo fmt --check)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo clippy --locked --all-targets -- -D warnings)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo test --locked && cargo test --locked --no-default-features)
uv build --package surgecore --wheel
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests
```

## Text Policy

Never include self-credit trailers (`Co-Authored-By`, `Generated with`) in commits, tags, release notes, or PR text.
Keep release notes focused on package behavior and user impact.
