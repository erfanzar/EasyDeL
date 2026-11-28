---
name: prepare-release
description: Prepare (and optionally publish) a release of an EasyDeL workspace package
---

# Prepare a Workspace Release

Prepare (and, only after explicit user confirmation, publish) a release for one of the 7 uv-workspace packages:
`easydel`, `spectrax`, `ejkernel`, `eformer`, `eray`, `surgecore`, or `eyvan`.

Follow `.agents/skills/release-workspace/SKILL.md` as the governing skill. The flow has two explicit stages — nothing
leaves the machine in Stage 1.

## Stage 0: Pre-Flight Checks

1. Verify a **clean git working tree across the entire repository** (`scripts/release.sh` aborts if
   `git status --porcelain` is non-empty).
2. Confirm version numbering and changelog readiness.
3. Run workspace checks:
   - `uv run lint-imports`
   - `uv run pre-commit run --all-files`
   - Affected tests on an accelerator (TPU: `env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu uv run pytest ...`; GPU: `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Use CPU only for non-computation suites (`libs/eray/tests`, `libs/surgecore/tests`, `libs/eyvan/tests`).
4. **If releasing `surgecore`** (Rust PyO3 crate), additionally run in `libs/surgecore`:

   ```bash
   cargo fmt --check
   PYO3_PYTHON="$(which python)" cargo clippy --locked --all-targets -- -D warnings
   PYO3_PYTHON="$(which python)" cargo test --locked
   cargo test --locked --no-default-features
   ```

   plus `ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests` from the repo root.

## Stage 1: Release Locally (No Network / No Tag Push)

1. Preview with dry run:

   ```bash
   scripts/release.sh <lib> <version> --dry-run
   ```

2. Run the release script (never hand-edit version pins):

   ```bash
   scripts/release.sh <lib> <version>
   ```

   This bumps `libs/<lib>/pyproject.toml` (plus `[package] version` in `libs/surgecore/Cargo.toml` and
   `libs/surgecore/Cargo.lock` for `surgecore`), syncs exact `==` sibling pins in `libs/easydel/pyproject.toml` (or
   refreshes `libs/easydel/README.md` for `easydel`), runs `uv lock`, and creates the local release commit.

## Stage 2: Publish (Outward-Facing — Confirm With User First)

- **Sibling-before-`easydel` ordering rule (`WORKSPACE.md`)**: `libs/easydel/pyproject.toml` pins exact sibling
  versions (`==`), so any bumped companion library (`spectrax`, `ejkernel`, `eformer`, `eray`, `surgecore`, or `eyvan`)
  **must** be published to PyPI before publishing an `easydel` release that depends on it.
- Preview the tag first:

  ```bash
  scripts/publish.sh <lib> --dry-run
  ```

- Only after explicit user confirmation, run:

  ```bash
  scripts/publish.sh <lib>
  ```

  This creates and pushes tag `<lib>-v<version>` to `origin`, triggering `.github/workflows/publish.yaml` (`uv build` /
  `uv publish` for Python libs; `surgecore-v*` triggers maturin `manylinux_2_28` x86_64/aarch64 + macOS arm64 abi3
  wheel and sdist jobs; `eyvan-v*` builds and stages the React dashboard before `uv build`).
- Push the branch separately with `git push` (the pre-push hook runs `scripts/subtree-sync.sh auto` for mirrors;
  `SUBTREE_SYNC_SKIP=1 git push` bypasses on mirror outage).

## Rules

- No self-credit trailers in release commits or notes.
- Report exactly which stage completed and what remains.
- If any check fails, stop and report rather than proceeding.
