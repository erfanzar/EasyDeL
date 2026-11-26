---
name: prepare-commit-pr
description: Prepare an EasyDeL commit or pull request safely. Use for staging, pre-commit, import layering, focused test selection, PR summaries, branch hygiene, release-policy checks, subtree mirror warnings, and enforcing the no self-credit trailer rule.
---

# Skill: Prepare Commit Or PR

Use this for commit and PR hygiene. For code review, load `.agents/skills/review-pr/SKILL.md`; for releases, load
`.agents/skills/release-workspace/SKILL.md`.

## First Reads

- `WORKSPACE.md` and `.pre-commit-config.yaml`
- `scripts/release.sh`, `scripts/publish.sh`, `scripts/subtree-sync.sh`
- `git status --short` and `git diff --stat`

## Worktree Discipline

- Do not stage unrelated user changes. When a file mixes user edits and yours, stage only the intended hunks.
- Never use destructive reset or checkout commands unless the user explicitly asks.
- Do not create legacy `CLAUDE.md` or `XERXES.md` files; update root `AGENTS.md` only when repo-wide rules actually
  change.
- **No new `os.environ` knobs**: verify the diff introduces no new env vars (wire toggles through config dataclasses,
  ContextVars, or explicit setters).

## Checks

Run focused tests selected via `.agents/skills/test-workspace/SKILL.md` for the touched packages. Never report a CPU run
of a computation test as validation — if no accelerator is available, state **unverified on hardware**.

Run workspace gates:

```bash
uv run lint-imports
uv lock --check                 # required whenever any pyproject.toml changes
uv run pre-commit run --all-files
```

Pre-commit caveats from `.pre-commit-config.yaml`:

- Auto-fixing hooks return `Failed` when they rewrite files — inspect the diff, restage intended edits, and rerun. Do
  not add `uv run` inside hook entries.
- `check-easydel-imports` runs `jax.devices()` with `JAX_PLATFORMS=''` and can fail or block if another process holds
  the `libtpu` lock.
- `cargo-fmt-surgecore` requires `cargo` on `PATH` when `libs/surgecore/**/*.rs` changes.
- Pre-push hooks enforce `uv lock --check` whenever any `libs/*/pyproject.toml` changes (mirrored by
  `.github/workflows/uv-lock-checkup.yaml`).

If `libs/surgecore` is modified, also run its Rust and CPU checks:

```bash
(cd libs/surgecore && cargo fmt --check)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo clippy --locked --all-targets -- -D warnings)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo test --locked && cargo test --locked --no-default-features)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests
```

## Release And Mirror Policy

Version bumps for all 7 workspace packages go through:

```bash
scripts/release.sh <easydel|spectrax|ejkernel|eformer|eray|surgecore|eyvan> <new-version> [--dry-run]
```

Never hand-edit versions or sibling dependency pins — see `.agents/skills/release-workspace/SKILL.md` (including the
mandatory sibling-before-`easydel` PyPI publish order).

The 6 standalone mirrors (`spectrax` -> `erfanzar/Spectrax`, `ejkernel` -> `erfanzar/ejkernel`, `eformer` ->
`erfanzar/eformer`, `eray` -> `erfanzar/eray`, `surgecore` -> `erfanzar/esurgecore`, `eyvan` -> `erfanzar/eyvan`) sync
via the pre-push hook:

```bash
DRY_RUN=1 scripts/subtree-sync.sh auto   # preview
scripts/subtree-sync.sh auto             # push out-of-date mirrors
```

`SUBTREE_SYNC_SKIP=1 git push ...` bypasses mirror sync only when the user knowingly requests it.

## Commit And PR Text

- Never add self-credit trailers or generated-by lines (`Co-Authored-By`, `Generated with`) in commits, PRs, tags, or
  release notes.
- PR summaries should cover: changed package surfaces, behavioral changes, exact tests run, skipped hardware or
  end-to-end checks, and release/migration notes when relevant.
- Keep claims aligned with actual verification — never claim TPU/GPU, eSurge, checkpoint, or benchmark validation unless
  that path actually ran on target hardware.
