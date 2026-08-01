---
name: prepare-commit-pr
description: Prepare an EasyDeL commit or pull request safely. Use for staging, pre-commit, import layering, focused test selection, PR summaries, branch hygiene, release-policy checks, subtree mirror warnings, and enforcing the no self-credit trailer rule.
---

# Skill: Prepare Commit Or PR

Use this for commit/PR hygiene. For code review, load
`.claude/skills/review-pr/SKILL.md`; for releases, load
`.claude/skills/release-workspace/SKILL.md`.

## First Reads

- `WORKSPACE.md`
- `.pre-commit-config.yaml`
- `scripts/release.sh`
- `scripts/publish.sh`
- `scripts/subtree-sync.sh`
- `git status --short`
- `git diff --stat`

## Worktree Discipline

- Do not stage unrelated user changes.
- If a file has user edits plus your edits, inspect the diff carefully and
  stage only the intended hunks.
- Never use destructive reset or checkout commands unless the user explicitly
  asks for that operation.
- Do not create or restore `CLAUDE.md` files for this repo unless the user
  reverses the current instruction.

## Checks

Run focused tests selected through `.claude/skills/test-workspace/SKILL.md` for
the touched package. Also run:

```bash
uv run lint-imports
uv run pre-commit run --all-files
```

Pre-commit hooks may auto-fix and return `Failed`. When that happens, review the
modifications, restage intended files, and rerun. Hook entries should use direct
repo commands as configured; do not add `uv run` inside hook definitions.

## Release And Version Policy

Version bumps go through:

```bash
scripts/release.sh <easydel|spectrax|ejkernel|eformer> <new-version> [--dry-run]
```

Do not hand-edit version pins or package dependency pins for a release. Use
`.claude/skills/release-workspace/SKILL.md` for the full release flow.

Foundation packages mirror to standalone repos via the pre-push hook:

```bash
scripts/subtree-sync.sh auto
```

`SUBTREE_SYNC_SKIP=1 git push ...` bypasses mirror sync when the user knowingly
wants that path.

## Commit And PR Text

Do not add self-credit trailers or generated-by lines in commits, PRs, tags, or
release notes. Forbidden examples include `Co-Authored-By` and
`Generated with`.

PR summaries should cover:

- changed package surfaces
- behavioral changes
- tests run
- skipped hardware or end-to-end checks
- release or migration notes when relevant

Keep PR claims aligned with verification. Do not claim TPU, eSurge, checkpoint,
or benchmark validation unless that path actually ran.
