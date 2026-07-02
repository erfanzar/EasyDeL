---
name: release-workspace
description: Release or publish one of the EasyDeL uv workspace packages. Use for version bumps, dry-run releases, publish tags, standalone repo subtree mirror sync, package dependency pins, or pre-push release hygiene for easydel, spectrax, ejkernel, or eformer.
---

# Skill: Release A Workspace Package

Use this when the user asks for a package release, version bump, publish, or
mirror-sync preparation.

## First Reads

- `WORKSPACE.md`
- `scripts/release.sh`
- `scripts/publish.sh`
- `scripts/subtree-sync.sh`
- `.pre-commit-config.yaml`
- target package `libs/<package>/pyproject.toml`

## Version Bump Rule

Use the release script. Do not hand-edit package versions or workspace pins.

```bash
scripts/release.sh <easydel|spectrax|ejkernel|eformer> <new-version> --dry-run
scripts/release.sh <easydel|spectrax|ejkernel|eformer> <new-version>
```

The script updates the target package version, updates dependent pins when
needed, refreshes the lockfile, stages release files, and creates a local commit.

## Publish Rule

Only publish when the user explicitly asks for publishing. Use:

```bash
scripts/publish.sh <easydel|spectrax|ejkernel|eformer> --dry-run
scripts/publish.sh <easydel|spectrax|ejkernel|eformer>
```

The publish script creates and pushes the package tag in the form
`<package>-v<version>`. Do not invent manual tags.

## Mirror Sync

Foundation packages mirror to standalone repositories via:

```bash
scripts/subtree-sync.sh auto
```

The pre-push hook runs subtree sync. `SUBTREE_SYNC_SKIP=1 git push ...` bypasses
the hook when the user knowingly wants to skip mirror sync.

## Verification

Before a release commit or publish, run:

```bash
uv run lint-imports
uv run pre-commit run --all-files
```

Run package-focused tests selected through `.agents/skills/test-workspace/SKILL.md`
when the release includes code changes beyond version metadata.

## Text Policy

Do not include self-credit trailers or generated-by lines in commits, tags,
release notes, or PR text. Keep release notes about package behavior and user
impact only.
