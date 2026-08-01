---
description: Prepare (and optionally publish) a release of a workspace package
argument-hint: <package> <version>, e.g. "ejkernel 0.0.83"
---

Prepare a release for: $ARGUMENTS

Follow `.claude/skills/release-workspace/SKILL.md` as the governing skill.
The flow is two explicit stages — nothing leaves the machine in stage 1:

1. **Pre-flight**: clean working tree for the released package; changelog/
   version sanity; affected tests green under the CPU env trio;
   `uv run lint-imports` passes.
2. **Release (local)**: `scripts/release.sh <lib> <version>` — bumps the
   version, syncs easydel's pins on sibling libs, refreshes locks, commits
   locally. Never hand-edit version pins.
3. **Publish (outward-facing — confirm with the user before running)**:
   `scripts/publish.sh <lib>` tags `<lib>-v<version>` and pushes the tag;
   CI (`.github/workflows/publish.yaml`) builds and publishes to PyPI.
   Push the branch separately with `git push` (pre-push hook subtree-syncs
   mirrors; `SUBTREE_SYNC_SKIP=1` to bypass on mirror outage).

Rules: no self-credit trailers in the release commit or notes; report
exactly which stage was completed and what remains; if any check failed,
stop and report rather than proceeding.
