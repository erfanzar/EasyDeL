---
name: review-pr
description: Multi-agent, high-signal correctness review for EasyDeL pull requests or branch diffs. Use when asked to review a PR, review uncommitted changes, or audit a branch for bugs, package-boundary violations, testing-policy violations, or release/commit hygiene before merge.
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git show:*), Bash(git merge-base:*), Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh pr comment:*), Bash(gh pr list:*), Bash(uv run lint-imports:*), Bash(uv run pytest:*), Bash(uv run pre-commit:*), Task
---

# Skill: Review PR

Provide a high-signal review. Findings must be bugs, broken contracts,
package-boundary violations, or concrete testing-policy violations. Avoid
style nits and speculative concerns.

## Required Context

1. Read `WORKSPACE.md`.
2. Identify changed files with `git diff --name-only` or `gh pr diff --name-only`.
3. Read the touched package's `pyproject.toml` plus relevant docs under
   `libs/<package>/docs/`.
4. For testing-policy questions, read `.claude/skills/test-workspace/SKILL.md`.
5. For operational changes, read `.claude/ops/OPS.md`.

## Early Stop

For a GitHub PR, first check:

```bash
gh pr view <PR> --json state,isDraft,title,body,author,comments
```

Stop if the PR is closed, draft, clearly trivial, or already reviewed by an
agent and no re-review was requested. Still review agent-authored PRs when the
user asks.

## Review Fan-Out

Use subagents when the harness supports them:

1. Gate agent: check early-stop conditions and summarize PR intent.
2. Context agent: return only the relevant `WORKSPACE.md`, package
   `pyproject.toml`, package docs, test-workspace skill, and ops docs for
   changed paths.
3. Two compliance agents: check changed files against `WORKSPACE.md`, package
   docs, test-workspace policy, and release/commit policy.
4. Two bug agents: inspect the diff for compile failures, missing imports,
   incorrect logic, bad shape/sharding/cache assumptions, or clear runtime
   regressions.
5. Validation agents: independently confirm every proposed issue before it is
   reported.

If subagents are unavailable, run the same passes manually and keep separate
notes for compliance vs bugs.

## What To Flag

Flag only issues you can ground in code or docs:

- Foundation libs importing `easydel` or each other.
- `easydel` changes that bypass existing registry, sharding, cache, operation,
  or conversion APIs.
- Kernel changes without an XLA/simple reference or backend parity path.
- eSurge changes that reuse stale cache shapes, accidentally enable MTP, break
  DP/KV-page placement, or claim a fix without an affected-path check.
- Tests that assert private state, helper dispatch, incidental strings, or
  tautologies instead of behavior.
- Manual version/pin edits that should go through `scripts/release.sh`.
- Commit/PR/release text with self-credit trailers or generated-by lines.

Do not flag:

- Mere style preferences.
- Missing broad coverage unless a specific changed behavior lacks any
  observable check.
- Performance concerns without a clear regression path.
- Issues a linter will trivially catch, unless the user asked for lint review.

## Verification

Run the cheapest relevant command after identifying likely issues:

```bash
uv run lint-imports
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <focused-test-path>
```

Do not use TPU for review verification unless the PR's claim depends on TPU and
the TPU is available to this process.

## Output

Lead with findings ordered by severity. Each finding needs:

- file and line
- why it is a real bug or contract violation
- the relevant rule or code path
- the smallest credible fix direction

If there are no findings, say so clearly and name any residual risk, such as
hardware checks not run.
