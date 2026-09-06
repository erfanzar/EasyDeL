---
description: High-signal review of the current working-tree diff or a branch/PR in the EasyDeL workspace
argument-hint: [PR number | branch | empty for working tree]
---

Review: $ARGUMENTS (if empty, review the uncommitted working-tree diff against HEAD; if a merge is pending, diff against
the merge-base with main).

Follow `.xerxes/skills/review-pr/SKILL.md` as the governing process and the
`reviewer` agent knowledge (`.xerxes/agents/reviewer.md`) for repo-specific hazards. If the diff touches
Mesh/PartitionSpec/shard_map/ pipeline stages, additionally run the checklist from
`.xerxes/skills/distributed-review/SKILL.md`.

Ground rules:

- Findings must be real bugs, broken contracts, boundary violations, or testing-policy violations — cite file:line and
  the rule or failure path.
- Validate every finding against the actual code before reporting it.
- Run the cheapest relevant verification (`uv run lint-imports`, focused pytest under the CPU env trio).
- No style nits, no speculative concerns, no lint-catchable issues.

Output findings ordered by severity with the smallest credible fix direction each. If clean, say so and name residual
risk (e.g., TPU paths unverified).
