# Agent Repository Guidance

Start with [AGENTS.md](../AGENTS.md) and [WORKSPACE.md](../WORKSPACE.md) at the repository root. This directory holds
the EasyDeL monorepo's harness-agnostic instructions, playbooks, runbooks, and project notes.

## Directory Layout and Composition

- `.agents/repo-map.yaml` — machine-readable map of workspace packages, subsystems, CI jobs, and ownership routing.
- `.agents/ops/OPS.md` — infrastructure and operational runbooks (TPU setup/bad-node recovery, `eray` CLI, GPU triage,
  stale ejkernel caches, eSurge triage, disk pressure). Check this before treating infra symptoms as code bugs.
- `.agents/agents/` — specialist role instructions (17 agents: architecture, review, debugging, testing, performance,
  sharding, quantization, inference, kernels, models, trainers, RLHF, data, JAX, TPU, GPU, docs).
- `.agents/skills/` — reusable domain and workflow playbooks (44 skills; index in
  [.agents/skills/README.md](skills/README.md)). Start non-trivial work from `.agents/skills/run-research/SKILL.md` and
  layer the matching domain skill.
- `.agents/commands/` — end-to-end task playbooks (planning, implementation, debugging, review, subsystem explanation,
  architecture summary, releases) that compose `.agents/agents/` and `.agents/skills/`.
- `.agents/projects/` — stateful multi-session research and debugging logs (index and lifecycle rules in
  [.agents/projects/README.md](projects/README.md)).

## Harness-Agnostic Usage

Read and follow these documents directly. Files in `.agents/` do not register tools, subagents, or slash commands on
their own; automatic discovery and delegation depend on the active agent harness. Use whatever tools and subagent
mechanisms are available in the current session.

## Maintenance Contract

- When adding, renaming, or removing a workspace package, skill, or agent, update `.agents/repo-map.yaml`,
  `.agents/skills/README.md`, `AGENTS.md`, and `WORKSPACE.md` in the same change.
- Ground every referenced file path, symbol, flag, and env var against the live tree (repo-root-relative paths).
- Promote durable operational lessons out of `.agents/projects/` into `.agents/ops/OPS.md`, and promote durable
  architectural invariants into `AGENTS.md`, `.agents/skills/`, or code docstrings.
