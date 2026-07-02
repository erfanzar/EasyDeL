# AGENTS.md — EasyDeL Stack

Briefing for AI agents working in this repository. Read
[CLAUDE.md](CLAUDE.md) for conventions and commands, and
[WORKSPACE.md](WORKSPACE.md) for the authoritative architecture. Non-trivial
work starts from `.agents/skills/run-research/SKILL.md` and layers a domain
skill on top; operational failures route through `.agents/ops/OPS.md`.

Hard rules that apply to every agent: the layering contract (foundation libs
never import easydel or each other), the CPU-test env trio, no self-credit
trailers in commits/PRs, no hand-edited version pins, and no claims about
scripts/paths/flags that were not verified in the repo.

## Available agents (`.agents/agents/`)

| agent | use when | scope |
| ----- | -------- | ----- |
| `architect` | designing a feature or refactor that spans more than one subsystem or package | decomposes work, names extension points, flags layering/API impact; does not implement |
| `reviewer` | a diff/PR needs a correctness review before merge | bugs, broken contracts, boundary violations, testing-policy violations; no style nits |
| `test-engineer` | choosing what to run, or writing tests for new behavior | test selection per package, repo-consistent test authoring, quality bar enforcement |
| `debugger` | a failure with an unknown cause | hypothesis → smallest falsifying probe → fix; keeps a paper trail |
| `perf-engineer` | a change might affect speed/memory, or a perf claim needs proof | compile cost, memory, communication, throughput; baseline-vs-candidate discipline |
| `docs-engineer` | public APIs moved, docs/docstrings need generation or repair | Sphinx api_docs regeneration, package docs, README conventions |
| `jax-expert` | tracing/transform semantics, spx state, donation, jit cache questions | JAX + spectrax transform correctness |
| `tpu-expert` | TPU-specific behavior: Pallas/Mosaic, pods, libtpu locks, multi-host | TPU execution + eray/gcloud operations |
| `sharding-expert` | anything touching Mesh/PartitionSpec/axis policies/PP stages | escale + infra sharding + spectrax MPMD/SPMD pipelines |
| `kernel-expert` | writing/porting/fixing a compute kernel | ejkernel modules/backends/registry + easydel operation adapters |
| `trainer-expert` | trainer plumbing, SFT/preference/distillation config or loss work | trainers/ foundation, TrainingArguments, loss math |
| `rlhf-expert` | online RL: GRPO family, PPO, rewards, rollouts | policy-gradient trainers, RewardProtocol, eSurge rollout integration |
| `inference-expert` | eSurge engine, API server, parsers, speculative decoding | inference/ subsystem end to end |
| `quantization-expert` | quantized weights/caches/matmuls, fused layouts, STE training | layers/quantization, layouts, eformer jaximus/mpric, ejkernel quantization |
| `model-expert` | adding/porting a model family or fixing HF conversion | modules/, caching/, parameters_transformation, auto classes |

## Choosing and combining

- Prefer the **narrowest agent that owns the failing surface**. A sharding
  error inside a trainer step is `sharding-expert` first, `trainer-expert`
  second.
- Domain experts are consulted **before** implementation when the change
  touches their invariants (sharding-expert for anything with a
  PartitionSpec; quantization-expert for anything with a fused/quantized
  layout; tpu-expert before interpreting TPU symptoms as code bugs).
- Fan out only when surfaces are independent (e.g., `kernel-expert` on the
  ejkernel impl while `test-engineer` drafts the parity test). Never two
  agents editing the same files.

## Implementation flow

1. **Scope** — `architect` (or the main session) states the goal, the touched
   packages, and the extension point; loads the matching skill from
   `.agents/skills/` (`add-easydel-model`, `add-easydel-trainer`,
   `add-ejkernel-kernel`, `build-dataset-pipeline`, ...).
2. **Consult** — pull in the domain expert(s) whose invariants the plan
   touches; adjust the plan, don't discover violations after coding.
3. **Implement** — smallest change that tests the hypothesis; follow the
   skill's "required surfaces" list (registrations, tests, docs, exports).
4. **Verify** — `test-engineer` selects the focused tests (CPU trio), then
   `uv run lint-imports` and pre-commit. Hardware-bound claims go to
   `tpu-expert`/`perf-engineer` or are explicitly reported as unverified.
5. **Review** — `reviewer` on the final diff (flow below).

## Review flow

`reviewer` follows `.agents/skills/review-pr/SKILL.md`: gate (skip
closed/draft/trivial) → context (WORKSPACE.md + touched package docs) →
parallel compliance and bug passes → independent validation of every finding
before it is reported. Findings must cite file/line and a concrete rule or
failure path; style preferences and speculative concerns are dropped.
Escalate a finding to a domain expert instead of guessing: numerics →
kernel-expert or jax-expert; partition specs → sharding-expert; cache shapes
or bucket compilation → inference-expert.

## Escalation rules

- **CPU-unverifiable claims** (Pallas lowering, TPU perf, eSurge runtime on
  device): stop at "unverified on hardware" or hand to `tpu-expert` /
  `perf-engineer` with a run plan. Never extrapolate CPU timings.
- **Infrastructure symptoms** (bad TPU node, libtpu lock, disk pressure):
  route to `.agents/ops/OPS.md` before treating as a code bug; destructive
  recovery (deleting TPU VMs) requires explicit user approval.
- **Cross-package API changes** (easydel needs a new spectrax/eformer/
  ejkernel surface): `architect` decides which side owns the change; the
  foundation lib gets the general mechanism, easydel gets the composition.
  Never solve it by violating the import contract.
- **Performance regressions found during review**: block only with a
  measured or clearly-reasoned regression path (`perf-engineer`), otherwise
  note-and-pass.
- **Anything touching release/publish/mirrors**: follow `release-workspace`
  and `prepare-commit-pr` skills; tags and pushes are user-approved actions.
