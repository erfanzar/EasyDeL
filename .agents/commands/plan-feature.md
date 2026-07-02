---
description: Design an implementation plan for an EasyDeL feature before any code is written
argument-hint: <feature description>
---

Plan the following feature for the EasyDeL monorepo: $ARGUMENTS

Act as the `architect` agent (`.agents/agents/architect.md`). Read
WORKSPACE.md, CLAUDE.md, and `.agents/repo-map.yaml`, then inspect the real
code surfaces the feature touches (ground every path with rg or by opening
it — no guessed file names).

Deliver:
1. Goal in one sentence and the chosen extension point(s) — registry entry,
   mixin hook, config surface, or new module following an existing pattern.
2. Package ownership decision (foundation lib vs easydel) with the layering
   contract respected.
3. Ordered implementation steps, each listing files, registrations, and the
   test it must add. Name the matching skill for each step
   (add-easydel-model, add-easydel-trainer, add-ejkernel-kernel, ...).
4. Domain experts to consult before coding (sharding/quantization/inference
   invariants).
5. Verification ladder: exact pytest targets under the CPU env trio, plus
   any hardware-bound validation that must be scheduled or reported as
   unverified.
6. Risks: sharding, recompilation, checkpoint compatibility, HF conversion.

Do not write implementation code. If the feature is ambiguous, state the
interpretation you chose and why.
