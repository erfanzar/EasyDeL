---
description: Root-cause a failure in the EasyDeL workspace with hypothesis-driven debugging
argument-hint: <symptom, failing test, or error message>
---

Investigate: $ARGUMENTS

Act as the `debugger` agent (`.xerxes/agents/debugger.md`). Before treating the symptom as a code bug, check
`.xerxes/ops/OPS.md` for infrastructure routes (TPU bad nodes, libtpu locks, disk pressure) — a matching route wins.

Process:

1. Reproduce with the smallest probe on the accelerator, using tiny configs (`libs/easydel/tests/modules/conftest.py`
   fixtures); the CPU trio only for non-computation logic (parsers, config/CLI, eray). If no accelerator is available,
   say so explicitly and report the result as unverified on hardware — never substitute a CPU run.
2. State a falsifiable hypothesis; run the one experiment that tests it; iterate. One variable per experiment.
3. Use the domain triage routes: NaN → softmax dtype → loss scale → kernel parity (`FORCE_NATIVE_RUNTIME=1`); sharding
   errors → active mesh + spec divisibility + fused-layout splitters; silently-lost state →
   `spx.jit(mutable=...)`; eSurge corruption → `PageTable.commit()` + DP page locality; suspicious "fixes" → clear
   `~/ejkernel-presistent-cache/` (override: `EJKERNEL_PERSISTENT_CACHE_DIR`) first.
4. For hunts longer than one session, keep
   `.xerxes/projects/<topic>.md` updated with hypotheses and negative results.

Deliver: root cause with evidence, the minimal fix (only if the user asked for a fix — otherwise report findings), and
the focused test that now covers the failure. Report failing output verbatim; never soften results.
