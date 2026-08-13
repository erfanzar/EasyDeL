---
description: Explain how an EasyDeL workspace subsystem works, grounded in the current code
argument-hint: <subsystem, e.g. "eSurge scheduler" or "fused layouts">
---

Explain the following subsystem of the EasyDeL monorepo: $ARGUMENTS

Ground rules:

1. Start from `.claude/repo-map.yaml` and CLAUDE.md's "Where things live"
   table to locate the subsystem, then read the actual source before explaining — the explanation must reflect the code
   as it is today, not general knowledge about similar systems.
2. Structure the answer as: (a) what it owns and where it sits in the layering, (b) the main flow through it with
   concrete class/function names and clickable file paths, (c) its extension points and the skill that covers them, (d)
   the gotchas that bite newcomers (verified ones only).
3. Prefer one worked example over exhaustive enumeration — e.g., trace one request through the eSurge scheduler, one
   weight tensor through HF conversion, one microbatch through a 1F1B schedule.
4. If the subsystem has a dedicated doc (`libs/*/docs/**`) or skill, cite it rather than paraphrasing it wholesale.
5. Flag anything you found that contradicts existing docs — that is a finding worth reporting, not a detail to silently
   reconcile.
