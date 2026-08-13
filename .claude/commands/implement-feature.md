---
description: Implement a planned or described feature in the EasyDeL workspace, end to end with tests
argument-hint: <feature or plan reference>
---

Implement in the EasyDeL monorepo: $ARGUMENTS

Follow the implementation flow from CLAUDE.md:

1. Load `.claude/skills/run-research/SKILL.md`, then the domain skill that matches the surface (add-easydel-model /
   add-easydel-trainer / add-ejkernel-kernel + port-ejkernel-to-easydel-operation / add-easydel-operation /
   add-easydel-layer / build-dataset-pipeline / tool-reasoning-parser / add-eformer-optimizer). Follow its "required
   surfaces" list exactly — registrations, exports, tests, docs.
2. If no plan exists yet and the change spans subsystems, produce a short plan first (see /plan-feature) before editing.
3. Implement the smallest complete change. Match surrounding code style (tp/jnp/spx aliases, ruff line length 121,
   Google docstrings). Never bypass a registry or hand-edit generated api_docs or version pins.
4. Write the tests the skill requires (patterns:
   `.claude/skills/generate-tests/SKILL.md`), then verify:

   ```bash
   ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
   XLA_FLAGS=--xla_force_host_platform_device_count=8 \
     uv run pytest <focused targets>
   uv run lint-imports
   ```

5. Report files changed, verification commands with outcomes, and remaining risk (especially anything only validatable
   on TPU/GPU). Do not claim training/serving readiness from constructor-only checks, and do not commit unless asked —
   and never with self-credit trailers.
