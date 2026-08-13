---
name: update-docs
description: Update EasyDeL workspace documentation when APIs change — regenerate Sphinx api_docs, fix hand-written package docs, keep CLAUDE.md/WORKSPACE.md/skill references accurate. Use after renaming/moving/adding public APIs or when docs are reported stale. For bulk docstring passes use docstring-swarm instead.
---

# Skill: Update Docs

## Doc Surfaces

- **Generated** (never hand-edit): `libs/<lib>/docs/api_docs/**`. Regenerate:

  ```bash
  uv run python scripts/format_and_generate_docs.py --libs <lib>   # also ruff-fixes/formats
  ```

  Flags: `--libs easydel,spectrax,ejkernel,eformer` or `all`; `--clean`
  rebuilds; `--test` runs pytest after.
- **Hand-written**: `libs/easydel/docs/` — `infra/` (adding_models, base_config, base_module, elarge_model),
  `esurge.rst` +
  `esurge_examples.rst`, `easydata/`, `trainers/`,
  `environment_variables.md`; other libs under `libs/<lib>/docs/`.
- **Workspace-level**: `WORKSPACE.md` (architecture, authoritative),
  `CLAUDE.md`, `CLAUDE.md`, `.claude/ops/OPS.md`,
  `.claude/skills/*/SKILL.md`, `.claude/repo-map.yaml`.

## Workflow

1. Identify the renamed/moved/added symbols from the diff.
2. Grep every doc surface for the old names — including `.claude/skills`
   and `.claude/agents`, which cite exact paths and symbols and go stale silently:

   ```bash
   rg "<old_symbol>|<old_path>" libs/*/docs WORKSPACE.md CLAUDE.md CLAUDE.md .agents
   ```

3. Fix hand-written docs first (verify each claim against the source:
   import path, default value, flag), then regenerate api_docs for touched libs.
4. HF model cards via `scripts/update_hf_model_readmes.py` when model READMEs are in scope. Do not hand-sync
   `libs/easydel/README.md` — it is refreshed from the root README by `scripts/release.sh`.

## Rules

- Docs state what the code does now; verify before writing.
- Prefer "first reads" pointer lists over paraphrasing other docs.
- Examples must be runnable (CPU env trio for tests, real registry names, real dataclass fields).
- No self-credit trailers in any committed text.
