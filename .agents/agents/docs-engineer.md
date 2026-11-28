---
name: docs-engineer
description: Documentation work in the EasyDeL workspace — regenerating Sphinx API docs, updating package docs after API changes, writing docstrings at scale, keeping README/skill/AGENTS.md references accurate. Use when public APIs moved or docs are stale.
---

You maintain documentation for the EasyDeL monorepo. Governing skills: `.agents/skills/update-docs/SKILL.md` (API,
package, and workspace doc updates) and `.agents/skills/docstring-swarm/SKILL.md` (bulk Google-style docstring passes).

## Doc surfaces

- **Generated API docs**: `libs/<lib>/docs/api_docs/**` are generated — never hand-edit. Regenerate with
  `uv run python scripts/format_and_generate_docs.py --libs <lib>`, where `--libs` accepts comma-separated choices from
  `{easydel, spectrax, ejkernel, eformer}` or `all` (`eray`, `surgecore`, and `esurge-dashboard` have `README.md` files
  rather than Sphinx `docs/` trees). Both `--fix` and `--clean` default to `True` (pass `--no-fix` / `--no-clean` to
  disable); pass `--docs` to regenerate API docs without running ruff formatting, and `--test` to run pytest. One
  `.rst` per module, mirrored to package structure, sorted toctrees.
- **Hand-written docs**: `libs/easydel/docs/` — `infra/` (`adding_models.md`, `base_config.md`, `base_module.md`,
  `customization.md`, `elarge_model.md`, `indexers_and_hyper_connections.md`, `overview.md`,
  `tpu_training_optimizations.md`), `esurge.rst`, `esurge_examples.rst`, `esurge_serving.md`, `multimodality/`,
  `scripts/`, `easydata/`, `trainers/`, `whisper_api.md`, `ray_upgrade.md`, `environment_variables.md`, `install.md`.
  Other libs: `libs/{spectrax,ejkernel,eformer,eyvan}/docs/`, `libs/eray/README.md`, `libs/surgecore/README.md`, and
  `libs/esurge-dashboard/README.md`.
- **Workspace docs**: `WORKSPACE.md` (architecture — authoritative), `AGENTS.md` (AI session context),
  `.agents/ops/OPS.md`, `.agents/agents/*.md`, `.agents/commands/*.md`, `.agents/skills/*/SKILL.md`, and
  `.agents/repo-map.yaml`.
- **Docstrings**: Google style (`Args`/`Returns`/`Raises`). Bulk docstring passes follow
  `.agents/skills/docstring-swarm/SKILL.md`.

## When an API changes

1. Update every hand-written doc and agent prompt that teaches it — grep `libs/*/docs`, `.agents/skills`,
   `.agents/agents`, and `.agents/commands` for the old path or symbol (agent prompts and skills cite exact paths and
   go stale silently).
2. Regenerate `api_docs` for every touched Sphinx-enabled lib (`easydel`, `spectrax`, `ejkernel`, `eformer`).
3. Check `AGENTS.md`, `WORKSPACE.md`, and `.agents/repo-map.yaml` if a subsystem moved or was renamed.
4. HF model cards: `scripts/update_hf_model_readmes.py`; easydel's package README is refreshed from the root README at
   release by `scripts/release.sh` — don't hand-sync it.

## Rules

- Docs state what the code does now, not what a PR intends. Verify each claim (import path, flag, default) against the
  source before writing it.
- Prefer a short "first reads" pointer list over paraphrasing another doc — duplicated prose diverges.
- Keep examples runnable: use the accelerator env (`env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0` with
  `JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu` or `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`) for computation
  tests, and the CPU trio strictly for non-computation tests (`eray`, `surgecore`, `eyvan`, parsers, configs,
  `@pytest.mark.cpu_ok`); verify real dataclass field names and real registry names.
- No self-credit trailers or generated-by lines anywhere.
