---
name: update-docs
description: Update EasyDeL workspace documentation when APIs change — regenerate Sphinx api_docs, fix hand-written package docs, keep AGENTS.md/WORKSPACE.md/skill references accurate. Use after renaming/moving/adding public APIs or when docs are reported stale. For bulk docstring passes use docstring-swarm instead.
---

# Skill: Update Docs

## Doc Surfaces

- **Generated Sphinx API docs (never hand-edit)**: `libs/<lib>/docs/api_docs/**` for `easydel`, `spectrax`, `ejkernel`,
  and `eformer` (the 4 libraries in `scripts/format_and_generate_docs.py` `LIBS`). Regenerate docs **only**:

  ```bash
  uv run python scripts/format_and_generate_docs.py --libs <lib> --docs
  ```

  **Important flag defaults & footguns** (`scripts/format_and_generate_docs.py`):
  - `--libs` accepts `easydel`, `spectrax`, `ejkernel`, `eformer`, comma-separated lists, or `all`.
  - Both `--fix` (`ruff check --fix --unsafe-fixes`) and `--clean` (wipe + recreate `docs/api_docs/`) default to
    **`True`** (disable with `--no-fix` / `--no-clean`).
  - **Always pass `--docs`** when only regenerating API docs: omitting it runs `ruff check --fix --unsafe-fixes` and
    `ruff format` across every `.py` file in `libs/<lib>/<lib>/`.
  - Avoid `--test`: it invokes bare `pytest <test_dir> -v` without accelerator env vars (tripping `conftest.py`'s CPU
    guard on computation suites). Run tests via `.agents/skills/test-workspace/SKILL.md`.
- **Packages outside `format_and_generate_docs.py`**: `libs/eray` and `libs/surgecore` have no `docs/` tree — update
  `libs/eray/README.md`, `libs/surgecore/README.md`, and `libs/surgecore/surgecore/_native.pyi` directly. `libs/eyvan`
  manages its own Markdown docs under `libs/eyvan/docs/` (with `libs/eyvan/scripts/generate_api_reference.py` and
  `libs/eyvan/scripts/enumerate_env_surface.py`).
- **Hand-written package docs**:
  - `libs/easydel/docs/`: `infra/` (`adding_models.md`, `base_config.md`, `base_module.md`, `customization.md`,
    `elarge_model.md`, `indexers_and_hyper_connections.md`, `overview.md`, `tpu_training_optimizations.md`),
    `esurge.rst`, `esurge_examples.rst`, `esurge_serving.md`, `easydata/`, `trainers/`, `multimodality/`, `scripts/`,
    `whisper_api.md`, `environment_variables.md`.
  - Foundation libs: `libs/{spectrax,ejkernel,eformer}/docs/`.
- **Workspace & agent docs**: `WORKSPACE.md` (authoritative architecture), `AGENTS.md`, `.agents/ops/OPS.md`,
  `.agents/skills/*/SKILL.md`, `.agents/agents/*.md`, `.agents/commands/*.md`, `.agents/repo-map.yaml`.

## Workflow

1. Identify renamed, moved, or added symbols and paths from the diff.
2. Search every doc surface (including `libs/*/README.md` and `.agents/{skills,agents,commands}`):

   ```bash
   rg "<old_symbol>|<old_path>" libs/*/docs libs/*/README.md WORKSPACE.md AGENTS.md .agents
   # Fallback when rg is unavailable:
   git grep -nE "<old_symbol>|<old_path>" -- 'libs/*/docs' 'libs/*/README.md' WORKSPACE.md AGENTS.md .agents
   ```

3. Fix hand-written docs first (verify every import path, default value, and flag against live source), then regenerate
   `api_docs` with `--docs` for touched libs.
4. **HF model cards**: `scripts/update_hf_model_readmes.py` pushes directly to Hugging Face Hub via `HF_TOKEN`. Always
   preview first with `--dry-run --output-dir <dir>` and obtain explicit user approval before pushing. Never hand-sync
   `libs/easydel/README.md` (`scripts/release.sh easydel ...` copies root `README.md`).

## Rules

- Document what the code does now; verify every path and symbol before writing.
- `libs/easydel/docs/environment_variables.md` follows the **no new `os.environ` knobs** policy — only document changes
  or removals of pre-existing env vars.
- Prefer short "First Reads" pointers over duplicating prose across files.
- Keep examples runnable (accelerator env for computation tests, CPU trio only for non-computation tests, real registry
  names, real dataclass fields).
- Never add self-credit trailers in any committed text.
