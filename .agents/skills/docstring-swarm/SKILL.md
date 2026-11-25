---
name: docstring-swarm
description: "Launch parallel workers to add and repair Google-style Sphinx-Napoleon docstrings across an EasyDeL workspace package — module, class, and function/method docstrings (public and private), expanding `**kwargs: Unpacked[TypedDict]` fields. Docstring-only diffs; never adds standalone variable/attribute docstrings or changes behavior."
---

# Skill: Docstring Swarm

Document or repair docstrings at scale across the EasyDeL workspace by fanning out parallel workers over disjoint file
batches. Every run obeys two invariants: **accuracy** (derived from reading the implementation, never guessed) and
**safety** (zero runtime behavior or code changes — docstrings only).

## EasyDeL Conventions

- **Style**: Google style parsed by `sphinx.ext.napoleon` (`libs/easydel/docs/conf.py`). Use reST **double backticks**
  (`` ``symbol`` ``) for inline code/values and `:class:` / `:func:` cross-references — never Markdown single backticks
  (Sphinx renders single backticks as default-role italics or broken refs).
- **Line width & ASCII**: `121` columns across all workspace `pyproject.toml` files. `libs/eformer/pyproject.toml` and
  `libs/eray/pyproject.toml` enforce BOTH `E501` (>121 cols) and `RUF002` (forbids ambiguous Unicode like em-dash,
  en-dash, multiplication sign, or curly quotes inside docstrings) — stick to plain ASCII (`--`, `-`, `x`, `"..."`).
- **Scope & skips**:
  - Skip `libs/*/docs/api_docs/**` (generated `.rst`), `.venv/`, `target/`, and `__pycache__/`. Tests are opt-in unless
    requested.
  - Preserve quoted `jaxtyping` annotations in `**/modeling_*.py` without altering code lines.
  - `libs/surgecore/`: Rust sources (`src/**/*.rs`) use `///` / `//!` rustdoc comments; Python stubs live in
    `libs/surgecore/surgecore/_native.pyi`.

## Hard Rules

1. **Docstring-only diffs**: never alter logic, signatures, imports, type annotations, decorators, or code formatting.
   Report suspected bugs in the final summary without fixing them.
2. **Never add standalone variable/attribute string-literal docstrings** (forbidden at module, class, and function
   scope):
   ```python
   LEARNING_RATE = 3e-4
   """Default learning rate."""  # FORBIDDEN -- put constants in module docstring or class Attributes:
   ```
3. **Document all in-scope units**: top-of-file module docstring, every class (public & private, documenting notable
   attributes under `Attributes:`), and every function/method (public, private, nested, overloads) with `Args:` /
   `Keyword Args:` (every param, type, default, constraint, `*args`/`**kwargs`), `Returns:` / `Yields:` (unpacking
   tuples/dicts), and `Raises:` (exception + trigger).
4. **Repair drift, minimize churn**: read each body first; fix stale/renamed/removed params and wrong return types, but
   leave already-accurate docstrings untouched.
5. **Expand `**kwargs: Unpacked[TypedDict]` (PEP 692)**: open the TypedDict (e.g. `EasyDeLBaseConfigDict`,
   `BaseTrainerCfg`), list each field under `Keyword Args:` noting it flows through `**kwargs`, and reflect any
   return-affecting fields in `Returns:`.

## Google / Sphinx-Napoleon Template

```python
"""Short summary of module purpose and main exported symbols."""


class StepConfig(TypedDict, total=False):
    clip_norm: float | None
    return_aux: bool


def apply_step(state: EasyDeLState, grads: PyTree, **kwargs: Unpacked[StepConfig]) -> EasyDeLState:
    """Return a new state with ``grads`` applied to ``state``.

    Args:
        state (EasyDeLState): Current immutable training state.
        grads (PyTree): Gradient pytree matching ``state.graphstate``.

    Keyword Args:
        clip_norm (float | None): Global gradient clip norm, or ``None`` to disable.
            Passed via ``**kwargs`` (:class:`StepConfig`).
        return_aux (bool): When ``True``, attach auxiliary norm metrics.
            Passed via ``**kwargs`` (:class:`StepConfig`).

    Returns:
        EasyDeLState: Updated training state with incremented step counter.

    Raises:
        ValueError: If ``clip_norm`` is non-positive.
    """
```

## Orchestration

1. **Partition**: enumerate in-scope files and split into disjoint per-worker batches by directory/module (~5–15 files
   per batch, fewer for large files). Never assign the same file to two workers.
2. **Fan out**: launch parallel subagents/workers, one per batch, handing each worker its exact file list, EasyDeL's
   121-col Google/Sphinx-Napoleon + ASCII rules, and the Hard Rules above. Require each worker to parse-check its edited
   files and return per-file counts (`documented` / `updated` / `unchanged`) plus any suspected bugs found (unfixed).

## Verification Gate

```bash
# 1. Syntax gate over existing modified .py files only
uv run python -c 'import ast, pathlib, subprocess
files = [p for p in subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=ACMR"], text=True).splitlines() if p.endswith(".py")]
for f in files: ast.parse(pathlib.Path(f).read_text(encoding="utf-8"), filename=f)
print(f"Parsed {len(files)} Python files cleanly.")'

# 2. Per-library ruff lint check (catches E501 / RUF002 in eformer & eray)
uv run ruff check --config libs/<lib>/pyproject.toml <modified-py-files>

# 3. Optional: regenerate Sphinx API docs if module docstrings or modules changed
uv run python scripts/format_and_generate_docs.py --libs <lib> --docs
```

- **Diff-shape check**: inspect `git diff` to confirm ONLY docstring literals changed and zero standalone variable
  docstrings were introduced; revert any accidental code edits.
- **Accuracy sample**: spot-check a random sample of modified files against their implementations for accurate
  signatures, `Unpacked[...]` expansion, and Sphinx double backticks before reporting done.
