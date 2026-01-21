# Repository Guidelines

## Project Structure & Module Organization

- `easydel/` holds the core library (models, trainers, data pipelines, serving).
- `tests/` contains pytest suites; add new coverage here.
- `docs/` contains Sphinx/Markdown docs; generated API docs live in `docs/api_docs/`.
- `scripts/` provides developer tooling (formatting, docs generation, conversions, docker helpers).
- `examples/` are runnable scripts; `tutorials/` are longer walkthroughs.
- `docker/`, `docker-compose*.yml`, and `autoscale/` cover container and Ray/TPU setups.
- `images/` and `dist/` store assets and packaged artifacts.

## Build, Test, and Development Commands

- `python scripts/format_and_generate_docs.py --all` formats Python (ruff) and regenerates API docs.
- `python scripts/format_and_generate_docs.py --docs` regenerates API docs only.
- `python scripts/format_and_generate_docs.py --all --test` runs formatting, docs, and pytest.
- `pytest` runs the test suite (defaults to `tests/`).
- `make -C docs html` builds Sphinx docs locally into `docs/_build/`.
- `python examples/esurge_metrics_example.py` runs a local example; see `examples/` for more.

## Coding Style & Naming Conventions

- Python 3.11+; 4-space indentation.
- Formatting/linting uses ruff (`ruff format`, `ruff check --fix`) with line length 121 (`pyproject.toml`).
- Use `snake_case` for modules/files; test files should be named `test_*.py`.

## Testing Guidelines

- Framework: pytest; tests live in `tests/`.
- Mark long-running tests with `@pytest.mark.slow`; run fast tests via `pytest -m "not slow"`.

## Commit & Pull Request Guidelines

- Commit history largely follows Conventional Commits: `feat:`, `fix:`, `chore:`, `refactor:`.
- Keep messages short and imperative; avoid bundling unrelated changes.
- PRs should address a single issue/feature, include a clear description, and link related issues.

## Configuration & Environment

- Runtime flags for JAX, TPU, and performance tuning are documented in `docs/environment_variables.md`.
