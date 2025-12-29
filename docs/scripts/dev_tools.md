# Development Tools

Scripts for code formatting and documentation generation.

## Script

```bash
python scripts/format_and_generate_docs.py --help
```

## What It Does

1. **Format code** — runs `ruff check --fix` and `ruff format` on Python files
2. **Generate API docs** — creates RST files for Sphinx from module docstrings

Used by pre-commit hooks and CI.

## Basic Usage

```bash
# Run all tasks (format + docs)
python scripts/format_and_generate_docs.py --all

# Or with no flags (same as --all)
python scripts/format_and_generate_docs.py

# Format only
python scripts/format_and_generate_docs.py --format

# Generate docs only
python scripts/format_and_generate_docs.py --docs

# Run tests too
python scripts/format_and_generate_docs.py --all --test
```

## Key Flags

| Flag          | Default   | Description                                                   |
| ------------- | --------- | ------------------------------------------------------------- |
| `--format`    | off       | Format code with ruff                                         |
| `--docs`      | off       | Generate API documentation                                    |
| `--test`      | off       | Run pytest                                                    |
| `--all`       | off       | Run format + docs (not test)                                  |
| `--fix`       | on        | Apply fixes automatically; use `--no-fix` for check-only      |
| `--clean`     | on        | Delete old docs before regenerating; use `--no-clean` to keep |
| `--directory` | `easydel` | Directory to format                                           |

## Typical Workflows

### Before Committing

```bash
# Format code and update docs
python scripts/format_and_generate_docs.py --all
```

### Check Without Modifying

```bash
# Check formatting only (CI mode)
python scripts/format_and_generate_docs.py --format --no-fix
```

### Regenerate All API Docs

```bash
# Clean and regenerate
python scripts/format_and_generate_docs.py --docs --clean
```

### Format Specific Directory

```bash
python scripts/format_and_generate_docs.py --format --directory easydel/trainers
```

## Output Locations

| Output      | Location         |
| ----------- | ---------------- |
| API docs    | `docs/api_docs/` |
| Ruff config | `pyproject.toml` |

## Pre-commit Integration

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: format-and-docs
        name: Format and generate docs
        entry: python scripts/format_and_generate_docs.py --all
        language: system
        pass_filenames: false
```
