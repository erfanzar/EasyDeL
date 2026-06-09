# Model Cards Script

Auto-generate and update Hugging Face model card README.md files.

## Script

```bash
python scripts/update_hf_model_readmes.py --help
```

## What It Does

1. Reads `config.json` from each HF model repo
2. Generates a README.md using `easydel/utils/readme_generator.py`
3. Optionally pushes the updated README to HF Hub

The generator infers model type, task, and attention mechanism from the config.

## Basic Usage

```bash
# Update all repos under an org
python scripts/update_hf_model_readmes.py \
  --author EasyDeL \
  --token $HF_TOKEN

# Update specific repos
python scripts/update_hf_model_readmes.py \
  --repo-id EasyDeL/Llama-3.1-8B \
  --repo-id EasyDeL/Qwen3-8B \
  --token $HF_TOKEN

# From a list file
python scripts/update_hf_model_readmes.py \
  --repos-file models.txt \
  --token $HF_TOKEN
```

## Dry Run Flow

Preview changes before pushing:

```bash
# 1. Generate READMEs locally without pushing
python scripts/update_hf_model_readmes.py \
  --author EasyDeL \
  --dry-run \
  --output-dir /tmp/readmes \
  --token $HF_TOKEN

# 2. Review generated files
ls /tmp/readmes/
cat /tmp/readmes/EasyDeL__Llama-3.1-8B.README.md

# 3. Push when satisfied
python scripts/update_hf_model_readmes.py \
  --author EasyDeL \
  --token $HF_TOKEN
```

## Key Flags

| Flag               | Default | Description                                  |
| ------------------ | ------- | -------------------------------------------- |
| `--repo-id`        | —       | Model repo id (repeatable)                   |
| `--repos-file`     | —       | File with one repo id per line               |
| `--author`         | —       | Update all repos owned by this HF user/org   |
| `--match`          | —       | Only process repos containing this substring |
| `--dry-run`        | off     | Generate but don't push                      |
| `--output-dir`     | —       | Write generated READMEs here for review      |
| `--commit-message` | auto    | Custom commit message                        |
| `--template-dir`   | —       | Custom Jinja template directory              |
| `--template-name`  | —       | Custom template filename                     |

## Template Override

To use custom templates:

```bash
python scripts/update_hf_model_readmes.py \
  --author EasyDeL \
  --template-dir /path/to/templates \
  --template-name my_readme.md.j2 \
  --token $HF_TOKEN
```

The default template is in `easydel/utils/readme_generator.py`.

## Models File Format

```text
# models.txt
EasyDeL/Llama-3.1-8B
EasyDeL/Qwen3-8B
# Comments are ignored
EasyDeL/Mistral-7B
```

## Output Format

The script reports:

- `[ok]` — README updated and pushed
- `[skip]` — No changes needed
- `[dry-run]` — Would update (dry-run mode)
- `[error]` — Failed to process

```md
[1/3] [ok] EasyDeL/Llama-3.1-8B
[2/3] [skip] EasyDeL/Qwen3-8B (no changes)
[3/3] [ok] EasyDeL/Mistral-7B
done: updated=2 skipped=1 failed=0
```
