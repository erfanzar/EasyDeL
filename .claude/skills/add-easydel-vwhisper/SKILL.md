---
name: add-easydel-vwhisper
description: Add, update, or debug the vWhisper speech-inference engine under libs/easydel/easydel/inference/vwhisper. Use for audio transcription/translation, server endpoints, CLI, generation, or audio preprocessing.
---

# Skill: Work On EasyDeL vWhisper

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work is inside
`libs/easydel/easydel/inference/vwhisper` or when the Whisper-based speech inference path changes.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/easydel/inference/vwhisper/__init__.py`
- `libs/easydel/easydel/inference/vwhisper/core.py`
- `libs/easydel/easydel/inference/vwhisper/config.py`
- `libs/easydel/easydel/inference/vwhisper/generation.py`
- `libs/easydel/easydel/inference/vwhisper/utils.py`
- `libs/easydel/easydel/inference/vwhisper/server.py`
- `libs/easydel/easydel/inference/vwhisper/cli.py`

## Typical Tasks

1. Add a transcription output format or decoding option by extending
   `vWhisperInferenceConfig` and post-processing in `core.py`.
2. Extend audio preprocessing (new input types, resampling, normalization) in
   `utils.py`.
3. Modify forced decoder IDs or add task/language handling in `generation.py`.
4. Add or modify API server endpoints or CLI flags in `server.py` / `cli.py`.

## Routing

- Underlying model changes: load `.claude/skills/add-easydel-model/SKILL.md`.
- Serving / OpenAI-compatible output issues: load
  `.claude/skills/debug-esurge/SKILL.md`.
- SpectraX `spx.export` / `bind` issues: load
  `.claude/skills/spectrax-core/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/vwhisper/
```

Also validate the CLI help and server startup paths on CPU before claiming readiness.
