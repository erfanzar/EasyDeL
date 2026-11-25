---
name: add-easydel-vwhisper
description: Add, update, or debug the vWhisper speech-inference engine under libs/easydel/easydel/inference/vwhisper. Use for audio transcription/translation, OpenAI Whisper response formats, standalone FastAPI server endpoints, CLI, compiled generation, or audio preprocessing.
---

# Skill: Work On EasyDeL vWhisper

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when working inside `libs/easydel/easydel/inference/vwhisper` or
modifying Whisper-based speech inference.

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

1. **OpenAI Whisper response formats & HTTP endpoints (`server.py`):** add or modify output formats
   (`ResponseFormat`: `json`, `text`, `srt`, `verbose_json`, `vtt`, `_format_whisper_response`, `_format_srt`,
   `_format_vtt`, `_normalize_segments`) and FastAPI routes (`create_whisper_app`). `vwhisper/server.py` is a
   standalone FastAPI app — it shares **no** code with eSurge or `eSurgeApiServer`.
2. **ASR decoding, batching & config (`core.py`, `config.py`):** `core.py` runs `tokenizer._decode_asr` returning
   `{"text": ..., **optional}` and manages `vWhisperInference`; `config.py` defines `vWhisperInferenceConfig`.
3. **Audio preprocessing (`utils.py`):** input decoding, resampling, chunking, and feature extraction.
4. **Forced decoder IDs & compiled generation (`generation.py`):** language/task token resolution and
   `_compiled_generate`.
5. **CLI flags (`cli.py`):** `--model`, `--host`, `--port`, `--dtype`, and server launch wiring.

## Contracts & Gotchas

- **Registry:** `vWhisperInference` registers via `@Registry.register("serve", "vwhisper")` in `core.py`.
- **Static-arg JIT cache-key invariant (`config.py` <-> `generation.py`):** `_compiled_generate` uses
  `@ejit(static_argnames=["graphdef", "inference_config", "return_timestamps"])`. `vWhisperInferenceConfig` drives JIT
  hashing/equality via `cache_key()` and `_canonicalize_cache_value()`. Any new field on `vWhisperInferenceConfig` that
  affects generation **must** be included in `cache_key()`, or `@ejit` will silently reuse stale compiled executables.
- **Empty `forced_decoder_ids` normalization (`core.py`, `generation.py`):** both normalize empty
  `forced_decoder_ids` to `{0: None}` because `WhisperForConditionalGeneration._force_generate` constructs
  `ForceTokensLogitsProcessor`, which treats `{0: None}` as a no-op and raises `ValueError` on an empty mapping.
- **`WhisperModel` singleton & upload safety (`server.py`):** `WhisperModel` is a thread-safe singleton that raises
  `ValueError` when re-initialized with conflicting model/config arguments unless `WhisperModel.reset()` is called.
  Uploads enforce the pre-existing `VWHISPER_MAX_UPLOAD_BYTES` limit (100 MiB default; do not add new `os.environ`
  knobs), clean up temp files via `_uploaded_temp_file`, and offload blocking inference with `asyncio.to_thread`.

## Routing

- Underlying Whisper model/config changes: `.agents/skills/add-easydel-model/SKILL.md`.
- SpectraX `spx.export` / `spx.bind` issues: `.agents/skills/spectrax-modules/SKILL.md`.
- Do **not** route vWhisper HTTP/server issues to `debug-esurge` — they are independent servers.

## Verification

Run the pure-host (`cpu_ok`) unit/regression tests and CLI check (CPU trio is allowed):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/test_vwhisper_config.py \
                libs/easydel/tests/inference/test_parsers_vwhisper_review_regressions.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
  uv run python -m easydel.inference.vwhisper.cli --help
```

Run Whisper numerical/model tests and live server startup on an accelerator (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report model/server numerics as unverified on
hardware:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/modules/test_whisper_numerics.py
```
