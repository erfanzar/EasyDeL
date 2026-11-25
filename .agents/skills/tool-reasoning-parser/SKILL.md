---
name: tool-reasoning-parser
description: Add, debug, or test EasyDeL tool-call and reasoning parsers for OpenAI-compatible inference. Use for ToolParserManager, ReasoningParserManager, DelegatingParser, streaming tool-call deltas, eSurge API authoritative tool_calls, shared_vocab caching, or parser auto-detection.
---

# Skill: Work On Tool And Reasoning Parsers

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill for parser behavior under `libs/easydel/easydel/inference/`.

## First Reads

- `WORKSPACE.md`
- Tool parsers: `libs/easydel/easydel/inference/tools/{abstract_tool.py,auto_detect.py,tool_calling_mixin.py,utils.py}`
  and `libs/easydel/easydel/inference/tools/parsers/`
- Reasoning parsers:
  `libs/easydel/easydel/inference/reasoning/{abstract_reasoning.py,auto_detect.py,basic_parsers.py,reasoning_mixin.py}`
  and `libs/easydel/easydel/inference/reasoning/parsers/`
- Cooperative state machine: `libs/easydel/easydel/inference/parsing/delegating_parser.py`
- OpenAI schemas & proxies: `libs/easydel/easydel/inference/{openai_api_modules.py,oai_proxies.py}`
- eSurge integration:
  - `libs/easydel/easydel/inference/esurge/engine/{bootstrap.py,admission.py,output_pipeline.py}`
  - `libs/easydel/easydel/inference/esurge/distributed/remote_engine.py`
  - `libs/easydel/easydel/inference/esurge/esurge_engine.py`
  - `libs/easydel/easydel/inference/esurge/server/api_server.py`

## Parser Contracts

- **Tool parser registration:** register with `@ToolParserManager.register_module(...)` and retrieve via
  `ToolParserManager.get_tool_parser(...)`. Implement both `extract_tool_calls` and `extract_tool_calls_streaming` when
  the format supports streaming.
- **Reasoning parser registration:** register with `@ReasoningParserManager.register_module(...)` and retrieve via
  `ReasoningParserManager.get_reasoning_parser(...)`. Implement both `extract_reasoning` and
  `extract_reasoning_streaming`.
- **Type-alias parity:** adding a new parser requires adding its name to `ToolParserName`
  (`libs/easydel/easydel/inference/tools/abstract_tool.py`) or `ReasoningParserName`
  (`libs/easydel/easydel/inference/reasoning/abstract_reasoning.py`).
- **Cached vocabulary (`shared_vocab`):** parsers are instantiated per admitted request inside `RequestAdmission`
  (`esurge/engine/admission.py`). Always obtain the vocab mapping via `shared_vocab(tokenizer)`
  (`reasoning/abstract_reasoning.py`, delegated by `ToolParser.vocab` in `tools/abstract_tool.py`) rather than calling
  `tokenizer.get_vocab()` per request (~32–64 ms overhead on every admission).
- **`DelegatingParser` state machine:** coordinates reasoning and tool parsers on shared output using the 4-phase
  `ParsePhase` state machine (`REASONING`, `CONTENT`, `TOOL_CALL`, `BUFFERING`) and `TrackedToolCallState` catch-up
  deltas, driven by `OutputPipeline` (`esurge/engine/output_pipeline.py`).

## eSurge Integration Rules

- Parser auto-detection and resolution live in module-level functions `auto_detect_tool_parser`,
  `auto_detect_reasoning_parser_name`, and `_resolve_parsers` in
  `libs/easydel/easydel/inference/esurge/engine/bootstrap.py` (also called from `distributed/remote_engine.py`), while
  per-request parser classes (`_tool_parser_class`, `_reasoning_parser_class`) are held and instantiated by
  `RequestAdmission` in `esurge/engine/admission.py`.
- The eSurge API server (`server/api_server.py`) must honor authoritative `tool_calls` returned by the engine — never
  reparse output that already carries structured tool calls.
- For serving, routing, or runner failures, also load `.agents/skills/debug-esurge/SKILL.md`.

## Dataset Normalization Route

OpenAI tool-call dataset normalization (`libs/easydel/easydel/scripts/normalize_openai_tool_dataset.py`) is canonically
documented in `.agents/skills/build-dataset-pipeline/SKILL.md` (use `--max-rows` for spot checks, and note that
`--drop-incomplete-tool-call-rows` defaults to `True` unless disabled with `--no-drop-incomplete-tool-call-rows`).

## Verification

Pure-Python parser and API-contract tests do no array computation (array-free or marked `@pytest.mark.cpu_ok`) and may
run under the CPU trio. Any test driving a real model or eSurge engine runs on an accelerator (TPU:
`env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu`; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/parsing/ \
                libs/easydel/tests/inference/tools/test_delegating_parser.py \
                libs/easydel/tests/inference/tools/test_auto_detect.py \
                libs/easydel/tests/inference/tools/parsers/test_extract.py \
                libs/easydel/tests/inference/tools/parsers/test_engine_streaming.py \
                libs/easydel/tests/inference/reasoning/test_parsers_extract.py \
                libs/easydel/tests/inference/reasoning/test_content_delta_alignment.py \
                libs/easydel/tests/inference/test_parsers_vwhisper_review_regressions.py \
                libs/easydel/tests/inference/test_shared_vocab.py \
                libs/easydel/tests/inference/esurge/test_engine_api_authoritative.py
```

Add streaming delta tests when changing incremental assembly, and final-response tests when changing batch extraction.
