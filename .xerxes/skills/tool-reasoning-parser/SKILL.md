---
name: tool-reasoning-parser
description: Add, debug, or test EasyDeL tool-call and reasoning parsers for OpenAI-compatible inference. Use for ToolParserManager, ReasoningParserManager, DelegatingParser, streaming tool-call deltas, eSurge API authoritative tool_calls, function calling, normalized tool datasets, or parser auto-detection.
---

# Skill: Work On Tool And Reasoning Parsers

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill for parser behavior in
`libs/easydel`.

## First Reads

- `WORKSPACE.md`
- `libs/easydel/easydel/inference/tools/abstract_tool.py`
- `libs/easydel/easydel/inference/tools/auto_detect.py`
- `libs/easydel/easydel/inference/tools/tool_calling_mixin.py`
- `libs/easydel/easydel/inference/tools/utils.py`
- `libs/easydel/easydel/inference/tools/parsers/`
- `libs/easydel/easydel/inference/reasoning/abstract_reasoning.py`
- `libs/easydel/easydel/inference/reasoning/auto_detect.py`
- `libs/easydel/easydel/inference/reasoning/basic_parsers.py`
- `libs/easydel/easydel/inference/reasoning/reasoning_mixin.py`
- `libs/easydel/easydel/inference/reasoning/parsers/`
- `libs/easydel/easydel/inference/parsing/delegating_parser.py`
- `libs/easydel/easydel/inference/openai_api_modules.py`
- `libs/easydel/easydel/inference/oai_proxies.py`
- `libs/easydel/easydel/inference/esurge/esurge_engine.py`
- `libs/easydel/easydel/inference/esurge/server/api_server.py`

## Parser Contracts

Tool parsers register through `ToolParserManager.register_module` and are retrieved with
`ToolParserManager.get_tool_parser`. Implement both batch and streaming behavior when the model format supports
streaming:

- `extract_tool_calls`
- `extract_tool_calls_streaming`

Reasoning parsers register through `ReasoningParserManager` and implement:

- `extract_reasoning`
- `extract_reasoning_streaming`

Use `DelegatingParser` when tool and reasoning parsing need to cooperate on the same generated text.

## eSurge Rules

Inspect `eSurge._auto_detect_tool_parser`,
`eSurge._auto_detect_reasoning_parser_name`, `tool_parser`,
`reasoning_parser_name`, `_tool_parser_class`, and `_reasoning_parser_class`
when parser selection is wrong.

The eSurge API server should honor authoritative `tool_calls` returned by the engine. Do not make the API server reparse
output that already has structured tool calls.

For serving or benchmark failures, also load `.xerxes/skills/debug-esurge/SKILL.md`.

## Dataset Route

For OpenAI tool-call dataset normalization, inspect:

- `libs/easydel/easydel/scripts/normalize_openai_tool_dataset.py`

Use `--max-rows` for a small validation run before producing the final dataset.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/tools/test_delegating_parser.py libs/easydel/tests/inference/tools/test_auto_detect.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/tools/parsers/test_extract.py libs/easydel/tests/inference/tools/parsers/test_engine_streaming.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/reasoning/test_parsers_extract.py libs/easydel/tests/inference/reasoning/test_content_delta_alignment.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/esurge/test_engine_api_authoritative.py
```

Add streaming tests when changing delta assembly. Add final-response tests when changing non-streaming parser output.
