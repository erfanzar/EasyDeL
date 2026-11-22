# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI agent harness RL environments and intercepting Token-in-Token-out (TITO) HTTP gateway.

Enables RL training directly through unmodified external coding/agent harnesses
(``xerxes``, ``claude_code``, ``codex``, ``hermes``, ``pi``, ``opencode``,
``aider``, ``goose``, or custom CLIs) with zero extra package dependencies.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import http.server
import ipaddress
import json
import posixpath
import socket
import threading
import typing as tp
import urllib.parse
import uuid

import numpy as np

from easydel.trainers._online_rl.trajectory import (
    OnlineRLTrajectory,
    TrajectorySegment,
    TrajectorySpan,
)
from easydel.trainers._shared.environment import ResetResult, StepResult

from ._base import SandboxEnvironment, register_environment
from .runtime import (
    ContainerConfig,
    ContainerSandboxSession,
    KiriRuntime,
    SandboxInfraError,
    SandboxPool,
    SandboxSession,
)
from .swe_repo import (
    _CLEAN_GIT_ENV,
    GradingSnapshot,
    _build_protected_predicate,
    _list_base_tree,
    _parse_python_test_command,
    _seed_repo_dir_files,
    apply_patch_via_stdin,
    capture_base_revision,
    extract_agent_workspace_changes,
    init_clean_git_repo,
    kill_agent_processes,
    run_verified_test_command,
    unscorable_step,
    verification_resolved,
)

ProtocolKind = tp.Literal["openai_chat", "openai_responses", "anthropic_messages"]
SubagentTrainingMode = tp.Literal["root_only", "all_streams"]

_DEFAULT_MAX_GATEWAY_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_CHUNK_LINE_BYTES = 4096
_HARNESS_VERIFY_ENV = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONPATH": "",
    "PYTHONSTARTUP": "",
    "BASH_ENV": "",
    "ENV": "",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_HARNESS_VERIFY_HOOK_BASENAMES = (
    "sitecustomize.py",
    "usercustomize.py",
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "pytest.py",
    "pytest",
    "_pytest",
)
_HARNESS_VERIFY_HOOK_SUFFIXES = (".pth",)


def _format_host_for_url(host: str) -> str:
    """Wrap bare IPv6 literals in ``[...]`` for valid ``http://<host>:<port>`` URLs."""
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]"
    return host


def _is_ipv6_host(host: str) -> bool:
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return isinstance(ipaddress.ip_address(bare), ipaddress.IPv6Address)
    except ValueError:
        return False


@dataclasses.dataclass(frozen=True)
class HarnessPreset:
    """Invocation and configuration template for an external CLI agent harness.

    Attributes:
        name: Canonical preset identifier.
        protocol: Primary wire protocol spoken by the harness.
        argv_template: Command argv list where ``{prompt}``, ``{model}``,
            ``{base_url}``, ``{base_url_v1}``, and ``{api_key}`` are interpolated.
        env_template: Environment variables injected into the harness process.
        config_files: Optional config files (``rel_path -> template_str``) written
            into the episode workspace before launching the harness.
        requires_git: Initialize a clean git repository before launching if absent.
    """

    name: str
    protocol: ProtocolKind
    argv_template: tuple[str, ...]
    env_template: dict[str, str] = dataclasses.field(default_factory=dict)
    config_files: dict[str, str] = dataclasses.field(default_factory=dict)
    requires_git: bool = True

    def render_argv(self, **vars_: str) -> list[str]:
        return [part.format(**vars_) for part in self.argv_template]

    def render_env(self, **vars_: str) -> dict[str, str]:
        return {k: v.format(**vars_) for k, v in self.env_template.items()}

    def render_files(self, **vars_: str) -> dict[str, str]:
        return {path.format(**vars_): content.format(**vars_) for path, content in self.config_files.items()}


_HARNESS_PRESETS: dict[str, HarnessPreset] = {
    "xerxes": HarnessPreset(
        name="xerxes",
        protocol="openai_chat",
        argv_template=("xerxes", "{prompt}"),
        env_template={
            "OPENAI_BASE_URL": "{base_url_v1}",
            "OPENAI_API_KEY": "{api_key}",
            "ANTHROPIC_BASE_URL": "{base_url}",
            "ANTHROPIC_API_KEY": "{api_key}",
        },
    ),
    "claude_code": HarnessPreset(
        name="claude_code",
        protocol="anthropic_messages",
        argv_template=(
            "claude",
            "-p",
            "{prompt}",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--model",
            "{model}",
        ),
        env_template={
            "ANTHROPIC_BASE_URL": "{base_url}",
            "ANTHROPIC_API_KEY": "{api_key}",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    ),
    "codex": HarnessPreset(
        name="codex",
        protocol="openai_responses",
        argv_template=("codex", "--yolo", "exec", "-m", "{model}", "{prompt}"),
        env_template={
            "OPENAI_BASE_URL": "{base_url_v1}",
            "OPENAI_API_KEY": "{api_key}",
        },
        requires_git=True,
    ),
    "hermes": HarnessPreset(
        name="hermes",
        protocol="openai_chat",
        argv_template=("hermes", "--provider", "openai", "-m", "{model}", "-z", "{prompt}"),
        env_template={
            "OPENAI_BASE_URL": "{base_url_v1}",
            "OPENAI_API_KEY": "{api_key}",
            "HERMES_INFERENCE_MODEL": "{model}",
            "HERMES_QUIET": "1",
        },
    ),
    "pi": HarnessPreset(
        name="pi",
        protocol="openai_chat",
        argv_template=(
            "pi",
            "--print",
            "--no-session",
            "--provider",
            "easydel",
            "--model",
            "{model}",
            "--api-key",
            "{api_key}",
            "--",
            "{prompt}",
        ),
        env_template={
            "OPENAI_API_KEY": "{api_key}",
            "PI_CODING_AGENT_DIR": ".pi/agent",
        },
        config_files={
            ".pi/agent/models.json": (
                "{{\n"
                '  "providers": {{\n'
                '    "easydel": {{\n'
                '      "baseUrl": "{base_url_v1}",\n'
                '      "api": "openai-completions",\n'
                '      "apiKey": "{api_key}",\n'
                '      "models": [{{"id": "{model}"}}]\n'
                "    }}\n"
                "  }}\n"
                "}}\n"
            )
        },
    ),
    "opencode": HarnessPreset(
        name="opencode",
        protocol="openai_chat",
        argv_template=("opencode", "run", "--model", "easydel/{model}", "{prompt}"),
        env_template={
            "OPENAI_BASE_URL": "{base_url_v1}",
            "OPENAI_API_KEY": "{api_key}",
        },
        config_files={
            "opencode.json": (
                "{{\n"
                '  "$schema": "https://opencode.ai/config.json",\n'
                '  "provider": {{\n'
                '    "easydel": {{\n'
                '      "npm": "@ai-sdk/openai-compatible",\n'
                '      "options": {{"baseURL": "{base_url_v1}", "apiKey": "{api_key}"}},\n'
                '      "models": {{"{model}": {{"name": "{model}"}}}}\n'
                "    }}\n"
                "  }}\n"
                "}}\n"
            )
        },
    ),
    "aider": HarnessPreset(
        name="aider",
        protocol="openai_chat",
        argv_template=(
            "aider",
            "--yes-always",
            "--no-auto-commits",
            "--model",
            "openai/{model}",
            "--openai-api-base",
            "{base_url_v1}",
            "--openai-api-key",
            "{api_key}",
            "--message",
            "{prompt}",
        ),
        env_template={
            "OPENAI_API_BASE": "{base_url_v1}",
            "OPENAI_API_KEY": "{api_key}",
        },
    ),
    "goose": HarnessPreset(
        name="goose",
        protocol="openai_chat",
        argv_template=("goose", "run", "--no-session", "-t", "{prompt}"),
        env_template={
            "GOOSE_PROVIDER": "openai",
            "OPENAI_HOST": "{base_url}",
            "OPENAI_API_KEY": "{api_key}",
            "GOOSE_MODEL": "{model}",
        },
    ),
}


def register_harness(preset: HarnessPreset) -> HarnessPreset:
    """Register a custom :class:`HarnessPreset` by ``preset.name``."""
    _HARNESS_PRESETS[preset.name.lower()] = preset
    return preset


def get_harness_preset(name_or_preset: str | HarnessPreset) -> HarnessPreset:
    """Look up a registered :class:`HarnessPreset`."""
    if isinstance(name_or_preset, HarnessPreset):
        return name_or_preset
    key = name_or_preset.lower()
    if key not in _HARNESS_PRESETS:
        raise KeyError(f"Unknown harness preset {name_or_preset!r}. Available: {sorted(_HARNESS_PRESETS)}")
    return _HARNESS_PRESETS[key]


def list_harnesses() -> list[str]:
    """Return all registered CLI harness preset names."""
    return sorted(_HARNESS_PRESETS.keys())


@dataclasses.dataclass(frozen=True)
class GatewayTurnOutput:
    """Model generation result returned to :class:`HarnessTITOGateway`."""

    token_ids: list[int]
    text: str
    logprobs: list[float] | None = None
    tool_calls: list[dict[str, tp.Any]] | None = None
    reasoning_text: str | None = None
    policy_version: int = 0
    finish_reason: str = "stop"


GatewayGenerateFn = tp.Callable[[list[int], list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None], GatewayTurnOutput]
MessagesEncodeFn = tp.Callable[[list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None], list[int]]


@dataclasses.dataclass
class _SegmentBuilder:
    segment_id: int
    stream_id: str
    is_root: bool
    messages: list[dict[str, tp.Any]] = dataclasses.field(default_factory=list)
    token_ids: list[int] = dataclasses.field(default_factory=list)
    action_mask: list[bool] = dataclasses.field(default_factory=list)
    behavior_logps: list[float] = dataclasses.field(default_factory=list)
    has_all_logps: bool = True
    spans: list[TrajectorySpan] = dataclasses.field(default_factory=list)
    versions: list[int] = dataclasses.field(default_factory=list)


class _EpisodeState:
    def __init__(self, episode_id: str, trajectory_id: int):
        self.episode_id = episode_id
        self.trajectory_id = trajectory_id
        self.lock = threading.Lock()
        self.stream_locks: dict[str, threading.Lock] = {}
        self.active_streams: dict[str, _SegmentBuilder] = {}
        self.sealed_segments: list[_SegmentBuilder] = []
        self.next_segment_id = 0
        self.finalized = False

    def get_stream_lock(self, stream_id: str) -> threading.Lock:
        with self.lock:
            lk = self.stream_locks.get(stream_id)
            if lk is None:
                lk = threading.Lock()
                self.stream_locks[stream_id] = lk
            return lk

    def alloc_segment(self, stream_id: str, *, is_root: bool) -> _SegmentBuilder:
        seg = _SegmentBuilder(segment_id=self.next_segment_id, stream_id=stream_id, is_root=is_root)
        self.next_segment_id += 1
        return seg


def _canonical_json_args(raw: tp.Any) -> str:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    try:
        return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
        return str(raw)


def _finalize_tool_calls(
    tool_calls: list[dict[str, tp.Any]] | None, *, id_prefix: str
) -> list[dict[str, tp.Any]] | None:
    """Return a copy of ``tool_calls`` where every entry has a stable ID and string JSON arguments."""
    if not tool_calls:
        return None
    out: list[dict[str, tp.Any]] = []
    for tc in tool_calls:
        fn_raw = tc.get("function")
        fn = dict(fn_raw) if isinstance(fn_raw, dict) else {}
        args_val = fn.get("arguments", "{}")
        if not isinstance(args_val, str):
            args_val = json.dumps(args_val, ensure_ascii=False)
        out.append(
            {
                "id": str(tc.get("id") or f"{id_prefix}_{uuid.uuid4().hex[:8]}"),
                "type": str(tc.get("type") or "function"),
                "function": {"name": str(fn.get("name", "")), "arguments": args_val},
            }
        )
    return out


def _normalize_openai_chat_messages(
    body: dict[str, tp.Any],
) -> tuple[list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None]:
    msgs = [dict(m) for m in body.get("messages", []) if isinstance(m, dict)]
    tools = body.get("tools")
    return msgs, (list(tools) if isinstance(tools, list) else None)


def _normalize_openai_responses_input(
    body: dict[str, tp.Any],
) -> tuple[list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None]:
    msgs: list[dict[str, tp.Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        msgs.append({"role": "system", "content": instructions})
    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        msgs.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call_output":
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": str(item.get("output", "")),
                    }
                )
            elif itype == "function_call":
                msgs.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")},
                            }
                        ],
                    }
                )
            elif itype == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    parts = [str(s.get("text", "")) for s in summary if isinstance(s, dict)]
                    rtext = "\n".join(p for p in parts if p)
                else:
                    rtext = str(item.get("text", ""))
                if rtext:
                    msgs.append({"role": "assistant", "content": f"<think>{rtext}</think>"})
            else:
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    parts = [str(c.get("text", "")) for c in content if isinstance(c, dict)]
                    content = "\n".join(p for p in parts if p)
                msgs.append({"role": role, "content": str(content)})
    tools = body.get("tools")
    return msgs, (list(tools) if isinstance(tools, list) else None)


def _normalize_anthropic_messages(
    body: dict[str, tp.Any],
) -> tuple[list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None]:
    msgs: list[dict[str, tp.Any]] = []
    system_val = body.get("system")
    if isinstance(system_val, str) and system_val:
        msgs.append({"role": "system", "content": system_val})
    elif isinstance(system_val, list):
        sys_txt = "\n".join(
            str(b.get("text", "")) for b in system_val if isinstance(b, dict) and b.get("type") == "text"
        )
        if sys_txt:
            msgs.append({"role": "system", "content": sys_txt})

    for raw in body.get("messages", []):
        if not isinstance(raw, dict):
            continue
        role = raw.get("role", "user")
        content = raw.get("content", "")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, tp.Any]] = []
            tool_results: list[dict[str, tp.Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(str(block.get("text", "")))
                elif btype == "thinking":
                    text_parts.append(f"<think>{block.get('thinking', '')}</think>")
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                            },
                        }
                    )
                elif btype == "tool_result":
                    tr_content = block.get("content", "")
                    if isinstance(tr_content, list):
                        tr_content = "\n".join(str(x.get("text", "")) for x in tr_content if isinstance(x, dict))
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": str(tr_content),
                        }
                    )
            if tool_results:
                msgs.extend(tool_results)
            if text_parts or tool_calls:
                entry: dict[str, tp.Any] = {"role": role, "content": "\n".join(text_parts)}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                msgs.append(entry)

    raw_tools = body.get("tools")
    norm_tools: list[dict[str, tp.Any]] | None = None
    if isinstance(raw_tools, list):
        norm_tools = []
        for t in raw_tools:
            if isinstance(t, dict) and "name" in t and "function" not in t:
                norm_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                        },
                    }
                )
            elif isinstance(t, dict):
                norm_tools.append(t)
    return msgs, norm_tools


def _canonical_tool_calls(tool_calls: tp.Any) -> list[dict[str, tp.Any]] | None:
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    norm: list[dict[str, tp.Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        fn_dict = fn if isinstance(fn, dict) else {}
        norm.append(
            {
                "id": str(tc.get("id", "")),
                "type": str(tc.get("type", "function")),
                "function": {
                    "name": str(fn_dict.get("name", "")),
                    "arguments": _canonical_json_args(fn_dict.get("arguments", "{}")),
                },
            }
        )
    return norm or None


def _canonical_msg(msg: dict[str, tp.Any]) -> str:
    content = msg.get("content", "")
    if content is None:
        content = ""
    return json.dumps(
        {
            "role": msg.get("role", ""),
            "content": content,
            "tool_calls": _canonical_tool_calls(msg.get("tool_calls")),
            "tool_call_id": msg.get("tool_call_id"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _is_prefix_messages(prefix: list[dict[str, tp.Any]], full: list[dict[str, tp.Any]]) -> bool:
    if not prefix or len(prefix) > len(full):
        return False
    return all(_canonical_msg(p) == _canonical_msg(f) for p, f in zip(prefix, full[: len(prefix)], strict=True))


def _build_protocol_response_and_messages(
    protocol: ProtocolKind, out: GatewayTurnOutput
) -> tuple[tp.Any, list[dict[str, tp.Any]]]:
    """Finalize tool IDs and derive protocol wire items + exact normalized assistant messages for TITO stitching."""
    if protocol == "openai_chat":
        final_tcs = _finalize_tool_calls(out.tool_calls, id_prefix="call")
        msg: dict[str, tp.Any] = {"role": "assistant", "content": out.text}
        if final_tcs:
            msg["tool_calls"] = final_tcs
        return final_tcs, [msg]

    if protocol == "openai_responses":
        final_tcs = _finalize_tool_calls(out.tool_calls, id_prefix="call")
        output_items: list[dict[str, tp.Any]] = []
        if out.reasoning_text:
            output_items.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:8]}",
                    "summary": [{"type": "summary_text", "text": out.reasoning_text}],
                }
            )
        if out.text:
            output_items.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": out.text, "annotations": []}],
                }
            )
        if final_tcs:
            for tc in final_tcs:
                fn = tc["function"]
                output_items.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex[:8]}",
                        "call_id": tc["id"],
                        "name": fn["name"],
                        "arguments": fn["arguments"],
                        "status": "completed",
                    }
                )
        assistant_msgs, _ = _normalize_openai_responses_input({"input": output_items})
        return output_items, assistant_msgs

    # protocol == "anthropic_messages"
    final_tcs = _finalize_tool_calls(out.tool_calls, id_prefix="toolu")
    blocks: list[dict[str, tp.Any]] = []
    if out.reasoning_text:
        blocks.append({"type": "thinking", "thinking": out.reasoning_text})
    if out.text:
        blocks.append({"type": "text", "text": out.text})
    if final_tcs:
        for tc in final_tcs:
            fn = tc["function"]
            raw_args = fn["arguments"]
            try:
                parsed_inp = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed_inp = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": parsed_inp,
                }
            )
    assistant_msgs, _ = _normalize_anthropic_messages({"messages": [{"role": "assistant", "content": blocks}]})
    return blocks, assistant_msgs


class _RequestBodyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _read_bounded_request_body(rfile: tp.Any, headers: tp.Any, max_bytes: int) -> bytes:
    """Read an HTTP request body (Content-Length or chunked) enforcing a strict ``max_bytes`` cap."""
    te = (headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in te:
        out = bytearray()
        while True:
            line = rfile.readline(_MAX_CHUNK_LINE_BYTES + 1)
            if not line or len(line) > _MAX_CHUNK_LINE_BYTES or not line.endswith(b"\n"):
                raise _RequestBodyError(400, "Malformed chunked transfer encoding")
            size_part = line.split(b";", 1)[0].strip()
            try:
                chunk_len = int(size_part, 16)
            except ValueError as exc:
                raise _RequestBodyError(400, "Invalid chunk size") from exc
            if chunk_len < 0:
                raise _RequestBodyError(400, "Negative chunk size")
            if chunk_len == 0:
                # Consume trailer headers up to the terminating empty line.
                for _ in range(64):
                    trailer = rfile.readline(_MAX_CHUNK_LINE_BYTES + 1)
                    if not trailer or trailer in (b"\r\n", b"\n"):
                        break
                break
            if len(out) + chunk_len > max_bytes:
                raise _RequestBodyError(413, f"Request body exceeds limit of {max_bytes} bytes")
            chunk = rfile.read(chunk_len)
            if len(chunk) < chunk_len:
                raise _RequestBodyError(400, "Truncated chunked request body")
            out.extend(chunk)
            crlf = rfile.read(2)
            if crlf != b"\r\n":
                raise _RequestBodyError(400, "Malformed chunk terminator")
        return bytes(out) if out else b"{}"

    cl_raw = headers.get("Content-Length")
    if cl_raw is None:
        return b"{}"
    try:
        length = int(cl_raw.strip())
    except ValueError as exc:
        raise _RequestBodyError(400, "Invalid Content-Length header") from exc
    if length < 0:
        raise _RequestBodyError(400, "Negative Content-Length header")
    if length > max_bytes:
        raise _RequestBodyError(413, f"Request body exceeds limit of {max_bytes} bytes")
    if length == 0:
        return b"{}"
    raw = rfile.read(length)
    if len(raw) < length:
        raise _RequestBodyError(400, "Incomplete request body")
    return raw


class HarnessTITOGateway:
    """Thread-safe HTTP gateway intercepting CLI harness LLM calls into TITO trajectories.

    Serves:
    * ``GET /v1/models``
    * ``POST /v1/chat/completions`` (JSON & SSE)
    * ``POST /v1/responses`` (JSON & SSE)
    * ``POST /v1/messages`` (Anthropic JSON & SSE)

    Performs prefix-stitching TITO caching per ``(episode_id, stream_id)`` and
    automatically splits new :class:`OnlineRLTrajectory` segments whenever a
    harness compacts or mutates earlier conversation history.
    """

    def __init__(
        self,
        *,
        generate_fn: GatewayGenerateFn,
        encode_messages_fn: MessagesEncodeFn,
        model_name: str = "easydel-rl-policy",
        host: str = "127.0.0.1",
        port: int = 0,
        advertise_host: str | None = None,
        public_base_url: str | None = None,
        max_request_bytes: int = _DEFAULT_MAX_GATEWAY_REQUEST_BYTES,
    ):
        """Create the gateway.

        Args:
            generate_fn: Policy turn generator.
            encode_messages_fn: Chat-message tokenizer used for TITO stitching.
            model_name: Model id reported to harnesses.
            host: Interface the HTTP server binds to. Container sandboxes cannot reach ``127.0.0.1`` on the
                trainer host; bind a reachable interface (e.g. ``"0.0.0.0"`` or the container bridge IP).
            port: TCP port (``0`` picks a free port).
            advertise_host: Host name/IP written into harness URLs instead of the bind address
                (the bound port is kept).
            public_base_url: Full URL (e.g. ``"http://gateway.internal:8080"``) written into harness URLs;
                overrides ``advertise_host``.
            max_request_bytes: Maximum accepted POST body size in bytes (larger bodies return HTTP 413).
        """
        if int(max_request_bytes) <= 0:
            raise ValueError(f"max_request_bytes must be positive, got {max_request_bytes}")
        self._advertise_host = advertise_host
        self._public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self._max_request_bytes = int(max_request_bytes)
        self._generate = generate_fn
        self._encode_messages = encode_messages_fn
        self.model_name = model_name
        self._episodes_lock = threading.Lock()
        self._episodes_by_token: dict[str, _EpisodeState] = {}
        self._next_traj_id = 0

        gateway = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: tp.Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:
                if self.path.rstrip("/") in ("/v1/models", "/models"):
                    payload = {
                        "object": "list",
                        "data": [{"id": gateway.model_name, "object": "model", "owned_by": "easydel"}],
                    }
                    self._send_json(200, payload)
                elif self.path == "/health":
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(404, {"error": {"message": f"Unknown path {self.path}"}})

            def do_POST(self) -> None:
                try:
                    raw = _read_bounded_request_body(self.rfile, self.headers, gateway._max_request_bytes)
                except _RequestBodyError as exc:
                    self._send_json(exc.status, {"error": {"message": exc.message}})
                    return
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(400, {"error": {"message": "Invalid UTF-8 or JSON request body"}})
                    return
                if not isinstance(body, dict):
                    self._send_json(400, {"error": {"message": "Request body must be a JSON object"}})
                    return

                token = gateway._extract_token(dict(self.headers))
                ep = gateway._lookup_episode(token)
                if ep is None:
                    self._send_json(401, {"error": {"message": "Unknown or closed rollout episode token"}})
                    return

                stream_id = (
                    self.headers.get("X-Agent-ID")
                    or self.headers.get("X-Xerxes-Agent-ID")
                    or self.headers.get("X-Stream-ID")
                    or "root"
                )
                clean_path = self.path.split("?", 1)[0].rstrip("/")
                stream = bool(body.get("stream", False))

                try:
                    if clean_path.endswith("/chat/completions"):
                        msgs, tools = _normalize_openai_chat_messages(body)
                        turn, final_tcs = gateway._step_turn(
                            ep, stream_id=stream_id, messages=msgs, tools=tools, protocol="openai_chat"
                        )
                        gateway._respond_openai_chat(self, turn, tool_calls=final_tcs, stream=stream)
                    elif clean_path.endswith("/responses"):
                        msgs, tools = _normalize_openai_responses_input(body)
                        turn, output_items = gateway._step_turn(
                            ep, stream_id=stream_id, messages=msgs, tools=tools, protocol="openai_responses"
                        )
                        gateway._respond_openai_responses(self, turn, output_items=output_items, stream=stream)
                    elif clean_path.endswith("/messages"):
                        msgs, tools = _normalize_anthropic_messages(body)
                        turn, blocks = gateway._step_turn(
                            ep, stream_id=stream_id, messages=msgs, tools=tools, protocol="anthropic_messages"
                        )
                        gateway._respond_anthropic_messages(self, turn, blocks=blocks, stream=stream)
                    else:
                        self._send_json(404, {"error": {"message": f"Unsupported endpoint {clean_path}"}})
                except Exception as exc:
                    self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

            def _send_json(self, status: int, data: dict[str, tp.Any]) -> None:
                encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

            def _send_sse(self, events: list[tuple[str | None, str]]) -> None:
                chunks: list[bytes] = []
                for ev_name, data_str in events:
                    if ev_name is not None:
                        chunks.append(f"event: {ev_name}\n".encode())
                    chunks.append(f"data: {data_str}\n\n".encode())
                body = b"".join(chunks)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

        bind_host = host[1:-1] if host.startswith("[") and host.endswith("]") else host
        if _is_ipv6_host(host):

            class _IPv6ThreadingHTTPServer(http.server.ThreadingHTTPServer):
                address_family = socket.AF_INET6

            server_cls: type[http.server.ThreadingHTTPServer] = _IPv6ThreadingHTTPServer
        else:
            server_cls = http.server.ThreadingHTTPServer

        self._server = server_cls((bind_host, port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="ed-tito-gateway", daemon=True)
        self._started = False
        self._closed = False

    def start(self) -> HarnessTITOGateway:
        if self._closed:
            raise RuntimeError("HarnessTITOGateway is closed.")
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{_format_host_for_url(str(host))}:{port}"

    @property
    def base_url_v1(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def advertised_base_url(self) -> str:
        """URL harness processes inside the sandbox should use to reach this gateway."""
        if self._public_base_url:
            return self._public_base_url
        if self._advertise_host:
            port = self._server.server_address[1]
            return f"http://{_format_host_for_url(self._advertise_host)}:{port}"
        return self.base_url

    @property
    def advertised_base_url_v1(self) -> str:
        return f"{self.advertised_base_url}/v1"

    @property
    def advertised_is_loopback(self) -> bool:
        """Whether :attr:`advertised_base_url` only resolves on the trainer host itself."""
        host = (urllib.parse.urlsplit(self.advertised_base_url).hostname or "").lower()
        if host in {"localhost", "0.0.0.0", "::", ""}:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host.endswith(".localhost")

    def open_episode(self, *, trajectory_id: int | None = None) -> tuple[str, str]:
        """Register a new rollout episode and return ``(episode_id, api_key)``."""
        with self._episodes_lock:
            tid = self._next_traj_id if trajectory_id is None else int(trajectory_id)
            self._next_traj_id = max(self._next_traj_id, tid + 1)
            ep_id = uuid.uuid4().hex
            api_key = f"sk-ed-{ep_id}"
            self._episodes_by_token[api_key] = _EpisodeState(ep_id, tid)
            return ep_id, api_key

    def finalize_episode(
        self,
        api_key: str,
        *,
        reward: float,
        subagent_mode: SubagentTrainingMode = "root_only",
    ) -> tuple[list[OnlineRLTrajectory], dict[str, int]]:
        """Close ``api_key`` and convert recorded streams/compactions into trajectories."""
        with self._episodes_lock:
            ep = self._episodes_by_token.pop(api_key, None)
        if ep is None:
            return [], {"segments": 0, "oldest_policy_version": 0, "latest_policy_version": 0}

        with ep.lock:
            ep.finalized = True
            all_segs = [*ep.sealed_segments, *ep.active_streams.values()]
            ep.active_streams.clear()

        kept = [s for s in all_segs if any(s.action_mask) and (subagent_mode == "all_streams" or s.is_root)]
        if not kept and subagent_mode == "root_only":
            kept = [s for s in all_segs if any(s.action_mask)]

        trajectories: list[OnlineRLTrajectory] = []
        all_versions: list[int] = []
        for idx, seg in enumerate(kept):
            is_last = idx == len(kept) - 1
            rewards = np.zeros(len(seg.token_ids), dtype=np.float32)
            if is_last:
                last_action_indices = [i for i, m in enumerate(seg.action_mask) if m]
                if last_action_indices:
                    rewards[last_action_indices[-1]] = float(reward)
            oldest_v = min(seg.versions) if seg.versions else 0
            latest_v = max(seg.versions) if seg.versions else 0
            all_versions.extend(seg.versions)

            trajectories.append(
                OnlineRLTrajectory(
                    input_ids=np.asarray(seg.token_ids, dtype=np.int32),
                    attention_mask=np.ones(len(seg.token_ids), dtype=np.bool_),
                    action_mask=np.asarray(seg.action_mask, dtype=np.bool_),
                    rewards=rewards,
                    behavior_logps=np.asarray(seg.behavior_logps, dtype=np.float32) if seg.has_all_logps else None,
                    spans=tuple(seg.spans),
                    segments=(
                        TrajectorySegment(
                            segment_id=0,
                            start=0,
                            stop=len(seg.token_ids),
                            kind="execution",
                            is_terminal=is_last,
                        ),
                    ),
                    trajectory_id=ep.trajectory_id,
                    policy_version=oldest_v,
                    metadata={
                        "episode_id": ep.episode_id,
                        "stream_id": seg.stream_id,
                        "segment_index": idx,
                        "total_segments": len(kept),
                        "oldest_policy_version": oldest_v,
                        "latest_policy_version": latest_v,
                    },
                )
            )

        meta = {
            "segments": len(trajectories),
            "oldest_policy_version": min(all_versions) if all_versions else 0,
            "latest_policy_version": max(all_versions) if all_versions else 0,
        }
        return trajectories, meta

    @staticmethod
    def _extract_token(headers: dict[str, str]) -> str:
        lower = {k.lower(): v for k, v in headers.items()}
        auth = lower.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return lower.get("x-api-key", "").strip()

    def _lookup_episode(self, token: str) -> _EpisodeState | None:
        with self._episodes_lock:
            return self._episodes_by_token.get(token)

    def _step_turn(
        self,
        ep: _EpisodeState,
        *,
        stream_id: str,
        messages: list[dict[str, tp.Any]],
        tools: list[dict[str, tp.Any]] | None,
        protocol: ProtocolKind = "openai_chat",
    ) -> tuple[GatewayTurnOutput, tp.Any]:
        stream_lock = ep.get_stream_lock(stream_id)
        with stream_lock:
            # Snapshot current stream state without mutating it; commit obs + act atomically only after _generate
            # succeeds so exceptions never leave dangling observation tokens or clobber active streams.
            with ep.lock:
                is_root = (stream_id == "root") and (
                    ("root" in ep.active_streams) or ((not ep.active_streams) and (not ep.sealed_segments))
                )
                seg = ep.active_streams.get(stream_id)
                is_continuation = seg is not None and _is_prefix_messages(seg.messages, messages)
                base_tokens = list(seg.token_ids) if (is_continuation and seg is not None) else []
                prev_msg_count = len(seg.messages) if (is_continuation and seg is not None) else 0

            obs_tokens: list[int] = []
            prompt_tokens: list[int] = []
            if is_continuation:
                delta_msgs = messages[prev_msg_count:]
                if delta_msgs:
                    obs_tokens = [int(t) for t in self._encode_messages(delta_msgs, None)]
                context_tokens = [*base_tokens, *obs_tokens]
            else:
                prompt_tokens = [int(t) for t in self._encode_messages(messages, tools)]
                if not prompt_tokens:
                    prompt_tokens = [0]
                context_tokens = list(prompt_tokens)

            out = self._generate(context_tokens, messages, tools)
            wire_items, assistant_msgs = _build_protocol_response_and_messages(protocol, out)
            act_ids = [int(t) for t in out.token_ids]

            with ep.lock:
                if ep.finalized:
                    return out, wire_items
                if is_continuation and seg is not None:
                    if obs_tokens:
                        obs_start = len(seg.token_ids)
                        seg.token_ids.extend(obs_tokens)
                        seg.action_mask.extend([False] * len(obs_tokens))
                        seg.behavior_logps.extend([0.0] * len(obs_tokens))
                        seg.spans.append(
                            TrajectorySpan(start=obs_start, stop=len(seg.token_ids), kind="observation", segment_id=0)
                        )
                else:
                    if seg is not None and any(seg.action_mask):
                        ep.sealed_segments.append(seg)
                    seg = ep.alloc_segment(stream_id, is_root=is_root or stream_id == "root")
                    ep.active_streams[stream_id] = seg
                    seg.token_ids.extend(prompt_tokens)
                    seg.action_mask.extend([False] * len(prompt_tokens))
                    seg.behavior_logps.extend([0.0] * len(prompt_tokens))
                    seg.spans.append(TrajectorySpan(start=0, stop=len(seg.token_ids), kind="prompt", segment_id=0))

                if act_ids:
                    act_start = len(seg.token_ids)
                    seg.token_ids.extend(act_ids)
                    seg.action_mask.extend([True] * len(act_ids))
                    if out.logprobs is not None and len(out.logprobs) == len(act_ids):
                        seg.behavior_logps.extend(float(lp) for lp in out.logprobs)
                    else:
                        seg.has_all_logps = False
                        seg.behavior_logps.extend([0.0] * len(act_ids))
                    seg.spans.append(
                        TrajectorySpan(start=act_start, stop=len(seg.token_ids), kind="action", segment_id=0)
                    )
                seg.versions.append(int(out.policy_version))
                seg.messages = [*messages, *assistant_msgs]

            return out, wire_items

    def _respond_openai_chat(
        self,
        handler: tp.Any,
        out: GatewayTurnOutput,
        *,
        tool_calls: list[dict[str, tp.Any]] | None = None,
        stream: bool,
    ) -> None:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        effective_tcs = tool_calls if tool_calls is not None else out.tool_calls
        finish = "tool_calls" if effective_tcs else out.finish_reason
        if not stream:
            msg: dict[str, tp.Any] = {"role": "assistant", "content": out.text}
            if out.reasoning_text:
                msg["reasoning_content"] = out.reasoning_text
            if effective_tcs:
                msg["tool_calls"] = effective_tcs
            handler._send_json(
                200,
                {
                    "id": cid,
                    "object": "chat.completion",
                    "model": self.model_name,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": len(out.token_ids),
                        "total_tokens": len(out.token_ids),
                    },
                },
            )
            return

        delta: dict[str, tp.Any] = {"role": "assistant", "content": out.text}
        if out.reasoning_text:
            delta["reasoning_content"] = out.reasoning_text
        if effective_tcs:
            delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(effective_tcs)]
        chunk1 = json.dumps(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": self.model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
        chunk2 = json.dumps(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": self.model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
        )
        handler._send_sse([(None, chunk1), (None, chunk2), (None, "[DONE]")])

    def _respond_openai_responses(
        self,
        handler: tp.Any,
        out: GatewayTurnOutput,
        *,
        output_items: list[dict[str, tp.Any]] | None = None,
        stream: bool,
    ) -> None:
        rid = f"resp_{uuid.uuid4().hex[:12]}"
        if output_items is None:
            output_items, _ = _build_protocol_response_and_messages("openai_responses", out)
        payload = {
            "id": rid,
            "object": "response",
            "status": "completed",
            "model": self.model_name,
            "output": output_items,
            "output_text": out.text,
        }
        if not stream:
            handler._send_json(200, payload)
            return
        handler._send_sse(
            [
                (
                    "response.created",
                    json.dumps(
                        {"type": "response.created", "response": {**payload, "status": "in_progress", "output": []}}
                    ),
                ),
                ("response.completed", json.dumps({"type": "response.completed", "response": payload})),
            ]
        )

    def _respond_anthropic_messages(
        self,
        handler: tp.Any,
        out: GatewayTurnOutput,
        *,
        blocks: list[dict[str, tp.Any]] | None = None,
        stream: bool,
    ) -> None:
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        if blocks is None:
            blocks, _ = _build_protocol_response_and_messages("anthropic_messages", out)
        has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks)
        stop_reason = "tool_use" if has_tool_use else "end_turn"
        payload = {
            "id": mid,
            "type": "message",
            "role": "assistant",
            "model": self.model_name,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": len(out.token_ids)},
        }
        if not stream:
            handler._send_json(200, payload)
            return
        events: list[tuple[str | None, str]] = [
            (
                "message_start",
                json.dumps({"type": "message_start", "message": {**payload, "content": [], "stop_reason": None}}),
            )
        ]
        for i, blk in enumerate(blocks):
            events.append(
                ("content_block_start", json.dumps({"type": "content_block_start", "index": i, "content_block": blk}))
            )
            events.append(("content_block_stop", json.dumps({"type": "content_block_stop", "index": i})))
        events.append(
            (
                "message_delta",
                json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason},
                        "usage": {"output_tokens": len(out.token_ids)},
                    }
                ),
            )
        )
        events.append(("message_stop", json.dumps({"type": "message_stop"})))
        handler._send_sse(events)

    def close(self) -> None:
        if self._started:
            self._server.shutdown()
            self._thread.join(timeout=2.0)
            self._started = False
        if not self._closed:
            self._server.server_close()
            self._closed = True


@register_environment(["harness", "cli_harness"])
class HarnessEnvironment(SandboxEnvironment):
    """RL environment that executes an unmodified CLI agent harness inside a sandbox.

    Supports ``xerxes``, ``claude_code``, ``codex``, ``hermes``, ``pi``,
    ``opencode``, ``aider``, ``goose``, or any custom :class:`HarnessPreset`.
    """

    def __init__(
        self,
        prompt: str = "",
        *,
        harness: str | HarnessPreset = "xerxes",
        gateway: HarnessTITOGateway | None = None,
        files: collections.abc.Mapping[str, bytes | str] | None = None,
        repo_dir: str | None = None,
        test_patch: str | None = None,
        fail_to_pass: collections.abc.Sequence[str] = (),
        pass_to_pass: collections.abc.Sequence[str] = (),
        verify_script: str | None = None,
        subagent_mode: SubagentTrainingMode = "root_only",
        harness_timeout: float = 300.0,
        verify_timeout: float = 60.0,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str = "kiri",
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        affinity_key: str | int | None = None,
    ):
        cfg = (
            container_config
            if container_config is not None
            else ContainerConfig(network=True, timeout=harness_timeout, use_jail="required")
        )
        super().__init__(
            runtime=runtime,
            backend=backend,
            endpoint=endpoint,
            container_config=cfg,
            max_steps=1,
            affinity_key=affinity_key,
        )
        self._prompt = prompt
        self._preset = get_harness_preset(harness)
        self._gateway = gateway
        self._initial_files = dict(files) if files is not None else None
        self._repo_dir = repo_dir
        self._test_patch = test_patch
        self._fail_to_pass = list(fail_to_pass)
        self._pass_to_pass = list(pass_to_pass)
        self._verify_script = verify_script
        self._subagent_mode: SubagentTrainingMode = subagent_mode
        self._harness_timeout = float(harness_timeout)
        self._verify_timeout = float(verify_timeout)
        self._last_trajectories: list[OnlineRLTrajectory] = []
        self._reset_seed: int | None = None
        self._base_sha: str | None = None
        self._grading_snapshot: GradingSnapshot | None = None
        self._base_tracked_paths: set[str] = set()
        self._cached_repo_files: dict[str, bytes] | None = None

    _infra_failure_step = staticmethod(unscorable_step)

    @property
    def last_trajectories(self) -> list[OnlineRLTrajectory]:
        """TITO trajectory segments captured during the most recent harness run."""
        return list(self._last_trajectories)

    def _all_test_commands(self) -> list[str]:
        return [*self._fail_to_pass, *self._pass_to_pass]

    def _seed_workspace(self, session: SandboxSession, *, needs_base: bool, capture_metadata: bool) -> str | None:
        if self._repo_dir is not None:
            if self._cached_repo_files is None:
                copied, seed_err = _seed_repo_dir_files(self._repo_dir, None)
                if seed_err is not None or copied is None:
                    return seed_err or "Failed to seed workspace from repo_dir."
                self._cached_repo_files = copied
            if self._cached_repo_files:
                session.write_files(self._cached_repo_files)
        if self._initial_files:
            session.write_files(self._initial_files)
        if self._preset.requires_git or needs_base:
            git_check = session.exec(
                ["/bin/sh", "-c", 'test "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd -P)"'],
                env=_CLEAN_GIT_ENV,
                timeout=5.0,
            )
            if git_check.infra_error:
                return git_check.infra_error
            if not git_check.ok:
                init_err = init_clean_git_repo(session, email="rl@easydel.local", name="EasyDeL RL")
                if init_err is not None:
                    return init_err
        if not capture_metadata:
            return None
        base_sha, infra_err = capture_base_revision(session)
        if infra_err:
            return infra_err
        if needs_base and base_sha is None:
            return "Could not record the workspace base revision."
        if needs_base and base_sha is not None:
            entries, list_err = _list_base_tree(session, base_sha)
            if entries is None:
                return list_err or "Could not record the protected files of the base revision."
            all_paths = {path for _, _, path in entries}
            is_protected = _build_protected_predicate(self._test_patch, self._all_test_commands(), all_paths)
            self._base_tracked_paths = all_paths
            self._grading_snapshot = {path: (mode, blob_id) for mode, blob_id, path in entries if is_protected(path)}
        self._base_sha = base_sha
        return None

    def reset(self, seed: int | None = None) -> ResetResult:
        self._reset_seed = seed
        self._last_trajectories = []
        self._base_sha = None
        self._grading_snapshot = None
        self._base_tracked_paths = set()
        self._cached_repo_files = None
        needs_base = bool(self._test_patch or self._fail_to_pass or self._pass_to_pass)
        try:
            session = self._open_session(seed=seed)
            try:
                self._check_gateway_reachability(session)
            except ValueError:
                self.close()
                raise
            err = self._seed_workspace(session, needs_base=needs_base, capture_metadata=True)
        except SandboxInfraError as exc:
            self.close()
            return ResetResult(self._prompt, info={"env_error": True, "env_error_message": str(exc)})
        if err is not None:
            self.close()
            return ResetResult(self._prompt, info={"env_error": True, "env_error_message": err})
        return ResetResult(self._prompt, info={"harness": self._preset.name, "env_error": False})

    def _check_gateway_reachability(self, session: SandboxSession) -> None:
        """Fail fast when a container sandbox would be handed a loopback-only gateway URL.

        Raises:
            ValueError: If ``session`` runs in a container and the gateway advertises a loopback address.
        """
        if self._gateway is None or not isinstance(session, ContainerSandboxSession):
            return
        if self._gateway.advertised_is_loopback:
            raise ValueError(
                "HarnessEnvironment: the TITO gateway advertises "
                f"{self._gateway.advertised_base_url!r}, which a container sandbox cannot reach. Bind the gateway "
                "to a reachable interface (HarnessTITOGateway(host='0.0.0.0', ...)) and pass "
                "`advertise_host=` (container-visible host/IP) or `public_base_url=`."
            )

    def _grade_workspace(self) -> tuple[float, dict[str, tp.Any], str | None]:
        agent_session = self.session
        kill_err = kill_agent_processes(agent_session)
        if kill_err is not None:
            return 0.0, {}, kill_err

        uses_swe_grading = bool(self._test_patch or self._fail_to_pass or self._pass_to_pass)
        if uses_swe_grading:
            if self._base_sha is None or self._grading_snapshot is None:
                return 0.0, {}, "Base revision was not recorded at reset; cannot grade safely."
            safe_patch, untracked_files, tamper_reason, extract_infra = extract_agent_workspace_changes(
                agent_session,
                self._base_sha,
                test_patch=self._test_patch,
                test_commands=self._all_test_commands(),
                base_tracked_paths=self._base_tracked_paths,
            )
            if extract_infra is not None:
                return 0.0, {}, extract_infra
            if tamper_reason is not None:
                return (
                    0.0,
                    {"resolved": False, "workspace_tampered": True, "error": tamper_reason},
                    None,
                )

            step_count = self._step_count
            grade_session = self._open_session(seed=self._reset_seed)
            self._step_count = step_count
            seed_err = self._seed_workspace(grade_session, needs_base=True, capture_metadata=False)
            if seed_err is not None:
                return 0.0, {}, seed_err

            if safe_patch:
                ap_agent = apply_patch_via_stdin(grade_session, safe_patch, timeout=15.0)
                if ap_agent.infra_error:
                    return 0.0, {}, ap_agent.infra_error
                if not ap_agent.ok:
                    return 0.0, {"resolved": False, "error": ap_agent.stderr.strip()}, None

            if untracked_files:
                try:
                    grade_session.write_files(untracked_files)
                except (OSError, ValueError) as exc:
                    return (
                        0.0,
                        {"resolved": False, "workspace_tampered": True, "error": f"could not apply untracked files: {exc}"},
                        None,
                    )

            if self._test_patch:
                ap = apply_patch_via_stdin(grade_session, self._test_patch, timeout=15.0)
                if ap.infra_error:
                    return 0.0, {}, ap.infra_error
                if not ap.ok:
                    return 0.0, {"resolved": False, "error": ap.stderr.strip()}, None

            if self._verify_script is not None:
                return self._run_verify_script(grade_session, neutralize_hooks=False)

            f2p: list[bool] = []
            for cmd in self._fail_to_pass:
                ok, infra_err = run_verified_test_command(grade_session, cmd, timeout=self._verify_timeout)
                if infra_err is not None:
                    return 0.0, {}, infra_err
                f2p.append(ok)

            p2p: list[bool] = []
            for cmd in self._pass_to_pass:
                ok, infra_err = run_verified_test_command(grade_session, cmd, timeout=self._verify_timeout)
                if infra_err is not None:
                    return 0.0, {}, infra_err
                p2p.append(ok)

            ok = verification_resolved(f2p, p2p, has_f2p=bool(self._fail_to_pass), has_p2p=bool(self._pass_to_pass))
            return (1.0 if ok else 0.0), {"resolved": ok, "f2p_passed": sum(f2p), "p2p_passed": sum(p2p)}, None

        # Pure verify_script mode (no SWE test suite).
        if self._verify_script is not None:
            return self._run_verify_script(agent_session, neutralize_hooks=True)

        return 0.0, {"resolved": False, "f2p_passed": 0, "p2p_passed": 0}, None

    def _run_verify_script(
        self, session: SandboxSession, *, neutralize_hooks: bool
    ) -> tuple[float, dict[str, tp.Any], str | None]:
        assert self._verify_script is not None
        if neutralize_hooks:
            chmod_res = session.exec(
                [
                    "/bin/sh",
                    "-c",
                    "chmod u+rwx . 2>/dev/null; find . -maxdepth 6 ! -type l -exec chmod u+rwx -- {} + 2>/dev/null || true",
                ],
                timeout=15.0,
            )
            if chmod_res.infra_error:
                return 0.0, {}, chmod_res.infra_error
            name_tests: list[str] = []
            for name in [*_HARNESS_VERIFY_HOOK_BASENAMES, *(f"*{s}" for s in _HARNESS_VERIFY_HOOK_SUFFIXES)]:
                name_tests += [*(["-o"] if name_tests else []), "-name", name]
            clean_res = session.exec(
                ["find", ".", "-maxdepth", "6", "(", *name_tests, ")", "-prune", "-exec", "rm", "-rf", "--", "{}", "+"],
                timeout=15.0,
            )
            if clean_res.infra_error:
                return 0.0, {}, clean_res.infra_error
            restore: dict[str, bytes | str] = {}
            for source in (self._cached_repo_files, self._initial_files):
                if not source:
                    continue
                for path, content in source.items():
                    base = posixpath.basename(path)
                    if base in _HARNESS_VERIFY_HOOK_BASENAMES or base.endswith(_HARNESS_VERIFY_HOOK_SUFFIXES):
                        restore[path] = content
            if restore:
                try:
                    session.write_files(restore)
                except (OSError, ValueError):
                    pass

        if _parse_python_test_command(self._verify_script) is not None:
            ok, infra_err = run_verified_test_command(session, self._verify_script, timeout=self._verify_timeout)
            if infra_err is not None:
                return 0.0, {}, infra_err
            return (1.0 if ok else 0.0), {"resolved": ok}, None

        vres = session.exec(
            ["/bin/sh", "-c", self._verify_script],
            env=_HARNESS_VERIFY_ENV,
            timeout=self._verify_timeout,
        )
        if vres.infra_error:
            return 0.0, {}, vres.infra_error
        return (1.0 if vres.ok else 0.0), {"resolved": vres.ok, "verify_output": vres.combined_output(2048)}, None

    def step(self, action: str = "") -> StepResult:
        """Launch the configured CLI harness for ``prompt`` (or override ``action``) and grade."""
        if self._gateway is None:
            raise RuntimeError("HarnessEnvironment requires a live `gateway: HarnessTITOGateway`.")
        self._gateway.start()
        task_text = action.strip() or self._prompt
        _ep_id, api_key = self._gateway.open_episode()
        finalized = False

        template_vars = {
            "prompt": task_text,
            "model": self._gateway.model_name,
            "base_url": self._gateway.advertised_base_url,
            "base_url_v1": self._gateway.advertised_base_url_v1,
            "api_key": api_key,
        }
        try:
            try:
                cfg_files = self._preset.render_files(**template_vars)
                if cfg_files:
                    self.session.write_files(cfg_files)

                argv = self._preset.render_argv(**template_vars)
                env_vars = self._preset.render_env(**template_vars)
                run_res = self.session.exec(argv, env=env_vars, timeout=self._harness_timeout)
            except SandboxInfraError as exc:
                return self._infra_failure_step(exc)
            if run_res.infra_error:
                return self._infra_failure_step(run_res.infra_error)

            try:
                reward, grade_info, infra_err = self._grade_workspace()
            except SandboxInfraError as exc:
                return self._infra_failure_step(exc)
            if infra_err is not None:
                return self._infra_failure_step(infra_err)

            trajectories, gw_meta = self._gateway.finalize_episode(
                api_key,
                reward=reward,
                subagent_mode=self._subagent_mode,
            )
            finalized = True
            self._last_trajectories = trajectories
            info = {
                **grade_info,
                **gw_meta,
                "harness": self._preset.name,
                "harness_returncode": run_res.returncode,
                "harness_timed_out": run_res.timed_out,
                "env_error": False,
            }
            return StepResult(
                observation=run_res.combined_output(4096),
                reward=reward,
                terminated=not run_res.timed_out,
                truncated=run_res.timed_out,
                info=info,
            )
        finally:
            if not finalized:
                self._gateway.finalize_episode(api_key, reward=0.0, subagent_mode=self._subagent_mode)


__all__ = (
    "GatewayTurnOutput",
    "HarnessEnvironment",
    "HarnessPreset",
    "HarnessTITOGateway",
    "get_harness_preset",
    "list_harnesses",
    "register_harness",
)
