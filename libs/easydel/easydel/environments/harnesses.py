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
import os
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
    capture_base_revision,
    kill_agent_processes,
    restore_grading_files,
    unscorable_step,
    verification_resolved,
)

ProtocolKind = tp.Literal["openai_chat", "openai_responses", "anthropic_messages"]
SubagentTrainingMode = tp.Literal["root_only", "all_streams"]


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
        self.active_streams: dict[str, _SegmentBuilder] = {}
        self.sealed_segments: list[_SegmentBuilder] = []
        self.next_segment_id = 0

    def alloc_segment(self, stream_id: str, *, is_root: bool) -> _SegmentBuilder:
        seg = _SegmentBuilder(segment_id=self.next_segment_id, stream_id=stream_id, is_root=is_root)
        self.next_segment_id += 1
        return seg


def _normalize_openai_chat_messages(
    body: dict[str, tp.Any],
) -> tuple[list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None]:
    msgs = [dict(m) for m in body.get("messages", [])]
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


def _canonical_msg(msg: dict[str, tp.Any]) -> str:
    return json.dumps(
        {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls"),
            "tool_call_id": msg.get("tool_call_id"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _is_prefix_messages(prefix: list[dict[str, tp.Any]], full: list[dict[str, tp.Any]]) -> bool:
    if not prefix or len(prefix) > len(full):
        return False
    return all(_canonical_msg(p) == _canonical_msg(f) for p, f in zip(prefix, full[: len(prefix)], strict=True))


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
        """
        self._advertise_host = advertise_host
        self._public_base_url = public_base_url.rstrip("/") if public_base_url else None
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
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    self._send_json(400, {"error": {"message": "Invalid JSON request body"}})
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
                        turn = gateway._step_turn(ep, stream_id=stream_id, messages=msgs, tools=tools)
                        gateway._respond_openai_chat(self, turn, stream=stream)
                    elif clean_path.endswith("/responses"):
                        msgs, tools = _normalize_openai_responses_input(body)
                        turn = gateway._step_turn(ep, stream_id=stream_id, messages=msgs, tools=tools)
                        gateway._respond_openai_responses(self, turn, stream=stream)
                    elif clean_path.endswith("/messages"):
                        msgs, tools = _normalize_anthropic_messages(body)
                        turn = gateway._step_turn(ep, stream_id=stream_id, messages=msgs, tools=tools)
                        gateway._respond_anthropic_messages(self, turn, stream=stream)
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

        self._server = http.server.ThreadingHTTPServer((host, port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="ed-tito-gateway", daemon=True)
        self._started = False

    def start(self) -> HarnessTITOGateway:
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

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
            return f"http://{self._advertise_host}:{port}"
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
    ) -> GatewayTurnOutput:
        with ep.lock:
            is_root = (stream_id == "root") and (
                ("root" in ep.active_streams) or ((not ep.active_streams) and (not ep.sealed_segments))
            )
            seg = ep.active_streams.get(stream_id)
            if seg is not None and _is_prefix_messages(seg.messages, messages):
                delta_msgs = messages[len(seg.messages) :]
                if delta_msgs:
                    obs_tokens = [int(t) for t in self._encode_messages(delta_msgs, None)]
                    if obs_tokens:
                        start = len(seg.token_ids)
                        seg.token_ids.extend(obs_tokens)
                        seg.action_mask.extend([False] * len(obs_tokens))
                        seg.behavior_logps.extend([0.0] * len(obs_tokens))
                        seg.spans.append(
                            TrajectorySpan(start=start, stop=len(seg.token_ids), kind="observation", segment_id=0)
                        )
            else:
                if seg is not None and any(seg.action_mask):
                    ep.sealed_segments.append(seg)
                seg = ep.alloc_segment(stream_id, is_root=is_root or stream_id == "root")
                ep.active_streams[stream_id] = seg
                prompt_tokens = [int(t) for t in self._encode_messages(messages, tools)]
                if not prompt_tokens:
                    prompt_tokens = [0]
                seg.token_ids.extend(prompt_tokens)
                seg.action_mask.extend([False] * len(prompt_tokens))
                seg.behavior_logps.extend([0.0] * len(prompt_tokens))
                seg.spans.append(TrajectorySpan(start=0, stop=len(seg.token_ids), kind="prompt", segment_id=0))

            context_tokens = list(seg.token_ids)

        out = self._generate(context_tokens, messages, tools)
        act_ids = [int(t) for t in out.token_ids] or [0]

        assistant_msg: dict[str, tp.Any] = {"role": "assistant", "content": out.text}
        if out.tool_calls:
            assistant_msg["tool_calls"] = out.tool_calls

        with ep.lock:
            start = len(seg.token_ids)
            seg.token_ids.extend(act_ids)
            seg.action_mask.extend([True] * len(act_ids))
            if out.logprobs is not None and len(out.logprobs) == len(act_ids):
                seg.behavior_logps.extend(float(lp) for lp in out.logprobs)
            else:
                seg.has_all_logps = False
                seg.behavior_logps.extend([0.0] * len(act_ids))
            seg.spans.append(TrajectorySpan(start=start, stop=len(seg.token_ids), kind="action", segment_id=0))
            seg.versions.append(int(out.policy_version))
            seg.messages = [*messages, assistant_msg]

        return out

    def _respond_openai_chat(self, handler: tp.Any, out: GatewayTurnOutput, *, stream: bool) -> None:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        finish = "tool_calls" if out.tool_calls else out.finish_reason
        if not stream:
            msg: dict[str, tp.Any] = {"role": "assistant", "content": out.text}
            if out.reasoning_text:
                msg["reasoning_content"] = out.reasoning_text
            if out.tool_calls:
                msg["tool_calls"] = out.tool_calls
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
        if out.tool_calls:
            delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(out.tool_calls)]
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

    def _respond_openai_responses(self, handler: tp.Any, out: GatewayTurnOutput, *, stream: bool) -> None:
        rid = f"resp_{uuid.uuid4().hex[:12]}"
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
        if out.tool_calls:
            for tc in out.tool_calls:
                fn = tc.get("function", {})
                output_items.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex[:8]}",
                        "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                        "status": "completed",
                    }
                )
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

    def _respond_anthropic_messages(self, handler: tp.Any, out: GatewayTurnOutput, *, stream: bool) -> None:
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        blocks: list[dict[str, tp.Any]] = []
        if out.reasoning_text:
            blocks.append({"type": "thinking", "thinking": out.reasoning_text})
        if out.text:
            blocks.append({"type": "text", "text": out.text})
        if out.tool_calls:
            for tc in out.tool_calls:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed_inp = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    parsed_inp = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                        "name": fn.get("name", ""),
                        "input": parsed_inp,
                    }
                )
        stop_reason = "tool_use" if out.tool_calls else "end_turn"
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
            self._server.server_close()
            self._thread.join(timeout=2.0)
            self._started = False


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
        self._base_sha: str | None = None

    _infra_failure_step = staticmethod(unscorable_step)

    @property
    def last_trajectories(self) -> list[OnlineRLTrajectory]:
        """TITO trajectory segments captured during the most recent harness run."""
        return list(self._last_trajectories)

    def reset(self, seed: int | None = None) -> ResetResult:
        self._last_trajectories = []
        self._base_sha = None
        try:
            session = self._open_session(seed=seed)
            try:
                self._check_gateway_reachability(session)
            except ValueError:
                self.close()
                raise
            if self._repo_dir is not None:
                copied: dict[str, bytes] = {}
                root = os.path.abspath(self._repo_dir)
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames if d not in {".git", ".pytest_cache", "__pycache__"}]
                    for fname in filenames:
                        full = os.path.join(dirpath, fname)
                        if os.path.isfile(full):
                            with open(full, "rb") as f:
                                copied[os.path.relpath(full, root)] = f.read()
                if copied:
                    session.write_files(copied)
            if self._initial_files:
                session.write_files(self._initial_files)
            needs_base = bool(self._test_patch or self._fail_to_pass or self._pass_to_pass)
            if self._preset.requires_git or needs_base:
                res = session.run_bash(
                    "git rev-parse --is-inside-work-tree >/dev/null 2>&1 || "
                    "(git init -q && git config user.email 'rl@easydel.local' && "
                    "git config user.name 'EasyDeL RL' && git add -A && "
                    "git commit -q --allow-empty -m 'base')",
                    timeout=10.0,
                )
                if res.infra_error:
                    return ResetResult(self._prompt, info={"env_error": True, "env_error_message": res.infra_error})
            base_sha, infra_err = capture_base_revision(session)
            if infra_err:
                return ResetResult(self._prompt, info={"env_error": True, "env_error_message": infra_err})
            if needs_base and base_sha is None:
                return ResetResult(
                    self._prompt,
                    info={"env_error": True, "env_error_message": "Could not record the workspace base revision."},
                )
            self._base_sha = base_sha
        except SandboxInfraError as exc:
            return ResetResult(self._prompt, info={"env_error": True, "env_error_message": str(exc)})
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
        session = self.session
        kill_err = kill_agent_processes(session)
        if kill_err is not None:
            return 0.0, {}, kill_err
        if self._test_patch or self._fail_to_pass or self._pass_to_pass:
            if self._base_sha is None:
                return 0.0, {}, "Base revision was not recorded at reset; cannot grade safely."
            restore_err = restore_grading_files(session, self._base_sha, self._test_patch)
            if restore_err is not None:
                return 0.0, {}, restore_err
        if self._test_patch:
            session.write_files({".ed_harness_eval.patch": self._test_patch})
            ap = session.exec(["git", "apply", ".ed_harness_eval.patch"], timeout=10.0)
            if ap.infra_error:
                return 0.0, {}, ap.infra_error
            if not ap.ok:
                return 0.0, {"resolved": False, "error": ap.stderr.strip()}, None

        if self._verify_script is not None:
            vres = session.run_bash(self._verify_script, timeout=self._verify_timeout)
            if vres.infra_error:
                return 0.0, {}, vres.infra_error
            return (1.0 if vres.ok else 0.0), {"resolved": vres.ok, "verify_output": vres.combined_output(2048)}, None

        f2p = []
        for cmd in self._fail_to_pass:
            r = session.run_bash(cmd, timeout=self._verify_timeout)
            if r.infra_error:
                return 0.0, {}, r.infra_error
            f2p.append(r.ok)

        p2p = []
        for cmd in self._pass_to_pass:
            r = session.run_bash(cmd, timeout=self._verify_timeout)
            if r.infra_error:
                return 0.0, {}, r.infra_error
            p2p.append(r.ok)

        ok = verification_resolved(f2p, p2p, has_f2p=bool(self._fail_to_pass), has_p2p=bool(self._pass_to_pass))
        return (1.0 if ok else 0.0), {"resolved": ok, "f2p_passed": sum(f2p), "p2p_passed": sum(p2p)}, None

    def step(self, action: str = "") -> StepResult:
        """Launch the configured CLI harness for ``prompt`` (or override ``action``) and grade."""
        if self._gateway is None:
            raise RuntimeError("HarnessEnvironment requires a live `gateway: HarnessTITOGateway`.")
        self._gateway.start()
        task_text = action.strip() or self._prompt
        _ep_id, api_key = self._gateway.open_episode()

        template_vars = {
            "prompt": task_text,
            "model": self._gateway.model_name,
            "base_url": self._gateway.advertised_base_url,
            "base_url_v1": self._gateway.advertised_base_url_v1,
            "api_key": api_key,
        }
        try:
            cfg_files = self._preset.render_files(**template_vars)
            if cfg_files:
                self.session.write_files(cfg_files)

            argv = self._preset.render_argv(**template_vars)
            env_vars = self._preset.render_env(**template_vars)
            run_res = self.session.exec(argv, env=env_vars, timeout=self._harness_timeout)
        except SandboxInfraError as exc:
            self._gateway.finalize_episode(api_key, reward=0.0, subagent_mode=self._subagent_mode)
            return self._infra_failure_step(exc)
        if run_res.infra_error:
            self._gateway.finalize_episode(api_key, reward=0.0, subagent_mode=self._subagent_mode)
            return self._infra_failure_step(run_res.infra_error)

        reward, grade_info, infra_err = self._grade_workspace()
        if infra_err is not None:
            self._gateway.finalize_episode(api_key, reward=0.0, subagent_mode=self._subagent_mode)
            return self._infra_failure_step(infra_err)

        trajectories, gw_meta = self._gateway.finalize_episode(
            api_key,
            reward=reward,
            subagent_mode=self._subagent_mode,
        )
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


__all__ = (
    "GatewayTurnOutput",
    "HarnessEnvironment",
    "HarnessPreset",
    "HarnessTITOGateway",
    "get_harness_preset",
    "list_harnesses",
    "register_harness",
)
