# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Streaming assistant turns: one chat-model interface, one tool loop, live events.

A chat model (an OpenAI-compatible endpoint, or the plane host's Claude Code)
streams one provider round as events:

* ``{"type": "text", "delta": str}`` -- visible answer text as it arrives;
* ``{"type": "thinking", "delta": str}`` -- reasoning text, when the model shares it;
* exactly one final ``{"type": "message", "message": {"role": "assistant",
  "content": str | None, "tool_calls": [{"id", "type": "function",
  "function": {"name", "arguments": str}}]}}``.

The loop turns rounds into the events a client renders while it waits:
``step``, ``text``, ``thinking``, ``tool_call``, ``tool_result``, ``draft``,
``request`` and finally ``done`` (whose ``reply`` is authoritative). Tools run
in the plane with the asking user's principal; a model never runs one itself.
"""

from __future__ import annotations

import json
import logging
import typing as tp

import httpx

logger = logging.getLogger(__name__)

PROVIDER_TIMEOUT_S = 120.0
MAX_TOOL_RESULT_CHARS = 12000
#: How much of a tool result a client sees in its ``tool_result`` event.
TOOL_RESULT_PREVIEW_CHARS = 400

StreamEvent: tp.TypeAlias = dict[str, tp.Any]


class AssistantProviderError(RuntimeError):
    """The provider could not answer; the message never carries a key or token."""


class ChatModel(tp.Protocol):
    """One provider and model that streams a single round of a conversation."""

    model: str

    def stream(self, messages: list[dict[str, tp.Any]], tools: list[dict[str, tp.Any]]) -> tp.AsyncIterator[StreamEvent]:
        """Yield ``text``/``thinking`` events, then exactly one ``message`` event."""
        ...


class ToolRunner(tp.Protocol):
    """What the loop needs from the per-user tool set (``AssistantTools``)."""

    drafts: list[dict[str, tp.Any]]
    requests: list[tp.Any]

    async def call(self, name: str, arguments: dict[str, tp.Any]) -> tp.Any:
        """Run one tool as the asking user and return ``(name, ok, result)``."""
        ...


def scrub(text: str, secret: str | None) -> str:
    return text.replace(secret, "[redacted]") if secret else text


class OpenAICompatibleChat:
    """An OpenAI-compatible ``/chat/completions`` endpoint, streamed over SSE.

    A provider that ignores ``stream`` and answers with one JSON document is
    accepted too: its content becomes a single ``text`` event.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._key = api_key
        self._transport = transport

    def __repr__(self) -> str:
        return f"OpenAICompatibleChat(url={self.url!r}, model={self.model!r})"

    async def stream(
        self, messages: list[dict[str, tp.Any]], tools: list[dict[str, tp.Any]]
    ) -> tp.AsyncIterator[StreamEvent]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        body: dict[str, tp.Any] = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = [{"type": "function", "function": spec} for spec in tools]
            body["tool_choice"] = "auto"
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S, transport=self._transport) as client:
                async with client.stream("POST", self.url, json=body, headers=headers) as response:
                    if response.status_code != 200:
                        raw = (await response.aread()).decode("utf-8", "replace")
                        logger.warning("assistant provider answered HTTP %s", response.status_code)
                        detail = scrub(raw[:300], self._key)
                        raise AssistantProviderError(
                            f"the assistant provider answered HTTP {response.status_code}: {detail}"
                        )
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        message = _whole_message(await response.aread())
                        if message.get("content"):
                            yield {"type": "text", "delta": str(message["content"])}
                        yield {"type": "message", "message": message}
                        return
                    async for event in _sse_round(response):
                        yield event
        except httpx.HTTPError as exc:
            logger.warning("assistant provider unreachable: %s", type(exc).__name__)
            raise AssistantProviderError(f"the assistant provider is unreachable ({type(exc).__name__})") from None


def _whole_message(raw: bytes) -> dict[str, tp.Any]:
    try:
        message = json.loads(raw)["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise AssistantProviderError("the assistant provider returned an unreadable answer") from None
    if not isinstance(message, dict):
        raise AssistantProviderError("the assistant provider returned an unreadable answer")
    return message


async def _sse_round(response: httpx.Response) -> tp.AsyncIterator[StreamEvent]:
    content: list[str] = []
    calls: dict[int, dict[str, tp.Any]] = {}
    async for line in response.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or {}
        if not isinstance(delta, dict):
            continue
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            yield {"type": "thinking", "delta": reasoning}
        text = delta.get("content")
        if isinstance(text, str) and text:
            content.append(text)
            yield {"type": "text", "delta": text}
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            index = call.get("index")
            index = index if isinstance(index, int) else len(calls)
            slot = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if call.get("id"):
                slot["id"] = str(call["id"])
            function = call.get("function") or {}
            if isinstance(function, dict):
                if function.get("name") and not slot["function"]["name"]:
                    slot["function"]["name"] = str(function["name"])
                if isinstance(function.get("arguments"), str):
                    slot["function"]["arguments"] += function["arguments"]
    tool_calls = []
    for index in sorted(calls):
        slot = calls[index]
        slot["id"] = slot["id"] or f"call_{index}"
        tool_calls.append(slot)
    message: dict[str, tp.Any] = {"role": "assistant", "content": "".join(content) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    yield {"type": "message", "message": message}


def _preview(result: tp.Any) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= TOOL_RESULT_PREVIEW_CHARS else text[:TOOL_RESULT_PREVIEW_CHARS] + "…"


async def run_turn_stream(
    model: ChatModel,
    tools: ToolRunner,
    history: list[dict[str, tp.Any]],
    *,
    system: str,
    tool_specs: list[dict[str, tp.Any]],
    max_steps: int,
) -> tp.AsyncIterator[StreamEvent]:
    """Run at most ``max_steps`` rounds, yielding every step, delta and tool call as it happens."""

    messages: list[dict[str, tp.Any]] = [{"role": "system", "content": system}, *history]
    trace: list[dict[str, tp.Any]] = []
    for step in range(1, max_steps + 1):
        yield {"type": "step", "step": step, "max_steps": max_steps}
        message: dict[str, tp.Any] | None = None
        async for event in model.stream(messages, tool_specs):
            if event.get("type") == "message":
                message = event["message"]
            else:
                yield event
        if message is None:
            raise AssistantProviderError("the assistant provider ended its answer without a message")
        calls = [call for call in message.get("tool_calls") or [] if isinstance(call, dict)]
        if not calls:
            yield _done(str(message.get("content") or ""), step, trace, tools)
            return
        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            call_id = str(call.get("id", ""))
            drafts_before, requests_before = len(tools.drafts), len(tools.requests)
            plans_before = len(_plans(tools))
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
            except ValueError:
                yield {"type": "tool_call", "id": call_id, "name": name, "arguments": None}
                outcome_name, ok, result = name, False, {"error": "arguments were not a JSON object"}
            else:
                yield {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}
                outcome_name, ok, result = await tools.call(name, arguments)
            yield {"type": "tool_result", "id": call_id, "name": outcome_name, "ok": ok, "summary": _preview(result)}
            for draft in tools.drafts[drafts_before:]:
                yield {"type": "draft", "draft": draft}
            for request in tools.requests[requests_before:]:
                yield {"type": "request", "request": request.model_dump(mode="json")}
            for plan in _plans(tools)[plans_before:]:
                yield {"type": "plan", "plan": plan}
            content = json.dumps(result, default=str)
            if len(content) > MAX_TOOL_RESULT_CHARS:
                content = content[:MAX_TOOL_RESULT_CHARS] + '..."(truncated)"'
            trace.append({"tool": name, "ok": ok})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    yield _done(
        f"I stopped after {max_steps} steps without a final answer; ask a narrower question.", max_steps, trace, tools
    )


def _plans(tools: ToolRunner) -> list[dict[str, tp.Any]]:
    """Research plans a tool set proposed this turn; a runner without the channel has none."""

    plans = getattr(tools, "plans", None)
    return plans if isinstance(plans, list) else []


def _done(reply: str, steps: int, trace: list[dict[str, tp.Any]], tools: ToolRunner) -> StreamEvent:
    return {
        "type": "done",
        "reply": reply,
        "steps": steps,
        "tool_calls": trace,
        "drafts": list(tools.drafts),
        "requests": [request.model_dump(mode="json") for request in tools.requests],
        "plans": list(_plans(tools)),
    }


async def collect_turn(events: tp.AsyncIterator[StreamEvent]) -> dict[str, tp.Any]:
    """Drain a turn's events and return its ``done`` payload (the non-streaming answer)."""

    async for event in events:
        if event.get("type") == "done":
            return {key: value for key, value in event.items() if key != "type"}
    raise AssistantProviderError("the assistant turn ended without an answer")


__all__ = (
    "PROVIDER_TIMEOUT_S",
    "AssistantProviderError",
    "ChatModel",
    "OpenAICompatibleChat",
    "StreamEvent",
    "collect_turn",
    "run_turn_stream",
    "scrub",
)
