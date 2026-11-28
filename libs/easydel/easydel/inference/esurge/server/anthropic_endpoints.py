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

"""Anthropic Messages API endpoints for :class:`eSurgeApiServer`.

Provides native Anthropic API v1 compatibility directly on ``eSurgeApiServer``
without introducing a separate server class or external proxy:

- ``POST /v1/messages`` (non-streaming JSON and streaming SSE, including
  ``thinking``, ``text``, and ``tool_use`` content blocks)
- ``POST /v1/messages/count_tokens`` (exact prompt token count using the
  engine's tokenizer and chat template)

All requests flow through ``eSurgeApiServer``'s shared authentication
(``x-api-key`` and ``Authorization: Bearer``), rate limiting, quota
enforcement, generation slot budgeting, disconnect cancellation, and engine
routing (local ``eSurge``, ``RoutedEngine``, and DP-replica fronts).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import traceback
import typing as tp
import uuid
from collections.abc import Iterator

from eformer.loggings import get_logger
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ...openai_api_modules import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionDefinition,
    ToolDefinition,
    UsageInfo,
)
from ..engine.chat_templating import format_chat_prompt
from ..esurge_engine import RequestOutput, eSurge

logger = get_logger("eSurgeAnthropicAPI")

_STREAM_END = "end"
_STREAM_ERROR = "error"
_SSE_PING_INTERVAL_S = 15.0

_STATUS_TO_ANTHROPIC_ERROR_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


class AnthropicMessagesRequest(BaseModel):
    """Official Anthropic Messages API ``POST /v1/messages`` request schema."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Target served model identifier.")
    messages: list[dict[str, tp.Any]] = Field(..., description="Ordered conversation messages.")
    max_tokens: int = Field(..., ge=1, description="Maximum number of output tokens to generate.")
    system: str | list[dict[str, tp.Any]] | None = Field(
        default=None, description="Optional system prompt as plain text or structured content blocks."
    )
    tools: list[dict[str, tp.Any]] | None = Field(
        default=None, description="Optional tool definitions using Anthropic's ``input_schema`` format."
    )
    tool_choice: dict[str, tp.Any] | str | None = Field(
        default=None, description="Optional tool selection policy (auto, any, none, or specific tool)."
    )
    stream: bool = Field(default=False, description="Whether to stream Server-Sent Events.")
    temperature: float | None = Field(default=None, description="Sampling temperature.")
    top_p: float | None = Field(default=None, description="Nucleus sampling probability mass.")
    top_k: int | None = Field(default=None, description="Top-k token sampling filter.")
    stop_sequences: list[str] | None = Field(default=None, description="Custom stop sequences.")
    thinking: dict[str, tp.Any] | None = Field(default=None, description="Extended/adaptive thinking configuration.")
    output_config: dict[str, tp.Any] | None = Field(
        default=None, description="Optional output configuration (for example reasoning ``effort``)."
    )
    metadata: dict[str, tp.Any] | None = Field(default=None, description="Optional request metadata.")
    chat_template_kwargs: dict[str, tp.Any] | None = Field(
        default=None, description="Optional extra keyword arguments forwarded to the chat template."
    )


class AnthropicCountTokensRequest(BaseModel):
    """Official Anthropic ``POST /v1/messages/count_tokens`` request schema."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Target served model identifier.")
    messages: list[dict[str, tp.Any]] = Field(..., description="Ordered conversation messages.")
    system: str | list[dict[str, tp.Any]] | None = Field(default=None, description="Optional system prompt.")
    tools: list[dict[str, tp.Any]] | None = Field(default=None, description="Optional tool definitions.")
    tool_choice: dict[str, tp.Any] | str | None = Field(default=None, description="Optional tool choice.")
    thinking: dict[str, tp.Any] | None = Field(default=None, description="Optional thinking configuration.")
    output_config: dict[str, tp.Any] | None = Field(default=None, description="Optional output configuration.")
    chat_template_kwargs: dict[str, tp.Any] | None = Field(
        default=None, description="Optional extra chat template keyword arguments."
    )


class AnthropicUsage(BaseModel):
    """Token usage payload returned by Anthropic Messages responses."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    """Official Anthropic Messages API response body."""

    id: str
    type: tp.Literal["message"] = "message"
    role: tp.Literal["assistant"] = "assistant"
    model: str
    content: list[dict[str, tp.Any]]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicCountTokensResponse(BaseModel):
    """Official Anthropic token count response body."""

    input_tokens: int


def create_anthropic_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """Create a JSON response matching the official Anthropic error envelope."""
    resolved_type = error_type or _STATUS_TO_ANTHROPIC_ERROR_TYPE.get(int(status_code), "api_error")
    headers = {"request-id": request_id} if request_id else None
    return JSONResponse(
        status_code=int(status_code),
        content={
            "type": "error",
            "error": {
                "type": resolved_type,
                "message": str(message),
            },
        },
        headers=headers,
    )


def _format_anthropic_sse(event: str, data: dict[str, tp.Any]) -> str:
    """Serialize one official Anthropic SSE event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _flatten_anthropic_text_blocks(value: tp.Any) -> str:
    """Flatten string or Anthropic content-block lists into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "thinking":
                    parts.append(str(item.get("thinking", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(value)


def _convert_anthropic_image_block(block: dict[str, tp.Any]) -> dict[str, tp.Any] | None:
    """Convert an Anthropic ``image`` content block into an OpenAI-style ``image_url`` part."""
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    src_type = source.get("type")
    if src_type == "base64":
        media_type = str(source.get("media_type") or "image/png")
        data = str(source.get("data") or "")
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    if src_type == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": str(source["url"])}}
    return None


def _parse_json_object(raw: tp.Any) -> dict[str, tp.Any]:
    """Best-effort parse of tool call arguments into a JSON object dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"_raw": raw}


def _map_finish_reason_to_anthropic(finish_reason: str | None, *, has_tool_use: bool) -> str:
    """Map eSurge/OpenAI finish reasons to Anthropic ``stop_reason`` values."""
    if has_tool_use or finish_reason in ("tool_calls", "function_call"):
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop_sequence":
        return "stop_sequence"
    return "end_turn"


def _convert_anthropic_tools(raw_tools: list[dict[str, tp.Any]] | None) -> list[ToolDefinition] | None:
    """Translate Anthropic tool definitions into OpenAI :class:`ToolDefinition` objects."""
    if not raw_tools:
        return None
    converted: list[ToolDefinition] = []
    for raw in raw_tools:
        if not isinstance(raw, dict):
            continue
        if "function" in raw and isinstance(raw["function"], dict):
            fn = raw["function"]
            name = fn.get("name")
            if not name:
                continue
            converted.append(
                ToolDefinition(
                    type="function",
                    function=FunctionDefinition(
                        name=str(name),
                        description=fn.get("description"),
                        parameters=fn.get("parameters") or {"type": "object", "properties": {}},
                    ),
                )
            )
            continue
        name = raw.get("name")
        if not name:
            continue
        params = raw.get("input_schema")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        converted.append(
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name=str(name),
                    description=raw.get("description"),
                    parameters=params,
                ),
            )
        )
    return converted or None


def _convert_anthropic_tool_choice(raw_choice: dict[str, tp.Any] | str | None) -> str | dict[str, tp.Any] | None:
    """Translate Anthropic ``tool_choice`` into OpenAI ``tool_choice`` syntax."""
    if raw_choice is None:
        return None
    if isinstance(raw_choice, str):
        lowered = raw_choice.strip().lower()
        if lowered == "any":
            return "required"
        return lowered
    if not isinstance(raw_choice, dict):
        return None
    choice_type = str(raw_choice.get("type", "")).strip().lower()
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and raw_choice.get("name"):
        return {"type": "function", "function": {"name": str(raw_choice["name"])}}
    return None


def _convert_anthropic_messages(
    raw_messages: list[dict[str, tp.Any]],
    system: str | list[dict[str, tp.Any]] | None = None,
) -> list[dict[str, tp.Any]]:
    """Normalize Anthropic ``system`` + ``messages`` into engine-ready chat messages.

    Preserves assistant ``thinking`` blocks in ``reasoning_content`` so chat
    templates that distinguish thinking from visible text (for example Qwen3,
    DeepSeek, and GLM) render multi-turn trajectories natively.
    """
    messages: list[dict[str, tp.Any]] = []
    sys_text = _flatten_anthropic_text_blocks(system).strip()
    if sys_text:
        messages.append({"role": "system", "content": sys_text})

    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role", "user"))
        content = raw.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content)})
            continue

        if role == "assistant":
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            tool_calls: list[dict[str, tp.Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    thinking_val = str(block.get("thinking", "") or "")
                    if thinking_val:
                        thinking_parts.append(thinking_val)
                elif btype == "text":
                    text_val = str(block.get("text", "") or "")
                    if text_val:
                        text_parts.append(text_val)
                elif btype == "tool_use":
                    raw_input = block.get("input", {})
                    args_str = raw_input if isinstance(raw_input, str) else json.dumps(raw_input, ensure_ascii=False)
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name", "")),
                                "arguments": args_str,
                            },
                        }
                    )
            entry: dict[str, tp.Any] = {"role": "assistant", "content": "\n".join(text_parts)}
            if thinking_parts:
                entry["reasoning_content"] = "\n".join(thinking_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
        else:
            tool_msgs: list[dict[str, tp.Any]] = []
            user_text_parts: list[str] = []
            user_media_parts: list[dict[str, tp.Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_result":
                    tr_text = _flatten_anthropic_text_blocks(block.get("content", ""))
                    if block.get("is_error"):
                        tr_text = f"[ERROR]\n{tr_text}"
                    tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id", "")),
                            "content": tr_text,
                        }
                    )
                elif btype == "text":
                    text_val = str(block.get("text", "") or "")
                    if text_val:
                        user_text_parts.append(text_val)
                        user_media_parts.append({"type": "text", "text": text_val})
                elif btype == "image":
                    img_part = _convert_anthropic_image_block(block)
                    if img_part is not None:
                        user_media_parts.append(img_part)

            if tool_msgs:
                messages.extend(tool_msgs)
            has_media = any(p.get("type") != "text" for p in user_media_parts)
            if has_media:
                messages.append({"role": "user", "content": user_media_parts})
            elif user_text_parts or not tool_msgs:
                messages.append({"role": "user", "content": "\n".join(user_text_parts)})

    return messages


def _resolve_anthropic_chat_template_kwargs(
    request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    processor: tp.Any,
) -> dict[str, tp.Any] | None:
    """Derive ``chat_template_kwargs`` from Anthropic ``thinking`` and ``output_config`` fields."""
    kwargs: dict[str, tp.Any] = dict(request.chat_template_kwargs or {})

    thinking = request.thinking
    if isinstance(thinking, dict) and "enable_thinking" not in kwargs:
        ttype = str(thinking.get("type", "")).strip().lower()
        if ttype == "disabled":
            kwargs["enable_thinking"] = False
        elif ttype in ("enabled", "adaptive"):
            kwargs["enable_thinking"] = True

    template_src = str(getattr(processor, "chat_template", "") or "")
    out_cfg = request.output_config
    if isinstance(out_cfg, dict) and out_cfg.get("effort") and "reasoning_effort" not in kwargs:
        if "reasoning_effort" in template_src:
            effort = str(out_cfg["effort"]).strip().lower()
            if effort in ("high", "max") and "'xhigh'" in template_src and f"'{effort}'" not in template_src:
                effort = "xhigh"
            kwargs["reasoning_effort"] = effort

    if "preserve_thinking" not in kwargs and "preserve_thinking" in template_src:
        kwargs["preserve_thinking"] = True

    return kwargs or None


def _extract_tool_call_fields(tc: tp.Any) -> tuple[str, str, tp.Any]:
    """Extract ``(id, name, arguments)`` from a dict or Pydantic ``ToolCall``."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        tc_id = str(tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}")
        if isinstance(fn, dict):
            return tc_id, str(fn.get("name") or ""), fn.get("arguments")
        return tc_id, str(getattr(fn, "name", "") or ""), getattr(fn, "arguments", None)
    fn = getattr(tc, "function", None)
    tc_id = str(getattr(tc, "id", None) or f"toolu_{uuid.uuid4().hex[:12]}")
    name = str(getattr(fn, "name", "") or "") if fn is not None else ""
    args = getattr(fn, "arguments", None) if fn is not None else None
    return tc_id, name, args


class AnthropicEndpointsMixin:
    """Mixin adding official Anthropic Messages API endpoints to :class:`eSurgeApiServer`."""

    def _anthropic_endpoints(self) -> list[tp.Any]:
        """Return ``EndpointConfig`` entries for ``/v1/messages`` and ``/v1/messages/count_tokens``."""
        from ...inference_engine_interface import EndpointConfig

        return [
            EndpointConfig(
                path="/v1/messages",
                handler=self.anthropic_messages,
                methods=["POST"],
                tags=["Anthropic"],
                summary=(
                    "Create a message using the Anthropic Messages API v1 protocol. "
                    "Supports streaming SSE and non-streaming JSON responses, extended thinking blocks, "
                    "and tool use backed directly by eSurge."
                ),
            ),
            EndpointConfig(
                path="/v1/messages/count_tokens",
                handler=self.anthropic_count_tokens,
                methods=["POST"],
                tags=["Anthropic"],
                summary=(
                    "Count input tokens for an Anthropic Messages payload using the served model's "
                    "tokenizer and chat template."
                ),
            ),
        ]

    def _build_internal_chat_request(
        self,
        request: AnthropicMessagesRequest,
        messages: list[dict[str, tp.Any]],
        tools: list[ToolDefinition] | None,
        tool_choice: str | dict[str, tp.Any] | None,
        chat_template_kwargs: dict[str, tp.Any] | None,
    ) -> ChatCompletionRequest:
        """Build an internal ``ChatCompletionRequest`` so shared sampling/refinement hooks apply uniformly."""
        chat_messages = [
            ChatMessage(
                role=str(m.get("role", "user")),
                content=m.get("content", ""),
                tool_call_id=m.get("tool_call_id"),
                tool_calls=m.get("tool_calls"),
                reasoning_content=m.get("reasoning_content"),
            )
            for m in messages
        ]
        req_kwargs: dict[str, tp.Any] = {
            "model": request.model,
            "messages": chat_messages,
            "max_tokens": int(request.max_tokens),
            "stop": request.stop_sequences,
            "stream": bool(request.stream),
            "tools": tools,
            "tool_choice": tool_choice,
            "chat_template_kwargs": chat_template_kwargs,
        }
        if request.temperature is not None:
            req_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            req_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            req_kwargs["top_k"] = request.top_k
        chat_req = ChatCompletionRequest(**req_kwargs)
        refine_cb = getattr(self, "_refine_chat_request_callback", None)
        if refine_cb:
            refined = refine_cb(chat_req)
            if refined is not None:
                chat_req = refined
        return chat_req

    async def anthropic_count_tokens(
        self,
        request: AnthropicCountTokensRequest,
        raw_request: Request,
    ) -> tp.Any:
        """Handle ``POST /v1/messages/count_tokens``."""
        request_id = str(uuid.uuid4())
        try:
            self._authorize_endpoint(
                raw_request,
                endpoint="/v1/messages/count_tokens",
                model=request.model,
                requested_tokens=0,
                stream=False,
                uses_tools=bool(request.tools),
                reconcile_immediately=True,
            )
            adapter = self._get_adapter(request.model)
            esurge = adapter.esurge
            messages = _convert_anthropic_messages(request.messages, request.system)
            if self._messages_have_multimodal_content(messages):
                self._reject_local_media_references(messages)
            tools = _convert_anthropic_tools(request.tools)
            raw_tools = [t.model_dump(exclude_none=True) for t in tools] if tools else None
            chat_template_kwargs = _resolve_anthropic_chat_template_kwargs(request, adapter.processor)

            prompt_text: str | None = None
            tokenizer = getattr(esurge, "tokenizer", None) or adapter.processor
            if hasattr(tokenizer, "apply_chat_template") and messages:
                try:
                    prompt_text = format_chat_prompt(
                        tokenizer,
                        messages,
                        tools=raw_tools,
                        add_generation_prompt=True,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                except Exception:
                    prompt_text = None

            if prompt_text is None:
                prompt_text = "\n".join(str(m.get("content") or "") for m in messages)

            token_count = int(self._count_tokens(prompt_text, request.model))
            return AnthropicCountTokensResponse(input_tokens=token_count)
        except HTTPException as exc:
            return create_anthropic_error_response(exc.status_code, str(exc.detail), request_id=request_id)
        except Exception as exc:
            logger.exception("Error in /v1/messages/count_tokens: %s", exc)
            return create_anthropic_error_response(500, str(exc), request_id=request_id)

    async def anthropic_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> tp.Any:
        """Handle ``POST /v1/messages`` (non-streaming JSON and streaming SSE)."""
        request_id = str(uuid.uuid4())
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        try:
            self._authorize_endpoint(
                raw_request,
                endpoint="/v1/messages",
                model=request.model,
                requested_tokens=int(request.max_tokens),
                stream=bool(request.stream),
                uses_tools=bool(request.tools),
                reconcile_immediately=False,
            )

            adapter = self._get_adapter(request.model)
            esurge = adapter.esurge

            if not request.messages:
                raise HTTPException(status_code=400, detail="messages: at least one message is required")

            messages = _convert_anthropic_messages(request.messages, request.system)
            is_multimodal = self._messages_have_multimodal_content(messages)
            if is_multimodal:
                self._reject_local_media_references(messages)

            tools = _convert_anthropic_tools(request.tools)
            tool_choice = _convert_anthropic_tool_choice(request.tool_choice)
            chat_template_kwargs = _resolve_anthropic_chat_template_kwargs(request, adapter.processor)

            chat_req = self._build_internal_chat_request(
                request,
                messages,
                tools,
                tool_choice,
                chat_template_kwargs,
            )
            extracted_tools = self.extract_tools(request=chat_req)

            if request.stream:
                response = await self._handle_anthropic_streaming(
                    request=request,
                    chat_req=chat_req,
                    esurge=esurge,
                    messages=messages,
                    tools=extracted_tools,
                    request_id=request_id,
                    msg_id=msg_id,
                    raw_request=raw_request,
                )
                return self._track_api_key_stream(response, raw_request)

            return await self._handle_anthropic_completion(
                request=request,
                chat_req=chat_req,
                esurge=esurge,
                messages=messages,
                tools=extracted_tools,
                request_id=request_id,
                msg_id=msg_id,
                raw_request=raw_request,
            )

        except asyncio.CancelledError:
            self._release_api_key_reservation(raw_request)
            raise
        except HTTPException as exc:
            self._release_api_key_reservation(raw_request)
            return create_anthropic_error_response(exc.status_code, str(exc.detail), request_id=request_id)
        except Exception as exc:
            self._release_api_key_reservation(raw_request)
            traceback.print_exc()
            logger.exception("Error in /v1/messages: %s", exc)
            return create_anthropic_error_response(500, str(exc), request_id=request_id)

    def _build_anthropic_response(
        self,
        request: AnthropicMessagesRequest,
        output: RequestOutput,
        msg_id: str,
        raw_request: Request,
    ) -> AnthropicMessagesResponse:
        """Convert a completed eSurge :class:`RequestOutput` into an Anthropic Messages response."""
        completion_tokens = int(output.num_generated_tokens or 0)
        prompt_tokens = int(self._prompt_token_count_from_output(output))
        self.metrics.total_tokens_generated += completion_tokens

        tokens_per_second = getattr(output, "tokens_per_second", 0.0) or 0.0
        if self.metrics.average_tokens_per_second == 0:
            self.metrics.average_tokens_per_second = tokens_per_second
        elif tokens_per_second > 0:
            self.metrics.average_tokens_per_second = (
                self.metrics.average_tokens_per_second * 0.9 + tokens_per_second * 0.1
            )

        self._record_api_key_usage(raw_request, prompt_tokens, completion_tokens)

        blocks: list[dict[str, tp.Any]] = []
        stop_reason = "end_turn"
        if output.outputs:
            completion = output.outputs[0]
            reasoning = getattr(completion, "reasoning_content", None)
            if reasoning:
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": str(reasoning),
                        "signature": f"sig_{msg_id}_0",
                    }
                )
            tool_calls = getattr(completion, "tool_calls", None)
            if tool_calls:
                visible_text = self._resolved_tool_call_visible_text(completion, output)
                if visible_text:
                    blocks.append({"type": "text", "text": str(visible_text)})
                for tc in tool_calls:
                    tc_id, fn_name, raw_args = _extract_tool_call_fields(tc)
                    if not fn_name:
                        continue
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": fn_name,
                            "input": _parse_json_object(raw_args),
                        }
                    )
                stop_reason = _map_finish_reason_to_anthropic(
                    getattr(completion, "finish_reason", None), has_tool_use=True
                )
            else:
                text = getattr(completion, "text", None) or ""
                if text or not blocks:
                    blocks.append({"type": "text", "text": str(text)})
                stop_reason = _map_finish_reason_to_anthropic(
                    getattr(completion, "finish_reason", None), has_tool_use=False
                )

        cache_info = UsageInfo.cache_fields(output)
        cache_read = int(cache_info.get("cache_read") or 0) if isinstance(cache_info, dict) else 0
        cache_written = int(cache_info.get("cache_written") or 0) if isinstance(cache_info, dict) else 0

        return AnthropicMessagesResponse(
            id=msg_id,
            model=request.model,
            content=blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=AnthropicUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cache_creation_input_tokens=cache_written,
                cache_read_input_tokens=cache_read,
            ),
        )

    async def _handle_anthropic_completion(
        self,
        *,
        request: AnthropicMessagesRequest,
        chat_req: ChatCompletionRequest,
        esurge: eSurge,
        messages: list[dict[str, tp.Any]],
        tools: list[dict[str, tp.Any]] | None,
        request_id: str,
        msg_id: str,
        raw_request: Request,
    ) -> AnthropicMessagesResponse:
        """Handle non-streaming ``POST /v1/messages``."""
        async with self._acquire_generation_slot(
            endpoint="/v1/messages",
            request_id=request_id,
            model=request.model,
            raw_request=raw_request,
            stream=False,
        ):
            try:
                sampling_params = self._prepare_sampling_params(chat_req, esurge)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            disconnect_done = asyncio.Event()
            disconnect_task = asyncio.create_task(
                self._abort_non_stream_request_on_disconnect(
                    raw_request=raw_request,
                    esurge=esurge,
                    request_id=request_id,
                    endpoint="/v1/messages",
                    model=request.model,
                    done_event=disconnect_done,
                )
            )

            def _run_chat() -> RequestOutput:
                return tp.cast(
                    RequestOutput,
                    esurge.chat(
                        messages=messages,
                        tools=tools,
                        tool_choice=chat_req.tool_choice,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        stream=False,
                        chat_template_kwargs=chat_req.chat_template_kwargs,
                    ),
                )

            def _reconcile_cancelled(partial_output: RequestOutput) -> None:
                p_tokens = self._prompt_token_count_from_output(partial_output)
                c_tokens = int(getattr(partial_output, "num_generated_tokens", 0) or 0)
                if p_tokens > 0 or c_tokens > 0:
                    self.metrics.total_tokens_generated += c_tokens
                    self._record_api_key_usage(raw_request, p_tokens, c_tokens)

            try:
                output = await self._run_in_thread_with_cancel_abort(
                    _run_chat,
                    abort_fn=lambda: self._abort_request_after_handler_cancel(
                        esurge=esurge,
                        request_id=request_id,
                        endpoint="/v1/messages",
                        model=request.model,
                    ),
                    on_cancelled_result=_reconcile_cancelled,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            finally:
                disconnect_done.set()
                disconnect_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await disconnect_task

            return self._build_anthropic_response(request, output, msg_id, raw_request)

    async def _handle_anthropic_streaming(
        self,
        *,
        request: AnthropicMessagesRequest,
        chat_req: ChatCompletionRequest,
        esurge: eSurge,
        messages: list[dict[str, tp.Any]],
        tools: list[dict[str, tp.Any]] | None,
        request_id: str,
        msg_id: str,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming SSE ``POST /v1/messages``."""
        try:
            sampling_params = self._prepare_sampling_params(chat_req, esurge)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        self._check_generation_slot_capacity(
            endpoint="/v1/messages",
            request_id=request_id,
            model=request.model,
            raw_request=raw_request,
            stream=True,
        )

        async def generate_stream():
            async with self._acquire_generation_slot(
                endpoint="/v1/messages",
                request_id=request_id,
                model=request.model,
                raw_request=raw_request,
                stream=True,
            ):
                queue = self._start_observed_stream_task(
                    lambda: tp.cast(
                        Iterator[ChatCompletionStreamResponse],
                        esurge.iter_chat_completion_stream(
                            model=request.model,
                            messages=messages,
                            tools=tools,
                            tool_choice=chat_req.tool_choice,
                            sampling_params=sampling_params,
                            request_id=request_id,
                            chat_template_kwargs=chat_req.chat_template_kwargs,
                            **self._stream_usage_callback_kwargs(esurge.iter_chat_completion_stream, raw_request),
                        ),
                    ),
                    lambda: esurge.abort_request(request_id),
                    raw_request,
                )

                completed = False
                disconnected = False
                started_message = False
                block_index = 0
                open_block_kind: str | None = None
                open_tool_stream_idx: int | None = None
                tool_slots: dict[int, dict[str, tp.Any]] = {}
                has_tool_use = False
                finish_reason: str | None = None
                prompt_tokens = 0
                completion_tokens = 0
                cache_read_tokens = 0
                cache_written_tokens = 0

                def _emit_message_start() -> str:
                    nonlocal started_message
                    started_message = True
                    return _format_anthropic_sse(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "model": request.model,
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {
                                    "input_tokens": prompt_tokens,
                                    "output_tokens": 1,
                                    "cache_creation_input_tokens": cache_written_tokens,
                                    "cache_read_input_tokens": cache_read_tokens,
                                },
                            },
                        },
                    )

                def _close_open_block() -> list[str]:
                    nonlocal block_index, open_block_kind, open_tool_stream_idx
                    frames: list[str] = []
                    if open_block_kind is None:
                        return frames
                    if open_block_kind == "thinking":
                        frames.append(
                            _format_anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "signature_delta",
                                        "signature": f"sig_{msg_id}_{block_index}",
                                    },
                                },
                            )
                        )
                    frames.append(
                        _format_anthropic_sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": block_index},
                        )
                    )
                    block_index += 1
                    open_block_kind = None
                    open_tool_stream_idx = None
                    return frames

                try:
                    while True:
                        if await raw_request.is_disconnected():
                            try:
                                esurge.abort_request(request_id)
                            except Exception:
                                logger.debug("Failed to abort request %s after disconnect", request_id, exc_info=True)
                            disconnected = True
                            break

                        try:
                            kind, payload = await asyncio.wait_for(queue.get(), timeout=_SSE_PING_INTERVAL_S)
                        except TimeoutError:
                            if not started_message:
                                yield _emit_message_start()
                            yield _format_anthropic_sse("ping", {"type": "ping"})
                            continue

                        if kind == _STREAM_END:
                            completed = True
                            break
                        if kind == _STREAM_ERROR:
                            raise tp.cast(Exception, payload)

                        chunk = tp.cast(ChatCompletionStreamResponse, payload)
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            if usage.prompt_tokens:
                                prompt_tokens = int(usage.prompt_tokens)
                            if usage.completion_tokens:
                                completion_tokens = int(usage.completion_tokens)
                            if getattr(usage, "cache_read", None):
                                cache_read_tokens = int(usage.cache_read or 0)
                            if getattr(usage, "cache_written", None):
                                cache_written_tokens = int(usage.cache_written or 0)
                            self._observe_api_key_stream_usage(raw_request, prompt_tokens, completion_tokens)
                            if any(choice.finish_reason is not None for choice in chunk.choices):
                                self._record_api_key_usage(raw_request, prompt_tokens, completion_tokens)

                        if not started_message:
                            yield _emit_message_start()

                        if not chunk.choices:
                            continue

                        choice = chunk.choices[0]
                        if choice.finish_reason is not None:
                            finish_reason = str(choice.finish_reason)
                        delta = choice.delta
                        if delta is None:
                            continue

                        # 1. Reasoning / thinking deltas.
                        r_delta = getattr(delta, "reasoning_content", None)
                        if r_delta:
                            if open_block_kind != "thinking":
                                for frame in _close_open_block():
                                    yield frame
                                yield _format_anthropic_sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": {"type": "thinking", "thinking": ""},
                                    },
                                )
                                open_block_kind = "thinking"
                            yield _format_anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "thinking_delta", "thinking": str(r_delta)},
                                },
                            )

                        # 2. Visible text deltas.
                        c_delta = getattr(delta, "content", None)
                        if c_delta:
                            if open_block_kind != "text":
                                for frame in _close_open_block():
                                    yield frame
                                yield _format_anthropic_sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                                open_block_kind = "text"
                            yield _format_anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "text_delta", "text": str(c_delta)},
                                },
                            )

                        # 3. Tool call deltas.
                        tc_deltas = getattr(delta, "tool_calls", None)
                        if tc_deltas:
                            has_tool_use = True
                            for tcd in tc_deltas:
                                t_idx = int(getattr(tcd, "index", 0) or 0)
                                t_id = getattr(tcd, "id", None)
                                fn = getattr(tcd, "function", None)
                                fn_name = getattr(fn, "name", None) if fn is not None else None
                                fn_args = getattr(fn, "arguments", None) if fn is not None else None

                                slot = tool_slots.get(t_idx)
                                if slot is None:
                                    slot = {
                                        "id": str(t_id or f"toolu_{uuid.uuid4().hex[:12]}"),
                                        "name": str(fn_name or ""),
                                        "started": False,
                                        "pending_args": [],
                                    }
                                    tool_slots[t_idx] = slot
                                else:
                                    if t_id:
                                        slot["id"] = str(t_id)
                                    if fn_name:
                                        slot["name"] += str(fn_name)

                                if fn_args:
                                    slot["pending_args"].append(str(fn_args))

                                if not slot["started"] and slot["name"]:
                                    for frame in _close_open_block():
                                        yield frame
                                    yield _format_anthropic_sse(
                                        "content_block_start",
                                        {
                                            "type": "content_block_start",
                                            "index": block_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": slot["id"],
                                                "name": slot["name"],
                                                "input": {},
                                            },
                                        },
                                    )
                                    open_block_kind = "tool_use"
                                    open_tool_stream_idx = t_idx
                                    slot["started"] = True

                                if slot["started"] and open_tool_stream_idx == t_idx and slot["pending_args"]:
                                    partial = "".join(slot["pending_args"])
                                    slot["pending_args"].clear()
                                    yield _format_anthropic_sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": block_index,
                                            "delta": {"type": "input_json_delta", "partial_json": partial},
                                        },
                                    )

                    if disconnected:
                        return

                    if not started_message:
                        yield _emit_message_start()

                    # Flush any remaining pending arguments for tool calls whose deltas arrived out of order.
                    for t_idx in sorted(tool_slots.keys()):
                        slot = tool_slots[t_idx]
                        if not slot["started"] and slot["name"]:
                            for frame in _close_open_block():
                                yield frame
                            yield _format_anthropic_sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": slot["id"],
                                        "name": slot["name"],
                                        "input": {},
                                    },
                                },
                            )
                            open_block_kind = "tool_use"
                            open_tool_stream_idx = t_idx
                            slot["started"] = True
                        if slot["started"] and slot["pending_args"] and open_tool_stream_idx == t_idx:
                            partial = "".join(slot["pending_args"])
                            slot["pending_args"].clear()
                            yield _format_anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "input_json_delta", "partial_json": partial},
                                },
                            )

                    for frame in _close_open_block():
                        yield frame

                    self.metrics.total_tokens_generated += completion_tokens
                    self._record_api_key_usage(raw_request, prompt_tokens, completion_tokens)

                    yield _format_anthropic_sse(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": _map_finish_reason_to_anthropic(finish_reason, has_tool_use=has_tool_use),
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": max(1, completion_tokens)},
                        },
                    )
                    yield _format_anthropic_sse("message_stop", {"type": "message_stop"})

                except Exception as exc:
                    self._mark_stream_failure()
                    logger.exception("Error during /v1/messages streaming: %s", exc)
                    yield _format_anthropic_sse(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(exc),
                            },
                        },
                    )
                finally:
                    await self._close_stream_queue_and_wait(queue)
                    if not completed:
                        try:
                            esurge.abort_request(request_id)
                        except Exception:
                            logger.debug("Failed to abort unfinished /v1/messages %s", request_id, exc_info=True)

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "request-id": request_id,
                "X-Request-ID": request_id,
            },
        )
