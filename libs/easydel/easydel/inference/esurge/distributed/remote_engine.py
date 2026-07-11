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

"""A client-side engine handle over the unified request plane.

:class:`RemoteEngineHandle` speaks to a remote eSurge engine's request
plane exactly like an in-group origin rank does — admissions forward as
``Admit`` messages, raw token deltas come back, and a private local
rendering stack (request registry, tokenizer/detokenizer workers, output
pipeline) turns them into ``RequestOutput`` objects — except it lives in a
process that is *not* part of the engine's JAX world. It authenticates
with the client role from the P0 handshake, bootstraps its admission
limits from the ``HelloOk`` engine spec, and exposes the engine-adapter
surface (``generate`` / ``stream`` / ``abort_request`` / limits /
``tokenizer``) so API servers and routers can treat it as an engine.

The transport is a single DEALER socket owned by one IO thread
(:class:`_ClientChannel`); caller threads enqueue sends through an outbox
with an inproc wakeup, mirroring the worker coordinator. All plane
semantics (blocking admits, delta application, abort/stop forwarding, id
retirement) are reused verbatim from :class:`OriginRequestPlane` — the
channel just presents the ``send_to_leader`` / ``set_plane_handler``
surface it expects.

Failure model is fail-fast per group: heartbeat loss or a ``Shutdown``
from the owner marks the handle dead, NACKs pending admissions, finishes
every in-flight request as aborted, and makes subsequent calls raise
:class:`RequestPlaneError`. There is no reconnect or request migration —
a router in front (``RoutedEngine``) simply stops picking this handle.
"""

from __future__ import annotations

import queue
import threading
import time
import typing
import uuid

from easydel.inference.sampling_params import SamplingParams
from easydel.workers.esurge.pipeline import WorkerManager

from ..engine.admission import RequestAdmission
from ..engine.output_pipeline import OutputPipeline
from ..engine.registry import RequestRegistry
from ..logger import logger
from . import wire
from .request_plane import OriginRequestPlane, RequestPlaneError, RequestPlaneUnavailable

if typing.TYPE_CHECKING:
    from ..esurge_engine import RequestOutput

_POLL_MS = 50

#: Client identities never coalesce; the rank field only labels logs/ids.
_CLIENT_RANK = -1


class _ClientChannel:
    """One DEALER + IO thread speaking the request plane's client role.

    Presents the two-method surface :class:`OriginRequestPlane` expects
    from its coordinator (``send_to_leader`` / ``set_plane_handler``) while
    owning the handshake, heartbeat liveness watch, and queue-stats intake
    itself.

    Args:
        leader_addr: Owner engine's control-plane host.
        control_port: Owner engine's control-plane port.
        auth_token: Shared plane secret.
        client_id: Stable identity string (also the ZMQ identity).
        connect_timeout_s: HelloOk wait budget.
        heartbeat_interval_s: Client beacon cadence (keeps ROUTER paths warm).
        heartbeat_timeout_s: Owner silence tolerated before declaring death.
        on_dead: Callback invoked once (with a reason) when the owner is lost.
    """

    def __init__(
        self,
        *,
        leader_addr: str,
        control_port: int,
        auth_token: str,
        client_id: str,
        connect_timeout_s: float = 15.0,
        heartbeat_interval_s: float = 1.0,
        heartbeat_timeout_s: float = 10.0,
        on_dead: typing.Callable[[str], None],
    ) -> None:
        self._endpoint = f"tcp://{leader_addr}:{int(control_port)}"
        self._auth_token = str(auth_token)
        self.client_id = str(client_id)
        self._connect_timeout_s = float(connect_timeout_s)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._on_dead = on_dead

        self.engine_spec: wire.EngineSpec | None = None
        self.queue_stats: wire.QueueStats = wire.QueueStats()
        self._plane_handler: typing.Callable[[wire.WireMessage, bytes | None], None] | None = None

        self._sock = None
        self._ctx = None
        self._outbox: queue.Queue = queue.Queue()
        self._wake_endpoint = f"inproc://esurge-client-wake-{id(self)}"
        self._wake_push = None
        self._wake_lock = threading.Lock()
        self._stop = threading.Event()
        self._io_thread: threading.Thread | None = None
        self._dead_reason: str | None = None

    # coordinator surface
    def set_plane_handler(self, handler: typing.Callable[[wire.WireMessage, bytes | None], None] | None) -> None:
        """Install the request-plane inbound handler (AdmitAck/OutputUpdate)."""
        self._plane_handler = handler

    def send_to_leader(self, header: wire.WireMessage, payload: bytes | None = None) -> None:
        """Queue a message for the owner from any thread."""
        self._outbox.put((wire.encode_message(header), payload))
        with self._wake_lock:
            if self._wake_push is not None:
                try:
                    self._wake_push.send(b"", flags=1)  # zmq.DONTWAIT
                except Exception:
                    pass

    # lifecycle
    @property
    def alive(self) -> bool:
        """Whether the owner link is up (handshaken and heartbeating)."""
        return self._dead_reason is None and self._sock is not None

    @property
    def dead_reason(self) -> str | None:
        """Why the link died, or ``None`` while healthy."""
        return self._dead_reason

    def start(self) -> None:
        """Connect, Hello as a client, and spawn the IO thread."""
        import zmq

        if self._sock is not None:
            return
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, self.client_id.encode())
        sock.setsockopt(zmq.LINGER, 500)
        sock.connect(self._endpoint)
        sock.send_multipart(
            [
                wire.encode_message(
                    wire.Hello(
                        rank=_CLIENT_RANK,
                        world_size=0,
                        config_fingerprint="",
                        auth=self._auth_token,
                        role=wire.ROLE_CLIENT,
                        client_id=self.client_id,
                    )
                )
            ]
        )
        if not sock.poll(int(self._connect_timeout_s * 1000)):
            sock.close(linger=0)
            raise RequestPlaneUnavailable(
                f"no HelloOk from engine owner at {self._endpoint} within {self._connect_timeout_s}s"
            )
        message = wire.decode_message(sock.recv_multipart()[0])
        if not isinstance(message, wire.HelloOk):
            sock.close(linger=0)
            raise RequestPlaneError(f"unexpected handshake reply {type(message).__name__}")
        self.engine_spec = message.engine_spec
        self._sock = sock
        self._io_thread = threading.Thread(target=self._io_loop, name="eSurgeRemoteEngineIO", daemon=True)
        self._io_thread.start()

    def close(self) -> None:
        """Stop the IO thread and close the socket."""
        self._stop.set()
        thread = self._io_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if self._sock is not None:
            self._sock.close(linger=500)
            self._sock = None

    def _mark_dead(self, reason: str) -> None:
        if self._dead_reason is not None:
            return
        self._dead_reason = reason
        try:
            self._on_dead(reason)
        except Exception:
            logger.exception("Remote engine handle: on_dead callback failed")

    # IO thread
    def _io_loop(self) -> None:
        import zmq

        sock = self._sock
        wake_pull = self._ctx.socket(zmq.PULL)
        wake_pull.bind(self._wake_endpoint)
        with self._wake_lock:
            self._wake_push = self._ctx.socket(zmq.PUSH)
            self._wake_push.connect(self._wake_endpoint)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        poller.register(wake_pull, zmq.POLLIN)
        last_owner_beat = time.monotonic()
        last_beat_sent = 0.0
        try:
            while not self._stop.is_set():
                try:
                    while True:
                        header, payload = self._outbox.get_nowait()
                        frames = [header]
                        if payload is not None:
                            frames.append(payload)
                        sock.send_multipart(frames)
                except queue.Empty:
                    pass

                now = time.monotonic()
                if now - last_beat_sent >= self._heartbeat_interval_s:
                    sock.send_multipart([wire.encode_message(wire.Heartbeat(rank=_CLIENT_RANK, ts=time.time()))])
                    last_beat_sent = now
                if now - last_owner_beat > self._heartbeat_timeout_s:
                    self._mark_dead(f"engine owner silent for more than {self._heartbeat_timeout_s}s")
                    return

                events = dict(poller.poll(_POLL_MS))
                if wake_pull in events:
                    while wake_pull.poll(0):
                        wake_pull.recv()
                if sock not in events:
                    continue
                frames = sock.recv_multipart()
                try:
                    message = wire.decode_message(frames[0])
                except Exception:
                    logger.warning("Remote engine handle: dropping undecodable frame")
                    continue
                if isinstance(message, wire.Heartbeat):
                    last_owner_beat = time.monotonic()
                    continue
                if isinstance(message, wire.QueueStats):
                    last_owner_beat = time.monotonic()
                    self.queue_stats = message
                    continue
                if isinstance(message, wire.Shutdown):
                    self._mark_dead(f"engine owner shut down ({message.reason})")
                    return
                last_owner_beat = time.monotonic()
                handler = self._plane_handler
                if handler is not None:
                    payload = frames[1] if len(frames) > 1 else None
                    try:
                        handler(message, payload)
                    except Exception:
                        logger.exception("Remote engine handle: plane handler failed")
        finally:
            with self._wake_lock:
                if self._wake_push is not None:
                    self._wake_push.close(linger=0)
                    self._wake_push = None
            wake_pull.close(linger=0)


class RemoteEngineHandle:
    """An engine-shaped client for one remote eSurge replica.

    Exposes the adapter surface (``generate``/``stream``/``abort_request``/
    ``esurge_name``/``max_model_len``/``max_num_seqs``/``tokenizer``/queue
    counters) backed entirely by the remote engine's request plane and a
    local rendering stack. Construction blocks until the handshake returns
    the engine spec and the tokenizer/detokenizer workers are up.

    Args:
        leader_addr: Owner engine's control-plane host.
        control_port: Owner engine's control-plane port.
        auth_token: Shared plane secret.
        client_id: Stable identity; auto-generated when omitted.
        tokenizer_source: Override for the tokenizer id/path; defaults to
            the owner's ``engine_spec.tokenizer_source``.
        connect_timeout_s: Handshake wait budget.
        admit_timeout_s: Per-admission ack deadline.
        heartbeat_timeout_s: Owner silence tolerated before the handle dies.
        decode_interval_tokens: Streaming decode cadence (tokens).
        decode_interval_secs: Streaming decode cadence (seconds).
    """

    def __init__(
        self,
        leader_addr: str,
        control_port: int,
        *,
        auth_token: str,
        client_id: str | None = None,
        tokenizer_source: str | None = None,
        connect_timeout_s: float = 15.0,
        admit_timeout_s: float = 120.0,
        heartbeat_timeout_s: float = 10.0,
        decode_interval_tokens: int = 4,
        decode_interval_secs: float = 0.05,
    ) -> None:
        self._channel = _ClientChannel(
            leader_addr=leader_addr,
            control_port=control_port,
            auth_token=auth_token,
            client_id=client_id or f"esurge-client-{uuid.uuid4().hex[:12]}",
            connect_timeout_s=connect_timeout_s,
            heartbeat_timeout_s=heartbeat_timeout_s,
            on_dead=self._on_owner_dead,
        )
        self._channel.start()
        spec = self._channel.engine_spec
        if spec is None or (not spec.tokenizer_source and not tokenizer_source):
            self._channel.close()
            raise RequestPlaneError(
                "engine owner returned no usable EngineSpec (missing tokenizer source); "
                "cannot bootstrap a remote handle"
            )
        self._spec = spec

        # Bootstrap failures after a successful handshake (bad tokenizer id,
        # worker spawn trouble) are permanent — close the transport and
        # re-raise as a plane error so callers don't retry them as
        # not-ready-yet.
        try:
            self._registry = RequestRegistry()
            self._worker_manager = WorkerManager(tokenizer_source or spec.tokenizer_source)
            self._tokenizer_client, self._detokenizer_client = self._worker_manager.start(
                detokenizer_max_states=max(64, int(spec.max_num_seqs) * 4),
                tokenizer_endpoint=None,
                detokenizer_endpoint=None,
            )
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source or spec.tokenizer_source)
        except Exception as exc:
            self._channel.close()
            manager = getattr(self, "_worker_manager", None)
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            raise RequestPlaneError(
                f"remote handle bootstrap failed (tokenizer source "
                f"{(tokenizer_source or spec.tokenizer_source)!r}): {exc}"
            ) from exc

        self._plane = OriginRequestPlane(
            coordinator=self._channel,
            submit_outputs=self._submit_outputs,
            alive=lambda: self._channel.alive,
            rank=_CLIENT_RANK,
            admit_timeout_s=admit_timeout_s,
            info=logger.debug,
        )
        self._output_pipeline = OutputPipeline(
            registry=self._registry,
            detokenizer_client=self._detokenizer_client,
            eos_token_ids=list(spec.eos_token_ids),
            decode_interval_tokens=decode_interval_tokens,
            decode_interval_secs=decode_interval_secs,
            on_stop_strings=self._on_stop_strings,
            on_activity=lambda: None,
            on_fatal=self._on_pipeline_fatal,
        )
        self._output_pipeline.start()
        self._admission = RequestAdmission(
            registry=self._registry,
            scheduler_submit=self._plane.submit_remote,
            tokenizer_client=self._tokenizer_client,
            tokenizer=self.tokenizer,
            context_config=_HandleContextConfig(),
            reserve_tokens=int(spec.reserve_tokens),
            max_model_len=int(spec.max_model_len),
            tool_parser_class=None,
            reasoning_parser_class=None,
            sampling_params_callback=lambda: None,
            generation_config_dict=None,
            primary_eos_token_id=(int(spec.eos_token_ids[0]) if spec.eos_token_ids else None),
            eos_token_ids=list(spec.eos_token_ids),
            extra_stops=[],
            ignore_stop_strings_in_reasoning=False,
            on_activity=lambda: None,
            info=logger.debug,
        )
        self._fatal: tuple[BaseException, str] | None = None

    # death paths
    def _on_owner_dead(self, reason: str) -> None:
        self._plane.fail_pending(reason)
        with self._registry.request_lock, self._registry.output_lock:
            for output in self._registry.outputs.values():
                if output.finished:
                    continue
                output.finished = True
                for completion in output.outputs:
                    if completion.finish_reason is None:
                        completion.finish_reason = "abort"
            self._registry.records.clear()
        self._registry.signal_all()
        logger.error("Remote engine handle %s lost its owner: %s", self._channel.client_id, reason)

    def _on_pipeline_fatal(self, exc: BaseException, tb: str) -> None:
        self._fatal = (exc, tb)
        self._registry.signal_all()

    def _on_stop_strings(self, hits: dict[str, str]) -> None:
        self._plane.forward_stop_hits(hits)

    def _submit_outputs(self, bundle) -> None:
        self._output_pipeline.submit(bundle)

    def _raise_if_unusable(self) -> None:
        if self._fatal is not None:
            raise RequestPlaneError(f"remote handle output pipeline failed: {self._fatal[0]}")
        if not self._channel.alive:
            raise RequestPlaneError(f"remote engine is unreachable: {self._channel.dead_reason or 'link closed'}")

    # adapter surface
    @property
    def esurge_name(self) -> str:
        """The remote engine's display name."""
        return self._spec.esurge_name

    @property
    def max_model_len(self) -> int:
        """The remote engine's context limit."""
        return int(self._spec.max_model_len)

    @property
    def max_num_seqs(self) -> int:
        """The remote engine's concurrency limit."""
        return int(self._spec.max_num_seqs)

    @property
    def page_size(self) -> int:
        """The remote engine's KV page size (prefix-affinity alignment)."""
        return int(self._spec.page_size)

    @property
    def alive(self) -> bool:
        """Whether the owner link is healthy."""
        return self._channel.alive and self._fatal is None

    @property
    def num_pending_requests(self) -> int:
        """Owner-reported waiting-queue depth (heartbeat-fresh)."""
        return int(self._channel.queue_stats.num_waiting)

    @property
    def num_running_requests(self) -> int:
        """Owner-reported running-set size (heartbeat-fresh)."""
        return int(self._channel.queue_stats.num_running)

    @property
    def pending_prefill_tokens(self) -> int:
        """Owner-reported uncomputed prompt tokens — the JSQ cost signal."""
        return int(self._channel.queue_stats.pending_prefill_tokens)

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = False,
        tool_parser_request: typing.Any | None = None,
    ) -> list[RequestOutput]:
        """Generate completions on the remote engine (blocking).

        Mirrors ``eSurge.generate`` semantics for the adapter surface.

        Args:
            prompts: One prompt or a batch.
            sampling_params: Generation parameters (default ``max_tokens=128``).
            request_id: Optional explicit id(s).
            use_tqdm: Accepted for signature parity; ignored.
            tool_parser_request: Accepted for signature parity; the handle
                runs no tool parsers.

        Returns:
            One finished ``RequestOutput`` per prompt.

        Raises:
            RequestPlaneError: If the owner rejects an admission or dies
                mid-generation.
        """
        del use_tqdm, tool_parser_request
        self._raise_if_unusable()
        if isinstance(prompts, str):
            prompts = [prompts]
        elif len(prompts) == 0:
            raise ValueError("Empty prompt list provided")
        if request_id is None:
            request_ids = [self._admission.generate_request_id() for _ in prompts]
        elif isinstance(request_id, str):
            if len(prompts) != 1:
                raise ValueError("request_id must be a list when providing multiple prompts.")
            request_ids = [request_id]
        else:
            request_ids = list(request_id)
            if len(request_ids) != len(prompts):
                raise ValueError("Length of request_id list must match number of prompts.")

        base_params = sampling_params or SamplingParams(max_tokens=128)
        for prompt, rid in zip(prompts, request_ids, strict=True):
            params = self._admission.prepare_sampling_params_for_request(base_params, request_id=rid, prompt=prompt)
            self._admission.add_request(rid, prompt, params)

        outputs: dict[str, RequestOutput] = {}
        while len(outputs) < len(request_ids):
            self._registry.wait_any(timeout=0.5)
            self._raise_if_unusable()
            with self._registry.output_lock:
                for rid in request_ids:
                    if rid in outputs:
                        continue
                    output = self._registry.outputs.get(rid)
                    if output is not None and output.finished:
                        outputs[rid] = output
        with self._registry.request_lock:
            for rid in request_ids:
                self._registry.events.pop(rid, None)
        return [outputs[rid] for rid in request_ids]

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        tool_parser_request: typing.Any | None = None,
    ) -> typing.Iterator[RequestOutput]:
        """Stream one request's incremental outputs from the remote engine.

        Args:
            prompts: One prompt (or single-element list).
            sampling_params: Generation parameters (default ``max_tokens=128``).
            request_id: Optional explicit id.
            tool_parser_request: Accepted for signature parity; ignored.

        Yields:
            ``RequestOutput`` snapshots as the origin-side pipeline renders
            new deltas, ending with ``finished=True``.
        """
        from easydel.inference.stream_protocol import compute_stream_delta_text

        from ..esurge_engine import CompletionOutput, RequestOutput

        del tool_parser_request
        self._raise_if_unusable()
        prompt = prompts[0] if isinstance(prompts, list) else prompts
        rid = request_id or self._admission.generate_request_id()
        params = self._admission.prepare_sampling_params_for_request(
            sampling_params or SamplingParams(max_tokens=128), request_id=rid, prompt=prompt
        )
        self._admission.add_request(rid, prompt, params)

        last_update_seq = -1
        last_accumulated_text = ""
        try:
            while True:
                self._registry.wait_request(rid, timeout=0.5)
                self._raise_if_unusable()
                snapshot = None
                with self._registry.output_lock:
                    output = self._registry.outputs.get(rid)
                    if output is not None and output.update_seq != last_update_seq:
                        last_update_seq = output.update_seq
                        # Snapshot-diff the delta against the last *yielded*
                        # text: pipeline updates between wakeups must not
                        # drop stream chunks.
                        snapshot_delta = compute_stream_delta_text(
                            output.accumulated_text, last_accumulated_text, output.delta_text
                        )
                        last_accumulated_text = output.accumulated_text
                        snapshot = RequestOutput(
                            request_id=output.request_id,
                            prompt=output.prompt,
                            prompt_token_ids=list(output.prompt_token_ids),
                            outputs=[
                                CompletionOutput(
                                    index=completion.index,
                                    text=completion.text,
                                    token_ids=list(completion.token_ids),
                                    finish_reason=completion.finish_reason,
                                    raw_text=completion.raw_text,
                                )
                                for completion in output.outputs
                            ],
                            finished=output.finished,
                            accumulated_text=output.accumulated_text,
                            delta_text=snapshot_delta,
                            raw_accumulated_text=output.raw_accumulated_text,
                            tokens_per_second=output.tokens_per_second,
                            num_generated_tokens=output.num_generated_tokens,
                            time_spent_generating=output.time_spent_generating,
                            first_token_time=output.first_token_time,
                            processing_time=output.processing_time,
                            update_seq=output.update_seq,
                            delta_seq=output.delta_seq,
                        )
                if snapshot is not None:
                    if (
                        not snapshot.finished
                        and snapshot.num_generated_tokens == 0
                        and not snapshot.delta_text
                    ):
                        continue
                    yield snapshot
                    if snapshot.finished:
                        return
        finally:
            with self._registry.request_lock:
                self._registry.events.pop(rid, None)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        """Chat interface over the remote engine (text-only).

        Renders the conversation through the handle's tokenizer chat
        template and forwards it as a plain generation. Multimodal content
        needs the owner's processor and is not supported through a remote
        handle.

        Args:
            messages: OpenAI-style message dicts.
            tools: Tool definitions rendered into the template; when
                present, special tokens are preserved so the model can emit
                tool calls.
            tool_choice: Accepted for signature parity.
            sampling_params: Generation parameters (default ``max_tokens=128``).
            request_id: Optional explicit id.
            stream: Yield incremental snapshots instead of blocking.
            chat_template: Optional template override.
            chat_template_kwargs: Extra ``apply_chat_template`` kwargs.

        Returns:
            A finished ``RequestOutput`` (``stream=False``) or an iterator
            of incremental snapshots (``stream=True``).
        """
        del tool_choice
        from ..engine.admission import clone_sampling_params
        from ..engine.chat_templating import format_chat_prompt

        params = clone_sampling_params(sampling_params or SamplingParams(max_tokens=128))
        if tools:
            params.skip_special_tokens = False
            params.logit_bias = None
        else:
            params.skip_special_tokens = True
        prompt = format_chat_prompt(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            chat_template=chat_template,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        if stream:
            return self.stream(prompt, sampling_params=params, request_id=request_id)
        return self.generate(prompt, sampling_params=params, request_id=request_id)[0]

    def iter_chat_completion_stream(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        """Bridge :meth:`chat` snapshots into OpenAI chat-completion chunks."""
        from ...stream_protocol import iter_chat_completion_stream_responses

        outputs = self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            sampling_params=sampling_params,
            request_id=request_id,
            stream=True,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        yield from iter_chat_completion_stream_responses(outputs, model=model)

    def iter_responses_stream(
        self,
        *,
        response_id: str,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        include_reasoning_summary: bool = False,
        final_response_overrides=None,
        created_at: int | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        """Bridge :meth:`chat` snapshots into OpenAI Responses-API frames."""
        from ...stream_protocol import iter_responses_stream_frames

        outputs = self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            sampling_params=sampling_params,
            request_id=request_id,
            stream=True,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        yield from iter_responses_stream_frames(
            outputs,
            response_id=response_id,
            model=model,
            include_reasoning_summary=include_reasoning_summary,
            final_response_overrides=final_response_overrides,
            created_at=created_at,
        )

    def abort_request(self, request_id: str) -> None:
        """Cancel a request: notify the owner, render the abort locally."""
        self._plane.notify_abort(request_id)
        with self._registry.request_lock, self._registry.output_lock:
            child_ids = [
                rid
                for rid, record in self._registry.records.items()
                if rid == request_id or record.parent_request_id == request_id
            ]
            for rid in child_ids:
                self._registry.records.pop(rid, None)
            output = self._registry.outputs.get(request_id)
            if output is not None and not output.finished:
                output.finished = True
                for completion in output.outputs:
                    if completion.finish_reason is None:
                        completion.finish_reason = "abort"
                output.update_seq += 1
        self._registry.signal(request_id)

    def close(self) -> None:
        """Tear down the transport, pipeline, and tokenizer workers."""
        self._channel.close()
        self._output_pipeline.stop()
        try:
            self._worker_manager.shutdown()
        except Exception:
            logger.debug("Remote engine handle worker shutdown failed", exc_info=True)


class _HandleContextConfig:
    """Default context policy for remote admissions (mirrors engine defaults)."""

    auto_truncate_prompt = True
    strict_context = False
    truncate_mode = "left"
    prefer_preserve_prompt = True
    auto_cap_new_tokens = True
    decode_truncated_prompt = False
