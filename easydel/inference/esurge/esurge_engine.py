# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import threading
import time
import traceback
import typing
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
from jax import numpy as jnp
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from easydel.inference.sampling_params import SamplingParams
from easydel.utils.helpers import get_logger

from .engine_types import EngineCoreOutputs
from .metrics import get_metrics_collector, initialize_metrics
from .request import EngineRequest, EngineRequestStatus
from .runners import eSurgeRunner
from .scheduler import Scheduler

if typing.TYPE_CHECKING:
    from easydel import AutoEasyDeLModelForCausalLM

logger = get_logger("eSurgeEngine")


@dataclass
class CompletionOutput:
    """Output of a single completion.

    Represents the generated output for a single completion within a batch request.
    Contains the generated text, token IDs, and optional probability information.

    Attributes:
        index: Position of this completion in the batch (0-indexed).
        text: The generated text string.
        token_ids: List of token IDs that were generated.
        cumulative_logprob: Cumulative log probability of the generated sequence.
        logprobs: Per-token log probabilities as dict mapping token_id to logprob.
        finish_reason: Reason for completion termination ('stop', 'length', 'eos_token', etc.).
    """

    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None


@dataclass
class RequestOutput:
    """Output of a generation request with comprehensive metrics.

    Contains the complete output for a generation request including generated text,
    performance metrics, and streaming support fields. Used for both batch and
    streaming generation modes.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Original prompt text.
        prompt_token_ids: Tokenized prompt as list of token IDs.
        outputs: List of CompletionOutput objects (one per n in sampling params).
        finished: Whether generation has completed.
        metrics: Dictionary of performance metrics (tokens, timing, etc.).
        accumulated_text: Full generated text accumulated so far.
        delta_text: Only the latest decoded text chunk (for streaming).
        tokens_per_second: Current generation throughput.
        num_generated_tokens: Total number of tokens generated.
        time_spent_generating: Total time spent in generation.
        first_token_time: Time to first token (TTFT) in seconds.
        processing_time: Total processing time including queuing.
        update_seq: Sequence number incremented on any update.
        delta_seq: Sequence number incremented only when delta_text changes.
    """

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = False
    metrics: dict[str, Any] | None = None

    accumulated_text: str = ""  # full text so far
    delta_text: str = ""  # only the latest decoded chunk
    tokens_per_second: float = 0.0
    num_generated_tokens: int = 0
    time_spent_generating: float = 0.0
    first_token_time: float | None = None
    processing_time: float = 0.0

    update_seq: int = 0
    delta_seq: int = 0

    def get_text(self) -> str:
        """Get the generated text from the first completion output.

        Returns:
            Generated text string, or empty string if no outputs.
        """
        return self.outputs[0].text if self.outputs else ""

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the request output.

        Returns:
            Dictionary containing key metrics: request_id, text, throughput,
            token count, timing, completion status and finish reason.
        """
        return {
            "request_id": self.request_id,
            "text": self.get_text(),
            "tokens_per_second": self.tokens_per_second,
            "num_generated_tokens": self.num_generated_tokens,
            "time_spent_generating": self.time_spent_generating,
            "finished": self.finished,
            "finish_reason": self.outputs[0].finish_reason if self.outputs else None,
        }


class eSurge:
    """High-level engine interface for text generation with eSurge.

    eSurge is a high-performance inference engine built on JAX that provides:
    - Efficient batched inference with paged attention
    - Continuous batching with background scheduling
    - Streaming generation with delta text tracking
    - Comprehensive monitoring and metrics
    - Thread-safe request handling

    The engine runs a background scheduler thread that continuously processes
    requests from the queue, enabling high throughput and low latency.
    """

    def __init__(
        self,
        model: str | AutoEasyDeLModelForCausalLM,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        max_model_len: int = 8192,
        min_input_pad: int = 16,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int | None = None,
        hbm_utilization: float = 0.85,
        page_size: int = 128,
        enable_prefix_caching: bool = True,
        auto_shard_model: bool = True,
        sharding_axis_dims: tuple[int, ...] = (1, 1, 1, -1, 1),
        compile_runner: bool = True,
        runner_verbose: bool = False,
        esurge_name: str | None = None,
        **kwargs,
    ):
        """Initialize the eSurge engine.

        Args:
            model: Model path (HuggingFace hub) or preloaded EasyDeL model instance.
            tokenizer: Tokenizer path or instance. If None, loads from model path.
            dtype: JAX dtype for model computations (default: bfloat16).
            max_model_len: Maximum sequence length the model can handle.
            min_input_pad: Minimum padding for input sequences.
            max_num_seqs: Maximum number of concurrent sequences.
            max_num_batched_tokens: Maximum tokens per batch (auto-computed if None).
            hbm_utilization: Target HBM memory utilization (0.0-1.0).
            page_size: Page size for paged attention KV cache.
            enable_prefix_caching: Enable caching of common prefixes.
            auto_shard_model: Automatically shard model across devices.
            sharding_axis_dims: Sharding configuration for model parallelism.
            compile_runner: JIT compile the runner for better performance.
            runner_verbose: Enable verbose logging in runner.
            esurge_name: Optional custom name for this engine instance.
            **kwargs: Additional configuration passed to model loading.

        Raises:
            ValueError: If tokenizer not provided and cannot be inferred.
        """
        from easydel import AutoEasyDeLModelForCausalLM, EasyDeLBaseConfigDict
        from easydel.layers.attention import AttentionMechanisms

        if jax.default_backend() != "tpu" and page_size <= 128:
            logger.warn(
                "for better performance and to utilize GPUs kernels (or even just on CPUs) "
                "better it's recommended to use `page_size>=256`."
            )
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.page_size = page_size
        if kwargs.pop("use_combined_forward", None) is not None:
            logger.warning("`use_combined_forward` is deprecated (the fused step will be used now).")
        if kwargs.pop("use_aot_forward", None) is not None:
            logger.warning("`use_aot_forward` is deprecated (the fused step will be used now).")
        if isinstance(model, str):
            self.model = AutoEasyDeLModelForCausalLM.from_pretrained(
                model,
                dtype=dtype,
                param_dtype=dtype,
                precision=jax.lax.Precision.DEFAULT,
                auto_shard_model=auto_shard_model,
                sharding_axis_dims=sharding_axis_dims,
                config_kwargs=EasyDeLBaseConfigDict(
                    attn_mechanism=kwargs.get("attn_mechanism", AttentionMechanisms.RAGGED_PAGE_ATTENTION),
                    attn_dtype=dtype,
                    kvdtype=dtype,
                    freq_max_position_embeddings=max_model_len,
                    mask_max_position_embeddings=max_model_len,
                    **kwargs.get("config_kwargs", {}),
                ),
                **{k: v for k, v in kwargs.items() if k not in ["attn_mechanism", "config_kwargs"]},
            )
        else:
            self.model = model

        if tokenizer is None:
            if isinstance(model, str):
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            else:
                raise ValueError("Tokenizer must be provided when using preloaded model")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self._monitoring_server = None
        self._dashboard = None
        self._dashboard_thread = None
        self._dashboard_urls = None
        self._monitoring_initialized = False
        self._esurge_name = esurge_name

        self.runner = eSurgeRunner(
            model=self.model,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_model_len=max_model_len,
            min_input_pad=min_input_pad,
            max_num_seqs=max_num_seqs,
            verbose=runner_verbose,
        )
        if compile_runner:
            self.runner.compile()

        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )

        # Streaming decode cadence
        self.decode_interval_tokens = 4
        self.decode_interval_secs = 0.02

        # State
        self._request_counter = 0
        self._active_requests: dict[str, dict] = {}
        self._request_outputs: dict[str, RequestOutput] = {}

        # Per-request events to support many concurrent streams
        self._request_events: dict[str, threading.Event] = {}

        # Locks and signals
        self._scheduler_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._output_lock = threading.RLock()
        self._counter_lock = threading.Lock()
        self._output_event = threading.Event()  # kept for generate()

        # Scheduler thread
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_running = False

        self.initiate()

    def _calculate_model_size(self, graphstate) -> str:
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            return f"{num_params / 1e9:.2f}"
        except Exception:
            return "unknown"

    def _get_model_type(self, model) -> str:
        return getattr(model.config, "model_type", "unknown").lower()

    def _get_model_name(self, model) -> str:
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(model.graphstate)
        return f"{model_type}-{model_size}b"

    @cached_property
    def esurge_name(self) -> str:
        return self._esurge_name or self._get_model_name(self.model)

    def initiate(self) -> None:
        """Start the background scheduler thread.

        Initiates a daemon thread that continuously runs the scheduler loop,
        processing requests from the queue and updating outputs. This must
        be called before using generate() or stream() methods.

        The scheduler thread will:
        1. Schedule requests from the waiting queue
        2. Execute model forward passes
        3. Update request outputs with generated tokens
        4. Signal waiting threads when updates are available
        """
        with self._scheduler_lock:
            if self._scheduler_running:
                logger.info("Scheduler loop is already running")
                return

            def _scheduler_loop():
                logger.info("Starting background scheduler loop")
                while self._scheduler_running:
                    try:
                        scheduler_output = self.scheduler.schedule()
                        model_output = self.runner.execute_model(scheduler_output)
                        engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
                        if engine_outputs:
                            self._process_engine_outputs(engine_outputs)
                    except Exception as e:
                        traceback.print_exc()
                        logger.error(f"Error in scheduler loop: {e}")
                        time.sleep(0.01)
                logger.info("Background scheduler loop stopped")

            self._scheduler_running = True
            self._scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
            self._scheduler_thread.start()
            logger.info("Background scheduler initiated")

    def terminate(self) -> None:
        """Stop the background scheduler thread.

        Gracefully shuts down the scheduler loop and waits for the thread
        to terminate. Should be called when the engine is no longer needed
        to free resources.
        """
        with self._scheduler_lock:
            if not self._scheduler_running:
                logger.info("Scheduler loop is not running")
                return
            logger.info("Stopping background scheduler loop...")
            self._scheduler_running = False
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5.0)
                if self._scheduler_thread.is_alive():
                    logger.warning("Scheduler thread did not stop gracefully")
                self._scheduler_thread = None
            logger.info("Background scheduler terminated")

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        """Generate completions for one or more prompts (blocking).

        Synchronous batch generation that waits for all completions to finish
        before returning. Suitable for batch processing scenarios.

        Args:
            prompts: Single prompt string or list of prompts.
            sampling_params: Generation parameters (temperature, max_tokens, etc.).
                Defaults to SamplingParams(max_tokens=128) if None.
            request_id: Optional request ID(s) for tracking. Auto-generated if None.
            use_tqdm: Show progress bar for batch generation.

        Returns:
            List of RequestOutput objects containing generated text and metrics.

        Raises:
            RuntimeError: If background scheduler is not running.

        Example:
            >>> outputs = engine.generate("What is AI?", SamplingParams(max_tokens=100))
            >>> print(outputs[0].get_text())
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if request_id is None:
            request_ids = [self._generate_request_id() for _ in prompts]
        elif isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = request_id

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=128)

        for prompt, req_id in zip(prompts, request_ids, strict=False):
            self._add_request(req_id, prompt, sampling_params)

        outputs = []
        pbar = None
        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=len(prompts), desc="Generating")

        completed = set()

        if not self._scheduler_running:
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        while len(completed) < len(prompts):
            self._output_event.wait(timeout=0.1)
            self._output_event.clear()
            with self._output_lock:
                for req_id in request_ids:
                    if req_id not in completed and req_id in self._request_outputs:
                        output = self._request_outputs[req_id]
                        if output.finished:
                            completed.add(req_id)
                            outputs.append(output)
                            if pbar:
                                pbar.update(1)

        if pbar:
            pbar.close()
        return outputs

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> Iterator[RequestOutput]:
        """Stream generation output as tokens are produced.

        Yields RequestOutput objects incrementally as new tokens are generated.
        The delta_text field contains only newly generated text since the last
        yield, while accumulated_text contains all text generated so far.

        Args:
            prompts: Single prompt string or list with one prompt.
            sampling_params: Generation parameters. Defaults to max_tokens=128.
            request_id: Optional request ID for tracking. Auto-generated if None.

        Yields:
            RequestOutput objects with incremental updates. Check delta_text
            for new content and finished flag for completion.

        Raises:
            ValueError: If empty prompt list provided.
            RuntimeError: If scheduler not running or request setup fails.

        Example:
            >>> for output in engine.stream("Tell me a story"):
            >>>     if output.delta_text:
            >>>         print(output.delta_text, end="", flush=True)
            >>>     if output.finished:
            >>>         break
        """
        if isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("Empty prompt list provided")
            prompt = prompts[0]
        else:
            prompt = prompts

        if request_id is None:
            request_id = self._generate_request_id()

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=128)

        self._add_request(request_id, prompt, sampling_params)

        if not self._scheduler_running:
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        last_update_seq = -1

        while True:
            req_event.wait(timeout=1.0)
            req_event.clear()

            snapshot = None
            with self._output_lock:
                ro = self._request_outputs.get(request_id)
                if ro is None:
                    break

                if ro.update_seq != last_update_seq:
                    # Snapshot without holding the lock during yield
                    outputs_copy = []
                    for comp in ro.outputs:
                        outputs_copy.append(
                            CompletionOutput(
                                index=comp.index,
                                text=comp.text,
                                token_ids=list(comp.token_ids),
                                cumulative_logprob=comp.cumulative_logprob,
                                logprobs=[dict(lp) for lp in comp.logprobs] if comp.logprobs else None,
                                finish_reason=comp.finish_reason,
                            )
                        )

                    snapshot = RequestOutput(
                        request_id=ro.request_id,
                        prompt=ro.prompt,
                        prompt_token_ids=list(ro.prompt_token_ids),
                        outputs=outputs_copy,
                        finished=ro.finished,
                        metrics=dict(ro.metrics) if ro.metrics is not None else None,
                        accumulated_text=ro.accumulated_text,
                        delta_text=ro.delta_text,
                        tokens_per_second=ro.tokens_per_second,
                        num_generated_tokens=ro.num_generated_tokens,
                        time_spent_generating=ro.time_spent_generating,
                        first_token_time=ro.first_token_time,
                        processing_time=ro.processing_time,
                        update_seq=ro.update_seq,
                    )
                    last_update_seq = ro.update_seq

            if snapshot is not None:
                yield snapshot
                if snapshot.finished:
                    break

        # Cleanup per-request event (output is preserved for generate or post-hoc access)
        with self._request_lock:
            self._request_events.pop(request_id, None)

    def _add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams) -> None:
        """Add a new request to the scheduler queue.

        Internal method that tokenizes the prompt, creates request tracking
        structures, and adds the request to the scheduler for processing.

        Args:
            request_id: Unique identifier for the request.
            prompt: Text prompt to generate from.
            sampling_params: Generation parameters.
        """
        tokenizer_output = self.tokenizer(prompt, return_tensors=None)
        token_ids = tokenizer_output["input_ids"]
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        self.scheduler.add_request(
            EngineRequest(
                request_id=request_id,
                prompt_token_ids=token_ids,
                sampling_params=sampling_params,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        )

        start_ts = time.perf_counter()
        ev = threading.Event()
        with self._request_lock:
            self._request_events[request_id] = ev
            self._active_requests[request_id] = {
                "prompt": prompt,
                "prompt_token_ids": token_ids,
                "generated_tokens": [],
                "last_decoded_index": 0,
                "start_time": start_ts,  # perf counter
                "first_token_time": None,
                "last_decode_time": start_ts,  # perf counter
            }

        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.start_request(request_id, len(token_ids))

        with self._output_lock:
            self._request_outputs[request_id] = RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=token_ids,
                outputs=[CompletionOutput(index=0, text="", token_ids=[])],
                finished=False,
                accumulated_text="",
                delta_text="",
                tokens_per_second=0.0,
                num_generated_tokens=0,
                time_spent_generating=0.0,
                first_token_time=None,
                processing_time=0.0,
                update_seq=0,
                delta_seq=0,
            )

    def _generate_request_id(self) -> str:
        """Generate a unique request ID.

        Returns:
            Unique request ID with format 'req-{uuid}-{counter}'.
        """
        with self._counter_lock:
            self._request_counter += 1
            return f"req-{uuid.uuid4().hex}-{self._request_counter}"

    def abort_request(self, request_id: str) -> None:
        """Abort an in-progress request.

        Marks the request as aborted and signals any waiting threads.
        The request will be removed from the scheduler queue if still waiting.

        Args:
            request_id: ID of the request to abort.
        """
        with self._scheduler_lock:
            if request_id in self.scheduler.requests:
                self.scheduler.requests[request_id].status = EngineRequestStatus.FINISHED_ABORTED

        with self._request_lock, self._output_lock:
            self._active_requests.pop(request_id, None)
            if request_id in self._request_outputs:
                ro = self._request_outputs[request_id]
                ro.finished = True
                ro.outputs[0].finish_reason = "aborted"
                ro.update_seq += 1

        # Notify both per-request and global waiters
        with self._request_lock:
            ev = self._request_events.get(request_id)
        if ev:
            ev.set()
        self._output_event.set()

    @property
    def num_pending_requests(self) -> int:
        """Get the number of requests waiting in queue.

        Returns:
            Count of requests in the waiting queue.
        """
        with self._scheduler_lock:
            return len(self.scheduler.waiting)

    @property
    def num_running_requests(self) -> int:
        """Get the number of actively running requests.

        Returns:
            Count of requests currently being processed.
        """
        with self._scheduler_lock:
            return len(self.scheduler.running)

    def _process_engine_outputs(self, engine_outputs: dict[int, EngineCoreOutputs]) -> None:
        """Process engine outputs and update request outputs (thread-safe).

        Core method that processes tokens from the model, performs incremental
        decoding, updates metrics, and signals waiting threads. Uses interval-based
        decoding to reduce tokenizer overhead during streaming.

        Args:
            engine_outputs: Dictionary mapping client IDs to engine outputs.

        The method:
        1. Extracts new tokens from engine outputs
        2. Performs interval-based decoding (every 4 tokens or 20ms)
        3. Updates accumulated and delta text
        4. Tracks performance metrics (TTFT, tokens/sec)
        5. Handles request completion
        6. Signals per-request events for streaming
        """
        metrics_collector = get_metrics_collector()

        # Update both request_data and public outputs atomically
        with self._request_lock, self._output_lock:
            for client_outputs in engine_outputs.values():
                for engine_output in client_outputs.outputs:
                    request_id = engine_output.request_id
                    ro = self._request_outputs.get(request_id)
                    rd = self._active_requests.get(request_id)
                    if ro is None or rd is None:
                        continue

                    text_changed = False
                    new_tokens = engine_output.new_token_ids
                    now = time.perf_counter()

                    if new_tokens:
                        rd["generated_tokens"].extend(new_tokens)
                        num_generated = len(rd["generated_tokens"])

                        # First token time
                        if rd["first_token_time"] is None and num_generated > 0:
                            rd["first_token_time"] = now - rd["start_time"]
                            if metrics_collector:
                                metrics_collector.record_first_token(request_id)

                        if metrics_collector:
                            metrics_collector.add_generated_tokens(request_id, len(new_tokens))

                        # Decode delta tokens on cadence
                        last_idx = rd["last_decoded_index"]
                        should_decode = (
                            num_generated - last_idx >= self.decode_interval_tokens
                            or (now - rd.get("last_decode_time", now)) >= self.decode_interval_secs
                        )
                        if should_decode and num_generated > last_idx:
                            delta = self.tokenizer.decode(
                                rd["generated_tokens"][last_idx:],
                                skip_special_tokens=True,
                            )
                            rd["last_decoded_index"] = num_generated
                            rd["last_decode_time"] = now

                            ro.accumulated_text += delta
                            ro.delta_text = delta
                            ro.delta_seq += 1
                            text_changed = True

                            comp = ro.outputs[0]
                            comp.text = ro.accumulated_text
                            comp.token_ids = list(rd["generated_tokens"])

                        # timings, tokens_per_second, etc...
                        ro.num_generated_tokens = len(rd["generated_tokens"])

                        # Timing and throughput
                        elapsed = now - rd["start_time"]
                        ro.processing_time = elapsed
                        ro.time_spent_generating = elapsed
                        ro.first_token_time = rd["first_token_time"]

                        if rd["first_token_time"] is not None and num_generated > 0:
                            generation_time = elapsed - rd["first_token_time"]
                            ro.tokens_per_second = num_generated / generation_time if generation_time > 0 else 0.0
                        else:
                            ro.tokens_per_second = 0.0

                        ro.num_generated_tokens = num_generated

                    # Completion handling
                    if engine_output.finished:
                        comp = ro.outputs[0]
                        ro.finished = True
                        comp.finish_reason = (
                            str(engine_output.finish_reason) if engine_output.finish_reason else "finished"
                        )

                        # Flush any remaining undecoded tokens
                        num_generated = len(rd["generated_tokens"])
                        last_idx = rd["last_decoded_index"]
                        if num_generated > last_idx:
                            delta = self.tokenizer.decode(
                                rd["generated_tokens"][last_idx:],
                                skip_special_tokens=True,
                            )
                            ro.accumulated_text += delta
                            ro.delta_text = delta
                            ro.delta_seq += 1  # NEW
                            comp.text = ro.accumulated_text
                            comp.token_ids = list(rd["generated_tokens"])
                            rd["last_decoded_index"] = num_generated
                            text_changed = True

                        elapsed = now - rd["start_time"]
                        num_prompt_tokens = (
                            len(rd["prompt_token_ids"]) if "prompt_token_ids" in rd else len(ro.prompt_token_ids)
                        )
                        num_generated_tokens = len(rd["generated_tokens"])

                        ro.processing_time = elapsed
                        ro.time_spent_generating = elapsed
                        ro.num_generated_tokens = num_generated_tokens
                        ro.first_token_time = rd.get("first_token_time")

                        if ro.first_token_time is not None and num_generated_tokens > 0:
                            generation_time = elapsed - ro.first_token_time
                            ro.tokens_per_second = num_generated_tokens / generation_time if generation_time > 0 else 0.0
                        else:
                            ro.tokens_per_second = 0.0

                        ro.metrics = {
                            "prompt_tokens": num_prompt_tokens,
                            "generated_tokens": num_generated_tokens,
                            "total_tokens": num_prompt_tokens + num_generated_tokens,
                            "processing_time": elapsed,
                            "first_token_time": ro.first_token_time,
                            "tokens_per_second": ro.tokens_per_second,
                        }

                        if metrics_collector:
                            metrics_collector.complete_request(
                                request_id,
                                finish_reason=comp.finish_reason,
                            )

                    ro.update_seq += 1
                    if text_changed or engine_output.finished:
                        ev = self._request_events.get(request_id)
                        if ev:
                            ev.set()

        self._output_event.set()

    def start_monitoring(
        self,
        dashboard_port: int = 11481,
        prometheus_port: int = 11184,
        dashboard_host: str = "0.0.0.0",
        enable_prometheus: bool = True,
        enable_dashboard: bool = True,
        enable_console: bool = False,
        log_file: str | None = None,
        log_interval: float = 10.0,
        history_size: int = 1000,
        enable_detailed_logging: bool = True,
    ) -> dict[str, str]:
        """Start monitoring services for the engine.

        Initializes various monitoring and observability services including
        Prometheus metrics, web dashboard, and console monitor.

        Args:
            dashboard_port: Port for web dashboard server.
            prometheus_port: Port for Prometheus metrics endpoint.
            dashboard_host: Host address to bind services to.
            enable_prometheus: Start Prometheus metrics server.
            enable_dashboard: Start web dashboard with real-time metrics.
            enable_console: Start console monitor with rich display.
            log_file: Optional file path for metrics logging.
            log_interval: Interval in seconds between metric logs.
            history_size: Number of historical metrics to retain.
            enable_detailed_logging: Enable detailed metric logging.

        Returns:
            Dictionary of service URLs:
            - 'prometheus': Prometheus metrics endpoint
            - 'dashboard': Web dashboard URL
            - 'health': Health check endpoint
            - 'api': API metrics endpoint

        Example:
            >>> urls = engine.start_monitoring(dashboard_port=8080)
            >>> print(f"Dashboard: {urls['dashboard']}")
        """
        if self._monitoring_initialized:
            logger.info("Monitoring already initialized for this eSurge instance")
            return self._dashboard_urls

        logger.info("Starting eSurge monitoring services...")

        if not get_metrics_collector():
            initialize_metrics(
                log_file=log_file,
                log_interval=log_interval,
                history_size=history_size,
                enable_detailed_logging=enable_detailed_logging,
            )
            logger.info(" Metrics collection initialized")

        urls = {}

        if enable_prometheus:
            try:
                from .monitoring import start_monitoring_server

                self._monitoring_server = start_monitoring_server(
                    prometheus_port=prometheus_port,
                    update_interval=1.0,
                )
                urls["prometheus"] = f"http://{dashboard_host}:{prometheus_port}/metrics"
                logger.info(f" Prometheus metrics: {urls['prometheus']}")
            except ImportError:
                logger.info(" Prometheus monitoring unavailable (install prometheus-client)")
            except Exception as e:
                logger.info(f" Failed to start Prometheus server: {e}")

        if enable_dashboard:
            try:
                from .dashboard import create_dashboard

                self._dashboard = create_dashboard(
                    host=dashboard_host,
                    port=dashboard_port,
                    debug=False,
                )

                def run_dashboard():
                    self._dashboard.run(log_level="warning")

                self._dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
                self._dashboard_thread.start()
                urls["dashboard"] = f"http://{dashboard_host}:{dashboard_port}"
                urls["health"] = f"http://{dashboard_host}:{dashboard_port}/health"
                urls["api"] = f"http://{dashboard_host}:{dashboard_port}/api/metrics"
                logger.info(f" Web dashboard: {urls['dashboard']}")
                logger.info(f" Health check: {urls['health']}")
            except ImportError:
                logger.info(" Web dashboard unavailable (install fastapi uvicorn)")
            except Exception as e:
                logger.info(f" Failed to start dashboard: {e}")

        if enable_console:
            try:
                from .monitoring import start_console_monitor

                logger.info(" Starting console monitor...")
                start_console_monitor(refresh_rate=1.0)
            except ImportError:
                logger.info(" Console monitor unavailable (install rich)")
            except Exception as e:
                logger.info(f" Failed to start console monitor: {e}")

        self._monitoring_initialized = True
        if urls:
            logger.info(" Monitoring services started successfully!")
            logger.info(" Metrics will be automatically collected during inference")
            if enable_dashboard:
                logger.info(f" Open {urls['dashboard']} to view real-time metrics")
        else:
            logger.info(" No monitoring services were started successfully")
        self._dashboard_urls = urls
        return urls

    def stop_monitoring(self) -> None:
        """Stop all monitoring services.

        Gracefully shuts down Prometheus server, dashboard, and console monitor
        if they are running.
        """
        if not self._monitoring_initialized:
            logger.info("No monitoring services to stop")
            return
        logger.info("Stopping eSurge monitoring services...")

        if self._monitoring_server:
            try:
                self._monitoring_server.stop()
                logger.info(" Prometheus server stopped")
            except Exception as e:
                logger.info(f" Error stopping Prometheus server: {e}")
            self._monitoring_server = None

        if self._dashboard_thread and self._dashboard_thread.is_alive():
            try:
                logger.info(" Dashboard server will stop with process")
            except Exception as e:
                logger.info(f" Error stopping dashboard: {e}")
            self._dashboard_thread = None
            self._dashboard = None

        self._monitoring_initialized = False
        logger.info(" Monitoring services stopped")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get current performance metrics summary.

        Returns:
            Dictionary containing:
            - requests_per_second: Current request throughput
            - average_latency: Average request latency
            - average_ttft: Average time to first token
            - average_throughput: Average tokens/second
            - total_completed: Total completed requests
            - total_failed: Total failed requests
            - total_tokens: Total tokens generated
            - active_requests: Currently active requests
            - queue_size: Pending requests in queue
            - running_requests: Currently running requests
        """
        metrics_collector = get_metrics_collector()
        if not metrics_collector:
            return {"error": "Metrics collection not initialized"}
        system_metrics = metrics_collector.get_system_metrics()
        return {
            "requests_per_second": system_metrics.requests_per_second,
            "average_latency": system_metrics.average_latency,
            "average_ttft": system_metrics.average_ttft,
            "average_throughput": system_metrics.average_throughput,
            "total_completed": system_metrics.total_requests_completed,
            "total_failed": system_metrics.total_requests_failed,
            "total_tokens": system_metrics.total_tokens_generated,
            "active_requests": len(self._active_requests),
            "queue_size": self.num_pending_requests,
            "running_requests": self.num_running_requests,
        }

    @property
    def monitoring_active(self) -> bool:
        return self._monitoring_initialized

    def __del__(self):
        if self._scheduler_running:
            try:
                self.terminate()
            except Exception:
                pass
        if self._monitoring_initialized:
            try:
                self.stop_monitoring()
            except Exception:
                pass
