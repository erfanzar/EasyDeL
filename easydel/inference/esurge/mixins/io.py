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

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from easydel.inference.sampling_params import SamplingParams

if TYPE_CHECKING:
    from ..esurge_engine import RequestOutput


class EngineIOMixin:
    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        """Generate completions for one or more prompts (blocking).

        Synchronous batch generation that waits for all completions to finish
        before returning. Suitable for batch processing scenarios where you need
        all results at once.

        Args:
            prompts: Single prompt string or list of prompts to generate from.
            sampling_params: Generation parameters controlling temperature, top_p,
                max_tokens, etc. Defaults to SamplingParams(max_tokens=128) if None.
            request_id: Optional request ID(s) for tracking. Auto-generated if None.
                Can be a single string (for single prompt) or list of strings.
            use_tqdm: Show progress bar for batch generation. Useful for tracking
                progress with multiple prompts.

        Returns:
            List of RequestOutput objects containing:
                - Generated text in the `text` field
                - Token IDs in the `token_ids` field
                - Performance metrics (tokens/sec, latency, etc.)
                - Finish reason ('stop', 'length', 'eos_token')

        Raises:
            RuntimeError: If background scheduler is not running. Call initiate() first.
            ValueError: If prompts and request_ids have mismatched lengths.

        Example:
            >>> # Single prompt generation
            >>> outputs = engine.generate(
            ...     "What is AI?",
            ...     SamplingParams(max_tokens=100, temperature=0.7)
            ... )
            >>> print(outputs[0].get_text())
            >>>
            >>> # Batch generation with progress bar
            >>> prompts = ["Question 1?", "Question 2?", "Question 3?"]
            >>> outputs = engine.generate(prompts, use_tqdm=True)
            >>> for i, output in enumerate(outputs):
            ...     print(f"Prompt {i}: {output.get_text()[:50]}...")
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        elif len(prompts) == 0:
            raise ValueError("Empty prompt list provided")

        if request_id is None:
            request_ids = [self._generate_request_id() for _ in prompts]
        elif isinstance(request_id, str):
            if len(prompts) != 1:
                raise ValueError("request_id must be a list when providing multiple prompts.")
            request_ids = [request_id]
        else:
            request_ids = list(request_id)
            if len(request_ids) != len(prompts):
                raise ValueError("Length of request_id list must match number of prompts.")

        base_sampling_params = sampling_params or SamplingParams(max_tokens=128)

        for prompt, req_id in zip(prompts, request_ids, strict=True):
            prompt_tokens = self._tokenize_prompt(req_id, prompt)
            effective_params = self._prepare_sampling_params_for_request(
                base_sampling_params,
                request_id=req_id,
                prompt=prompt,
            )
            self._add_request(req_id, prompt, effective_params, prompt_token_ids=prompt_tokens)

        outputs = []
        pbar = None
        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=len(prompts), desc="Generating")

        completed = set()

        if not self._scheduler_running:
            self._raise_if_scheduler_failed()
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        while len(completed) < len(prompts):
            self._output_event.wait(timeout=0.1)
            self._output_event.clear()
            self._raise_if_scheduler_failed()
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

        # Cleanup per-request events (outputs retained per max_request_outputs).
        with self._request_lock:
            for output in outputs:
                rid = output.request_id
                self._request_events.pop(rid, None)
                n_samples = len(output.outputs)
                if n_samples > 1:
                    for sample_idx in range(n_samples):
                        self._request_events.pop(f"{rid}-{sample_idx}", None)
        if self._max_request_outputs is not None:
            with self._output_lock:
                for output in outputs:
                    self._track_finished_output(output.request_id)
        return outputs

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> Iterator[RequestOutput]:
        """Stream generation output as tokens are produced.

        Yields RequestOutput objects incrementally as new tokens are generated,
        enabling real-time streaming of generated text. Perfect for interactive
        applications and chat interfaces.

        Args:
            prompts: Single prompt string or list with one prompt. For multiple
                prompts, use generate() instead.
            sampling_params: Generation parameters controlling temperature, top_p,
                max_tokens, etc. Defaults to SamplingParams(max_tokens=128).
            request_id: Optional request ID for tracking. Auto-generated if None.

        Yields:
            RequestOutput objects with incremental updates:
                - delta_text: Only the newly generated text since last yield
                - accumulated_text: Full text generated so far
                - finished: True when generation is complete
                - tokens_per_second: Current generation throughput
                - num_generated_tokens: Total tokens generated so far

        Raises:
            ValueError: If empty prompt list provided.
            RuntimeError: If scheduler not running or request setup fails.

        Example:
            >>> # Basic streaming
            >>> for output in engine.stream("Tell me a story"):
            ...     if output.delta_text:
            ...         print(output.delta_text, end="", flush=True)
            ...     if output.finished:
            ...         break
            >>>
            >>> # Monitor generation speed
            >>> for output in engine.stream("Long prompt here..."):
            ...     if output.delta_text:
            ...         print(output.delta_text, end="")
            ...     if output.num_generated_tokens % 10 == 0:
            ...         print(f"\n[{output.tokens_per_second:.1f} tok/s]", end="")
        """
        if isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("Empty prompt list provided")
            prompt = prompts[0]
        else:
            prompt = prompts

        if request_id is None:
            request_id = self._generate_request_id()

        base_sampling_params = sampling_params or SamplingParams(max_tokens=128)

        prompt_tokens = self._tokenize_prompt(request_id, prompt)
        effective_params = self._prepare_sampling_params_for_request(
            base_sampling_params,
            request_id=request_id,
            prompt=prompt,
        )
        self._add_request(request_id, prompt, effective_params, prompt_token_ids=prompt_tokens)

        if not self._scheduler_running:
            self._raise_if_scheduler_failed()
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        last_update_seq = -1
        last_accumulated_text = ""
        last_accumulated_reasoning = ""
        from ..esurge_engine import CompletionOutput, RequestOutput

        try:
            while True:
                req_event.wait(timeout=1.0)
                req_event.clear()
                self._raise_if_scheduler_failed()

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
                                    tool_calls=comp.tool_calls,
                                    reasoning_content=comp.reasoning_content,
                                )
                            )

                        snapshot_delta = self._compute_snapshot_delta_text(
                            ro.accumulated_text,
                            last_accumulated_text,
                            ro.delta_text,
                        )
                        snapshot_reasoning_delta = self._compute_snapshot_delta_text(
                            ro.reasoning_content or "",
                            last_accumulated_reasoning,
                            ro.delta_reasoning_content or "",
                        )

                        snapshot = RequestOutput(
                            request_id=ro.request_id,
                            prompt=ro.prompt,
                            prompt_token_ids=list(ro.prompt_token_ids),
                            outputs=outputs_copy,
                            finished=ro.finished,
                            metrics=dict(ro.metrics) if ro.metrics is not None else None,
                            accumulated_text=ro.accumulated_text,
                            delta_text=snapshot_delta,
                            tokens_per_second=ro.tokens_per_second,
                            num_generated_tokens=ro.num_generated_tokens,
                            time_spent_generating=ro.time_spent_generating,
                            first_token_time=ro.first_token_time,
                            processing_time=ro.processing_time,
                            update_seq=ro.update_seq,
                            delta_seq=ro.delta_seq,
                            tool_calls=ro.tool_calls,
                            delta_tool_calls=ro.delta_tool_calls,
                            reasoning_content=ro.reasoning_content,
                            delta_reasoning_content=snapshot_reasoning_delta or None,
                        )
                        last_update_seq = ro.update_seq
                        last_accumulated_text = ro.accumulated_text
                        last_accumulated_reasoning = ro.reasoning_content or ""

                if snapshot is not None:
                    yield snapshot
                    if snapshot.finished:
                        break
        finally:
            with self._output_lock:
                ro = self._request_outputs.get(request_id)
                n_samples = len(ro.outputs) if ro is not None else 0
                finished = ro.finished if ro is not None else True
            if finished:
                with self._request_lock:
                    self._request_events.pop(request_id, None)
                    if n_samples > 1:
                        for sample_idx in range(n_samples):
                            self._request_events.pop(f"{request_id}-{sample_idx}", None)
                if self._max_request_outputs is not None:
                    with self._output_lock:
                        self._track_finished_output(request_id)
            else:
                self.abort_request(request_id)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        """High-level chat interface compatible with vLLM and OpenAI APIs.

        Provides a convenient chat-based interface for conversational AI applications.
        Automatically formats messages using the model's chat template and handles
        both streaming and non-streaming responses. Supports multimodal content
        (images and videos) for vision-language models.

        Args:
            messages: List of message dictionaries representing the conversation history.
                Each message must have 'role' and 'content' keys. Content can be:
                - A string for text-only messages
                - A list of content items for multimodal messages (OpenAI format)

                Text-only example:
                    [{"role": "user", "content": "Hello!"}]

                Multimodal example:
                    [{"role": "user", "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": "Describe this image"}
                    ]}]

            tools: Optional list of tool/function definitions for function calling.
                Format should match the model's expected tool schema.
            sampling_params: Generation parameters controlling temperature, top_p,
                max_tokens, etc. Defaults to SamplingParams(max_tokens=128) if None.
            request_id: Optional unique identifier for tracking this request.
                Auto-generated if None.
            stream: If True, returns an iterator yielding incremental RequestOutput
                objects with delta_text for real-time streaming. If False, returns
                a single RequestOutput with the complete response.
            chat_template: Optional custom Jinja2 template to override the tokenizer's
                default chat template. Useful for models with non-standard formats.

        Returns:
            - If stream=False: Single RequestOutput object containing the complete
              assistant response with all metrics and generated text.
            - If stream=True: Iterator[RequestOutput] yielding incremental updates
              with delta_text containing newly generated text chunks.

        Raises:
            ValueError: If messages format is invalid or empty, or if multimodal
                content is provided but no processor is configured.
            RuntimeError: If scheduler is not running or tokenizer lacks chat template.

        Example:
            >>> # Text-only chat
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Explain quantum computing"}
            ... ]
            >>> response = engine.chat(messages)
            >>> print(response.get_text())
            >>>
            >>> # Multimodal chat with images (requires processor)
            >>> from PIL import Image
            >>> image = Image.open("photo.jpg")
            >>> messages = [
            ...     {"role": "user", "content": [
            ...         {"type": "image", "image": image},
            ...         {"type": "text", "text": "What's in this image?"}
            ...     ]}
            ... ]
            >>> response = engine.chat(messages)
            >>> print(response.get_text())
            >>>
            >>> # Streaming multimodal chat
            >>> for chunk in engine.chat(messages, stream=True):
            ...     print(chunk.delta_text, end="", flush=True)

        Note:ƒ
            For multimodal support, you must configure the engine with a processor
            during initialization: eSurge(..., processor=AutoProcessor.from_pretrained(...))
        """
        has_multimodal = self._messages_have_multimodal_content(messages)

        if has_multimodal:
            return self._chat_multimodal(
                messages=messages,
                tools=tools,
                sampling_params=sampling_params,
                request_id=request_id,
                stream=stream,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
            )
        else:
            prompt = self._format_chat_prompt(
                messages,
                tools=tools,
                add_generation_prompt=True,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
            )

            if stream:
                return self.stream(prompt, sampling_params=sampling_params, request_id=request_id)
            else:
                outs = self.generate(
                    prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    use_tqdm=False,
                )
                return outs[0]

    def _messages_have_multimodal_content(self, messages: list[dict]) -> bool:
        """Check if messages contain multimodal content (images/videos).

        Args:
            messages: List of chat message dictionaries.

        Returns:
            True if any message contains image or video content.
        """
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type in ("image", "image_url", "input_image", "video", "video_url", "input_video"):
                            return True
                        if any(k in item for k in ("image", "image_url", "video", "video_url")):
                            return True
        return False

    def _chat_multimodal(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        """Handle multimodal chat with images/videos.

        Internal method that processes vision-language model requests.

        Args:
            messages: Chat messages with multimodal content.
            tools: Optional tool definitions.
            sampling_params: Generation parameters.
            request_id: Optional request ID.
            stream: Whether to stream output.
            chat_template: Optional custom chat template.

        Returns:
            RequestOutput (non-streaming) or Iterator[RequestOutput] (streaming).

        Raises:
            ValueError: If no processor is configured for multimodal content.
        """
        if self._multimodal_manager is None:
            raise ValueError(
                "Multimodal content detected but no processor configured. "
                "Initialize eSurge with: processor=<tokenizer-or-processor> (e.g. AutoProcessor/AutoTokenizer)."
            )

        if request_id is None:
            request_id = self._generate_request_id()

        base_sampling_params = sampling_params or SamplingParams(max_tokens=128)

        images, videos = self._multimodal_manager.extract_media_from_messages(messages)

        pixel_values, image_grid_thw = self._multimodal_manager.process_images(images)
        pixel_values_videos, video_grid_thw = self._multimodal_manager.process_videos(videos)

        # Create mm_features for caching and batching support
        mm_features = []
        if images:
            mm_features.extend(self._multimodal_manager.process_images_to_features(images))
        if videos:
            mm_features.extend(self._multimodal_manager.process_videos_to_features(videos))

        prompt_token_ids = self._multimodal_manager.tokenize_multimodal(
            messages=messages,
            images=images,
            videos=videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        prompt = self._format_chat_prompt(
            messages,
            tools=tools,
            add_generation_prompt=True,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )

        effective_params = self._prepare_sampling_params_for_request(
            base_sampling_params,
            request_id=request_id,
            prompt=prompt,
        )

        # Add request with vision data
        self._add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=effective_params,
            prompt_token_ids=prompt_token_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_features=mm_features,
        )

        if not self._scheduler_running:
            self._raise_if_scheduler_failed()
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        if stream:
            return self._stream_multimodal_request(request_id)
        else:
            return self._wait_for_request(request_id)

    def _stream_multimodal_request(self, request_id: str) -> Iterator[RequestOutput]:
        """Stream output for a multimodal request.

        Args:
            request_id: ID of the multimodal request to stream.

        Yields:
            RequestOutput snapshots with incremental updates.

        Raises:
            RuntimeError: If request event is missing.
        """
        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        last_update_seq = -1
        last_accumulated_text = ""
        last_accumulated_reasoning = ""
        from ..esurge_engine import CompletionOutput, RequestOutput

        try:
            while True:
                req_event.wait(timeout=1.0)
                req_event.clear()
                self._raise_if_scheduler_failed()

                snapshot = None
                with self._output_lock:
                    ro = self._request_outputs.get(request_id)
                    if ro is None:
                        break

                    if ro.update_seq != last_update_seq:
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
                                    tool_calls=comp.tool_calls,
                                    reasoning_content=comp.reasoning_content,
                                )
                            )

                        snapshot_delta = self._compute_snapshot_delta_text(
                            ro.accumulated_text,
                            last_accumulated_text,
                            ro.delta_text,
                        )
                        snapshot_reasoning_delta = self._compute_snapshot_delta_text(
                            ro.reasoning_content or "",
                            last_accumulated_reasoning,
                            ro.delta_reasoning_content or "",
                        )

                        snapshot = RequestOutput(
                            request_id=ro.request_id,
                            prompt=ro.prompt,
                            prompt_token_ids=list(ro.prompt_token_ids),
                            outputs=outputs_copy,
                            finished=ro.finished,
                            metrics=dict(ro.metrics) if ro.metrics is not None else None,
                            accumulated_text=ro.accumulated_text,
                            delta_text=snapshot_delta,
                            tokens_per_second=ro.tokens_per_second,
                            num_generated_tokens=ro.num_generated_tokens,
                            time_spent_generating=ro.time_spent_generating,
                            first_token_time=ro.first_token_time,
                            processing_time=ro.processing_time,
                            update_seq=ro.update_seq,
                            delta_seq=ro.delta_seq,
                            tool_calls=ro.tool_calls,
                            delta_tool_calls=ro.delta_tool_calls,
                            reasoning_content=ro.reasoning_content,
                            delta_reasoning_content=snapshot_reasoning_delta or None,
                        )
                        last_update_seq = ro.update_seq
                        last_accumulated_text = ro.accumulated_text
                        last_accumulated_reasoning = ro.reasoning_content or ""

                if snapshot is not None:
                    yield snapshot
                    if snapshot.finished:
                        break
        finally:
            with self._output_lock:
                ro = self._request_outputs.get(request_id)
                n_samples = len(ro.outputs) if ro is not None else 0
                finished = ro.finished if ro is not None else True
            if finished:
                with self._request_lock:
                    self._request_events.pop(request_id, None)
                    if n_samples > 1:
                        for sample_idx in range(n_samples):
                            self._request_events.pop(f"{request_id}-{sample_idx}", None)
            else:
                self.abort_request(request_id)

    def _wait_for_request(self, request_id: str) -> RequestOutput:
        """Wait for a request to complete and return the output.

        Blocks until the request finishes generation.

        Args:
            request_id: ID of the request to wait for.

        Returns:
            Completed RequestOutput with all generated text.

        Raises:
            RuntimeError: If request event is missing.
        """
        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        output: RequestOutput | None = None
        while True:
            req_event.wait(timeout=1.0)
            req_event.clear()
            self._raise_if_scheduler_failed()
            with self._output_lock:
                output = self._request_outputs.get(request_id)
                if output is not None and output.finished:
                    break

        # Request is finished; cleanup per-request events (retention honors max_request_outputs).
        n_samples = len(output.outputs) if output is not None else 0
        with self._request_lock:
            self._request_events.pop(request_id, None)
            if n_samples > 1:
                for sample_idx in range(n_samples):
                    self._request_events.pop(f"{request_id}-{sample_idx}", None)

        assert output is not None
        if self._max_request_outputs is not None:
            with self._output_lock:
                self._track_finished_output(request_id)
        return output
