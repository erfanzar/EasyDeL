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

import asyncio
import dataclasses
import queue
import time
import typing as tp

import jax
from eformer.common_types import NOT_GIVEN, _Empty
from eformer.loggings import get_logger

from ..decoders import SmartBytecodeDecoder
from ..sampling_params import SamplingParams
from .core import vDriver, vEngine
from .request_type import vSurgeRequest
from .utils import (
    ActiveRequest,
    ActiveRequestMetadata,
    AsyncMultifuture,
    ReturnSample,
    calculate_pefill_lengths,
    text_tokens_to_string,
)

if tp.TYPE_CHECKING:
    from easydel.infra.base_module import EasyDeLBaseModule
    from easydel.infra.utils import ProcessingClassType

logger = get_logger("vSurge")


@dataclasses.dataclass
class ProcessState:
    """Internal state for tracking a single generation sequence.

    Attributes:
        id (int): Identifier for this generation sequence (0 to n-1).
        active_request (ActiveRequest): The ActiveRequest object submitted to the driver.
        channel_iter (tp.AsyncIterator[list[ReturnSample]]): Asynchronous iterator for
            receiving results from the driver.
        driver_buffer (list[list[ReturnSample]]): Buffer for driver responses in
            server-side tokenization.
        finished_streaming (bool): True if this generation sequence has finished.
        current_step_text (tp.Union[str, list]): Text or token IDs generated in the current step.
            Type is `str` for server-side/bytecode, `list[int]` for client-side token IDs.
        all_token_ids (list[int]): Accumulated list of all token IDs generated so far.
        full_accumulated_text (tp.Union[str, list]): Full generated text or list of
            per-token decoded strings. Type is `str` for server-side/bytecode, `list[str]`
            (per-token decode) if client-side without bytecode.
        time_spent (float): Cumulative time spent computing for this sequence.
        tps (float): Tokens per second for this sequence.
        num_tokens (int): Total number of tokens generated for this sequence.
        decoded_text_upto_previous_step (tp.Optional[str]): Text decoded up to the
            previous step, used for diffing with `bytecode_decode`.
    """

    id: int
    active_request: ActiveRequest
    channel_iter: tp.AsyncIterator[list[ReturnSample]]
    driver_buffer: list[list[ReturnSample]] = dataclasses.field(default_factory=list)
    finished_streaming: bool = False
    current_step_text: str | list = ""
    all_token_ids: list[int] = dataclasses.field(default_factory=list)
    full_accumulated_text: str | list = ""
    time_spent: float = 0.0
    tps: float = 0.0
    num_tokens: int = 0
    decoded_text_upto_previous_step: str | None = None
    buffered_tokens: list[int] = dataclasses.field(default_factory=list)


class vSurge:
    """High-level interface for high-throughput text generation.

    vSurge orchestrates text generation using an underlying vDriver, managing
    request queuing, result processing, and tokenization/detokenization.
    It supports both streaming and non-streaming generation with features
    like bytecode decoding for handling malformed UTF-8 sequences.

    Features:
        - Batch and streaming generation
        - Client-side and server-side tokenization
        - Smart bytecode decoding for robust text handling
        - Asynchronous request processing
        - Multiple sampling strategies

    Attributes:
        driver: Underlying vDriver instance
        vsurge_name: Name of this vSurge instance
        bytecode_decode: Default bytecode decoding setting
        smart_decoder: Smart decoder for malformed characters
        processor: Tokenizer/processor from the driver

    Example:
        >>> driver = vDriver.from_pretrained("model-name")
        >>> surge = vSurge(driver, bytecode_decode=True)
        >>> surge.start()
        >>>
        >>> # Synchronous generation
        >>> response = surge.generate_once(
        ...     "Write a story about",
        ...     max_tokens=100
        ... )
        >>> print(response.text)
        >>>
        >>> # Asynchronous streaming
        >>> async for chunk in surge.generate(
        ...     "Explain quantum computing",
        ...     stream=True
        ... ):
        ...     print(chunk.text, end="")
    """

    def __init__(
        self,
        driver: vDriver,
        vsurge_name: str | None = None,
        bytecode_decode: bool = False,
    ):
        """Initialize vSurge instance.

        Args:
            driver: The underlying vDriver instance for model execution.
            vsurge_name: Optional name for this instance (defaults to driver name).
            bytecode_decode: Enable bytecode decoding for handling malformed UTF-8.
                When True, accumulates tokens before decoding to handle byte
                fallback tokens properly. Can be overridden per-request.
        """
        self._driver = driver
        self._vsurge_name = vsurge_name or driver.driver_name
        self._bytecode_decode = bytecode_decode
        self._smart_decoder = SmartBytecodeDecoder(self.processor)

    def compile(self):
        """Compile the underlying driver for optimized execution.

        Pre-compiles JAX functions for faster inference. Should be called
        before starting generation for best performance.
        """
        self.driver.compile()

    @property
    def vsurge_name(self) -> str:
        """Get the name of this vSurge instance.

        Returns:
            Name identifier for this instance.
        """
        return self._vsurge_name

    @property
    def bytecode_decode(self) -> bool:
        """Get the default bytecode decoding setting.

        Returns:
            True if bytecode decoding is enabled by default.
        """
        return self._bytecode_decode

    @property
    def driver(self) -> vDriver:
        """Get the underlying vDriver instance.

        Returns:
            The vDriver used for model execution.
        """
        return self._driver

    @property
    def smart_decoder(self) -> SmartBytecodeDecoder:
        """Get the smart bytecode decoder.

        Returns:
            Decoder for handling malformed UTF-8 sequences.
        """
        return self._smart_decoder

    @property
    def processor(self) -> ProcessingClassType:
        """Get the tokenizer/processor.

        Returns:
            The tokenizer used for text processing.
        """
        return self.driver.processor

    def start(self):
        """Start the underlying driver.

        Initializes the driver and makes it ready to accept generation requests.
        Must be called before submitting any requests.

        Returns:
            Result from driver.start() operation.
        """
        return self.driver.start()

    def stop(self):
        """Stops the underlying driver gracefully."""
        return self.driver.stop()

    def pause(self):
        """Pauses the underlying driver, halting new request processing."""
        return self.driver.pause()

    def resume(self):
        """Resumes the underlying driver after a pause."""
        return self.driver.resume()

    def replace_graphstate(self, state):
        """Replaces the graph state of the underlying driver.

        Args:
            state: The new graph state to apply.
        """
        return self.driver.replace_graphstate(state=state)

    def get_device_memory_stats(self) -> dict | None:
        """Gets device memory statistics from the driver, if available.

        Returns:
            tp.Optional[dict]: A dictionary of memory stats or None.
        """
        if hasattr(self.driver, "get_device_memory_stats"):
            return self.driver.get_device_memory_stats()
        return None

    def get_vdriver_metrics(self, aggregated: bool = True, window_size: int = 100) -> dict | None:
        """Gets operational metrics from the driver, if available.

        Args:
            aggregated (bool): If True, returns aggregated metrics. Defaults to True.
            window_size (int): Window size for non-aggregated metrics. Defaults to 100.

        Returns:
            tp.Optional[dict]: A dictionary of metrics or None.
        """
        if hasattr(self.driver, "get_metrics"):
            return self.driver.get_metrics(aggregated=aggregated, window_size=window_size)
        return None

    @classmethod
    def from_model(
        cls,
        model: EasyDeLBaseModule,
        processor: ProcessingClassType,
        max_concurrent_decodes: int | None = None,
        max_concurrent_prefill: int | None = None,
        extra_eos_token_ids: int | list[int] | None = None,
        prefill_lengths: int | list[int] | None = None,
        max_prefill_length: int | None = None,
        max_length: int | None = None,
        interleaved_mode: bool = False,
        slot_clear_steps: int = 0,
        vsurge_name: str | None = None,
        verbose: bool = True,
        bytecode_decode: bool = False,
        seed: int = 894,
        **kwargs,
    ) -> vSurge:
        """Instantiates vSurge from a model and processor.

        This class method configures and creates a `vSurge` instance, setting up
        the necessary driver and engine stack with appropriate parameters for
        decoding, prefill concurrency, memory management, and other generation settings.

        Args:
            model (EasyDeLBaseModule): The EasyDeL model instance to be used for inference.
            processor (ProcessingClassType): The tokenizer or processor compatible with the model.
            max_concurrent_decodes (tp.Optional[int]): Maximum number of decode calls
                allowed in parallel. Defaults to the number of available JAX devices.
            max_concurrent_prefill (tp.Optional[int]): Maximum number of concurrent
                prefill steps. Defaults to 1.
            prefill_lengths (tp.Optional[tp.Union[int, list[int]]]): Custom prefill lengths.
                If an integer is provided, it's treated as `max_prefill_length`, and
                a list of prefill lengths is computed. If a list of integers is provided,
                it's used directly, and its maximum value must match `max_prefill_length`.
                If None, prefill lengths are computed automatically based on
                `max_prefill_length` and `num_pages`.
            max_prefill_length (tp.Optional[int]): The maximum number of tokens allowed
                during the prefill phase. Defaults to half of `max_length`.
            max_length (tp.Optional[int]): The maximum sequence length for decoding.
                Defaults to the model's `granted_mask_max_position_embedding`.
            num_pages (tp.Optional[int]): Number of paging groups for memory partitioning.
                Used for computing `prefill_lengths` if not explicitly provided.
                Defaults to 128 in that case.
            tokens_per_page (tp.Optional[int]): Number of tokens allocated per page in
                the decoding workspace.
            interleaved_mode (bool): If True, enables interleaved decoding and prefill
                scheduling. Defaults to False.
            slot_clear_steps (int): Number of steps after which stale memory slots
                are cleared. Defaults to 0.
            vsurge_name (tp.Optional[str]): Optional name identifier for the created
                `vSurge` instance.
            verbose (bool): Enables logging and verbose driver output. Defaults to True.
            bytecode_decode (bool): Default bytecode decoding behavior for the new
                `vSurge` instance. Defaults to False.
            seed (int): Random seed for consistent decoding behavior. Defaults to 894.

        Returns:
            vSurge: A fully configured `vSurge` instance ready for inference.

        Raises:
            ValueError: If `prefill_lengths` is provided as a list and its maximum
                value does not match `max_prefill_length`.
            TypeError: If `prefill_lengths` is of an unsupported type.
        """
        max_length = max_length or model.config.granted_mask_max_position_embedding
        max_prefill_length = max_prefill_length or max_length // 2
        max_concurrent_prefill = max_concurrent_prefill or 1
        max_concurrent_decodes = max_concurrent_decodes or jax.device_count()

        actual_prefill_lengths: list[int]
        if isinstance(prefill_lengths, int):
            max_prefill_length = prefill_lengths
            actual_prefill_lengths = calculate_pefill_lengths(max_prefill_length=max_prefill_length)
            logger.info(f"Computed prefill lengths are {actual_prefill_lengths}")
        elif isinstance(prefill_lengths, list):
            if not all(isinstance(i, int) for i in prefill_lengths):
                raise ValueError("`prefill_lengths` must be a list of integers if provided as a list.")
            if max(prefill_lengths) != max_prefill_length:
                raise ValueError("The maximum value in `prefill_lengths` list must match `max_prefill_length`.")
            actual_prefill_lengths = prefill_lengths
        elif prefill_lengths is None:
            actual_prefill_lengths = calculate_pefill_lengths(max_prefill_length=max_prefill_length)
            logger.info(f"Computed prefill lengths are {actual_prefill_lengths}")
        else:
            raise TypeError("`prefill_lengths` must be an int, list of ints, or None.")

        return vSurge(
            driver=vDriver(
                engine=vEngine(
                    model=model,
                    processor=processor,
                    extra_eos_token_ids=extra_eos_token_ids,
                    max_concurrent_prefill=max_concurrent_prefill,
                    max_concurrent_decodes=max_concurrent_decodes,
                    prefill_lengths=actual_prefill_lengths,
                    max_prefill_length=max_prefill_length,
                    max_length=max_length,
                    seed=seed,
                ),
                interleaved_mode=interleaved_mode,
                slot_clear_steps=slot_clear_steps,
                verbose=verbose,
            ),
            vsurge_name=vsurge_name,
            bytecode_decode=bytecode_decode,
        )

    def count_tokens(self, text_or_conversation: str | list) -> int:
        """Counts tokens in a string or conversation list.

        Uses the underlying driver's processor. If the input is a list (assumed
        to be a conversation), it attempts to apply the chat template if available.

        Args:
            text_or_conversation (tp.Union[str, list]): A single string or a list of
                message dictionaries (e.g., `[{"role": "user", "content": "Hello"}]`).

        Returns:
            int: The total number of tokens in the input.

        Raises:
            ValueError: If the input type is unsupported or if tokenization fails.
        """
        try:
            if isinstance(text_or_conversation, str):
                return len(self.processor(text=text_or_conversation)["input_ids"])
            elif isinstance(text_or_conversation, list):
                if hasattr(self.processor, "apply_chat_template"):
                    tokenized = self.processor.apply_chat_template(
                        conversation=text_or_conversation,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    return len(tokenized if isinstance(tokenized, list) else tokenized.get("input_ids", []))
                else:
                    full_text = " ".join(
                        [msg.get("content", "") for msg in text_or_conversation if isinstance(msg.get("content"), str)]
                    )
                    return len(self.processor(text=full_text)["input_ids"])
            else:
                raise ValueError(f"Unsupported input type for token counting: {type(text_or_conversation)}")
        except Exception as e:
            logger.error(f"Error during token counting: {e}")
            raise ValueError(f"Failed to count tokens: {e}") from e

    async def _generate_batch(
        self,
        requests: list[vSurgeRequest],
        bytecode_decode: bool | _Empty = NOT_GIVEN,
    ) -> list[ReturnSample]:
        """Generates text completions for a batch of requests without streaming.

        Internal method to handle batch generation logic.

        Args:
            requests (list[vSurgeRequest]): A list of `vSurgeRequest` objects.
            bytecode_decode (tp.Union[bool, _Empty]): If provided (not `NOT_GIVEN`),
                overrides the instance's default `bytecode_decode` setting for this call.
                If `NOT_GIVEN`, the instance's default is used.

        Returns:
            list[ReturnSample]: A list of `ReturnSample` objects, one per input request.
            The `ReturnSample.accumulated_text` field holds the final generated text(s)
            for each of the 'n' generations within that request.
        """
        bytecode_decode = self.bytecode_decode if isinstance(bytecode_decode, _Empty) else bytecode_decode

        async def _collect_last_yielded_item(
            request_obj: vSurgeRequest,
        ) -> ReturnSample | None:
            last_item: ReturnSample | None = None
            async for result_list in self.complete(request_obj, bytecode_decode=bytecode_decode):
                if result_list:
                    last_item = result_list[0]
            return last_item

        tasks = [asyncio.create_task(_collect_last_yielded_item(req)) for req in requests]
        try:
            final_samples_per_request: list[ReturnSample | None] = await asyncio.gather(*tasks)
        except Exception as e:
            for task_to_cancel in tasks:
                if not task_to_cancel.done():
                    task_to_cancel.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise e

        processed_final_results: list[ReturnSample] = []
        for i, last_step_sample in enumerate(final_samples_per_request):
            original_req = requests[i]
            if last_step_sample is not None:
                processed_final_results.append(
                    ReturnSample(
                        text=last_step_sample.accumulated_text,
                        token_ids=last_step_sample.token_ids,
                        accumulated_text=last_step_sample.accumulated_text,
                        time_spent_computing=last_step_sample.time_spent_computing,
                        tokens_per_second=last_step_sample.tokens_per_second,
                        num_generated_tokens=last_step_sample.num_generated_tokens,
                    )
                )
            else:
                empty_text_val_single: str | list
                if original_req.is_client_side_tokenization and not bytecode_decode:
                    empty_text_val_single = []
                else:
                    empty_text_val_single = ""
                num_n = original_req.n if original_req.n > 0 else 0
                processed_final_results.append(
                    ReturnSample(
                        text=[empty_text_val_single] * num_n,
                        token_ids=[[]] * num_n,
                        accumulated_text=[empty_text_val_single] * num_n,
                        time_spent_computing=[0.0] * num_n,
                        tokens_per_second=[0.0] * num_n,
                        num_generated_tokens=[0] * num_n,
                    )
                )
        return processed_final_results

    def _create_default_return_sample_for_request(self, request: vSurgeRequest, bytecode_decode: bool) -> ReturnSample:
        """Creates a default ReturnSample for a request.

        This helper is used by `_generate_stream` to provide a consistent object
        structure when a stream ends or if it never produces data, ensuring that
        the yielded list always contains `ReturnSample` objects.

        Args:
            request (vSurgeRequest): The original `vSurgeRequest` for which to create
                a default `ReturnSample`.
            bytecode_decode (bool): The effective bytecode decoding setting, used to
                determine the structure of empty text/token fields.

        Returns:
            ReturnSample: A `ReturnSample` instance with default (empty/zeroed) values,
            structured according to `request.sampling_params.n` and the effective `bytecode_decode`
            and `request.is_client_side_tokenization` settings.
        """
        num_n = request.sampling_params.n if request.sampling_params.n > 0 else 0
        empty_text_val_single: str | list
        if request.is_client_side_tokenization and not bytecode_decode:
            empty_text_val_single = []
        else:
            empty_text_val_single = ""

        return ReturnSample(
            text=[empty_text_val_single] * num_n,
            token_ids=[[]] * num_n,
            accumulated_text=[empty_text_val_single] * num_n,
            time_spent_computing=[0.0] * num_n,
            tokens_per_second=[0.0] * num_n,
            num_generated_tokens=[0] * num_n,
        )

    async def _generate_stream(
        self,
        requests: list[vSurgeRequest],
        bytecode_decode: bool | _Empty = NOT_GIVEN,
    ) -> tp.AsyncGenerator[list[ReturnSample], None]:
        """Generates text completions for a batch of requests with streaming.

        Internal method to handle streaming logic. Each yield provides the latest
        state for all input requests. If a stream for a particular request finishes,
        its slot in the yielded list will contain its *last actual `ReturnSample`*
        repeatedly. If it never produced data or errored early, a default
        `ReturnSample` (structured according to the request's 'n' value) is used.

        Args:
            requests (list[vSurgeRequest]): A list of `vSurgeRequest` objects.
            bytecode_decode (tp.Union[bool, _Empty]): If provided (not `NOT_GIVEN`),
                overrides the instance's default `bytecode_decode` setting for this call.
                If `NOT_GIVEN`, the instance's default is used.

        Yields:
            tp.AsyncGenerator[list[ReturnSample], None]: An asynchronous generator.
            Each yield is a list of `ReturnSample` objects. Each `ReturnSample` in
            the list corresponds to an input request. Its fields (e.g., `text`,
            `token_ids`) are lists themselves, where each element corresponds to one
            of the `n` generations for that specific request. `ReturnSample.text`
            contains the current chunk of text/tokens for each of the 'n' generations.
        """
        bytecode_decode = self.bytecode_decode if isinstance(bytecode_decode, _Empty) else bytecode_decode
        num_original_requests = len(requests)
        if num_original_requests == 0:
            return

        stream_iterators = [self.complete(req, bytecode_decode=bytecode_decode).__aiter__() for req in requests]

        latest_data_from_streams: list[ReturnSample] = [
            self._create_default_return_sample_for_request(req, bytecode_decode) for req in requests
        ]
        active_stream_indices = list(range(num_original_requests))
        has_yielded_at_least_once = False

        async def fetch_next_for_sync(iterator, idx_tag):
            try:
                item_list = await iterator.__anext__()
                return idx_tag, item_list[0] if item_list else None, None
            except StopAsyncIteration:
                return idx_tag, None, StopAsyncIteration()
            except Exception as exc:
                return idx_tag, None, exc

        while active_stream_indices:
            tasks_for_this_sync_step = [
                asyncio.create_task(fetch_next_for_sync(stream_iterators[original_idx], original_idx))
                for original_idx in active_stream_indices
            ]

            if not tasks_for_this_sync_step:
                break

            done_tasks, _ = await asyncio.wait(
                tasks_for_this_sync_step,
                return_when=asyncio.ALL_COMPLETED,
            )

            newly_finished_indices_this_poll = []
            critical_error_occurred: Exception | None = None
            any_new_data_arrived_in_this_step = False

            for task in done_tasks:
                original_idx, fetched_sample, task_error_status = task.result()

                if task_error_status:
                    if original_idx not in newly_finished_indices_this_poll:
                        newly_finished_indices_this_poll.append(original_idx)
                    if not isinstance(task_error_status, StopAsyncIteration):
                        critical_error_occurred = task_error_status
                        logger.error(
                            f"Error in stream {original_idx} for prompt "
                            f"'{requests[original_idx].prompt[:30]}...': {task_error_status}"
                        )
                else:
                    if fetched_sample is not None:
                        latest_data_from_streams[original_idx] = fetched_sample
                        any_new_data_arrived_in_this_step = True

            if critical_error_occurred:
                logger.error(f"Critical error occurred during stream processing: {critical_error_occurred}")
                active_stream_indices.clear()
                yield list(latest_data_from_streams)
                has_yielded_at_least_once = True
                raise critical_error_occurred

            if newly_finished_indices_this_poll:
                for finished_idx in newly_finished_indices_this_poll:
                    if finished_idx in active_stream_indices:
                        active_stream_indices.remove(finished_idx)

            should_yield_this_step = (
                not has_yielded_at_least_once
                or any_new_data_arrived_in_this_step
                or bool(newly_finished_indices_this_poll)
                or not active_stream_indices
            )

            if should_yield_this_step:
                yield list(latest_data_from_streams)
                has_yielded_at_least_once = True

            if not active_stream_indices:
                break

        if not has_yielded_at_least_once and num_original_requests > 0:
            yield list(latest_data_from_streams)

    async def generate(
        self,
        prompts: str | tp.Sequence[str],
        sampling_params: SamplingParams | tp.Sequence[SamplingParams] | None = None,
        stream: bool = False,
        bytecode_decode: bool | _Empty = NOT_GIVEN,
    ) -> tp.Generator[ReturnSample] | list[ReturnSample] | tp.AsyncGenerator[list[ReturnSample], None]:
        """Generates text completions for given prompts.

        This is the main public method for text generation. It handles single or
        multiple prompts and can operate in batch or streaming mode.

        Args:
            prompts (tp.Union[str, tp.Sequence[str]]): A single prompt string or a
                sequence of prompt strings.
            sampling_params (tp.Optional[tp.Union[SamplingParams, tp.Sequence[SamplingParams]]]):
                Sampling parameters.
                - If `prompts` is a single string, this can be a single `SamplingParams`
                  object or None (to use default `SamplingParams`).
                - If `prompts` is a sequence, this can be:
                    - A single `SamplingParams` object (applied to all prompts).
                    - A sequence of `SamplingParams` objects (one per prompt).
                    - None (default `SamplingParams` used for all prompts).
                Must match the structure of `prompts` if provided as a sequence.
            stream (bool): If True, returns an async generator that yields results
                incrementally. If False, returns a list with all results after
                completion. Defaults to False.
            bytecode_decode (tp.Union[bool, _Empty]): Overrides the instance's default
                `bytecode_decode` setting for this specific generation call. If `NOT_GIVEN`
                (the default), the instance's default `self.bytecode_decode` is used.
                If True, enables special handling for bytecode tokenizers.

        Returns:
            tp.Union[list[ReturnSample], tp.AsyncGenerator[list[ReturnSample], None]]:
            - If `stream` is False (batch mode): `list[ReturnSample]`.
              Each `ReturnSample` corresponds to an input prompt. Its `accumulated_text`
              field contains the final generated text(s). The structure of
              `ReturnSample.accumulated_text` (and `.text`, `.token_ids`) is a list
              representing the `n` generations (e.g., `list[str]` or `list[list[int]]`).
            - If `stream` is True (streaming mode): `tp.AsyncGenerator[list[ReturnSample], None]`.
              Each yield is `list[ReturnSample]`, one entry per input prompt.
              A `ReturnSample` contains the latest generated chunk (in its `.text` field) and
              cumulative data. If a specific prompt's stream has finished, its slot in the
              yielded list will contain the *last actual `ReturnSample` it produced* repeatedly
              until all streams are finished. If a stream never produces data, it will yield
              a default `ReturnSample`.

        Raises:
            ValueError: If `prompts` and `sampling_params` have mismatched lengths
                or unsupported types.
            RuntimeError: If issues occur during the generation process (e.g., queue full).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            if sampling_params is not None and not isinstance(sampling_params, SamplingParams):
                raise ValueError("If prompts is str, sampling_params must be SamplingParams or None.")
            sampling_params = [sampling_params if sampling_params else SamplingParams()]
        elif isinstance(prompts, tp.Sequence):
            if not prompts:
                if stream:

                    async def empty_gen_for_empty_prompts():
                        if False:
                            yield []

                    return empty_gen_for_empty_prompts()
                return []
            if sampling_params is None:
                sampling_params = [SamplingParams()] * len(prompts)
            elif isinstance(sampling_params, SamplingParams):
                sampling_params = [sampling_params] * len(prompts)
            elif isinstance(sampling_params, tp.Sequence):
                if len(prompts) != len(sampling_params):
                    raise ValueError("Lengths of prompts and sampling_params lists must match.")
            else:
                raise ValueError("sampling_params must be SamplingParams, list of SamplingParams, or None.")
        else:
            raise ValueError("prompts must be a string or a sequence of strings.")

        if not prompts:
            if stream:

                async def empty_gen_final():
                    if False:
                        yield []

                return empty_gen_final()
            return []

        requests_list = [
            vSurgeRequest.from_sampling_params(p, sp) for p, sp in zip(prompts, sampling_params, strict=False)
        ]

        effective_bytecode_decode = self.bytecode_decode if isinstance(bytecode_decode, _Empty) else bytecode_decode

        if stream:
            return self._generate_stream(requests_list, bytecode_decode=effective_bytecode_decode)
        else:
            return await self._generate_batch(requests_list, bytecode_decode=effective_bytecode_decode)

    def should_buffer_response(self, response: list[ReturnSample]) -> bool:
        """Determines if a driver response needs buffering for server-side detokenization.

        Buffering is typically needed if the response contains special tokens
        (e.g., byte fallbacks) indicating incomplete detokenization.

        Args:
            response (list[ReturnSample]): A list of `ReturnSample` objects from the
                driver for a single generation stream's current step. `ReturnSample.text`
                is expected to be a list of strings/byte-strings from the driver.

        Returns:
            bool: True if buffering is needed, False otherwise.
        """
        for item in response:
            if (
                item.text
                and isinstance(item.text, list)
                and item.text
                and isinstance(item.text[-1], str)
                and item.text[-1].startswith("<0x")
                and item.text[-1].endswith(">")
            ):
                return True
        return False

    def process_client_side_tokenization_response(
        self,
        response: list[ReturnSample],
        generation_idx_tag: int,
    ) -> list[ReturnSample]:
        """Processes driver responses for client-side tokenization.

        This method primarily tags the `ReturnSample` objects with the correct
        `generation_idx`. For client-side tokenization, the `ReturnSample.text`
        field from the driver is expected to contain the new token IDs for the current step.
        These are moved to the `token_ids` field of the output `ReturnSample`.

        Args:
            response (list[ReturnSample]): list of `ReturnSample` objects from the driver.
                `ReturnSample.text` is expected to be `list[int]` (new token IDs).
            generation_idx_tag (int): The index of this generation stream (0 to n-1).

        Returns:
            list[ReturnSample]: A list of `ReturnSample` objects, updated with
            `generation_idx_tag` and with new token IDs in the `token_ids` field.
        """
        samples = []
        for sample_from_driver in response:
            new_sample = dataclasses.replace(
                sample_from_driver,
                generation_idx=generation_idx_tag,
                token_ids=list(sample_from_driver.text) if isinstance(sample_from_driver.text, list) else [],
            )
            samples.append(new_sample)
        return samples

    def process_server_side_tokenization_response(
        self,
        current_driver_response: list[ReturnSample],
        buffer_for_this_gen: list[list[ReturnSample]],
        generation_idx_tag: int,
        state: ProcessState = None,
    ) -> list[ReturnSample]:
        """Enhanced version with smart bytecode decoding."""

        items_to_process_tuples = (
            list(zip(*buffer_for_this_gen, current_driver_response, strict=False))
            if buffer_for_this_gen
            else [(r,) for r in current_driver_response]
        )
        processed_samples_for_yield = []

        for single_sequence_all_steps_tuple in items_to_process_tuples:
            text_segments_for_detok_this_chunk = []
            latest_driver_sample_this_step = single_sequence_all_steps_tuple[-1]
            tps = latest_driver_sample_this_step.tokens_per_second
            num_gen_tokens_total_for_seq = latest_driver_sample_this_step.num_generated_tokens
            time_spent = latest_driver_sample_this_step.time_spent_computing
            driver_cumulative_text = latest_driver_sample_this_step.accumulated_text
            token_ids_this_step = list(latest_driver_sample_this_step.token_ids or [])
            if self.bytecode_decode and state is not None:
                buffered_tokens = getattr(state, "buffered_tokens", [])
                previous_decoded_text = getattr(state, "decoded_text_upto_previous_step", "")

                if token_ids_this_step:
                    current_text_chunk, new_buffered_tokens, had_malformed = self.smart_decoder.decode_with_recovery(
                        token_ids_this_step, previous_decoded_text, buffered_tokens
                    )
                    state.buffered_tokens = new_buffered_tokens
                    if had_malformed and new_buffered_tokens:
                        text_for_this_yield_step_chunk = current_text_chunk
                    else:
                        try:
                            full_decoded = self.processor.decode(state.all_token_ids, skip_special_tokens=True)
                            if not self.smart_decoder.contains_malformed_chars(full_decoded):
                                driver_cumulative_text = full_decoded
                        except Exception:
                            pass
                        text_for_this_yield_step_chunk = current_text_chunk
                else:
                    text_for_this_yield_step_chunk = ""
            else:
                for raw_step_sample_from_buffer_or_current in single_sequence_all_steps_tuple:
                    if isinstance(raw_step_sample_from_buffer_or_current.text, list):
                        text_segments_for_detok_this_chunk.extend(raw_step_sample_from_buffer_or_current.text)
                    elif isinstance(raw_step_sample_from_buffer_or_current.text, str | bytes):
                        text_segments_for_detok_this_chunk.append(raw_step_sample_from_buffer_or_current.text)

                text_for_this_yield_step_chunk = text_tokens_to_string(text_segments_for_detok_this_chunk)

            processed_samples_for_yield.append(
                ReturnSample(
                    text=text_for_this_yield_step_chunk,
                    token_ids=token_ids_this_step,
                    accumulated_text=driver_cumulative_text,
                    time_spent_computing=time_spent,
                    tokens_per_second=tps,
                    num_generated_tokens=num_gen_tokens_total_for_seq,
                    generation_idx=generation_idx_tag,
                )
            )

        return processed_samples_for_yield

    async def complete(
        self,
        request: vSurgeRequest,
        bytecode_decode: bool | _Empty = NOT_GIVEN,
    ) -> tp.AsyncGenerator[list[ReturnSample], None]:
        """Performs text generation for a single `vSurgeRequest`.

        This asynchronous generator streams results for `request.sampling_params.n` parallel
        generations. Each yield is a list containing a single `ReturnSample` object.
        This `ReturnSample` aggregates the current state of all `n` generations.

        Args:
            request (vSurgeRequest): The text generation request.
            bytecode_decode (tp.Union[bool, _Empty]): If provided (not `NOT_GIVEN`),
                overrides the instance's default `bytecode_decode` setting for this call.
                If `NOT_GIVEN`, the instance's default `self.bytecode_decode` is used.
                If True, enables special handling for bytecode tokenizers where text is
                always decoded from the full accumulated list of token IDs.

        Yields:
            tp.AsyncGenerator[list[ReturnSample], None]: A generator yielding lists,
            each containing one `ReturnSample`. This `ReturnSample`'s fields
            (e.g., `text`, `token_ids`, `accumulated_text`) are lists,
            with each element corresponding to one of the `n` requested generations.
            - `ReturnSample.text`: `list[Union[str, list[int]]]`. Current text/token_ids chunk.
                - If `bytecode_decode` is True OR `request.is_client_side_tokenization` is False:
                  `str` chunk.
                - Else (`bytecode_decode` is False AND `request.is_client_side_tokenization` is True):
                  `list[int]` of new token IDs for the current step.
            - `ReturnSample.token_ids`: `list[list[int]]`. All token IDs generated so far for each 'n'.
            - `ReturnSample.accumulated_text`: `list[Union[str, list[str]]]`. Full text or
              list of per-token decoded strings for each 'n'.
                - If `bytecode_decode` is True OR `request.is_client_side_tokenization` is False:
                  `str` of full decoded text.
                - Else (`bytecode_decode` is False AND `request.is_client_side_tokenization` is True):
                  `list[str]` (each string being a per-token decode of all tokens so far).
        """
        if request.sampling_params.n == 0:
            return

        bytecode_decode = self.bytecode_decode if isinstance(bytecode_decode, _Empty) else bytecode_decode

        logger.debug(
            f"complete called for request (prompt: '...{request.prompt[-50:]}') with n={request.sampling_params.n}, "
            f"bytecode_decode={bytecode_decode}, client_side_tok={request.is_client_side_tokenization}"
        )

        gen_states: list[ProcessState] = []
        for i in range(request.sampling_params.n):
            return_channel = AsyncMultifuture()
            start_time = request.metadata.start_time if request.metadata else time.time()
            prefill_enqueue_time = time.perf_counter()
            active_request = ActiveRequest(
                return_channel=return_channel,
                sampling_params=request.sampling_params,
                prefill_content=request.prompt,
                is_client_side_tokenization=request.is_client_side_tokenization,
                metadata=ActiveRequestMetadata(start_time=start_time, prefill_enqueue_time=prefill_enqueue_time),
            )
            try:
                self.driver.submit_request(active_request)
            except queue.Full as e:
                for state_idx in range(i):
                    if hasattr(gen_states[state_idx].active_request.return_channel, "cancel"):
                        gen_states[state_idx].active_request.return_channel.cancel()
                raise RuntimeError(f"Prefill queue full for generation {i + 1}/{request.sampling_params.n}") from e

            initial_current_step_text: str | list
            initial_full_accumulated_text: str | list
            if request.is_client_side_tokenization and not bytecode_decode:
                initial_current_step_text = []
                initial_full_accumulated_text = []
            else:
                initial_current_step_text = ""
                initial_full_accumulated_text = ""

            gen_states.append(
                ProcessState(
                    id=i,
                    active_request=active_request,
                    channel_iter=active_request.return_channel.__aiter__(),
                    current_step_text=initial_current_step_text,
                    full_accumulated_text=initial_full_accumulated_text,
                    decoded_text_upto_previous_step="" if bytecode_decode else None,
                )
            )

        num_gens_fully_finished = 0
        active_gen_indices_to_poll = list(range(request.sampling_params.n))

        async def _fetch_next_from_channel(iterator, original_idx_tag):
            try:
                return original_idx_tag, await iterator.__anext__(), None
            except StopAsyncIteration:
                return original_idx_tag, None, StopAsyncIteration()
            except Exception as exc:
                return original_idx_tag, None, exc

        while num_gens_fully_finished < request.sampling_params.n:
            tasks_to_poll = [
                asyncio.create_task(_fetch_next_from_channel(gen_states[idx].channel_iter, idx))
                for idx in active_gen_indices_to_poll
            ]
            if not tasks_to_poll:
                break

            done_tasks, pending_tasks = await asyncio.wait(tasks_to_poll, return_when=asyncio.FIRST_COMPLETED)
            newly_set_current_step_text_flags = [False] * request.sampling_params.n

            for task in done_tasks:
                original_idx, item_from_driver, error_status = task.result()
                state = gen_states[original_idx]

                if isinstance(error_status, StopAsyncIteration):
                    state.finished_streaming = True
                    num_gens_fully_finished += 1
                    if original_idx in active_gen_indices_to_poll:
                        active_gen_indices_to_poll.remove(original_idx)
                    if not request.is_client_side_tokenization and state.driver_buffer:
                        last_buffered_sample = (
                            state.driver_buffer[-1][0]
                            if state.driver_buffer and state.driver_buffer[-1]
                            else ReturnSample(text=[], token_ids=[])
                        )
                        dummy_driver_resp_for_flush = [
                            ReturnSample(
                                text=[],
                                token_ids=[],
                                accumulated_text=last_buffered_sample.accumulated_text,
                                time_spent_computing=last_buffered_sample.time_spent_computing,
                                tokens_per_second=last_buffered_sample.tokens_per_second,
                                num_generated_tokens=last_buffered_sample.num_generated_tokens,
                            )
                        ]
                        processed_outputs = self.process_server_side_tokenization_response(
                            dummy_driver_resp_for_flush,
                            state.driver_buffer,
                            original_idx,
                            state,
                        )
                        state.driver_buffer = []
                        if processed_outputs:
                            final_chunk_from_buffer = processed_outputs[0]
                            state.time_spent = final_chunk_from_buffer.time_spent_computing
                            state.tps = final_chunk_from_buffer.tokens_per_second
                            state.num_tokens = final_chunk_from_buffer.num_generated_tokens
                            if bytecode_decode and state.all_token_ids:
                                full_decoded_text_now = self.processor.decode(
                                    state.all_token_ids,
                                    skip_special_tokens=True,
                                )
                                prev_decoded_text = state.decoded_text_upto_previous_step or ""
                                current_text_chunk = (
                                    full_decoded_text_now[len(prev_decoded_text) :]
                                    if full_decoded_text_now.startswith(prev_decoded_text)
                                    else full_decoded_text_now
                                )
                                state.current_step_text = current_text_chunk
                                state.full_accumulated_text = full_decoded_text_now
                            else:
                                state.current_step_text = final_chunk_from_buffer.text
                                state.full_accumulated_text = final_chunk_from_buffer.accumulated_text
                            if state.current_step_text:
                                newly_set_current_step_text_flags[original_idx] = True
                    continue
                if error_status is not None:
                    logger.error(f"Error in stream {original_idx} for request: {error_status}")
                    for p_task in pending_tasks:
                        p_task.cancel()
                    if pending_tasks:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                    for i_gen in range(request.sampling_params.n):
                        if i_gen != original_idx and not gen_states[i_gen].finished_streaming:
                            if hasattr(
                                gen_states[i_gen].active_request.return_channel,
                                "set_exception",
                            ):
                                gen_states[i_gen].active_request.return_channel.set_exception(error_status)
                            gen_states[i_gen].finished_streaming = True
                    raise error_status

                driver_step_output_list = tp.cast(list[ReturnSample], item_from_driver)

                if request.is_client_side_tokenization:
                    processed_outputs = self.process_client_side_tokenization_response(
                        driver_step_output_list, original_idx
                    )
                    if processed_outputs:
                        chunk = processed_outputs[0]
                        new_token_ids_this_step = list(chunk.token_ids)
                        state.time_spent = chunk.time_spent_computing
                        state.tps = chunk.tokens_per_second
                        state.num_tokens = chunk.num_generated_tokens
                        if new_token_ids_this_step:
                            state.all_token_ids.extend(new_token_ids_this_step)
                            newly_set_current_step_text_flags[original_idx] = True
                            if bytecode_decode:
                                buffered_tokens = getattr(state, "buffered_tokens", [])
                                prev_decoded_text = state.decoded_text_upto_previous_step or ""
                                current_text_chunk, new_buffered_tokens, had_malformed = (
                                    self.smart_decoder.decode_with_recovery(
                                        new_token_ids_this_step,
                                        prev_decoded_text,
                                        buffered_tokens,
                                    )
                                )
                                state.buffered_tokens = new_buffered_tokens
                                state.current_step_text = current_text_chunk
                                if not had_malformed or not new_buffered_tokens:
                                    try:
                                        full_decoded_text_now = self.processor.decode(
                                            state.all_token_ids,
                                            skip_special_tokens=True,
                                        )
                                        if not self.smart_decoder.contains_malformed_chars(full_decoded_text_now):
                                            state.full_accumulated_text = full_decoded_text_now
                                            state.decoded_text_upto_previous_step = full_decoded_text_now
                                        else:
                                            state.full_accumulated_text = prev_decoded_text + current_text_chunk
                                            state.decoded_text_upto_previous_step = state.full_accumulated_text
                                    except Exception:
                                        state.full_accumulated_text = prev_decoded_text + current_text_chunk
                                        state.decoded_text_upto_previous_step = state.full_accumulated_text
                                else:
                                    state.full_accumulated_text = prev_decoded_text + current_text_chunk
                                    state.decoded_text_upto_previous_step = state.full_accumulated_text
                            else:
                                state.current_step_text = new_token_ids_this_step
                                state.full_accumulated_text = [
                                    self.processor.decode([tok_id], skip_special_tokens=True)
                                    for tok_id in state.all_token_ids
                                ]
                else:  # Server-side tokenization
                    if self.should_buffer_response(driver_step_output_list):
                        state.driver_buffer.append(driver_step_output_list)
                    else:
                        processed_outputs = self.process_server_side_tokenization_response(
                            driver_step_output_list,
                            state.driver_buffer,
                            original_idx,
                            state,
                        )
                        state.driver_buffer = []
                        if processed_outputs:
                            chunk = processed_outputs[0]
                            new_token_ids_this_step = list(chunk.token_ids)
                            if new_token_ids_this_step:
                                state.all_token_ids.extend(new_token_ids_this_step)
                            state.time_spent = chunk.time_spent_computing
                            state.tps = chunk.tokens_per_second
                            state.num_tokens = chunk.num_generated_tokens
                            if chunk.text or not state.all_token_ids:
                                newly_set_current_step_text_flags[original_idx] = True
                                if bytecode_decode and state.all_token_ids:
                                    full_decoded_text_now = self.processor.decode(
                                        state.all_token_ids,
                                        skip_special_tokens=True,
                                    )
                                    prev_decoded_text = state.decoded_text_upto_previous_step or ""
                                    current_text_chunk = (
                                        full_decoded_text_now[len(prev_decoded_text) :]
                                        if full_decoded_text_now.startswith(prev_decoded_text)
                                        else full_decoded_text_now
                                    )
                                    state.current_step_text = current_text_chunk
                                    state.full_accumulated_text = full_decoded_text_now
                                    state.decoded_text_upto_previous_step = full_decoded_text_now
                                else:
                                    state.current_step_text = chunk.text
                                    state.full_accumulated_text = chunk.accumulated_text
                                    if bytecode_decode:
                                        state.decoded_text_upto_previous_step = chunk.accumulated_text

            for task_to_cancel in pending_tasks:
                task_to_cancel.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

            any_stream_produced_new_text_this_cycle = any(newly_set_current_step_text_flags)
            all_streams_definitively_done = (
                num_gens_fully_finished == request.sampling_params.n and not active_gen_indices_to_poll
            )

            if any_stream_produced_new_text_this_cycle or all_streams_definitively_done:
                texts_for_yield, all_tokens_for_yield, full_texts_for_yield = [], [], []
                times_spent_for_yield, tps_for_yield, num_tokens_for_yield = [], [], []

                for idx in range(request.sampling_params.n):
                    s = gen_states[idx]
                    if newly_set_current_step_text_flags[idx]:
                        texts_for_yield.append(s.current_step_text)
                    elif request.is_client_side_tokenization and not bytecode_decode:
                        texts_for_yield.append([])
                    else:
                        texts_for_yield.append("")
                    all_tokens_for_yield.append(list(s.all_token_ids))
                    full_texts_for_yield.append(s.full_accumulated_text)
                    times_spent_for_yield.append(s.time_spent)
                    tps_for_yield.append(s.tps)
                    num_tokens_for_yield.append(s.num_tokens)

                aggregated_sample = ReturnSample(
                    text=texts_for_yield,
                    token_ids=all_tokens_for_yield,
                    accumulated_text=full_texts_for_yield,
                    time_spent_computing=times_spent_for_yield,
                    tokens_per_second=tps_for_yield,
                    num_generated_tokens=num_tokens_for_yield,
                )
                yield [aggregated_sample]

                for idx_reset in range(request.sampling_params.n):
                    if newly_set_current_step_text_flags[idx_reset]:
                        if request.is_client_side_tokenization and not bytecode_decode:
                            gen_states[idx_reset].current_step_text = []
                        else:
                            gen_states[idx_reset].current_step_text = ""
            if all_streams_definitively_done:
                break

        for state_to_finalize in gen_states:
            if not state_to_finalize.finished_streaming:
                if hasattr(state_to_finalize.active_request.return_channel, "close"):
                    state_to_finalize.active_request.return_channel.close()
                state_to_finalize.finished_streaming = True

    def __repr__(self):
        """Returns a string representation of the vSurge instance."""
        is_live = self.driver.live if hasattr(self.driver, "live") else "N/A"
        return f"vSurge(name={self.vsurge_name}, live={is_live})"

    __str__ = __repr__
