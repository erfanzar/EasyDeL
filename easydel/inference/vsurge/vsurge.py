# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""
This module defines the vSurge system, a high-throughput inference engine
for EasyDeL models, designed for high-throughput and low-latency text generation.

vSurge orchestrates text generation requests, managing the underlying driver
(vDriver for standard attention or oDriver for paged attention) and processing
responses. It provides a high-level interface for submitting text generation
requests and handles the complexities of request queuing, tokenization,
detokenization, and asynchronous processing.

Key features of vSurge include:

- **High-throughput:** vSurge is designed to handle a large number of concurrent
  text generation requests.
- **Low-latency:** vSurge minimizes latency by using asynchronous processing
  and optimized inference engines.
- **Memory-efficient:** vSurge supports paged attention via the oDriver, which
  allows for efficient memory management when generating long sequences.
- **Flexibility:** vSurge supports both standard attention (vDriver) and paged
  attention (oDriver), allowing users to choose the best option for their
  specific needs.
- **Easy integration:** vSurge integrates seamlessly with other modules in the
  EasyDeL library.

The following diagram illustrates the architecture of the vSurge system:

```mermaid
graph LR
    A[vSurge] --> B(vDriver or oDriver);
    B --> C(vEngine or oEngine);
    C --> D(EasyDeLBaseModule);
    C --> E(Processor);
    B --> F(ActiveRequest Queue);
    F --> B;
    A --> G(ReturnSample Queue);
    G --> A;
```

In this diagram:

- `vSurge` is the main class that orchestrates the text generation process.
- `vDriver` and `oDriver` are the underlying inference drivers that handle the
  actual text generation. `vDriver` uses standard attention, while `oDriver`
  uses paged attention for memory-efficient inference.
- `vEngine` and `oEngine` are the inference engines that perform the forward
  pass of the model.
- `EasyDeLBaseModule` is the EasyDeL model instance.
- `Processor` is the processor/tokenizer instance.
- `ActiveRequest Queue` is the queue that holds the incoming text generation
  requests.
- `ReturnSample Queue` is the queue that holds the generated text samples.
"""

from __future__ import annotations

import asyncio
import dataclasses
import queue
import time
import typing as tp

import jax

from easydel.inference.utilities import SamplingParams
from easydel.utils.helpers import get_logger

from .core import vDriver, vEngine
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
else:
    ProcessingClassType = tp.Any
    EasyDeLBaseModule = tp.Any


logger = get_logger("vSurge")


class vSurgeMetadata:
    """
    Tracks timing information for requests processed by the vSurge.

    This class is used to store metadata related to a request's processing,
    such as the start time. It helps in measuring the latency and performance
    of the vSurge system.

    Attributes:
        start_time (float): The Unix timestamp (seconds) when the request processing started.
    """

    def __init__(self):
        """
        Initializes the metadata, capturing the current time as the start time.
        """
        self.start_time = time.time()


@dataclasses.dataclass
class vSurgeRequest:
    """
    Represents a request specifically for text completion within the vSurge system.

    This dataclass encapsulates all parameters necessary for a text generation
    request, providing a structured way to configure the generation process.

    Attributes:
        prompt (str): The input prompt for text completion. This is the text that the model will use as a starting point
            for generating new text.
        max_tokens (int): The maximum number of tokens to generate. This limits the length of the generated text.
        top_p (float): The nucleus sampling probability. Only tokens comprising the top_p cumulative probability are
            considered. Defaults to 1.0.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 0
            (no top-k filtering).
        min_p (float): The minimum probability for a token to be considered. Defaults to 0.0.
        n (int): The number of independent samples to generate for each prompt. Defaults to 1.
        stop (tp.Optional[tp.Union[str, tp.List[str]]]): A string or list of strings that, if generated, will cause the
            generation to stop. Defaults to None.
        temperature (float): The sampling temperature. Higher values make the output more random, lower values make it
            more deterministic. Defaults to 0.7.
        presence_penalty (float): Penalty applied to tokens based on their presence in the generated text so far.
            Discourages repetition. Defaults to 0.0.
        frequency_penalty (float): Penalty applied to tokens based on their frequency in the generated text so far.
            Discourages frequent tokens. Defaults to 0.0.
        repetition_penalty (float): Penalty applied to repeated tokens. A value > 1.0 discourages repetition.
            Defaults to 1.0.
        metadata (vSurgeMetadata | None): Metadata associated with the request, such as start time.
            Automatically initialized if None.
        is_client_side_tokenization (bool): If True, indicates that the prompt is already tokenized and the
            client expects token IDs as output. Defaults to False.
    """

    prompt: str
    max_tokens: int

    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    n: int = 1
    stop: str | list[str] | None = None

    temperature: float = 0.7
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    metadata: vSurgeMetadata | None = None
    is_client_side_tokenization: bool = False

    @classmethod
    def from_sampling_params(cls, prompt: str, sampling_params: SamplingParams):
        """
        Creates a vSurgeRequest instance from a prompt and SamplingParams.

        Args:
            prompt (str): The input prompt string.
            sampling_params (SamplingParams): An object containing sampling parameters.

        Returns:
            vSurgeRequest: A new vSurgeRequest instance initialized with the
                provided prompt and sampling parameters.
        """
        return vSurgeRequest(
            prompt=prompt,
            max_tokens=sampling_params.max_tokens,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            min_p=sampling_params.min_p,
            n=sampling_params.n,
            stop=sampling_params.stop,
            temperature=sampling_params.temperature,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            repetition_penalty=sampling_params.repetition_penalty,
        )

    def __post_init__(self):
        """
        Ensures metadata is initialized and validates the prompt type.
        `is_client_side_tokenization` is also explicitly set to False by default here.
        """
        if self.metadata is None:
            self.metadata = vSurgeMetadata()
        self.is_client_side_tokenization = False
        assert isinstance(self.prompt, str), "prompt should be a single string"


class vSurge:
    """
    Orchestrates high-throughput text generation by interacting with a vDriver or oDriver.

    The vSurge class acts as the main interface for submitting text generation
    requests to the underlying inference engine. It manages the driver
    (either a vDriver for standard attention or an oDriver for paged attention),
    handles request queuing, and processes responses, including tokenization
    and detokenization if needed.

    Attributes:
        _driver (vDriver): The underlying inference driver instance
            (either vDriver for standard attention or oDriver for paged attention).
        _vsurge_name (str): The name of this vSurge instance, defaulting to the
            driver's name.
    """

    def __init__(
        self,
        driver: vDriver,
        vsurge_name: str | None = None,
    ):
        """
        Initializes the vSurge instance.

        Args:
            driver (vDriver): The underlying driver instance
                (vDriver or oDriver) that will handle the actual inference.
            vsurge_name (str | None): An optional name for this vSurge instance.
                If None, it defaults to the name of the provided driver.
        """
        self._driver = driver
        self._vsurge_name = vsurge_name or driver.driver_name

    def compile(self):
        """
        Compiles the underlying driver.

        This typically involves JIT compilation of the model's forward pass for
        optimized execution.
        """
        self.driver.compile()

    @property
    def vsurge_name(self) -> str:
        """
        Returns the name of the vSurge instance.

        Returns:
            str: The name of this vSurge instance.
        """
        return self._vsurge_name

    @property
    def driver(self) -> vDriver:
        """
        Provides access to the underlying driver instance.

        Returns:
            vDriver: The vDriver or oDriver instance used by
                this vSurge.
        """
        return self._driver

    @property
    def processor(self) -> ProcessingClassType:
        """
        Returns the processor/tokenizer associated with the underlying driver.

        The processor is used for tokenizing prompts and detokenizing generated
        token IDs.

        Returns:
            ProcessingClassType: The processor (e.g., a Hugging Face tokenizer)
                instance.
        """
        return self.driver.processor

    def start(self):
        """
        Starts the underlying driver.

        This initializes the driver's processing loops and makes it ready to
        accept inference requests.
        """
        return self.driver.start()

    def stop(self):
        """
        Stops the underlying driver.

        This gracefully shuts down the driver's processing loops.
        """
        return self.driver.stop()

    def pause(self):
        """
        Pauses the underlying driver.

        This temporarily halts the processing of new requests by the driver.
        """
        return self.driver.pause()

    def resume(self):
        """
        Resumes the underlying driver after it has been paused.

        This allows the driver to continue processing requests.
        """
        return self.driver.resume()

    def replace_graphstate(self, state):
        """
        Replaces the graph state of the underlying driver.

        This is an advanced feature, typically used for dynamic model updates or
        state management in complex scenarios.

        Args:
            state: The new graph state to be applied to the driver.
        """
        return self.driver.replace_graphstate(state=state)

    @classmethod
    def from_model(
        cls,
        model: EasyDeLBaseModule,
        processor: ProcessingClassType,
        max_concurrent_decodes: int | None = None,
        max_concurrent_prefill: int | None = None,
        prefill_lengths: int | None = None,
        max_prefill_length: int | None = None,
        max_length: int | None = None,
        num_pages: int | None = None,
        tokens_per_page: int | None = None,
        interleaved_mode: bool = False,
        detokenizing_blocks: int = 8,
        slot_clear_steps: int = 512,
        vsurge_name: str | None = None,
        verbose: bool = True,
        seed: int = 894,
    ) -> vSurge:
        """
        Instantiates a `vSurge` object from a given model and processor, configuring
        decoding and prefill concurrency, memory management, and generation parameters.

        This method constructs a `vSurge` engine and driver stack, handling default
        fallbacks and validations for parameters not explicitly provided.

        Args:
            model (EasyDeLBaseModule):
                The EasyDeL model instance used for inference.

            processor (ProcessingClassType):
                The tokenizer or processor compatible with the model.

            max_concurrent_decodes (int, optional):
                Maximum number of decode calls allowed in parallel. Defaults to the
                number of available JAX devices.

            max_concurrent_prefill (int, optional):
                Maximum number of concurrent prefill steps. Defaults to 1.

            prefill_lengths (int, optional):
                Custom prefill lengths for each decode group. If not provided, they are
                computed automatically.

            max_prefill_length (int, optional):
                The maximum number of tokens allowed during the prefill phase. Defaults
                to half of `max_length`.

            max_length (int, optional):
                The maximum sequence length for decoding. Defaults to the model's
                `granted_mask_max_position_embedding`.

            num_pages (int, optional):
                Number of paging groups for memory partitioning and prefill scheduling.

            tokens_per_page (int, optional):
                Number of tokens allocated per page in the decoding workspace.

            interleaved_mode (bool, optional):
                If True, enables interleaved decoding and prefill scheduling.
                Defaults to False.

            detokenizing_blocks (int, optional):
                Number of detokenization blocks used during output post-processing.
                Defaults to 8.

            slot_clear_steps (int, optional):
                Number of steps after which stale memory slots are cleared.
                Defaults to 512.

            vsurge_name (str, optional):
                Optional name identifier for the created `vSurge` instance.

            verbose (bool, optional):
                Enables logging and verbose driver output. Defaults to True.

            seed (int, optional):
                Random seed for consistent decoding behavior. Defaults to 894.

        Returns:
            vSurge:
                A fully configured `vSurge` instance ready for inference.

        Raises:
            ValueError:
                If `prefill_lengths` is provided but its maximum value does not match
                `max_prefill_length`.
        """
        max_length = max_length or model.config.granted_mask_max_position_embedding

        max_prefill_length = max_prefill_length or max_length // 2

        max_concurrent_prefill = max_concurrent_prefill or 1
        max_concurrent_decodes = max_concurrent_decodes or jax.device_count()

        if prefill_lengths is not None:
            if max(prefill_lengths) != max_prefill_length:
                raise ValueError("The maximum value in `prefill_lengths` must match `max_prefill_length`.")
        else:
            prefill_lengths = calculate_pefill_lengths(
                max_prefill_length=max_prefill_length,
                num_pages=128,
            )
            logger.info(f"Computed prefill lengths are {prefill_lengths}")
        return vSurge(
            driver=vDriver(
                engine=vEngine(
                    model=model,
                    processor=processor,
                    max_concurrent_prefill=max_concurrent_prefill,
                    max_concurrent_decodes=max_concurrent_decodes,
                    prefill_lengths=prefill_lengths,
                    max_prefill_length=max_prefill_length,
                    num_pages=num_pages,
                    tokens_per_page=tokens_per_page,
                    max_length=max_length,
                    seed=seed,
                ),
                interleaved_mode=interleaved_mode,
                detokenizing_blocks=detokenizing_blocks,
                slot_clear_steps=slot_clear_steps,
                verbose=verbose,
            ),
            vsurge_name=vsurge_name,
        )

    def count_tokens(self, text_or_conversation: str | list) -> int:
        """
        Counts the number of tokens in a given string or conversation list.

        Uses the underlying driver's processor to tokenize the input. If the input
        is a list (assumed to be a conversation in OpenAI chat format), it attempts
        to apply the chat template if available, otherwise concatenates content fields.

        Args:
            text_or_conversation (tp.Union[str, list]): Either a single string or a
                list of message dictionaries (e.g., `[{"role": "user", "content": "Hello"}]`).

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
                    return len(tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized)
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

    def should_buffer_response(self, response: list[ReturnSample]) -> bool:
        """
        Determines if a response from the driver needs buffering for server-side detokenization.

        This is typically true if the response contains special tokens (e.g., byte fallbacks)
        that indicate incomplete detokenization.

        Args:
            response (list[ReturnSample]): A list of `ReturnSample` objects from the
                driver for a single generation stream's current step. Each `ReturnSample`'s
                `text` attribute is a list of strings/byte-strings from the driver.

        Returns:
            bool: True if buffering is needed, False otherwise.
        """
        for item in response:
            if (
                item.text
                and isinstance(item.text, list)
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
        """
        Processes driver responses when tokenization is handled client-side.

        This method primarily tags the `ReturnSample` objects with the correct
        `generation_idx`, which identifies which of the 'n' parallel generations
        this sample belongs to.

        Args:
            response (list[ReturnSample]): List of `ReturnSample` objects from the
                driver for the current step of a single generation stream (one of 'n').
                The `text` field in these samples is expected to be token IDs.
            generation_idx_tag (int): The index of this generation stream (0 to n-1).

        Returns:
            list[ReturnSample]: A list of `ReturnSample` objects, each updated with
                the `generation_idx_tag`.
        """
        samples = []
        for sample_from_driver in response:
            new_sample = dataclasses.replace(
                sample_from_driver,
                generation_idx=generation_idx_tag,
            )
            samples.append(new_sample)
        return samples

    def process_server_side_tokenization_response(
        self,
        current_driver_response: list[ReturnSample],
        buffer_for_this_gen: list[list[ReturnSample]],
        generation_idx_tag: int,
    ) -> list[ReturnSample]:
        """
        Processes driver responses when tokenization/detokenization is server-side.

        This method handles detokenization of text segments, including potentially
        combining buffered segments from previous steps with current segments,
        especially to correctly decode byte fallbacks or other special tokens.

        Args:
            current_driver_response (list[ReturnSample]): List of `ReturnSample`
                objects from the driver for the current step of a single generation stream.
            buffer_for_this_gen (list[list[ReturnSample]]): A buffer containing lists
                of `ReturnSample` objects from previous steps of the same generation stream,
                kept when `should_buffer_response` was True.
            generation_idx_tag (int): The index of this generation stream (0 to n-1).

        Returns:
            list[ReturnSample]: A list of processed `ReturnSample` objects, typically
                containing one `ReturnSample` with the detokenized text for the current
                chunk, tagged with `generation_idx_tag`.
        """
        items_to_process_tuples = (
            list(zip(*buffer_for_this_gen, current_driver_response, strict=False))
            if buffer_for_this_gen
            else [(r,) for r in current_driver_response]
        )

        processed_samples_for_yield = []

        for single_sequence_all_steps_tuple in items_to_process_tuples:
            text_segments_for_detok_this_chunk = []

            latest_raw_sample_in_sequence = single_sequence_all_steps_tuple[-1]
            tps = latest_raw_sample_in_sequence.tokens_per_second
            num_gen_tokens_total_for_seq = latest_raw_sample_in_sequence.num_generated_tokens
            time_spent = latest_raw_sample_in_sequence.time_spent_computing

            full_accumulated_text_from_driver = latest_raw_sample_in_sequence.accumulated_text
            if isinstance(full_accumulated_text_from_driver, list) and full_accumulated_text_from_driver:
                full_accumulated_text_from_driver = "".join(
                    s for s in full_accumulated_text_from_driver if isinstance(s, str)
                )
            elif not isinstance(full_accumulated_text_from_driver, str):
                full_accumulated_text_from_driver = ""
            all_token_ids_for_this_gen_sequence_total = latest_raw_sample_in_sequence.token_ids
            if not isinstance(all_token_ids_for_this_gen_sequence_total, list):
                all_token_ids_for_this_gen_sequence_total = []

            for raw_step_sample in single_sequence_all_steps_tuple:
                if isinstance(raw_step_sample.text, list):
                    text_segments_for_detok_this_chunk.extend(raw_step_sample.text)

            text_for_this_yield_step_chunk = text_tokens_to_string(text_segments_for_detok_this_chunk)

            processed_samples_for_yield.append(
                ReturnSample(
                    text=text_for_this_yield_step_chunk,
                    token_ids=all_token_ids_for_this_gen_sequence_total,
                    accumulated_text=full_accumulated_text_from_driver,
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
    ) -> tp.AsyncGenerator[list[ReturnSample], None]:
        """
        Performs text generation for a single `vSurgeRequest`, handling `n` parallel generations.

        This asynchronous generator streams results. Each yield is a list containing
        a single `ReturnSample` object. This `ReturnSample` aggregates the current
        state of all `n` generations requested by the input `vSurgeRequest`.
        For example, `ReturnSample.text` will be a list where each element is the
        latest text chunk (or list of token IDs if client-side tokenization)
        for one of the `n` generations.

        Args:
            request (vSurgeRequest): The text generation request.

        Yields:
            tp.List[ReturnSample]: A list containing one `ReturnSample` object.
                This object's fields (e.g., `text`, `token_ids`) are lists,
                with each element corresponding to one of the `n` requested generations.
                - `text`: `List[Union[str, List[int]]]` - current text/token_ids chunk for each 'n'.
                - `token_ids`: `List[List[int]]` - all token IDs so far for each 'n'.
                - `accumulated_text`: `List[Union[str, List[str]]]` - full text/token strings for each 'n'.

        Raises:
            RuntimeError: If the prefill queue is full or if an error occurs during generation.
        """
        if request.n == 0:
            if False:
                yield []
            return

        gen_states = []
        for i in range(request.n):
            return_channel = AsyncMultifuture()
            active_req = ActiveRequest(
                max_tokens=request.max_tokens,
                prefill_content=request.prompt,
                is_client_side_tokenization=request.is_client_side_tokenization,
                return_channel=return_channel,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                stop=request.stop,
                temperature=request.temperature,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                repetition_penalty=request.repetition_penalty,
                metadata=ActiveRequestMetadata(
                    start_time=request.metadata.start_time if request.metadata else time.time(),
                    prefill_enqueue_time=time.perf_counter(),
                ),
            )
            try:
                self._driver.place_request_on_prefill_queue(active_req)
            except queue.Full as e:
                raise RuntimeError(f"Prefill queue full for generation {i + 1}/{request.n}") from e

            gen_states.append(
                {
                    "id": i,
                    "active_request": active_req,
                    "channel_iter": active_req.return_channel.__aiter__(),
                    "driver_buffer": [],
                    "finished_streaming": False,
                    "current_step_text": [] if request.is_client_side_tokenization else "",
                    "all_token_ids": [],
                    "full_accumulated_text": [] if request.is_client_side_tokenization else "",
                    "time_spent": 0.0,
                    "tps": 0.0,
                    "num_tokens": 0,
                }
            )

        num_gens_fully_finished = 0
        active_gen_indices_to_poll = list(range(request.n))

        while num_gens_fully_finished < request.n:
            tasks_to_poll = []
            indices_polled_this_round = []
            for gen_idx_val in active_gen_indices_to_poll:
                state = gen_states[gen_idx_val]

                async def fetch_next(iterator, original_idx_tag):
                    try:
                        return original_idx_tag, await iterator.__anext__(), None
                    except StopAsyncIteration:
                        return original_idx_tag, None, StopAsyncIteration
                    except Exception as exc:
                        return original_idx_tag, None, exc

                tasks_to_poll.append(asyncio.create_task(fetch_next(state["channel_iter"], gen_idx_val)))
                indices_polled_this_round.append(gen_idx_val)

            if not tasks_to_poll:
                break

            done_tasks, pending_tasks = await asyncio.wait(
                tasks_to_poll,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            newly_set_current_step_text_flags = [False] * request.n
            for task in done_tasks:
                original_idx, item_from_driver, error_status = task.result()
                state = gen_states[original_idx]

                if error_status is StopAsyncIteration:
                    state["finished_streaming"] = True
                    num_gens_fully_finished += 1
                    if original_idx in active_gen_indices_to_poll:
                        active_gen_indices_to_poll.remove(original_idx)

                    if not request.is_client_side_tokenization and state["driver_buffer"]:
                        dummy_driver_resp = [
                            ReturnSample(
                                text=[],
                                token_ids=[],
                                accumulated_text=state["driver_buffer"][-1][0].accumulated_text,
                                time_spent_computing=state["driver_buffer"][-1][0].time_spent_computing,
                                tokens_per_second=state["driver_buffer"][-1][0].tokens_per_second,
                                num_generated_tokens=state["driver_buffer"][-1][0].num_generated_tokens,
                            )
                        ]
                        processed_outputs = self.process_server_side_tokenization_response(
                            dummy_driver_resp, state["driver_buffer"], original_idx
                        )
                        if processed_outputs:
                            final_chunk = processed_outputs[0]
                            state["current_step_text"] = final_chunk.text
                            newly_set_current_step_text_flags[original_idx] = True
                            state["all_token_ids"] = final_chunk.token_ids
                            state["full_accumulated_text"] = final_chunk.accumulated_text
                            state["time_spent"], state["tps"], state["num_tokens"] = (
                                final_chunk.time_spent_computing,
                                final_chunk.tokens_per_second,
                                final_chunk.num_generated_tokens,
                            )
                        state["driver_buffer"] = []
                    continue

                if error_status is not None:
                    raise error_status

                driver_step_output_list = tp.cast(list[ReturnSample], item_from_driver)

                if request.is_client_side_tokenization:
                    processed_outputs = self.process_client_side_tokenization_response(
                        driver_step_output_list, original_idx
                    )
                    if processed_outputs:
                        chunk = processed_outputs[0]
                        state["current_step_text"] = chunk.text
                        newly_set_current_step_text_flags[original_idx] = True
                        state["all_token_ids"].extend(chunk.token_ids)
                        state["full_accumulated_text"] = [
                            self.processor.decode([tok_id]) for tok_id in state["all_token_ids"]
                        ]
                        state["time_spent"], state["tps"], state["num_tokens"] = (
                            chunk.time_spent_computing,
                            chunk.tokens_per_second,
                            chunk.num_generated_tokens,
                        )
                else:
                    if self.should_buffer_response(driver_step_output_list):
                        state["driver_buffer"].append(driver_step_output_list)

                    else:
                        processed_outputs = self.process_server_side_tokenization_response(
                            driver_step_output_list, state["driver_buffer"], original_idx
                        )
                        state["driver_buffer"] = []
                        if processed_outputs:
                            chunk = processed_outputs[0]
                            state["current_step_text"] = chunk.text
                            newly_set_current_step_text_flags[original_idx] = True
                            state["all_token_ids"] = chunk.token_ids
                            state["full_accumulated_text"] = chunk.accumulated_text
                            state["time_spent"], state["tps"], state["num_tokens"] = (
                                chunk.time_spent_computing,
                                chunk.tokens_per_second,
                                chunk.num_generated_tokens,
                            )

            any_stream_produced_new_text_this_cycle = any(newly_set_current_step_text_flags)
            all_streams_definitively_done = num_gens_fully_finished == request.n and not active_gen_indices_to_poll

            if any_stream_produced_new_text_this_cycle or all_streams_definitively_done:
                texts_for_yield = []
                for idx in range(request.n):
                    if newly_set_current_step_text_flags[idx]:
                        texts_for_yield.append(gen_states[idx]["current_step_text"])
                    elif request.is_client_side_tokenization:
                        texts_for_yield.append([])
                    else:
                        texts_for_yield.append("")
                all_tokens_for_yield = [list(gs["all_token_ids"]) for gs in gen_states]
                full_texts_for_yield = [gs["full_accumulated_text"] for gs in gen_states]
                times_spent_for_yield = [gs["time_spent"] for gs in gen_states]
                tps_for_yield = [gs["tps"] for gs in gen_states]
                num_tokens_for_yield = [gs["num_tokens"] for gs in gen_states]

                if request.n == 1:
                    aggregated_sample = ReturnSample(
                        text=texts_for_yield,
                        token_ids=all_tokens_for_yield,
                        accumulated_text=full_texts_for_yield,
                        time_spent_computing=times_spent_for_yield,
                        tokens_per_second=tps_for_yield,
                        num_generated_tokens=num_tokens_for_yield,
                    )
                else:
                    aggregated_sample = ReturnSample(
                        text=texts_for_yield,
                        token_ids=all_tokens_for_yield,
                        accumulated_text=full_texts_for_yield,
                        time_spent_computing=times_spent_for_yield,
                        tokens_per_second=tps_for_yield,
                        num_generated_tokens=num_tokens_for_yield,
                    )
                yield [aggregated_sample]

                for idx_reset in range(request.n):
                    if newly_set_current_step_text_flags[idx_reset]:
                        rsp = [] if request.is_client_side_tokenization else ""
                        gen_states[idx_reset]["current_step_text"] = rsp

    async def _generate_batch(
        self,
        requests: list[vSurgeRequest],
    ) -> list[ReturnSample]:
        """
        Generates text completions for a batch of requests without streaming.

        This method processes multiple `vSurgeRequest` objects concurrently and
        returns a list of `ReturnSample` objects, each containing the final,
        fully generated text for the corresponding input request.

        Args:
            requests (tp.List[vSurgeRequest]): A list of `vSurgeRequest` objects.

        Returns:
            tp.List[ReturnSample]: A list of `ReturnSample` objects. Each object
                corresponds to an input request.
                - If a request completes successfully, its `ReturnSample` contains:
                    - `text`: The final accumulated text(s). Type `List[Union[str, List[str]]]`.
                              (e.g., `["final text"]` or `[["tok1", "tok2"]]` for n=1;
                              `["text1", "text2"]` or `[["t1"], ["t2"]]` for n>1).
                    - `token_ids`: Final token IDs. Type `List[List[int]]`.
                - If a request `req` with `req.n=1` results in an empty generation:
                    - `text`: `""` (if server-side tokenization) or `[]` (if client-side).
                    - `token_ids`: `[]`.
                - If a request `req` with `req.n > 0` (but not 1) results in empty generation:
                    - `text`: List of `n` empty strings/lists (e.g., `["", ""]` or `[[], []]`).
                    - `token_ids`: List of `n` empty lists (e.g., `[[], []]`).

        Raises:
            Exception: Propagates exceptions from the underlying `complete` calls.
        """

        async def _collect_last_yielded_item(
            request_obj: vSurgeRequest,
        ) -> ReturnSample | None:
            last_item: ReturnSample | None = None
            async for result_list in self.complete(request_obj):
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
                final_text_to_use = last_step_sample.accumulated_text
                final_time = last_step_sample.time_spent_computing
                final_tps = last_step_sample.tokens_per_second
                final_num_tokens = last_step_sample.num_generated_tokens

                processed_final_results.append(
                    ReturnSample(
                        text=final_text_to_use,
                        token_ids=last_step_sample.token_ids,
                        accumulated_text=last_step_sample.accumulated_text,
                        time_spent_computing=final_time,
                        tokens_per_second=final_tps,
                        num_generated_tokens=final_num_tokens,
                    )
                )
            else:
                empty_text_val_single = [] if original_req.is_client_side_tokenization else ""
                if original_req.n == 1:
                    processed_final_results.append(
                        ReturnSample(
                            text=empty_text_val_single,
                            token_ids=[],
                            accumulated_text=empty_text_val_single,
                            time_spent_computing=0.0,
                            tokens_per_second=0.0,
                            num_generated_tokens=0,
                        )
                    )
                else:
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

    async def _generate_stream(
        self, requests: list[vSurgeRequest]
    ) -> tp.AsyncGenerator[list[ReturnSample | None], None]:
        """
        Generates text completions for a batch of requests with streaming.

        This asynchronous generator processes multiple `vSurgeRequest` objects
        concurrently. It yields lists of `ReturnSample` objects (or None if a
        particular stream has finished). Each `ReturnSample` in the yielded list
        corresponds to one of the input requests and contains the latest chunk
        of generated text for that request (aggregating its 'n' internal generations).

        Args:
            requests (tp.List[vSurgeRequest]): A list of `vSurgeRequest` objects.

        Yields:
            tp.AsyncGenerator[tp.List[tp.Optional[ReturnSample]], None]:
                An asynchronous generator. Each yield is a list where each element
                is either a `ReturnSample` (containing the current chunk for one
                of the input requests) or `None` (if that request's stream has ended).
                Each `ReturnSample` object will have its fields (e.g., `text`, `token_ids`)
                as lists, corresponding to the `n` generations of its original `vSurgeRequest`.
                - `ReturnSample.text`: `List[Union[str, List[int]]]` - current chunk for each 'n'.
        """
        num_original_requests = len(requests)
        if num_original_requests == 0:
            if False:
                yield []
            return

        stream_iterators = [self.complete(req).__aiter__() for req in requests]

        latest_data_from_streams: list[ReturnSample | None] = [None] * num_original_requests

        active_stream_indices = list(range(num_original_requests))

        while active_stream_indices:
            tasks_for_this_sync_step = []

            for original_idx in active_stream_indices:

                async def fetch_next_for_sync(iterator, idx_tag):
                    try:
                        item_list = await iterator.__anext__()
                        return idx_tag, item_list[0] if item_list else None, None
                    except StopAsyncIteration:
                        return idx_tag, None, StopAsyncIteration()
                    except Exception as exc:
                        return idx_tag, None, exc

                tasks_for_this_sync_step.append(
                    asyncio.create_task(fetch_next_for_sync(stream_iterators[original_idx], original_idx))
                )

            if not tasks_for_this_sync_step:
                break
            done_tasks, _ = await asyncio.wait(
                tasks_for_this_sync_step,
                return_when=asyncio.ALL_COMPLETED,
            )

            newly_finished_indices_this_step = []
            error_occurred = None

            for task in done_tasks:
                original_idx, fetched_sample, error = task.result()

                if error:
                    if isinstance(error, StopAsyncIteration):
                        newly_finished_indices_this_step.append(original_idx)
                    else:
                        error_occurred = error
                        break
                else:
                    if fetched_sample is not None:
                        latest_data_from_streams[original_idx] = fetched_sample

            if error_occurred:
                logger.error(f"Error during _generate_stream sync step: {error_occurred}")
                raise error_occurred

            for idx_to_remove in newly_finished_indices_this_step:
                if idx_to_remove in active_stream_indices:
                    active_stream_indices.remove(idx_to_remove)

            yield list(latest_data_from_streams)
            if not active_stream_indices:
                break

    async def generate(
        self,
        prompts: str | tp.Sequence[str],
        sampling_params: SamplingParams | tp.Sequence[SamplingParams] | None = None,
        stream: bool = False,
    ) -> list[ReturnSample] | tp.AsyncGenerator[list[ReturnSample], None]:
        """
        Generates text completions for given prompts.

        This is the main public method for text generation. It can handle single
        or multiple prompts, with corresponding sampling parameters. Output can be
        streamed or returned as a batch.

        Args:
            prompts (tp.Union[str, tp.Sequence[str]]):
                A single prompt string or a sequence of prompt strings.
            sampling_params (tp.Optional[SamplingParams | tp.Sequence[SamplingParams]]):
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

        Returns:
            tp.Union[tp.List[ReturnSample], tp.AsyncGenerator[tp.List[ReturnSample], None]]:
                - If `stream` is False (batch mode):
                    Returns `tp.List[ReturnSample]`. Each `ReturnSample` in this list
                    corresponds to one input prompt. The `text` field of this `ReturnSample`
                    (and `token_ids`, `accumulated_text`) will be structured based on the `n`
                    value in its `SamplingParams` and whether generation was empty:
                    - For `n=1` and successful generation: `text` is `List[Union[str, List[str]]]`
                      (e.g., `["final text"]` or `[["tok1", "tok2"]]`).
                    - For `n>1` and successful generation: `text` is `List[Union[str, List[str]]]`
                      (e.g., `["text1", "text2"]` or `[["t1"], ["t2"]]`).
                    - For `n=1` and empty generation: `text` is `str` (e.g., `""`) or `List[int]` (e.g., `[]`).
                    - For `n>1` (or `n=0`) and empty generation: `text` is `List[str]` or `List[List[int]]`
                      (e.g., `["", ""]` or `[[], []]`).
                - If `stream` is True (streaming mode):
                    Returns `tp.AsyncGenerator[tp.List[tp.Optional[ReturnSample]], None]`.
                    Each yield from the generator is `tp.List[tp.Optional[ReturnSample]]`.
                    This list has one entry per input prompt. Each entry is either:
                    - A `ReturnSample` object containing the latest generated chunk for that prompt.
                      The `text` field (and similar fields like `token_ids`, `accumulated_text`)
                      of this `ReturnSample` is always `tp.List[tp.Union[str, tp.List[int]]]`,
                      where each element in this inner list corresponds to one of the `n`
                      generations for that prompt (e.g., `["chunk for n0"]` or `[[ids_n0], [ids_n1]]`).
                    - `None`, if the stream for that specific prompt has finished.

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

                async def empty_gen():
                    if False:
                        yield []

                return empty_gen()
            return []

        requests = [vSurgeRequest.from_sampling_params(p, sp) for p, sp in zip(prompts, sampling_params, strict=False)]
        if stream:
            return self._generate_stream(requests)
        else:
            return await self._generate_batch(requests)

    def __repr__(self):
        return f"vSurge(name={self.vsurge_name}, live={self.driver.live})"

    __str__ = __repr__
