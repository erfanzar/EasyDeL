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

"""BaseCfg and eSurgeCfg TypedDicts."""

from __future__ import annotations

import collections.abc
import typing as tp
from typing import Any, NotRequired, TypedDict

from easydel.infra.base_config import EasyDeLBaseConfigDict

from .model import OperationConfigsDict

if tp.TYPE_CHECKING:
    from easydel.inference.reasoning.abstract_reasoning import ReasoningParserName
    from easydel.inference.sampling_params import SamplingParams
    from easydel.inference.tools.abstract_tool import ToolParserName


class BaseCfg(TypedDict, total=False):
    """Container for base model configuration values and ejkernel operation overrides.

    Attributes:
        values: Base model configuration dictionary. Accepts an
            ``EasyDeLBaseConfigDict`` or a plain ``dict`` of key-value
            overrides applied during model initialization.
        operation_configs: Per-operation kernel configuration overrides.
            Maps operation names to their custom settings. ``None`` keeps
            the default kernel configurations.
    """

    values: NotRequired[EasyDeLBaseConfigDict | dict[str, Any]]
    operation_configs: NotRequired[OperationConfigsDict | None]


class eSurgeCfg(TypedDict, total=False):
    """Configuration for the eSurge high-throughput inference engine (PagedAttention-based).

    Attributes:
        max_model_len: Maximum sequence length (prompt + generation) the
            model can handle.
        min_input_pad: Minimum padding applied to input sequences for
            efficient batching.
        min_token_pad: Minimum token padding. ``None`` uses the engine
            default.
        max_num_seqs: Maximum number of sequences processed concurrently.
        max_num_seq_buckets: Bucket sizes for batching sequences of
            similar lengths. ``None`` for automatic selection.
        max_num_batched_tokens: Maximum total tokens across all sequences
            in a batch. ``None`` for automatic selection.
        hbm_utilization: Fraction of HBM (High Bandwidth Memory) to use
            for the KV cache (0.0--1.0).
        page_size: Number of tokens per KV cache page.
        use_aot_forward: Whether to use an ahead-of-time compiled forward
            pass.
        bind_graphstate_for_aot: Whether to bind the graph state into the
            AOT-compiled function for reduced overhead.
        enable_window_aware_runtime_cap: Whether to derive eSurge's runtime
            request cap from live KV-window page demand instead of the
            heuristic cache-cap estimate.
        enable_prefix_caching: Enable prefix caching to reuse the KV
            cache for shared prompt prefixes.
        auto_shard_model: Automatically shard the model across available
            devices.
        sharding_axis_dims: Axis dimensions for model sharding (e.g.,
            ``[1, 1, 1, -1]``).
        compile_runner: Whether to JIT-compile the runner step function.
        async_scheduling: Enable async sampled-token scheduling/materialization
            between scheduler iterations.
        runner_verbose: Enable verbose logging from the runner.
        verbose: Enable verbose logging from the engine.
        overlap_execution: Overlap decode execution with scheduling for
            lower latency.
        sampler_metrics: Collect and report sampling metrics.
        data_parallelism_axis: Name of the data parallelism axis (e.g.,
            ``"dp"``).
        esurge_name: Optional name identifier for this engine instance.
        reserve_tokens: Number of tokens to reserve for generation
            headroom.
        auto_truncate_prompt: Automatically truncate prompts that exceed
            ``max_model_len``.
        auto_cap_new_tokens: Automatically cap ``max_new_tokens`` to fit
            within ``max_model_len``.
        strict_context: Reject requests that would exceed the context
            window rather than truncating.
        truncate_mode: Truncation strategy when ``auto_truncate_prompt``
            is enabled (``"left"``, ``"right"``, or ``"middle"``).
        prefer_preserve_prompt: Prefer preserving the prompt over
            generation tokens when context is tight.
        decode_truncated_prompt: Include a note in output when the prompt
            was truncated.
        destroy_pages_on_pause: Free KV cache pages when a sequence is
            paused.
        detokenizer_max_states: Maximum concurrent detokenizer states.
            ``None`` for unlimited.
        worker_startup_timeout: Seconds to wait for spawned tokenizer and
            detokenizer workers to bind. ``None`` uses the engine default
            or environment override.
        idle_reset_seconds: Time in seconds of inactivity before
            resetting engine state. ``None`` to disable.
        idle_reset_min_interval: Minimum interval in seconds between idle
            resets.
        tokenizer_endpoint: Remote tokenizer endpoint URL. ``None`` for
            local tokenization.
        detokenizer_endpoint: Remote detokenizer endpoint URL. ``None``
            for local detokenization.
        sampling_params_callback: Callback to transform sampling params
            before use.
        extra_eos_token_ids: Additional token IDs treated as
            end-of-sequence.
        extra_stops: Additional stop strings for generation.
        silent_mode: Suppress all engine logging output.
        tool_parser: Name of the tool-call parser for structured output.
        reasoning_parser: Name of the reasoning/chain-of-thought parser.
        ignore_stop_strings_in_reasoning: Do not apply stop strings
            within reasoning blocks.
        distributed_mode: Enable distributed multi-host inference.
        distributed_role: Role in the distributed setup (``"auto"``,
            ``"coordinator"``, or ``"worker"``).
        distributed_service_name: DNS service name for distributed worker
            discovery.
        distributed_world_size: Total number of distributed workers.
            ``None`` for auto-detection.
        distributed_rank: Rank of this worker in the distributed group.
            ``None`` for auto-assignment.
        distributed_control_port: TCP port for the distributed control
            plane.
        distributed_control_bind_host: Host address to bind the control
            plane to.
        distributed_advertise_addr: Address advertised to other workers
            for connectivity.
        distributed_auth_token: Shared secret for authenticating
            distributed workers.
        distributed_step_timeout_s: Timeout in seconds for each
            distributed step.
        distributed_connect_timeout_s: Timeout in seconds for initial
            distributed connection.
        distributed_verify_sampling_digest: Verify sampling parameter
            consistency across workers.
    """

    max_model_len: NotRequired[int]
    min_input_pad: NotRequired[int]
    min_token_pad: NotRequired[int | None]
    max_num_seqs: NotRequired[int]
    max_num_seq_buckets: NotRequired[collections.abc.Sequence[int] | None]
    max_num_batched_tokens: NotRequired[int | None]
    hbm_utilization: NotRequired[float]
    page_size: NotRequired[int]
    use_aot_forward: NotRequired[bool]
    bind_graphstate_for_aot: NotRequired[bool]
    enable_window_aware_runtime_cap: NotRequired[bool]
    enable_prefix_caching: NotRequired[bool]
    auto_shard_model: NotRequired[bool]
    sharding_axis_dims: NotRequired[collections.abc.Sequence[int]]
    compile_runner: NotRequired[bool]
    async_scheduling: NotRequired[bool]
    runner_verbose: NotRequired[bool]
    verbose: NotRequired[bool]
    overlap_execution: NotRequired[bool]
    sampler_metrics: NotRequired[bool]
    data_parallelism_axis: NotRequired[str]
    esurge_name: NotRequired[str | None]
    reserve_tokens: NotRequired[int | None]
    auto_truncate_prompt: NotRequired[bool]
    auto_cap_new_tokens: NotRequired[bool]
    strict_context: NotRequired[bool]
    truncate_mode: NotRequired[tp.Literal["left", "right", "middle"]]
    prefer_preserve_prompt: NotRequired[bool]
    decode_truncated_prompt: NotRequired[bool]
    destroy_pages_on_pause: NotRequired[bool]
    detokenizer_max_states: NotRequired[int | None]
    worker_startup_timeout: NotRequired[float | None]
    idle_reset_seconds: NotRequired[float | None]
    idle_reset_min_interval: NotRequired[float]
    tokenizer_endpoint: NotRequired[str | None]
    detokenizer_endpoint: NotRequired[str | None]
    sampling_params_callback: NotRequired[
        tp.Callable[["SamplingParams", dict[str, tp.Any]], "SamplingParams | None"] | None
    ]
    long_prefill_token_threshold: NotRequired[int | None]
    extra_eos_token_ids: NotRequired[list[int] | None]
    extra_stops: NotRequired[str | list[str] | None]
    silent_mode: NotRequired[bool]
    tool_parser: NotRequired["ToolParserName | None"]
    reasoning_parser: NotRequired["ReasoningParserName | None"]
    ignore_stop_strings_in_reasoning: NotRequired[bool]

    # Distributed inference controls
    distributed_mode: NotRequired[bool]
    distributed_role: NotRequired[tp.Literal["auto", "coordinator", "worker"]]
    distributed_service_name: NotRequired[str | None]
    distributed_world_size: NotRequired[int | None]
    distributed_rank: NotRequired[int | None]
    distributed_control_port: NotRequired[int]
    distributed_control_bind_host: NotRequired[str]
    distributed_advertise_addr: NotRequired[str | None]
    distributed_auth_token: NotRequired[str | None]
    distributed_step_timeout_s: NotRequired[float]
    distributed_connect_timeout_s: NotRequired[float]
    distributed_verify_sampling_digest: NotRequired[bool]
