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

"""eSurge Model Runner - High-performance inference execution engine.

This module implements the core execution logic for the eSurge inference engine,
providing efficient model execution with advanced features like paged attention,
dynamic batching, and compilation caching.

Key Components:
    ExecutionManager: Manages compiled execution functions for different batch/token configurations
    eSurgeRunner: Main runner class that orchestrates model execution

Architecture:
    The module uses a two-stage compilation strategy:
    1. Pre-compilation of functions for different token/batch size combinations
    2. Runtime selection of appropriate compiled function based on input shape

Performance Features:
    - Paged attention for efficient KV cache management
    - Vectorized operations for batch processing
    - Pre-allocated buffers to minimize memory allocation
    - Compilation caching to avoid recompilation
    - Progress logging for long compilation processes

Example:
    >>> from easydel.infra import EasyDeLBaseModule
    >>> from easydel.inference.esurge.runners import eSurgeRunner
    >>>
    >>> # Initialize model
    >>> model = EasyDeLBaseModule.from_pretrained("model-name")
    >>>
    >>> # Create runner
    >>> runner = eSurgeRunner(
    ...     model=model,
    ...     max_model_len=2048,
    ...     max_num_seqs=8,
    ...     hbm_utilization=0.9
    ... )
    >>>
    >>> # Compile for different configurations
    >>> runner.compile()
    >>>
    >>> # Execute model
    >>> output = runner.execute_model(scheduler_output)
"""

from __future__ import annotations

import time
import typing
from bisect import bisect_left
from concurrent.futures import Future

import flax
import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from ..metrics import get_metrics_collector
from ..outputs import ModelRunnerOutput
from ..scheduler import SchedulerOutput
from ..utils import model_uses_mrope
from .async_types import AsyncPreResults
from .execution_manager import ExecutionManager
from .sequence_buffer import (
    SequenceBuffer,
    build_allowed_mask,
    build_sampling_arrays,
    fill_slice,
    move_row,
    pack_prompts,
    swap_rows,
)
from .states import CachedRequestState

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

logger = get_logger("eSurge")


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int, min_input_pad: int) -> int:
    """Calculate padded request count for compilation efficiency.

    Pads the number of requests to powers of 2 (up to 8) or the nearest
    power of 2 above 8. This reduces the number of unique compilations
    needed while maintaining good utilization.

    Args:
        x: Actual number of requests
        upper_limit: Maximum allowed requests

    Returns:
        int: Padded request count, capped at upper_limit

    Example:
        >>> _get_padded_num_reqs_with_upper_limit(3, 32)   # Returns 8
        >>> _get_padded_num_reqs_with_upper_limit(10, 32)  # Returns 16
        >>> _get_padded_num_reqs_with_upper_limit(20, 16)  # Returns 16
    """
    res = min_input_pad if x <= min_input_pad else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


class eSurgeRunner:
    """High-performance model runner for efficient batched inference.

    The eSurgeRunner orchestrates model execution with advanced features:
    - Paged attention for memory-efficient KV cache management
    - Dynamic batching with request scheduling
    - Pre-allocated buffers for zero-copy operations
    - Vectorized token processing
    - Compilation caching for different batch/sequence configurations

    The runner maintains an internal state of active requests and manages
    their lifecycle from prompt processing through token generation.

    Architecture:
        Request Flow:
        1. Scheduler provides requests to execute
        2. Runner updates internal state (add/remove requests)
        3. Prepares inputs with proper padding and batching
        4. Executes model using pre-compiled functions
        5. Processes sampled tokens and updates buffers
        6. Returns results to scheduler

    Memory Management:
        - Pre-allocated buffers for common operations
        - Paged KV cache with configurable page size
        - Efficient slot mapping for attention
        - Buffer reuse across batches

    Attributes:
        model: The EasyDeL model to run
        metadata: Paged attention metadata
        max_num_seqs: Maximum concurrent sequences
        max_model_len: Maximum sequence length
        executor_manager: Manages compiled functions
        sequence_buffer: Manages active sequences
        requests: Active request states

    Example:
        >>> runner = eSurgeRunner(
        ...     model=model,
        ...     max_model_len=2048,
        ...     max_num_seqs=8,
        ...     hbm_utilization=0.9,
        ...     page_size=128
        ... )
        >>>
        >>> # Compile for all configurations
        >>> runner.compile()
        >>>
        >>> # Execute requests from scheduler
        >>> output = runner.execute_model(scheduler_output)
        >>>
        >>> # Process results
        >>> for req_id, tokens in zip(output.req_ids, output.sampled_token_ids):
        ...     print(f"Request {req_id}: {tokens}")
    """

    def __init__(
        self,
        model: EasyDeLBaseModule,
        hbm_utilization: float = 0.5,
        page_size: int = 128,
        max_model_len: int = 2**13,
        min_input_pad: int = 256,
        min_token_pad: int | None = None,
        max_num_seqs: int = 16,
        max_num_seq_buckets: list[int] | None = None,
        use_aot_forward: bool = True,
        verbose: bool = False,
        enable_overlap_execution: bool = False,
        enable_sampler_metrics: bool = False,
    ):
        logger.debug(f"Initializing eSurgeRunner with {max_model_len=}, {max_num_seqs=}")
        logger.debug(f"Configuration: {hbm_utilization=}, {page_size=}")
        self.model = model.esurge_compatible_model
        self.metadata = model.create_ragged_page_cache_config(
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_length=max_model_len,
        )
        self.max_num_seq_buckets = self._init_seq_buckets(max_num_seq_buckets, max_num_seqs, min_input_pad)
        self.max_num_seqs = max_num_seqs
        self.max_num_reqs = self.max_num_seq_buckets[-1]

        self.max_model_len = max_model_len
        self.min_input_pad = max(min_input_pad, self.max_num_seq_buckets[0])
        self.page_size = int(self.metadata.page_size)
        self.max_pages_per_req = int(self.metadata.max_num_pages_per_req)

        min_token_pad_i = self.min_input_pad if min_token_pad is None else int(min_token_pad)
        min_token_pad_i = min(min_token_pad_i, int(self.max_model_len))
        self.num_tokens_paddings = self._get_token_paddings(
            min_token_size=min_token_pad_i,
            max_token_size=self.max_model_len,
            padding_gap=0,
        )
        self.max_num_tokens = self.num_tokens_paddings[-1]

        logger.debug("Creating ExecutionManager and initializing pages cache")
        self.executor_manager = ExecutionManager(
            model=model,
            use_aot_forward=use_aot_forward,
            min_input_pad=self.min_input_pad,
            max_model_len=max_model_len,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            metadata=self.metadata,
            verbose=verbose,
        )
        self.log_it = logger.info if verbose else logger.debug
        self._setup_variables()
        self.enable_overlap_execution = enable_overlap_execution
        self.enable_sampler_metrics = enable_sampler_metrics

        # Perf logging state (kept lightweight; no allocations in the hot path).
        self._perf_iteration = 0
        self._perf_tps_ema: float | None = None
        self._perf_alpha = 0.2

        # Async scheduling state
        self._pre_async_results: AsyncPreResults | None = None
        self._executor: typing.Any = None  # ThreadPoolExecutor, typed as Any to avoid circular import
        logger.debug("eSurgeRunner initialization complete")

    @property
    def mesh(self):
        return self.model.mesh

    @property
    def _empty_sharding(self):
        return jax.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

    @staticmethod
    def _get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int) -> list[int]:
        """Generate padding sizes for efficient compilation.

        Args:
            min_token_size: Minimum token size (must be power of 2)
            max_token_size: Maximum token size to cover
            padding_gap: Gap between padding sizes (0 for exponential growth)

        Returns:
            List of padding sizes
        """
        if not ((min_token_size & (min_token_size - 1) == 0) and min_token_size > 0):
            logger.error(f"Invalid min_token_size={min_token_size}, must be power of 2")
            raise ValueError(f"min_token_size must be a power of 2, got {min_token_size}")
        paddings = []
        num = min_token_size

        if padding_gap == 0:
            while num <= max_token_size:
                paddings.append(num)
                num *= 2
        else:
            while num <= padding_gap:
                paddings.append(num)
                num *= 2
            num //= 2
            while num < max_token_size:
                num += padding_gap
                paddings.append(num)
        if paddings[-1] != max_token_size:
            paddings.append(max_token_size)
        return paddings

    @staticmethod
    def _get_request_paddings(min_bucket: int, max_bucket: int) -> list[int]:
        min_bucket = max(1, min(min_bucket, max_bucket))
        buckets: list[int] = []
        current = min_bucket
        while current < max_bucket:
            buckets.append(current)
            current *= 2
        if not buckets or buckets[-1] != max_bucket:
            buckets.append(max_bucket)
        return buckets

    def _init_seq_buckets(
        self,
        user_buckets: list[int] | None,
        max_num_seqs: int,
        min_input_pad: int,
    ) -> list[int]:
        if user_buckets:
            buckets = sorted({int(b) for b in user_buckets if 0 < int(b) <= max_num_seqs})
        else:
            buckets = self._get_request_paddings(min_input_pad, max_num_seqs)
        if not buckets or buckets[-1] != max_num_seqs:
            buckets.append(max_num_seqs)
        return buckets

    def _get_current_bucket(self, num_reqs: int) -> int:
        """Select the smallest bucket that can accommodate num_reqs.

        Args:
            num_reqs: Number of active requests

        Returns:
            Smallest sufficient bucket size from self.max_num_seq_buckets
        """
        if num_reqs <= 0:
            return self.max_num_seq_buckets[0]
        for bucket in self.max_num_seq_buckets:
            if num_reqs <= bucket:
                return bucket
        return self.max_num_seq_buckets[-1]

    def _setup_variables(self):
        """Initialize internal variables and preallocate reusable buffers."""
        self.num_reqs_max_model_len = min(self.metadata.get_max_num_seqs(), self.max_num_reqs)
        self.num_reqs_most_model_len = self.num_reqs_max_model_len
        self.requests: dict[str, CachedRequestState] = {}
        logger.debug(f"Token padding sizes: {len(self.num_tokens_paddings)} levels, max={self.max_num_tokens}")

        logger.debug(
            f"Creating sequence buffer for max_num_reqs={self.max_num_reqs}, max_model_len={self.max_model_len}"
        )
        self.sequence_buffer = SequenceBuffer(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.model.config.get_text_config().vocab_size,
            page_sizes=[self.metadata.page_size],
            sharding=self._empty_sharding,
        )

        self.arange = jnp.arange(self.max_num_tokens, dtype=jnp.int32)
        self.arange_np = jnp.arange(self.max_num_reqs, dtype=jnp.int32)

        self.input_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32, device=self._empty_sharding)
        self.position_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32, device=self._empty_sharding)
        self.num_tokens_paddings_arr = jnp.array(self.num_tokens_paddings, dtype=jnp.int32, device=self._empty_sharding)
        self.scheduled_full_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32, device=self._empty_sharding)
        self.req_num_tokens_full_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32, device=self._empty_sharding)
        self.active_mask_full_buf = jnp.zeros((self.max_num_reqs,), dtype=bool, device=self._empty_sharding)

        # Host-side scratch buffers (avoid per-step NumPy allocations in hot path).
        self._scheduled_full_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._active_mask_full_cpu = np.zeros((self.max_num_reqs,), dtype=bool)
        self._req_num_tokens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)

        # VLM host-side scratch buffers keyed by `num_tokens_static` (avoid repeated
        # large allocations while keeping the step-function input pytree stable).
        self._vlm_cpu_buffers: dict[
            int,
            tuple[
                np.ndarray,  # prefill_embeds_cpu
                np.ndarray,  # prefill_embeds_mask_cpu
                np.ndarray | None,  # mrope_position_ids_cpu
                np.ndarray | None,  # visual_pos_masks_cpu
                list[np.ndarray] | None,  # deepstack_visual_embeds_cpu
            ],
        ] = {}

    def _get_vlm_cpu_buffers(
        self,
        *,
        num_tokens_static: int,
        uses_mrope_model: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, list[np.ndarray] | None]:
        num_tokens_static = int(num_tokens_static)
        cached = self._vlm_cpu_buffers.get(num_tokens_static)
        if cached is None:
            hidden_size = int(getattr(self.model.config.get_text_config(), "hidden_size", 0) or 1)

            prefill_embeds_cpu = np.zeros((num_tokens_static, hidden_size), dtype=np.float16)
            prefill_embeds_mask_cpu = np.zeros((num_tokens_static,), dtype=bool)

            mrope_position_ids_cpu = None
            visual_pos_masks_cpu = None
            deepstack_visual_embeds_cpu = None
            if uses_mrope_model:
                mrope_position_ids_cpu = np.zeros((3, num_tokens_static), dtype=np.int32)
                deepstack_indexes = getattr(
                    getattr(self.model.config, "vision_config", None),
                    "deepstack_visual_indexes",
                    None,
                )
                deepstack_layers = len(deepstack_indexes) if deepstack_indexes else 0
                if deepstack_layers:
                    visual_pos_masks_cpu = np.zeros((num_tokens_static,), dtype=bool)
                    deepstack_visual_embeds_cpu = [
                        np.zeros((num_tokens_static, hidden_size), dtype=np.float16) for _ in range(deepstack_layers)
                    ]

            cached = (
                prefill_embeds_cpu,
                prefill_embeds_mask_cpu,
                mrope_position_ids_cpu,
                visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu,
            )
            self._vlm_cpu_buffers[num_tokens_static] = cached

        (
            prefill_embeds_cpu,
            prefill_embeds_mask_cpu,
            mrope_position_ids_cpu,
            visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu,
        ) = cached

        # Clear masks/position ids each step; large embed buffers are overwritten only
        # for the masked regions, and ignored otherwise.
        prefill_embeds_mask_cpu.fill(False)
        if mrope_position_ids_cpu is not None:
            mrope_position_ids_cpu.fill(0)
        if visual_pos_masks_cpu is not None:
            visual_pos_masks_cpu.fill(False)

        return cached

    def _precompile_jitted_helpers(
        self,
        reqs_padds: list[int],
        prompt_len_buckets: list[int],
        precompile_allowed_mask: bool = False,
        allowed_max: int = 512,
    ) -> None:
        logger.info("Precompiling eSurgeRunner helper kernels")

        B = self.max_num_reqs
        T = self.max_model_len
        V = int(self.model.config.get_text_config().vocab_size)

        token_ids = jnp.zeros((B, T), dtype=jnp.int32)
        num_prompt_tokens = jnp.zeros((B,), dtype=jnp.int32)

        temperature = jnp.zeros((B,), dtype=jnp.float32)
        min_p = jnp.zeros((B,), dtype=jnp.float32)
        top_p = jnp.ones((B,), dtype=jnp.float32)
        top_k = jnp.zeros((B,), dtype=jnp.int32)

        for pr_len in prompt_len_buckets:
            pr_len = min(pr_len, self.max_model_len)
            for pr_reqs in reqs_padds:
                try:
                    lowered = pack_prompts.lower(
                        token_ids,
                        num_prompt_tokens,
                        padded_num_reqs=pr_reqs,
                        padded_prompt_len=pr_len,
                        pad_id=V,
                    )
                    _ = lowered.compile()
                    logger.debug(f"pack_prompts compiled for (padded_num_reqs={pr_reqs}, padded_prompt_len={pr_len})")
                except Exception as e:
                    logger.debug(f"pack_prompts skip ({pr_reqs}, {pr_len}): {e}")

        for pr_reqs in reqs_padds:
            try:
                lowered = build_sampling_arrays.lower(
                    temperature,
                    min_p,
                    top_p,
                    top_k,
                    jnp.int32(min(pr_reqs, B)),  # num_reqs <= padded_num_reqs
                    padded_num_reqs=pr_reqs,
                )
                _ = lowered.compile()
                logger.debug(f"build_sampling_arrays compiled for (padded_num_reqs={pr_reqs})")
            except Exception as e:
                logger.debug(f"build_sampling_arrays skip ({pr_reqs}): {e}")

        for pr_reqs in reqs_padds:
            try:
                lowered = fill_slice.lower(
                    temperature,
                    jnp.float32(0.0),
                    int(pr_reqs),
                    int(pr_reqs),
                )
                _ = lowered.compile()
                logger.debug(f"fill_slice compiled for (num_reqs={pr_reqs}, padded_num_reqs={pr_reqs})")
            except Exception as e:
                logger.debug(f"fill_slice skip ({pr_reqs}): {e}")

        try:
            _ = swap_rows.lower(token_ids, jnp.int32(0), jnp.int32(1)).compile()
            _ = move_row.lower(token_ids, jnp.int32(0), jnp.int32(1)).compile()
            logger.debug("swap_rows and move_row compiled")
        except Exception as e:
            logger.debug(f"swap_rows/move_row skip: {e}")

        if precompile_allowed_mask:
            max_allowed = int(min(allowed_max, V))
            allowed_ids_padded = jnp.zeros((B, max_allowed), dtype=jnp.int32)
            allowed_lens = jnp.zeros((B,), dtype=jnp.int32)
            try:
                lowered = build_allowed_mask.lower(
                    allowed_ids_padded,
                    allowed_lens,
                    vocab_size=int(V),
                    max_allowed=max_allowed,
                )
                _ = lowered.compile()
                logger.debug(f"build_allowed_mask compiled for (B={B}, V={V}, max_allowed={max_allowed})")
            except Exception as e:
                logger.debug(f"build_allowed_mask skip (V={V}, max_allowed={max_allowed}): {e}")

        logger.info("Helper kernel precompilation finished")

    def compile(self):
        """Compile the model for all token padding sizes."""
        logger.info("Starting eSurgeRunner compilation")
        logger.debug(
            f"Compiling for {len(self.num_tokens_paddings)} token padding sizes: {self.num_tokens_paddings[:5]}..."
            if len(self.num_tokens_paddings) > 5
            else f"Compiling for token padding sizes: {self.num_tokens_paddings}"
        )

        self.executor_manager.compile(
            num_tokens_paddings=self.num_tokens_paddings,
            num_reqs_max_model_len=self.num_reqs_max_model_len,
            max_pages_per_req=self.max_pages_per_req,
            max_num_reqs=self.max_num_reqs,
            metadata=self.metadata,
            num_reqs_paddings=self.max_num_seq_buckets,
        )

        self._precompile_jitted_helpers(
            reqs_padds=self.max_num_seq_buckets,
            prompt_len_buckets=[min(n, self.max_model_len) for n in self.num_tokens_paddings],
            precompile_allowed_mask=False,
            allowed_max=4096,
        )

    def update_model_weights(
        self,
        model: EasyDeLBaseModule | None = None,
        *,
        graphdef=None,
        graphstate=None,
        graphother=None,
        reset_state: bool = True,
    ) -> None:
        """Update the runner's model weights/graphs and optionally reset state.

        Args:
            model: Optional EasyDeL model instance providing new weights. If
                omitted, graph components must be supplied explicitly.
            graphdef: Optional graphdef override.
            graphstate: Optional graphstate override.
            graphother: Optional graphother override.
            reset_state: When True (default) reinitializes internal buffers and
                cached requests to ensure the new weights are applied cleanly.

        Raises:
            RuntimeError: If active requests exist while reset_state is True.
        """
        if reset_state and self.requests:
            raise RuntimeError("Cannot update model weights while requests are active")

        if model is None:
            assert graphdef is not None
            assert graphstate is not None
            assert graphother is not None
            model = flax.nnx.merge(graphdef, graphstate, graphother)

        model = model.esurge_compatible_model
        graphdef = model.graphdef
        self.model = model

        self.executor_manager.update_graphs(
            model=model,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
        )

        if reset_state:
            self._setup_variables()

    def destroy_kv_cache(self) -> None:
        """Destroy the current ragged KV cache to release memory."""
        logger.info("Destroying eSurgeRunner ragged KV cache pages")
        self.executor_manager.kv_pages = None

    def initialize_kv_cache(self) -> None:
        """Reinitialize the ragged KV cache if it has been destroyed."""
        if self.executor_manager.kv_pages is not None:
            logger.debug("KV cache already initialized; skipping reallocation")
            return
        logger.info("Reinitializing eSurgeRunner ragged KV cache pages")
        self.executor_manager.kv_pages = self.model.init_ragged_pages(self.metadata)

    def _precompute_vlm_prefill(self, req_state: CachedRequestState) -> None:
        """Precompute prompt embeddings (+ optional mRoPE indices) for VLM requests.

        Some VLM base models compute mRoPE indices via NumPy/data-dependent control-flow
        which is not compatible with JIT/AOT inside the compiled eSurge step. We run
        those parts eagerly here and store host-side arrays for later reuse.
        """
        if req_state.vision_processed:
            return

        uses_mrope = model_uses_mrope(self.model)

        # If raw vision inputs are missing but the request is marked as "has_vision"
        # (e.g. only cached mm_features remain), skip precompute and treat it as processed.
        if req_state.pixel_values is None and req_state.pixel_values_videos is None:
            req_state._vision_processed = True
            return

        if req_state.prefill_inputs_embeds is not None and (
            not uses_mrope or req_state.prefill_position_ids is not None
        ):
            req_state.clear_vision_data()
            return

        prompt_ids = np.asarray(req_state.prompt_token_ids, dtype=np.int32)[None, :]
        input_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
        attention_mask = jnp.ones(input_ids.shape, dtype=jnp.int32)

        try:
            embed_kwargs: dict[str, typing.Any] = {"attention_mask": attention_mask}
            if req_state.pixel_values is not None:
                embed_kwargs["pixel_values"] = req_state.pixel_values
                if req_state.image_grid_thw is not None:
                    embed_kwargs["image_grid_thw"] = req_state.image_grid_thw
            if req_state.pixel_values_videos is not None:
                embed_kwargs["pixel_values_videos"] = req_state.pixel_values_videos
                if req_state.video_grid_thw is not None:
                    embed_kwargs["video_grid_thw"] = req_state.video_grid_thw

            inputs_embeds, info = self.model.compute_embedding_with_info(input_ids, **embed_kwargs)
        except Exception as exc:
            logger.warning(f"VLM precompute failed for req_id={req_state.req_id}: {exc}")
            return

        # Store host-side views (keeps compiled step free of vision preprocessing).
        embeds_host = np.asarray(jax.device_get(inputs_embeds))
        req_state.prefill_inputs_embeds = embeds_host[0]

        if getattr(info, "position_ids", None) is not None:
            pos_host = np.asarray(jax.device_get(info.position_ids))
            if pos_host.ndim == 3:
                pos_host = pos_host[:, 0, :]
            req_state.prefill_position_ids = pos_host.astype(np.int32, copy=False)

        if getattr(info, "rope_deltas", None) is not None:
            req_state.prefill_rope_deltas = np.asarray(jax.device_get(info.rope_deltas)).astype(np.int32, copy=False)

        if getattr(info, "visual_pos_masks", None) is not None:
            mask_host = np.asarray(jax.device_get(info.visual_pos_masks))
            if mask_host.ndim == 2:
                mask_host = mask_host[0]
            req_state.prefill_visual_pos_masks = mask_host.astype(bool, copy=False)

        if getattr(info, "deepstack_visual_embeds", None) is not None:
            ds_list = []
            for arr in info.deepstack_visual_embeds:
                ds_list.append(np.asarray(jax.device_get(arr)))
            req_state.prefill_deepstack_visual_embeds = ds_list

        # Raw vision tensors are no longer needed once embeddings are cached.
        req_state.clear_vision_data()

    def _update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Update internal states based on scheduler output.

        Synchronizes the runner's internal state with the scheduler's decisions.
        Handles request lifecycle: adding new requests, removing finished ones,
        updating cached requests, and managing the sequence buffer.

        State Updates:
            1. Remove finished requests from tracking
            2. Remove unscheduled requests from buffer
            3. Add new requests with their metadata
            4. Update cached request states
            5. Reorganize sequence buffer for efficiency

        Args:
            scheduler_output: Contains request scheduling decisions including:
                - finished_req_ids: Requests that completed
                - scheduled_new_reqs: New requests to add
                - scheduled_cached_reqs: Existing requests to update
                - num_scheduled_tokens: Tokens to generate per request

        Returns:
            True if state changed (requests added/removed), indicating
            potential buffer reorganization. False if no changes occurred.

        Side Effects:
            - Updates self.requests dictionary
            - Modifies sequence buffer contents
            - May trigger buffer condensation

        Note:
            This method is called at the beginning of each execution cycle
            to ensure the runner's state matches the scheduler's decisions.
        """
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # 2) Remove finished from sequence buffer (functional)
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # 3) Remove unscheduled requests from buffer
        scheduled_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
        cached_req_ids = set(self.sequence_buffer.req_id_to_index.keys())
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        for req_id in unscheduled_req_ids:
            req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # 4) Add new requests to tracking
        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.sampling_params is not None, "Pooling not supported in TPU"
            req_id = new_req_data.req_id
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=new_req_data.sampling_params,
                generator=None,
                page_ids=new_req_data.page_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                # Vision-language model data
                pixel_values=new_req_data.pixel_values,
                image_grid_thw=new_req_data.image_grid_thw,
                pixel_values_videos=new_req_data.pixel_values_videos,
                video_grid_thw=new_req_data.video_grid_thw,
                mm_features=new_req_data.mm_features,
            )
            req_ids_to_add.append(req_id)

        # 5) Update cached requests and page tables
        req_data = scheduler_output.scheduled_cached_reqs
        upd_req_indices: list[int] = []
        upd_num_computed_vals: list[int] = []
        batched_page_rows: list[tuple[int, tuple[list[int], ...]]] = []

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue

            nct = req_data.num_computed_tokens[i]
            new_page_ids = req_data.new_page_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            req_state.num_computed_tokens = nct
            if not resumed_from_preemption:
                for page_ids, new_ids in zip(req_state.page_ids, new_page_ids, strict=False):
                    page_ids.extend(new_ids)
            else:
                req_state.page_ids = new_page_ids

            req_index = self.sequence_buffer.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue

            upd_req_indices.append(req_index)
            upd_num_computed_vals.append(int(nct))
            if resumed_from_preemption:
                # Resumed requests may provide a full replacement page table.
                self.sequence_buffer.page_table.add_row(new_page_ids, req_index)
            else:
                # Running requests report only incremental page allocations.
                if any(len(ids) for ids in new_page_ids):
                    batched_page_rows.append((req_index, new_page_ids))

        if upd_req_indices:
            # num_computed_tokens is now a NumPy array, use standard indexing
            idx_arr = np.array(upd_req_indices, dtype=np.int32)
            val_arr = np.array(upd_num_computed_vals, dtype=np.int32)
            self.sequence_buffer.num_computed_tokens[idx_arr] = val_arr

        if batched_page_rows:
            indices = [ix for ix, _ in batched_page_rows]
            pages_per_req = [ids for _, ids in batched_page_rows]
            self.sequence_buffer.page_table.append_rows_batch(pages_per_req, indices)

        # 6) Add new / reinserted requests
        # Sort in reverse order and pop() to get highest indices first (avoid reusing index 0/1)
        # This prevents KV cache corruption from repeatedly reusing low indices
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            # Pop() from reverse-sorted list gives highest index first
            reuse_index = removed_req_indices.pop() if removed_req_indices else None
            self.sequence_buffer.add_request(req_state, reuse_index)

        # 7) Condense to remove holes
        if removed_req_indices:
            self.sequence_buffer.condense(removed_req_indices)

        # Drop cached VLM prompt helpers once prefill is complete to free host RAM.
        for req_state in self.requests.values():
            if (
                req_state.prefill_inputs_embeds is not None
                and req_state.num_computed_tokens >= req_state.num_prompt_tokens
            ):
                req_state.prefill_inputs_embeds = None
                req_state.prefill_position_ids = None
                req_state.prefill_visual_pos_masks = None
                req_state.prefill_deepstack_visual_embeds = None

        has_changes = len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0
        return has_changes

    def _modify_prev_results(self) -> None:
        """Apply previous iteration's tokens to sequence buffer.

        This method is called at the beginning of each iteration when async
        scheduling is enabled. It retrieves the tokens that were sampled
        asynchronously in the previous iteration and applies them to the
        sequence buffer.

        The method blocks until the async token transfer is complete, then
        updates the token_ids array and request output_token_ids lists.

        Note:
            This method should only be called when self._pre_async_results is not None.
        """
        if self._pre_async_results is None:
            return

        pre_req_ids = self._pre_async_results.req_ids
        pre_next_tokens = self._pre_async_results.next_tokens
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_indices = self._pre_async_results.discard_sampled_tokens_req_indices

        # Block until tokens are ready (async copy to host completes)
        next_tokens_cpu = np.asarray(jax.device_get(pre_next_tokens))
        selected_token_ids = np.expand_dims(next_tokens_cpu[: len(pre_req_ids)], 1)

        # Mask out discarded tokens
        valid_sampled_token_ids = [token_id for token_id in selected_token_ids]
        for i in pre_discard_indices:
            valid_sampled_token_ids[i] = np.array([])

        # Apply tokens to sequence buffer
        for pre_req_idx, req_state, _ in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if len(sampled_ids) == 0:
                continue

            # Check if request is still active
            req_id = pre_req_ids[pre_req_idx]
            if req_id not in self.sequence_buffer.req_id_to_index:
                continue

            req_idx = self.sequence_buffer.req_id_to_index[req_id]
            assert req_state is self.requests[req_id], "Request state mismatch"

            # Update token_ids array (replace placeholder)
            end_idx = self.sequence_buffer.num_tokens_no_spec[req_idx]
            start_idx = end_idx - 1
            assert end_idx <= self.max_model_len, f"Token count {end_idx} exceeds max_model_len {self.max_model_len}"

            self.sequence_buffer.token_ids[req_idx, start_idx:end_idx] = sampled_ids
            # Replace placeholder in output_token_ids
            req_state.output_token_ids[-1] = int(sampled_ids[-1])

    def _update_placeholder(
        self,
        discard_sampled_tokens_req_indices: list[int],
        request_seq_lens: list[tuple[int, CachedRequestState, int]],
    ) -> dict[str, int]:
        """Set placeholders for tokens not yet generated.

        When async scheduling is enabled, this method is called after the
        forward pass to set placeholder tokens (0) for requests that will
        generate tokens. The actual tokens will be filled in during the
        next iteration via _modify_prev_results().

        Args:
            discard_sampled_tokens_req_indices: Indices of requests whose
                tokens should be discarded (e.g., partial prefill).
            request_seq_lens: List of (req_idx, req_state, seq_len) tuples
                for requests that generated tokens.

        Returns:
            Mapping from request ID to index for placeholder replacement.

        Note:
            This method updates num_tokens_no_spec and num_tokens in the
            sequence buffer, and appends placeholder (0) to output_token_ids.
        """
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_set = set(discard_sampled_tokens_req_indices)

        for req_idx, req_state, _ in request_seq_lens:
            if req_idx in discard_set:
                continue

            start_idx = self.sequence_buffer.num_tokens_no_spec[req_idx]
            end_idx = start_idx + 1  # Assume 1 token (no spec decode yet)

            assert end_idx <= self.max_model_len, (
                f"Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: {self.max_model_len}"
            )

            # Update buffer state
            self.sequence_buffer.num_tokens_no_spec[req_idx] = end_idx
            self.sequence_buffer.num_tokens[req_idx] = end_idx

            # Add placeholder (0) to output
            req_state.output_token_ids.extend([0])
            placeholder_req_id_to_index[req_state.req_id] = req_idx

        return placeholder_req_id_to_index

    def _reorder_decode_first(self, scheduler_output: SchedulerOutput) -> None:
        """Reorder active requests so decode tokens are placed first."""
        i, j = 0, self.sequence_buffer.num_reqs - 1
        while i < j:
            i_req_id = self.sequence_buffer.req_ids[i]
            j_req_id = self.sequence_buffer.req_ids[j]
            if i_req_id is None or j_req_id is None:
                break

            i_is_decode = (
                scheduler_output.num_scheduled_tokens.get(i_req_id, 0) == 1
                and self.sequence_buffer.num_computed_tokens[i] > 0
            )
            j_is_decode = (
                scheduler_output.num_scheduled_tokens.get(j_req_id, 0) == 1
                and self.sequence_buffer.num_computed_tokens[j] > 0
            )

            if i_is_decode:
                i += 1
            elif not j_is_decode:
                j -= 1
            else:
                # Swap to move a decode request forward.
                self.sequence_buffer.swap_states(i, j)
                i += 1
                j -= 1

    def _execute_model_impl(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute the model on scheduled requests.

        Main entry point for model execution. Processes all scheduled requests
        in batches, handling state updates, input preparation, model execution,
        and token processing.

        The method handles:
        1. State synchronization with scheduler
        2. Batch-wise processing of requests
        3. Token generation and sampling
        4. Buffer updates and metrics logging

        Args:
            scheduler_output: Output from the scheduler containing:
                - Requests to process
                - Tokens to generate per request
                - Finished/new/cached request information

        Returns:
            ModelRunnerOutput: Contains:
                - req_ids: List of processed request IDs
                - sampled_token_ids: Generated tokens per request
                - logprobs: Log probabilities (if requested)
                - Timing and debugging information

        Note:
            The method processes requests in batches when they exceed
            the maximum model length, ensuring all requests are handled
            efficiently without exceeding memory constraints.
        """
        execution_start_time = time.time()

        updating_states_start = time.time()
        self._update_states(scheduler_output)
        updating_states_time = time.time() - updating_states_start

        # Apply previous async results if available
        prev_async_start = time.time()
        if self._pre_async_results is not None:
            self._modify_prev_results()
            self._pre_async_results = None  # Clear after applying
        prev_async_time = time.time() - prev_async_start

        # Align ordering with TPU runner: decode requests first.
        if self.sequence_buffer.num_reqs > 1:
            self._reorder_decode_first(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                finished_sending=None,
                finished_recving=None,
                num_nans_in_logits=None,
            )

        start_index = 0
        total_step_time = 0.0
        total_post_proc_time = 0.0

        req_ids_all: list[str] = []
        sampled_token_ids_all: list[list[int]] = []
        token_logprobs: dict[str, float] = {}

        # Window-level perf aggregation (a single scheduler step can span multiple windows).
        num_windows = 0
        total_exec_time = 0.0
        total_sample_time = 0.0
        total_prep_time = 0.0
        total_prep_host_time = 0.0
        total_prep_put_time = 0.0
        total_prep_extra_put_time = 0.0
        total_execute_overhead_time = 0.0
        total_runner_host_time = 0.0
        total_d2h_time = 0.0
        token_buckets_used: set[int] = set()
        req_buckets_used: set[int] = set()

        cfg = getattr(self.model, "config", None)
        task_type = getattr(self.model, "_task_type", None)
        is_vlm_model = task_type == "image-text-to-text" or (
            cfg is not None
            and (getattr(cfg, "image_token_id", None) is not None or getattr(cfg, "video_token_id", None) is not None)
            and callable(getattr(self.model, "get_image_features", None))
        )
        uses_mrope_model = model_uses_mrope(self.model)

        while start_index < self.sequence_buffer.num_reqs:
            host_start = time.time()
            num_reqs_total = self.sequence_buffer.num_reqs
            scheduled_list: list[int] = []
            req_ids_window = []
            for i in range(start_index, min(num_reqs_total, start_index + self.num_reqs_max_model_len)):
                rid = self.sequence_buffer.req_ids[i]
                req_ids_window.append(rid)
                scheduled_list.append(int(scheduler_output.num_scheduled_tokens.get(rid, 0)) if rid is not None else 0)

            while scheduled_list and scheduled_list[-1] == 0:
                scheduled_list.pop()
                req_ids_window.pop()

            num_reqs = len(scheduled_list)
            if num_reqs == 0:
                break
            end_index = start_index + num_reqs

            total_scheduled = sum(scheduled_list)
            idx = bisect_left(self.num_tokens_paddings, total_scheduled)
            if idx >= len(self.num_tokens_paddings):
                idx = len(self.num_tokens_paddings) - 1
            num_tokens_static = int(self.num_tokens_paddings[idx])

            # Select optimal bucket for current batch size
            # This determines which compiled function to use
            current_bucket = self._get_current_bucket(num_reqs)
            padded_num_reqs = current_bucket  # Use bucket size for compilation lookup

            if num_reqs > 0:
                # Keep scheduled and active_mask as CPU arrays
                scheduled_full_cpu = self._scheduled_full_cpu
                scheduled_full_cpu.fill(0)
                scheduled_full_cpu[: len(scheduled_list)] = scheduled_list

                # Packed view of the per-request target lengths for the current window.
                # Avoid per-step dict lookups; SequenceBuffer keeps this aligned with its ordering.
                req_num_tokens_np = self._req_num_tokens_cpu
                req_num_tokens_np.fill(0)
                req_num_tokens_np[:num_reqs] = self.sequence_buffer.num_tokens[start_index:end_index]

                active_mask_full_cpu = self._active_mask_full_cpu
                active_mask_full_cpu.fill(False)
                for i, rid in enumerate(req_ids_window):
                    if rid is not None:
                        active_mask_full_cpu[i] = True

                self.req_num_tokens_full_buf = jax.device_put(req_num_tokens_np, self._empty_sharding)

            mrope_position_ids_cpu = None
            prefill_embeds_cpu = None
            prefill_embeds_mask_cpu = None
            visual_pos_masks_cpu = None
            deepstack_visual_embeds_cpu = None
            if is_vlm_model:
                # Precompute per-request VLM prompt embeddings outside the compiled step.
                for rid in req_ids_window:
                    if rid is None:
                        continue
                    req_state = self.requests.get(rid)
                    if req_state is None:
                        continue
                    if req_state.has_vision and not req_state.vision_processed:
                        self._precompute_vlm_prefill(req_state)

                (
                    prefill_embeds_cpu,
                    prefill_embeds_mask_cpu,
                    mrope_position_ids_cpu,
                    visual_pos_masks_cpu,
                    deepstack_visual_embeds_cpu,
                ) = self._get_vlm_cpu_buffers(
                    num_tokens_static=num_tokens_static,
                    uses_mrope_model=uses_mrope_model,
                )
                if uses_mrope_model:
                    visual_off = 0

                off = 0
                for req_idx, rid in enumerate(req_ids_window):
                    if rid is None:
                        continue
                    n = int(scheduled_list[req_idx])
                    if n <= 0:
                        continue

                    req_state = self.requests.get(rid)
                    start_tok = int(self.sequence_buffer.num_computed_tokens[req_idx])
                    end_tok = start_tok + n

                    if uses_mrope_model and mrope_position_ids_cpu is not None:
                        # mRoPE position ids: use precomputed prompt indices when available, otherwise
                        # fall back to a constant delta-adjusted 1D position broadcast.
                        if (
                            req_state is not None
                            and req_state.prefill_position_ids is not None
                            and start_tok < req_state.num_prompt_tokens
                        ):
                            prompt_end = min(end_tok, req_state.num_prompt_tokens)
                            prompt_n = int(prompt_end - start_tok)
                            if prompt_n > 0:
                                mrope_position_ids_cpu[:, off : off + prompt_n] = req_state.prefill_position_ids[
                                    :, start_tok:prompt_end
                                ]

                            if prompt_n < n:
                                delta = 0
                                if req_state.prefill_rope_deltas is not None:
                                    delta = int(np.asarray(req_state.prefill_rope_deltas).reshape(-1)[0])
                                idxs = np.arange(start_tok + prompt_n, end_tok, dtype=np.int32) + np.int32(delta)
                                mrope_position_ids_cpu[:, off + prompt_n : off + n] = np.broadcast_to(
                                    idxs[None, :], (3, idxs.shape[0])
                                )
                        else:
                            delta = 0
                            if req_state is not None and req_state.prefill_rope_deltas is not None:
                                delta = int(np.asarray(req_state.prefill_rope_deltas).reshape(-1)[0])
                            idxs = np.arange(start_tok, end_tok, dtype=np.int32) + np.int32(delta)
                            mrope_position_ids_cpu[:, off : off + n] = np.broadcast_to(idxs[None, :], (3, n))

                    # Embedding overrides: use precomputed prompt embeddings when available.
                    if (
                        prefill_embeds_cpu is not None
                        and prefill_embeds_mask_cpu is not None
                        and req_state is not None
                        and req_state.prefill_inputs_embeds is not None
                        and start_tok < req_state.num_prompt_tokens
                    ):
                        prompt_end = min(end_tok, req_state.num_prompt_tokens)
                        prompt_n = int(prompt_end - start_tok)
                        if prompt_n > 0:
                            prefill_embeds_cpu[off : off + prompt_n] = req_state.prefill_inputs_embeds[
                                start_tok:prompt_end
                            ]
                            prefill_embeds_mask_cpu[off : off + prompt_n] = True

                            if visual_pos_masks_cpu is not None and req_state.prefill_visual_pos_masks is not None:
                                mask_slice = req_state.prefill_visual_pos_masks[start_tok:prompt_end]
                                visual_pos_masks_cpu[off : off + prompt_n] = mask_slice

                                num_before = int(req_state.prefill_visual_pos_masks[:start_tok].sum())
                                num_in = int(mask_slice.sum())
                                if (
                                    uses_mrope_model
                                    and num_in
                                    and deepstack_visual_embeds_cpu is not None
                                    and req_state.prefill_deepstack_visual_embeds is not None
                                ):
                                    ds_list = req_state.prefill_deepstack_visual_embeds
                                    for layer_idx, buf in enumerate(deepstack_visual_embeds_cpu):
                                        if layer_idx >= len(ds_list):
                                            break
                                        buf[visual_off : visual_off + num_in] = ds_list[layer_idx][
                                            num_before : num_before + num_in
                                        ]
                                    visual_off += num_in

                    off += n

            # Get page table as CPU array (already on CPU, no transfer needed)
            page_table_cpu = self.sequence_buffer.page_table[0].get_cpu_tensor()
            page_table_version = getattr(self.sequence_buffer.page_table[0], "cpu_version", None)
            total_runner_host_time += time.time() - host_start
            step_start = time.time()
            (
                out_tokens_win,
                valid_mask_win,
                self.input_ids_buf,
                self.position_ids_buf,
                _hidden_states,
                _logits,
                window_metrics,
            ) = self.executor_manager.execute(
                num_tokens=num_tokens_static,
                scheduled_full_cpu=scheduled_full_cpu,
                req_num_tokens_full=self.req_num_tokens_full_buf,
                active_mask_full_cpu=active_mask_full_cpu,
                input_ids_buf=self.input_ids_buf,
                position_ids_buf=self.position_ids_buf,
                padded_num_reqs=padded_num_reqs,
                token_ids_cpu=self.sequence_buffer.token_ids,
                num_computed_tokens_cpu=self.sequence_buffer.num_computed_tokens,
                temperature_cpu=self.sequence_buffer.temperature,
                top_p_cpu=self.sequence_buffer.top_p,
                top_k_cpu=self.sequence_buffer.top_k,
                min_p_cpu=self.sequence_buffer.min_p,
                page_table_cpu=page_table_cpu,
                page_table_version=page_table_version,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            )

            # account for device time (blocking already happened inside execute())
            total_step_time += time.time() - step_start
            num_windows += 1
            total_exec_time += float(window_metrics.get("exec_time", 0.0))
            total_sample_time += float(window_metrics.get("sample_time", 0.0))
            total_prep_time += float(window_metrics.get("prep_time", 0.0))
            total_prep_host_time += float(window_metrics.get("prep_host_time", 0.0))
            total_prep_put_time += float(window_metrics.get("prep_put_time", 0.0))
            total_prep_extra_put_time += float(window_metrics.get("prep_extra_put_time", 0.0))
            total_execute_overhead_time += float(window_metrics.get("execute_overhead_time", 0.0))
            token_buckets_used.add(int(window_metrics.get("token_bucket", num_tokens_static)))
            req_buckets_used.add(int(window_metrics.get("padded_num_reqs", padded_num_reqs)))

            d2h_start = time.time()
            tokens_np = np.asarray(out_tokens_win)
            valid_np = np.asarray(valid_mask_win)
            logits_np = np.asarray(_logits) if self.enable_sampler_metrics and _logits is not None else None
            total_d2h_time += time.time() - d2h_start

            # Track for async scheduling
            request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
            discard_sampled_tokens_req_indices: list[int] = []

            up_wtime = time.time()
            for i, rid in enumerate(req_ids_window):
                if rid is None:
                    continue
                req_ids_all.append(rid)

                if valid_np[i]:
                    tid = int(tokens_np[i])

                    # Get request state and sequence length
                    if rid in self.requests:
                        req_state = self.requests[rid]
                        seq_len = req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens.get(rid, 0)

                        req_idx = self.sequence_buffer.req_id_to_index.get(rid)
                        if req_idx is not None and 0 <= seq_len < self.max_model_len:
                            self.sequence_buffer.token_ids[req_idx, seq_len] = tid

                        # Check if async scheduling is enabled
                        if scheduler_output.async_scheduling:
                            # Async mode: don't append yet, will be done in next iteration
                            request_seq_lens.append((i, req_state, seq_len))
                        else:
                            # Sync mode: append immediately
                            sampled_token_ids_all.append([tid])
                            req_state.output_token_ids.append(tid)
                    else:
                        # No request state, append in sync mode
                        sampled_token_ids_all.append([tid])

                    if self.enable_sampler_metrics and logits_np is not None and i < logits_np.shape[0]:
                        try:
                            token_logprobs[rid] = logits_np[i]
                        except Exception:
                            pass
                else:
                    sampled_token_ids_all.append([])
                    discard_sampled_tokens_req_indices.append(i)

            up_wtime_took = time.time() - up_wtime
            total_post_proc_time += up_wtime_took

            start_index = end_index

        metrics_start = time.time()
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_runner_metrics(
                execution_time=time.time() - execution_start_time,
                batch_size=len(req_ids_all),
                num_tokens=scheduler_output.total_num_scheduled_tokens,
            )
        metrics_time = time.time() - metrics_start

        total_time = time.time() - execution_start_time
        self._perf_iteration += 1

        total_tokens = int(scheduler_output.total_num_scheduled_tokens)
        wall_tps = total_tokens / total_time if total_time > 0 else 0.0
        if self._perf_tps_ema is None:
            self._perf_tps_ema = wall_tps
        else:
            self._perf_tps_ema = self._perf_alpha * wall_tps + (1.0 - self._perf_alpha) * self._perf_tps_ema

        def _fmt_bucket(values: set[int]) -> str:
            if not values:
                return "?"
            if len(values) == 1:
                return str(next(iter(values)))
            vals = sorted(values)
            return f"{vals[0]}-{vals[-1]}"

        num_new = len(scheduler_output.scheduled_new_reqs)
        num_cached = scheduler_output.scheduled_cached_reqs.num_reqs
        num_finished = len(scheduler_output.finished_req_ids)

        step_gap_time = total_step_time - (total_prep_time + total_exec_time + total_sample_time)
        step_gap_time = max(0.0, step_gap_time)

        misc_time = total_time - (
            updating_states_time
            + prev_async_time
            + total_runner_host_time
            + total_d2h_time
            + total_post_proc_time
            + total_prep_time
            + total_exec_time
            + total_sample_time
            + total_execute_overhead_time
            + step_gap_time
            + metrics_time
        )
        misc_time = max(0.0, misc_time)

        prep_detail = ""
        if (total_prep_host_time + total_prep_put_time + total_prep_extra_put_time) > 0:
            prep_detail = (
                f"(host={total_prep_host_time * 1e3:.2f}ms put={total_prep_put_time * 1e3:.2f}ms "
                f"extra={total_prep_extra_put_time * 1e3:.2f}ms) "
            )

        self.log_it(
            f"[perf] it={self._perf_iteration:06d} "
            f"win={num_windows} "
            f"reqs={len(req_ids_all)}(new={num_new},cached={num_cached},fin={num_finished},pad={_fmt_bucket(req_buckets_used)}) "
            f"tok={total_tokens}/b{_fmt_bucket(token_buckets_used)} "
            f"tps={wall_tps:,.0f} ema={self._perf_tps_ema:,.0f} "
            f"runner={total_runner_host_time * 1e3:.2f}ms d2h={total_d2h_time * 1e3:.2f}ms "
            f"prep={total_prep_time * 1e3:.2f}ms {prep_detail}"
            f"fwd={total_exec_time * 1e3:.2f}ms samp={total_sample_time * 1e3:.2f}ms "
            f"ovh={total_execute_overhead_time * 1e3:.2f}ms metrics={metrics_time * 1e3:.2f}ms "
            f"async={prev_async_time * 1e3:.2f}ms "
            f"step={total_step_time * 1e3:.2f}ms gap={step_gap_time * 1e3:.2f}ms "
            f"sync={updating_states_time * 1e3:.2f}ms post={total_post_proc_time * 1e3:.2f}ms misc={misc_time * 1e3:.2f}ms "
            f"total={total_time * 1e3:.2f}ms"
        )

        # Handle async scheduling return
        if scheduler_output.async_scheduling:
            # Set placeholders for current batch
            placeholder_req_id_to_index = self._update_placeholder(
                discard_sampled_tokens_req_indices,
                request_seq_lens,
            )

            # Async copy to host (non-blocking)
            next_tokens_jax = jnp.array(tokens_np, dtype=jnp.int32)
            next_tokens = jax.copy_to_host_async(next_tokens_jax)

            # Store async results for next iteration
            self._pre_async_results = AsyncPreResults(
                req_ids=req_ids_all,
                next_tokens=next_tokens,
                request_seq_lens=request_seq_lens,
                discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                placeholder_req_id_to_index=placeholder_req_id_to_index,
            )

            # Return immediately (non-blocking)
            req_id_to_out_index = {rid: i for i, rid in enumerate(req_ids_all)}
            return ModelRunnerOutput(
                req_ids=req_ids_all,
                req_id_to_index=req_id_to_out_index,
                sampled_token_ids=[],  # Empty, will be filled in next iteration
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={rid: None for rid in req_ids_all},
                finished_sending=None,
                finished_recving=None,
                token_logprobs=token_logprobs or None,
            )

        # Stable mapping for scheduler indexing
        req_id_to_out_index = {rid: i for i, rid in enumerate(req_ids_all)}
        return ModelRunnerOutput(
            req_ids=req_ids_all,
            req_id_to_index=req_id_to_out_index,
            sampled_token_ids=sampled_token_ids_all,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={rid: None for rid in req_ids_all},
            finished_sending=None,
            finished_recving=None,
            token_logprobs=token_logprobs or None,
        )

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        return self._execute_model_impl(scheduler_output)

    def execute_model_async(self, scheduler_output: SchedulerOutput) -> Future[ModelRunnerOutput]:
        """Execute model asynchronously in a background thread.

        This method enables async scheduling by executing the model in a separate
        thread, allowing the caller to continue scheduling the next batch while
        the current batch is being processed.

        The async execution workflow:
            1. Submit model execution to thread pool executor
            2. Return immediately with a Future object
            3. Caller can schedule next batch while this executes
            4. Use wait_for_execution(future) to get results when needed

        Args:
            scheduler_output: Scheduling decisions for this iteration

        Returns:
            Future[ModelRunnerOutput]: Future that will contain the model output
                when execution completes. Can be waited on using wait_for_execution().

        Raises:
            RuntimeError: If async execution is not enabled (executor not initialized)

        Note:
            This method requires async scheduling to be enabled and the executor
            to be initialized. Initialize the executor by calling
            initialize_async_executor() first.

        Example:
            >>> # Initialize async executor first
            >>> runner.initialize_async_executor()
            >>>
            >>> # Execute asynchronously
            >>> future = runner.execute_model_async(scheduler_output)
            >>>
            >>> # Do other work while model executes...
            >>> next_schedule = scheduler.schedule()
            >>>
            >>> # Wait for current execution to finish
            >>> output = runner.wait_for_execution(future)
        """
        if self._executor is None:
            raise RuntimeError(
                "Async execution not enabled. Call initialize_async_executor() first "
                "or check that async_scheduling is enabled in scheduler config."
            )
        return self._executor.submit(self._execute_model_impl, scheduler_output)

    def initialize_async_executor(self) -> None:
        """Initialize the thread pool executor for async model execution.

        This method creates a single-threaded executor that will be used to
        run model execution in the background, enabling async scheduling.

        Side Effects:
            - Creates self._executor as a ThreadPoolExecutor with 1 worker
            - Existing executor is shutdown if present

        Note:
            This should be called before using execute_model_async().
            The executor uses a single worker to maintain execution order.
        """
        if self._executor is not None:
            logger.debug("Shutting down existing executor before reinitializing")
            self._executor.shutdown(wait=True)

        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eSurgeAsync")
        logger.debug("Initialized async executor for model execution")

    def reset_state(self) -> None:
        """Clear sequence state and request bookkeeping.

        Useful when pausing or resetting the runner to ensure no stale pages
        or request metadata linger between sessions.
        """
        self.requests.clear()
        self.sequence_buffer.clear()
        self._pre_async_results = None

    def wait_for_execution(self, future: Future) -> ModelRunnerOutput:
        """Wait for an async execution to complete and return the result.

        Args:
            future: The Future object returned by execute_model_async()

        Returns:
            ModelRunnerOutput: The completed model execution output

        Note:
            This call blocks until the future completes.
        """
        return future.result()

    def shutdown(self) -> None:
        """Cleanup resources including async executor if present."""
        if self._executor is not None:
            logger.debug("Shutting down async executor")
            self._executor.shutdown(wait=True)
            self._executor = None
