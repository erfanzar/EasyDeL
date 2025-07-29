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

import hashlib
from dataclasses import field
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, SkipValidation, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from easydel.utils.helpers import get_logger

logger = get_logger(__name__)

# Type aliases for better readability
PreemptionMode = Literal["swap", "recompute"]
SchedulingPolicy = Literal["fcfs", "priority"]


@dataclass
class SchedulerConfig:
    """
    Configuration for the EasyDeL scheduler that manages request batching and execution.

    This class defines parameters that control how the scheduler batches requests,
    manages memory, and handles prefill operations. The configuration is validated
    to ensure consistency and optimal performance.

    Attributes:
        max_num_batched_tokens: Maximum number of tokens that can be processed in a single batch.
                               If None, will be computed based on other parameters.
        max_num_seqs: Maximum number of sequences that can be processed concurrently.
                     If None, defaults to 128.
        max_model_len: Maximum sequence length supported by the model.
                      If None, defaults to 8192.
        max_num_partial_prefills: Maximum number of partial prefill operations that can run concurrently.
        max_long_partial_prefills: Maximum number of long partial prefill operations.
        long_prefill_token_threshold: Token count threshold to classify a prefill as "long".
        num_lookahead_slots: Number of lookahead slots for speculative execution.
        cuda_graph_sizes: List of CUDA graph sizes for optimization.
        delay_factor: Factor for introducing delays in scheduling (0.0 = no delay).
        enable_chunked_prefill: Whether to enable chunked prefill processing.
        is_multimodal_model: Whether this is a multimodal model requiring special handling.
        preemption_mode: Strategy for handling memory pressure ("swap" or "recompute").
        num_scheduler_steps: Number of steps the scheduler takes per iteration.
        multi_step_stream_outputs: Whether to stream outputs in multi-step mode.
        send_delta_data: Whether to send only delta data in responses.
        policy: Scheduling policy for request ordering.
        disable_hybrid_kv_cache_manager: Whether to disable hybrid KV cache management.
    """

    # Core capacity parameters
    max_num_batched_tokens: SkipValidation[int | None] = Field(
        default=None, description="Maximum tokens per batch. Auto-computed if None."
    )
    max_num_seqs: SkipValidation[int | None] = Field(
        default=None, description="Maximum concurrent sequences. Defaults to 128."
    )
    max_model_len: SkipValidation[int | None] = Field(
        default=None, description="Maximum sequence length. Defaults to 8192."
    )

    # Prefill configuration
    max_num_partial_prefills: int = Field(default=1, ge=1, description="Maximum concurrent partial prefills.")
    max_long_partial_prefills: int = Field(default=1, ge=1, description="Maximum concurrent long partial prefills.")
    long_prefill_token_threshold: int = Field(
        default=0, ge=0, description="Token threshold for classifying long prefills."
    )

    # Performance optimization
    num_lookahead_slots: int = Field(default=0, ge=0, description="Lookahead slots for speculative execution.")
    cuda_graph_sizes: list[int] = Field(default_factory=list, description="CUDA graph sizes for optimization.")
    delay_factor: float = Field(default=0.0, ge=0.0, description="Scheduling delay factor.")

    # Feature flags
    enable_chunked_prefill: SkipValidation[bool | None] = Field(
        default=None, description="Enable chunked prefill processing."
    )
    is_multimodal_model: bool = Field(default=False, description="Whether this is a multimodal model.")
    disable_hybrid_kv_cache_manager: bool = Field(default=False, description="Disable hybrid KV cache management.")

    # Scheduling behavior
    preemption_mode: PreemptionMode | None = Field(default=None, description="Memory pressure handling strategy.")
    num_scheduler_steps: int = Field(default=1, ge=1, description="Scheduler steps per iteration.")
    multi_step_stream_outputs: bool = Field(default=True, description="Stream outputs in multi-step mode.")
    send_delta_data: bool = Field(default=False, description="Send only delta data in responses.")
    policy: SchedulingPolicy = Field(default="fcfs", description="Request scheduling policy.")

    # Computed fields (set during post_init)
    max_num_encoder_input_tokens: int = field(init=False, repr=False)
    encoder_cache_size: int = field(init=False, repr=False)
    chunked_prefill_enabled: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Initialize computed fields and apply default values.

        This method:
        1. Sets default values for None fields
        2. Computes derived parameters
        3. Logs important configuration decisions
        4. Validates parameter consistency
        """
        self._set_default_values()
        self._compute_batch_token_limits()
        self._setup_encoder_parameters()
        self._configure_chunked_prefill()
        self._configure_partial_prefills()
        self._setup_cuda_graphs()

    def _set_default_values(self) -> None:
        """Set default values for optional parameters."""
        if self.max_model_len is None:
            self.max_model_len = 8192
            logger.debug("Set default max_model_len to %d", self.max_model_len)

        if self.max_num_seqs is None:
            self.max_num_seqs = 128
            logger.debug("Set default max_num_seqs to %d", self.max_num_seqs)

    def _compute_batch_token_limits(self) -> None:
        """Compute max_num_batched_tokens based on configuration."""
        if self.max_num_batched_tokens is not None:
            return

        # Base computation
        if self.enable_chunked_prefill:
            if self.num_scheduler_steps > 1:
                base_tokens = max(self.max_model_len, 2048)
            else:
                base_tokens = 2048
        else:
            base_tokens = max(self.max_model_len, 2048)

        # Multimodal adjustment
        if self.is_multimodal_model:
            base_tokens = max(base_tokens, 5120)
            logger.debug("Increased batch tokens for multimodal model to %d", base_tokens)

        # Apply sequence limit
        sequence_limit = self.max_num_seqs * self.max_model_len
        self.max_num_batched_tokens = min(sequence_limit, base_tokens)

        logger.info(
            "Computed max_num_batched_tokens: %d (base: %d, sequence_limit: %d)",
            self.max_num_batched_tokens,
            base_tokens,
            sequence_limit,
        )

    def _setup_encoder_parameters(self) -> None:
        """Set up encoder-related parameters."""
        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

    def _configure_chunked_prefill(self) -> None:
        """Configure chunked prefill settings."""
        self.chunked_prefill_enabled = bool(self.enable_chunked_prefill)

        if self.chunked_prefill_enabled:
            logger.info("Chunked prefill enabled with max_num_batched_tokens=%d", self.max_num_batched_tokens)

    def _configure_partial_prefills(self) -> None:
        """Configure partial prefill settings."""
        if self.max_num_partial_prefills <= 1:
            return

        # Auto-compute threshold if not set
        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

        logger.info(
            "Concurrent partial prefills enabled: "
            "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
            "long_prefill_token_threshold=%d",
            self.max_num_partial_prefills,
            self.max_long_partial_prefills,
            self.long_prefill_token_threshold,
        )

    def _setup_cuda_graphs(self) -> None:
        """Set up default CUDA graph sizes if not provided."""
        if not self.cuda_graph_sizes:
            default_size = min(self.max_num_seqs * 2, 512)
            self.cuda_graph_sizes = [default_size]
            logger.debug("Set default CUDA graph size to %d", default_size)

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        """
        Validate configuration parameters for consistency and correctness.

        Returns:
            Self: The validated configuration instance.

        Raises:
            ValueError: If any validation check fails.
        """
        self._validate_batch_token_constraints()
        self._validate_sequence_constraints()
        self._validate_lookahead_slots()
        self._validate_scheduler_steps()
        self._validate_partial_prefill_constraints()

        return self

    def _validate_batch_token_constraints(self) -> None:
        """Validate batch token constraints."""
        if self.max_num_batched_tokens < self.max_model_len and not self.chunked_prefill_enabled:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes the system reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= max_num_seqs ({self.max_num_seqs})."
            )

    def _validate_sequence_constraints(self) -> None:
        """Validate sequence-related constraints."""
        max_theoretical_tokens = self.max_num_seqs * self.max_model_len

        if self.max_num_batched_tokens > max_theoretical_tokens:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs * max_model_len (%d). "
                "This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                max_theoretical_tokens,
            )

    def _validate_lookahead_slots(self) -> None:
        """Validate lookahead slot configuration."""
        if self.num_lookahead_slots < 0:
            raise ValueError(f"num_lookahead_slots ({self.num_lookahead_slots}) must be >= 0.")

    def _validate_scheduler_steps(self) -> None:
        """Validate scheduler step configuration."""
        if self.num_scheduler_steps < 1:
            raise ValueError(f"num_scheduler_steps ({self.num_scheduler_steps}) must be >= 1.")

    def _validate_partial_prefill_constraints(self) -> None:
        """Validate partial prefill configuration."""
        if self.max_num_partial_prefills < 1:
            raise ValueError(f"max_num_partial_prefills ({self.max_num_partial_prefills}) must be >= 1.")

        if self.max_num_partial_prefills > 1:
            if not self.chunked_prefill_enabled:
                raise ValueError("Chunked prefill must be enabled to set max_num_partial_prefills > 1.")

            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    f"long_prefill_token_threshold ({self.long_prefill_token_threshold}) "
                    f"cannot exceed max_model_len ({self.max_model_len})."
                )

        if not (1 <= self.max_long_partial_prefills <= self.max_num_partial_prefills):
            raise ValueError(
                f"max_long_partial_prefills ({self.max_long_partial_prefills}) "
                f"must be between 1 and max_num_partial_prefills ({self.max_num_partial_prefills})."
            )

    @cached_property
    def is_multi_step(self) -> bool:
        """
        Check if multi-step scheduling is enabled.

        Returns:
            bool: True if num_scheduler_steps > 1.
        """
        return self.num_scheduler_steps > 1

    @cached_property
    def effective_batch_size(self) -> int:
        """
        Calculate the effective batch size considering all constraints.

        Returns:
            int: The effective maximum batch size.
        """
        return min(self.max_num_batched_tokens, self.max_num_seqs * self.max_model_len)

    @cached_property
    def memory_efficiency_ratio(self) -> float:
        """
        Calculate memory efficiency ratio.

        Returns:
            float: Ratio of effective batch size to theoretical maximum.
        """
        theoretical_max = self.max_num_seqs * self.max_model_len
        return self.effective_batch_size / theoretical_max if theoretical_max > 0 else 0.0

    def compute_hash(self) -> str:
        """
        Compute a hash of the configuration for caching and comparison.

        Returns:
            str: MD5 hash of the configuration parameters.
        """
        # Include all significant parameters that affect behavior
        hash_factors = [
            self.max_num_batched_tokens,
            self.max_num_seqs,
            self.max_model_len,
            self.max_num_partial_prefills,
            self.max_long_partial_prefills,
            self.long_prefill_token_threshold,
            self.num_lookahead_slots,
            tuple(self.cuda_graph_sizes),  # Convert list to tuple for hashing
            self.delay_factor,
            self.enable_chunked_prefill,
            self.is_multimodal_model,
            self.preemption_mode,
            self.num_scheduler_steps,
            self.multi_step_stream_outputs,
            self.send_delta_data,
            self.policy,
            self.disable_hybrid_kv_cache_manager,
        ]

        hash_str = hashlib.md5(str(hash_factors).encode(), usedforsecurity=False).hexdigest()

        return hash_str

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the configuration for logging and debugging.

        Returns:
            dict: Summary of key configuration parameters and computed values.
        """
        return {
            "capacity": {
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_num_seqs": self.max_num_seqs,
                "max_model_len": self.max_model_len,
                "effective_batch_size": self.effective_batch_size,
                "memory_efficiency_ratio": f"{self.memory_efficiency_ratio:.2%}",
            },
            "prefill": {
                "chunked_prefill_enabled": self.chunked_prefill_enabled,
                "max_num_partial_prefills": self.max_num_partial_prefills,
                "max_long_partial_prefills": self.max_long_partial_prefills,
                "long_prefill_token_threshold": self.long_prefill_token_threshold,
            },
            "scheduling": {
                "policy": self.policy,
                "num_scheduler_steps": self.num_scheduler_steps,
                "is_multi_step": self.is_multi_step,
                "preemption_mode": self.preemption_mode,
            },
            "optimization": {
                "num_lookahead_slots": self.num_lookahead_slots,
                "cuda_graph_sizes": self.cuda_graph_sizes,
                "delay_factor": self.delay_factor,
            },
            "features": {
                "is_multimodal_model": self.is_multimodal_model,
                "multi_step_stream_outputs": self.multi_step_stream_outputs,
                "send_delta_data": self.send_delta_data,
                "disable_hybrid_kv_cache_manager": self.disable_hybrid_kv_cache_manager,
            },
        }

    def validate_compatibility(self, other: "SchedulerConfig") -> list[str]:
        """
        Check compatibility with another scheduler configuration.

        Args:
            other: Another SchedulerConfig to compare against.

        Returns:
            list[str]: List of compatibility issues (empty if compatible).
        """
        issues = []

        # Check critical parameters that must match
        critical_params = [
            ("max_model_len", "Maximum model length"),
            ("is_multimodal_model", "Multimodal model flag"),
            ("chunked_prefill_enabled", "Chunked prefill setting"),
        ]

        for param, description in critical_params:
            if getattr(self, param) != getattr(other, param):
                issues.append(f"{description} mismatch: {getattr(self, param)} vs {getattr(other, param)}")

        # Check capacity constraints
        if self.max_num_batched_tokens != other.max_num_batched_tokens:
            issues.append(f"Batch token limit mismatch: {self.max_num_batched_tokens} vs {other.max_num_batched_tokens}")

        if self.max_num_seqs != other.max_num_seqs:
            issues.append(f"Sequence limit mismatch: {self.max_num_seqs} vs {other.max_num_seqs}")

        return issues

    def optimize_for_hardware(
        self, available_memory_gb: float, num_gpus: int = 1, gpu_memory_utilization: float = 0.9
    ) -> "SchedulerConfig":
        """
        Create an optimized configuration for specific hardware constraints.

        Args:
            available_memory_gb: Available GPU memory in GB.
            num_gpus: Number of GPUs available.
            gpu_memory_utilization: Target GPU memory utilization ratio.

        Returns:
            SchedulerConfig: New optimized configuration.
        """
        bytes_per_token = 16
        available_bytes = available_memory_gb * 1024**3 * gpu_memory_utilization

        optimal_batch_tokens = int(available_bytes // bytes_per_token)

        # Create new config with optimized values
        new_config = SchedulerConfig(
            max_num_batched_tokens=min(optimal_batch_tokens, self.max_num_batched_tokens or optimal_batch_tokens),
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_partial_prefills=min(num_gpus * 2, self.max_num_partial_prefills),
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            num_lookahead_slots=self.num_lookahead_slots,
            cuda_graph_sizes=self.cuda_graph_sizes.copy(),
            delay_factor=self.delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            is_multimodal_model=self.is_multimodal_model,
            preemption_mode=self.preemption_mode,
            num_scheduler_steps=self.num_scheduler_steps,
            multi_step_stream_outputs=self.multi_step_stream_outputs,
            send_delta_data=self.send_delta_data,
            policy=self.policy,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
        )

        logger.info(
            "Optimized config for %.1fGB memory: batch_tokens=%d, estimated_utilization=%.1f%%",
            available_memory_gb,
            new_config.max_num_batched_tokens,
            (new_config.max_num_batched_tokens * bytes_per_token / available_bytes) * 100,
        )

        return new_config

    def copy(self, **overrides) -> "SchedulerConfig":
        """
        Create a copy of the configuration with optional parameter overrides.

        Args:
            **overrides: Parameters to override in the new configuration.

        Returns:
            SchedulerConfig: New configuration instance.
        """
        # Get current values
        current_values = {
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "max_num_partial_prefills": self.max_num_partial_prefills,
            "max_long_partial_prefills": self.max_long_partial_prefills,
            "long_prefill_token_threshold": self.long_prefill_token_threshold,
            "num_lookahead_slots": self.num_lookahead_slots,
            "cuda_graph_sizes": self.cuda_graph_sizes.copy(),
            "delay_factor": self.delay_factor,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "is_multimodal_model": self.is_multimodal_model,
            "preemption_mode": self.preemption_mode,
            "num_scheduler_steps": self.num_scheduler_steps,
            "multi_step_stream_outputs": self.multi_step_stream_outputs,
            "send_delta_data": self.send_delta_data,
            "policy": self.policy,
            "disable_hybrid_kv_cache_manager": self.disable_hybrid_kv_cache_manager,
        }

        # Apply overrides
        current_values.update(overrides)

        return SchedulerConfig(**current_values)

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"SchedulerConfig("
            f"batch_tokens={self.max_num_batched_tokens}, "
            f"max_seqs={self.max_num_seqs}, "
            f"max_len={self.max_model_len}, "
            f"chunked_prefill={self.chunked_prefill_enabled}, "
            f"multi_step={self.is_multi_step})"
        )
