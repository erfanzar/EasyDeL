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

"""eSurge: Experimental high-performance inference engine.

eSurge is an experimental inference engine designed for efficient
batched text generation with advanced caching and scheduling capabilities.
It provides fine-grained control over memory management and request scheduling.

Key Features:
    - Advanced KV cache management with page-based allocation
    - Multiple attention pattern support (full, sliding window, chunked)
    - Flexible request scheduling (FCFS, priority-based)
    - Prefix caching for improved efficiency
    - Continuous batching support
    - Vision-language model support with multimodal processing
    - Prometheus metrics integration for monitoring
    - Rich console monitoring with real-time metrics display

Components:
    Config: Main configuration classes for scheduler and cache
    CacheCoordinator: KV cache coordination and management
    Scheduler: Request scheduling and batching
    eSurgeRunner: Model execution runner
    eSurge: Main engine interface

Classes:
    eSurge: High-level engine interface for text generation.
    Config: Unified configuration combining scheduler and cache settings.
    SchedulerConfig: Configuration for request scheduling behavior.
    CacheConfig: Configuration for KV cache management.
    RequestOutput: Container for generation results and metrics.
    CompletionOutput: Individual completion within a request.
    EngineRequest: Internal request tracking object.
    EngineRequestStatus: Enumeration of request lifecycle states.
    Scheduler: Request scheduler with batching support.
    SchedulerOutput: Output from scheduler decisions.
    eSurgeRunner: Model execution and forward pass runner.
    CacheCoordinator: KV cache allocation and management.
    MetricsCollector: Centralized metrics collection system.

Example:
    >>> from easydel.inference.esurge import (
    ...     eSurge,
    ...     Config,
    ...     SchedulerConfig,
    ...     CacheConfig
    ... )
    >>>
    >>> # Create configuration
    >>> config = Config(
    ...     scheduler_config=SchedulerConfig(
    ...         max_num_seqs=16,
    ...         max_num_batched_tokens=2048,
    ...         max_model_len=8192
    ...     ),
    ...     cache_config=CacheConfig(
    ...         num_pages=1000,
    ...         page_size=16,
    ...         enable_prefix_caching=True
    ...     )
    ... )
    >>>
    >>> # Initialize and use the engine
    >>> engine = eSurge(model="model-name", max_model_len=8192)
    >>> engine.initiate()
    >>>
    >>> # Generate text with streaming
    >>> for output in engine.stream("Tell me about AI"):
    ...     print(output.delta_text, end="", flush=True)

Note:
    eSurge is experimental and APIs may change in future versions.
    For production use, ensure thorough testing and monitoring.
"""

from .config import CacheConfig, Config, SchedulerConfig
from .core import (
    AttentionSpec,
    CacheCoordinator,
    CacheCoordinatorNoPrefixCache,
    CacheGroupsConfig,
    CacheGroupSpec,
    CacheManager,
    CachePages,
    CacheSpec,
    ChunkedLocalAttentionManager,
    ChunkedLocalAttentionSpec,
    FullAttentionManager,
    FullAttentionSpec,
    HybridCacheCoordinator,
    MambaManager,
    MambaSpec,
    PagePool,
    SingleTypeCacheManager,
    SlidingWindowManager,
    SlidingWindowSpec,
    UnitaryCacheCoordinator,
    create_kv_cache_specs_from_config,
)
from .distributed import (
    DistributedController,
    StepDispatch,
    compute_sampled_digest,
    make_config_fingerprint,
    resolve_distributed_role,
    resolve_service_hosts,
)
from .esurge_engine import CompletionOutput, RequestOutput, eSurge
from .metrics import (
    CacheMetrics,
    MetricsCollector,
    ModelRunnerMetrics,
    RequestMetrics,
    SchedulerMetrics,
    SystemMetrics,
    get_metrics_collector,
    initialize_metrics,
    log_metrics_summary,
)
from .monitoring import (
    PrometheusMetrics,
    RichConsoleMonitor,
    eSurgeMonitoringServer,
    start_console_monitor,
    start_monitoring_server,
    stop_monitoring,
)
from .multimodal import MultiModalManager, VisionEncoderCache
from .request import EngineRequest, EngineRequestStatus
from .runners import SequenceBuffer, eSurgeRunner
from .scheduler import (
    CachedRequestData,
    FCFSRequestQueue,
    NewRequestData,
    PriorityRequestQueue,
    RequestQueue,
    Scheduler,
    SchedulerInterface,
    SchedulerOutput,
)
from .server import eSurgeApiServer

__all__ = (
    "AttentionSpec",
    "CacheConfig",
    "CacheCoordinator",
    "CacheCoordinatorNoPrefixCache",
    "CacheGroupSpec",
    "CacheGroupsConfig",
    "CacheManager",
    "CacheMetrics",
    "CachePages",
    "CacheSpec",
    "CachedRequestData",
    "ChunkedLocalAttentionManager",
    "ChunkedLocalAttentionSpec",
    "CompletionOutput",
    "Config",
    "DistributedController",
    "EngineRequest",
    "EngineRequestStatus",
    "FCFSRequestQueue",
    "FullAttentionManager",
    "FullAttentionSpec",
    "HybridCacheCoordinator",
    "MambaManager",
    "MambaSpec",
    "MetricsCollector",
    "ModelRunnerMetrics",
    "MultiModalManager",
    "NewRequestData",
    "PagePool",
    "PriorityRequestQueue",
    "PrometheusMetrics",
    "RequestMetrics",
    "RequestOutput",
    "RequestQueue",
    "RichConsoleMonitor",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerInterface",
    "SchedulerMetrics",
    "SchedulerOutput",
    "SequenceBuffer",
    "SingleTypeCacheManager",
    "SlidingWindowManager",
    "SlidingWindowSpec",
    "StepDispatch",
    "SystemMetrics",
    "UnitaryCacheCoordinator",
    "VisionEncoderCache",
    "compute_sampled_digest",
    "create_kv_cache_specs_from_config",
    "eSurge",
    "eSurgeApiServer",
    "eSurgeMonitoringServer",
    "eSurgeRunner",
    "get_metrics_collector",
    "initialize_metrics",
    "log_metrics_summary",
    "make_config_fingerprint",
    "resolve_distributed_role",
    "resolve_service_hosts",
    "start_console_monitor",
    "start_monitoring_server",
    "stop_monitoring",
)
