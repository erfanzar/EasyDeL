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

Components:
    Config: Main configuration classes
    CacheCoordinator: KV cache coordination and management
    Scheduler: Request scheduling and batching
    eSurgeRunner: Model execution runner
    eSurge: Main engine interface

Example:
    >>> from easydel.inference.esurge import (
    ...     eSurge,
    ...     Config,
    ...     SchedulerConfig,
    ...     CacheConfig
    ... )
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
    >>> engine = eSurge(config=config)

Note:
    eSurge is experimental and APIs may change in future versions.
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
)
from .dashboard import create_dashboard, eSurgeWebDashboard
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
    "SystemMetrics",
    "UnitaryCacheCoordinator",
    "create_dashboard",
    "eSurge",
    "eSurgeApiServer",
    "eSurgeMonitoringServer",
    "eSurgeRunner",
    "eSurgeWebDashboard",
    "get_metrics_collector",
    "initialize_metrics",
    "log_metrics_summary",
    "start_console_monitor",
    "start_monitoring_server",
    "stop_monitoring",
)
