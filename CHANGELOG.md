# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **eSurge Complete Inference Engine**: Added comprehensive high-performance inference engine
  - New `eSurge` class with batched generation, streaming, and async support
  - Support for dynamic batching, KV caching, and prefix caching optimizations
  - Comprehensive metrics collection system with real-time monitoring
  - Web dashboard for monitoring inference performance and system status
  - Prometheus metrics integration for production monitoring
  - Function calling support with structured output validation
  - OpenAI-compatible API server implementation

- **eSurge Core Components**:
  - `eSurgeRunner`: JAX-based model execution with optimized batching
  - `Scheduler`: Advanced request scheduling with preemption and priority handling
  - `CacheManager`: Intelligent KV cache management with prefix caching
  - `PagePool`: Memory-efficient page-based cache allocation
  - `MetricsCollector`: Thread-safe metrics collection and aggregation
  - `RequestQueue`: Priority-based request queuing system

- **eSurge Infrastructure**:
  - Complete engine types and output structures
  - Request management with status tracking
  - Sequence buffer for efficient token handling
  - Page table implementation for memory management
  - Comprehensive utilities and helper functions

### Fixed

- **eSurge Metrics Dashboard**: Fixed scheduler and cache metrics showing zero values
  - Scheduler metrics now properly display waiting/running request counts and batch sizes
  - Cache metrics now show actual page utilization and hit rates instead of always showing zero
  - Both metrics are automatically collected during inference operations

- **Code Quality**: Fixed trailing whitespace issues in vWhisper module documentation
  - Removed trailing whitespace from docstring examples in `easydel/inference/vwhisper/__init__.py`
  - All pre-commit linting checks now pass

### Changed

- **eSurge ExecutionManager**: Enhanced compilation strategies with AOT/JIT flexibility
  - Changed `use_aot_forward` parameter default to `True` for better out-of-box performance
  - Added support for both AOT (Ahead-of-Time) and JIT (Just-In-Time) compilation modes
  - AOT mode pre-compiles functions for optimal production performance
  - JIT mode allows dynamic compilation with graph definition as static argument
  - Improved static argument handling based on compilation mode selection

- **Inference Architecture**: Migrated from JAX to NumPy for CPU operations in page management
  - Page table and cache management now use NumPy for better CPU performance
  - Improved memory efficiency and reduced overhead for cache operations

- **Ray Integration**: Pinned Ray version and removed TypeError handling for sharding
  - Fixed TPU setup script to use explicit Python path for UV installation
  - Improved base module `to_state` method with partition rules and sharding support

### Removed

- **Deprecated Components**: Cleaned up old inference implementations
  - Removed outdated `surge_api.py`, `surge_eval.py`, `surge_generation.py` test files
  - Removed obsolete `loss_utils_test.py` from infrastructure

### Technical Details

- **New Files Added**:
  - `easydel/inference/esurge/engine.py`: Main eSurge engine implementation
  - `easydel/inference/esurge/metrics.py`: Comprehensive metrics collection system
  - `easydel/inference/esurge/dashboard.py`: Web-based monitoring dashboard
  - `easydel/inference/esurge/monitoring.py`: Prometheus and console monitoring
  - `easydel/inference/esurge/function_calling_mixin.py`: Function calling support
  - `easydel/inference/esurge/server/`: Complete API server implementation
  - `easydel/inference/function_calling.py`: Function calling utilities
  - `easydel/inference/function_calling_handler.py`: Function call processing
  - Multiple test files for eSurge functionality and performance evaluation

- **Enhanced Modules**:
  - All eSurge core components updated with improved error handling and performance
  - Inference engine interface standardized across all inference backends
  - Sampling parameters and logits processing unified
  - OpenAI API modules enhanced with eSurge integration

- **Infrastructure Improvements**:
  - Base configuration, module, and state classes enhanced
  - Loss utilities optimized and consolidated
  - Generation mixins improved with better protocol support
  - Modeling outputs standardized across all components
  - Utils enhanced with better helper functions and checkpoint management
