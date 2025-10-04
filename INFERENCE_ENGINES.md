# EasyDeL Inference Stack Overview

EasyDeL ships several inference back-ends so you can choose between raw generation, high-throughput batch serving, and monitored production deployments. This note summarises the major components and how they relate to the broader ecosystem (including external projects like [`ejkernel`](https://github.com/erfanzar/ejkernel)).

## 1. `vinference` (Baseline Inference Engine)
- **Path**: `easydel/inference/vinference`
- **Exports**: `vInference`, `vInferenceApiServer`
- **Purpose**: Core OpenAI-compatible inference pipeline with prompt batching, sampling utilities (`sampling_params.py`), and logits processors.
- **Usage**: Ideal for quick local generation or lightweight API serving.

## 2. `vsurge` (High-Throughput Batch Engine)
- **Path**: `easydel/inference/vsurge`
- **Key modules**: `core/driver.py`, `core/scheduler.py`, `core/engine.py`, `server/api_server.py`, `request_type.py`
- **Highlights**:
  - Request batching with prioritised scheduling (prefill/decoding separation).
  - Supports bytecode decode, interleaved mode, and concurrency controls via `EasyDeLElargeModel.set_vsurge()`.
  - FastAPI server integration (`vSurgeApiServer`) mirrors OpenAI endpoints.
  - Documentation/examples: `docs/vsurge_example.rst`, `docs/vsurge_api_server_example.rst`.
- **When to choose**: Latency-sensitive streaming or large batch deployments where throughput matters more than per-request observability.

## 3. `esurge` (Monitored Inference Engine)
- **Path**: `easydel/inference/esurge`
- **Key modules**: `esurge_engine.py`, `metrics.py`, `dashboard.py`, `server/api_server.py`, `monitoring.py`
- **Highlights**:
  - Drop-in engine (`eSurge`) with metrics instrumentation, Prometheus exporters, rich web dashboard, priority queues, cache controls, and server adapters.
  - Tight integration with `EasyDeLElargeModel` via `set_esurge()` / `build_esurge()`.
  - Examples: `examples/esurge_metrics_example.py`, `examples/esurge_monitoring_demo.py`, documentation in `docs/esurge.rst`.
- **When to choose**: Production scenarios requiring detailed monitoring, multi-tenant scheduling, or observability-first architecture.

## 4. OpenAI Proxies & Mixed Backends
- **Path**: `easydel/inference/oai_proxies.py`
- **Feature**: Proxy that can route requests to EasyDeL engines (`vsurge`, `esurge`) or upstream OpenAI-compatible endpoints.
- **Value**: Simplifies migrations and hybrid deployments.

## 5. Relationship to `ejkernel`
- [`ejkernel`](https://github.com/erfanzar/ejkernel) is an external project maintained by the EasyDeL author providing high-performance fused kernels (e.g., FlashAttention variants, quantised matmuls) for JAX/Flax.
- EasyDeL can consume these kernels to accelerate both training and inference (especially inside `vsurge`/`esurge` engines). Keep an eye on that repository for low-level optimisations that complement the inference stack.

## 6. Choosing an Engine
| Scenario | Recommended Engine | Notes |
| -------- | ------------------ | ----- |
| Quick local experiments | `vinference` | Minimal setup, direct API parity. |
| Throughput-focused serving | `vsurge` | Batch scheduling, concurrency controls, streaming-friendly. |
| Production with deep monitoring | `esurge` | Metrics, dashboards, priority queues, rich observability. |
| Hybrid/external routing | `oai_proxies` | Bridges EasyDeL engines with external services. |

## 7. Next Steps
- Decide which engine matches your deployment constraints and configure it via `EasyDeLElargeModel` helpers (`set_vsurge`, `set_esurge`, etc.).
- Use the provided docs/examples for operational guidance (`docs/vsurge_*`, `docs/esurge.rst`).
- Explore `ejkernel` for kernel-level optimisations if you’re targeting top-end throughput.

_Last updated: 2025-10-03_
