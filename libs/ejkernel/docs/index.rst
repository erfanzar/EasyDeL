ejkernel 🔮
==========

Overview
--------

ejKernel is a production-grade kernel library for JAX that provides highly optimized implementations of deep learning operations with automatic multi-backend support. The library features a sophisticated configuration management system with autotuning, comprehensive type safety, and seamless execution across GPUs, TPUs, and CPUs.

Key Findings
------------

Architectural Strengths
~~~~~~~~~~~~~~~~~~~~~~~

✅ **Layered Architecture**: Clean separation between user API, operations, execution, and implementations

✅ **Multi-Backend Support**: Seamless support for GPU (Triton), TPU (Pallas), and CPU (XLA)

✅ **Automatic Optimization**: Sophisticated autotuning with multi-tier configuration management

✅ **Type Safety**: Comprehensive type annotations with runtime validation

✅ **Performance**: State-of-the-art implementations with custom gradients

Supported Operations
~~~~~~~~~~~~~~~~~~~~

**Attention Mechanisms**

- Flash Attention v2 (memory-efficient exact attention)
- Ring Attention (distributed sequence parallelism)
- Page Attention (KV-cache optimized inference)
- Block Sparse Attention (configurable sparse patterns)
- Gated Linear Attention (GLA)
- Lightning Attention (layer-dependent decay)
- Multi-head Latent Attention (MLA)
- Ragged Page Attention v2/v3 (variable-length paged attention)
- Ragged Decode Attention (variable-length decoding)
- Kernel Delta Attention (delta-rule linear attention)
- Unified Attention (vLLM-style paged attention)
- Prefill Page Attention (prefill phase handling)

**State Space Models**

- State Space v1 (Mamba1-style SSM with 2D A matrix)
- State Space v2 (Mamba2-style SSM with per-head scalar A)

**Other Operations**

- Grouped MatMul (efficient batched matrix operations for MoE)
- Mean Pooling (variable-length sequence aggregation)
- Recurrent (optimized RNN/LSTM/GRU operations)
- Native Sparse (block-sparse matrix computations)

Design Patterns Identified
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Registry Pattern** for kernel discovery and routing
- **Strategy Pattern** for configuration selection
- **Chain of Responsibility** for fallback mechanisms
- **Factory Pattern** for kernel specialization
- **Template Method** for platform-specific customization

Innovation Highlights
~~~~~~~~~~~~~~~~~~~~~

🚀 **7-Tier Configuration Selection**: Override → Overlay → Cache → Persistent → Autotune → Heuristics → Error

🚀 **Device-Aware Caching**: Fingerprint-based optimal configuration storage

🚀 **Platform-Agnostic Registry**: Automatic backend selection with priorities

🚀 **Custom VJP Integration**: Memory-efficient gradient computation with O(N) complexity

🚀 **Type-Safe Configurations**: Dataclass-based configs with auto-conversion

Project Statistics
~~~~~~~~~~~~~~~~~~

- **Supported Operations**: 20+ attention mechanisms, SSMs, and utilities
- **Backend Implementations**: 4 (Triton, Pallas, XLA, CUDA)
- **Test Coverage**: Comprehensive unit, integration, and performance tests
- **Type Coverage**: 100% of public APIs with jaxtyping annotations
- **Platform Support**: GPU (NVIDIA/AMD), TPU, CPU

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from ejkernel.modules import flash_attention

   # Basic usage - automatic configuration selection
   output = flash_attention(
       query, key, value,
       causal=True,
       dropout_prob=0.1
   )

   # With advanced features
   output = flash_attention(
       query, key, value,
       causal=True,
       sliding_window=128,        # Local attention window
       logits_soft_cap=30.0,      # Gemma-2 style soft capping
   )

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Basic installation
   pip install ejkernel

   # GPU support (CUDA/ROCm)
   pip install ejkernel[gpu]

   # TPU support
   pip install ejkernel[tpu]

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   install

.. toctree::
   :maxdepth: 2
   :caption: Architecture & Design:

   project_overview
   kernel_registry_system
   ops_system_architecture
   maskinfo_guide
   kernel_implementations
   module_operations
   utilities_and_helpers
   test_suite_and_examples
   comprehensive_architecture_report

.. toctree::
   :maxdepth: 2
   :caption: User Guides:

   api/index
   api/modules
   api/types
   api/ops

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api_docs/index
