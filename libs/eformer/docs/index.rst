eformer 🔮
==========

eformer is a powerful and flexible JAX-based package designed to accelerate and simplify machine learning and deep learning workflows. It provides a comprehensive suite of tools and utilities for efficient model development, training, and deployment.

**eformer** (EasyDel Former) is a utility library designed to simplify and enhance the development of machine learning models using JAX. It provides a collection of tools for sharding, custom PyTrees, quantization, and optimized operations, making it easier to build and scale models efficiently.

Features
--------

- **Sharding Utilities (`escale`)**: Tools for efficient sharding and distributed computation in JAX.
- **Custom PyTrees (`jaximus`)**: Enhanced utilities for creating custom PyTrees and `ArrayValue` objects, updated from Equinox.
- **Custom Calling (`callib`)**: A tool for custom function calls and direct integration with Triton kernels in JAX.
- **Optimizer Factory**: A flexible factory for creating and configuring optimizers like AdamW, Adafactor, Lion, and RMSProp.
- **Custom Operations and Kernels**:
  - Flash Attention 2 for GPUs/TPUs (via Triton and Pallas).
  - 8-bit and NF4 quantization for efficient model.
- **Quantization Support**: Tools for 8-bit and NF4 quantization, enabling memory-efficient model deployment.

Installation
------------

You can install `eformer` via pip:

```bash
pip install eformer
```

.. _eformer:

Zare Chavoshi, Erfan. "eformer is a collection of functions and utilities that can help with various tasks when using Flax and JAX.""


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: APIs

    api_docs/apis


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Getting Started

    contributing