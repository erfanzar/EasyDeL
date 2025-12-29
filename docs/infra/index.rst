Infrastructure Guide
====================

The EasyDeL infrastructure module (``easydel.infra``) provides the foundational classes and utilities
that all EasyDeL models are built upon. This guide covers how the infrastructure works, how to use it,
and how to customize and extend it for your own needs.

.. toctree::
   :maxdepth: 2
   :caption: Infrastructure Documentation:

   overview.md
   base_config.md
   base_module.md
   customization.md
   adding_models.md
   elarge_model.md


Quick Start
-----------

The infrastructure module provides three core components:

1. **EasyDeLBaseConfig** - Configuration management for models
2. **EasyDeLBaseModule** - Base class for all neural network modules
3. **EasyDeLState** - Training state management

.. code-block:: python

   from easydel import (
       EasyDeLBaseConfig,
       EasyDeLBaseModule,
       EasyDeLState,
   )

   # Load a pretrained model
   import easydel as ed

   model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-7b-hf",
       dtype=jnp.bfloat16,
       auto_shard_model=True,
   )

   # Create training state
   state = model.to_state(optimizer=optax.adamw(1e-4))


Key Concepts
------------

Sharding and Distribution (5D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EasyDeL uses 5D sharding with axes: ``(dp, fsdp, ep, tp, sp)``

- **dp**: Data Parallelism (batch splitting)
- **fsdp**: Fully Sharded Data Parallelism
- **ep**: Expert Parallelism (for MoE models)
- **tp**: Tensor Parallelism (layer sharding)
- **sp**: Sequence Parallelism

.. code-block:: python

   config = LlamaConfig.from_pretrained(
       "meta-llama/Llama-2-7b-hf",
       sharding_axis_dims=(1, -1, 1, 1, 1),  # Full FSDP
   )

   # For MoE models with expert parallelism
   config = DeepseekConfig.from_pretrained(
       "deepseek/DeepSeek-V3",
       sharding_axis_dims=(1, 4, 8, 1, 1),  # FSDP + EP
   )

See :doc:`base_config` for details on sharding configuration.


Model Customization
^^^^^^^^^^^^^^^^^^^

All components are designed to be customizable:

.. code-block:: python

   class MyCustomConfig(EasyDeLBaseConfig):
       def get_partition_rules(self, fully_sharded=True):
           # Custom sharding rules
           return (...)

See :doc:`customization` for detailed customization guides.


Adding New Models
^^^^^^^^^^^^^^^^^

EasyDeL makes it straightforward to add new model architectures:

.. code-block:: python

   @register_module("causal-lm", MyModelConfig)
   class MyModelForCausalLM(EasyDeLBaseModule):
       # Your model implementation
       pass

See :doc:`adding_models` for a complete guide.


High-Level Training API
^^^^^^^^^^^^^^^^^^^^^^^

The ``eLargeModel`` class provides a fluent API for training:

.. code-block:: python

   from easydel.infra import eLargeModel

   elm = (
       eLargeModel
       .from_pretrained("meta-llama/Llama-2-7b-hf")
       .set_dtype(dtype="bf16")
       .set_sharding(axis_dims=(1, -1, 1, 1, 1))  # 5D sharding
       .add_dataset(dataset_name="your-dataset")
       .set_trainer(trainer_type="sft", learning_rate=2e-5)
   )

   elm.train()

See :doc:`elarge_model` for the complete eLargeModel guide.


API Reference
-------------

For detailed API documentation, see:

- :doc:`../api_docs/infra/index` - Complete API reference
- :doc:`../api_docs/infra/base_config` - EasyDeLBaseConfig API
- :doc:`../api_docs/infra/base_module` - EasyDeLBaseModule API
- :doc:`../api_docs/infra/base_state` - EasyDeLState API
- :doc:`../api_docs/infra/elarge_model/index` - eLargeModel API
