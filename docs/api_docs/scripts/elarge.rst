easydel.scripts.elarge
======================

Unified YAML Runner for eLargeModel
-----------------------------------

The ``elarge`` script provides a command-line interface for running eLargeModel
pipelines defined in YAML configuration files.

Usage
^^^^^

.. code-block:: bash

   # Basic usage
   python -m easydel.scripts.elarge config.yaml

   # With --config flag
   python -m easydel.scripts.elarge --config config.yaml

   # Dry run (parse and print without executing)
   python -m easydel.scripts.elarge config.yaml --dry-run

YAML Structure
^^^^^^^^^^^^^^

Configuration files have two main sections:

1. **Configuration**: Model, sharding, data, and trainer settings
2. **Actions**: Sequential operations to execute

.. code-block:: yaml

   model:
     name_or_path: "meta-llama/Llama-2-7b-hf"

   loader:
     dtype: bf16

   sharding:
     axis_dims: [1, -1, 1, 1, 1]

   actions:
     - validate
     - train

Available Actions
^^^^^^^^^^^^^^^^^

- ``validate``: Validate configuration
- ``train``: Run training
- ``eval``: Run evaluation with lm-evaluation-harness
- ``serve`` / ``server``: Start OpenAI-compatible API server
- ``print`` / ``show``: Print eLargeModel summary
- ``dump_config`` / ``config``: Print normalized configuration
- ``to_json`` / ``save_json``: Save config to JSON
- ``to_yaml`` / ``save_yaml``: Save config to YAML

Example: Training
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   model:
     name_or_path: meta-llama/Llama-2-7b-hf

   loader:
     dtype: bf16

   sharding:
     axis_dims: [1, -1, 1, 1, 1]

   mixture:
     informs:
       - type: json
         data_files: "train/*.json"
         content_field: text
     batch_size: 32

   trainer:
     trainer_type: sft
     learning_rate: 2.0e-5
     num_train_epochs: 3
     save_directory: ./output

   actions:
     - validate
     - train

Example: Evaluation
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   model:
     name_or_path: meta-llama/Llama-2-7b-hf

   loader:
     dtype: bf16

   esurge:
     max_model_len: 4096
     max_num_seqs: 64

   actions:
     - validate
     - eval:
         tasks: [hellaswag, mmlu]
         engine: esurge
         num_fewshot: 5
         output_path: ./results.json

Example: API Server
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   model:
     name_or_path: meta-llama/Llama-2-7b-chat-hf

   loader:
     dtype: bf16

   esurge:
     max_model_len: 4096
     max_num_seqs: 256

   actions:
     - validate
     - serve:
         host: 0.0.0.0
         port: 8000
         enable_function_calling: true

API Reference
^^^^^^^^^^^^^

.. automodule:: easydel.scripts.elarge
   :members:
   :undoc-members:
   :show-inheritance:

See Also
^^^^^^^^

- :doc:`../../infra/elarge_model` - Complete eLargeModel guide
- :doc:`../../esurge` - eSurge inference engine documentation
