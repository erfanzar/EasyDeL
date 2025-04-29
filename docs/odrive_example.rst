oDrive/oEngine Example
======================

The `oDrive` engine is one of the available engine drivers for EasyDeL's `vSurge` inference server. It is specifically designed for efficient inference by leveraging **Paged Attention**. Paged Attention is a key technique that optimizes the management of the KV (Key-Value) cache, which stores the intermediate attention outputs during sequence generation. By dividing the KV cache into fixed-size "pages," `oDrive` can handle variable-length sequences more efficiently, reduce memory fragmentation, and increase the utilization of High Bandwidth Memory (HBM).

The `oDrive` engine is instantiated using the `ed.vSurge.create_odriver` class method. Below is an example demonstrating how to create an `oDrive` instance:

.. code-block:: python

    surge = ed.vSurge.create_odriver(
        model=model,
        processor=processor,
        max_prefill_length=prefill_length,
        prefill_lengths=[prefill_length], 
        page_size=page_size,
        hbm_utilization=hbm_utilization,
        max_concurrent_prefill=max_concurrent_decodes,
        max_concurrent_decodes=max_concurrent_decodes,
        seed=877,
        vsurge_name="my_odrive_server", 
    )

Parameters:
-----------

The `create_odriver` function accepts several parameters to configure the `oDrive` engine:

:model: The loaded EasyDeL model instance that will be used for inference.
:processor: The tokenizer or processor object required for encoding input prompts and decoding generated tokens.
:max_prefill_length: The maximum sequence length allowed during the initial prompt processing (prefill) phase. Prompts longer than this may be truncated.
:prefill_lengths: An integer specifying a maximum prefill length to optimize kernels for, or None. This helps the engine optimize for specific input lengths.
:page_size: The size of memory pages used for managing the KV cache. A core parameter for the Paged Attention mechanism, influencing memory allocation granularity.
:hbm_utilization: The target utilization ratio for High Bandwidth Memory (HBM) allocated for the KV cache. This helps control memory usage.
:max_concurrent_prefill: The maximum number of prefill requests that the engine can process simultaneously. This affects the throughput of initial prompt processing.
:max_concurrent_decodes: The maximum number of decoding steps that can be executed concurrently across all active inference requests. This parameter also effectively limits the total number of concurrent requests the driver can handle.
:seed: A random seed used for operations within the engine, contributing to reproducibility.
:vsurge_name: (Optional) A string identifier assigned to this specific `vSurge` instance. Defaults to the driver's name if not provided.

The `oDrive` engine handles the underlying inference logic, manages the KV cache efficiently using Paged Attention (configured via `page_size` and `hbm_utilization`), and schedules incoming requests based on parameters like `max_concurrent_prefill` and `max_concurrent_decodes`.