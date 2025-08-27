vSurge API Server Example
=========================

The ``vSurgeApiServer`` class in EasyDeL provides a convenient way to wrap a pre-configured ``vSurge`` instance and expose it as an OpenAI-compatible API endpoint. This allows you to serve your vSurge model using standard API calls.

To initialize and run the API server, you can use the following pattern:

.. code-block:: python

    import jax
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    import easydel as ed


    pretrained_model_name_or_path = "Qwen/Qwen3-8B"

    dtype = param_dtype = jnp.bfloat16
    max_length = 8192
    prefill_length = 4096
    partition_axis = ed.PartitionAxis()
    processor = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    processor.padding_side = "left"
    processor.pad_token_id = processor.eos_token_id

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=param_dtype,
        dtype=dtype,
        partition_axis=partition_axis,
        precision=jax.lax.Precision.DEFAULT,
    )

    max_concurrent_decodes = 64
    max_concurrent_prefill = 1

    surge = ed.vSurge.from_model(
        model=model,
        processor=processor,
        max_concurrent_prefill=max_concurrent_prefill,
        max_concurrent_decodes=max_concurrent_decodes,
        seed=877,
    )

    ed.vSurgeApiServer(surge, max_workers=max_concurrent_decodes).fire()

The `max_workers` parameter controls the size of the API server's internal thread pool used to process incoming requests. It's important to set this value appropriately based on the capabilities and configuration of the underlying `vSurge` instance to avoid overloading the server or the model.

Parameters:
------------

*   **surge**: The pre-configured ``vSurge`` instance that the API server will use for inference. This instance should be set up with your desired model and configuration, as shown in the :doc:`vsurge_example`.
*   **max_workers**: An integer specifying the maximum number of concurrent requests the server can handle in its internal thread pool. This value should be chosen considering the capacity of the underlying ``vSurge`` instance to manage load effectively.

The ``.fire()`` Method:
-----------------------

The ``.fire()`` method is used to start the Uvicorn server, which hosts the API. It has several optional arguments to configure the server's behavior:

*   ``host`` (default: "0.0.0.0"): The host address to bind the server to.
*   ``port`` (default: 11556): The port number to listen on for API requests.
*   ``log_level`` (default: "info"): The logging level for the server.
*   ``ssl_keyfile`` (optional): Path to the SSL key file for HTTPS.
*   ``ssl_certfile`` (optional): Path to the SSL certificate file for HTTPS.

OpenAI API Compatibility:
-------------------------

The ``vSurgeApiServer`` is designed to be compatible with the OpenAI API specification, primarily focusing on chat-based inference. This means you can interact with the server using standard OpenAI client libraries or tools. The key endpoint ``/v1/chat/completions`` is supported, allowing you to perform chat-based inference requests. While other endpoints like `/v1/models` might be partially supported or planned, the main functionality currently revolves around chat completions.
