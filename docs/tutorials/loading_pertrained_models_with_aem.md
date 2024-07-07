# AutoEasyDeLModelForCausalLM

**In essence, `AutoEasyDeLModelForCausalLM` is your EasyDeL shortcut for using pre-trained causal language models from the Hugging Face Hub.** 

- **Think of it as a specialized model loader designed to streamline your workflow when working with LLMs like Cohere, Llama, Mixtral, etc., within the EasyDeL framework.**

**Here's what it does:**

1. **Downloads:** Fetches the model architecture and pre-trained weights from the Hugging Face Hub based on the model name or path you provide.
2. **Converts:** Transforms the model from its original PyTorch format to a JAX-based representation compatible with EasyDeL.
3. **Shards (Distributes):**  Optionally splits the model's parameters across multiple GPUs or TPUs for efficient distributed training, allowing you to work with much larger models and datasets. 

**Benefits:**

- **Simplified Model Loading:** No need to manually download, convert, and shard models. 
- **Distributed Training Ready:**  Easily configure your model for distributed training with various sharding strategies.
- **Optimized for Performance:**  Options to control data types, precision, and other settings for efficient training and inference.

**In short, `AutoEasyDeLModelForCausalLM` makes it significantly easier to leverage the power of pre-trained causal language models within your EasyDeL projects.** 


## Usage and Argument Explanation
#### Note
this usage and argument explanation is updated at 6/21/2024



Let's break down each argument with a focus on practical usage and decision-making:

```python
class AutoEasyDeLModelForCausalLM: 
    # ... (rest of the class definition)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str, # Path or name of the pretrained model (e.g., "gpt2", "facebook/bart-large-cnn") 
        device=jax.devices("cpu")[0],      # Device for initial model loading (CPU recommended for large models)
        dtype: jax.numpy.dtype = jax.numpy.float32, # Data type for model computations (float32 is standard)
        param_dtype: jax.numpy.dtype = jax.numpy.float32, # Data type for storing model parameters (float32 is standard)
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"), # Computational precision (use "fastest" for optimal performance on your hardware)
        sharding_axis_dims: Sequence[int] = (1, -1, 1, 1), # Sharding dimensions for (dp, fsdp, tp, sp)
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"), # Names corresponding to sharding dimensions
        partition_axis: PartitionAxis = PartitionAxis(), # For advanced sharding with EasyDeL's PartitionAxis module (usually leave as default) 
        shard_attention_computation: bool = True, # Shard attention for better memory efficiency (recommended: True) 
        input_shape: Tuple[int, int] = (1, 1),   # Input shape for the model (batch size, sequence length), used for initialization
        shard_fns: Optional[Mapping[tuple, Callable] | dict] = None, # Custom sharding functions (advanced use)
        backend: Optional[str] = None,           # Backend for JAX ("cpu", "gpu", "tpu"; usually auto-detected)
        config_kwargs: Optional[Mapping[str, Any]] = None, # Keyword arguments to pass to the model's configuration
        auto_shard_params: bool = False,        # Automatically determine sharding (if True, `shard_fns` are ignored)
        partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]] = None, # Rules for auto-sharding (if `auto_shard_params` is True)
        load_in_8bit: bool = False,             # Load model weights in 8-bit precision (for memory efficiency)
        bit_targeted_params: Optional[List[str]] = None, # List of parameter names to convert to 8-bit (if `load_in_8bit` is True)
        verbose_params: bool = False,            # Print the number of parameters in the loaded models
        from_torch: bool = True,                 # Load the model from PyTorch weights (Hugging Face)
        **kwargs,                                 # Additional keyword arguments passed to the model's initialization 
    ) -> Tuple[BaseNNXModule, dict]:
        # ... (rest of the method implementation)
```

**Explanation:**

1. **`pretrained_model_name_or_path`:**
   - **Purpose:** Specifies the pre-trained model you want to load.
   - **Example:** `"gpt2"`, `"facebook/bart-large-cnn"`, `"google/flan-t5-xl"`
   - **Note:**  Refer to the Hugging Face Model Hub for available models.

2. **`device`:**
   - **Purpose:**  The device where the model is initially loaded. For very large models, it's highly recommended to start with CPU (`jax.devices("cpu")[0]`) to avoid out-of-memory errors on the GPU. You can move the model to your desired device later. 
   - **Examples:** `jax.devices("cpu")[0]`, `jax.devices("gpu")[0]`

3. **`dtype` and `param_dtype`:**
   - **Purpose:**
     - `dtype`: Data type used for model computations during training and inference. 
     - `param_dtype`: Data type used to store the model's parameters. 
   - **Common Choices:**
     - `jax.numpy.float32` (default): Standard single-precision floating-point.
     - `jax.numpy.bfloat16`:  Reduced precision (16-bit brain floating-point), can speed up training and save memory, but might slightly impact accuracy. 
   - **Note:** For most cases, the default `float32` is suitable. Use `bfloat16` if you need to reduce memory consumption or speed up training, especially for large models.

4. **`precision`:** 
   - **Purpose:** Controls the precision of JAX computations.
   - **Recommended:** `jax.lax.Precision("fastest")` lets JAX choose the most efficient precision based on your hardware.
   - **Other Options:** `jax.lax.Precision("high")`, `jax.lax.Precision("default")` - offer more control but might be slower. 

5. **`sharding_axis_dims` and `sharding_axis_names`:**
   - **Purpose:** Define how the model's parameters are split (sharded) across multiple devices for distributed training.
   - **Understanding Sharding:** 
     - **Data Parallelism (DP):** Replicates the model on each device and splits the data. Good for smaller models that can fit on a single device.
     - **Fully Sharded Data Parallelism (FSDP):**  Shards model parameters and optimizer states across devices.  Essential for very large models that exceed the memory of a single device. 
     - **Tensor Parallelism (TP):** Splits individual layers (tensors) of the model across devices.
     - **Sequence Parallelism (SP):**  Used to parallelize Sequences and model (if you haven't change default behaviors).
   - **Example (8 GPUs, FSDP along the second axis):**
      ```python
      sharding_axis_dims = (1, 8, 1, 1)  # (dp, fsdp, tp, sp)
      sharding_axis_names = ("dp", "fsdp", "tp", "sp") 
      ```

6. **`partition_axis`:**
   - **Purpose:** Used for advanced sharding control in conjunction with EasyDeL's `PartitionAxis` module. Not typically modified for common use cases. 

7. **`shard_attention_computation`:**
   - **Purpose:**  Determines whether to shard attention computations, which is generally recommended for improved memory efficiency.
   - **Recommended:** `True` 

8. **`input_shape`:**
   - **Purpose:**  Specifies the expected input shape for your model (batch size, sequence length). This is used during model initialization.

9. **`shard_fns`, `auto_shard_params`, `partition_rules`:**
   - **Purpose:** 
     - `shard_fns`: Allows you to provide custom sharding functions (advanced use case).
     - `auto_shard_params`:  If set to `True`, EasyDeL will automatically determine a sharding strategy, overriding any provided `shard_fns`.
     - `partition_rules`: When using `auto_shard_params`, you can specify rules to guide EasyDeL's automatic sharding process. 
   - **Note:** For most cases, start with `auto_shard_params=True` and let EasyDeL handle sharding. Use the more advanced options only if you need very specific control.

10. **`load_in_8bit` and `bit_targeted_params`:**
   - **Purpose:**
     - `load_in_8bit`: Enables loading model weights in 8-bit precision for reduced memory usage. 
     - `bit_targeted_params`: If `load_in_8bit` is `True`, you can specify which parameter types (e.g., `"kernel"` for weights, `"embedding"`) should be converted to 8-bit. If `None`, all applicable parameters are converted.
   - **Caution:** Using 8-bit precision can affect model accuracy, so carefully evaluate its impact on your specific task.

11. **`verbose_params`:**
   - **Purpose:** If `True`, prints the number of parameters in the loaded PyTorch and EasyDeL models, helping you understand the model size and memory usage.

12. **`from_torch`:**
   - **Purpose:** Indicates that you are loading a model from PyTorch weights (the common case when using pre-trained models from the Hugging Face Hub).
   - **Note:** EasyDeL also supports loading models from its own saved parameters (`BaseNNXModule.from_pretrained`). In that case, set `from_torch=False`. 

13. **`**kwargs`:**
   - **Purpose:** Catches any additional keyword arguments that might be passed to the model's configuration during initialization.

**Experiment and Profile:**

It's important to experiment with different sharding strategies, precision settings, and other arguments to find the optimal configuration for your hardware and model size. EasyDeL's flexibility allows you to fine-tune these aspects to maximize training efficiency.
