# GPU and TPU Inference with EasyDeL
In this tutorial, we’ll walk through setting up and performing efficient inference on both GPU and TPU devices using EasyDeL. This setup utilizes model parallelism and quantization for optimized performance.

### Requirements
Ensure you have EasyDeL, Transformers, JAX, and Torch installed in your environment.
(torch is not needed in case that your not loading a torch model and using already converted EasyDeL Models).


```python
!pip install git+https://github.com/erfanzar/EasyDeL.git -q
!pip install jax[cuda12] -q
# or install jax for TPUs
!pip install torch -q # For GPUS only
```

#### 1. Import Required Libraries
Begin by importing the necessary libraries.

---


```python
import easydel as ed
import jax
import transformers
from jax import numpy as jnp
import torch
```


---

#### 2. Configure Model and Inference Parameters
Define the model and inference settings. Adjust the sharding_axis_dims, dtype, and precision for either GPU or TPU:

##### Model Configuration


```python
pretrained_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
max_length = 8192  # Maximum length of input sequences
num_devices = jax.device_count()

# Initialize the model with specific sharding and quantization settings
model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    sharding_axis_dims=(1, 1, 1, -1),  # Adjust this based on device type
		# Use Sequence Sharding or Tensor Parallelization for this
    auto_shard_params=True,
    dtype=jnp.float16, 
    param_dtype=jnp.float16,
    precision=None,
    input_shape=(num_devices, max_length),
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_method=ed.EasyDeLQuantizationMethods.A8BIT,  # Use 8-bit quantization for inference efficiency
    config_kwargs=ed.EasyDeLBaseConfigDict(
        quantize_kv_cache=True,
        attn_dtype=jnp.float16,
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,  # Faster attention mechanism
        mask_max_position_embeddings=max_length,
        freq_max_position_embeddings=max_length,
    ),
)
```

#### Key Parameters

- *Sharding Axis*: Set to (1, 1, 1, -1) to optimize for sequence sharding or tensor parallelization on TPUs or GPUs.

- *Quantization*: We use 8-bit quantization to reduce memory usage and improve inference speed.

- *Attention Mechanism*: FLASH_ATTN2 provides efficient attention handling for large sequences.

- *Precision*: Set to float16 for efficient computation on hardware accelerators.

---

#### 3. Prepare the Tokenizer
Load the tokenizer for preprocessing the input text.


```python
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
```


---

#### 4. Initialize the Inference Class

Set up the vInference class to handle inference requests. Configure generation parameters to control text generation, such as temperature, top-k sampling, and maximum token length.


```python
inference = ed.vInference(
	model=model,
	params=params,
	tokenizer=tokenizer,
	generation_config=ed.vInferenceConfig(
		temperature=0.8,
		top_k=10,
		top_p=0.95,
		bos_token_id=model.generation_config.bos_token_id,
		eos_token_id=model.generation_config.eos_token_id,
		pad_token_id=model.generation_config.pad_token_id,
		streaming_chunks=32,
		max_new_tokens=1024,
	),
)
```

#### Generation Configuration

- *Temperature*: Controls randomness; higher values produce more diverse outputs.
- *Top-k and Top-p*: Top-k sampling selects the k most likely next tokens, while top-p sampling dynamically adjusts the number of tokens based on cumulative probability.
- *Max New Tokens*: Limits the number of tokens generated per inference.

---


#### 5. Precompile the Model for Inference.

For inference, it is beneficial to precompile the model to optimize execution.


```python
inference.precompile(batch_size=1)
```


---

#### 6. Deploy an API Server for Inference
Use vInferenceApiServer to expose the model as an API, allowing asynchronous requests.


```python
api_inference = ed.vInferenceApiServer(
	{inference.inference_name: inference}
)  # you can load multi inferences together
api_inference.fire()
```


This server will be ready to receive inference requests, making it ideal for deploying in a production environment.

---

#### Summary
This tutorial demonstrated how to configure and run inference on both `GPU` and `TPU` devices with `EasyDeL`. The setup used sharding, quantization, and efficient attention mechanisms to optimize inference. Adjustments in precision, sharding configuration, and precompilation steps allow seamless usage across different hardware.
