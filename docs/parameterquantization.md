# 8-bit EasyDeL Models

## What's 8-bit quantization? How does it help ?

Quantization in the context of deep learning is the process of constraining the number of bits that represent the
weights and biases of the model.

Weights and Biases numbers that we need in backpropagation.

In 8-bit quantization, each weight or bias is represented using only 8 bits as opposed to the typical 32 bits used in
single-precision floating-point format (float32).

## Why does it use less GPU/TPU Memory?

The primary advantage of using 8-bit quantization is the reduction in model size and memory usage. Here's a simple
explanation:

A float32 number takes up 32 bits of memory.
A 8-bit quantized number takes up only 8 bits of memory.
So, theoretically, you can fit 4 times more 8-bit quantized numbers into the same memory space as float32 numbers. This
allows you to load larger models into the GPU memory or use smaller GPUs that might not have been able to handle the
model otherwise.

The amount of memory used by an integer in a computer system is directly related to the number of bits used to represent
that integer.

Memory Usage for 8-bit Integer
A 8-bit integer uses 8 bits of memory.

Memory Usage for 32-bit Integer
A 32-bit integer uses 32 bits of memory.

Conversion to Bytes
To convert these to bytes (since memory is often measured in bytes):

- 1 byte = 8 bits
- 8-bit integer would use ( 8/8 = 1 ) bytes.
- A 16-bit integer would use ( 16/8 = 2 ) bytes.

## Example of Using Parameters Quantization in EasyDeL

in case of serving models or using them with `JAX` The Easiest and the best way you can find
is EasyDeL (you can explore more if you want) you have 4 ways to use models

1. Create The Pipeline and everything from scratch yourself.
2. Use JAXServer API from EasyDeL.
3. use ServeEngine from EasyDeL.
4. use builtin generate method from HuggingFace Transformers and EasyDeL

let assume we want to run a 7B model on only 12 GB of vram let just jump into codding

## Using Quantized Model via generate Function

let assume we want to run `Qwen/Qwen1.5-7B-Chat`

```python
from jax import numpy as jnp
from easydel import AutoEasyDeLModelForCausalLM

import torch

repo_id = "Qwen/Qwen1.5-7B-Chat"
model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
    repo_id,
    sharding_axis_dims=(1, 1, 1, -1),
    config_kwargs=dict(
        gradient_checkpointing="",
        use_scan_mlp=False,  # Turn this one if you want to go beyond 32K sequence length.
        shard_attention_computation=True,
        use_sharded_kv_caching=True
    ),
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    auto_shard_params=True,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="cpu"  # this one will be passed to transformers.AutoModelForCausalLM
)
# Now params are loaded as 8-bit fjformer kernel

```

### TIP

you can do everything but training with 8Bit params in EasyDeL, and use pickle to saving and loading them