## EasyDeL SFT Training Tutorial: Fine-tuning Qwen2-7B for Chat

This tutorial guides you through fine-tuning the Qwen2-7B model for chat using EasyDeL's powerful SFT (Supervised Fine-tuning) Trainer.  We'll leverage distributed training on multiple GPUs/TPUs to handle this large language model efficiently. 

**1. Setup and Installation**

First, ensure you have the necessary libraries and a Hugging Face Hub token:

```bash
!python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('YOUR_HF_TOKEN')" 
!wandb login YOUR_WANDB_API_KEY
```

Replace `YOUR_HF_TOKEN` and `YOUR_WANDB_API_KEY` with your actual credentials. 

**2. Import Libraries**

```python
import jax
jax.devices("cpu")[0]  # JAX Force PJRT to be intialized.
import easydel as ed
from datasets import load_dataset
from flax.core import FrozenDict
from transformers import AutoTokenizer, AutoConfig
from jax import numpy as jnp, sharding
from huggingface_hub import HfApi
import os

PartitionSpec = sharding.PartitionSpec
api = HfApi()
```

**3. Hardware Check**

It's useful to check available GPUs/TPUs memory:

```python
def print_accelerator_status():
    for idx, device in enumerate(jax.devices()):
        platform = device.platform
        status = device.memory_stats()
        bytes_in_use = status["bytes_in_use"] / 1e9
        available  = status["bytes_reservable_limit"] / 1e9 - bytes_in_use
        print(f"{platform=} {idx} : {available=} | {bytes_in_use=}")

print_accelerator_status()
```

**4. Configuration**

Set up essential parameters for the model, training, and data:

```python
# Model & Training Configuration
sharding_axis_dims = (1, 1, 1, -1) # Sharding across the last axis
max_length = 8192  
input_shape = (1, max_length)
pretrained_model_name_or_path = "Qwen/Qwen2-7B" 
new_repo_id = "your-username/NewModelTrainedUsingEasyDeL" # Your Hugging Face repo

dtype = jnp.bfloat16  # Data type for reduced memory usage
use_lora = False
block_size = 128
attn_mechanism = "flash" # Use (blocwise, wise_ring, ring, local_ring) for really-long-context-size (Check docs also)
partition_axis = ed.PartitionAxis()
from_torch = False 
```

**5. Load Model and Tokenizer**

```python
# Load pre-trained model with sharding
model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    device=jax.devices('cpu')[0], # Load to CPU first for large models
    input_shape=input_shape,
    device_map="auto",
    auto_shard_params=True,
    sharding_axis_dims=sharding_axis_dims,
    verbose_params=True,
    config_kwargs=dict(
        use_scan_mlp=False, # Turn this On for really-long-context-size
        attn_mechanism=attn_mechanism,
        partition_axis=partition_axis
    ),
    partition_axis=partition_axis,
    param_dtype=dtype,
    dtype=dtype,
    from_torch=from_torch
)

config = model.config 
model_parameters = FrozenDict({"params": params})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True
)
```

**6. Model and Tokenizer Adjustments**

```python
model_use_tie_word_embedding = config.tie_word_embeddings 

# Add basic configurations for sharded attention
config.add_basic_configurations(
    attn_mechanism=attn_mechanism,
    shard_attention_computation=True,
)

# Set up model class initialization arguments
configs_to_initialize_model_class = {
    "config": config,
    "dtype": dtype,
    "param_dtype": dtype,
    "input_shape": input_shape
}

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right" 
```

**7. (Optional) LoRA (Low-Rank Adaptation) Configuration**

to enable LoRA, a technique for reducing memory usage during fine-tuning:

```python 
rapture_config = ed.LoraRaptureConfig(
    model_parameters,
    lora_dim=64,
    fully_fine_tune_parameters=["embed_tokens"], 
    lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj", "o_proj"],
    verbose=True
) if use_lora else None
```

**8. Load and Prepare the Dataset** 

```python
# Load your SFT dataset 
train_dataset = load_dataset("your-dataset-name", split="train") 

# ... (Add any necessary data preprocessing steps here)... 
```

**(Optional) Advanced: Custom Partition Rules**

This section allows for more fine-grained control over how model parameters are sharded. You can uncomment and customize this if needed:

```python
# rules = (
#     ('model/embed_tokens/embedding', PartitionSpec("tp",('fsdp', 'sp'),)),
#     # ... other rules ...
# )
# config.get_partition_rules = lambda _: rules
```

**9.  Set up Training Arguments**

```python
train_arguments = ed.TrainingArguments(
    model_class=ed.get_modules_by_type(config.model_type)[1],
    configs_to_initialize_model_class=configs_to_initialize_model_class,
    custom_rule=config.get_partition_rules(True), 
    num_train_epochs=4,
    learning_rate=1.5e-5,
    learning_rate_end=9e-6,
    warmup_steps=50,
    optimizer=ed.EasyDeLOptimizers.ADAMW,
    scheduler=ed.EasyDeLSchedulers.WARM_UP_COSINE,
    weight_decay=0.02,
    total_batch_size=16,
    max_sequence_length=max_length,
    gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=sharding_axis_dims,
    use_pjit_attention_force=False,
    gradient_accumulation_steps=1,
    init_input_shape=input_shape,
    dtype=dtype,
    param_dtype=dtype,
    step_start_point=0,
    do_last_save=False,
    model_name=new_repo_id.split("/")[-1].split("-v0")[0],
    training_time="7H", 
    force_batch_and_gradient_accumulation_steps_calculation=False,
    rapture_config=rapture_config, # For LoRA (if used)
    track_memory=False, # Requires Go installation
)
```

**10. Create and Run the SFT Trainer**

```python
trainer = ed.SFTTrainer(
    arguments=train_arguments,
    train_dataset=train_dataset,
    eval_dataset=None, # Add an evaluation dataset if needed
    tokenizer=tokenizer,
    dataset_text_field=None, # Replace with the name of the text field in your dataset
    formatting_func=lambda x: [tokenizer.apply_chat_template(x["conversation"], tokenize=False)],
    packing=False, 
    num_of_sequences=max_length,
    dataset_num_proc=32 
)

# Train the model
output = trainer.train(
    model_parameters=model_parameters if not use_lora else None,
    state=None 
)
```

**11. Save the Fine-tuned Model**

```python
api.create_repo(new_repo_id, private=True, exist_ok=True)

file_path = "/".join(output.checkpoint_path.split("/")[:-1])
output.state.module.save_pretrained(file_path, output.state.params["params"], float_dtype=dtype) 

# Upload to Hugging Face Hub
api.upload_folder(
    repo_id=new_repo_id,
    folder_path=file_path,
)
```

This revised tutorial provides a detailed breakdown of the provided EasyDeL SFT training pipeline, making it easier to understand and adapt for your own chat-based fine-tuning tasks.

You've encountered a comprehensive example showcasing the power of EasyDeL for training large language models! However, remember that this pipeline merely scratches the surface of what EasyDeL can achieve.
For those seeking advanced configurations or unique features:
Dive into the Documentation: The EasyDeL documentation is your treasure trove! Explore detailed explanations, customization options, and more advanced use cases.

https://easydel.readthedocs.io/en/latest/index.html
EasyDeL prides itself on flexibility and performance. If you haven't found the specific arguments or settings you need, chances are they're waiting for you in the docs! Don't hesitate to unlock the full potential of EasyDeL!