# Supervised Fine-Tuning with SFTTrainer

This tutorial guides you through the process of fine-tuning a model using the `SFTTrainer` class from the `EasyDeL` library. This setup allows you to customize training with ease, and it’s designed to handle various configurations for supervised fine-tuning (`SFT`). Remember, there are many more options and possibilities—explore the documentation to unlock the full potential of `EasyDeL`!

### 1. Importing Required Libraries


```python
import easydel as ed
from easydel.utils.analyze_memory import SMPMemoryMonitor  # Optional for memory analysis
import jax
from transformers import AutoTokenizer
from jax import numpy as jnp, sharding, lax, random as jrnd
from huggingface_hub import HfApi
import datasets
from flax.core import FrozenDict

PartitionSpec, api = sharding.PartitionSpec, HfApi() 
```


In this section, we import the essential libraries needed for fine-tuning a language model. `EasyDeL` helps manage the model training, `jax` provides efficient numerical operations, and `transformers` is used for tokenizer management. `SMPMemoryMonitor` is optionally imported for monitoring memory usage, which can be particularly useful when working with large models. For more advanced usage and utilities, explore the documentation!

--------------------------
### 2. Defining Configuration Parameters


```python
sharding_axis_dims = (1, -1, 1, 1)
max_length = 8192
input_shape = (len(jax.devices()), max_length)
pretrained_model_name_or_path = "google/gemma-2-2b-it"
pretrained_model_name_or_path_tokenizer = pretrained_model_name_or_path
new_repo_id = "EasyDeL/sft-gemma-2-2b-it"
dtype = jnp.bfloat16
```


Here, we define the basic configuration for our training process, including the dimensions for sharding, maximum sequence length, and the input shape for the model. Adjusting these parameters allows you to tailor the model’s memory and computational efficiency to your setup. Remember, there are more options available in the documentation that can help you fine-tune your setup further!

--------------------------
### 3. Loading the Pretrained Model


```python
model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
	pretrained_model_name_or_path,
	input_shape=input_shape,
	auto_shard_params=True,
	sharding_axis_dims=sharding_axis_dims,
	config_kwargs=ed.EasyDeLBaseConfigDict(
		use_scan_mlp=False,
		attn_dtype=jnp.float32,
		freq_max_position_embeddings=max_length,
		mask_max_position_embeddings=max_length,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
	),
	param_dtype=dtype,
	dtype=dtype,
	precision=lax.Precision("fastest"),
)
```

This section initializes the model using `AutoEasyDeLModelForCausalLM` with custom `parameters`. The configuration options like `attention mechanism` and model precision allow you to optimize for speed or memory efficiency. You can also explore different attention mechanisms and configurations by checking out the documentation.

-------------------------

### 4. Setting Up the Tokenizer


```python
config = model.config
model_use_tie_word_embedding = config.tie_word_embeddings
model_parameters = FrozenDict({"params": params})

tokenizer = AutoTokenizer.from_pretrained(
	pretrained_model_name_or_path_tokenizer, trust_remote_code=True
)
tokenizer.pad_token = (
	tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
)
tokenizer.padding_side = "right"
```

Here, we load the `tokenizer` using `transformers` and adjust the padding token if necessary. We also wrap the model parameters using `FrozenDict` for efficient handling during training. Tokenization is crucial in preparing your data for model input, and `EasyDeL` allows you to tweak this process—explore the documentation for more customizations!

------

### 5. Preparing the Dataset


```python
train_dataset = datasets.concatenate_datasets(
	[
		# Your SFT Datasets come here
	]
)
```


In this step, we prepare the training dataset. You can load multiple `datasets` and combine them for fine-tuning, depending on your needs. `Data processing` can be adapted to your specific use case. Be sure to explore the documentation for different dataset handling techniques!

----

### 6. Configuring Training Parameters


```python
train_arguments = ed.TrainingArguments(
	num_train_epochs=3,
	learning_rate=8e-5,
	learning_rate_end=9e-6,
	warmup_steps=100,
	optimizer=ed.EasyDeLOptimizers.ADAMW,
	scheduler=ed.EasyDeLSchedulers.WARM_UP_COSINE,
	weight_decay=0.02,
	total_batch_size=16,
	max_sequence_length=max_length,
	gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
	sharding_array=sharding_axis_dims,
	gradient_accumulation_steps=1,
	init_input_shape=input_shape,
	dtype=dtype,
	param_dtype=dtype,
	model_name=new_repo_id.split("/")[-1].split("-v")[0],
	training_time="7H",
	track_memory=False,
)
```

We define the training configuration using `TrainingArguments`. This includes parameters like the `learning rate`, `number of training epochs`, `optimizer`, and `scheduler` type. Adjust these settings to find the optimal training regimen for your dataset and hardware. `EasyDeL` offers many more options for customizing training—check the documentation for more possibilities.

----

### 7. Initializing the SFTTrainer


```python
trainer = ed.SFTTrainer(
	arguments=train_arguments,
	model=model,
	train_dataset=train_dataset,
	eval_dataset=None,
	tokenizer=tokenizer,
	dataset_text_field=None,
	formatting_func=lambda x: [
		tokenizer.apply_chat_template(x["conversation"], tokenize=False)
	],
	packing=True,
	num_of_sequences=max_length,
	dataset_num_proc=128,
)
```

We set up the `SFTTrainer` with the `model`, `tokenizer`, and `training dataset`. The `SFTTrainer` allows for customization of how text inputs are formatted and packed for training. You can adjust the formatting_func or other parameters as needed. Many other training configurations are available, so be sure to explore the documentation for deeper customization.

----

### 8. Training the Model and Uploading to Hugging Face


```python
output = trainer.train(model_parameters=model_parameters, state=None)
api.create_repo(new_repo_id, private=True, exist_ok=True)
file_path = "/".join(output.checkpoint_path.split("/")[:-1])
output.state.module.save_pretrained(
	file_path, output.state.params["params"], float_dtype=dtype
)
api.upload_folder(
	repo_id=new_repo_id, folder_path=file_path, ignore_patterns="events.out.tfevents.*"
)
```

In this final step, we train the model using the `SFTTrainer` and save the trained model to the specified path. We then create a private repository on the Hugging Face Hub and upload the trained model files. This allows you to easily share or deploy your fine-tuned model. For more details on managing checkpoints or integrating with other tools, refer to the documentation.

----

With these steps, you have fine-tuned a model using `EasyDeL's` `SFTTrainer`! Remember, this tutorial covers the basics, but the library offers many more options to explore. Dive into the documentation to fully customize your `fine-tuning` workflow and get the best results. Happy fine-tuning!
