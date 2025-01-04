# EasyDeL Tutorial

This tutorial demonstrates the essential steps for using the `EasyDeL` library with JAX and Hugging Face for training a causal language model. You'll learn how to set up your model, tokenizer, dataset, and training configuration. Remember, this is just the beginning—there are many more options you can tweak, so check out the documentation to unlock the full potential of `EasyDeL`!

1. Importing Required Libraries


```python
import easydel as ed
import jax
from transformers import AutoTokenizer
from jax import numpy as jnp, sharding, lax
from huggingface_hub import HfApi
import datasets

PartitionSpec, api = sharding.PartitionSpec, HfApi() 
```

In this section, we import the necessary libraries, including `EasyDeL`, JAX, and transformers for model and tokenizer management. `HfApi` is used to interact with Hugging Face's model hub, while `datasets` provides a way to handle training data. There are many additional modules and functions that can be leveraged for more customized workflows—check the documentation for further details!

----

2. Defining Configuration Parameters


```python
sharding_axis_dims = (1, -1, 1, 1)
sharding_axis_names = ("dp", "fsdp", "tp", "sp")
max_length = 2048
num_devices = len(jax.devices())
input_shape = (num_devices, max_length)
pretrained_model_name_or_path = "google/gemma-2-2b-it"
pretrained_model_name_or_path_tokenizer = pretrained_model_name_or_path
new_repo_id = "EasyDeL/dpo-gemma-2-2b-it"
dtype = jnp.bfloat16
partition_axis = ed.PartitionAxis()
```

Here, we define various configuration parameters for model training. This includes the dimensions for sharding across devices, the maximum sequence length, and the model and tokenizer identifiers. The input shape and data types for model parameters are also set. These configurations provide flexibility in adjusting the model's behavior to better suit your hardware and data needs. For more advanced options, refer to the documentation.

----

3. Loading the Pretrained Model


```python
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
	pretrained_model_name_or_path,
	input_shape=input_shape,
	device_map="auto",
	# device_map help to load first pytorch model in case that you are facing heavy loads!, we don't need that
	# if your loading from a easystate, or an easydel model which was saved using `ed_model.save_pretraiend``.
	auto_shard_model=True,
	sharding_axis_dims=sharding_axis_dims,
	sharding_axis_names=sharding_axis_names,
	config_kwargs=ed.EasyDeLBaseConfigDict(
		use_scan_mlp=False,
		partition_axis=partition_axis,
		attn_dtype=jnp.float32,
		freq_max_position_embeddings=max_length,
		mask_max_position_embeddings=max_length,
		attn_mechanism=ed.AttentionMechanisms.SDPA,  
		# feel free to change attention to whatever mechanism you want (e.g FLASH_ATTN2, SPLASH, ...)
		# if your using kaggle TPUs, i highly suggest to don't use SDPA attention and switch to VANILLA.
	),
	partition_axis=partition_axis,
	param_dtype=dtype,
	dtype=dtype,
	precision=lax.Precision("fastest"),
)
```

This section shows how to load a pretrained model with `AutoEasyDeLModelForCausalLM` and customize various parameters like device mapping and precision. The `config_kwargs` allows fine-tuning of the attention mechanism and MLP settings. Feel free to explore other configuration options to fit your model needs better—see the documentation for more choices!

----

4. Setting Up the Tokenizer


```python
tokenizer = AutoTokenizer.from_pretrained(
	pretrained_model_name_or_path_tokenizer,
	trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
tokenizer.padding_side = "right"
```

We initialize a tokenizer using the same model identifier, setting the padding token and adjusting padding alignment. Tokenization is crucial for preparing text inputs for training. `EasyDeL` supports other tokenization options and configurations—refer to the documentation to explore them.

----

5. Preparing the Dataset


```python
train_dataset = (
	datasets.concatenate_datasets(
		[
			datasets.load_dataset(
				"argilla/ultrafeedback-binarized-preferences",
				split="train",
			),
			# show case of loading multi-datasets.
		]
	)
	.shuffle()
	.shuffle()
)
# DPOTrainer needs data to be formated like this (prompt, chosen, rejected).
train_dataset = train_dataset.rename_column("chosen_response", "chosen")
train_dataset = train_dataset.rename_column("rejected_response", "rejected")
train_dataset = train_dataset.rename_column("instruction", "prompt")
```

Here, we load and preprocess a training dataset, using the datasets library to concatenate and shuffle it. The columns are renamed to match the input fields required for training. You can modify this to use different datasets, perform more advanced preprocessing, or integrate other data sources. Check the docs for more options!

----

6. Configuring Training Parameters


```python
training_arguments = ed.DPOConfig(
	# DPO Configs
	loss_type="hinge",
	
	beta=0.1,
	
	label_smoothing=0.0,
	truncation_mode="keep_end",
	
	max_length=max_length,
	max_prompt_length=max_length - (max_length // 4),
	max_completion_length=max_length - (max_length // 4),

	# Other Configs.
	num_train_epochs=1,
	learning_rate=1.5e-5,
	learning_rate_end=9e-6,
	warmup_steps=100,
	
	optimizer=ed.EasyDeLOptimizers.ADAMW,
	scheduler=ed.EasyDeLSchedulers.COSINE,
	
	weight_decay=0.02,
	total_batch_size=8,

	max_sequence_length=max_length,
	gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
	sharding_array=sharding_axis_dims,
	gradient_accumulation_steps=1,
	init_input_shape=input_shape,
	dtype=dtype,
	param_dtype=dtype,
	model_name=new_repo_id.split("/")[-1].split("-v0")[0],
	training_time_limit="7H",
	track_memory=True,
)
```

We configure the training process with `DPOConfig`, specifying settings like loss type, learning rate, and optimizer. This configuration allows users to fine-tune their training settings to achieve optimal results. Feel free to tweak these parameters as you explore different models and training strategies—there's more in the documentation to guide you!

----

7. Preparing Model State


```python
model_state = model.to_easydel_state(params=params)
```

This converts the model and its parameters into a format compatible with `EasyDeL` training workflows. It's a simple step but an important one before starting the training with dpo trainer. For further details on how model_state can be customized or extended, consult the documentation.

----

8. Initializing the Trainer


```python
trainer = ed.DPOTrainer(
	arguments=training_arguments,
	model_state=model_state,
	ref_model_state=model_state,  # since we dont have a renfrence model for now, but you can use other models as reference.
	tokenizer=tokenizer,
	train_dataset=train_dataset,
)
```

We set up a `DPOTrainer` to handle the training process, including the model, tokenizer, and dataset. Using a reference model allows for comparative training, but in this example, we're using the same model state for simplicity. Many more configurations are possible, so explore the docs for deeper customization!

----

9. Training the Model


```python
# Except long compilation time. like 7-8 min
# since a lot of functions like loss functions model state and ref model state etc are being compiled this process
# can take up to 46 min for larger models (Llama70B) and it depends a lot on a processor for example on GPUs it's a lot faster.

output = trainer.train()
```

Finally, we start the training process. The output includes metrics and results from the training run, which you can use to monitor progress. Remember, you can customize everything from the training loop to data augmentation. Refer to the documentation for advanced training options.

----

With these steps, you've set up and trained a model using EasyDeL! There's a lot more you can do—be sure to dive into the documentation to explore further options and tailor this workflow to your specific needs. Happy training!
