# Training a Causal Language Model with EasyDeL
In this tutorial, we will guide you through setting up and training a causal language model using the `EasyDeL` library. The example uses the Llama model architecture, demonstrating how to set up the model, prepare a training dataset, and configure the training process.

----

### 1. Install Required Libraries
First, let's install the necessary libraries and set up authentication for accessing the Hugging Face and Weights & Biases platforms.


```
!pip install git+https://github.com/huggingface/transformers -U -q
!pip install git+https://github.com/erfanzar/easydel.git -U -q
!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q -U

# Configure git and login to Hugging Face Hub
!git config --global credential.helper store
!huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN --add-to-git-credential

# Login to Weights & Biases
!wandb login YOUR_WANDB_API_KEY
```

Make sure to replace `YOUR_HUGGINGFACE_TOKEN` and `YOUR_WANDB_API_KEY` with your actual tokens. This step will set up the environment for training and allow you to track experiments.

----

### 2. Import Required Libraries
After installing the dependencies, import the necessary libraries for training.


```python
import easydel as ed
from easydel.utils.analyze_memory import SMPMemoryMonitor  # Optional: For checking memory usage
import jax
from transformers import AutoTokenizer
from jax import numpy as jnp, sharding, lax, random as jrnd
from huggingface_hub import HfApi
import datasets
from flax.core import FrozenDict

# Set up sharding and API utilities
PartitionSpec, api = sharding.PartitionSpec, HfApi()

```


----

### 3. Model Configuration and Initialization
Here, we define model parameters and load a pretrained Llama model using EasyDeL.


```python
sharding_axis_dims = (1, -1, 1, 1)
max_length = 2048
input_shape = (len(jax.devices()), max_length)
pretrained_model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
pretrained_model_name_or_path_tokenizer = pretrained_model_name_or_path
new_repo_id = "EasyDeL/Llama-3.2-3B-Instruct"
dtype = jnp.bfloat16

# Load the pretrained model with automatic sharding
model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    input_shape=input_shape,
    auto_shard_model=True,
    sharding_axis_dims=sharding_axis_dims,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        use_scan_mlp=False,
        attn_dtype=jnp.float32,
        freq_max_position_embeddings=max_length,
        mask_max_position_embeddings=max_length,
        attn_mechanism=ed.AttentionMechanisms.VANILLA
    ),
    param_dtype=dtype,
    dtype=dtype,
    precision=lax.Precision("fastest"),
)
```

This code initializes a Llama model with sharding for efficient multi-device training. Adjust the model name to load other pretrained models from the Hugging Face Hub.

----

### 4. Prepare the Tokenizer
We set up a tokenizer for the model, which is responsible for converting text into input IDs for training.




```python
config = model.config
model_use_tie_word_embedding = config.tie_word_embeddings
model_parameters = FrozenDict({"params": params})

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path_tokenizer, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
tokenizer.padding_side = "right"
```


----

### 5. Load and Prepare the Dataset
In this step, load your training data using the datasets library and apply the tokenizer.


```python
# Replace with your dataset loading code
train_dataset = datasets.concatenate_datasets([
    # Add datasets here
])

# Tokenize the dataset using a chat template function (adjust as needed)
tokenized_dataset = train_dataset.map(
    lambda x: tokenizer.apply_chat_template(x["conversation"], tokenize=True, return_dict=True),
    remove_columns=train_dataset.column_names
)

# (Optional) Pack sequences to optimize training
packed_dataset = ed.pack_sequences(tokenized_dataset, max_length)
```

Make sure to replace the placeholder with your actual dataset details. You can explore different preprocessing methods depending on your dataset.

----

### 6. Define Training Arguments
Configure the training process by setting up various arguments such as learning rate, batch size, and training epochs.


```python
train_arguments = ed.TrainingArguments(
    num_train_epochs=1,
    learning_rate=9e-5,
    learning_rate_end=9e-6,
    warmup_steps=100,
    optimizer=ed.EasyDeLOptimizers.ADAMW,
    scheduler=ed.EasyDeLSchedulers.WARM_UP_COSINE,
    weight_decay=0.02,
    total_batch_size=48,
    max_sequence_length=max_length,
    gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=sharding_axis_dims,
    gradient_accumulation_steps=1,
    init_input_shape=input_shape,
    dtype=dtype,
    param_dtype=dtype,
    model_name=new_repo_id.split("/")[-1].split("-v")[0],
    training_time="7H",
    track_memory=True, # Req go-lang
)
```

The arguments can be adjusted based on your computational resources and training goals. Check the documentation for more details on each parameter.

----

### 7. Train the Model
Now, create the trainer and start training.


```python
trainer = ed.CausalLanguageModelTrainer(
	arguments=train_arguments,
	model=model,
	dataset_train=packed_dataset,
)

output = trainer.train(model_parameters=model_parameters, state=None)
```

This code will start the training process and save model checkpoints at specified intervals.

----

### 8. Save and Upload the Model
After training, save the model and upload it to the Hugging Face Hub.


```python
# Create a new repository or update an existing one
api.create_repo(new_repo_id, private=True, exist_ok=True)

# Save the trained model
file_path = "/".join(output.checkpoint_path.split("/")[:-1])
output.state.module.save_pretrained(
	file_path, output.state.params["params"], float_dtype=dtype
)

# Upload the model to the Hugging Face Hub
api.upload_folder(
	repo_id=new_repo_id,
	folder_path=file_path,
	ignore_patterns="events.out.tfevents.*",
)
```

This step allows you to save and share your trained model, making it accessible for future use.

----

### Conclusion
You've now trained a causal language model using `EasyDeL`! Feel free to explore more configuration options and try different datasets to see how the model's performance varies. For further customization and detailed explanations, refer to the `EasyDeL` documentation.

Happy training!
