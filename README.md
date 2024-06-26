# EasyDeL 🔮

EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning models.
With a primary focus on Jax/Flax, EasyDeL aims to provide convenient and effective solutions for training Flax/Jax
models on TPU/GPU for both serving and training purposes.

## Key Features

1. **Trainers**: EasyDeL offers a range of trainers, including DPOTrainer, ORPOTrainer, SFTTrainer, and VideoCLM
   Trainer, tailored for specific training requirements.

2. **Serving and API Engines**: EasyDeL provides serving and API engines for efficiently using and serving large
   language models (LLMs) in JAX, enabling seamless integration into various applications.

3. **Quantization Support**: EasyDeL supports quantization methods for all models, allowing for efficient inference and
   training.

4. **Bit Operation Support**: EasyDeL supports 8, 6, and 4-bit operations for inference and training in JAX, optimizing
   performance and resource utilization.

5. **Diverse Model Support**: EasyDeL offers a wide range of models in JAX that have never been implemented before, such
   as Falcon, Qwen2, Phi2, Mixtral, Qwen2Moe, Cohere, Dbrx, Phi3, and MPT.

6. **FlashAttention Integration**: EasyDeL integrates FlashAttention in JAX for GPUs and TPUs, enhancing performance and
   efficiency.

7. **Automatic LLM Serving**: EasyDeL enables automatic serving of LLMs with mid and high-level APIs in both JAX and
   PyTorch, simplifying deployment and integration.

8. **LLM Training and Fine-tuning**: EasyDeL provides LLM trainer and fine-tuner capabilities in JAX, allowing for
   efficient training and customization of language models.

9. **Video CLM Training and Fine-tuning**: EasyDeL supports Video CLM trainer and fine-tuner for models such as Falcon,
   Qwen2, Phi2, MPT, Mixtral, Grok-1, and Qwen2Moe, enabling advanced video-related applications.

10. **Performance Optimization**: EasyDeL provides various features to enhance the training process and optimize
    performance, such as LoRA (Low-Rank Adaptation of Large Language Models), RingAttention, FlashAttention, BlockWise
    FFN, and Efficient Attention support (through the FJFormer backbone).

11. **Model Conversion**: EasyDeL supports automatic conversion of models from JAX-EasyDeL to PyTorch-HF and vice versa,
    facilitating seamless integration with different frameworks.

With its comprehensive set of features and tools, EasyDeL aims to streamline and accelerate the training and deployment
of machine learning models, particularly in the domain of large language models and video-related applications.

### Latest News 🔥
- Gemma2 Is now available.
- StableLM and DBrX model bugs are fixed.
- All the Models have their AttentionStructure Changed, and now they are faster and more memory efficient for inference.
- OLMo models are supported.
- Now Flash attention will automatically switch on pallas_flash in case that your on GPUs.
- GPU Flash Attention Bugs are now fixed.
- changing required version of jax from `>=0.4.23` to `>=0.4.28`, but `0.4.29` or higher versions (if available) are
  recommended.
- Import structure is now like the older versions and multi-host(GPU) training issues are fixed.
- KV cache quantization is improved and now have +%21 accuracy from old version (%69)
- adding `EasyDeLFlaxPretrainedModel.from_pretrained` and `EasyDeLFlaxPretrainedModel.save_pretrained` to give users
  experience of being free and not only using easydel states
- removing *(q,k,v,b,a)_partition_specs and using `PartitionAxis` instead of them.
- Sharding Strategies are changed.
- Now EasyDeL is more Memory efficient Multi-GPUs
- Aya Model is Supported.
- Falcon Model is Updated and now Falcon11B is supported wih flash attention support.
- `pallas_flash` is now available for CPU/GPU/TPU with custom pallas kernel.
- DeepseekV2 Model is Added (beta mood).
- OpenELM Model is Added.
- EasyDeL project structure has changed now you have to import EasyDel as `easydel`.
- `ORPOTrainer` is Added
- Phi3 Model bugs are fixed, Arctic Model is added.

> [!TIP]
>
> use `ed.AttentionModule.test_attentions()` to find the best attention mechanism
> that works for you
> ```python
> import easydel as ed
> ed.AttentionModule.test_attentions()
> ```

## Documentation 💫

> [!IMPORTANT]
> Documents and Examples are ready at [Here](https://easydel.readthedocs.io/en/latest/)
> Please have that in mind that EasyDeL is in the loop of fast-development
> so we might have API changes.

### Hands on Code Kaggle Examples

1. [script](https://www.kaggle.com/citifer/easydel-causal-language-model-trainer-example) for mindset of using EasyDeL
   CausalLanguageModelTrainer on kaggle, but you can do much more.
2. [script](https://www.kaggle.com/code/citifer/easydel-sfttrainer-example) SuperVised Finetuning with EasyDeL.

## Serving and Generation

### EasyDeL Generation Pipeline: Your Guide to Text Generation with JAX

The `GenerationPipeline` class in EasyDeL provides a streamlined interface for generating text using pre-trained
language
models within the JAX framework that support token streaming option. This introduction will guide you through its
purpose, potential applications, and basic usage.

#### What it Does:

At its core, the GenerationPipeline takes your input text (provided as `input_ids` and optionally an `attention_mask`)
and
uses a pre-trained language model to predict the most likely following tokens. This process is repeated iteratively,
generating new text one token at a time, until a stopping condition is met (e.g., reaching a maximum length or
encountering a special end-of-sequence token).

**Here's how it works:**

1. **Initialization:**
    - You provide a pre-trained `EasyDeLFlaxPretrainedModel`, typically an instance
      of `EasyDeLFlaxPretrainedModelForCausalLM`.
    - You provide the corresponding model parameters (`params`).
    - A `PreTrainedTokenizer` instance handles tokenization, ensuring compatibility between your text and the model.
    - Optionally, you can customize generation behavior using a `GenerationPipelineConfig` object.

2. **Generating Text:**
    - Call the `generate` method with your input text represented as `input_ids` and an optional `attention_mask`.
    - The pipeline iteratively generates new tokens, extending the input sequence.
    - You can either receive each generated token as it's produced or use a `TextIteratorStreamer` to handle streaming
      output.

**Example Usage:**

```python
import easydel as ed
from transformers import AutoTokenizer
from jax import numpy as jnp

# Load your pre-trained model and tokenizer
model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# Create a GenerationPipeline
pipeline = ed.GenerationPipeline(model=model, params=params, tokenizer=tokenizer)

# Prepare your input
input_text = "The quick brown fox jumps over the "

```

**Key Points:**

- **Input Format:** The `generate` method expects `input_ids` (numerical representation of tokens) and optionally
  an `attention_mask` to specify relevant input positions.
- **Output Handling:** You can either iterate over individual generated tokens or employ a `TextIteratorStreamer` for
  streaming output.
- **Customization:** Tailor the generation process with options like `max_new_tokens`, `temperature`, `top_k` sampling,
  and more using the `GenerationPipelineConfig`.

The `GenerationPipeline` offers a user-friendly interface to harness the power of EasyDeL's language models for a wide
range of text generation applications.


> [!NOTE]
> you can use `EasyDeLServeEngine` which is a Serve API Engine for production purpose sice that's more stable provide
> versioned
> API and efficient.

## EasyDeLState A Snapshot of Your EasyDeL Model

The `EasyDeLState` class acts like a comprehensive container that holds all the essential information about your EasyDeL
model at a given point in time. Think of it as a snapshot of your model. It includes:

* **Training Progress:**
    * `step`: Tracks the current training step.
* **Model Itself:**
    * `module`:  Holds the actual instance of your EasyDeL model.
    * `module_config`: Stores the model's configuration settings.
    * `module_config_args`:  Keeps track of arguments used to create the configuration (useful for reloading).
    * `apply_fn`:  References the core function that applies your model to data.
* **Learned Parameters:**
    * `params`: Contains the trained weights and biases of your model.
* **Optimizer Information:**
    * `tx`: Stores the optimizer you're using to update the model's parameters (e.g., AdamW).
    * `opt_state`: Keeps track of the optimizer's internal state (this is important for things like momentum in
      optimizers).
    * `tx_init`: Remembers the initial settings used to create the optimizer (again, for reloading purposes).
* **Additional Settings:**
    * `hyperparameters`:  Provides a flexible place to store other hyperparameters related to your model or training
      process.

**Key Capabilities of EasyDeLState:**

* **Initialization (`create`)**: Lets you create a brand new `EasyDeLState` to start training.
* **Loading (`load`, `load_state`, `from_pretrained`)**: Enables you to reload a saved model from a checkpoint file or
  even a pre-trained model from a repository like Hugging Face Hub.
* **Saving (`save_state`)**: Allows you to save your model's current state, including its parameters and optimizer
  state.
* **Optimizer Management (`apply_gradients`, `free_opt_state`, `init_opt_state`)**: Provides methods for updating the
  model's parameters using gradients, releasing optimizer memory, and re-initializing the optimizer if needed.
* **Sharding (`shard_params`)**:  Helps you distribute your model's parameters efficiently across multiple devices (
  important for training large models).
* **PyTorch Conversion (`to_pytorch`)**:  Gives you a way to convert your EasyDeL model to its PyTorch equivalent.

**In Essence:**

`EasyDeLState` streamlines the process of managing, saving, loading, and even converting your EasyDeL models. It ensures
that you can easily work with your models and maintain consistency throughout your machine learning workflow.

## Supervised Fine-Tuning with EasyDeL

EasyDeL supports both DPO and SFT Trainers, so dealing with LLMs in jax is a lot easier right now
let have an example of using Supervised Fine-Tuner in JAX with EasyDeL

```python
import jax.lax
from easydel import (
    TrainArguments,
    AutoEasyDeLModelForCausalLM,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    EasyDeLGradientCheckPointers,
    SFTTrainer,
    PartitionAxis,
    conversations_formatting_function  # i have added this one for newcomers so if they 
    # don't know what's going on they can use this pre created prompter
)
from datasets import load_dataset
import flax
from jax import numpy as jnp
from transformers import AutoTokenizer

huggingface_repo_id_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

max_length = 4096
dtype = jnp.bfloat16
input_shape = (1, 1)
partition_axis = PartitionAxis()
sharding_axis_dims = (1, -1, 1, 1)  # Change to 1,1,1,-1 for Sequence Sharding
model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
    huggingface_repo_id_or_path,
    dtype=dtype,
    param_dtype=dtype,
    precision=jax.lax.Precision("fastest"),
    auto_shard_params=True,
    sharding_axis_dims=sharding_axis_dims,
    verbose_params=True,
    config_kwargs=dict(
        use_scan_mlp=False,
        partition_axis=partition_axis
    ),
    partition_axis=partition_axis,
)

tokenizer = AutoTokenizer.from_pretrained(
    huggingface_repo_id_or_path,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
configs_to_initialize_model_class = {
    "config": model.config,
    "dtype": dtype,
    "param_dtype": dtype,
    "input_shape": input_shape
}

train_arguments = TrainArguments(
    model_class=type(model),
    model_name="SFT-EasyDeL",
    num_train_epochs=3,
    configs_to_initialize_model_class=configs_to_initialize_model_class,
    learning_rate=5e-5,
    learning_rate_end=1e-6,
    optimizer=EasyDeLOptimizers.ADAMW,
    scheduler=EasyDeLSchedulers.WARM_UP_COSINE,
    weight_decay=0.01,
    total_batch_size=32,
    max_training_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
    max_sequence_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=sharding_axis_dims,
    # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    remove_ckpt_after_load=True,
    gradient_accumulation_steps=8,
    loss_re_mat="",
    dtype=dtype,
    param_dtype=dtype,
    init_input_shape=input_shape,
)


def prompter(sample):
    return [conversations_formatting_function(tokenizer, messages_field="messages")(sample)]


train_dataset = load_dataset("HuggingFaceH4/deita-10k-v0-sft", split="train_sft")
trainer = SFTTrainer(
    arguments=train_arguments,
    train_dataset=train_dataset,
    eval_dataset=None,  # we don't have eval dataset rn :)
    tokenizer=tokenizer,
    dataset_text_field=None,
    formatting_func=prompter,
    packing=True,
    num_of_sequences=max_length,
)

output = trainer.train(flax.core.FrozenDict({"params": params}))
print(f"Hey ! , here's where your model saved {output.checkpoint_path}")
```

> [!NOTE]
> You Can use Lora too, for DPO, ORPO and SFT Trainers.

## FineTuning

with using EasyDeL FineTuning LLM (CausalLanguageModels) are easy as much as possible with using Jax and Flax
and having the benefit of TPUs for the best speed here's a simple code to use in order to finetune your
own Model

Days Has Been Passed and now using easydel in Jax is way more similar to HF/PyTorch Style
now it's time to finetune our model

```python
import jax.lax
from easydel import (
    TrainArguments,
    CausalLanguageModelTrainer,
    AutoEasyDeLModelForCausalLM,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    EasyDeLGradientCheckPointers,
    PartitionAxis
)
from datasets import load_dataset
import flax
from jax import numpy as jnp
from transformers import AutoTokenizer

huggingface_repo_id_or_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

max_length = 4096
dtype = jnp.bfloat16
input_shape = (1, 1)
partition_axis = PartitionAxis()
sharding_axis_dims = (1, -1, 1, 1)  # Change to 1,1,1,-1 for Sequence Sharding
model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
    huggingface_repo_id_or_path,
    dtype=dtype,
    param_dtype=dtype,
    precision=jax.lax.Precision("fastest"),
    auto_shard_params=True,
    sharding_axis_dims=sharding_axis_dims,
    verbose_params=True,
    config_kwargs=dict(
        use_scan_mlp=False,
        partition_axis=partition_axis
    ),
    partition_axis=partition_axis,
)

tokenizer = AutoTokenizer.from_pretrained(
    huggingface_repo_id_or_path,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
configs_to_initialize_model_class = {
    "config": model.config,
    "dtype": dtype,
    "param_dtype": dtype,
    "input_shape": input_shape
}

train_arguments = TrainArguments(
    model_class=type(model),
    model_name="my_first_model_to_train_using_easydel",
    num_train_epochs=3,
    configs_to_initialize_model_class=configs_to_initialize_model_class,
    learning_rate=5e-5,
    learning_rate_end=1e-6,
    optimizer=EasyDeLOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
    scheduler=EasyDeLSchedulers.LINEAR,
    # "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
    weight_decay=0.01,
    total_batch_size=64,
    max_training_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
    max_sequence_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=sharding_axis_dims,
    # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    remove_ckpt_after_load=True,
    gradient_accumulation_steps=8,
    loss_re_mat="",
    dtype=dtype,
    param_dtype=dtype,
    init_input_shape=input_shape,
)


def ultra_chat_prompting_process(
        data_chunk
):
    user_part = [
        chunk["content"] for chunk in data_chunk["messages"] if chunk["role"] == "user"
    ]
    assistant_part = [
        chunk["content"] for chunk in data_chunk["messages"] if chunk["role"] == "assistant"
    ]

    prompt = ""

    for uc, ac in zip(user_part, assistant_part):
        prompt += f"<|user|>\n{uc}</s>\n<|assistant|>\n{ac}</s>\n"

    return {"prompt": prompt}


tokenization_process = lambda data_chunk: tokenizer(
    data_chunk["prompt"],
    add_special_tokens=False,
    max_length=max_length,
    padding="max_length"
)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
dataset_train = dataset["train_gen"].map(ultra_chat_prompting_process, num_proc=12)
dataset_train = dataset_train.map(
    tokenization_process,
    num_proc=12,
    remove_columns=dataset_train.column_names
)

# you can do the same for evaluation process dataset

trainer = CausalLanguageModelTrainer(
    train_arguments,
    dataset_train,
    checkpoint_path=None
)

output = trainer.train(flax.core.FrozenDict({"params": params}))
print(f"Hey ! , here's where your model saved {output.checkpoint_path}")
```

> [!TIP]
> you can then convert it to pytorch for better use I don't recommend jax/flax for hosting models since
> pytorch is better option for gpus

## DPO Fine-tuning

`DPOTrainer` is the new Trainer in EasyDeL, so you might have except some bugs in process but as far as i have tested
everything works just fine, and you can consider it the first DPO Trainer in JAX/Flax let have an example and see how
you can fine-tune your own model with DPOTrainer

> [!TIP]
> In case that you want a better script to learn about `DPOTrainer` you can see examples
> at [here](https://github.com/erfanzar/EasyDeL/blob/main/examples/training/dpo/dpo_training_example.py) which contain
> DPO Tuning a Mixtral model with Intel DPO dataset.

```python
import easydel
from easydel import (
    TrainArguments,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    EasyDeLGradientCheckPointers,
    DPOTrainer,
    EasyDeLState,
    easystate_to_huggingface_model
)

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, LlamaForCausalLM as module_pt
from jax import numpy as jnp
import jax
from jax.sharding import PartitionSpec
from fjformer import GenerateRNG
from typing import Optional, Dict
from datasets import Dataset

rng_g = GenerateRNG()
api = HfApi()

max_length = 512  # Overall maximum length
max_target_length = 1024  # Maximum Length for target column in Dataset
max_prompt_length = 1024  # Maximum Length for prompt column in Dataset

model_name_or_path = "erfanzar/LinguaMatic-Tiny"
ref_model_name_or_path = "teknium/OpenHermes-2.5-Mistral-7B"
dtype = jnp.bfloat16

sharding_axis_dims = (1, -1, 1, 1)
sharding_axis_names = ("dp", "fsdp", "tp", "sp")


def extract_anthropic_prompt(prompt_and_response):
    """
    Extract the anthropic prompt from a prompt and response pair.
    """
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }

    return dataset.map(split_prompt_and_responses)


arguments = TrainArguments(
    model_name="EasyDeL-DPO",
    num_train_epochs=5,
    learning_rate=1e-4,
    learning_rate_end=3e-5,
    warmup_steps=200,
    optimizer=EasyDeLOptimizers.ADAMW,
    scheduler=EasyDeLSchedulers.LINEAR,
    weight_decay=0.02,
    total_batch_size=128,
    gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=sharding_axis_dims,
    fully_sharded_data_parallel=True,
    gradient_accumulation_steps=2,
    dtype=dtype,
    param_dtype=dtype,
    step_start_point=0,
    training_time="7H",
    do_train=True,
    do_eval=True,
    track_memory=False  # Performance boost.
    # You can set other options too or play with them but for now I just stick with these arguments.
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

train_dataset = get_hh("train", sanity_check=True)
eval_dataset = get_hh("test", sanity_check=True)

state = EasyDeLState.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    dtype=dtype,
    param_dtype=dtype,
    init_optimizer_state=False,
    free_optimizer_state=True,
    sharding_axis_dims=sharding_axis_dims,
    sharding_axis_names=sharding_axis_names,
    partition_axis=easydel.PartitionAxis(
        batch_axis=("dp", "fsdp"),
        query_sequence_axis="sp",
        key_sequence_axis="sp",
        head_axis="tp",
        attention_dim_axis=None
    )
)

ref_state = EasyDeLState.from_pretrained(
    pretrained_model_name_or_path=ref_model_name_or_path,
    dtype=dtype,
    param_dtype=dtype,
    init_optimizer_state=False,
    free_optimizer_state=True,
    sharding_axis_dims=sharding_axis_dims,
    sharding_axis_names=sharding_axis_names,
    partition_axis=easydel.PartitionAxis(
        batch_axis=("dp", "fsdp"),
        query_sequence_axis="sp",
        key_sequence_axis="sp",
        head_axis="tp",
        attention_dim_axis=None
    )
)

dpo_trainer = DPOTrainer(
    model_state=state,
    ref_model_state=ref_state,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    arguments=arguments,
    max_length=max_length,
    max_target_length=max_target_length,
    max_prompt_length=max_prompt_length,
    ref_model_init_kwargs=None,  # In case that you pass the ref_model_state a string you have to pass this one too
    model_init_kwargs=None,  # In case that you pass the model_state a string you have to pass this one too
    dataset_map_arguments={
        "num_proc": 8,
        "batched": True,
        "batch_size": 100,
    },
    auto_shard_model_state=True,
    auto_shard_ref_model_state=True,
    loss_type="sigmoid",
    data_collator=None,  # Pass None in order to use default data_collector (you can create your own)
)

output = dpo_trainer.train()

easydel_jax_model = output.state  # Here's you EasyDeL Model

with jax.default_device(jax.devices("cpu")[0]):
    model = easystate_to_huggingface_model(
        state=EasyDeLState.load_state(
            output.checkpoint_path
        ),
        base_huggingface_module=module_pt,
        config=dpo_trainer.model_state.module.config
    )  # Here's you PyTorch Model

model.push_to_hub("<REPO_ID>", private=False)  # Hope you love open-source too :)
tokenizer.push_to_hub("<REPO_ID>", private=False)  # Hope you love open-source too :)
```

now you have trained your first model Using DPOTrainer in JAX with EasyDeL.

> [!TIP]
> The API of EasyDeL DPO Trainer is similar to DPO Trainer in TRL from HuggingFace so that means
> you have freedom and have access to a hackable and changeable code.

## EasyDeLState

EasyDeLState is new and cool feature in EasyDeL and have a lot of options like
storing `Model Parameters`, _Optimizer State,
Model Config, Model Type, Optimizer and Scheduler Configs_

Let see and examples of using EasyDeLState

### Fine-tuning

Fine-tuning from a previous State or a new state

```python
from easydel import (
    AutoEasyDeLConfig,
    EasyDeLState,
    PartitionAxis
)
from transformers import AutoTokenizer
from jax import numpy as jnp, lax
import jax

huggingface_model_repo_id = "REPO_ID"
checkpoint_name = "CKPT_NAME"

state = EasyDeLState.from_pretrained(
    pretrained_model_name_or_path=huggingface_model_repo_id,
    filename=checkpoint_name,
    optimizer="adamw",
    scheduler="none",
    tx_init=None,
    device=jax.devices('cpu')[0],  # Offload Device
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=lax.Precision("fastest"),
    sharding_axis_dims=(1, -1, 1, 1),
    # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    sharding_axis_names=("dp", "fsdp", "tp", "sp"),
    partition_axis=PartitionAxis(
        batch_axis=("dp", "fsdp"),
        query_sequence_axis="sp",
        key_sequence_axis="sp",
        head_axis="tp",
        attention_dim_axis=None
    ),
    shard_attention_computation=True,
    input_shape=(1, 1),
    backend=None,
    init_optimizer_state=False,
    free_optimizer_state=True,
    verbose=True,
    state_shard_fns=None,
)

config = AutoEasyDeLConfig.from_pretrained(
    huggingface_model_repo_id
)

tokenizer = AutoTokenizer.from_pretrained(
    huggingface_model_repo_id,
    trust_remote_code=True
)

max_length = config.max_position_embeddings

configs_to_initialize_model_class = {
    'config': config,
    'dtype': jnp.bfloat16,
    'param_dtype': jnp.bfloat16,
    'input_shape': (8, 8)
}
```

`EasyDeLState` also has `.load_state()` and `.save_state()` with some other usable options like `.free_opt_state()`
which
free optimizer state or `.shard_params()` which shard parameters you can read docs in order to find out more about these
options.

### Converting to Huggingface and Pytorch

Let see how you can convert a EasyDeLMistral Model to Huggingface Pytorch Mistral Model from a trained State

```python

from transformers import MistralForCausalLM
from easydel import (
    AutoEasyDeLConfig,
    EasyDeLState,
    easystate_to_huggingface_model
)
import jax

huggingface_model_repo_id = "REPO_ID"

config = AutoEasyDeLConfig.from_pretrained(
    huggingface_model_repo_id
)
with jax.default_device(jax.devices("cpu")[0]):
    model = easystate_to_huggingface_model(
        state=EasyDeLState.load_state(
            "PATH_TO_CKPT"
        ),  # You can Pass EasyDeLState here
        base_huggingface_module=MistralForCausalLM,  # type: ignore
        config=config
    )

model = model.half()  # it's a huggingface model now
```

### Other Use Cases

`EasyDeLState` have a general use you can use it everywhere in easydel for example for a stand-alone model
, serve, fine-tuning and many other features, it's up to you to test how creative you are 😇.

## AttentionModule: A Versatile Attention Mechanism Factory

The `AttentionModule` class is designed to simplify the creation and execution of different attention mechanisms within
your EasyDeL models. It provides a unified interface for working with various attention types, allowing you to easily
switch between them and experiment with different configurations.

**Key Features:**

* **Mechanism Selection:** The `attn_mechanism` argument lets you choose the specific attention algorithm you want to
  use (e.g., "vanilla," "flash," "splash," "ring," "cudnn").
* **Sharding and Partitioning:** The class supports advanced JAX sharding techniques to distribute attention
  computations across multiple devices for efficient processing of large models. It handles partitioning of query, key,
  value, bias, and attention weight matrices using `PartitionSpec`.
* **Blockwise Attention:** Enables the use of blockwise attention for increased memory efficiency, especially with long
  sequences.
* **Caching Support:** Facilitates the use of attention caching to speed up inference and generation tasks.
* **Dropout and Determinism:** Allows for applying dropout to attention weights and controlling the deterministic
  behavior of the attention computation.
* **Testing Utility:**  Provides a `test_attentions` method to compare different attention mechanisms in terms of
  accuracy, gradient stability, and computation time.

**How it Works:**

1. **Initialization:**
    - During initialization, you provide the desired `attn_mechanism`, JAX `mesh` for sharding, scaling
      factor (`sm_scale`), number of attention heads, head dimensions, and other configuration parameters.
    - The class automatically sets default values for many parameters based on the chosen attention mechanism and the
      provided EasyDeL configuration (`base_module_class`).
2. **Calling the Module:**
    - When you call the `AttentionModule` object, you pass in the query, key, and value states, along with optional
      parameters like attention masks, biases, and causal flags.
    - The module internally selects the appropriate attention function based on the specified `attn_mechanism`.
    - It performs any necessary sharding and partitioning based on the configured partition specifications.
    - The attention computation is executed, and the attention outputs (and optionally attention weights) are returned.

**Advantages:**

* **Flexibility:**  Allows you to easily switch between different attention mechanisms without major code changes.
* **Efficiency:**  Supports advanced JAX sharding for distributed computation, enabling the handling of large models.

_Flash Attention works on TPU with ease but for gpu there are still some improvements in process._

## EasyDeLXRapTure for layer tuning and LoRA

in case of using LoRA and applying that on the EasyDeL models there are some other things
that you might need to config on your own but a lot of things being handled by EasyDeL so let just jump into an example
for LoRA fine-tuning section and use _EasyDeLXRapTure_ in for mistral models with flash attention example

```python
from flax.core import FrozenDict
from easydel import (
    TrainArguments,
    CausalLanguageModelTrainer,
    AutoEasyDeLModelForCausalLM,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    EasyDeLGradientCheckPointers,
    EasyDeLXRapTureConfig
)
from datasets import load_dataset
import flax
from jax import numpy as jnp
from transformers import AutoTokenizer

huggingface_repo_id_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

model, params = AutoEasyDeLModelForCausalLM.from_pretrained(huggingface_repo_id_or_path, )

max_length = 8196
model_parameters = FrozenDict({"params": params})

dtype = jnp.bfloat16
param_dtype = jnp.bfloat16  # you can change that if you want 

tokenizer = AutoTokenizer.from_pretrained(
    huggingface_repo_id_or_path,
    trust_remote_code=True
)

model.config.add_basic_configurations(
    attn_mechanism="flash",  # Using FlashAttention
    block_b=1,
    block_q=1024,
    block_k=1024,
    block_k_major=1024,
)

tokenizer.pad_token = tokenizer.eos_token
configs_to_initialize_model_class = {
    "config": model.config,
    "dtype": dtype,
    "param_dtype": param_dtype,
    "input_shape": (1, 1)
}

rapture = EasyDeLXRapTureConfig(
    parameters=model_parameters,
    lora_dim=64,
    fully_fine_tune_parameters=["embed_tokens"],  # Model layer to be fully fine tuned
    lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj", "o_proj"],  # LoRA Layer Targets you can pass this to none
    # For only Layer Tuning or transfer learning
    verbose=True
)

train_arguments = TrainArguments(
    model_class=type(model),
    model_name="EasyDeL-Lora-Example",
    num_train_epochs=3,
    configs_to_initialize_model_class=configs_to_initialize_model_class,
    learning_rate=1e-4,  # Using higher learning rate is recommended
    learning_rate_end=8e-5,
    optimizer=EasyDeLOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
    scheduler=EasyDeLSchedulers.LINEAR,
    # "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
    weight_decay=0.01,
    total_batch_size=512,
    max_training_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
    max_sequence_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=(1, -1, 1, 1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    remove_ckpt_after_load=True,
    gradient_accumulation_steps=1,
    loss_re_mat="",
    dtype=dtype,
    param_dtype=param_dtype,
    rapture_config=rapture,
    merge_lora_rapture_parameters=True  # turning this off is still not supported and not recommended to do so
    # What this does ? this will merge the lora parameters with the original model parameters and the end of training
)


def ultra_chat_prompting_process(
        data_chunk
):
    user_part = [
        chunk["content"] for chunk in data_chunk["messages"] if chunk["role"] == "user"
    ]
    assistant_part = [
        chunk["content"] for chunk in data_chunk["messages"] if chunk["role"] == "assistant"
    ]

    prompt = ""

    for uc, ac in zip(user_part, assistant_part):
        prompt += f"<|user|>\n{uc}</s>\n<|assistant|>\n{ac}</s>\n"

    return {"prompt": prompt}


tokenization_process = lambda data_chunk: tokenizer(
    data_chunk["prompt"],
    add_special_tokens=False,
    max_length=max_length,
    padding="max_length"
)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
dataset_train = dataset["train_gen"].map(ultra_chat_prompting_process, num_proc=12)
dataset_train = dataset_train.map(
    tokenization_process,
    num_proc=12,
    remove_columns=dataset_train.column_names
)

# you can do the same for evaluation process dataset

trainer = CausalLanguageModelTrainer(
    train_arguments,
    dataset_train,
    checkpoint_path=None
)

output = trainer.train()  # you should not pass the parameters in Trainer.train anymore when
# you are using LoRA or transfer Learning
print(f"Hey ! , here's where your model saved {output.checkpoint_path}")
```

## Contributing

EasyDeL is an open-source project, and contributions are welcome. If you would like to contribute to EasyDeL, please
fork the repository, make your changes, and submit a pull request. The team behind EasyDeL will review your changes and
merge them if they are suitable.

## License 📜

EasyDeL is a Fully Open-Source released under the Apache v2 license. Please see the LICENSE file in the root directory
of this project for
more information.

## Contact

If you have any questions or comments about EasyDeL, you can reach out to me

## Citing EasyDeL 🥶

To cite this repository:

```misc
@misc{Zare Chavoshi_2023,
    title={EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training Flax/Jax Models on TPU/GPU for both Serving and Training purposes.},
    url={https://github.com/erfanzar/EasyDeL},
    journal={EasyDeL Easy and Fast DeepLearning with JAX},
    publisher={Erfan Zare Chavoshi},
    author={Zare Chavoshi, Erfan},
    year={2023}
} 
```

## Refrences

- _[young-geng/EasyLM](https://github.com/young-geng/EasyLM)_: Large language models (LLMs) made easy, EasyLM is a one
  stop solution for pre-training, finetuning, evaluating and serving LLMs in JAX/Flax.
- _[Sea-Snell/JAXSeq](https://github.com/Sea-Snell/JAXSeq)_: Train very large language models in Jax.

- _[lhao499/RingAttention](https://github.com/lhao499/RingAttention)_:Transformers with Arbitrarily Large Context