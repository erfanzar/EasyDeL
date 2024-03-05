# EasyDeL 🔮

EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine
learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training
Flax/Jax Models on TPU/GPU for both Serving and Training purposes. Additionally, EasyDeL will support mojo and will be
rewritten for mojo as well.

Some of the key features provided by EasyDeL include:

- Serving and API Engines for Using and serving LLMs in JAX as efficiently as possible.
- Support for 8, 6, and 4 BIT inference and training in JAX
- A wide range of models in Jax is supported which have never been implemented before such as Falcon, Qwen2, Phi2,
  Mixtral, and MPT ...
- Integration of flashAttention in JAX for GPUs and TPUs
- Automatic serving of LLMs with mid and high-level APIs in both JAX and PyTorch
- LLM Trainer and fine-tuner in JAX
- Video CLM Trainer and Fine-tunerFalcon, Qwen2, Phi2, Mixtral, and MPT ...
- RLHF (Reinforcement Learning from Human Feedback) in Jax (Beta Stage)
- DPOTrainer(Supported) and SFTTrainer(Developing Stage)
- Various other features to enhance the training process and optimize performance.
- LoRA: Low-Rank Adaptation of Large Language Models
- RingAttention, Flash Attention, BlockWise FFN, and Efficient Attention are supported for more than 90 % of models
  ([FJFormer](https://github.com/erfanzar/FJFormer) Backbone).
- Serving and API Engines for Using and serving LLMs in JAX as efficient as possible.
- Automatic Converting Models from JAX-EasyDeL to PyTorch-HF and reverse

> [!NOTE]
> In case of using EasyDel on Kaggle TPUs you need to upgrade tensorflow version too.

> [!NOTE]
> EasyDel Will only support JAX>=0.4.22 Due to miss computations being happened in older version and also not bein able
> to run Flash Attention and Splash Attention (Splash Attention is still under Process)

> [!NOTE]
> These features collectively aim to simplify and accelerate the training of machine learning models, making it more
> efficient and accessible for developers working with Jax/Flax.

## Documentation 💫

> [!IMPORTANT]
> Documents and Examples are ready at [Here](https://erfanzar.github.io/EasyDeL)
> Please have that in mind that EasyDel is in the loop of fast-development
> so we might have API changes

## Serving

you can read docs or examples to see how `JAXServer` works but let me show you how you can simply host and serve any
model that supported by `EasyDeL` fo this example ill just use `gemma-7-it` by google but you can use any model as you
wish.

```shell
python -m examples.jax_serve_example \
  --prompter_type="gemma" \ 
  --share_gradio=True \ 
  --sharding_axis_dims=1,1,1,-1 \
  --attn_mechanism="normal" \
  --scan_ring_attention=True \
  --max_sequence_length=8192 \ 
  --max_new_tokens_ratio=25 \
  --max_compile_tokens=256 \ 
  --block_k=128 \
  --block_q=128 \
  --pretrained_model_name_or_path="google/gemma-7b-it" \
  --dtype="bf16"
```

> [!NOTE]
> you can use `EasyServe` which is a Serve API Engine for production purpose sice that's more stable provide versioned
> API and efficient.

## RLHF(Reinforcement Learning From Human Feedback)

> RLHF or Reinforcement Learning From Human Feedback is Available At the moment, but it's still
> under heavy development , because I don't have enough experience with Reinforcement Learning at the moment so its
> still in beta version but it's works and ill soon release a Tutorial For that

## FineTuning

with using EasyDel FineTuning LLM (CausalLanguageModels) are easy as much as possible with using Jax and Flax
and having the benefit of TPUs for the best speed here's a simple code to use in order to finetune your
own Model

Days Has Been Passed and now using easydel in Jax is way more similar to HF/PyTorch Style
now it's time to finetune our model

```python
import jax.numpy
from EasyDel import (
    TrainArguments,
    CausalLanguageModelTrainer,
    AutoEasyDelModelForCausalLM,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers
)
from datasets import load_dataset
import flax
from jax import numpy as jnp
from transformers import AutoTokenizer

huggingface_repo_id_or_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

model, params = AutoEasyDelModelForCausalLM.from_pretrained(huggingface_repo_id_or_path, )

max_length = 2048
tokenizer = AutoTokenizer.from_pretrained(
    huggingface_repo_id_or_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
configs_to_initialize_model_class = {
    "config": model.config,
    "dtype": jnp.bfloat16,
    "param_dtype": jnp.bfloat16,
    "input_shape": (1, 1)
}

train_arguments = TrainArguments(
    model_class=type(model),
    model_name="my_first_model_to_train_using_easydel",
    num_train_epochs=3,
    configs_to_initialize_model_class=configs_to_initialize_model_class,
    learning_rate=5e-5,
    learning_rate_end=1e-6,
    optimizer=EasyDelOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
    scheduler=EasyDelSchedulers.LINEAR,
    # "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
    weight_decay=0.01,
    total_batch_size=64,
    max_training_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
    max_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=(1, 1, 1, -1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, 1, 1, -1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    use_pjit_attention_force=False,
    remove_ckpt_after_load=True,
    gradient_accumulation_steps=8,
    loss_re_mat="",
    dtype=jnp.bfloat16
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

## LLMServe

To use EasyDeL in your project, you will need to import the library in your Python script and use its various functions
and classes. Here is an example of how to import EasyDeL and use its Model class:

```python
from EasyDel.modules import AutoEasyDelModelForCausalLM
from EasyDel.serve import JAXServer
from transformers import AutoTokenizer
import jax

model_huggingface_repo_id = "meta-llama/Llama.md-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_huggingface_repo_id, trust_remote_code=True)
model, params = AutoEasyDelModelForCausalLM.from_pretrained(
    model_huggingface_repo_id,
    jax.devices("cpu")[0],
    jax.numpy.float16,
    jax.numpy.float16,
    jax.lax.Precision("fastest"),
    (1, 1, 1, -1),
    device_map="auto"
)

server = JAXServer.from_parameters(
    model=model,
    config_model=model.config,
    tokenizer=tokenizer,
    params=model.params,
    add_params_field=True
)

response_printed = 0
for response, tokens_used in server.process(
        "String To The Model", stream=True
):
    print(response[response_printed:], end="")
    response_printed = len(response)
```

## DPO Fine-tuning

`DPOTrainer` is the new Trainer in EasyDeL, so you might have except some bugs in process but as far as i have tested
everything works just fine, and you can consider it the first DPO Trainer in JAX/Flax let have an example and see how
you can fine-tune your own model with DPOTrainer

> [!TIP]
> In case that you want a better script to learn about `DPOTrainer` you can see examples
> at [here](https://github.com/erfanzar/EasyDeL/blob/main/examples/training/dpo/dpo_training_example.py) which contain
> DPO Tuning a Mixtral model with Intel DPO dataset.

```python
from EasyDel import (
    TrainArguments,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    DPOTrainer,
    EasyDelState,
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

sharding_axis_dims = (1, 1, 1, -1)
sharding_axis_names = ("dp", "fsdp", "tp", "sp")
query_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Query Partition Spec for Model
key_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Key Partition Spec for Model
value_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Value Partition Spec for Model
bias_partition_spec = PartitionSpec(
    ("dp", "fsdp"), None, None, None
)  # Attention Mask / Bias Partition Spec for Model
attention_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Attention Score / Weight Partition Spec for Model

ref_model_query_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Query Partition Spec for Ref Model
ref_model_key_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Key Partition Spec for Ref Model
ref_model_value_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Value Partition Spec for Ref Model
ref_model_bias_partition_spec = PartitionSpec(
    ("dp", "fsdp"), None, None, None
)  # Attention Mask / Bias Partition Spec for Ref Model
ref_model_attention_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Attention Score / Weight Partition Spec for Ref Model


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
    optimizer=EasyDelOptimizers.ADAMW,
    scheduler=EasyDelSchedulers.LINEAR,
    weight_decay=0.02,
    total_batch_size=128,
    gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
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

state = EasyDelState.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    dtype=dtype,
    param_dtype=dtype,
    init_optimizer_state=False,
    free_optimizer_state=True,
    sharding_axis_dims=sharding_axis_dims,
    sharding_axis_names=sharding_axis_names,
    query_partition_spec=query_partition_spec,
    key_partition_spec=key_partition_spec,
    value_partition_spec=value_partition_spec,
    bias_partition_spec=bias_partition_spec,
    attention_partition_spec=attention_partition_spec,
)

ref_state = EasyDelState.from_pretrained(
    pretrained_model_name_or_path=ref_model_name_or_path,
    dtype=dtype,
    param_dtype=dtype,
    init_optimizer_state=False,
    free_optimizer_state=True,
    sharding_axis_dims=sharding_axis_dims,
    sharding_axis_names=sharding_axis_names,
    query_partition_spec=ref_model_query_partition_spec,
    key_partition_spec=ref_model_key_partition_spec,
    value_partition_spec=ref_model_value_partition_spec,
    bias_partition_spec=ref_model_bias_partition_spec,
    attention_partition_spec=ref_model_attention_partition_spec,
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
        state=EasyDelState.load_state(
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

## EasyDelState

EasyDelState is new and cool feature in EasyDeL and have a lot of options like
storing `Model Parameters`, _Optimizer State,
Model Config, Model Type, Optimizer and Scheduler Configs_

Let see and examples of using EasyDelState

### Fine-tuning

Fine-tuning from a previous State or a new state

```python
from EasyDel import (
    AutoEasyDelConfig,
    EasyDelState
)
from transformers import AutoTokenizer
from jax import numpy as jnp, lax
import jax

huggingface_model_repo_id = "REPO_ID"
checkpoint_name = "CKPT_NAME"

state = EasyDelState.from_pretrained(
    pretrained_model_name_or_path=huggingface_model_repo_id,
    filename=checkpoint_name,
    optimizer="adamw",
    scheduler="none",
    tx_init=None,
    device=jax.devices('cpu')[0],  # Offload Device
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=lax.Precision("fastest"),
    sharding_axis_dims=(1, 1, 1, -1),
    # the way to shard model across gpu,cpu or TPUs using sharding array (1, 1, 1, -1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    sharding_axis_names=("dp", "fsdp", "tp", "sp"),
    query_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    key_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    value_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    bias_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None),
    attention_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    use_shard_map=False,
    input_shape=(1, 1),
    backend=None,
    init_optimizer_state=False,
    free_optimizer_state=True,
    verbose=True,
    state_shard_fns=None,
)

config = AutoEasyDelConfig.from_pretrained(
    huggingface_model_repo_id
)

config.use_pjit_attention_force = False

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

`EasyDelState` also has `.load_state()` and `.save_state()` with some other usable options like `.free_opt_state()`
which
free optimizer state or `.shard_params()` which shard parameters you can read docs in order to find out more about these
options.

### Converting to Huggingface and Pytorch

Let see how you can convert a EasyDelMistral Model to Huggingface Pytorch Mistral Model from a trained State

```python

from transformers import MistralForCausalLM
from EasyDel import (
    AutoEasyDelConfig,
    EasyDelState,
    easystate_to_huggingface_model
)
import jax

huggingface_model_repo_id = "REPO_ID"

config = AutoEasyDelConfig.from_pretrained(
    huggingface_model_repo_id
)
with jax.default_device(jax.devices("cpu")[0]):
    model = easystate_to_huggingface_model(
        state=EasyDelState.load_state(
            "PATH_TO_CKPT"
        ),  # You can Pass EasyDelState here
        base_huggingface_module=MistralForCausalLM,
        config=config
    )

model = model.half()  # it's a huggingface model now
```

### Other Use Cases

`EasyDelState` have a general use you can use it everywhere in easydel for example for a stand-alone model
, serve, fine-tuning and many other features, it's up to you to test how creative you are 😇.

## Flash Attention and Splash Attention Are Here 🥵

here's a simple example about how can you use Flash Attention in EasyDeL

```python
# Config is built in config for every model (EasyDelPretrainedConfig)
config.add_basic_configurations(
    attn_mechanism="flash",  # flash , normal or splash (not fully supported yet on GPU,TPU) 
    block_b=1,
    block_q=512,
    block_k=512,
    block_k_major=512
)
```

_Flash Attention works on TPU with ease but for gpu there are still some improvements in process._

## EasyDeLXRapTure for layer tuning and LoRA

in case of using LoRA and applying that on the EasyDeL models there are some other things
that you might need to config on your own but a lot of things being handled by EasyDeL so let just jump into an example
for LoRA fine-tuning section and use _EasyDeLXRapTure_ in for mistral models with flash attention example

```python
from flax.core import FrozenDict
from EasyDel import (
    TrainArguments,
    CausalLanguageModelTrainer,
    AutoEasyDelModelForCausalLM,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    EasyDeLXRapTureConfig
)
from datasets import load_dataset
import flax
from jax import numpy as jnp
from transformers import AutoTokenizer

huggingface_repo_id_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

model, params = AutoEasyDelModelForCausalLM.from_pretrained(huggingface_repo_id_or_path, )

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
    optimizer=EasyDelOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
    scheduler=EasyDelSchedulers.LINEAR,
    # "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
    weight_decay=0.01,
    total_batch_size=512,
    max_training_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
    max_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=(1, 1, 1, -1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, 1, 1, -1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    use_pjit_attention_force=False,
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

EasyDeL is an Fully Open-Source released under the Apache v2 license. Please see the LICENSE file in the root directory
of this project for
more information.

## Contact

If you have any questions or comments about EasyDeL, you can reach out to me

## Citing EasyDeL 🥶

To cite this repository:

```misc
@misc{Zare Chavoshi_2023,
    title={EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training Flax/Jax Models on TPU/GPU for both Serving and Training purposes.},
    url={https://github.com/erfanzar/EasyDel},
    journal={EasyDeL Easy and Fast DeepLearning with JAX},
    publisher={Erfan Zare Chavoshi},
    author={Zare Chavoshi, Erfan},
    year={2023}
} 
```
