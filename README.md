# EasyDeL 🔮

EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine
learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training
Flax/Jax Models on TPU/GPU for both Serving and Training purposes. Additionally, EasyDeL will support mojo and will be
rewritten for mojo as well.

Some of the key features provided by EasyDeL include:

- Support for 8, 6, and 4 BIT inference and training in JAX
- Wide Range of models in Jax are supported which have never been implemented before such as _falcon, Qwen2, Phi2,
  MPT ..._
- Integration of flashAttention in JAX for GPUs and TPUs
- Automatic serving of LLMs with mid and high-level APIs in both JAX and PyTorch
- LLM Trainer and fine-tuner in JAX
- RLHF (Reinforcement Learning from Human Feedback) in Jax (Beta Stage)
- And various other features to enhance the training process and optimize performance.
- LoRA: Low-Rank Adaptation of Large Language Models

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

you can read docs or examples to see how `JAXServer` works but let me show you how you can simply host and serve a
LLama2
chat model (70B model is supported too)

```shell
python -m examples.serving.causal-lm.llama-2-chat \
  --pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --max_length=4096 \
  --max_new_tokens=2048 --max_compile_tokens=32 --temperature=0.6 \
  --top_p=0.95 --top_k=50 \
  --dtype="fp16" --use_prefix_tokenizer

```

> [!NOTE]
> you can use all the llama models not just "meta-llama/Llama-2-7b-chat-hf"
> float16 or float32 , bfloat16 are supported dtype and make sure to use --use_prefix_tokenizer,
> and you will get links or api to use model from gradio app chat/instruct or FastAPI apis

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
    sharding_array=(1, -1, 1, 1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
    # everything training will be in fully FSDP automatic and share data between devices
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
    (1, -1, 1, 1),
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

## EasyDelState

EasyDelState is new and cool feature in easydel and have a lot of options like
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
    sharding_axis_dims=(1, -1, 1, 1),
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
    sharding_array=(1, -1, 1, 1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
    # everything training will be in fully FSDP automatic and share data between devices
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

EasyDeL is released under the Apache v2 license. Please see the LICENSE file in the root directory of this project for
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
