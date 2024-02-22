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


def ultra_chat_prompting_sample(
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