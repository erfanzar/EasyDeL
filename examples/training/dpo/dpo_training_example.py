from EasyDel import (
    TrainArguments,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    DPOTrainer,
    EasyDelState
)

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from jax import numpy as jnp
from fjformer import GenerateRNG
from jax.sharding import PartitionSpec

rng_g = GenerateRNG()
api = HfApi()


def llama_prompt(
        message: str,
        chat_history: list = None,
        system: str = None
) -> str:
    do_strip = False
    texts = [f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"] if system is not None else ["<s>[INST] "]
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f"{user_input} [/INST] {response.strip()} </s><s>[INST] ")
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)


def llama_format(example):
    if len(example["system"]) > 0:
        system = example["system"]
    else:
        system = None

    message = example["question"]
    prompt = llama_prompt(
        message,
        [],
        system
    )
    chosen = example['chosen'] + "</s><s>[INST]"
    rejected = example['rejected'] + "</s><s>[INST]"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def main():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ref_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    max_length = 1536  # Overall maximum length
    max_target_length = 1024  # Maximum Length for target column in Dataset
    max_prompt_length = 1024  # Maximum Length for prompt column in Dataset

    dtype = jnp.bfloat16

    sharding_axis_dims = (1, -1, 1, 1)
    sharding_axis_names = ("dp", "fsdp", "tp", "sp")

    query_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
    key_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
    value_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
    bias_partition_spec = PartitionSpec(("dp", "fsdp"), None, None, None)
    attention_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)

    ref_model_query_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
    ref_model_key_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
    ref_model_value_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
    ref_model_bias_partition_spec = PartitionSpec(("dp", "fsdp"), None, None, None)
    ref_model_attention_partition_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)

    dataset = load_dataset("Intel/orca_dpo_pairs")['train']

    original_columns = dataset.column_names

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = dataset.map(
        llama_format,
        remove_columns=original_columns
    )

    train_args = TrainArguments(
        model_name="EasyDeL-DPO-Example",
        num_train_epochs=3,
        learning_rate=1.2e-4,
        learning_rate_end=4e-5,
        warmup_steps=200,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.LINEAR,
        weight_decay=0.02,
        total_batch_size=8 * 10,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=(1, -1, 1, 1),
        gradient_accumulation_steps=1,
        dtype=dtype,
        param_dtype=dtype,
        training_time="7H",
        do_train=True,
        do_eval=False,
        track_memory=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    model_state = EasyDelState.from_pretrained(
        pretrained_model_name_or_path=model_name,
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
        config_kwargs=dict(
            gradient_checkpointing="nothing_saveable"
        )
    )
    print("Model State is Loaded.")

    ref_model_state = EasyDelState.from_pretrained(
        pretrained_model_name_or_path=ref_model_name,
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
        load_in_8bit=True
    )
    print("Ref Model State is Loaded.")

    dpo_trainer = DPOTrainer(
        model_state=model_state,
        ref_model_state=ref_model_state,
        arguments=train_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_target_length=max_target_length,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
    )

    dpo_trainer.train()


if __name__ == "__main__":
    main()
