import os

from EasyDel import (
    AutoEasyDelModelForCausalLM,
    TrainArguments,
    CausalLanguageModelTrainer,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    get_modules_by_type,
    easystate_to_huggingface_model, EasyDelState
)
from transformers import MixtralForCausalLM
from datasets import load_dataset
from flax.core import FrozenDict
from transformers import AutoTokenizer
from jax import numpy as jnp
import jax


def main():
    pretrained_model_name_or_path = "mistralai/Mixtral-8x7B-v0.1"
    dtype = jnp.bfloat16
    max_length = 4096

    model, params = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device=jax.devices("cpu")[0],
        input_shape=(1, 1),
        device_map="auto",
        sharding_axis_dims=(1, 1, 1, -1)
    )

    model_parameters = FrozenDict({"params": params})

    model.config.add_basic_configurations(
        attn_mechanism="flash",  # Using Flash Attention here you can simply just set this to normal or ring
        block_b=1,
        block_q=128,
        block_k=128,
        block_k_major=128,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "DATASET_NAME",
        split="train",
    )

    def tokenization_process(data_chunk):
        return tokenizer(
            data_chunk["prompt"],
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length"
        )

    dataset = dataset.map(
        tokenization_process,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names
    )

    train_args = TrainArguments(
        model_class=get_modules_by_type(model.config.model_type)[1],
        configs_to_initialize_model_class={
            "config": model.config,
            "dtype": dtype,
            "param_dtype": dtype,
            "input_shape": (1, max_length)
        },
        custom_rule=model.config.get_partition_rules(True),
        model_name="Mixtral-Tune",
        num_train_epochs=2,
        learning_rate=1e-5,
        learning_rate_end=7e-6,
        warmup_steps=200,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.LINEAR,
        weight_decay=0.02,
        total_batch_size=32,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=(1, 1, 1, -1),
        use_pjit_attention_force=False,
        gradient_accumulation_steps=4,
        init_input_shape=(1, max_length),
        dtype=dtype,
        param_dtype=dtype,
        step_start_point=0,
        training_time="10H",  # Set training limit time to 10 hours you can set this to None
        wandb_entity=None  # Set WANDB team to send logs to
        # Read docs for more and better understanding of options
    )

    trainer = CausalLanguageModelTrainer(
        train_args,
        dataset.shuffle().shuffle().shuffle(),
        checkpoint_path=None  # In Case of resuming from a checkpoint you can pass checkpoint path here and simply just
        # don't create and run model and params steps above.
    )

    output = trainer.train(
        model_parameters=model_parameters,  # pass this as none in case of resuming from last checkpoint
        state=None
    )

    with jax.default_device(jax.devices("cpu")[0]):  # Converting EasyDel model to huggingface model and offloading that
        # on to cpu.
        model = easystate_to_huggingface_model(
            state=EasyDelState.load_state(
                output.checkpoint_path
            ),
            base_huggingface_module=MixtralForCausalLM,
            config=model.config
        )
    model = model.half()
    model.push_to_hub("EasyDeL-Mixtral")


if __name__ == "__main__":
    main()
