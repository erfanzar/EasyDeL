import os

import flax.core
from transformers import AutoTokenizer

from lib.python.EasyDel.trainer import conversations_formatting_function

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
from lib.python.EasyDel.trainer.sft import SFTTrainer
from lib.python.EasyDel import (
    TrainArguments,
    FlaxMistralForCausalLM,
    MistralConfig
)
from jax import numpy as jnp
from datasets import load_dataset


def create_prompt_creator(tokenizer):
    def _tr(field):
        return [
            {"role": "user", "content": field["conversation"][0]["input"]},
            {"role": "assistant", "content": field["conversation"][0]["output"]}
        ]

    def _pc(
            sample
    ):
        sample = conversations_formatting_function(tokenizer, messages_field="conversation")(  # type: ignore
            {"conversation": _tr(sample)}
        )
        return [sample]

    return _pc


def main():
    sequence_length = 128
    config = MistralConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=256,
        gradient_checkpointing="",
        max_position_embeddings=sequence_length,
    )

    dove = load_dataset("LDJnr/Pure-Dove", )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    prompt_creator = create_prompt_creator(tokenizer)

    model = FlaxMistralForCausalLM(config=config, _do_init=True)
    params = model.params

    dtype = jnp.float32
    trainer = SFTTrainer(
        arguments=TrainArguments(
            model_name="SFTTrainer-Debug",
            num_train_epochs=3,
            total_batch_size=2,
            gradient_accumulation_steps=2,
            use_wandb=False,
            model_class=type(model),
            do_shard_fns=False,
            do_train=True,
            do_eval=True,
            max_sequence_length=sequence_length,
            configs_to_initialize_model_class={
                "config": model.config,
                "input_shape": (1, 1),
                "dtype": dtype,
                "param_dtype": dtype
            },
            dtype=dtype,
            param_dtype=dtype,
            track_memory=False,
            learning_rate=5e-4,
            label_smoothing_factor=0.1,
            z_loss=0.0001,
            train_on_inputs=True,
            save_steps=50,
            save_total_limit=1,
            do_last_save=False,
        ),
        train_dataset=dove["train"],
        eval_dataset=dove["train"],  # we don't have eval dataset rn :)
        tokenizer=tokenizer,
        dataset_text_field=None,
        formatting_func=prompt_creator,
        packing=False,
        num_of_sequences=1024
    )
    return trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))


if __name__ == "__main__":
    res = main()
