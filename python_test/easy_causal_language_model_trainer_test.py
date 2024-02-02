import os

import flax.core

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from lib.python.EasyDel import (
    CausalLanguageModelTrainer,
    AutoEasyDelModelForCausalLM,
    TrainArguments
)
from jax import numpy as jnp, random
from datasets import Dataset


def main():
    model, params = AutoEasyDelModelForCausalLM.from_pretrained('gpt2')
    data_row_size = 100
    sequence_length = 128

    def data_generator():
        for i in range(data_row_size):
            yield {
                "attention_mask": jnp.ones(
                    (1, sequence_length), dtype="i4"
                ),
                "input_ids": random.randint(
                    random.PRNGKey(0), (1, sequence_length), 0, 32000, dtype="i4"
                )
            }

    example_data = Dataset.from_generator(data_generator, )
    dtype = jnp.float32
    trainer = CausalLanguageModelTrainer(
        arguments=TrainArguments(
            model_name="Lora-Test",
            num_train_epochs=100,
            total_batch_size=1,
            gradient_accumulation_steps=1,
            use_wandb=False,
            model_class=type(model),
            do_shard_fns=False,
            max_sequence_length=sequence_length,
            configs_to_initialize_model_class={
                "config": model.config,
                "input_shape": (1, 1),
                "dtype": dtype,
                "param_dtype": dtype
            },
            dtype=dtype,
            param_dtype=dtype,
            track_memory=False
        ),
        dataset_train=example_data,
    )

    trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))


if __name__ == "__main__":
    main()
