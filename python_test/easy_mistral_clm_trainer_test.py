import os

import flax.core

from EasyDel import Qwen2Config

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from lib.python.EasyDel import (
    CausalLanguageModelTrainer,
    AutoEasyDelModelForCausalLM,
    TrainArguments,
    FlaxMistralForCausalLM,
    MistralConfig
)
from jax import numpy as jnp, random
from datasets import Dataset


def main():
    sequence_length = 128
    data_row_size = 100
    config = MistralConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=256,
        gradient_checkpointing="",
        max_position_embeddings=sequence_length,
    )

    model = FlaxMistralForCausalLM(config=config, _do_init=True)
    params = model.params

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
            model_name="MistralSmoothZlossTest",
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
            label_smoothing_factor=0.1,
            z_loss=0.0001
        ),
        dataset_train=example_data,
        dataset_eval=example_data,
    )

    trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))


if __name__ == "__main__":
    main()
