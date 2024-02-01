import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from lib.python.EasyDel import (
    EasyDeLXRapTureConfig,
    CausalLanguageModelTrainer,
    AutoEasyDelModelForCausalLM,
    TrainArguments
)
from jax import numpy as jnp, random
from flax.core import FrozenDict
from datasets import Dataset


def main():
    model, params = AutoEasyDelModelForCausalLM.from_pretrained('erfanzar/LLamaStory-70M')
    data_row_size = 1_000
    sequence_length = 128
    rab_config = EasyDeLXRapTureConfig(
        64,
        fully_fine_tune_parameters=["embed_tokens"],
        lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj", "o_proj"],
        verbose=False
    )

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
    print(example_data)
    dtype = jnp.float32
    print(len(example_data))
    trainer = CausalLanguageModelTrainer(
        arguments=TrainArguments(
            model_name="Lora-Test",
            num_train_epochs=100,
            # rapture_config=rab_config,
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
            param_dtype=dtype
        ),
        dataset_train=example_data,
    )

    trainer.train(
        model_parameters=FrozenDict({"params": params})
    )


if __name__ == "__main__":
    main()
