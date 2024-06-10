import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)  # noqa: E402
sys.path.append(
    os.path.join(
        dirname,
        "..",
    )
)  # noqa: E402


os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from easydel import (  # noqa: E402
    EasyDeLXRapTureConfig,
    CausalLanguageModelTrainer,
    AutoEasyDeLModelForCausalLM,
    TrainArguments,
)
from jax import numpy as jnp, random  # noqa: E402
from datasets import Dataset  # noqa: E402


def main():
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        "erfanzar/LLamaStory-70M"
    )
    data_row_size = 1_000
    sequence_length = 128
    rab_config = EasyDeLXRapTureConfig(
        parameters={"params": params},
        lora_dim=64,
        fully_fine_tune_parameters=[],
        lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj", "o_proj"],
        verbose=False,
    )

    def data_generator():
        for i in range(data_row_size):
            yield {
                "attention_mask": jnp.ones((1, sequence_length), dtype="i4"),
                "input_ids": random.randint(
                    random.PRNGKey(0), (1, sequence_length), 0, 32000, dtype="i4"
                ),
            }

    example_data = Dataset.from_generator(
        data_generator,
    )
    dtype = jnp.float32
    trainer = CausalLanguageModelTrainer(
        arguments=TrainArguments(
            model_name="Lora-Test",
            num_train_epochs=100,
            rapture_config=rab_config,
            total_batch_size=64,
            gradient_accumulation_steps=2,
            use_wandb=False,
            model_class=type(model),
            do_shard_fns=True,
            max_sequence_length=sequence_length,
            configs_to_initialize_model_class={
                "config": model.config,
                "input_shape": (1, 1),
                "dtype": dtype,
                "param_dtype": dtype,
            },
            dtype=dtype,
            param_dtype=dtype,
            track_memory=False,
        ),
        dataset_train=example_data,
    )

    trainer.train()


if __name__ == "__main__":
    main()
