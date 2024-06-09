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
import flax.core  # noqa: E402

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from src.python.easydel import (  # noqa: E402
    CausalLanguageModelTrainer,
    TrainArguments,
    FlaxMistralForCausalLM,
    MistralConfig,
)
from jax import numpy as jnp, random  # noqa: E402
from datasets import Dataset, IterableDataset  # noqa: E402


def main(use_iterable_dataset: bool):
    sequence_length = 128
    NUM_TRAIN_EXAMPLES = 50
    NUM_EVAL_EXAMPLES = 12
    TOTAL_BATCH_SIZE = 2
    NUM_TRAIN_EPOCHS = 3
    max_training_steps = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
    max_evaluation_steps = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE
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

    def data_generator(num_rows: int):
        for i in range(num_rows):
            yield {
                "attention_mask": jnp.ones((sequence_length,), dtype="i4"),
                "input_ids": random.randint(
                    random.PRNGKey(0), (sequence_length,), 0, 32000, dtype="i4"
                ),
            }

    if not use_iterable_dataset:
        example_train_data = Dataset.from_generator(
            data_generator, gen_kwargs={"num_rows": NUM_TRAIN_EXAMPLES}
        )
        example_eval_data = Dataset.from_generator(
            data_generator, gen_kwargs={"num_rows": NUM_EVAL_EXAMPLES}
        )
    else:
        example_train_data = IterableDataset.from_generator(
            data_generator, gen_kwargs={"num_rows": NUM_TRAIN_EXAMPLES}
        )
        example_eval_data = IterableDataset.from_generator(
            data_generator, gen_kwargs={"num_rows": NUM_EVAL_EXAMPLES}
        )
    dtype = jnp.float32
    trainer = CausalLanguageModelTrainer(
        arguments=TrainArguments(
            model_name="MistralSmoothZlossTest",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            total_batch_size=TOTAL_BATCH_SIZE,
            gradient_accumulation_steps=2,
            max_training_steps=max_training_steps,
            max_evaluation_steps=max_evaluation_steps,
            model_class=type(model),
            do_shard_fns=False,
            do_train=True,
            do_eval=True,
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
            use_wandb=False,
            learning_rate=5e-4,
            label_smoothing_factor=0.1,
            z_loss=0.0001,
            train_on_inputs=True,
            save_steps=50,
            save_total_limit=1,
            do_last_save=True,
        ),
        dataset_train=example_train_data,
        dataset_eval=example_eval_data,
    )
    print(trainer._get_information())
    output = trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))
    trainer.save_pretrained(output.state, to_torch=True)


if __name__ == "__main__":
    main(use_iterable_dataset=True)
