import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)  # noqa: E402
sys.path.append(
    os.path.join(
        dirname,
        "../../src",
    )
)  # noqa: E402
import jax  # noqa

# jax.config.update("jax_platform_name", "cpu")  # CPU Test !
import flax.core  # noqa: E402
from datasets import Dataset, IterableDataset  # noqa: E402
from easydel import (  # noqa: E402
    AttentionMechanisms,
    CausalLanguageModelTrainer,
    FlaxMistralForCausalLM,
    MistralConfig,
    TrainArguments,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
)

from jax import numpy as jnp, random  # noqa: E402

TOTAL_BATCH_SIZE = 32
NUM_TRAIN_EXAMPLES = TOTAL_BATCH_SIZE * 15
NUM_EVAL_EXAMPLES = TOTAL_BATCH_SIZE * 15
NUM_TRAIN_EPOCHS = 3


def main(use_iterable_dataset: bool):
    sequence_length = 512
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
        attn_dtype=jnp.float16,
        attn_mechanism=AttentionMechanisms.pallas_flash,
        block_k=128,
        block_q=128,
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
    dtype = jnp.float16
    trainer = CausalLanguageModelTrainer(
        arguments=TrainArguments(
            model_name="CLM_TEST",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            total_batch_size=TOTAL_BATCH_SIZE,
            gradient_accumulation_steps=2,
            max_training_steps=max_training_steps,
            max_evaluation_steps=max_evaluation_steps,
            model_class=type(model),
            do_shard_fns=True,
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
            track_memory=True,
            use_wandb=True,
            learning_rate=5e-4,
            label_smoothing_factor=0.1,
            z_loss=0.0001,
            train_on_inputs=True,
            do_last_save=True,
            training_time="80Min",
            optimizer=EasyDeLOptimizers.ADAMW,
            scheduler=EasyDeLSchedulers.COSINE,
            clip_grad=1.0,
            warmup_steps=5
        ),
        dataset_train=example_train_data,
        dataset_eval=example_eval_data,
    )

    output = trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))
    trainer.save_pretrained(output.state, to_torch=True)


if __name__ == "__main__":
    main(use_iterable_dataset=True)
