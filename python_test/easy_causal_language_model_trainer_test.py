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
import flax.core  # noqa: E402

from easydel import (  # noqa: E402
    CausalLanguageModelTrainer,
    TrainArguments,
    FlaxMixtralForCausalLM,
    MixtralConfig,
)
from jax import numpy as jnp, random  # noqa: E402
from datasets import Dataset  # noqa: E402
from fjformer import GenerateRNG  # noqa: E402

SEQUENCE_LENGTH = 128
DATA_ROW_SIZE = 10000
BATCH_SIZE = 32

MODEL_CONFIG = MixtralConfig(
    hidden_size=128,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=4,
    intermediate_size=256,
    gradient_checkpointing="",
    max_position_embeddings=SEQUENCE_LENGTH,
    use_scan_mlp=False,
    num_local_experts=8,
    num_experts_per_tok=2,
    output_router_logits=True,
)

RNG_GEN = GenerateRNG(seed=42)


def train():
    model = FlaxMixtralForCausalLM(config=MODEL_CONFIG, _do_init=True)
    params = model.params

    def data_generator():
        for i in range(DATA_ROW_SIZE):
            yield {
                "attention_mask": jnp.ones(SEQUENCE_LENGTH, dtype="i4"),
                "input_ids": random.randint(
                    RNG_GEN.rng, (SEQUENCE_LENGTH,), 0, 32000, dtype="i4"
                ),
            }

    example_data = Dataset.from_generator(
        data_generator,
    )
    dtype = jnp.float32
    trainer = CausalLanguageModelTrainer(
        arguments=TrainArguments(
            model_name="CLM-Test",
            num_train_epochs=1,
            total_batch_size=1,
            gradient_accumulation_steps=1,
            use_wandb=False,
            model_class=type(model),
            do_shard_fns=False,
            max_sequence_length=SEQUENCE_LENGTH,
            configs_to_initialize_model_class={
                "config": model.config,
                "input_shape": (1, 1),
                "dtype": dtype,
                "param_dtype": dtype,
            },
            dtype=dtype,
            param_dtype=dtype,
            track_memory=False,
            save_optimizer_state=True,
            max_training_steps=DATA_ROW_SIZE // BATCH_SIZE,
            learning_rate=3e-4,
            optimizer="adamw",
            scheduler="cosine",
        ),
        dataset_train=example_data,
    )
    output = trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))
    return output.checkpoint_path


def re_train(checkpoint_path: str | os.PathLike):
    model = FlaxMixtralForCausalLM(config=MODEL_CONFIG, _do_init=False)

    def data_generator():
        for i in range(DATA_ROW_SIZE):
            yield {
                "attention_mask": jnp.ones((SEQUENCE_LENGTH,), dtype="i4"),
                "input_ids": random.randint(
                    RNG_GEN.rng, (SEQUENCE_LENGTH,), 0, 32000, dtype="i4"
                ),
            }

    # example_data = IterableDataset.from_generator(data_generator, )
    example_data = Dataset.from_generator(
        data_generator,
    )
    dtype = jnp.float32
    trainer = CausalLanguageModelTrainer(
        arguments=TrainArguments(
            model_name="CLM-Test",
            num_train_epochs=4,
            total_batch_size=1,
            gradient_accumulation_steps=1,
            use_wandb=False,
            model_class=type(model),
            do_shard_fns=False,
            max_sequence_length=SEQUENCE_LENGTH,
            configs_to_initialize_model_class={
                "config": model.config,
                "input_shape": (1, 1),
                "dtype": dtype,
                "param_dtype": dtype,
            },
            dtype=dtype,
            param_dtype=dtype,
            track_memory=False,
            max_training_steps=DATA_ROW_SIZE // BATCH_SIZE,
            learning_rate=3e-4,
            optimizer="adamw",
            scheduler="cosine",
        ),
        dataset_train=example_data,
        checkpoint_path=checkpoint_path,
    )

    output = trainer.train()
    return output.checkpoint_path


if __name__ == "__main__":
    # re_train(train())
    train()
