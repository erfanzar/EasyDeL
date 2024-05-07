import multiprocessing

import jax.numpy
from flax.core import FrozenDict

from easydel import MistralConfig, FlaxMistralForCausalLM
from src.python.easydel.trainer.orpo import ORPOTrainer
from src.python.easydel import TrainArguments, EasyDeLState
from transformers import AutoTokenizer
from datasets import load_dataset
from jax import numpy as jnp

SEQUENCE_LENGTH = 128
NUM_TRAIN_EXAMPLES = 50
NUM_EVAL_EXAMPLES = 12
TOTAL_BATCH_SIZE = 1
NUM_TRAIN_EPOCHS = 3
MAX_TRAINING_STEPS = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
MAX_EVALUATION_STEPS = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE


def orpo_main():
    #####################
    # Model & Tokenizer #
    #####################
    with jax.default_device(jax.devices("cpu")[0]):
        model_name_or_path = "erfanzar/LLamaStory-70M"
        conf = MistralConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=4,
            intermediate_size=256,
            gradient_checkpointing="",
            max_position_embeddings=SEQUENCE_LENGTH * 4,
        )

        model = FlaxMistralForCausalLM(
            config=conf,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=True,
            input_shape=(8, 8)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #################
    # Dataset       #
    #################
    ds = load_dataset("orpo-explorers/OpenHermesPreferences-10k", split="train[:1%]")
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    train_dataset = ds.map(
        process,
        # num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    ################
    # Training     #
    ################
    dtype = jnp.float32
    trainer = ORPOTrainer(
        max_prompt_length=SEQUENCE_LENGTH,
        max_length=SEQUENCE_LENGTH * 2,
        max_completion_length=SEQUENCE_LENGTH * 2,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        arguments=TrainArguments(
            model_name="ORPO",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            total_batch_size=TOTAL_BATCH_SIZE,
            gradient_accumulation_steps=2,
            model_class=type(model),
            do_shard_fns=False,
            do_train=True,
            configs_to_initialize_model_class={
                "config": model.config,
                "input_shape": (8, 8),
                "dtype": dtype,
                "param_dtype": dtype,
            },
            dtype=dtype,
            param_dtype=dtype,
            track_memory=False,
            use_wandb=False,
            learning_rate=5e-4,
            do_last_save=True,
            init_input_shape=(8, 8)
        ),

    )

    trainer.train(model_parameters=FrozenDict({"params": model.params}))


if __name__ == "__main__":
    orpo_main()
