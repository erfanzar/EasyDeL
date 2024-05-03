import multiprocessing

import jax.numpy

from EasyDel import LlamaConfig, FlaxLlamaForCausalLM
from lib.python.EasyDel.trainer.orpo import ORPOTrainer
from lib.python.EasyDel import TrainArguments, EasyDelState
from transformers import AutoTokenizer
from datasets import load_dataset
from jax import numpy as jnp


def orpo_main():
    ################
    # Model & Tokenizer
    ################
    with jax.default_device(jax.devices("cpu")[0]):
        model_name_or_path = "erfanzar/LLamaStory-70M"
        conf = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            use_scan_mlp=False
        )
        module = FlaxLlamaForCausalLM(
            config=conf,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=True,
            input_shape=(8, 8)
        )
        model = module.to_easydel_state(params=module.params)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    ds = load_dataset("orpo-explorers/OpenHermesPreferences-10k", split="train[:2%]")
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    train_dataset = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    ################
    # Training
    ################
    trainer = ORPOTrainer(
        model_state=model,
        arguments=TrainArguments(
            num_train_epochs=2,
            max_sequence_length=128,
            do_train=True,
            model_name="ORPO",
            dtype=jax.numpy.float16,
            total_batch_size=1,
            use_wandb=False
        ),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=128,
        max_prompt_length=128,
        max_completion_length=128,
    )

    trainer.train()


if __name__ == "__main__":
    orpo_main()
