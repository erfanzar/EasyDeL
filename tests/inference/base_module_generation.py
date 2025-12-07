import os
import time

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/huggingface-dataset"
os.environ["HF_HOME"] = "/dev/shm/huggingface"
os.environ["ENABLE_DISTRIBUTED_INIT"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["EASYDEL_AUTO"] = "1"

import jax
from flax import nnx as nn
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding
from transformers import AutoTokenizer

import easydel as ed

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def main():
    sharding_axis_dims = (1, 1, 1, -1, 1)
    max_model_len = 2048

    # _base = ed.AutoEasyDeLModelForCausalLM
    _base = ed.AutoEasyDeLModelForImageTextToText

    # pretrained_model_name_or_path = "Qwen/Qwen3-0.6B"
    pretrained_model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
    # pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"

    model = _base.from_pretrained(
        pretrained_model_name_or_path,
        auto_shard_model=True,
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_model_len,
            mask_max_position_embeddings=max_model_len,
            kvdtype=jnp.bfloat16,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            decode_attn_mechanism=ed.AttentionMechanisms.VANILLA,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        precision=jax.lax.Precision.DEFAULT,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

    messages = [
        {
            "role": "system",
            "content": (
                "Please reason step by step, and put your final answer within \\boxed{}. and give 3 different responses"
            ),
        },
        {"role": "user", "content": prompt},
    ]

    ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="jax",
        return_dict=True,
        max_length=max_model_len // 2,
        padding="max_length",
        padding_side="left",
        add_generation_prompt=True,
        truncation=True,
        truncation_side="left",
    )

    model.generation_config.max_new_tokens = max_model_len // 2
    model.generation_config.temperature = 0.4
    model.generation_config.top_k = 0
    model.generation_config.top_p = 0.95
    static_argnums = (0, 5)

    @ed.ejit(static_argnums=static_argnums)
    def generate(
        graphdef,
        graphstate,
        graphother,
        input_ids,
        attention_mask,
        generation_config,
    ):
        module = nn.merge(graphdef, graphstate, graphother)
        with module.mesh:
            return module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

    output = generate(
        *model.split_module(),
        ids["input_ids"],
        ids["attention_mask"],
        model.generation_config,
    )

    print(tokenizer.decode(output.sequences[0][max_model_len // 2 :], skip_special_tokens=True))
    time_spent = time.time()
    output = generate(
        *model.split_module(),
        ids["input_ids"],
        ids["attention_mask"],
        model.generation_config,
    )
    time_spent = time.time() - time_spent
    tokens = jnp.sum(output.sequences[0][max_model_len // 2 :] != tokenizer.pad_token_id)
    print("TPS:", tokens / time_spent)
    print("Num Tokens Generated", tokens)


if __name__ == "__main__":
    main()
