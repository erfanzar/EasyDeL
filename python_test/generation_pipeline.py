import dataclasses
import warnings
from typing import Optional, Union, Dict, Any

import flax.core
import jax

from src.python.easydel import (
    FlaxLlamaForCausalLM,
    LlamaConfig,
    GenerationPipelineConfig,
    GenerationPipeline
)
from jax import numpy as jnp, random, lax, jit
from transformers import AutoTokenizer, PreTrainedTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=4,
        max_position_embeddings=512,
        use_scan_mlp=False
    )
    model = FlaxLlamaForCausalLM(
        config=config,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        precision=lax.Precision("fastest"),
        _do_init=True
    )
    tokens = tokenizer("SOME TEXT", return_tensors="np", max_length=32, padding="max_length")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    pipeline = GenerationPipeline(
        model=model,
        params=model.params,
        tokenizer=tokenizer,
        add_params_field=True,
        generation_config=GenerationPipelineConfig(max_new_tokens=128, do_sample=True)
    )
    for token in pipeline.generate(input_ids, attention_mask):
        print(token, end="")
    print("\n")
    print("*" * 50)
    for token in pipeline.generate(input_ids, attention_mask):
        print(token, end="")


if __name__ == "__main__":
    main()
