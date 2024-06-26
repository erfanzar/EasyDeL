import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

try:
    import easydel as ed
except ModuleNotFoundError:
    dirname = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(dirname)  # noqa: E402
    sys.path.append(
        os.path.join(
            dirname,
            "../src",
        )
    )

from easydel import (  # noqa: E402
    FlaxLlamaForCausalLM,
    GenerationPipeline,
    GenerationPipelineConfig,
    LlamaConfig,
)
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        max_position_embeddings=512,
        use_scan_mlp=False,
        axis_dims=(1, 1, 1, -1),
        quantize_kv_cache=False,
    )
    model = FlaxLlamaForCausalLM(
        config=config,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        precision=lax.Precision("fastest"),
        input_shape=(1, 2),
        _do_init=True,
        seed=81,
    )
    tokenizer.padding_side="left"
    tokens = tokenizer(
        "SOME TEXT", return_tensors="np", max_length=32, padding="max_length"
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    params = model.params
    # params = fjformer.linen.quantize_int8_parameters(["kernel", "embedding"], params)
    pipeline = GenerationPipeline(
        model=model,
        params=params,
        tokenizer=tokenizer,
        generation_config=GenerationPipelineConfig(
            max_new_tokens=128,
            temprature=0.8,
            top_p=0.95,
            top_k=10,
            eos_token_id=23070,
            length_penalty=1.2,
            repetition_penalty=1.2,
        ),
    )
    for token in pipeline.generate(input_ids, attention_mask):
        print(token, end="")
    print("\n")
    print("*" * 50)
    for token in pipeline.generate(input_ids, attention_mask):
        print(token, end="")
    # streamer = TextIteratorStreamer(tokenizer=tokenizer)
    # threading.Thread(target=pipeline.generate, args=(input_ids, attention_mask, streamer)).start()
    # for char in streamer:
    #     print(char, end="")


if __name__ == "__main__":
    main()
