import flax.core
import transformers

from lib.python.EasyDel.serve.api.serve import EasyServe, EasyServeConfig
from lib.python.EasyDel import MixtralConfig, FlaxMixtralForCausalLM
from jax import numpy as jnp, lax


def main():
    mistral_config = MixtralConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        bits=None,
        max_position_embeddings=512
    )

    mistral_model = FlaxMixtralForCausalLM(
        config=mistral_config,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        precision=None,
        _do_init=True
    )

    params = mistral_model.params

    serve_config = EasyServeConfig(
        verbose=True,
        dtype="fp16",
        pre_compile=False,
        max_new_tokens=256,
        max_compile_tokens=32,
        max_length=mistral_config.max_position_embeddings
    )

    server = EasyServe.from_parameters(
        llm=mistral_model,
        params={"params": params},
        serve_config=serve_config,
        tokenizer=transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1"),
        partition_rules=mistral_config.get_partition_rules(True),
        shard_parameters=False,
    )

    print(server)
    server.fire()


if __name__ == "__main__":
    main()
