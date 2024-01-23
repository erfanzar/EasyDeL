import transformers

from lib.python.EasyDel.serve.api.serve import EasyServe, EasyServeConfig
from lib.python.EasyDel import MixtralConfig, FlaxMixtralForCausalLM
from jax import numpy as jnp, lax


def main():
    mixtral_config = MixtralConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_local_experts=4,
    )

    mixtral_model = FlaxMixtralForCausalLM(
        config=mixtral_config,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        precision=lax.Precision("fastest"),
        _do_init=True
    )

    params = mixtral_model.params

    serve_config = EasyServeConfig(
        verbose=True,
        dtype="fp16",
        pre_compile=False
    )

    server = EasyServe.from_parameters(
        llm=mixtral_model,
        params=params,
        serve_config=serve_config,
        tokenizer=transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1"),
        partition_rules=mixtral_config.get_partition_rules(True),
        shard_parameters=False,
    )

    print(server)


if __name__ == "__main__":
    main()
