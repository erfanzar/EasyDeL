import transformers

from lib.python.EasyDel import EasyServe, EasyServeConfig, MixtralConfig, FlaxMixtralForCausalLM
from jax import numpy as jnp, lax


def main():
    max_position_embeddings = 512
    config = MixtralConfig(
        hidden_size=256,
        intermediate_size=200,
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2
    )

    model = FlaxMixtralForCausalLM(
        config=config,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        precision=None,
        _do_init=True,
        input_shape=(1, max_position_embeddings)
    )

    params = model.params

    serve_config = EasyServeConfig(
        verbose=True,
        dtype="fp16",
        pre_compile=False,
        max_new_tokens=32,
        max_compile_tokens=32,
        max_sequence_length=max_position_embeddings,
        use_prefix_tokenizer=False
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        trust_remote_code=True
    )
    server = EasyServe.from_parameters(
        llm=model,
        params={
            "params": params,
        },
        serve_config=serve_config,
        tokenizer=tokenizer,
        partition_rules=config.get_partition_rules(True),
        shard_parameters=False,
    )
    print(server)
    server.fire()


if __name__ == "__main__":
    main()
