import transformers

from easydel import EasyServe, EasyServeConfig, EasyClient, AutoEasyDeLModelForCausalLM
from jax import numpy as jnp, lax


def main():
    pretrained_model_name_or_path = "Qwen/Qwen1.5-0.5B-Chat"
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        load_in_8bit=True,
        precision=lax.Precision("fastest"),
        auto_shard_params=True
    )
    serve_config = EasyServeConfig(
        verbose=True,
        dtype="fp16",
        pre_compile=False,
        max_new_tokens=2048,
        max_compile_tokens=128,
        max_sequence_length=2048,
        use_prefix_tokenizer=False
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    server = EasyServe.from_parameters(
        llm=model,
        params={"params": params},
        serve_config=serve_config,
        tokenizer=tokenizer,
        partition_rules=model.config.get_partition_rules(True),
        shard_parameters=False,
    )
    print(server)
    server.fire()


def gen_test():
    client = EasyClient(host="localhost", port=2059)
    response = None
    for response in client.generate(conversation=[{"role": "user", "content": "Hello World"}]):
        print(response.response, end="")

    print("\n\nINFO:")
    print(f"{response.generation_duration=}")
    print(f"{response.tokens_pre_second=}")
    print(f"{response.num_token_generated=!r}")


if __name__ == "__main__":
    main()
