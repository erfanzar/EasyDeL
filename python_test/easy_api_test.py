import flax.core
import transformers

from lib.python.EasyDel.serve.api.serve import EasyServe, EasyServeConfig
from lib.python.EasyDel import Qwen1Config, FlaxQwen1ForCausalLM
from jax import numpy as jnp, lax


def modify_qwen_tokenizer(tokenizer):
    tokenizer.eos_token_id = tokenizer.im_end_id
    tokenizer.bos_token_id = tokenizer.im_start_id
    tokenizer.pad_token_id = tokenizer.im_end_id
    return tokenizer


def main():
    max_position_embeddings = 512
    config = Qwen1Config(
        hidden_size=256,
        intermediate_size=200,
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=4,
        num_attention_heads=2,
        cv_channels=64
    )

    model = FlaxQwen1ForCausalLM(
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

    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)
    server = EasyServe.from_parameters(
        llm=model,
        params={
            "params": params,
        },
        serve_config=serve_config,
        tokenizer=modify_qwen_tokenizer(tokenizer),
        partition_rules=config.get_partition_rules(True),
        shard_parameters=False,
    )
    server.tokenizer = modify_qwen_tokenizer(server.tokenizer)
    print(server)

    server.prefix_tokenizer = modify_qwen_tokenizer(server.prefix_tokenizer)
    server.fire()


if __name__ == "__main__":
    main()
