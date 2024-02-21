from EasyDel import JAXServer, JAXServerConfig
import jax
from jax import numpy as jnp, lax


def main():
    server_config = JAXServerConfig(
        max_sequence_length=256000,
        max_compile_tokens=512,
        max_new_tokens=4096 * 10,
        dtype="bf16"
    )
    server = JAXServer.from_torch_pretrained(
        server_config=server_config,
        pretrained_model_name_or_path="LargeWorldModel/LWM-Text-Chat-256K",
        device=jax.devices('cpu')[0],
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        precision=jax.lax.Precision("fastest"),
        sharding_axis_dims=(1, 1, 4, -1),
        sharding_axis_names=("dp", "fsdp", "tp", "sp"),
        input_shape=(1, server_config.max_sequence_length),
        model_config_kwargs=dict(
            fully_sharded_data_parallel=False,
            attn_mechanism="ring"
        )
    )

    for response, _ in server.process(
            "EasyDeL is and Open-Source Library to make training and serving process of LLMs efficient and "
    ):
        print(response, end="")


if __name__ == "__main__":
    main()
