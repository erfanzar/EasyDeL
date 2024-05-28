import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# from jax.sharding import PartitionSpec
# import jax
# from jax import numpy as jnp, lax
# from src.python.easydel import AttentionModule
from pixelyai_core import PixelClient


def main():
    for char in PixelClient("107.159.172.69:40338")(prompt="hello world"):
        print(char, end="")
    # spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)

    # print(
    #     AttentionModule.test_attentions(
    #         sequence_length=32,
    #         chunk_size=16,
    #         axis_dims=(1, 1, 1, -1)
    #     ).to_string())


if __name__ == '__main__':
    main()
