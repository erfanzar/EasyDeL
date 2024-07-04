import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)  # noqa: E402
sys.path.append(
    os.path.join(
        dirname,
        "../../src",
    )
)  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_platform_name", "cpu")  # CPU Test !
from easydel import engine_client  # noqa: E402


def main():
    print("Gradio " + "*" * 50)
    for response in engine_client.generate_gradio(
        "http://127.0.0.1:11552/",
        conversation=[{"content": "hi", "role": "user"}],
    ):
        print(response.sequence_stream, end="")
    print()
    print(f"{response.tokens_per_second=}")
    print(f"{response.elapsed_time=}")

    print("WebSocket " + "*" * 50)
    for response in engine_client.generate_websocket(
        "127.0.0.1:11551",
        conversation=[{"content": "hi", "role": "user"}],
    ):
        print(response.response, end="")
    print()
    print(response.progress)


if __name__ == "__main__":
    main()
