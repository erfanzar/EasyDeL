import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from lib.python.EasyDel import JAXServer, JAXServerConfig
import jax


def main():
    print(jax.devices("gpu"))
    server = JAXServer.from_torch_pretrained(
        server_config=JAXServerConfig(
            max_new_tokens=8192,
            max_sequence_length=8192,
            max_compile_tokens=128
        ),
        pretrained_model_name_or_path="google/gemma-1.1-2b-it",
        load_in_8bit=True,
        sharding_axis_dims=(1, 1, 1, -1),
        device_map="cpu"
    )

    server.gradio_inference().launch(server_name="0.0.0.0")


if __name__ == '__main__':
    main()
