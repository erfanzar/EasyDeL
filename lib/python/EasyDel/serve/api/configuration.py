from typing import Sequence
from jax.sharding import PartitionSpec
from dataclasses import dataclass


@dataclass
class ServeConfig:
    """
    :param host: str: Set the host address of the server
    :param port: int: Specify the port number that the server will run on
    :param batch_size: int: Set the batch size of the model
    :param contains_auto_format: bool: Determine whether the input text contains auto-formatting
    :param max_length: int: Set the maximum length of the text that can be generated
    :param max_new_tokens: int: Determine how many tokens can be added to the vocabulary
    :param max_compile_tokens: int: Set the maximum number of tokens that can be streamed at a time
    :param generation_ps: jax.sharding.PartitionSpec : PartitionSpec to use for sharding data
    :param temperature: float: Control the randomness of the output
    :param top_p: float: Control the diversity of the text generated
    :param top_k: int: Limit the number of tokens that can be generated
    :param logging: bool: Print out the progress of the server
    :param mesh_axes_names: Sequence[str]: Specify the names of the axes in the mesh tensor
    :param mesh_axes_shape: Sequence[int]: Specify the shape of the mesh
    :param dtype: str: Specify the data type of the model
    :param stream_tokens_for_gradio: bool: Determine whether the stream tokens
    :param use_prefix_tokenizer: bool: Determine if the tokenizer should be used to generate tokens
    :param pre_compile: bool: Pre-compile the model
    :return: Nothing

    """
    host: str = "0.0.0.0"
    port: int = 2059
    batch_size: int = 1
    contains_auto_format: bool = True
    max_length: int = 4096
    max_new_tokens: int = 4096
    max_compile_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    logging: bool = True
    mesh_axes_names: Sequence[str] = ("dp", "fsdp", "tp", "sp")
    mesh_axes_shape: Sequence[int] = (1, -1, 1, 1)
    generation_ps: PartitionSpec = PartitionSpec("dp", "fsdp")
    dtype: str = "fp16"
    stream_tokens_for_gradio: bool = True
    use_prefix_tokenizer: bool = True
    pre_compile: bool = True
    verbose: bool = True
