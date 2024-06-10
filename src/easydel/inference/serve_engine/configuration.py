from typing import Sequence, Optional, NamedTuple
from jax.sharding import PartitionSpec


class EasyDeLServeEngineConfig(NamedTuple):
    """
    Args:
        host: str: Set the host address of the server
        port: int: Specify the port number that the server will run on
        batch_size: int: Set the batch size of the model
        max_sequence_length: int: Set the maximum length of the text
            that can be generated
        max_new_tokens: int: Determine how many tokens can be added to
            the vocabulary
        max_compile_tokens: int: Set the maximum number of tokens that
            can be streamed at a time
        generation_ps: jax.sharding.PartitionSpec : PartitionSpec to use
            for sharding data
        temperature: float: Control the randomness of the output
        top_p: float: Control the diversity of the text generated
        top_k: int: Limit the number of tokens that can be generated
        logging: bool: Print out the progress of the server
        mesh_axes_names: Sequence[str]: Specify the names of the axes in
            the mesh tensor
        mesh_axes_shape: Sequence[int]: Specify the shape of the mesh
        dtype: str: Specify the data type of the model
        use_prefix_tokenizer: bool: Determine if the tokenizer should be
            used to generate tokens
        pre_compile: bool: Pre-compile the model

    Returns:
        Nothing
    """
    host: str = "0.0.0.0"
    port: int = 2059

    batch_size: int = 1
    max_sequence_length: int = 4096
    max_new_tokens: int = 4096
    max_compile_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.2
    greedy: bool = False

    logging: bool = True

    mesh_axes_names: Sequence[str] = ("dp", "fsdp", "tp", "sp")
    mesh_axes_shape: Sequence[int] = (1, -1, 1, 1)
    generation_ps: PartitionSpec = PartitionSpec("dp", "fsdp")
    dtype: str = "fp16"

    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None

    use_prefix_tokenizer: bool = True
    pre_compile: bool = True

    verbose: bool = True

    use_mxn_break_point: bool = True
