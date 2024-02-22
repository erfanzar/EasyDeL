from typing import Sequence, Optional
from jax.sharding import PartitionSpec
from dataclasses import dataclass


@dataclass
class EasyServeConfig:
    """
    :param host: str: Set the host address of the server
    :param port: int: Specify the port number that the server will run on
    :param batch_size: int: Set the batch size of the model
    :param max_sequence_length: int: Set the maximum length of the text that can be generated
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
    :param use_prefix_tokenizer: bool: Determine if the tokenizer should be used to generate tokens
    :param pre_compile: bool: Pre-compile the model
    :return: Nothing

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

    def __repr__(self):

        """
        The __repr__ function is used to generate a string representation of an object.
        This function should return a string that can be parsed by the Python interpreter
        to recreate the object. The __repr__ function is called when you use print() on an
        object, or when you type its name in the REPL.

        :param self: Refer to the instance of the class
        :return: A string representation of the object
        """
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):

                try:
                    repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
                    string += repr_src if len(repr_src) < 500 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
                except TypeError:
                    ...

        return string + ")"

    def __str__(self):

        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
