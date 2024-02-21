import chex
import flax
from jax.experimental.mesh_utils import create_device_mesh
from transformers import PretrainedConfig, FlaxPreTrainedModel
import jax
from jax import numpy as jnp
from typing import Sequence, Union, Optional, Literal, Tuple, Any
from dataclasses import dataclass
from jax.sharding import PartitionSpec, Mesh

AVAILABLE_ATTENTION_MECHANISMS = Literal["normal", "flash", "splash", "ring"]


def set_attrs_smartly(self, attr_name: str, default: Any, new_attr: Any):
    if not hasattr(self, attr_name):
        setattr(self, attr_name, default)
    if not new_attr == Ellipsis:
        setattr(self, attr_name, new_attr)


@dataclass
class EasyMethod:
    TRAIN: str = "train"
    SERVE: str = "serve"
    EVAL: str = "serve"
    CONVERT: str = "convert"


class EasyDelPretrainedConfig(PretrainedConfig):
    """
    It initializes all the attributes of an object, and it's called when you create a new instance of that class.
    :param self: Refer to the instance of the class
    :param axis_dims: Sequence[int]: Specify the number of dimensions for each axis
    :param axis_names: Sequence[str]: Set the names of the axes
    :param attn_mechanism: Literal["normal", "flash", "splash", "ring"]: attention mechanism to use
    :param block_k: int: block size of key_states
    :param block_q: int: block size of query_states
    :param block_b: int: block size of bias
    :param block_q_major_dkv: int: block size of block_q_major_dkv
    :param block_k_major_dkv: int: block size of block_k_major_dkv
    :param block_k_dkv: int: block size of block_k_dkv
    :param block_q_dkv: int: block size of block_q_dkv
    :param block_k_major_dq: int: block size of block_k_major_dq
    :param block_k_dq: int: block size of block_k_dq
    :param block_q_dq: int: block size of block_q_dq
    :param query_partition_spec: PartitionSpec: Specify the partitioning of the query tensor
    :param key_partition_spec: PartitionSpec: Partition the key matrix
    :param value_partition_spec: PartitionSpec: Specify the partitioning of the value tensor
    :param bias_partition_spec: PartitionSpec: Specify the Attention Bias partition spec
    :param attention_partition_spec: PartitionSpec: Specify the partitioning of the attention weights
    :param use_shard_map: bool: whenever to use shard_map for attention
    :param use_scan_mlp: bool: Determine whether to use scan_mlp or not
    :param backend: Optional[None]: Specify the backend to use
    """

    def __init__(
            self,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = "normal",
            block_k: int = 128,
            block_q: int = 128,
            block_b: int = 1,
            block_k_major: int = 128,
            block_q_major_dkv: int | None = None,
            block_k_major_dkv: int | None = None,
            block_k_dkv: int | None = None,
            block_q_dkv: int | None = None,
            block_k_major_dq: int | None = None,
            block_k_dq: int | None = None,
            block_q_dq: int | None = None,
            query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            key_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            value_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None, None),
            attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            use_sharded_kv_caching: bool = True,
            backend: Optional[None] = jax.default_backend(),
            easy_method: Literal["train", "serve", "convert"] = EasyMethod.TRAIN,
            bits: Optional[int] = None,
            scan_ring_attention: bool = True,
            scan_attention_layers: bool = False,
            use_scan_mlp: bool = True,
            scan_mlp_chunk_size: int = 1024,
            **kwargs
    ):
        self.query_partition_spec = query_partition_spec
        self.key_partition_spec = key_partition_spec
        self.value_partition_spec = value_partition_spec
        self.bias_partition_spec = bias_partition_spec
        self.attention_partition_spec = attention_partition_spec
        self.use_shard_map = use_shard_map
        self.axis_dims = axis_dims
        self.axis_names = axis_names
        self.backend = backend if backend is not None else ""
        self.easy_method = easy_method
        self.attn_mechanism = attn_mechanism
        self.block_b = block_b
        self.block_k = block_k
        self.block_q = block_q
        self.block_k_major = block_k_major
        self.block_q_major_dkv = block_q_major_dkv or block_q
        self.block_k_major_dkv = block_k_major_dkv or block_k
        self.block_k_dkv = block_k_dkv or block_k
        self.block_q_dkv = block_q_dkv or block_q
        self.block_k_major_dq = block_k_major_dq or block_k
        self.block_k_dq = block_k_dq or block_k
        self.block_q_dq = block_q_dq or block_q
        self.bits = bits
        self.scan_attention_layers = scan_attention_layers
        self.scan_ring_attention = scan_ring_attention
        self.use_sharded_kv_caching = use_sharded_kv_caching
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        super().__init__(**kwargs)

    @staticmethod
    def create_mesh(
            axis_dims: Sequence[int] = (1, -1, 1, 1), axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"), backend=""
    ):
        """
        The create_mesh function creates a mesh object that can be used to shard arrays.

        :param axis_dims: Sequence[int]: Specify the dimensions of the mesh
        :param axis_names: Sequence[str]: Name the axes of the mesh
        :param backend: Specify the backend to use
        :return: A mesh object

        """
        array_devices = jax.numpy.ones(
            (len(jax.devices() if backend == "" else jax.devices(backend)), 1))
        resh = array_devices.reshape(axis_dims).shape

        return Mesh(
            create_device_mesh(resh), axis_names
        )

    def jax_mesh(self) -> Mesh:
        """
        The jax_mesh function is a helper function that creates a Mesh object from the
        axis_dims and axis_names attributes of an object, which are assumed to be lists of integers and strings, respectively.
        The backend attribute is also used if it exists.

        :param self: Refer to the object itself
        :return: A jaxMesh

        """
        return self.create_mesh(
            axis_dims=[v for k, v in self.axis_dims.items()] if isinstance(self.axis_dims, dict) else self.axis_dims,
            axis_names=[v for k, v in self.axis_names.items()] if isinstance(self.axis_names,
                                                                             dict) else self.axis_names,
            backend=(self.backend if self.backend is not None else "") if hasattr(
                self, 'backend') else ""
        )

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):

        """
        The get_partition_rules function is used to specify how the parameters of a model are partitioned across devices.

        :param self: Access the attributes of the class
        :param fully_sharded_data_parallel: bool: Determine whether the model is fully sharded or not
        :return: A tuple of tuples
        """
        if not fully_sharded_data_parallel:
            raise NotImplementedError()
        else:
            return (
                ('.*', PartitionSpec(("fsdp", "sp")))
            )

    def get_axis_dims(self) -> Sequence[int]:
        """
        The get_axis_dims function returns a sequence of integers representing the dimensions of each axis.

        :param self: Represent the instance of the class
        :return: The dimensions of the axes

        """
        return self.axis_dims

    def get_axis_names(self) -> Sequence[str]:
        """
        The get_axis_names function returns a list of the names of the axes.

        :param self: Represent the instance of the class
        :return: A list of the names of all axes

        """
        return self.axis_names

    def get_backend(self) -> str:
        """
        The get_backend function returns the backend that is currently being used.
        If no backend has been set, it will return the default JAX backend.

        :param self: Bind the method to an object
        :return: The backend platform

        """
        return self.backend if not self.backend == "" else jax.lib.xla_bridge.get_backend().platform

    def add_basic_configurations(
            self,
            axis_dims: Sequence[int] = ...,
            axis_names: Sequence[str] = ...,
            attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = ...,
            block_k: int = ...,
            block_q: int = ...,
            block_b: int = ...,
            block_k_major: int = ...,
            block_q_major_dkv: int | None = ...,
            block_k_major_dkv: int | None = ...,
            block_k_dkv: int | None = ...,
            block_q_dkv: int | None = ...,
            block_k_major_dq: int | None = ...,
            block_k_dq: int | None = ...,
            block_q_dq: int | None = ...,
            query_partition_spec: PartitionSpec = ...,
            key_partition_spec: PartitionSpec = ...,
            value_partition_spec: PartitionSpec = ...,
            bias_partition_spec: PartitionSpec = ...,
            attention_partition_spec: PartitionSpec = ...,
            use_shard_map: bool = ...,
            use_sharded_kv_caching: bool = ...,
            backend: Optional[None] = ...,
            easy_method: Literal["train", "serve", "convert"] = ...,
            bits: Optional[int] = ...,
            scan_ring_attention: bool = ...,
            scan_attention_layers: bool = ...,
            use_scan_mlp: bool = ...,
            scan_mlp_chunk_size: int = ...
    ):
        """
        It initializes all the attributes of an object, and it's called when you create a new instance of that class.
        :param self: Refer to the instance of the class
        :param axis_dims: Sequence[int]: Specify the number of dimensions for each axis
        :param axis_names: Sequence[str]: Set the names of the axes
        :param attn_mechanism: Literal["normal", "flash", "splash"]: attention mechanism to use
        :param block_k: int: block size of key_states
        :param block_q: int: block size of query_states
        :param block_b: int: block size of bias
        :param block_k_major: int: block size if key major
        :param block_q_major_dkv: int: block size of block_q_major_dkv
        :param block_k_major_dkv: int: block size of block_k_major_dkv
        :param block_k_dkv: int: block size of block_k_dkv
        :param block_q_dkv: int: block size of block_q_dkv
        :param block_k_major_dq: int: block size of block_k_major_dq
        :param block_k_dq: int: block size of block_k_dq
        :param block_q_dq: int: block size of block_q_dq
        :param query_partition_spec: PartitionSpec: Specify the partitioning of the query tensor
        :param key_partition_spec: PartitionSpec: Partition the key matrix
        :param value_partition_spec: PartitionSpec: Specify the partitioning of the value tensor
        :param bias_partition_spec: PartitionSpec: Specify the Attention Bias partition spec
        :param attention_partition_spec: PartitionSpec: Specify the partitioning of the attention weights
        :param use_shard_map: bool: whenever to use shard_map for attention
        :param use_sharded_kv_caching: bool: whenever to use shard_map and sharding for key and value
        :param backend: Optional[None]: Specify the backend to use
        :param easy_method: Literal["train", "serve", "convert"]: EasyDel Quantization Method to be applied for
        :param bits: Optional[int]: Model bits for quantization
        :param scan_ring_attention: bool: Whether to use can for ring attention
        :param scan_attention_layers: bool: Whether to use can for attention layers
        :param use_scan_mlp: bool: Determine whether to use scan_mlp or not
        :param scan_mlp_chunk_size: int: Size of chunks in scan MLP.
        """
        set_attrs_smartly(self, "axis_dims", (1, -1, 1, 1), axis_dims)
        set_attrs_smartly(self, "axis_names", ("dp", "fsdp", "tp", "sp"), axis_names)

        set_attrs_smartly(self, "block_q", 128, block_q)
        set_attrs_smartly(self, "block_k", 128, block_k)
        set_attrs_smartly(self, "block_b", 1, block_b)

        set_attrs_smartly(self, "query_partition_spec", PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                          query_partition_spec)
        set_attrs_smartly(self, "key_partition_spec", PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                          key_partition_spec)
        set_attrs_smartly(self, "value_partition_spec", PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                          value_partition_spec)
        set_attrs_smartly(self, "bias_partition_spec", PartitionSpec(("dp", "fsdp"), None, None, None),
                          bias_partition_spec)
        set_attrs_smartly(self, "attention_partition_spec", PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                          attention_partition_spec)

        set_attrs_smartly(self, "backend", jax.default_backend(), backend)
        set_attrs_smartly(self, "use_shard_map", False, use_shard_map)
        set_attrs_smartly(self, "use_sharded_kv_caching", True, use_sharded_kv_caching)
        set_attrs_smartly(self, "attn_mechanism", "normal", attn_mechanism)

        set_attrs_smartly(self, "block_k_dkv", block_k_dkv or self.block_k, block_k_dkv)
        set_attrs_smartly(self, "block_q_dkv", block_q_dkv or self.block_q, block_q_dkv)

        set_attrs_smartly(self, "block_q_major_dkv", block_q_major_dkv or self.block_q, block_q_major_dkv)
        set_attrs_smartly(self, "block_k_major_dkv", block_k_major_dkv or self.block_k, block_k_major_dkv)

        set_attrs_smartly(self, "block_k_major", block_k_major or self.block_k, block_k_major)
        set_attrs_smartly(self, "block_k_major_dq", block_k_major_dq or self.block_k, block_k_major_dq)

        set_attrs_smartly(self, "block_k_dq", block_k_dq or self.block_k, block_k_dq)
        set_attrs_smartly(self, "block_q_dq", block_q_dq or self.block_q, block_q_dq)

        set_attrs_smartly(self, "easy_method", EasyMethod.TRAIN, easy_method)
        set_attrs_smartly(self, "bits", None, bits)
        set_attrs_smartly(self, "scan_attention_layers", True, scan_attention_layers)
        set_attrs_smartly(self, "scan_ring_attention", False, scan_ring_attention)
        set_attrs_smartly(self, "use_scan_mlp", True, use_scan_mlp)
        set_attrs_smartly(self, "scan_mlp_chunk_size", 1024, scan_mlp_chunk_size)

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
                    pass
        return string + ")"

    def __str__(self):

        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()


class EasyDelFlaxPretrainedModel(FlaxPreTrainedModel):
    def __init__(
            self,
            config: PretrainedConfig,
            module: flax.linen.Module,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,  # Ignored
            precision: Optional[Union[jax.lax.Precision, str]] = None,  # Ignored
            _do_init: bool = True,
    ):
        super().__init__(
            config=config,
            module=module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init
        )

    def get_input_embeddings(self):
        """
        The get_input_embeddings function returns the embedding layer of the model.

        :param self: Refer to the current object
        :return: The embedding layer of the model
        """
        raise NotImplementedError()

    def set_input_embeddings(self, value):
        """
        The set_input_embeddings function is used to set the embedding module of the model.

        :param self: Represent the instance of the class
        :param value: Set the embeddings of the model
        """
        raise NotImplementedError()

    def get_output_embeddings(self):
        """
        The get_output_embeddings function returns the output embeddings of a model.

        :param self: Represent the instance of the class
        :return: The output embeddings of the model
        """
        raise NotImplementedError()

    def set_output_embeddings(self, new_embeddings):
        """
        The set_output_embeddings function is used to set the output embeddings of a model.
        This function can be used to change the output embedding layer of a pretrained model in order to finetune it
        to some downstream task. Changing this layer has an effect only if the model has already been fine-tuned on some
        task (e.g., for classification). If you are training your own language models, you should call this function before
        you start training.

        :param self: Represent the instance of the class
        :param new_embeddings: Set the embeddings of the output layer
        :return: A new embedding layer
        """
        raise NotImplementedError()

    def set_decoder(self, decoder):
        """
        The set_decoder function is used to set the decoder for a given encoder.

        :param self: Refer to the object itself
        :param decoder: Set the decoder for a given encoder
        :return: A decoder
        """
        raise NotImplementedError()

    def get_decoder(self):
        """
        The get_decoder function is used to create a decoder object.

        :param self: Represent the instance of the class
        :return: A decoder object
        """
        raise NotImplementedError()

    def init_cache(self, batch_size: int, max_length: int):
        raise NotImplementedError("init_cache is not Implemented Yet!")

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[chex.Array] = None):
        """
        The prepare_inputs_for_generation function is used to prepare the inputs for a generation task.

        :param self: Access variables that belong to the class
        :param input_ids: Pass in the input tokens
        :param max_length: Set the length of the sequence to be generated
        :param attention_mask: Optional[chex.Array]: Mask the attention weights
        :return: A dictionary of the past_key_values, attention_mask and position ids

        """
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones(
            (batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[
                                            None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

    def __call__(
            self,
            input_ids: chex.Array,
            attention_mask: chex.Array = None,
            position_ids: chex.Array = None,
            params: dict = None,
            past_key_values: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
            add_params_field: bool = False,
            **kwargs
    ):
        raise NotImplementedError("Not Implemented Yet")

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
                    pass
        return string + ")"

    def __str__(self):

        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
