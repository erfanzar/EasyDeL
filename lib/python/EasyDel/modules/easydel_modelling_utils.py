import chex
from jax.experimental.mesh_utils import create_device_mesh
from transformers import PretrainedConfig, FlaxPreTrainedModel
import jax
from jax import numpy as jnp
from typing import Sequence, Union, Optional
from dataclasses import dataclass


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
    :param q_ps: jax.sharding.PartitionSpec: Specify the partitioning of the query tensor
    :param k_ps: jax.sharding.PartitionSpec: Partition the key matrix
    :param v_ps: jax.sharding.PartitionSpec: Specify the partitioning of the value tensor
    :param b_ps: jax.sharding.PartitionSpec: Specify the Attention Bias partition spec
    :param a_ps: jax.sharding.PartitionSpec: Specify the partitioning of the attention weights
    :param use_shard_map: bool: whenever to use shard_map for attention
    :param backend: Optional[None]: Specify the backend to use
    :param easy_method: EasyMethod: Specify the use of model to init the QDot Method for (e.q TRAIN,SERVE,...)
    """

    def __init__(
            self,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            q_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None),
            k_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None),
            v_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None),
            b_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), None, None, None),
            a_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            backend: Optional[None] = jax.default_backend(),
            easy_method: EasyMethod = EasyMethod.TRAIN,
            **kwargs
    ):
        self.q_ps = q_ps
        self.k_ps = k_ps
        self.v_ps = v_ps
        self.b_ps = b_ps
        self.a_ps = a_ps
        self.use_shard_map = use_shard_map
        self.axis_dims = axis_dims
        self.axis_names = axis_names
        self.backend = backend if backend is not None else ""
        self.easy_method = easy_method
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

        return jax.sharding.Mesh(
            create_device_mesh(resh), axis_names
        )

    def jax_mesh(self) -> jax.sharding.Mesh:
        """
        The jax_mesh function is a helper function that creates a jax.sharding.Mesh object from the
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
                ('.*', jax.sharding.PartitionSpec(("fsdp", "sp")))
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

    def add_partitions(
            self,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            q_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None
            ),
            k_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None
            ),
            v_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None
            ),
            b_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), None, None, None
            ),
            a_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp", "tp", None
            ),
            use_shard_map: bool = False,
            backend: Optional[str] = None,
    ):
        """
            It initializes all the attributes of an object, and it's called when you create a new instance of that class.
            :param self: Refer to the instance of the class
            :param axis_dims: Sequence[int]: Specify the number of dimensions for each axis
            :param axis_names: Sequence[str]: Set the names of the axes
            :param q_ps: jax.sharding.PartitionSpec: Specify the partitioning of the query tensor
            :param k_ps: jax.sharding.PartitionSpec: Partition the key matrix
            :param v_ps: jax.sharding.PartitionSpec: Specify the partitioning of the value tensor
            :param b_ps: jax.sharding.PartitionSpec: Specify the Attention Bias partition spec
            :param a_ps: jax.sharding.PartitionSpec: Specify the partitioning of the attention weights
            :param use_shard_map: bool: whenever to use shard_map for attention
            :param backend: Optional[None]: Specify the backend to use
            """
        self.axis_dims = axis_dims
        self.axis_names = axis_names
        self.q_ps = q_ps
        self.k_ps = k_ps
        self.v_ps = v_ps
        self.b_ps = b_ps
        self.a_ps = a_ps
        self.backend = backend
        self.use_shard_map = use_shard_map


class EasyDelFlaxPretrainedModel(FlaxPreTrainedModel):

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

    def __str__(self):
        padded_model = "\t" + "\n\t".join(self.module.__str__().split("\n"))
        string = f"{self.__class__.__name__}(\n{padded_model}\n)"
        return string

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
