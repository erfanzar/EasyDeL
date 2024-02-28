import copy
import functools
import os
from typing import Optional, Mapping, Callable, Dict, Any

import flax.core
import gradio as gr
import jax
import msgpack
import tqdm
import transformers
import uvicorn
from fastapi import FastAPI
from fjformer import make_shard_and_gather_fns, match_partition_rules, with_sharding_constraint

from ..etils.etils import get_logger
from ..smi import get_mem, initialise_tracking
from jax import numpy as jnp
from jax.experimental import mesh_utils
from flax.serialization import from_bytes
from fjformer.checkpoint import get_dtype
from jax.sharding import Mesh, PartitionSpec
from transformers import GenerationConfig
from ..utils.utils import RNG
import multiprocessing as mp
from typing import Union, Sequence, List
import chex
from .utils import InstructRequest, ChatRequest
from jax.experimental.pjit import pjit
from .gradio_user_interface_base import GradioUserInference
from ..modules.auto_easydel_model import AutoEasyDelModelForCausalLM
from dataclasses import dataclass
import logging

logger = get_logger(__name__)


@dataclass
class JAXServerConfig:
    """
    :param host: str: Set the host address of the server
    :param port: int: Specify the port number that the server will run on
    :param batch_size: int: Set the batch size of the model
    :param max_sequence_length: int: Set the maximum length of the text that can be generated
    :param max_new_tokens: int: Determine how many tokens can be added to the vocabulary
    :param max_compile_tokens: int: Set the maximum number of tokens that can be streamed at a time
    :param generation_ps: PartitionSpec : PartitionSpec to use for sharding data
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

    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None

    logging: bool = True

    mesh_axes_names: Sequence[str] = ("dp", "fsdp", "tp", "sp")
    mesh_axes_shape: Sequence[int] = (1, -1, 1, 1)
    generation_ps: PartitionSpec = PartitionSpec("dp", "fsdp")

    dtype: str = "fp16"

    stream_tokens_for_gradio: bool = True
    use_prefix_tokenizer: bool = True
    pre_compile: bool = True

    use_mxn_break_point: bool = True

    def __post_init__(self):
        assert self.max_new_tokens % self.max_compile_tokens == 0, (
            f"max_new_tokens should be divisible by max_compile_tokens  {self.max_new_tokens % self.max_compile_tokens}"
        )

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


class JAXServer(GradioUserInference):

    def __init__(self, server_config=None):

        """
        The __init__ function is called when the class is instantiated.
        It sets up all the attributes that will be used by other methods in the class.


        :param self: Refer to the current instance of a class
        :param server_config: Pass the JAXServerConfig object
        :return: A fastapi object
        
        """
        (
            self.process_uvicorn,
            self.prefix_tokenizer,
            self.params,
            self.tokenizer,
            self.model,
            self.partition_specs,
            self.generate_function,
            self.greedy_generate_function
        ) = [None] * 8
        assert server_config is None or isinstance(server_config,
                                                   JAXServerConfig), "server_config can be None or JAXServerConfig Type"
        if server_config is None:
            server_config = JAXServerConfig()

        self.server_config = server_config
        self._funcs_generated = False
        self.number_of_served_request_until_last_up_time = 0

        self.rng_generator = RNG(42)
        initialise_tracking(0.5)
        array = jnp.ones((len(jax.devices()), 1)).reshape(self.server_config.mesh_axes_shape)
        self.mesh = Mesh(mesh_utils.create_device_mesh(array.shape), self.server_config.mesh_axes_names)

        self.app = FastAPI()
        self.app.post("/chat")(self.forward_chat)
        self.app.post("/instruct")(self.forward_instruct)
        self.app.get("/status")(self.status)
        self.app = gr.mount_gradio_app(self.app, self.gradio_inference(), "/gradio_chat")

    def status(self):
        """
        The status function returns a dictionary with the following keys:
            server_config: A dictionary containing all the configuration parameters for this server.
            devices: A string describing which devices are available to JAX.
            number_of_backends: The number of backends available to JAX.  This is usually equal to the number of GPUs
            on your machine, but can be less if you have not installed CUDA or if you have disabled some GPUs in your
             system BIOS settings (e.g., because they are defective).  It can also be more than one if you have multiple
              machines connected via MPI and running under Horov

        :param self: Represent the instance of the class
        :return: A dictionary with the following keys:
        
        """
        return {
            "server_config": {k: v for k, v in self.server_config.__dict__.items()},
            "devices": f"{jax.devices()}",
            "number_of_backends": len(jax.devices()),
            "status": "Ready",
            "number_of_served_request_until_last_up_time": f"{self.number_of_served_request_until_last_up_time}",
            "memory": f"{get_mem()}"
        }

    @staticmethod
    def get_memory():
        """
        The get_memory function returns the total memory of the system in bytes.


        :return: The amount of memory used by the program
        
        """
        return get_mem()

    def configure_generate_functions(self, model, tokenizer):

        """
        The configure_generate_functions function is used to configure the generation functions for a given model.

        :param self: Access variables within the class
        :param model: Generate the model
        :param tokenizer: Get the eos_token_id, pad_token_id and bos token id
        :return: A function that takes in three parameters:
        
        """
        assert self.partition_specs is not None, "you should first shard params with using ``shard_params`` method"

        if tokenizer.pad_token is None:
            logging.info(
                "Tokenizer does not contain padding token setting padding token to eos token for open end generation")
            tokenizer.pad_token = tokenizer.eos_token

        try:
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            self.prefix_tokenizer = copy.deepcopy(tokenizer)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
            self.tokenizer = copy.deepcopy(tokenizer)
        except:
            logger.warning(
                f"The class Model of Tokenizer {type(tokenizer)} do not support deepcopy option "
            )
            if self.server_config.use_prefix_tokenizer:
                tokenizer.padding_side = "left"
                tokenizer.truncation_side = "left"
            else:
                tokenizer.padding_side = "right"
                tokenizer.truncation_side = "right"
            self.prefix_tokenizer = tokenizer

        @functools.partial(
            pjit,
            in_shardings=(self.partition_specs, PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def greedy_generate(parameters, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, self.server_config.generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, self.server_config.generation_ps)
            predict = model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_new_tokens=self.server_config.max_compile_tokens,

                    eos_token_id=self.server_config.eos_token_id or tokenizer.eos_token_id,
                    pad_token_id=self.server_config.pad_token_id or tokenizer.pad_token_id,
                    bos_token_id=self.server_config.bos_token_id or tokenizer.bos_token_id,

                    do_sample=False,
                    num_beams=1,
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        @functools.partial(
            pjit,
            in_shardings=(self.partition_specs, PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def generate(parameters, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, self.server_config.generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, self.server_config.generation_ps)
            predict = model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_new_tokens=self.server_config.max_compile_tokens,

                    eos_token_id=self.server_config.eos_token_id or tokenizer.eos_token_id,
                    pad_token_id=self.server_config.pad_token_id or tokenizer.pad_token_id,
                    bos_token_id=self.server_config.bos_token_id or tokenizer.bos_token_id,

                    temperature=self.server_config.temperature,
                    do_sample=True,
                    num_beams=1,
                    top_p=self.server_config.top_p,
                    top_k=self.server_config.top_k,
                    repetition_penalty=self.server_config.repetition_penalty
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        self.generate_function = generate
        self.greedy_generate_function = greedy_generate
        self._funcs_generated = True

    def auto_configure(self, model, params, tokenizer, partition_rules):
        """
        The auto_configure function is a helper function that will automatically configure the model for distributed training.
        It does this by:
            1) sharding the parameters of the model based on partition_rules, and then
            2) configuring generate functions to be used in distributed training.

        :param self: Represent the instance of the class
        :param model: Configure the model
        :param params: Store the parameters that are used to configure the model
        :param tokenizer: Tokenize the input text
        :param partition_rules: Specify how the parameters should be partitioned
        :return: A dictionary with the following keys:
        
        """
        self.shard_params(params=params, partition_rules=partition_rules)
        self.configure_generate_functions(model, tokenizer)

    def generate(
            self,
            params: Union[flax.core.FrozenDict, dict],
            input_ids: chex.Array,
            attention_mask: chex.Array,
    ):
        """
        The generate function is used to generate a sequence of tokens from the model.

        :param self: Access variables that belong to the class
        :param params: Union[flax.core.FrozenDict, dict]: Pass the parameters of the model to be used in generating text
        :param input_ids: chex.Array: Pass the input to the model
        :param attention_mask: chex.Array: Mask the padding tokens
        :return: The logits of the model
        
        """
        if not self._funcs_generated:
            raise NotImplementedError(
                "this method will be implemented automatically after using ``configure_generate_functions`` function"
            )
        else:
            with self.mesh:
                return self.generate_function(
                    params, input_ids, attention_mask
                )

    @classmethod
    def load(
            cls,
            model: transformers.FlaxPreTrainedModel,
            config_model: transformers.PretrainedConfig,
            tokenizer: transformers.PreTrainedTokenizer,
            path: Union[str, os.PathLike],
            server_config=None,
            add_params_field: bool = True,
            init_shape: tuple = (1, 1),
            do_memory_log: bool = False,
            verbose: bool = True
    ):
        """
        The load function is used to load a pretrained model from disk.

        :param cls: Refer to the class itself
        :param model: transformers.FlaxPreTrainedModel: Initialize the server
        :param config_model: transformers.PretrainedConfig: Get the partition rules
        :param tokenizer: transformers.PreTrainedTokenizer: Load the tokenizer from the model
        :param path: Union[str, os.PathLike]: Specify the path to the checkpoint file
        :param server_config: Configure the server
        :param add_params_field: bool: Add a params field to the server
        :param init_shape: tuple: Specify the shape of the input to be used for generating shard_fns
        :param do_memory_log: bool: Log the memory usage of the server
        :param verbose: bool: Print the compilation process
        :return: A server
        
        """
        assert hasattr(model,
                       "init_weights"), "model must contain init_weights func in order to init params for shard_fns"
        assert hasattr(config_model,
                       "get_partition_rules"), "config_model must contain get_partition_rules functions"
        server = cls(server_config=server_config)
        logging.info(
            "running _init() func in order to make shard_fns"
        )
        with jax.default_device(jax.devices("cpu")[0]):
            def _init():
                return model.init_weights(jax.random.PRNGKey(0), init_shape)

            shape = jax.eval_shape(_init)
        logging.info(
            "matching partition rules"
        )
        rules = match_partition_rules(params=shape, rules=config_model.get_partition_rules(True))

        with server.mesh:
            shard_fns, _ = make_shard_and_gather_fns(rules, get_dtype(server.server_config.dtype))
            logging.info(
                "loading checkpoints"
            )

            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
            server.params = {}
            with open(path, "rb") as stream:
                unpacker = msgpack.Unpacker(stream, read_size=83886080, max_buffer_size=0)
                pbar = tqdm.tqdm(unpacker)
                for key, value in pbar:
                    key = tuple(key)
                    tensor = from_bytes(None, value)
                    tensor = shard_fns[key](tensor)
                    server.params[key] = tensor
                    if do_memory_log:
                        pbar.write(server.get_memory())
                    pbar.set_description("Sharding Params")
        server.params = flax.traverse_util.unflatten_dict(server.params)
        server.params = {"params": server.params} if add_params_field else server.params

        server.rules = {"params": rules} if add_params_field else rules
        logging.info(
            "configuring generate functions for the server"
        )
        server.configure_generate_functions(model, tokenizer)

        if server.server_config.pre_compile:
            server.compile(verbose=verbose)
        return server

    @classmethod
    def from_torch_pretrained(
            cls,
            server_config: JAXServerConfig,
            pretrained_model_name_or_path: str,
            device=jax.devices('cpu')[0],
            dtype: jax.numpy.dtype = jax.numpy.float32,
            param_dtype: jax.numpy.dtype = jax.numpy.float32,
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            key_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            value_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None, None),
            attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            input_shape: Sequence[int] = (1, 1),
            shard_fns: Optional[Mapping[tuple, Callable]] = None,
            backend: Optional[str] = None,
            add_params_field: bool = True,
            do_memory_log: bool = False,
            model_config_kwargs: Optional[Mapping[str, Any]] = None,
            verbose: bool = True,
            **kwargs
    ):

        model, params = AutoEasyDelModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            sharding_axis_names=sharding_axis_names,
            sharding_axis_dims=sharding_axis_dims,
            query_partition_spec=query_partition_spec,
            attention_partition_spec=attention_partition_spec,
            value_partition_spec=value_partition_spec,
            key_partition_spec=key_partition_spec,
            bias_partition_spec=bias_partition_spec,
            use_shard_map=use_shard_map,
            shard_fns=shard_fns,
            input_shape=input_shape,
            backend=backend,
            config_kwargs=model_config_kwargs,
            **kwargs
        )

        return cls.from_parameters(
            model=model,
            config_model=model.config,
            tokenizer=transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            params=params,
            server_config=server_config,
            verbose=verbose,
            do_memory_log=do_memory_log,
            add_params_field=add_params_field
        )

    @classmethod
    def from_parameters(
            cls,
            model: transformers.FlaxPreTrainedModel,
            config_model: transformers.PretrainedConfig,
            tokenizer: transformers.PreTrainedTokenizer,
            params: Dict,
            server_config: JAXServerConfig = None,
            add_params_field: bool = True,
            do_memory_log: bool = False,
            verbose: bool = True
    ):
        """
        The from_parameters function is used to load a model from the parameters of a pretrained model.
        It takes in the following arguments:
            - cls: The class of the server you are loading, this should be Server or TPU_Server depending on
            what backend you want to use.
            - model: A FlaxPreTrainedModel object that contains all of your models functions and parameters. This can
             be found in transformers/flax_utils/models/*model*.py
                where *model* is replaced with whatever transformer you are using (e.g., bert). You can also create
                 your own custom

        :param cls: Create a new instance of the class
        :param model: transformers.FlaxPreTrainedModel: Load the model
        :param config_model: transformers.PretrainedConfig: Get the partition rules
        :param tokenizer: transformers.PreTrainedTokenizer: Tokenize the input text
        :param params: Dict: Pass in the parameters of the model
        :param server_config: Pass in the server_config file for the server
        :param add_params_field: bool: Add a params field to the server
        :param do_memory_log: bool: Log the memory usage of the server
        :param verbose: bool: Print out the status of the compilation
        :return: A server object
        
        """
        assert hasattr(model, "init_weights"), (
            "model must contain init_weights func in order to init params for shard_fns"
        )
        assert hasattr(config_model, "get_partition_rules"), (
            "config_model must contain get_partition_rules functions"
        )
        server = cls(server_config=server_config)

        with server.mesh:
            logging.info(
                "matching partition rules"
            )
            partition_specs = match_partition_rules(params=params, rules=config_model.get_partition_rules(True))
            shard_fns, _ = make_shard_and_gather_fns(partition_specs, get_dtype(server.server_config.dtype))
            logging.info(
                "sharding parameters across all of the chosen backend(tpu/gpu/cpu)s"
            )
            params = flax.traverse_util.flatten_dict(params)
            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
            pbar = tqdm.tqdm(params.keys())
            for key in pbar:
                key = tuple(key)
                params[key] = shard_fns[key](params[key])
                if do_memory_log:
                    pbar.write(server.get_memory())
                pbar.set_description("Sharding Params")
            server.params = flax.traverse_util.unflatten_dict(params)
            server.params = {"params": server.params} if add_params_field else server.params
        server.partition_specs = {"params": partition_specs} if add_params_field else partition_specs
        logging.info(
            "configuring generate functions for the server"
        )
        server.configure_generate_functions(model, tokenizer)
        if server.server_config.pre_compile:
            server.compile(verbose=verbose)
        return server

    def compile(self, verbose: bool = True) -> bool:
        """
        The compile function is used to compile the model for use in inference.
        It does this by running through all possible combinations of rules and actions,
        and compiling them into functions that can be called later on during inference.
        This allows us to avoid having to recompile the model every time we want to run it,
        which would be very slow.

        :param self: Represent the instance of the class
        :param verbose: bool: Print out the compiling process
        :return: True, but what does it do?
        """
        assert self._funcs_generated, "funcs are not generated yet"
        assert self.partition_specs is not None, "rules should not be None"
        if self.server_config.use_prefix_tokenizer:
            if verbose:
                logger.info("Compiling greedy generate function")

            r, a = [None] * 2
            for r, a in self.sample(
                    string="",
                    max_new_tokens=self.server_config.max_compile_tokens,
                    greedy=True
            ):
                ...
            if verbose:
                logger.info("Compiling non-greedy generate function")
            for r, a in self.sample(
                    string="",
                    max_new_tokens=self.server_config.max_compile_tokens,
                    greedy=False
            ):
                ...

        else:
            logger.warning(
                "Skip Compiling the compiling process is useless "
                "when you are not using prefix tokenizer",
            )
        return True

    def greedy_generate(self,
                        params: Union[flax.core.FrozenDict, dict],
                        input_ids: chex.Array,
                        attention_mask: chex.Array,
                        ):
        """
        The greedy_generate function is a helper function that takes in the model parameters, input_ids and attention_mask
        and returns the generated tokens. It uses greedy search to generate tokens one at a time.


        :param self: Refer to the object itself
        :param params: Union[flax.core.FrozenDict, dict]: Pass the parameters to the model
        :param input_ids: chex.Array: Pass in the input sequence
        :param attention_mask: chex.Array: Mask the input tokens
        :param : Specify the parameters of the model
        :return:  generated_ids
        
        """
        if not self._funcs_generated:
            raise NotImplementedError(
                "this method will be implemented automatically after using ``configure_generate_functions`` function"
            )
        else:
            with self.mesh:
                return self.greedy_generate_function(
                    params, input_ids, attention_mask
                )

    def shard_params(self, params, partition_rules):

        """
        The shard_params function takes in a set of parameters and a partition rule.
        The partition rule is used to determine how the parameters should be sharded across devices.
        For example, if we have two devices, one with 4GB of memory and another with 8GB of memory,
        we may want to shard our model such that the device with more memory has more parameters on it.
        This function returns an updated version of params where each parameter is now stored on its own device.

        :param self: Bind the instance of the class to a method
        :param params: Pass the parameters of the model to be sharded
        :param partition_rules: Specify how the parameters should be partitioned
        :return: The sharded parameters
        
        """
        logging.log(
            logging.INFO,
            "the parameters will be sharded and ba saved inside server you can access them by ``JAXServer.params``")
        rules = match_partition_rules(params=params, rules=partition_rules)
        self.partition_specs = rules
        shard_fns, _ = make_shard_and_gather_fns(rules, get_dtype(self.server_config.dtype))

        with self.mesh:
            self.params = jax.tree_map(
                lambda f, p: f(p), shard_fns, params
            )

        return self.params

    def forward_chat(self, data: ChatRequest):

        """
        The forward_chat function is the main function of this class.
        It takes in a ChatRequest object, which contains a prompt and history.
        The prompt is the user"s input to be processed by the chatbot, while history
        is an array of previous inputs and outputs from both sides (user and bot).
        The forward_chat function then formats these inputs into one string that can be processed by our model.
        This formatted string is then passed through our sample() method, which returns an output response as well as
        how many tokens were used to generate it.

        :param self: Access the attributes and methods of the class
        :param data: ChatRequest: Pass in the data from the request
        :return: A dictionary with the following keys:
        
        """
        if not self._funcs_generated:
            return {
                "status": "down"
            }

        string = self.format_chat(
            prompt=data.prompt,
            system=None,
            history=data.history
        )

        response, used_tokens = [None] * 2
        for response, used_tokens in self.sample(
                string=string,
                greedy=data.greedy,
                max_new_tokens=None
        ):
            ...
        self.number_of_served_request_until_last_up_time += 1
        return {
            "input": f"{string}",
            "response": response,
            "tokens_used": used_tokens,
        }

    @staticmethod
    def format_instruct(system: str, instruction: str) -> str:
        """
        Here you will get the system and instruction from user, and you can apply your prompting style
        """
        raise NotImplementedError()

    @staticmethod
    def format_chat(history: List[List[str]], prompt: str, system: Union[str, None]) -> str:
        """
        Here you will get the system, prompt and history from user, and you can apply your prompting style
        """
        raise NotImplementedError()

    def forward_instruct(self, data: InstructRequest):
        """
        The forward_instruct function is the main function of this class.
        It takes in a InstructRequest object, which contains the system and instruction to be processed.
        The function then formats the input string using format_instruct, and passes it into sample().
        sample() returns a tuple containing (response, used_tokens). The response is returned as part of
        the response dictionary. If no valid responses are found by sample(), None will be returned instead.

        :param self: Bind the method to the object
        :param data: InstructRequest: Pass the system and instruction to the function
        :return: A dictionary with three keys:
        
        """
        if not self._funcs_generated:
            return {
                "status": "down"
            }

        response, used_tokens = [None] * 2
        string = self.format_instruct(
            system=data.system,
            instruction=data.instruction
        )
        for response, used_tokens in self.sample(
                string=string,
                greedy=data.greedy,
                max_new_tokens=None
        ):
            ...
        self.number_of_served_request_until_last_up_time += 1
        return {
            "input": f"{string}",
            "response": response,
            "tokens_used": used_tokens,
        }

    def forward_instruct_non_api(self, prompt, system, greedy):
        """
        The forward_instruct_non_api function is a wrapper for the forward_instruct function.
        It takes in a prompt, system, and greedy flag as arguments and returns the response from
        the forward_instruct function. The purpose of this wrapper is to allow users to call
        forward_instruct without having to create an InstructRequest object.

        :param self: Represent the instance of the class
        :param prompt: Pass the instruction to the system
        :param system: Specify which system to use for the instruction
        :param greedy: Determine whether the system should return
        :return: The response from the forward_instruct function
        
        """
        data = InstructRequest(
            prompt=prompt,
            system=system,
            greedy=greedy
        )
        return self.forward_instruct(data)

    def forward_chat_non_api(self, prompt, history, greedy):
        """
        The forward_chat_non_api function is a wrapper for the forward_chat function.
        It takes in a prompt, history, and greedy parameter and returns the response from
        the forward_chat function. The purpose of this wrapper is to allow users to use
        the chatbot without having to create ChatRequest objects.

        :param self: Represent the instance of the class
        :param prompt: Pass the user's input to the model
        :param history: Pass the history of the conversation to the model
        :param greedy: Determine whether the model should use a greedy search
        :return: A chat-response object
        
        """
        data = ChatRequest(
            prompt=prompt,
            history=history,
            greedy=greedy
        )
        return self.forward_chat(data)

    def sample_gradio(
            self,
            prompt: str,
            history: List[List[str]],
            system_prompt: str | None,
            mode: str,
            max_sequence_length: int,
            max_new_tokens: int,
            max_compile_tokens: int,
            greedy: bool,
            temperature: float,
            top_p: float,
            top_k: int,
            repetition_penalty: float
    ):
        if mode.lower() == "chat":
            string = self.format_chat(
                history=history,
                system=system_prompt,
                prompt=prompt
            )
        elif mode.lower() == "instruct":
            history = []
            string = self.format_instruct(
                system=system_prompt,
                instruction=prompt
            )
        else:
            raise ValueError("UnKnown Mode for sample_gradio available modes are only Chat or Instruct")
        history.append([prompt, ""])
        responses = ""
        for response, _ in self.sample(
                string=string,
                greedy=greedy,
                max_new_tokens=max_new_tokens,
        ):
            responses += response
            history[-1][-1] = responses
            yield "", history

    def sample(self,
               string: str,
               *,
               greedy: bool = False,
               max_new_tokens: int = None,
               **kwargs
               ):
        """
        The sample function is the main function of a model. It takes in an input string and returns a list of strings
        that are generated from that input string. The sample function can be called multiple times with different inputs,
        and each time it will return a new set of outputs based on those inputs.

        :param self: Access the class attributes
        :param string: str: Pass the string that we want to generate
        :param *: Pass a variable number of arguments to a function
        :param greedy: bool: Determine whether to use the greedy or non-greedy version of the generate function
        :param max_new_tokens: int: Set the number of tokens to generate
        :param kwargs: Pass any additional parameters to the sample function
        :return: A generator that yields the predicted text and the number of tokens generated
        
        """

        fixed_pad = self.server_config.max_sequence_length - self.server_config.max_compile_tokens
        tokens = self.prefix_tokenizer(
            string,
            max_length=fixed_pad,
            padding="max_length",
            return_tensors="jax"
        ) if self.server_config.use_prefix_tokenizer else self.tokenizer(
            string,
            return_tensors="jax"
        )

        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        num_generated_tokens = 0

        for _ in range((max_new_tokens or self.server_config.max_new_tokens) // self.server_config.max_compile_tokens):
            inputs_to_gen = dict(
                params=self.params,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predicted_token = self.greedy_generate(**inputs_to_gen) if greedy else self.generate(**inputs_to_gen)
            predicted_token = predicted_token[
                predicted_token != self.tokenizer.pad_token_id if (
                        self.server_config.pad_token_id is None
                ) else predicted_token != self.server_config.pad_token_id
            ]
            if predicted_token.ndim == 1:
                predicted_token = predicted_token.reshape(1, -1)
            num_generated_tokens += predicted_token.shape[-1]
            plus_attn_mask = jnp.ones((len(attention_mask), self.server_config.max_compile_tokens), dtype=jnp.int32)

            input_ids = jnp.concatenate(
                (input_ids, predicted_token), axis=-1
            )[:, -fixed_pad:]

            attention_mask = jnp.concatenate(
                (attention_mask, plus_attn_mask), dtype=jnp.int32,
                axis=-1
            )[:, -fixed_pad:]

            returns = (
                self.tokenizer.decode(input_ids[0][-num_generated_tokens:], skip_special_tokens=True),
                num_generated_tokens
            )

            yield returns

            if self.server_config.use_mxn_break_point:
                if predicted_token.shape[-1] != self.server_config.max_compile_tokens:
                    break

            if (
                    predicted_token[0][-1] == (self.server_config.eos_token_id or self.tokenizer.eos_token_id)
                    or
                    predicted_token[0][-1] == (self.server_config.eos_token_id or self.prefix_tokenizer.eos_token_id)
            ):
                break

    def fire(self):
        """
        The fire function is a wrapper around the uvicorn.run function that allows you
         to run your model in a separate process
        from the main one. This is useful for running models on GPUs, as it prevents any
        other processes from using them while
        the model is being served.

        :param self: Refer to the instance of the class
        :return: A process, which is a child of the main process
        
        """
        assert self._funcs_generated, "you have to first add your model and parameters into server before using fire " \
                                      "with using ``configure_generate_functions``"

        def run():
            uvicorn.run(self.app, host=self.server_config.host, port=self.server_config.port)

        self.process_uvicorn = mp.Process(target=run)
        self.process_uvicorn.start()

    def end(self):
        """
        The end function is used to stop the server.
            It will wait for the process to end before returning.

        :param self: Represent the instance of the class
        :return: The process_uvicorn
        
        """
        if self.process_uvicorn is not None:
            self.process_uvicorn.join()
        else:
            logging.warning("you have to fire server before ending that this command will be ignored")

    def gradio_inference(self):
        return self.build_inference(
            sample_func=self.sample_gradio,
            max_sequence_length=self.server_config.max_sequence_length,
            max_new_tokens=self.server_config.max_new_tokens,
            max_compile_tokens=self.server_config.max_compile_tokens,
        )
