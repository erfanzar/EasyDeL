import copy
import functools
import os
import typing

import flax.core
import gradio as gr
import jax
import msgpack
import tqdm
import transformers
import uvicorn
from fastapi import FastAPI
from fjformer import make_shard_and_gather_fns, match_partition_rules, with_sharding_constraint
from ..smi import get_mem, initialise_tracking
from jax import numpy as jnp
from jax.experimental import mesh_utils
from flax.serialization import from_bytes
from fjformer.load._load import get_float_dtype_by_name
from jax.sharding import Mesh, PartitionSpec as Ps
from transformers import GenerationConfig
import logging
from ..utils.utils import RNG, prefix_str
import multiprocessing as mp
from typing import Union, Sequence
import chex
from .utils import InstructRequest, ChatRequest, seafoam
from jax.experimental.pjit import pjit


class JAXServerConfig:
    def __init__(
            self,
            host: str = "0.0.0.0",
            port: int = 2059,
            batch_size: int = 1,
            contains_auto_format: bool = True,
            max_length: int = 4096,
            max_new_tokens: int = 4096,
            max_stream_tokens: int = 64,
            temperature: float = 0.1,
            top_p: float = 0.95,
            top_k: int = 50,
            logging: bool = True,
            mesh_axes_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            mesh_axes_shape: Sequence[int] = (1, -1, 1, 1),
            generation_ps: jax.sharding.PartitionSpec = Ps("dp", "fsdp"),
            dtype: str = 'fp16',
            stream_tokens_for_gradio: bool = True,
            use_prefix_tokenizer: bool = True,
            pre_compile: bool = True,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an instance of this class, which are:
            host: str = &quot;0.0.0.0&quot;
                The IP address to listen on for incoming requests from clients

        :param self: Represent the instance of the class
        :param host: str: Set the host address of the server
        :param port: int: Specify the port number that the server will run on
        :param batch_size: int: Set the batch size of the model
        :param contains_auto_format: bool: Determine whether the input text contains auto-formatting
        :param max_length: int: Set the maximum length of the text that can be generated
        :param max_new_tokens: int: Determine how many tokens can be added to the vocabulary
        :param max_stream_tokens: int: Set the maximum number of tokens that can be streamed at a time
        :param generation_ps: jax.sharding.PartitionSpec : PartitionSpec to use for sharding data
        :param temperature: float: Control the randomness of the output
        :param top_p: float: Control the diversity of the text generated
        :param top_k: int: Limit the number of tokens that can be generated
        :param logging: bool: Print out the progress of the server
        :param mesh_axes_names: Sequence[str]: Specify the names of the axes in the mesh tensor
        :param &quot;mp&quot;): Define the mesh_axes_names
        :param mesh_axes_shape: Sequence[int]: Specify the shape of the mesh
        :param dtype: str: Specify the data type of the model
        :param stream_tokens_for_gradio: bool: Determine whether the stream tokens
        :param use_prefix_tokenizer: bool: Determine if the tokenizer should be used to generate tokens
        :param pre_compile: bool: Pre-compile the model
        :param : Set the host address
        :return: Nothing
        
        """
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.contains_auto_format = contains_auto_format
        self.max_length = max_length
        self.generation_ps = generation_ps
        self.max_new_tokens = max_new_tokens
        self.max_stream_tokens = max_stream_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.logging = logging
        self.mesh_axes_names = mesh_axes_names
        self.mesh_axes_shape = mesh_axes_shape
        self.dtype = dtype
        self.stream_tokens_for_gradio = stream_tokens_for_gradio
        self.use_prefix_tokenizer = use_prefix_tokenizer
        self.pre_compile = pre_compile
        assert max_new_tokens % max_stream_tokens == 0, \
            'max_new_tokens should be divisible by  max_new_tokens' \
            f'{max_new_tokens % max_stream_tokens}'

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise KeyError(f'{item} not found !')

    def __setitem__(self, key, value):
        setattr(self, key, value)


class JAXServer(object):

    def __init__(self, config=None):

        """
        The __init__ function is called when the class is instantiated.
        It sets up all the attributes that will be used by other methods in the class.


        :param self: Refer to the current instance of a class
        :param config: Pass the JAXServerConfig object
        :return: A fastapi object
        
        """
        self.process_uvicorn, self.prefix_tokenizer, self.params, self.tokenizer, self.model, \
            self.rules, self.generate_function, self.greedy_generate_function = [None] * 8
        assert config is None or isinstance(config, JAXServerConfig), 'config can be None or JAXServerConfig Type'
        if config is None:
            self.config = JAXServerConfig()
        else:
            self.config = config
        self._funcs_generated = False
        self.number_of_served_request_until_last_up_time = 0

        self.rng_generator = RNG(42)
        initialise_tracking(0.5)
        array = jnp.ones((len(jax.devices()), 1)).reshape(self.config.mesh_axes_shape)
        self.mesh = Mesh(mesh_utils.create_device_mesh(array.shape), self.config.mesh_axes_names)

        self.app = FastAPI()
        self.app.post('/chat')(self.forward_chat)
        self.app.post('/instruct')(self.forward_instruct)
        self.app.get('/status')(self.status)
        self.gradio_app_chat = self.create_gradio_ui_chat()
        self.gradio_app_instruct = self.create_gradio_ui_instruct()
        self.app = gr.mount_gradio_app(self.app, self.gradio_app_chat, '/gradio_chat')
        self.app = gr.mount_gradio_app(self.app, self.gradio_app_instruct, '/gradio_instruct')

    def status(self):
        """
        The status function returns a dictionary with the following keys:
            config: A dictionary containing all of the configuration parameters for this server.
            devices: A string describing which devices are available to JAX.
            number_of_backends: The number of backends available to JAX.  This is usually equal to the number of GPUs on your machine, but can be less if you have not installed CUDA or if you have disabled some GPUs in your system BIOS settings (e.g., because they are defective).  It can also be more than one if you have multiple machines connected via MPI and running under Horov

        :param self: Represent the instance of the class
        :return: A dictionary with the following keys:
        
        """
        return {
            'config': {k: v for k, v in self.config.__dict__.items()},
            'devices': f"{jax.devices()}",
            'number_of_backends': len(jax.devices()),
            'status': 'Ready',
            'number_of_served_request_until_last_up_time': f"{self.number_of_served_request_until_last_up_time}",
            'memory': f"{get_mem()}"
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
        assert self.rules is not None, 'you should first shard params with using ``shard_params`` method'

        if tokenizer.pad_token is None:
            logging.info(
                'Tokenizer does not contain padding token setting padding token to eos token for open end generation')
            tokenizer.pad_token = tokenizer.eos_token

        try:
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'left'
            self.prefix_tokenizer = copy.deepcopy(tokenizer)
            tokenizer.padding_side = 'right'
            tokenizer.truncation_side = 'right'
            self.tokenizer = copy.deepcopy(tokenizer)
        except:
            prefix_str(
                'Warning', f'The class Model of Tokenizer {type(tokenizer)} do not support deepcopy option '
            )
            if self.config.use_prefix_tokenizer:
                tokenizer.padding_side = 'left'
                tokenizer.truncation_side = 'left'
            else:
                tokenizer.padding_side = 'right'
                tokenizer.truncation_side = 'right'
            self.prefix_tokenizer = tokenizer

        @functools.partial(
            pjit,
            in_shardings=(self.rules, Ps(), Ps()),
            out_shardings=(Ps())
        )
        def greedy_generate(parameters, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, self.config.generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, self.config.generation_ps)
            predict = model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_new_tokens=self.config.max_stream_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,

                    do_sample=False,
                    num_beams=1,
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        @functools.partial(
            pjit,
            in_shardings=(self.rules, Ps(), Ps()),
            out_shardings=(Ps())
        )
        def generate(parameters, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, self.config.generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, self.config.generation_ps)
            predict = model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_new_tokens=self.config.max_stream_tokens,

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,

                    temperature=self.config.temperature,
                    do_sample=True,
                    num_beams=1,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
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

    def generate(self,
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
                'this method will be implemented automatically after using ``configure_generate_functions`` function'
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
            path: typing.Union[str, os.PathLike],
            config=None,
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
        :param path: typing.Union[str, os.PathLike]: Specify the path to the checkpoint file
        :param config: Configure the server
        :param add_params_field: bool: Add a params field to the server
        :param init_shape: tuple: Specify the shape of the input to be used for generating shard_fns
        :param do_memory_log: bool: Log the memory usage of the server
        :param verbose: bool: Print the compilation process
        :return: A server
        
        """
        assert hasattr(model,
                       'init_weights'), 'model must contain init_weights func in order to init params for shard_fns'
        assert hasattr(config_model,
                       'get_partition_rules'), 'config_model must contain get_partition_rules functions'
        server = cls(config=config)
        logging.info(
            'running _init() func in order to make shard_fns'
        )
        with jax.default_device(jax.devices('cpu')[0]):
            def _init():
                return model.init_weights(jax.random.PRNGKey(0), init_shape)

            shape = jax.eval_shape(_init)
        logging.info(
            'matching partition rules'
        )
        rules = match_partition_rules(params=shape, rules=config_model.get_partition_rules(True))

        with server.mesh:
            shard_fns, _ = make_shard_and_gather_fns(rules, get_float_dtype_by_name(server.config.dtype))
            logging.info(
                'loading checkpoints'
            )

            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
            server.params = {}
            with open(path, 'rb') as stream:
                unpacker = msgpack.Unpacker(stream, read_size=83886080, max_buffer_size=0)
                pbar = tqdm.tqdm(unpacker)
                for key, value in pbar:
                    key = tuple(key)
                    tensor = from_bytes(None, value)
                    tensor = shard_fns[key](tensor)
                    server.params[key] = tensor
                    if do_memory_log:
                        pbar.write(server.get_memory())
                    pbar.set_description('Sharding Params')
        server.params = flax.traverse_util.unflatten_dict(server.params)
        server.params = {'params': server.params} if add_params_field else server.params

        server.rules = {'params': rules} if add_params_field else rules
        logging.info(
            'configuring generate functions for the server'
        )
        server.configure_generate_functions(model, tokenizer)

        if server.config.pre_compile:
            server.compile(verbose=verbose)
        return server

    @classmethod
    def load_from_params(
            cls,
            model: transformers.FlaxPreTrainedModel,
            config_model: transformers.PretrainedConfig,
            tokenizer: transformers.PreTrainedTokenizer,
            params: typing.Dict,
            config=None,
            add_params_field: bool = True,
            do_memory_log: bool = False,
            verbose: bool = True
    ):
        """
        The load_from_params function is used to load a model from the parameters of a pretrained model.
        It takes in the following arguments:
            - cls: The class of the server you are loading, this should be Server or TPU_Server depending on what backend you want to use.
            - model: A FlaxPreTrainedModel object that contains all of your models functions and parameters. This can be found in transformers/flax_utils/models/*model*.py
                where *model* is replaced with whatever transformer you are using (e.g., bert). You can also create your own custom

        :param cls: Create a new instance of the class
        :param model: transformers.FlaxPreTrainedModel: Load the model
        :param config_model: transformers.PretrainedConfig: Get the partition rules
        :param tokenizer: transformers.PreTrainedTokenizer: Tokenize the input text
        :param params: typing.Dict: Pass in the parameters of the model
        :param config: Pass in the config file for the server
        :param add_params_field: bool: Add a params field to the server
        :param do_memory_log: bool: Log the memory usage of the server
        :param verbose: bool: Print out the status of the compilation
        :return: A server object
        
        """
        assert hasattr(model,
                       'init_weights'), 'model must contain init_weights func in order to init params for shard_fns'
        assert hasattr(config_model,
                       'get_partition_rules'), 'config_model must contain get_partition_rules functions'
        server = cls(config=config)

        with server.mesh:
            logging.info(
                'matching partition rules'
            )
            rules = match_partition_rules(params=params, rules=config_model.get_partition_rules(True))
            shard_fns, _ = make_shard_and_gather_fns(rules, get_float_dtype_by_name(server.config.dtype))
            logging.info(
                'sharding parameters across all of the chosen backend(tpu/gpu/cpu)s'
            )
            params = flax.traverse_util.flatten_dict(params)
            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
            pbar = tqdm.tqdm(params.keys())
            for key in pbar:

                key = tuple(key)
                params[key] = shard_fns[key](params[key])

                if do_memory_log:
                    pbar.write(server.get_memory())
                pbar.set_description('Sharding Params')
            server.params = flax.traverse_util.unflatten_dict(params)
            server.params = {'params': server.params} if add_params_field else server.params
        server.rules = {'params': rules} if add_params_field else rules
        logging.info(
            'configuring generate functions for the server'
        )
        server.configure_generate_functions(model, tokenizer)
        if server.config.pre_compile:
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
        assert self._funcs_generated, 'funcs are not generated yet'
        assert self.rules is not None, 'rules should not be None'
        if self.config.use_prefix_tokenizer:
            if verbose:
                print('\033[1;91mCompiling Model Forwards Greedy/NonGreedy(Generate)')
                print('Compiling Greedy Funcs')

            r, a = [None] * 2
            for r, a in self.process(
                    string='',
                    max_new_tokens=self.config.max_stream_tokens,
                    greedy=True
            ):
                ...
            print('Compiling NonGreedy(Generate) Funcs\033[1;0m')
            for r, a in self.process(
                    string='',
                    max_new_tokens=self.config.max_stream_tokens,
                    greedy=False
            ):
                ...

        else:
            print(
                '\033[1;91mSkip Compiling the compiling process is useless '
                'when you are not using prefix tokenizer\033[1;0m')
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
                'this method will be implemented automatically after using ``configure_generate_functions`` function'
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
            'the parameters will be sharded and ba saved inside server you can access them by ``JAXServer.params``')
        rules = match_partition_rules(params=params, rules=partition_rules)
        self.rules = rules
        shard_fns, _ = make_shard_and_gather_fns(rules, get_float_dtype_by_name(self.config.dtype))

        with self.mesh:
            self.params = jax.tree_map(
                lambda f, p: f(p), shard_fns, params
            )

        return self.params

    def forward_chat(self, data: ChatRequest):

        """
        The forward_chat function is the main function of this class.
        It takes in a ChatRequest object, which contains a prompt and history.
        The prompt is the user's input to be processed by the chatbot, while history
        is an array of previous inputs and outputs from both sides (user and bot).
        The forward_chat function then formats these inputs into one string that can be processed by our model.
        This formatted string is then passed through our process() method, which returns an output response as well as how many tokens were used to generate it.

        :param self: Access the attributes and methods of the class
        :param data: ChatRequest: Pass in the data from the request
        :return: A dictionary with the following keys:
        
        """
        if not self._funcs_generated:
            return {
                'status': "down"
            }

        string = self.format_chat(
            prompt=data.prompt,
            system=None,
            history=data.history
        )

        response, used_tokens = [None] * 2
        for response, used_tokens in self.process(
                string=string,
                greedy=data.greedy,
                max_new_tokens=None
        ):
            ...
        self.number_of_served_request_until_last_up_time += 1
        return {
            'input': f'{string}',
            'response': response,
            'tokens_used': used_tokens,
        }

    @staticmethod
    def format_instruct(system: str, instruction: str) -> str:
        """
        Here you will get the system and instruction from user, and you can apply your prompting style
        """
        raise NotImplementedError()

    @staticmethod
    def format_chat(history: typing.List[str], prompt: str, system: typing.Union[str, None]) -> str:
        """
        Here you will get the system, prompt and history from user, and you can apply your prompting style
        """
        raise NotImplementedError()

    def forward_instruct(self, data: InstructRequest):
        """
        The forward_instruct function is the main function of this class.
        It takes in a InstructRequest object, which contains the system and instruction to be processed.
        The function then formats the input string using format_instruct, and passes it into process().
        process() returns a tuple containing (response, used_tokens). The response is returned as part of
        the response dictionary. If no valid responses are found by process(), None will be returned instead.

        :param self: Bind the method to the object
        :param data: InstructRequest: Pass the system and instruction to the function
        :return: A dictionary with three keys:
        
        """
        if not self._funcs_generated:
            return {
                'status': "down"
            }

        response, used_tokens = [None] * 2
        string = self.format_instruct(
            system=data.system,
            instruction=data.instruction
        )
        for response, used_tokens in self.process(
                string=string,
                greedy=data.greedy,
                max_new_tokens=None
        ):
            ...
        self.number_of_served_request_until_last_up_time += 1
        return {
            'input': f'{string}',
            'response': response,
            'tokens_used': used_tokens,
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

    def process(self,
                string: str,
                *,
                greedy: bool = False,
                max_new_tokens: int = None,
                **kwargs
                ):
        """
        The process function is the main function of a model. It takes in an input string and returns a list of strings
        that are generated from that input string. The process function can be called multiple times with different inputs,
        and each time it will return a new set of outputs based on those inputs.

        :param self: Access the class attributes
        :param string: str: Pass the string that we want to generate
        :param *: Pass a variable number of arguments to a function
        :param greedy: bool: Determine whether to use the greedy or non-greedy version of the generate function
        :param max_new_tokens: int: Set the number of tokens to generate
        :param kwargs: Pass any additional parameters to the process function
        :return: A generator that yields the predicted text and the number of tokens generated
        
        """

        fixed_pad = self.config.max_length - self.config.max_stream_tokens
        tokens = self.prefix_tokenizer(
            string,
            max_length=fixed_pad,
            padding='max_length',
            return_tensors='jax'
        ) \
            if self.config.use_prefix_tokenizer else \
            self.tokenizer(
                string,
                return_tensors='jax'
            )

        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        num_generated_tokens = 0

        for _ in range((max_new_tokens or self.config.max_new_tokens) // self.config.max_stream_tokens):
            inputs_to_gen = dict(
                params=self.params,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predicted_token = self.greedy_generate(**inputs_to_gen) if greedy else self.generate(**inputs_to_gen)

            num_generated_tokens += predicted_token.shape[-1]
            plus_attn_mask = jnp.ones((len(attention_mask), self.config.max_stream_tokens), dtype=jnp.int32)

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
            if (
                    predicted_token[0][-1] == self.tokenizer.eos_token_id
                    or
                    predicted_token[0][-1] == self.prefix_tokenizer.eos_token_id
            ):
                break

    def process_gradio_chat(self, prompt, history, max_new_tokens, system, greedy):
        """
        The process_gradio_chat function is a wrapper for the process function.
        It takes in a prompt, history, max_new_tokens and system as arguments.
        The string variable is set to the output of format_chat with the given history and prompt.
        If stream tokens are not enabled then it will append an empty response to history and iterate through all responses from process until there are no more left (the last one). It will then return an empty string along with this new updated version of history. If stream tokens are enabled it appends an empty response to the end of our current list of histories (history) and iterates through

        :param self: Refer to the object itself
        :param prompt: Add the user's input to the history
        :param history: Keep track of the conversation
        :param max_new_tokens: Limit the number of tokens that can be generated by the model
        :param system: Determine whether the message is from the user or system
        :param greedy: Determine if the model should generate a response token by token or all at once
        :return: A tuple of two values:
        
        """
        string = self.format_chat(history=history, prompt=prompt, system=system)

        if not self.config.stream_tokens_for_gradio:
            response = ''
            for response, _ in self.process(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
            ):
                pass
            history.append([prompt, response])
        else:
            history.append([prompt, ''])
            for response, _ in self.process(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
            ):
                history[-1][-1] = response
                yield '', history
        return '', history

    def process_gradio_instruct(self, instruction, system, max_new_tokens, greedy):
        """
        The process_gradio_instruct function is a wrapper for the process function.
        It takes in an instruction and system, formats them into a string, and then passes that string to the process function.
        The response from this call to process is returned as output.

        :param self: Refer to the instance of the class
        :param instruction: Pass in the instruction from the user
        :param system: Determine which system to use for the instruction
        :param max_new_tokens: Limit the number of new tokens that can be added to the vocabulary
        :param greedy: Determine whether the model should be greedy or not
        :return: A tuple of two strings:
        
        """
        string = self.format_instruct(instruction=instruction, system=system)
        if not self.config.stream_tokens_for_gradio:
            response = ''
            for response, _ in self.process(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
            ):
                pass

        else:
            response = ''
            for response, _ in self.process(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
                    stream=True
            ):
                yield '', response
        return '', response

    def create_gradio_ui_chat(self):
        """
        The create_gradio_ui_chat function creates a Gradio UI for the chatbot.

        :param self: Represent the instance of the class
        :return: A block
        
        """
        with gr.Blocks(
                theme=seafoam) as block:
            gr.Markdown("# <h1> <center>Powered by [EasyDeL](https://github.com/erfanzar/EasyDel) </center> </h1>")
            with gr.Row():
                history = gr.Chatbot(elem_id="EasyDel", label="EasyDel", container=True, height=600)

            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(show_label=False, placeholder='Message Box', container=False)
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button(variant="primary")
                        stop = gr.Button(value='Stop ')
                        clear = gr.Button(value='Clear Conversation')

            with gr.Row():
                with gr.Accordion('Advanced Options', open=False):
                    max_new_tokens = gr.Slider(value=self.config.max_new_tokens, maximum=10000,
                                               minimum=self.config.max_stream_tokens,
                                               label='Max New Tokens', step=self.config.max_stream_tokens, )

                    system = gr.Textbox(show_label=False, placeholder='System Prompt', container=False, value='')
                    greedy = gr.Checkbox(value=False, label='Greedy Search')

            inputs = [prompt, history, max_new_tokens, system, greedy]
            sub_event = submit.click(fn=self.process_gradio_chat, inputs=inputs, outputs=[prompt, history])

            def clear_():
                return []

            clear.click(fn=clear_, outputs=[history])
            txt_event = prompt.submit(fn=self.process_gradio_chat, inputs=inputs, outputs=[prompt, history])

            stop.click(fn=None, inputs=None, outputs=None, cancels=[txt_event, sub_event])

        block.queue()
        return block

    def create_gradio_ui_instruct(self):
        """
        The create_gradio_ui_instruct function creates a Gradio UI for the EasyDeL model.
        The function takes in an instance of the EasyDeL class and returns a Gradio UI object.


        :param self: Represent the instance of the class
        :return: A block
        
        """
        with gr.Blocks(
                theme=seafoam) as block:
            gr.Markdown("# <h1> <center>Powered by [EasyDeL](https://github.com/erfanzar/EasyDel) </center> </h1>")
            with gr.Row():
                pred = gr.TextArea(elem_id="EasyDel", label="EasyDel", container=True)

            with gr.Row():
                submit = gr.Button(variant="primary")
                stop = gr.Button(value='Stop ')
                clear = gr.Button(value='Clear Conversation')
            with gr.Column():
                prompt = gr.Textbox(show_label=False, placeholder='Instruct Message', container=False)

            with gr.Row():
                with gr.Accordion('Advanced Options', open=False):
                    system = gr.Textbox(value='',
                                        show_label=False, placeholder='System Message', container=False)
                    max_new_tokens = gr.Slider(value=self.config.max_new_tokens, maximum=10000,
                                               minimum=self.config.max_stream_tokens,
                                               label='Max New Tokens', step=self.config.max_stream_tokens, )

                    greedy = gr.Checkbox(value=False, label='Greedy Search')

            inputs = [prompt, system, max_new_tokens, greedy]
            sub_event = submit.click(fn=self.process_gradio_instruct, inputs=inputs, outputs=[prompt, pred])

            def clear_():
                return ''

            clear.click(fn=clear_, outputs=[pred])
            txt_event = prompt.submit(fn=self.process_gradio_instruct, inputs=inputs, outputs=[prompt, pred])

            stop.click(fn=None, inputs=None, outputs=None, cancels=[txt_event, sub_event])

        block.queue()
        return block

    def fire(self):
        """
        The fire function is a wrapper around the uvicorn.run function that allows you to run your model in a separate process
        from the main one. This is useful for running models on GPUs, as it prevents any other processes from using them while
        the model is being served.

        :param self: Refer to the instance of the class
        :return: A process, which is a child of the main process
        
        """
        assert self._funcs_generated, 'you have to first add your model and parameters into server before using fire ' \
                                      'with using ``configure_generate_functions``'

        def run():
            uvicorn.run(self.app, host=self.config.host, port=self.config.port)

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
            logging.warning('you have to fire server before ending that this command will be ignored')

    def launch(self,
               share_chat: bool = False,
               share_inst: bool = False
               ):
        """
        The launch function is a wrapper for the launch function of the gradio.Interface class.
        It takes two boolean arguments: share_chat and share_inst, which are used to determine whether to
        share the chatbot interface and/or instruction interface respectively. If both are set to False, then neither
        interface will be shared (this is useful if you want to run your app locally). If one or both of them are True,
        then they will be shared on Gradio's servers with a unique URL that can be accessed by anyone in order for them
        to interact with your app.

        :param self: Represent the instance of the class
        :param share_chat: bool: Determine if the chatbot should be shared
        :param share_inst: bool: Share the instructions for the app
        :return: A dictionary with the share urls for chat and instructions
        
        """
        share_kwargs = {}
        assert not share_chat or not share_inst, 'you have to pass at least one of sharing options True'
        if share_chat:
            self.gradio_app_chat.launch(share=True)
            share_kwargs['chat'] = self.gradio_app_chat.share_url
        if share_inst:
            self.gradio_app_instruct.launch(share=True)
            share_kwargs['inst'] = self.gradio_app_instruct.share_url
        return share_kwargs
