import copy
import functools
import logging
import warnings

import jax
import uvicorn
from fastapi import FastAPI
from fjformer import with_sharding_constraint, match_partition_rules, make_shard_and_gather_fns, get_dtype
from jax import numpy as jnp

from ...etils.etils import get_logger
from ...modules.easydel_modelling_utils import EasyDelFlaxPretrainedModel
from flax.core import FrozenDict
from transformers import PreTrainedTokenizerBase, GenerationConfig
from typing import Callable, Mapping, Tuple, List
from .configuration import EasyServeConfig
from jax.sharding import PartitionSpec, Mesh
from jax.experimental.pjit import pjit
from dataclasses import dataclass
from .dantics import GenerateAPIRequest, ConversationItem

logger = get_logger(__name__)


@dataclass
class LLMBaseReq:
    greedy_generate_function: Callable
    non_greedy_generate_function: Callable
    tokenizer: PreTrainedTokenizerBase
    prefix_tokenizer: PreTrainedTokenizerBase


class EasyServe:
    def __init__(
            self,
            llm: EasyDelFlaxPretrainedModel,
            params: FrozenDict | dict,
            tokenizer: PreTrainedTokenizerBase,
            prefix_tokenizer: PreTrainedTokenizerBase,
            greedy_generate_function: Callable,
            non_greedy_generate_function: Callable,
            serve_config: EasyServeConfig,
    ):
        self.llm = llm
        self.params = params
        self.tokenizer = tokenizer
        self.prefix_tokenizer = prefix_tokenizer
        self.greedy_generate_function = greedy_generate_function
        self.non_greedy_generate_function = non_greedy_generate_function
        self.serve_config = serve_config
        if serve_config.pre_compile:
            self.compile(verbose=serve_config.verbose)
        self.fast_api = FastAPI(
            title="EasyDeL"
        )
        self.fast_api.post("/generate/v1/conversation")(self.api_generate_request)

    def get_generation_function(self, greedy: bool):
        return self.greedy_generate_function if greedy else self.non_greedy_generate_function

    def conversation_template(
            self, conversation: List[ConversationItem]
    ) -> str:
        """
        The conversation_template function takes a list of ConversationItem objects and returns a string.
        The string is formatted as follows:
            &lt;s&gt;[INST] &lt;&lt;SYS&gt;&gt;{system_message}&lt;&lt;/SYS&gt;&gt;\n\n{user_message} [/INST]
            {assistant_message} &lt;/s&gt;&lt;s&gt;[INST] ...
        where system message, user message, and assistant message are the content fields of the ConversationItem objects.
        If there is no system message in the conversation, then it will be omitted from the template.

        :param self: Refer to the current instance of a class
        :param conversation: List[ConversationItem]: Pass in the conversation items
        :return: A string that is a concatenation of the messages in the conversation

        """
        system = None
        do_strip = False
        for conv in conversation:
            if conv.role == "system":
                system = conv.content
        messages = [f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"] if system is not None else ["<s>[INST] "]
        for conv in conversation:
            content = conv.content.strip() if do_strip else conv.content
            role = conv.role
            do_strip = True
            if role == "user" or role == "assistant":
                messages.append(f"{content} [/INST] " if role == "user" else f"{content} </s><s>[INST] ")
        return ''.join(messages)

    def api_generate_request(self, request: GenerateAPIRequest):

        """
        The api_generate_request function is the main function that will be called by the API.
        It takes in a GenerateAPIRequest object, which contains all the information needed to generate a response.
        The request object has two fields: conversation and max_new_tokens. The conversation field is an array of strings,
        which are each one turn in a conversation (i.e., one person's utterance). The max_new_tokens field specifies
         how many tokens can be generated at most for this response.

        :param self: Access the attributes and methods of the class
        :param request: GenerateAPIRequest: Pass the request object to the api_generate_request function
        :return: A dictionary with the following keys {"response", "num_token_generated", "greedy", "model_prompt"}
        """
        prompt = self.conversation_template(
            request.conversation
        )
        response: str | None = None
        num_token_generated: int | None = None
        for response, num_token_generated in self.sample(
                string=prompt,
                max_new_tokens=request.max_new_tokens or self.serve_config.max_new_tokens,
                greedy=request.greedy or self.serve_config.greedy
        ):
            ...
        return {
            "response": response,
            "num_token_generated": num_token_generated,
            "greedy": request.greedy,
            "model_prompt": prompt
        }

    @staticmethod
    def create_shard_and_gather_functions(
            parameters: dict,
            partition_rules: Tuple[Tuple[str, PartitionSpec]],
            dtype: jax.numpy.dtype | str = "fp16"
    ):

        """
        The create_shard_and_gather_functions function takes in a dictionary of parameters,
        a tuple of partition rules, and an optional dtype. It then matches the partition rules to the
        parameters and creates shard functions for each parameter. The shard functions are used to
        split up a parameter into shards (or partitions) that can be stored on different devices.
        The gather function is used to combine all the shards back together again.

        :param parameters: dict: Specify the parameters of the model
        :param partition_rules: Tuple[Tuple[str,  PartitionSpec]]: Specify which parameters to partition
        :param dtype: jax.numpy.dtype | str: Specify the data type of the parameters
        :return: A tuple of three elements:
        """
        partition_specs = match_partition_rules(partition_rules, parameters)
        shard_fns, gather_fns = make_shard_and_gather_fns(
            partition_specs=partition_specs,
            dtype_specs=get_dtype(dtype)
        )
        return shard_fns, gather_fns, partition_specs

    @staticmethod
    def shard_parameters(
            mesh: Mesh,
            params: FrozenDict | dict,
            partition_rules: Tuple[Tuple[str, PartitionSpec]],
            serve_config: EasyServeConfig,
    ):

        """
        The shard_parameters function takes a set of parameters and partitions them according to the partition_rules.

        :param mesh: Mesh: Create a mesh object that is used to shard the parameters
        :param params: FrozenDict | dict: Pass in the parameters of the model
        :param partition_rules: Tuple[Tuple[str, PartitionSpec]]: Specify the partitioning rules for each parameter
        :param serve_config: EasyServeConfig: Specify the dtype of the parameters
        :param : Create a mesh of devices
        :return: sharded parameters
        """

        partition_specs = match_partition_rules(params=params, rules=partition_rules)
        shard_fns, _ = make_shard_and_gather_fns(partition_specs, get_dtype(serve_config.dtype))

        with mesh:
            params = jax.tree_map(
                lambda func, param: func(param), shard_fns, params
            )

        return params

    @staticmethod
    def create_generation_functions_and_tokenizers(
            model: EasyDelFlaxPretrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            serve_config: EasyServeConfig,
            partition_specs: dict[str, PartitionSpec]
    ) -> LLMBaseReq:
        """
        The create_generation_functions_and_tokenizers function is used to create the functions that will be used for
        generation. It also creates a tokenizer object that can be used to encode and decode text. The function takes in
        a model, a tokenizer, an EasyServeConfig object (which contains all the parameters needed for generation), and
        partition_specs which are specifications about how data should be partitioned across devices.

        :param model: EasyDelFlaxPretrainedModel: Create the model and tokenizer
        :param tokenizer: PreTrainedTokenizerBase: Create a tokenizer object
        :param serve_config: EasyServeConfig: Create the generation function
        :param partition_specs: dict[str, PartitionSpec]: Specify the sharding of the model parameters
        :return: An LLMBaseReq object
        """
        if tokenizer.pad_token is None:
            logging.info(
                "Tokenizer does not contain padding token setting padding token to eos token for open end generation")
            tokenizer.pad_token = tokenizer.eos_token

        try:
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            prefix_tokenizer = copy.deepcopy(tokenizer)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
            tokenizer = copy.deepcopy(tokenizer)

        except:
            warnings.warn(
                f"The class Model of Tokenizer {type(tokenizer)} do not support deepcopy option "
            )
            if serve_config.use_prefix_tokenizer:
                tokenizer.padding_side = "left"
                tokenizer.truncation_side = "left"
            else:
                tokenizer.padding_side = "right"
                tokenizer.truncation_side = "right"
            prefix_tokenizer = tokenizer

        @functools.partial(
            pjit,
            in_shardings=(partition_specs, PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def greedy_generate_function(
                parameters,
                input_ids,
                attention_mask
        ):
            input_ids = with_sharding_constraint(input_ids, serve_config.generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, serve_config.generation_ps)
            predict = model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_new_tokens=serve_config.max_compile_tokens,

                    eos_token_id=serve_config.eos_token_id or tokenizer.eos_token_id,
                    pad_token_id=serve_config.pad_token_id or tokenizer.pad_token_id,
                    bos_token_id=serve_config.bos_token_id or tokenizer.bos_token_id,

                    do_sample=False,
                    num_beams=1,
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        @functools.partial(
            pjit,
            in_shardings=(partition_specs, PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def non_greedy_generate_function(
                parameters,
                input_ids,
                attention_mask
        ):
            input_ids = with_sharding_constraint(input_ids, serve_config.generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, serve_config.generation_ps)
            predict = model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_new_tokens=serve_config.max_compile_tokens,

                    eos_token_id=serve_config.eos_token_id or tokenizer.eos_token_id,
                    pad_token_id=serve_config.pad_token_id or tokenizer.pad_token_id,
                    bos_token_id=serve_config.bos_token_id or tokenizer.bos_token_id,

                    temperature=serve_config.temperature,
                    repetition_penalty=serve_config.repetition_penalty,
                    do_sample=True,
                    num_beams=1,
                    top_p=serve_config.top_p,
                    top_k=serve_config.top_k,
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        return LLMBaseReq(
            greedy_generate_function=greedy_generate_function,
            non_greedy_generate_function=non_greedy_generate_function,
            tokenizer=tokenizer,
            prefix_tokenizer=prefix_tokenizer
        )

    @classmethod
    def from_parameters(
            cls,
            llm: EasyDelFlaxPretrainedModel,
            params: dict,
            tokenizer: PreTrainedTokenizerBase,
            serve_config: EasyServeConfig,
            partition_rules: Tuple[Tuple[str, PartitionSpec]],
            shard_parameters: bool = True,
    ):

        """
        The from_parameters function is the main entry point for creating a model that can be served.
        It takes in a pretrained model, parameters, tokenizer and serve_config as input and returns an object of type
        EasyServe.

        :param cls: Create a new instance of the class
        :param llm: EasyDelFlaxPretrainedModel: Pass the model to the class
        :param params: dict: Pass the parameters of the model
        :param tokenizer: PreTrainedTokenizerBase: Create the tokenizer and prefix_tokenizer
        :param serve_config: EasyServeConfig: Configure the model for serving
        :param partition_rules: Tuple[Tuple[str, PartitionSpec]]: Partition the parameters of the model
        :param shard_parameters: bool: Specify whether the parameters should be sharded or not
        :param : Shard the parameters of the model
        :return: A EasyServe object
        """
        shard_fns, gather_fns, partition_specs = cls.create_shard_and_gather_functions(
            parameters=params,
            partition_rules=partition_rules,
            dtype=serve_config.dtype
        )
        llm_base_req = cls.create_generation_functions_and_tokenizers(
            model=llm,
            tokenizer=tokenizer,
            partition_specs=partition_specs,
            serve_config=serve_config
        )

        if shard_parameters:
            params = cls.shard_parameters(
                params=params,
                partition_rules=partition_rules,
                serve_config=serve_config,
                mesh=llm.config.jax_mesh()
            )

        return cls(
            llm=llm,
            serve_config=serve_config,
            tokenizer=llm_base_req.tokenizer,
            prefix_tokenizer=llm_base_req.prefix_tokenizer,
            params=params,
            greedy_generate_function=llm_base_req.greedy_generate_function,
            non_greedy_generate_function=llm_base_req.non_greedy_generate_function,
        )

    def sample(
            self,
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
        :param greedy: bool: Determine whether to use the greedy or non-greedy version of the generate function
        :param max_new_tokens: int: Set the number of tokens to generate
        :param kwargs: Pass any additional parameters to the process function
        :return: A generator that yields the predicted text and the number of tokens generated

        """
        with self.llm.config.jax_mesh():
            fixed_pad = self.serve_config.max_sequence_length - self.serve_config.max_compile_tokens
            tokens = self.prefix_tokenizer(
                string,
                max_length=fixed_pad,
                padding="max_length",
                return_tensors="jax"
            ) if self.serve_config.use_prefix_tokenizer else self.tokenizer(
                string,
                return_tensors="jax"
            )

            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
            num_generated_tokens = 0

            for _ in range(
                    (max_new_tokens or self.serve_config.max_new_tokens) // self.serve_config.max_compile_tokens):

                predicted_token = self.get_generation_function(greedy=greedy)(
                    self.params,
                    input_ids,
                    attention_mask
                )

                num_generated_tokens += predicted_token.shape[-1]
                plus_attn_mask = jnp.ones(
                    (len(attention_mask), self.serve_config.max_compile_tokens),
                    dtype="i4"
                )

                input_ids = jnp.concatenate(
                    (input_ids, predicted_token), dtype="i4",
                    axis=-1
                )[:, -fixed_pad:]

                attention_mask = jnp.concatenate(
                    (attention_mask, plus_attn_mask), dtype="i4",
                    axis=-1
                )[:, -fixed_pad:]

                returns = (
                    self.tokenizer.decode(input_ids[0][-num_generated_tokens:], skip_special_tokens=True),
                    num_generated_tokens
                )

                yield returns

                if self.serve_config.use_mxn_break_point:
                    if self.serve_config.max_compile_tokens != predicted_token.shape[-1]:
                        break
                if (
                        predicted_token[0][-1] == (self.serve_config.eos_token_id or self.tokenizer.eos_token_id)
                        or
                        predicted_token[0][-1] == (self.serve_config.eos_token_id or self.prefix_tokenizer.eos_token_id)
                ):
                    break

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
        if self.serve_config.use_prefix_tokenizer:
            if verbose:
                logger.info("Compiling greedy generate function")
            response, tokens = [None] * 2
            for response, tokens in self.sample(
                    string="",
                    max_new_tokens=self.serve_config.max_compile_tokens,
                    greedy=True
            ):
                ...
            if verbose:
                logger.info("Compiling non-greedy generate function")
            for response, tokens in self.sample(
                    string="",
                    max_new_tokens=self.serve_config.max_compile_tokens,
                    greedy=False
            ):
                ...

        else:
            logger.warning(
                "Skip Compiling the compiling process is useless "
                "when you are not using prefix tokenizer",
            )
        return True

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

    def fire(self):
        uvicorn.run(
            self.fast_api,
            host=self.serve_config.host,
            port=self.serve_config.port,
        )
