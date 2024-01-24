import copy
import functools
import logging
import warnings

import flax.core
import jax
import termcolor
import uvicorn
from fastapi import FastAPI
from fjformer import with_sharding_constraint, match_partition_rules, make_shard_and_gather_fns, get_dtype
from jax import numpy as jnp
from ...modules.easydel_modelling_utils import EasyDelFlaxPretrainedModel
from flax.core import FrozenDict
from transformers import PreTrainedTokenizerBase, GenerationConfig
from typing import Callable, Mapping, Tuple, List
from .configuration import EasyServeConfig
from jax.sharding import PartitionSpec, Mesh
from jax.experimental.pjit import pjit
from dataclasses import dataclass
from .dantics import GenerateAPIRequest, ConversationItem


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

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,

                    temperature=serve_config.temperature,
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
            fixed_pad = self.serve_config.max_length - self.serve_config.max_compile_tokens
            tokens = self.prefix_tokenizer(
                string,
                max_length=fixed_pad,
                padding="max_length",
                return_tensors="jax"
            ) \
                if self.serve_config.use_prefix_tokenizer else \
                self.tokenizer(
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
                if (
                        predicted_token[0][-1] == self.tokenizer.eos_token_id
                        or
                        predicted_token[0][-1] == self.prefix_tokenizer.eos_token_id
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
                termcolor.cprint(
                    "Compiling Model Forwards Greedy/Non-Greedy(Generate)",
                    color="cyan",
                    force_color=True
                )
                termcolor.cprint(
                    "Compiling Greedy Functions",
                    color="cyan",
                    force_color=True
                )

            response, tokens = [None] * 2
            for response, tokens in self.sample(
                    string="",
                    max_new_tokens=self.serve_config.max_compile_tokens,
                    greedy=True
            ):
                ...
            if verbose:
                termcolor.cprint(
                    "Compiling Non-Greedy(Generate) Functions",
                    color="cyan",
                    force_color=True
                )
            for response, tokens in self.sample(
                    string="",
                    max_new_tokens=self.serve_config.max_compile_tokens,
                    greedy=False
            ):
                ...

        else:
            termcolor.cprint(
                "Skip Compiling the compiling process is useless "
                "when you are not using prefix tokenizer",
                color="red", force_color=True
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
                repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
                string += repr_src if len(repr_src) < 500 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
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
