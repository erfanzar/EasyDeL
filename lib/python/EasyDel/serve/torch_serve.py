import threading
import typing

import gradio as gr
import transformers
import uvicorn
from fastapi import FastAPI
from typing import List
from transformers import TextIteratorStreamer
import logging
import multiprocessing as mp
import torch
from .utils import ChatRequest, InstructRequest
from .gradio_user_interface_base import GradioUserInference
from dataclasses import dataclass


@dataclass
class PytorchServerConfig:
    """
    It sets up the instance of the class, and defines all its attributes.

    :param host: Specify the ip address of the server
    :param port: Specify the port number that will be used by the server
    :param batch_size: Determine the number of samples to be generated in a single batch
    :param contains_auto_format: Determine whether the input text contains auto_formatting
    :param max_length: Set the maximum length of a sentence
    :param max_new_tokens: Limit the number of new tokens that can be generated in a single batch
    :param temperature: Control the randomness of the generated text
    :param top_p: Control the probability of sampling from the top candidates
    :param top_k: Limit the number of tokens that are considered for each token
    :param logging: Control whether the server will print out
    :param dtype: Specify the data type of the tensors
    :param max_number_of_gpus: Limit the number of gpus used by the server
    :param max_gpu_perc_to_use: Specify the maximum percentage of gpu memory that can be used by the server
    :param max_compile_tokens: int: Limit the number of tokens that can be streamed to a single client
    """
    host: str = "0.0.0.0"
    port: int = 2059
    batch_size: int = 1
    contains_auto_format: bool = True
    max_length: int = 2048
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    logging: bool = True
    dtype: str = "fp16"
    max_number_of_gpus: typing.Optional[int] = None
    max_gpu_perc_to_use: float = 0.95
    max_compile_tokens: int = 1


class PyTorchServer(GradioUserInference):

    def __init__(self, config: PytorchServerConfig):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function can accept arguments, which are passed at instantiation.

        :param self: Represent the instance of the class
        :param config: PytorchServerConfig: Pass the configuration parameters to the class
        :return: The app, which is a fastapi object
        
        """
        self.model, self.tokenizer = [None] * 2

        self.config = config
        self.process_uvicorn = None
        self.app = FastAPI()
        self.number_of_served_request_until_last_up_time = 0
        self.device_rolling = self.get_gpu_memory(self.config.max_number_of_gpus)
        self.dict_max_memory_sharding = {
            i: str(
                int(
                    mem * self.config.max_gpu_perc_to_use
                )
            ) + "GiB" for i, mem in
            enumerate(self.device_rolling)
        }
        self.app.post("/chat")(self.forward_chat_fast_api)
        self.app.post("/instruct")(self.forward_instruct_fast_api)
        self.app.get("/status")(self.status)
        self.app = gr.mount_gradio_app(self.app, self.gradio_inference(), "/gradio_chat")

    @staticmethod
    def get_gpu_memory(num_gpus_req=None):

        """
        The get_gpu_memory function returns the amount of available GPU memory in GB.

        :param num_gpus_req: Specify the number of gpus to be used
        :return: The amount of free memory on each gpu
        
        """
        gpu_m = []
        dc = torch.cuda.device_count()
        num_gpus = torch.cuda.device_count() if num_gpus_req is None else min(num_gpus_req, dc)

        for gpu_id in range(num_gpus):
            with torch.cuda.device(gpu_id):
                gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_m.append(
                    (gpu_properties.total_memory / (1024 ** 3)) - (torch.cuda.memory_allocated() / (1024 ** 3)))
        return gpu_m

    def get_model_load_kwargs(self):
        """
        The get_model_load_kwargs function is used to set the torch_dtype, device_map and max_memory parameters for loading a model.

        :param self: Bind the method to an object
        :return: A dictionary with the following keys:
        
        """
        if self.config.dtype == "fp16":
            dtype = torch.float16
        elif self.config.dtype == "fp32":
            dtype = torch.float32
        elif self.config.dtype == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError("unknown type available types are [fp32 fp16 bf16]")
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "max_memory": self.dict_max_memory_sharding
        }
        return load_kwargs

    def status(self):

        """
        The status function returns a dictionary with the following keys:
            config: A dictionary of configuration parameters.
            devices: The number of GPUs available to the server.
            device_sharding: Whether device sharding is enabled. If True, then each request will be served by
            a different GPU (if multiple GPUs are available). If False, then all requests will be served by
            the same GPU (or CPU if no GPUs are available). This parameter can also be set in your client"s
            initialization function via torch-serve"s DeviceShardingStrategy
            class. See https://pytorch-lightning.readthedoc

        :param self: Represent the instance of the class
        :return: A dictionary with the following keys:
        
        """
        return {
            "config": {k: v for k, v in self.config.__dict__.items()},
            "devices": f"{torch.cuda.device_count()}",
            "device_sharding": self.device_rolling,
            "max_memory": self.dict_max_memory_sharding,
            "status": "Ready",
            "number_of_served_request_until_last_up_time": f"{self.number_of_served_request_until_last_up_time}"
        }

    def forward_instruct_fast_api(self, data: InstructRequest):
        """
        The forward_instruct_fast_api function is a ReST API endpoint that takes in an InstructRequest object and returns
        a response. The InstructRequest object contains the following fields:
            - system (str): A string representing the name of the system to be instructed. This should match one of the
                systems defined in your config file, or else it will default to &quot;default&quot;. If you want to instruct multiple
                systems at once, use forward_instruct_fast instead.

        :param self: Refer to the object itself
        :param data: InstructRequest: Pass in the data that is used to generate the response
        :return: A dictionary with a single key, response
        
        """
        string = self.format_instruct(
            system=data.system,
            instruction=data.instruction
        )
        response = self.process(
            string=string,
            max_length=self.config.max_length,
            temperature=data.temperature,
            stream=False,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens
        )
        return {
            "response": response
        }

    def forward_chat_fast_api(self, data: ChatRequest):
        """
        The forward_chat_fast_api function is a ReST API endpoint that takes in a ChatRequest object and returns the response from the model.

        :param self: Refer to the object itself
        :param data: ChatRequest: Pass the data from the serve_engine to the function
        :return: A dictionary with a single key, response
        
        """
        string = self.format_chat(
            system=data.system,
            history=data.history,
            prompt=data.prompt,
        )
        response = self.process(
            string=string,
            max_length=self.config.max_length,
            temperature=data.temperature,
            stream=False,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens
        )
        return {
            "response": response
        }

    @staticmethod
    def format_instruct(system: str, instruction: str) -> str:
        """
        The format_instruct function is used to format the instruction string
            for a particular system.  The function takes two arguments:

        :param system: str: Determine which system the instruction is for
        :param instruction: str: Store the instruction that is being passed in
        :return: The instruction in the format of the system
        
        """
        raise NotImplementedError()

    @staticmethod
    def format_chat(history: List[List[str]], prompt: str, system: str = None) -> str:
        """
        The format_chat function takes a list of strings, representing the chat history,
        and returns a string that is formatted in such a way that it can be printed to the screen.
        The prompt argument is used to indicate which user's turn it currently is. The system argument
        is used for messages from the system (e.g., &quot;You are now connected!&quot;). If no value for system
        is provided, then this function should return None.

        :param history: List[str]: Store the chat history
        :param prompt: str: Display the prompt to the user
        :param system: str: Add a system message to the chat history
        :return: A string that contains the history of a chat
        
        """
        raise NotImplementedError()

    def process(
            self,
            string: str,
            max_new_tokens: int = None,
            max_length: int = None,
            temperature: float = 0.6,
            top_k=50,
            top_p=0.9,
            stream: bool = True,
            sample: bool = True
    ):
        """
        The process function is the main function of this class. It takes a string as input and returns a generator that yields strings.

        :param self: Represent the instance of the class
        :param string: str: Pass the string to be generated
        :param max_new_tokens: int: Limit the number of new tokens that can be generated
        :param max_length: int: Set the maximum length of the generated text
        :param temperature: float: Control the randomness of the text generation
        :param top_k: Filter out the top k tokens with the highest probability
        :param top_p: Control the probability of sampling from the top n tokens
        :param stream: bool: Determine whether to stream the output or not
        :param sample: bool: Indicate whether to sample from the distribution or take the argmax
        :return: A generator
        
        """
        assert self.model is not None, "you should first load model with ``load`` method"
        tokens = self.tokenizer(
            string,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)

        iterator_streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        if stream:
            kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=iterator_streamer,
                generation_config=transformers.GenerationConfig(
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_length=max_length or self.config.max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                    num_beams=1,
                    do_sample=sample,
                )
            )
            thread_ = threading.Thread(
                target=self.model.generate,
                kwargs=kwargs
            )
            thread_.start()
            for string in iterator_streamer:
                yield string
        else:
            kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=transformers.GenerationConfig(
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_length=max_length or self.config.max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                    num_beams=1
                )
            )
            pred = self.tokenizer.decode(self.model.generate(
                **kwargs
            ).logits[0])
            return pred

    def load(self, pretrained_model_name_or_path: str, tokenizer_repo: str = None, auto_config: bool = True, **kwargs):
        """
        The load function is used to load a model from the HuggingFace Model Hub.

        :param self: Represent the instance of the class
        :param pretrained_model_name_or_path: str: Specify the name of the model to be loaded
        :param tokenizer_repo: str: Specify the repo id of the tokenizer
        :param auto_config: bool: Determine whether the model should be loaded with a config file or not
        :param kwargs: Pass a variable number of keyword arguments to the function
        :return: A tuple of model and tokenizer
        
        """
        load_kwargs = kwargs if not auto_config else self.get_model_load_kwargs()
        load_kwargs = load_kwargs | kwargs
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **load_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_repo or pretrained_model_name_or_path,
            trust_remote_code=True
        )

        self.model = model
        self.tokenizer = tokenizer

    def process_gradio(
            self,
            prompt: str,
            history: List[List[str]],
            system_prompt: str | None,
            mode: str,
            max_length: int,
            max_new_tokens: int,
            max_compile_tokens: int,
            greedy: bool,
            temperature: float,
            top_p: float,
            top_k: int
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
            raise ValueError("UnKnown Mode for process_gradio available modes are only Chat or Instruct")
        history.append([prompt, ""])
        responses = ""
        for response in self.process(
                string=string,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_length=max_length,
                top_p=top_p,
                top_k=top_k,
                stream=True
        ):
            responses += response
            history[-1][-1] = responses
            yield "", history

    def gradio_inference(self):
        return self.build_inference(
            sample_func=self.process_gradio,
            max_length=self.config.max_length,
            max_new_tokens=self.config.max_new_tokens,
            max_compile_tokens=1,
        )

    def fire(self):
        """
        The fire function starts the uvicorn server in a separate process.

        :param self: Represent the instance of the class
        :return: A process that runs the uvicorn server
        
        """

        def run():
            uvicorn.run(self.app, host=self.config.host, port=self.config.port)

        self.process_uvicorn = mp.Process(target=run)
        self.process_uvicorn.start()

    def end(self):
        """
        The end function is used to stop the server.
            It will wait for the process to end before returning.

        :param self: Represent the instance of the class
        :return: A boolean value
        
        """
        if self.process_uvicorn is not None:
            self.process_uvicorn.join()
        else:
            logging.warning("you have to fire server before ending that this command will be ignored")
