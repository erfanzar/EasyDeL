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
from ..utils import RNG
import multiprocessing as mp
from ..utils import prefix_str
from typing import Union, Sequence
import chex
from .utils import InstructRequest, ChatRequest, seafoam
from jax.experimental.pjit import pjit


class JaxServerConfig:
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
            mesh_axes_names: Sequence[str] = ('dp', 'fsdp', 'mp'),
            mesh_axes_shape: Sequence[int] = (1, -1, 1),
            dtype: str = 'fp16',
            stream_tokens_for_gradio: bool = True,
            use_prefix_tokenizer: bool = True,
            pre_compile: bool = True,
    ):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.contains_auto_format = contains_auto_format
        self.max_length = max_length
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

        self.process_uvicorn, self.prefix_tokenizer, self.params, self.tokenizer, self.model, \
            self.rules, self._generate, self._greedy_generate = [None] * 8
        assert config is None or isinstance(config, JaxServerConfig), 'config can be None or JaxServerConfig Type'
        if config is None:
            self.config = JaxServerConfig()
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
        return get_mem()

    def configure_generate_functions(self, model, tokenizer):

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
            input_ids = with_sharding_constraint(input_ids, Ps(('dp', 'fsdp')))
            attention_mask = with_sharding_constraint(attention_mask, Ps(('dp', 'fsdp')))
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
            input_ids = with_sharding_constraint(input_ids, Ps(('dp', 'fsdp')))
            attention_mask = with_sharding_constraint(attention_mask, Ps(('dp', 'fsdp')))
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

        self._generate = generate
        self._greedy_generate = greedy_generate
        self._funcs_generated = True

    def auto_configure(self, model, params, tokenizer, partition_rules):
        self.shard_params(params=params, partition_rules=partition_rules)
        self.configure_generate_functions(model, tokenizer)

    def generate(self,
                 params: Union[flax.core.FrozenDict, dict],
                 input_ids: chex.Array,
                 attention_mask: chex.Array,
                 ):
        if not self._funcs_generated:
            raise NotImplementedError(
                'this method will be implemented automatically after using ``configure_generate_functions`` function'
            )
        else:
            with self.mesh:
                return self._generate(
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
        if not self._funcs_generated:
            raise NotImplementedError(
                'this method will be implemented automatically after using ``configure_generate_functions`` function'
            )
        else:
            with self.mesh:
                return self._greedy_generate(
                    params, input_ids, attention_mask
                )

    def shard_params(self, params, partition_rules):

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
        data = InstructRequest(
            prompt=prompt,
            system=system,
            greedy=greedy
        )
        return self.forward_instruct(data)

    def forward_chat_non_api(self, prompt, history, greedy):
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
        tokens = self.prefix_tokenizer(
            string,
            max_length=self.config.max_length - self.config.max_stream_tokens,
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
        pad = self.config.max_length - self.config.max_stream_tokens

        for _ in range((max_new_tokens or self.config.max_new_tokens) // self.config.max_stream_tokens):
            predicted_token = self.greedy_generate(
                params=self.params,
                input_ids=input_ids,
                attention_mask=attention_mask
            ) if greedy else self.generate(
                params=self.params,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            num_generated_tokens += predicted_token.shape[-1]

            input_ids = jnp.concatenate(
                (input_ids, predicted_token), axis=-1
            )[:, -pad:]
            attention_mask = jnp.concatenate(
                (attention_mask, jnp.ones((len(attention_mask), self.config.max_stream_tokens), dtype=jnp.int32)),
                axis=-1
            )[:, -pad:]

            yield self.tokenizer.decode(input_ids[0][-num_generated_tokens:],
                                        skip_special_tokens=True), num_generated_tokens
            if predicted_token[0][-1] == self.tokenizer.eos_token_id or predicted_token[0][
                -1] == self.prefix_tokenizer.eos_token_id:
                break

    def process_gradio_chat(self, prompt, history, max_new_tokens, system, greedy):
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
        assert self._funcs_generated, 'you have to first add your model and parameters into server before using fire ' \
                                      'with using ``configure_generate_functions``'

        def run():
            uvicorn.run(self.app, host=self.config.host, port=self.config.port)

        self.process_uvicorn = mp.Process(target=run)
        self.process_uvicorn.start()

    def end(self):
        if self.process_uvicorn is not None:
            self.process_uvicorn.join()
        else:
            logging.warning('you have to fire server before ending that this command will be ignored')

    def launch(self,
               share_chat: bool = False,
               share_inst: bool = False
               ):
        share_kwargs = {}
        assert not share_chat or not share_inst, 'you have to pass at least one of sharing options True'
        if share_chat:
            self.gradio_app_chat.launch(share=True)
            share_kwargs['chat'] = self.gradio_app_chat.share_url
        if share_inst:
            self.gradio_app_instruct.launch(share=True)
            share_kwargs['inst'] = self.gradio_app_instruct.share_url
        return share_kwargs
