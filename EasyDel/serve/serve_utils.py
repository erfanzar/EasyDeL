import typing

import gradio as gr
import uvicorn
from fastapi import FastAPI
from jax import numpy as jnp

import jax
from flax.core import freeze
from jax.experimental import mesh_utils
from fjutils import utils
from flax.traverse_util import unflatten_dict
from fjutils import easylm
import multiprocessing as mp
from ml_collections import ConfigDict
from pydantic import BaseModel
from EasyDel.serve.theme import seafoam

dtypes = {
    'fp16': jnp.float16,
    'bf16': jnp.bfloat16,
    'fp32': jnp.float32,
    'fp64': jnp.float64,

}


def get_dtype(dtype):
    if isinstance(dtype, str):
        dtype = dtypes[dtype]
    return dtype


read_ckpt = utils.read_ckpt
create_shard_gather_fns = easylm.make_shard_and_gather_fns
match_partition_rules = easylm.match_partition_rules
with_sharding_constraint = easylm.with_sharding_constraint
get_jax_mesh = easylm.get_jax_mesh


def shard_params(params, partition_rules,
                 shard_mesh_shape=(1, -1, 1),
                 backend='gpu',
                 shard_mesh=('dp', 'fsdp', 'mp'), do_unf=True,
                 dtype='fp16'):
    dtype = get_dtype(dtype)
    params = unflatten_dict(params) if do_unf else params
    params = freeze(params)
    mxd = jax.device_count(backend)
    rsp = jnp.asarray([1, mxd, 1]).reshape(shard_mesh_shape)
    phs_mesh = mesh_utils.create_device_mesh(rsp.tolist(), )
    mesh = jax.sharding.Mesh(phs_mesh, shard_mesh)
    ps = match_partition_rules(
        partition_rules,
        params
    )
    with mesh:
        shard_fns, _ = create_shard_gather_fns(
            ps, dtype
        )
        params = jax.tree_util.tree_map(lambda fn, x: fn(x), shard_fns, params)
    return params, mesh


class InstructRequest(BaseModel):
    system: typing.Optional[typing.List[str]] = None
    prompt: typing.List[str] = None
    temperature: typing.Optional[float] = None


class ChatRequest(BaseModel):
    prompt: str
    context: typing.List[str] = ''
    temperature: typing.Optional[float] = None


class JAXServer(object):

    def __init__(self, config=None):
        self.config = self.get_default_config(config)
        self.app = FastAPI()
        self.number_of_served_request_until_last_up_time = 0
        self.app.post('/chat')(self.forward_chat)
        self.app.post('/instruct')(self.forward_instruct)
        self.app.get('/status')(self.status)
        self.app = gr.mount_gradio_app(self.app, self.create_gradio_ui(), '/gradio_app')

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.host = '0.0.0.0'
        config.port = 2059

        config.instruct_format = '### SYSTEM:\n{system}\n### INSTRUCT:\n{instruct}\n### ASSISTANT:\n'
        config.chat_format = '<|prompter|>{prompt}</s><|assistant|>{assistant}</s>'

        config.batch_size = 1

        config.system_prefix = ''
        config.system = ''

        config.prompt_prefix_instruct = ''
        config.prompt_postfix_instruct = ''

        config.prompt_prefix_chat = '<|prompter|>'
        config.prompt_postfix_chat = '</s>'

        config.is_instruct = False

        config.chat_prefix = ''
        config.contains_auto_format = True
        config.max_length = 2048
        config.temperature = 0.1
        config.logging = False

        config.dtype = 'bfloat16'
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def status(self):
        return {
            'config': {k: v for k, v in self.config.__dict__.items()},
            'devices': f"{jax.devices()}",
            'number_of_backends': len(jax.devices()),
            'status': 'Ready',
            'number_of_served_request_until_last_up_time': f"{self.number_of_served_request_until_last_up_time}"
        }

    def forward_chat(self, data: ChatRequest):
        ...

    def forward_instruct(self, data: InstructRequest):
        ...

    @staticmethod
    def forward(data):
        return NotImplementedError

    def process_chat(self, text, history, max_new_tokens, max_length, temperature):
        print(self)
        return 'NotImplementedYet'

    def process_chat_history(self, history: typing.List):
        if len(history) == 0:
            return ''
        else:
            message_history = ''
            for message in history:
                message_history += self.config.chat_format.format(prompt=message[0], assistant=message[1])

        return message_history

    def create_gradio_ui(self):
        with gr.Blocks(
                theme=seafoam) as block:
            gr.Markdown("#<h1><center>Powered by [EasyDeL](https://github.com/erfanzar/EasyDel)</center></h1>")
            with gr.Row():
                cache = gr.Chatbot(elem_id="EasyDel", label="EasyDel").style(container=True, height=600)

            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(show_label=False, placeholder='Message Box').style(container=False)
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button(variant="primary")
                        stop = gr.Button(value='Stop ')
                        clear = gr.Button(value='Clear Conversation')

            with gr.Row():
                with gr.Accordion('Advanced Options', open=False):
                    max_new_tokens = gr.Slider(value=2048, maximum=3072, minimum=1, label='Max New Tokens', step=1, )
                    max_length = gr.Slider(value=2048, maximum=4096, minimum=1, label='Max Length', step=1)
                    temperature = gr.Slider(value=0.2, maximum=1, minimum=0.1, label='Temperature', step=0.01)

            inputs = [text, cache, max_new_tokens, max_length, temperature]
            sub_event = submit.click(fn=self.process_chat, inputs=inputs, outputs=[text, cache])

            def clear_():
                return []

            clear.click(fn=clear_, outputs=[cache])
            txt_event = text.submit(fn=self.process_chat, inputs=inputs, outputs=[text, cache])

            stop.click(fn=None, inputs=None, outputs=None, cancels=[txt_event, sub_event])

        block.queue()
        return block

    def fire(self):
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)


class PyTorchServer(object):

    def __init__(self, config=None):
        import torch as tr
        self.torch = tr
        self.config = self.get_default_config(config)
        self.app = FastAPI()
        self.number_of_served_request_until_last_up_time = 0
        self.device_rolling = self.get_gpu_memory(self.config.max_number_of_gpus)
        self.dict_max_memory_sharding = {i: str(int(mem * self.config.max_gpu_perc_to_use)) + 'GiB' for i, mem in
                                         enumerate(self.device_rolling)}
        self.app.post('/chat')(self.forward_chat)
        self.app.post('/instruct')(self.forward_instruct)
        self.app.get('/status')(self.status)
        self.app = gr.mount_gradio_app(self.app, self.create_gradio_ui(), '/gradio_app')

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.host = '0.0.0.0'
        config.port = 2059

        config.instruct_format = '### SYSTEM:\n{system}\n### INSTRUCT:\n{instruct}\n### ASSISTANT:\n'
        config.chat_format = '<|prompter|>{prompt}</s><|assistant|>{assistant}</s>'

        config.batch_size = 1

        config.system_prefix = ''
        config.system = ''

        config.prompt_prefix_instruct = ''
        config.prompt_postfix_instruct = ''

        config.prompt_prefix_chat = '<|prompter|>'
        config.prompt_postfix_chat = '</s>'

        config.is_instruct = False

        config.chat_prefix = ''
        config.contains_auto_format = True
        config.max_length = 2048
        config.temperature = 0.1
        config.logging = False

        config.max_gpu_perc_to_use = 0.90
        config.dtype = 'float16'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def get_gpu_memory(self, num_gpus_req=None):

        gpu_m = []
        dc = self.torch.cuda.device_count()
        num_gpus = self.torch.cuda.device_count() if num_gpus_req is None else min(num_gpus_req, dc)

        for gpu_id in range(num_gpus):
            with self.torch.cuda.device(gpu_id):
                gpu_properties = self.torch.cuda.get_device_properties(self.torch.cuda.current_device())
                gpu_m.append(
                    (gpu_properties.total_memory / (1024 ** 3)) - (self.torch.cuda.memory_allocated() / (1024 ** 3)))
        return gpu_m

    def get_model_load_kwargs(self):
        if self.config.dtype == 'float16':
            dtype = self.torch.float16
        elif self.config.dtype == 'float32':
            dtype = self.torch.float32
        elif self.config.dtype == 'bfloat16':
            dtype = self.torch.bfloat16
        else:
            raise ValueError('unknown type available types are [float32 float16 bfloat16]')
        load_kwargs = {
            'torch_dtype': dtype,
            'device_map': 'auto',
            'max_memory': self.dict_max_memory_sharding
        }
        return load_kwargs

    def status(self):

        return {
            'config': {k: v for k, v in self.config.__dict__.items()},
            'devices': f"{self.torch.cuda.device_count()}",
            'device_sharding': self.device_rolling,
            'max_memory': self.dict_max_memory_sharding,
            'status': 'Ready',
            'number_of_served_request_until_last_up_time': f"{self.number_of_served_request_until_last_up_time}"
        }

    @staticmethod
    def forward(data):
        return NotImplementedError

    def forward_chat(self, data: ChatRequest):
        ...

    def forward_instruct(self, data: InstructRequest):
        ...

    def process_chat(self, text, history, max_new_tokens, max_length, temperature):
        print(self)
        return 'NotImplementedYet'

    def process_chat_history(self, history: typing.List):
        if len(history) == 0:
            return ''
        else:
            message_history = ''
            for message in history:
                message_history += self.config.chat_format.format(prompt=message[0], assistant=message[1])

        return message_history

    def create_gradio_ui(self):
        with gr.Blocks(
                theme=seafoam) as block:
            gr.Markdown("#<h1><center>Powered by [EasyDeL](https://github.com/erfanzar/EasyDel)</center></h1>")
            with gr.Row():
                cache = gr.Chatbot(elem_id="EasyDel", label="EasyDel").style(container=True, height=600)

            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(show_label=False, placeholder='Message Box').style(container=False)
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button(variant="primary")
                        stop = gr.Button(value='Stop ')
                        clear = gr.Button(value='Clear Conversation')

            with gr.Row():
                with gr.Accordion('Advanced Options', open=False):
                    max_new_tokens = gr.Slider(value=2048, maximum=3072, minimum=1, label='Max New Tokens', step=1, )
                    max_length = gr.Slider(value=2048, maximum=4096, minimum=1, label='Max Length', step=1)
                    temperature = gr.Slider(value=0.2, maximum=1, minimum=0.1, label='Temperature', step=0.01)

            inputs = [text, cache, max_new_tokens, max_length, temperature]
            sub_event = submit.click(fn=self.process_chat, inputs=inputs, outputs=[text, cache])

            def clear_():
                return []

            clear.click(fn=clear_, outputs=[cache])
            txt_event = text.submit(fn=self.process_chat, inputs=inputs, outputs=[text, cache])

            stop.click(fn=None, inputs=None, outputs=None, cancels=[txt_event, sub_event])

        block.queue()
        return block

    def fire(self):
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
