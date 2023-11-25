import threading
import gradio as gr
import transformers
import uvicorn
from fastapi import FastAPI
from typing import List
from ml_collections import ConfigDict
from transformers import TextIteratorStreamer
import logging
import multiprocessing as mp
import torch
from .utils import ChatRequest, InstructRequest, seafoam


class PytorchServerConfig:
    def __init__(self,
                 host='0.0.0.0',
                 port=2059,
                 batch_size=1,
                 contains_auto_format=True,
                 max_length=2048,
                 max_new_tokens=2048,
                 temperature=0.8,
                 top_p=0.95,
                 top_k=50,
                 logging=True,
                 dtype='fp16',
                 max_number_of_gpus=None,
                 max_gpu_perc_to_use=0.95,
                 max_stream_tokens: int = 1
                 ):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.contains_auto_format = contains_auto_format
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.logging = logging
        self.dtype = dtype
        self.max_number_of_gpus = max_number_of_gpus
        self.max_gpu_perc_to_use = max_gpu_perc_to_use
        self.max_stream_tokens = max_stream_tokens


class PyTorchServer(object):

    def __init__(self, config: PytorchServerConfig):
        self.model, self.tokenizer = [None] * 2

        # logging.warning('PytorchServer is not built fully yet at this version')

        self.config = config
        self.process_uvicorn = None
        self.app = FastAPI()
        self.number_of_served_request_until_last_up_time = 0
        self.device_rolling = self.get_gpu_memory(self.config.max_number_of_gpus)
        self.dict_max_memory_sharding = {i: str(int(mem * self.config.max_gpu_perc_to_use)) + 'GiB' for i, mem in
                                         enumerate(self.device_rolling)}
        self.app.post('/chat')(self.forward_chat_fast_api)
        self.app.post('/instruct')(self.forward_instruct_fast_api)
        self.app.get('/status')(self.status)
        self.app = gr.mount_gradio_app(self.app, self.create_gradio_ui_chat(), '/gradio_chat')
        self.app = gr.mount_gradio_app(self.app, self.create_gradio_ui_instruct(), '/gradio_instruct')

    @staticmethod
    def get_gpu_memory(num_gpus_req=None):

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
        if self.config.dtype == 'fp16':
            dtype = torch.float16
        elif self.config.dtype == 'fp32':
            dtype = torch.float32
        elif self.config.dtype == 'bf16':
            dtype = torch.bfloat16
        else:
            raise ValueError('unknown type available types are [fp32 fp16 bf16]')
        load_kwargs = {
            'torch_dtype': dtype,
            'device_map': 'auto',
            'max_memory': self.dict_max_memory_sharding
        }
        return load_kwargs

    def status(self):

        return {
            'config': {k: v for k, v in self.config.__dict__.items()},
            'devices': f"{torch.cuda.device_count()}",
            'device_sharding': self.device_rolling,
            'max_memory': self.dict_max_memory_sharding,
            'status': 'Ready',
            'number_of_served_request_until_last_up_time': f"{self.number_of_served_request_until_last_up_time}"
        }

    def forward_instruct_fast_api(self, data: InstructRequest):
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
            'response': response
        }

    def forward_chat_fast_api(self, data: ChatRequest):
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
            'response': response
        }

    @staticmethod
    def format_instruct(system: str, instruction: str) -> str:
        raise NotImplementedError()

    @staticmethod
    def format_chat(history: List[str], prompt: str, system: str = None) -> str:
        raise NotImplementedError()

    def process(self,
                string: str,
                max_new_tokens: int = None,
                max_length: int = None,
                temperature: float = 0.6,
                top_k=50,
                top_p=0.9,
                stream: bool = True,
                sample: bool = True

                ):
        assert self.model is not None, 'you should first load model with ``load`` method'
        tokens = self.tokenizer(
            string,
            return_tensors='pt'
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
                    do_sample=sample
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

    def load(self, repo_id: str, tokenizer_repo: str = None, auto_config: bool = True, **kwargs):
        load_kwargs = kwargs if not auto_config else self.get_model_load_kwargs()
        load_kwargs = load_kwargs | kwargs
        model = transformers.AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            **load_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_repo or repo_id,
            trust_remote_code=True
        )

        self.model = model
        self.tokenizer = tokenizer

    def process_gradio_chat(self,
                            prompt: str,
                            history: List[str],
                            max_new_tokens: int,
                            temperature: float,
                            max_length: int,
                            top_p: float,
                            top_k: int
                            ):
        string = self.format_chat(prompt=prompt, history=history, system=None)
        history.append([prompt, ''])
        responses = ''
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
            yield '', history

    def process_gradio_instruct(self,
                                instruction: str,
                                system: str,
                                max_new_tokens: int,
                                temperature: float,
                                max_length: int,
                                top_p: float,
                                top_k: int
                                ):
        string = self.format_instruct(system=system, instruction=instruction)
        responses = ''
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
            yield '', response

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
                                               label='Max New Tokens', step=self.config.max_stream_tokens)
                    max_length = gr.Slider(value=self.config.max_length, maximum=self.config.max_length, minimum=1,
                                           label='Max Length', step=1)
                    temperature = gr.Slider(value=0.2, maximum=1, minimum=0.1, label='Temperature', step=0.01)
                    top_p = gr.Slider(value=0.9, maximum=1, minimum=0.1, label='Top P', step=0.01)
                    top_k = gr.Slider(value=50, maximum=100, minimum=1, label='Top K', step=1)

            inputs = [
                prompt,
                history,
                max_new_tokens,
                temperature,
                max_length,
                top_p,
                top_k
            ]
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
                system = gr.Textbox(value='You Are an helpful AI Assistant, generate good and helpful answers',
                                    show_label=False, placeholder='System Message', container=False)

            with gr.Row():
                with gr.Accordion('Advanced Options', open=False):
                    max_new_tokens = gr.Slider(value=self.config.max_new_tokens, maximum=10000,
                                               minimum=self.config.max_stream_tokens,
                                               label='Max New Tokens', step=self.config.max_stream_tokens, )
                    max_length = gr.Slider(value=self.config.max_length, maximum=self.config.max_length, minimum=1,
                                           label='Max Length', step=1)
                    temperature = gr.Slider(value=0.2, maximum=1, minimum=0.1, label='Temperature', step=0.01)
                    top_p = gr.Slider(value=0.9, maximum=1, minimum=0.1, label='Top P', step=0.01)
                    top_k = gr.Slider(value=50, maximum=100, minimum=1, label='Top K', step=1)

            inputs = [
                prompt,
                system,
                max_new_tokens,
                temperature,
                max_length,
                top_p,
                top_k
            ]
            sub_event = submit.click(fn=self.process_gradio_instruct, inputs=inputs, outputs=[prompt, pred])

            def clear_():
                return ''

            clear.click(fn=clear_, outputs=[pred])
            txt_event = prompt.submit(fn=self.process_gradio_instruct, inputs=inputs, outputs=[prompt, pred])

            stop.click(fn=None, inputs=None, outputs=None, cancels=[txt_event, sub_event])

        block.queue()
        return block

    def fire(self):
        def run():
            uvicorn.run(self.app, host=self.config.host, port=self.config.port)

        self.process_uvicorn = mp.Process(target=run)
        self.process_uvicorn.start()

    def end(self):
        if self.process_uvicorn is not None:
            self.process_uvicorn.join()
        else:
            logging.warning('you have to fire server before ending that this command will be ignored')
