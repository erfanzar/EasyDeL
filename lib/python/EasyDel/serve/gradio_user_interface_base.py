import typing

import gradio as gr
from typing import List
from .utils import seafoam


class GradioUserInference:
    @staticmethod
    def chat_interface_components(
            sample_func: typing.Callable,
            max_length: int,
            max_new_tokens: int,
            max_compile_tokens: int
    ):
        """
        The function `chat_interface_components` creates the components for a chat interface, including
        a chat history, message box, buttons for submitting, stopping, and clearing the conversation,
        and sliders for advanced options.
        """
        _max_length = max_length
        _max_new_tokens = max_new_tokens
        _max_compile_tokens = max_compile_tokens
        with gr.Row():
            with gr.Column(scale=4):
                history = gr.Chatbot(
                    elem_id="EasyDel",
                    label="EasyDel",
                    container=True,
                    height=800
                )
                prompt = gr.Textbox(placeholder='Message Box', container=False)
            with gr.Column(scale=1):
                gr.Markdown(
                    "# <h1><center>Powered by [EasyDeL](https://github.com/erfanzar/EasyDel)</center></h1>"
                )
                system_prompt = gr.Textbox(
                    value="",
                    label="System Prompt",
                    placeholder='System Prompt',
                    container=False
                )
                max_length = gr.Slider(
                    value=_max_length,
                    minimum=1,
                    maximum=_max_length,
                    step=1,
                    label="Maximum Length"
                )

                max_new_tokens = gr.Slider(
                    value=_max_new_tokens,
                    maximum=_max_length,
                    minimum=_max_compile_tokens,
                    label='Max New Tokens',
                    step=_max_compile_tokens
                )

                max_compile_tokens = gr.Slider(
                    value=max_compile_tokens,
                    maximum=_max_new_tokens,
                    minimum=max_compile_tokens,
                    label='Max Compile Tokens (JAX)',
                    step=max_compile_tokens,
                )
                temperature = gr.Slider(
                    value=0.8,
                    maximum=1,
                    minimum=0.1,
                    label='Temperature',
                    step=0.01
                )
                top_p = gr.Slider(
                    value=0.9,
                    maximum=1,
                    minimum=0.1,
                    label='Top P',
                    step=0.01
                )
                top_k = gr.Slider(
                    value=50,
                    maximum=100,
                    minimum=1,
                    label='Top K',
                    step=1
                )
                mode = gr.Dropdown(
                    choices=["Chat", "Instruction"],
                    value="Chat",
                    label="Mode",
                    multiselect=False,
                )
                greedy = gr.Checkbox(
                    value=False,
                    label="Greedy"
                )

                stop = gr.Button(
                    value='Stop'
                )
                clear = gr.Button(
                    value='Clear Conversation'
                )
                submit = gr.Button(
                    value="Run",
                    variant="primary"
                )
        inputs = [
            prompt,
            history,
            system_prompt,
            mode,
            max_length,
            max_new_tokens,
            max_compile_tokens,
            greedy,
            temperature,
            top_p,
            top_k
        ]

        clear.click(fn=lambda: [], outputs=[history])
        sub_event = submit.click(
            fn=sample_func, inputs=inputs, outputs=[prompt, history]
        )
        txt_event = prompt.submit(
            fn=sample_func, inputs=inputs, outputs=[prompt, history]
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[txt_event, sub_event]
        )

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
        raise NotImplementedError()

    def build_inference(
            self,
            sample_func: typing.Callable,
            max_length: int,
            max_new_tokens: int,
            max_compile_tokens: int
    ) -> gr.Blocks:
        """
        The function "build_inference" returns a gr.Blocks object that model
        interface components.
        :return: a gr.Blocks object.
        """
        with gr.Blocks(
                theme=seafoam
        ) as block:
            self.chat_interface_components(
                sample_func=sample_func,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                max_compile_tokens=max_compile_tokens
            )
        return block
