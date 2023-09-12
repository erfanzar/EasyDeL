import EasyDel
import jax.lax
from EasyDel import JAXServer
from fjutils import get_float_dtype_by_name
from EasyDel.transform import llama_from_pretrained
from transformers import AutoTokenizer
import gradio as gr

import argparse

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant in the rule of a operator. Always answer " \
                        "as helpfully as possible, while being safe.  Your answers should not" \
                        " include any harmful, unethical, racist, sexist, toxic, dangerous, or " \
                        "illegal content. Please ensure that your responses are socially unbiased " \
                        "and positive in nature.\nIf a question does not make any sense, or is not " \
                        "factually coherent, explain why instead of answering something not correct. If " \
                        "you don't know the answer to a question, please don't share false information. " \
                        "and some time you will receive and extra data between " \
                        "tag of [EXTRA-DATA] and [/EXTRA-DATA] and you have to answer based on that extra data if you" \
                        "received one"


def get_prompt_llama2_format(message: str, chat_history,
                             system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


class Llama2Host(JAXServer):
    def __init__(self, config=None):
        super().__init__(config=config)

    @classmethod
    def load_from_torch(cls, repo_id, config=None):
        with jax.default_device(jax.devices('cpu')[0]):
            param, config_model = llama_from_pretrained(
                repo_id
            )
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = EasyDel.FlaxLlamaForCausalLM(
            config=config_model,
            dtype=get_float_dtype_by_name(config['dtype'] if config is not None else 'fp16'),
            param_dtype=get_float_dtype_by_name(config['dtype'] if config is not None else 'fp16'),
            precision=jax.lax.Precision('fastest'),
            _do_init=False
        )
        return cls.load_from_params(
            config_model=config_model,
            model=model,
            config=config,
            params=param,
            tokenizer=tokenizer,
            add_params_field=True,
            do_memory_log=False
        )

    @classmethod
    def load_from_jax(cls, repo_id, checkpoint_path, config_repo=None, config=None):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        config_model = EasyDel.LlamaConfig.from_pretrained(config_repo or repo_id)
        model = EasyDel.FlaxLlamaForCausalLM(
            config=config_model,
            dtype=get_float_dtype_by_name(config['dtype'] if config is not None else 'fp16'),
            param_dtype=get_float_dtype_by_name(config['dtype'] if config is not None else 'fp16'),
            precision=jax.lax.Precision('fastest'),
            _do_init=False
        )
        return cls.load(
            path=path,
            config_model=config_model,
            model=model,
            config=config,
            tokenizer=tokenizer,
            add_params_field=True,
            do_memory_log=False
        )

    def process_gradio_chat(self, prompt, history, max_new_tokens, greedy, pbar=gr.Progress()):
        string = get_prompt_llama2_format(
            message=prompt,
            chat_history=history,
            system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        response, _ = self.process(
            string=string,
            greedy=greedy,
            max_new_tokens=max_new_tokens,
            pbar=pbar
        )
        history.append([prompt, response])
        return '', history

    def process_gradio_instruct(self, prompt, system, max_new_tokens, greedy, pbar=gr.Progress()):
        string = get_prompt_llama2_format(system_prompt=DEFAULT_SYSTEM_PROMPT, message=prompt, chat_history=[])
        response, _ = self.process(
            string=string,
            greedy=greedy,
            max_new_tokens=max_new_tokens,
            pbar=pbar
        )
        return '', response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for Llama2.")
    parser.add_argument(
        '--repo_id',
        default='meta-llama/Llama-2-7b-chat-hf',
        help='HuggingFace Repo to load model From'
    )
    parser.add_argument(
        "--contains_auto_format",
        default=False,
        action="store_true",
        help="Whether the input text contains auto-format tokens.",
    )
    parser.add_argument(
        "--max_length",
        default=4096,
        type=int,
        help="The maximum length of the input text.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=2048,
        type=int,
        help="The maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--max_stream_tokens",
        default=32,
        type=int,
        help="The maximum number of tokens to generate per stream.",
    )
    parser.add_argument(
        "--temperature",
        default=0.6,
        type=float,
        help="The temperature of the sampling distribution.",
    )
    parser.add_argument(
        "--top_p",
        default=0.95,
        type=float,
        help="The top-p probability cutoff for the sampling distribution.",
    )
    parser.add_argument(
        "--top_k",
        default=50,
        type=int,
        help="The top-k number of tokens to keep for the sampling distribution.",
    )
    parser.add_argument(
        "--logging",
        default=False,
        action="store_true",
        help="Whether to log the generation process.",
    )
    parser.add_argument(
        "--mesh_axes_names",
        default=["dp", "fsdp", "mp"],
        nargs="+",
        help="The names of the mesh axes.",
    )
    parser.add_argument(
        "--mesh_axes_shape",
        default=[1, -1, 1],
        nargs="+",
        type=int,
        help="The shapes of the mesh axes.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        help="The data type to use for the generation.",
    )
    parser.add_argument(
        "--use_prefix_tokenizer",
        default=False,
        action="store_true",
        help="Whether to use a prefix tokenizer.",
    )
    args = parser.parse_args()
    configs = {
        "repo_id": args.repo_id,
        "contains_auto_format": args.contains_auto_format,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "max_stream_tokens": args.max_stream_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "logging": args.logging,
        "mesh_axes_names": args.mesh_axes_names,
        "mesh_axes_shape": args.mesh_axes_shape,
        "dtype": args.dtype,
        "use_prefix_tokenizer": args.use_prefix_tokenizer
    }
    for key, value in configs.items():
        print('\033[1;36m{:<30}\033[1;0m : {:>30}'.format(key.replace('_', ' '), f"{value}"))
    server = Llama2Host.load_from_torch(
        repo_id=args.repo_id,
        config=configs
    )
    try:
        print('\033[1;36mLaunching Chat App ...\033[1;0m')
        server.gradio_app_chat.launch(share=True)
        print('\033[1;36mLaunching Instruct App ...\033[1;0m')
        server.gradio_app_instruct.launch(share=True)
        print('\033[1;36mLaunching Server APIS (Fire) ...\033[1;0m')
        server.fire()
    except KeyboardInterrupt:
        print('Exiting ...')
        server.end()
        exit(0)
