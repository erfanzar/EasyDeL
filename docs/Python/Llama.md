# About Llama Models

* **Introduction**

Llama models are a family of large language models (LLMs) developed by Meta AI. They are trained on a massive dataset of
text and code, and they can be used for a variety of tasks, such as text generation, translation, summarization,
question answering, code generation, and natural language inference.

* **Model Architecture**

Llama models are based on the Transformer architecture, which is a neural network architecture that has been shown to be
very effective for natural language processing tasks. The Transformer architecture uses self-attention to learn
long-range dependencies between words in a sentence.

* **Training Data**

Llama models are trained on a massive dataset of text and code. The text dataset includes text from a variety of
sources, such as books, articles, and websites. The code dataset includes code from a variety of programming languages,
such as Python, Java, and C++.

* **Fine-tuning**

After being pre-trained on a massive dataset, Llama models can be fine-tuned for specific tasks. Fine-tuning involves
training the model on a smaller dataset of data that is relevant to the specific task.

* **Applications**

Llama models can be used for a variety of tasks, such as:

    * Text generation: Llama models can be used to generate text, such as poems, code, scripts, and musical pieces.
    * Translation: Llama models can be used to translate text from one language to another.
    * Summarization: Llama models can be used to summarize text.
    * Question answering: Llama models can be used to answer questions about text.
    * Code generation: Llama models can be used to generate code.
    * Natural language inference: Llama models can be used to determine the relationship between two sentences.

* **Availability**

Llama models are available for free for research and commercial use. They can be downloaded from the Hugging Face Hub.

* **Limitations**

Llama models are still under development, and they have some limitations. For example, they can sometimes generate
incorrect or misleading text. They can also be biased, reflecting the biases that are present in the training data.

* **Future Work**

Llama models are a promising new technology with the potential to be used for a variety of applications. Future work on
Llama models will focus on improving their accuracy, reducing their bias, and making them more robust to errors.

* Text generation
* Translation
* Summarization
* Question answering
* Code generation
* Natural language inference

Here is a table comparing the different sizes of Llama models:

| Model     | Parameters |
|-----------|------------|
| Llama 7B  | 7 billion  |
| Llama 13B | 13 billion |
| Llama 33B | 33 billion |
| Llama 65B | 65 billion |
| Llama 70B | 70 billion |

## How to Use/Load Them in EasyDel

```python
import jax
from EasyDel.transform import llama_from_pretrained

params, config = llama_from_pretrained(
    'meta-llama/Llama-2-7b',
    device  # Offload on CPU
)
```

also keep that in mind that returned `config` includes `.get_partition_rules(fsdp=True)`

#### Use With JaxServer

```python
from EasyDel.serve import JAXServer
from EasyDel.modules.llama import FlaxLlamaForCausalLM
import jax
from EasyDel.transform import llama_from_pretrained
from transformers import AutoTokenizer

params, config = llama_from_pretrained(
    'meta-llama/Llama-2-7b',
    device  # Offload on CPU
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant and act as wanted"


class Llama2JaxServer(JAXServer):
    def process_gradio_chat(self, prompt, history, max_new_tokens, system, greedy):

        system = None if system == '' else system
        string = self.prompt_llama2_model(
            message=prompt,
            chat_history=history or [],
            system_prompt=system or DEFAULT_SYSTEM_PROMPT
        )
        if not self.config.stream_tokens_for_gradio:
            response = ''
            for response, _ in self.process(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
            ):
                ...
            history.append([prompt, response])
        else:
            history.append([prompt, ''])
            for response, _ in self.process(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens
            ):
                history[-1][-1] = response
                yield '', history

        return '', history

    def process_gradio_instruct(self, prompt, system, max_new_tokens, greedy):
        string = self.prompt_llama2_model(system_prompt=DEFAULT_SYSTEM_PROMPT, message=prompt, chat_history=[])
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

    @staticmethod
    def prompt_llama2_model(message: str, chat_history,
                            system_prompt: str) -> str:

        do_strip = False
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)


server = Llama2JaxServer.load_from_params(
    params=params,
    model=FlaxLlamaForCausalLM(
        config=config,
        dtype=jax.numpy.bfloat16,  # Im on TPUs
        param_dtype=jax.numpy.bfloat16,  # Im on TPUs
        precision=jax.lax.Precision('fastest'),
        _do_init=False,
        input_shape=(1, 1024)
    ),
    config_model=config,
    add_params_field=True,
    tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b'),
    verbose=False,
    do_memory_log=True,
    config={
        "max_length": 4096,
        "max_new_tokens": 4096,
        "max_stream_tokens": 64,
        "dtype": 'bf16',
        "use_prefix_tokenizer": True,
        'pre_compile': True
    }
)

server.fire()  # Launch FastAPI functions

shared_urls = server.launch(
    share_chat=True,
    share_inst=True
)
```

Done 😇 this method can be used for all the llama models