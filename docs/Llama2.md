## About Llama2 Models

**Llama2 Models**

Llama2 Models is a family of pretrained and fine-tuned large language models (LLMs) developed by Meta AI. The models are
trained on a massive dataset of text and code, and can be used for a variety of tasks, including

* Natural language understanding (NLU)
* Natural language generation (NLG)
* Machine translation
* Text summarization
* Question answering
* Code generation

The Llama2 models are available under the Apache 2.0 license, which means that they can be freely used, modified, and
redistributed.

**Model Architecture**

The Llama2 models are based on the Transformer architecture, which is a neural network architecture that has been shown
to be very effective for NLP tasks. The models are trained using a technique called masked language modeling, which
involves predicting the missing words in a sequence of text.

**Model Sizes**

The Llama2 models come in a variety of sizes, ranging from 7 billion to 70 billion parameters. The larger models have
more capacity to learn complex patterns in language, but they are also more computationally expensive to train and
deploy.

**Fine-tuning**

The Llama2 models are pretrained on a massive dataset of text and code, but they can be further fine-tuned on a specific
task to improve their performance. Fine-tuning involves training the model on a dataset of labeled data for the specific
task.

**Use Cases**

The Llama2 models can be used for a variety of tasks, including:

* Natural language understanding (NLU): The Llama2 models can be used to understand the meaning of text, such as
  identifying the entities and relationships in a sentence.
* Natural language generation (NLG): The Llama2 models can be used to generate text, such as writing different kinds of
  creative content, like poems, code, scripts, musical pieces, email, letters, etc.
* Machine translation: The Llama2 models can be used to translate text from one language to another.
* Text summarization: The Llama2 models can be used to summarize a text document into a shorter, more concise version.
* Question answering: The Llama2 models can be used to answer questions about a text document.
* Code generation: The Llama2 models can be used to generate code, such as Python scripts or Java classes.

**Availability**

The Llama2 models are available through the Hugging Face Hub. The models are also available in the TensorFlow Hub , the
PyTorch Hub and EasyDel.

**Conclusion**

The Llama2 models are a powerful family of LLMs that can be used for a variety of tasks. The models are open source and
available for free, making them a valuable resource for researchers and developers.

## How to Use/Load Them in EasyDel

```python
from EasyDel import AutoEasyDelModelForCausalLM
model, params = AutoEasyDelModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b',
    # other kwargs
)
```

also keep that in mind that returned `config` includes `.get_partition_rules(fsdp=True)`

#### Use With JaxServer

```python
from EasyDel.serve import JAXServer, JAXServerConfig
import jax
from transformers import AutoTokenizer

from EasyDel import AutoEasyDelModelForCausalLM

model, params = AutoEasyDelModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b',
    # other kwargs
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant and act as wanted"


class Llama2JaxServer(JAXServer):
    def sample_gradio_chat(self, prompt, history, max_new_tokens, system, greedy):

        system = None if system == "" else system
        string = self.prompt_llama2_model(
            message=prompt,
            chat_history=history or [],
            system_prompt=system or DEFAULT_SYSTEM_PROMPT
        )
        if not self.server_config.stream_tokens_for_gradio:
            response = ""
            for response, _ in self.sample(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
            ):
                ...
            history.append([prompt, response])
        else:
            history.append([prompt, ""])
            for response, _ in self.sample(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens
            ):
                history[-1][-1] = response
                yield "", history

        return "", history

    def sample_gradio_instruct(self, prompt, system, max_new_tokens, greedy):
        string = self.prompt_llama2_model(system_prompt=DEFAULT_SYSTEM_PROMPT, message=prompt, chat_history=[])
        if not self.server_config.stream_tokens_for_gradio:
            response = ""
            for response, _ in self.sample(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
            ):
                pass
        else:
            response = ""
            for response, _ in self.sample(
                    string=string,
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
                    stream=True
            ):
                yield "", response
        return "", response

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
        return "".join(texts)


server = Llama2JaxServer.from_parameters(
    params=params,
    model=model,
    config_model=model.config,
    add_params_field=True,
    tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b'),
    verbose=False,
    do_memory_log=True,
    server_config=JAXServerConfig()
)

server.fire()  # Launch FastAPI functions

shared_urls = server.launch(
    share_chat=True,
    share_inst=True
)
```

Done 😇 this method can be used for all the llama2 models