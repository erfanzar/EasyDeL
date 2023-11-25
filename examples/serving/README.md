# EasyDel Serving Models

## JAXServer

`JaxServer` is the class that host Jax/Flax Models in EasyDel and you can configure and use your Jax
Model in EasyDel JAXServer with less than 50 lines of Code

## Example of using

### Example of Serving LLama 7B Instruct

```python
from EasyDel import JAXServer
from huggingface_hub import hf_hub_download
from src import EasyDel
from transformers import AutoTokenizer


def main():
    # Hosting Pretrained EasyDel ITDF model
    ref = hf_hub_download('erfanzar/EasyDelCollection', 'ITDF-OpenLlama-easydel-v0')

    conf = EasyDel.configs.configs.llama_configs['7b']
    config = EasyDel.LlamaConfig(**conf, rotary_type='complex')
    config.use_flash_attention = False
    config.use_mlp_attention = False
    config.rope_scaling = None
    config.max_sequence_length = 2048
    config.use_pjit_attention_force = False
    model = EasyDel.FlaxLlamaForCausalLM(config, _do_init=False)
    tokenizer = AutoTokenizer.from_pretrained('erfanzar/JaxLLama')

    server = JAXServer.load(
        path=ref,
        model=model,
        tokenizer=tokenizer,
        config_model=config,
        add_params_field=True,
        config=None,
        init_shape=(1, 1)
    )
    # Launching the server
    server.fire()  # Launching Server Post and Get APIs with FastApi on https://ip:port/
    server.gradio_app_instruct.launch(share=True)  # Up On https://ip:port/gradio_instructy
    server.gradio_app_chat.launch(share=True)  # Up On https://ip:port/gradio_chat


if __name__ == "__main__":
    main()
```

you can also load model itself from parameters like

```python
import EasyDel.transform

model_id = 'meta-llama/Llama.md-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

params, config = EasyDel.transform.llama_from_pretrained(
    model_id=model_id
)

with jax.default_device(jax.devices('cpu')[0]):
    model = EasyDel.FlaxLlamaForCausalLM(
        config=config,
        _do_init=False,
        dtype='float16',
        param_dtype='float16'
    )

server = JAXServer.load_from_params(
    model=model,
    config_model=config,
    config=config_server,
    tokenizer=tokenizer,
    params=params,
    add_params_field=True
)
```

### FastAPI 🌪

#### Instruct API

to Override this api you have to code `forward_instruct` just like what you want the default implementation of this
function is

```python
def forward_instruct(self, data: InstructRequest):
    if not self._funcs_generated:
        return {
            'status': "down"
        }

    string = self.config.instruct_format.format(instruct=data.prompt, system=data.system)
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
```

* BaseModel Class For PYData in FastAPI :

```python
class InstructRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    temperature: Optional[float] = None
    greedy: Optional[bool] = False
```

* And here's an example of using this api via python and creating a simple client with using `requests` library in
  python :

```python
import requests

content = {
    'prompt': 'can you code a simple neural network in c++ for me',
    'system': 'You are an AI assistant generate short and useful response',
    'temperature': 0.1,
    'greedy': False
}

response = requests.post(
    url='http://ip:port/instruct',
    json=content
).json()

print(response['response'])
# Response of model
print(response['input'])
# The input passed to the model

```

#### Chat API

to Override this api you have to code `forward_chat` just like what you want the default implementation of this function
is

```python
def forward_chat(self, data: ChatRequest):
    if not self._funcs_generated:
        return {
            'status': "down"
        }

    history = self.process_chat_history(data.history or [])
    history += self.config.prompt_prefix_chat + data.prompt + self.config.prompt_postfix_chat

    response, used_tokens = [None] * 2
    for response, used_tokens in self.process(
            string=history,
            greedy=data.greedy,
            max_new_tokens=None
    ):
        ...
    self.number_of_served_request_until_last_up_time += 1
    return {
        'input': f'{history}',
        'response': response,
        'tokens_used': used_tokens,
    }
```

* BaseModel Class For PYData in FastAPI :

```python
class ChatRequest(BaseModel):
    prompt: str
    history: Union[List[List], None] = None
    temperature: Optional[float] = None
    greedy: Optional[bool] = False
```

* And here's an example of using this api via python and creating a simple client with using `requests` library in
  python :

```python
import requests

content = {
    'prompt': 'can you code a simple neural network in c++ for me',
    'history': [
        ['hello how are you', 'Hello\nthanks, im here to assist you you have any question that i could help you with']
    ],
    'temperature': 0.1,
    'greedy': False
}

response = requests.post(
    url='http://ip:port/chat',
    json=content
).json()

print(response['response'])
# Response of model
print(response['input'])
# The input passed to the model

```

#### Status

Simply by sending a get API to `https://ip:port/status` you will receive base information about the server and
how it being run, num cores in use, number of generated prompt , number of request and ..

### Example of using Server Without hosting it

```python
from EasyDel.serve import JAXServer
from huggingface_hub import hf_hub_download
import EasyDel
from transformers import AutoTokenizer


def main():
    # Hosting Pretrained EasyDel ITDF model
    ref = hf_hub_download('erfanzar/EasyDelCollection', 'ITDF-OpenLlama-easydel-v0')

    conf = EasyDel.configs.configs.llama_configs['7b']
    config = EasyDel.LlamaConfig(**conf, rotary_type='complex')
    config.use_flash_attention = False
    config.use_mlp_attention = False
    config.rope_scaling = None
    config.max_sequence_length = 2048
    config.use_pjit_attention_force = False
    model = EasyDel.FlaxLlamaForCausalLM(config, _do_init=False)
    tokenizer = AutoTokenizer.from_pretrained('erfanzar/JaxLLama')

    server = JAXServer.load(
        path=ref,
        model=model,
        tokenizer=tokenizer,
        config_model=config,
        add_params_field=True,
        init_shape=(1, 1),
        config=None,
    )
    # Predicting with Server
    text = "what is different between llama and alpacas"
    pred = server.forward_instruct_non_api(text, 'You Are an helpful AI Assistant, generate long and useful responses',
                                           False)
    # Now Pred is dict that contains input that have been passed to the model and answer
    print('Input:\n', pred['input'])
    print('Response:\n', pred['response'])


if __name__ == "__main__":
    main()
```

and your output depends on you configs bae prompting type should be something like

```text
Input:
    ### SYSTEM:
    You Are an helpful AI Assistant, generate long and useful responses
    ### INSTRUCT:
    what is different between llama and alpacas
    ### ASSISTANT:

Response:
    Llamas are domesticated members of the Camelidae family. They are known for their long, thick, 
    luxurious coats, which are used to make coats, blankets, and other items. Alpacas are also domesticated members of the 
    Camelidae family, but they are smaller and have shorter, thinner coats. They are raised for their fiber, which is used 
    to make yarn and textiles. Both llamas and alpacas are native to South America, but they are now found in other parts 
    of the world as well.
```

### Predicting with model from base

model also contains the function of process that will only take a string in and will return the output for that without
any type of prompt engineering or anything like that soe you can use this function to override your own functions

```python
string = 'hello my name id daniel and i'
response, num_generated_tokens = server.process(
    string=string,
    greedy=False,
    max_new_tokens=8,
)

print(string + response)

# Output is "hello my name id daniel and i work at a coffe shop" 
```

### Creating your own generation function

`JAXServer` uses  `configure_generate_functions` to generate functions and that looks like

```python
import jax.sharding
import logging
import copy
import functools
from fjformer import with_sharding_constraint
from jax.experimental.pjit import pjit
from transformers import GenerationConfig

Ps = jax.sharding.PartitionSpec


def configure_generate_functions(self, model, tokenizer):
    # Your override ed function in case that you want to create custom configure generation configs must contain
    # These attributes
    assert self.rules is not None, 'you should first shard params with using ``shard_params`` method'

    if tokenizer.pad_token is None:
        logging.info(
            'Tokenizer does not contain padding token setting padding token to eos token for open end generation')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    self.prefix_tokenizer = copy.deepcopy(tokenizer)
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    self.tokenizer = copy.deepcopy(tokenizer)

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

                temperature=self.config.temperature,
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
```

### Configs

_These are all the attributes in Config_

1. host
2. port
3. instruct_format
4. chat_format
5. system_prefix
6. system
7. prompt_prefix_instruct
8. prompt_postfix_instruct
9. prompt_prefix_chat
10. prompt_postfix_chat
11. chat_prefix
12. contains_auto_format
13. max_length
14. max_new_tokens
15. max_stream_tokens
16. temperature
17. top_p
18. top_k
19. logging
20. mesh_axes_names
21. mesh_axes_shape
22. dtype
23. seed
24. use_prefix_tokenizer

### Host (str)

The hostname or IP address of the server.(By Default its up on localhost `0.0.0.0`)

### Port (int)

The port number of the server.

``Defualt is 2059``

### instruct_format (str)

The format of the instructions that will be sent to the server.

for example the default is :

```text
'### SYSTEM:\n{system}\n### INSTRUCT:\n{instruct}\n### ASSISTANT:\n'
```

### chat_format (str)

The format of the chat messages that will be sent to the server.

for example the default is :

```text
<|prompter|>{prompt}</s><|assistant|>{assistant}</s>
```

### system_prefix (str)

The prefix that will be used for system messages from the server.

``Defualt is ""``

### system (str)

The default system Message Like `You are an AI assitant generate good and useful responses`

### prompt_prefix_instruct (str)

The prefix that will be used for prompts in instructions.

``Defualt is ""``

### prompt_postfix_instruct (str)

The postfix that will be used for prompts in instructions.

``Defualt is ""``

### prompt_prefix_chat (str)

The prefix that will be used for prompts in chat messages.

for example the default is :

```text
'<|prompter|>'
```

### prompt_postfix_chat (str)

The postfix that will be used for prompts in chat messages.

for example the default is :

```text
'</s><|assistant|>'
```

### chat_prefix (str)

The prefix that will be used for chat messages from the server.

``Defualt is ""``

### contains_auto_format (bool)

Whether the instructions and chat messages should be checked for the auto format.

``Defualt is True``

#### Note If you want to use customized autoformat you should override the function process_chat_history

```python
import typing


def process_chat_history(self, history: typing.List):
    if len(history) == 0:
        return ''
    else:
        message_history = ''
        for message in history:
            message_history += self.config.chat_format.format(prompt=message[0], assistant=message[1])

    return message_history
```

### max_length (int)

The maximum length of the instructions and chat messages.

``Defualt is 2048``

### max_new_tokens (int)

The maximum number of new tokens that can be generated in each instruction or chat message.

``Defualt is 2048``

### max_stream_tokens (int)

The maximum number of tokens that can be generated in a stream of instructions or chat
messages.

``Defualt is 32``

### temperature (float)

The temperature of the language model.

``Defualt is 0.1``

### top_p (float)

The top-p probability threshold for the language model.

``Defualt is 0.95``

### top_k (int)

The top-k tokens to consider for the language model.

``Defualt is 50``

### logging (bool)

Whether to enable logging.

``Defualt is False``

### mesh_axes_names (list of str)

The names of the mesh axes.

``Defualt is ('dp','fsdp','mp)``

##### Note do Not change Mesh Axes names in case of you dont want to use fully customized model

### mesh_axes_shape (list of int)

The shapes of the mesh axes.

``Defualt is (1,-1,1)``

### dtype (str)

The data type of the model.

``Defualt is fp16``

### seed (int)

The random seed.

``Defualt is 556``

### use_prefix_tokenizer (bool)

Whether to use a prefix tokenizer for the language model.

``Defualt is True``