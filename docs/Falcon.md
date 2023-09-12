# About Falcon Models

Sure, here is a document about Falcon Models:

**Falcon Models**

Falcon Models is a family of large language models (LLMs) developed by the Technology Innovation Institute (TII) in Abu
Dhabi. The models are trained on a massive dataset of text and code, and can be used for a variety of tasks, including

* Natural language understanding (NLU)
* Natural language generation (NLG)
* Machine translation
* Text summarization
* Question answering
* Code generation

The Falcon models are available under the Apache 2.0 license, which means that they can be freely used, modified, and
redistributed.

**Falcon-40B**

The Falcon-40B is the largest model in the Falcon family. It has 40 billion parameters, and is trained on a dataset of
500 billion words. The model is capable of state-of-the-art performance on a variety of NLP tasks.

**Falcon-7B**

The Falcon-7B is a smaller version of the Falcon-40B. It has 7 billion parameters, and is trained on a dataset of 100
billion words. The model is still capable of achieving strong performance on NLP tasks, but it is more efficient to
train and deploy.

**Falcon-180B**

The Falcon-180B is the newest model in the Falcon family. It has 180 billion parameters, and is trained on a dataset of
2 trillion words. The model is the largest openly available LLM, and it is capable of achieving state-of-the-art
performance on a variety of NLP tasks.

**Use Cases**

The Falcon models can be used for a variety of tasks, including:

* Natural language understanding (NLU): The Falcon models can be used to understand the meaning of text, such as
  identifying the entities and relationships in a sentence.
* Natural language generation (NLG): The Falcon models can be used to generate text, such as writing different kinds of
  creative content, like poems, code, scripts, musical pieces, email, letters, etc.
* Machine translation: The Falcon models can be used to translate text from one language to another.
* Text summarization: The Falcon models can be used to summarize a text document into a shorter, more concise version.
* Question answering: The Falcon models can be used to answer questions about a text document.
* Code generation: The Falcon models can be used to generate code, such as Python scripts or Java classes.

**Availability**

The Falcon models are available through the Hugging Face Hub. The models are also available in the TensorFlow Hub and
the PyTorch Hub ( and EasyDel).

**Conclusion**

The Falcon models are a powerful family of LLMs that can be used for a variety of tasks. The models are open source and
available for free, making them a valuable resource for researchers and developers.

## How to Use/Load Them in EasyDel

```python
import jax
from EasyDel.transform import falcon_from_pretrained

params, config = falcon_from_pretrained(
    'tiiuae/falcon-7b',
    device=jax.devices('cpu')[0]  # Offload on CPU
)
```

also keep that in mind that returned `config` includes `.get_partition_rules(fsdp=True)`

#### Use With JaxServer

```python
from EasyDel import JAXServer, FlaxFalconForCausalLM
import jax
from EasyDel.transform import falcon_from_pretrained
from transformers import AutoTokenizer

params, config = falcon_from_pretrained(
    'tiiuae/falcon-7b',
    device=jax.devices('cpu')[0]  # Offload on CPU
)


class FalconJaxServer(JAXServer):
    ...
    # You have to Custom this one yourself as you 
    # need read JaxServer Documents inorder to learn how


server = FalconJaxServer.load_from_params(
    params=params,
    model=FlaxFalconForCausalLM(
        config=config,
        dtype=jax.numpy.bfloat16,  # Im on TPUs
        param_dtype=jax.numpy.bfloat16,  # Im on TPUs
        precision=jax.lax.Precision('fastest'),
        _do_init=False,
        input_shape=(1, 1024)
    ),
    config_model=config,
    add_param_field=True,
    tokenizer=AutoTokenizer.from_pretrained('tiiuae/falcon-7b'),
    verbose=False,
    do_memory_log=True,
    config={
        "max_length": 2048,
        "max_new_tokens": 2048,
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

Done 😇 this method can be used for all the Falcon models