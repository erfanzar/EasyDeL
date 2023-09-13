# About MosaicMPT Models

**MosaicMPT Models**

MosaicMPT Models is a family of large language models (LLMs) developed by MosaicML. The models are trained on a massive
dataset of text and code, and can be used for a variety of tasks, including

* Natural language understanding (NLU)
* Natural language generation (NLG)
* Machine translation
* Text summarization
* Question answering
* Code generation

The MosaicMPT models are available under the Apache 2.0 license, which means that they can be freely used, modified, and
redistributed.

**Model Architecture**

The MosaicMPT models are based on the Transformer architecture, which is a neural network architecture that has been
shown to be very effective for NLP tasks. The models are trained using a technique called masked language modeling,
which involves predicting the missing words in a sequence of text.

**Model Sizes**

The MosaicMPT models come in a variety of sizes, ranging from 7 billion to 70 billion parameters. The larger models have
more capacity to learn complex patterns in language, but they are also more computationally expensive to train and
deploy.

**MosaicPretrainedTransformer (MPT) Architecture**

The MosaicPretrainedTransformer (MPT) architecture is a modified transformer architecture that is optimized for
efficient training and inference. The MPT architecture includes the following changes:

* Performance-optimized layer implementations
* Architecture changes that provide greater training stability
* Elimination of context length limits by replacing positional embeddings with Attention with Linear Biases (ALiBi)

Thanks to these modifications, MPT models can be trained with high throughput efficiency and stable convergence. MPT
models can also be served efficiently with both standard HuggingFace pipelines and NVIDIA's FasterTransformer.

**Use Cases**

The MosaicMPT models can be used for a variety of tasks, including:

* Natural language understanding (NLU): The MosaicMPT models can be used to understand the meaning of text, such as
  identifying the entities and relationships in a sentence.
* Natural language generation (NLG): The MosaicMPT models can be used to generate text, such as writing different kinds
  of creative content, like poems, code, scripts, musical pieces, email, letters, etc.
* Machine translation: The MosaicMPT models can be used to translate text from one language to another.
* Text summarization: The MosaicMPT models can be used to summarize a text document into a shorter, more concise
  version.
* Question answering: The MosaicMPT models can be used to answer questions about a text document.
* Code generation: The MosaicMPT models can be used to generate code, such as Python scripts or Java classes.

**Availability**

The MosaicMPT models are available through the Hugging Face Hub. The models are also available in the TensorFlow Hub,
the PyTorch Hub and EasyDel.

**Conclusion**

The MosaicMPT models are a powerful family of LLMs that can be used for a variety of tasks. The models are open source
and available for free, making them a valuable resource for researchers and developers.

## How to Use/Load Them in EasyDel

```python
import jax
from EasyDel.transform import mpt_from_pretrained

params, config = mpt_from_pretrained(
    'mosaicml/mpt-7b',
    device=jax.devices('cpu')[0]  # Offload on CPU
)
```

also keep that in mind that returned `config` includes `.get_partition_rules(fsdp=True)`

#### Use With JaxServer

```python
from EasyDel import JAXServer, FlaxMptForCausalLM
import jax
from EasyDel.transform import mpt_from_pretrained
from transformers import AutoTokenizer

params, config = mpt_from_pretrained(
    'mosaicml/mpt-7b',
    device=jax.devices('cpu')[0]  # Offload on CPU
)


class MPTJaxServer(JAXServer):
    ...
    # You have to Custom this one yourself as you 
    # need read JaxServer Documents inorder to learn how


server = MPTJaxServer.load_from_params(
    params=params,
    model=FlaxMptForCausalLM(
        config=config,
        dtype=jax.numpy.bfloat16,  # Im on TPUs
        param_dtype=jax.numpy.bfloat16,  # Im on TPUs
        precision=jax.lax.Precision('fastest'),
        _do_init=False,
        input_shape=(1, 1024)
    ),
    config_model=config,
    add_params_field=True,
    tokenizer=AutoTokenizer.from_pretrained('mosaicml/mpt-7b'),
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

Done 😇 this method can be used for all the MosaicMPT models