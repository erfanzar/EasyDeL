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

### Open an issue or a request to update this section 