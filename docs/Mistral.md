## Mistral Models

Mistral LLM models. Mistral AI is a French startup that develops large language models (LLMs). Mistral's first LLM,
Mistral-7B-v0.1, was released in October 2023. It is a 7 billion parameter decoder-based LM with a number of
architectural innovations, including sliding window attention, grouped query attention, and byte-fallback BPE tokenizer.
Mistral-7B-v0.1 has been shown to achieve state-of-the-art performance on a number of NLP benchmarks, including GLUE,
SuperGLUE, and the Stanford Question Answering Dataset.

Mistral AI has not yet released a commercial version of Mistral-7B-v0.1, but it is available for free download and
evaluation. The company is also working on developing larger and more powerful LLMs, including a 100 billion parameter
model.

Mistral's LLMs have been praised for their ability to generate creative and informative text, as well as their ability
to perform a wide range of NLP tasks, such as translation, question answering, and summarization. However, some concerns
have been raised about the potential for Mistral's LLMs to be used to generate harmful content, such as instructions on
how to make bombs or how to self-harm.

Overall, Mistral AI is a promising startup in the field of LLM development. Its LLMs have the potential to be used in a
wide range of applications, such as customer service, education, and creative writing. However, it is important to be
aware of the potential risks associated with using LLMs, such as the risk of generating harmful content.

**README.md**

**Mistral LLM models**

Mistral LLM models are a set of large language models (LLMs) developed by Mistral AI, a French startup. Mistral's LLMs
are trained on massive datasets of text and code, and can be used to perform a variety of NLP tasks, including:

* Text generation
* Translation
* Question answering
* Summarization
* Code generation
* Creative writing

**Mistral-7B-v0.1** is the first LLM released by Mistral AI. It is a 7 billion parameter decoder-based LM with a number
of architectural innovations, including sliding window attention, grouped query attention, and byte-fallback BPE
tokenizer. Mistral-7B-v0.1 has been shown to achieve state-of-the-art performance on a number of NLP benchmarks,
including GLUE, SuperGLUE, and the Stanford Question Answering Dataset.

**To use a Mistral LLM model:**

1. Download the model weights from the Mistral AI website: https://mistral.ai/.
2. Install the necessary dependencies, such as the Transformers library.
3. Load the model weights into a Python script or notebook.
4. Call the model's `generate()` method to generate text, translate languages, answer questions, or perform other NLP
   tasks.

**Mistral LLM models are still under development, but they have the potential to be used in a wide range of
applications.** If you are interested in using Mistral's LLMs, please visit the Mistral AI website: https://mistral.ai/
for more information.

# Mistral Model In EasyDel

using Mistral Models are the same as all the other models in EasyDel Collection but let take a look at how can we train
or finetune a Mistral model

```python
from EasyDel.trainer import TrainArguments, CausalLanguageModelTrainer
from datasets import load_dataset
from transformers import AutoTokenizer
from jax import numpy as jnp
import flax
import EasyDel
from EasyDel import (
    AutoEasyDelModelForCausalLM,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers
)

model_huggingface_repo_id = 'mistralai/Mistral-7B-v0.1'
dataset_train = load_dataset('<TOKENIZED_MISTRAL_DATASET_AT_HUGGINGFACE>')
tokenizer = AutoTokenizer.from_pretrained(model_huggingface_repo_id, trust_remote_code=True)
model, params = AutoEasyDelModelForCausalLM.from_pretrained(model_huggingface_repo_id)
config = model.config
config.freq_max_position_embeddings = config.max_position_embeddings  # 32768
config.max_position_embeddings = 4096  # Let use context length of 4096 for training
config.c_max_position_embeddings = config.max_position_embeddings

max_sequence_length = config.max_position_embeddings

train_args = TrainArguments(
    model_class=EasyDel.FlaxMistralForCausalLM,
    configs_to_initialize_model_class={
        'config': config,
        'dtype': jnp.bfloat16,
        'param_dtype': jnp.bfloat16,
        'input_shape': (1, 1)
    },
    custom_rule=config.get_partition_rules(True),
    model_name='Test',
    num_train_epochs=2,
    learning_rate=4e-5,
    learning_rate_end=5e-6,
    optimizer=EasyDelOptimizers.ADAMW,
    scheduler=EasyDelSchedulers.WARM_UP_COSINE,
    weight_decay=0.01,
    total_batch_size=32,
    max_training_steps=None,
    do_train=True,
    do_eval=False,
    backend='tpu',
    max_sequence_length=max_sequence_length,
    gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=(1, -1, 1, 1),
    gradient_accumulation_steps=8,
    remove_ckpt_after_load=True,
    ids_to_pop_from_dataset=['token_type_ids'],
    loss_re_mat="",
    dtype=jnp.bfloat16
)

trainer = CausalLanguageModelTrainer(
    train_args,
    dataset_train['train'],
    checkpoint_path=None
)

output = trainer.train(flax.core.FrozenDict({'params': params}))
# And Here were EasyDel goes brrrrrr and start training 
```