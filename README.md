# EasyDeL

EasyDeL (Easy Deep Learning) is an open-source library designed to accelerate and optimize the training process of
machine learning models. This library is primarily focused on Jax/Flax and plans to offer easy and fine solutions to
train Flax/Jax Models on the `TPU/GPU` both for Serving and Training

#### Note this Library needs golang to run (for some tracking stuff on TPU/GPU/CPU)

#### Ubuntu GO installation

```shell
sudo apt-get update && apt-get upgrade -y
sudo apt-get install golang -y 
```

#### Manjaro/Arch GO installation

```shell
sudo pacman -Syyuu go
```

_you can install other version too but easydel required at least version of 0.4.10_

```shell
!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
```

on GPUs be like

```shell
pip install --upgrade pip
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

```shell
pip install --upgrade pip
# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Documentation

Tadadad (Magic Sound) 💫 finally documents are ready at [EasyDel/Docs](https://erfanzar.github.io/EasyDeL/docs)

## Installation

### Available on PyPi

To install EasyDeL, you can use pip:

```bash
pip install easydel
```

## Tutorials

_Tutorials on how to use and train or serve your models with EasyDel is available at examples dir_

1. [_Serving_](https://github.com/erfanzar/EasyDeL/tree/main/examples/serving)

2. [_Train_](https://github.com/erfanzar/EasyDeL/tree/main/examples/training/causal-lm)

3. [_Use Llama 2 Models_](https://github.com/erfanzar/EasyDeL/blob/main/LLAMA.md)

## Available Models Are

- **_Llama_**     (Support `FSDP`, `MP`,` DP`)(_Supports gradient checkpointing_)
- **_Llama2_**    (Support `FSDP`, `MP`,` DP`)(_Supports gradient checkpointing_)
- **_GPT-J_**     (Support `FSDP`, `MP`,` DP`)(_Supports gradient checkpointing_)
- **_LT_**        (Support `FSDP`, `MP`, `DP`)(_Supports gradient checkpointing_)
- **_MosaicMPT_** (Support `FSDP`, `MP`,` DP`)(_Supports gradient checkpointing_)
- **_GPTNeoX_**   (Support `FSDP`, `MP`, `DP`)(_Supports gradient checkpointing_)
- **_Falcon_**    (Support `FSDP`, `MP`, `DP`)(_Supports gradient checkpointing_)
- **_Palm_**      (Support `FSDP`, `MP`, `DP`)(_Supports gradient checkpointing_)
- **_T5_**        (Support `FSDP`, `MP`, `DP`)(_Supports gradient checkpointing_)
- **_OPT_**       (Support `FSDP`, `MP`, `DP`)(_Supports gradient checkpointing_)

[//]: # (- **_XGen_**      _Soon_)

- _LLama GPT-J MosaicMPT Falcon supports Flash Attention_

you can also tell me the model you want in Flax/Jax version and ill try my best to build it ;)

## Serving

you can read docs or examples to see how `JAXServer` works but let me show you how you can simply host and serve a
LLama2
chat model (70B model is supported too)

```shell
python -m examples.serving.causal-lm.llama-2-chat \
  --repo_id='meta-llama/Llama.md-2-7b-chat-hf' --max_length=4096 \
  --max_new_tokens=2048 --max_stream_tokens=32 --temperature=0.6 \
  --top_p=0.95 --top_k=50 \
  --dtype='fp16' --use_prefix_tokenizer

```

you can use all of the llama models not just 'meta-llama/Llama-2-7b-chat-hf'

'fp16' Or 'fp32' , 'bf16' are supported dtype

make sure to use --use_prefix_tokenizer

and you will get links or api to use model from gradio app chat/instruct or FastAPI apis

## RLHF(Reinforcement Learning From Human Feedback)

`RLHF` or Reinforcement Learning From Human Feedback is Available At the moment, but it's still
under heavy development , because i don't have enough experience with Reinforcement Learning at the moment so its still
in beta version but it's works and ill soon release a Tutorial For that

## FineTuning

with using EasyDel FineTuning LLM (CausalLanguageModels) are easy as much as possible with using Jax and Flax
and having the benefit of TPUs for the best speed here's a simple code to use in order to finetune your own MPT / LLama
or any other models supported by EasyDel

#### Step One

download converted model weights in order to finetune or convert the weights of the model you want to use
transform in the library example

```python
import jax
from EasyDel.transform import mpt_convert_pt_to_flax_7b
from fjutils.utils import save_ckpt

number_of_layers = 32  # its 32 hidden layers for Mpt 7B
device = jax.devices('cpu')[0]  # offload on CPU

pytorch_model_state_dict = None  # StateDict of the model should be this one
flax_params = mpt_convert_pt_to_flax_7b(pytorch_model_state_dict, number_of_layers, device)
save_ckpt(flax_params, 'flax_param_easydel')
```

#### Step Two

now it's time to finetune or model

```python
import jax.numpy
from EasyDel import TrainArguments, CausalLMTrainer
from datasets import load_dataset
from EasyDel.configs import configs

max_length = 4096
conf = configs.mpt_configs['7b']
conf['max_sequence_length'] = max_length
conf['max_seq_len'] = max_length

train_args = TrainArguments(
    model_id='erfanzar/FlaxMpt-7B',
    # right now you should use model supported with remote code from huggingface all model are supported and uploaded
    model_name='my_first_model_to_train_using_easydel',
    num_train_epochs=3,
    learning_rate=1e-5,
    learning_rate_end=1e-6,
    optimizer='lion',  # 'adamw', 'lion', 'adafactor','warm_up_cosine' are supported
    scheduler='linear',  # 'linear' or 'cosine' or 'none'
    weight_decay=0.01,
    total_batch_size=16,
    max_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend='tpu',  # default backed is set to cpu so you must define you want to use tpu cpu or gpu
    max_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing='nothing_saveable',
    sharding_array=(1, -1, 1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1)
    # everything training will be in fully fsdp automatic and share data between devices
    use_pjit_attention_force=False,
    extra_configs=conf,
    remove_ckpt_after_load=True,
    gradient_accumulation_steps=8
)
dataset = load_dataset('TRAIN_DATASET')
dataset_train = dataset['train']
dataset_eval = dataset['eval']

trainer = CausalLMTrainer(train_args,
                          dataset_train,
                          ckpt_path='flax_param_easydel')
model_and_extra_outputs = trainer.train()
print(f'Hey ! , here\'s where your model saved {model_and_extra_outputs.last_save_file_name}')


```

you can then convert it to pytorch for better use I don't recommend jax/flax for hosting models since
pytorch is better option for gpus

## Usage

To use EasyDeL in your project, you will need to import the library in your Python script and use its various functions
and classes. Here is an example of how to import EasyDeL and use its Model class:

```python
from EasyDel import JAXServer, FlaxLlamaForCausalLM, LlamaConfig
from EasyDel.transform import llama_from_pretrained
from transformers import AutoTokenizer

import jax

model_id = 'meta-llama/Llama.md-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

params, config = llama_from_pretrained(model_id, jax.devices('cpu')[0])
model = FlaxLlamaForCausalLM(
    config,
    dtype='float16',
    param_dtype='float16',
    _do_init=False,
)
server = JAXServer.load_from_params(
    model=model,
    config_model=config,
    tokenizer=tokenizer,
    params=model.params,
    add_params_field=True
)

predictions = server.process(
    'String To The Model'
)
```

## Contributing

EasyDeL is an open-source project, and contributions are welcome. If you would like to contribute to EasyDeL, please
fork the repository, make your changes, and submit a pull request. The team behind EasyDeL will review your changes and
merge them if they are suitable.

## License

EasyDeL is released under the Apache v2 license. Please see the LICENSE file in the root directory of this project for
more information.

## Contact

If you have any questions or comments about EasyDeL, you can reach out to me
