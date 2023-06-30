# EasyDeL

EasyDeL (Easy Deep Learning) is an open-source library designed to accelerate and optimize the training process of
machine learning models. This library is primarily focused on Jax/Flax and plans to offer easy and fine solutions to
train Flax/Jax Models on the `TPU/GPU`

## Installation

To install EasyDeL, you can use pip:
### Availalbe on PyPi
```bash
pip install easydel
```

## Available Models Are

- Llama     (Support `FSDP`, `MP`,` DP`)
- GPT-J     (Support `FSDP`, `MP`,` DP`)
- LT        (Support `FSDP`, `MP`, `DP`)
- MosaicMPT (Support `FSDP`, `MP`,` DP`)
- GPTNeoX   (Support `FSDP`, `MP`, `DP`)
- Falcon    (Support `FSDP`, `MP`, `DP`)
- Palm      (Support `FSDP`, `MP`, `DP`)
- T5        (Support `FSDP`, `MP`, `DP`)

you can also tell me the model you want in Flax/Jax version and ill try my best to build it ;)

## FineTuning

with using EasyDel FineTuning LLM (CausalLanguageModels) are easy as much as possible with using Jax and Flax
and having the benefit of TPUs for the best speed here's a simple code to use in order to finetune your own MPT / LLama 
or any other models supported by EasyDel

#### Step One

download converted model weights in order to finetune or convert the weights of the model you want to use
weight_convertor in the library example

```python
import jax
from EasyDel.weight_convertor.mpt import convert_pt_to_flax_7b
from fjutils.utils import save_ckpt

number_of_layers = 32  # its 32 hidden layers for Mpt 7B
device = jax.devices('cpu')[0]  # offload on CPU

pytorch_model_state_dict = None  # StateDict of the model should be this one
flax_params = convert_pt_to_flax_7b(pytorch_model_state_dict, number_of_layers, device)
save_ckpt(flax_params, 'flax_param_easydel')
```

#### Step Two

now it's time to finetune or model

```python
import jax.numpy
from EasyDel import TrainArguments, finetuner
from datasets import load_dataset

max_length = 4096
train_args = TrainArguments(
    model_id='erfanzar/FlaxMpt-7B',
    # right now you should use model supported with remote code from huggingface all model are supported and uploaded
    model_name='my_first_model_to_train_using_easydel',
    num_train_epochs=3,
    learning_rate=1e-5,
    learning_rate_end=1e-6,
    optimizer='lion',  # 'adamw', 'lion', 'adafactor' are supported
    scheduler='linear',  # 'linear' or 'cosine' or 'none'
    weight_decay=0.01,
    total_batch_size=16,
    max_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend='tpu',  # default backed is set to cpu so you must define you want to use tpu cpu or gpu
    max_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing='nothing_saveable',
    sharding_array=(1, -1, 1)  # the way to shard model across gpu,cpu or TPUs with using sharding array (1, -1, 1)
    # everything training will be in fully fsdp automatic and share data between devices
)
dataset = load_dataset('TRAIN_DATASET')
dataset_train = dataset['train']
dataset_eval = dataset['eval']
model_and_extra_outputs = finetuner(
    dataset_train=dataset_train,
    dataset_eval=dataset_eval,
    training_arguments=train_args,
    ckpt_path='flax_param_easydel',
    use_wandb=True,
    fully_fsdp=True,
    extra_configs={
        'max_seq_len': max_length
        # this one is working for mpt config models check source for other models or see config.json file
    },
    dtype=jax.numpy.bfloat16,
    param_dtype=jax.numpy.bfloat16,
    use_pjit_attention_force=False,
)
print(f'Hey ! , here\'s where your model saved {model_and_extra_outputs.last_save_file_name}')

```

you can then convert it to pytorch for better use I don't recommend jax/flax for hosting models since 
pytorch is better option for gpus

## Usage

To use EasyDeL in your project, you will need to import the library in your Python script and use its various functions
and classes. Here is an example of how to import EasyDeL and use its Model class:

```python

from EasyDel import FlaxLlamaForCausalLM, LlamaConfig

config = LlamaConfig.from_pretrained('owner/repo_id')
# in case building from config
model = FlaxLlamaForCausalLM(
    config=config,
    _do_init=True,  # To init Params (doing this manual is a better idea)
)

# in case of loading
```

or simply just load a params

```python
# fjutils is an inside library for EasyDeL
from fjutils.utils import read_ckpt

params = read_ckpt('path_to_ckpt_(ostFormat,EasyDeLFormat,EasyLMFormat)',
                   shard_fns=None  # shard fns in case to use with pjit to shard model
                   )

```

or loading with train state just like

```python
from fjutils import StreamingCheckpointer

ckpt_s = StreamingCheckpointer(
    StreamingCheckpointer.get_default_config(),
    'ckpt_dir'
)
train_state, params = ckpt_s.load_checkpoint(
    'params::path_to_ckpt_(ostFormat,EasyDeLFormat,EasyLMFormat)'
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

If you have any questions or comments about EasyDeL, you can reach out to the team behind EasyDeL
