# EasyDeL

EasyDeL (Easy Deep Learning) is an open-source library designed to accelerate and optimize the training process of
machine learning models. This library is primarily focused on Jax/Flax and plans to offer easy and fine solutions to
train Flax/Jax Models on the `TPU/GPU`

## Installation

To install EasyDeL, you can use pip:
### Not on PyPi yet!
```bash
pip install easydel
```

## Available Models Are

- Llama     (Support `FSDP`, `MP`,` DP`)
- GPT-J     (Support `FSDP`, `MP`,` DP`)
- LT        (Support `FSDP`, `MP`, `DP`)
- MosaicMPT (Support `FSDP`, `MP`,` DP`)

you can also tell me the model you want in Flax/Jax version and ill try my best to build it ;)
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
