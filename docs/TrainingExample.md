# About EasyDel.CausalLMTrainer

What Will we cover in this tutorial

- Inputs args/kwargs for CausalLMTrainer
- How to use
- Example of Training a Llama2 Model with CausalLMTrainer with jax on TPU/CPU/GPU
- Reconvert Model to HuggingFace PyTorch

## What Do we need to start

if you want to run this code in Kaggle or GoogleCloud you are fine to just run this Code to install dependencies

```shell
pip install fjformer==0.0.7 datasets gradio wandb -U -q
# EasyDel Can work with any version of JAX >= 0.4.4
pip install jax[tpu]==0.4.14 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('HF_TOKEN_Here')"
wandb login WANDB_TOKEN_HERE
apt-get update && apt-get upgrade -y -q
apt-get install golang -y -q
```

## Inputs to CausalLMTrainer

```arguments: TrainArguments```

```dataset_train: Dataset```

```dataset_eval: Dataset = None```

```finetune: bool = True```

```ckpt_path: typing.Union[str, os.PathLike] = None```

```_do_init_fns: bool = True```

### arguments

trainer takes in a TrainArguments to initialize wandb, utilities, funcs, partitions, optimizers and extra

```python
class TrainArguments(
    OrderedDict
):
    def __init__(
            self,
            model_name: str,  # Name For Model 
            num_train_epochs: int,  # Number of Total Epochs for Train
            model_id: str = None,  # ID For model [Optional for load Model From HuggingFace Repo]
            model_class=None,
            # Model class to initialize the Model or you can pass the HuggingFace Repo ID to `model_id` field
            total_batch_size: int = 32,
            # Total Batch size without counting gradient_accumulation_steps
            max_steps: Union[int, None] = None,
            # Max Number of Train Steps
            optimizer: str = 'lion',
            # Optimizer For Model
            scheduler: str = 'linear',
            # Scheduler For Model
            learning_rate: Union[int, float] = 5e-5,
            # Start OF Learning Rate
            learning_rate_end: Union[None, float] = 5e-6,
            # End OF Learning Rate in case of using scheduler='linear'
            gradient_accumulation_steps: int = 1,
            # Gradient Accumulation Steps
            weight_decay: float = 0.01,
            gradient_checkpointing: str = 'nothing_saveable',
            max_length: Union[int, None] = 4096,
            sharding_array: Union[tuple, int] = ((1, -1, 1)),
            # PJIT Partition Sharding Mesh Size for (DP,FSDP,MP) 
            is_fine_tuning: bool = True,
            do_train: bool = True,
            do_eval: bool = False,
            do_test: Union[bool, None] = False,
            backend: Union[str, None] = None,
            extra_optimizer_kwargs: dict = None,
            # Extra kwargs to be passed to optimizer for initialization
            save_steps: Union[int, None] = None,
            save_dir: str = 'easydel_ckpt',
            use_pjit_attention_force: bool = False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            fully_fsdp=True,
            use_wandb: bool = True,
            custom_rule=None,
            # Use Custom Partition Rules or use Default Ones 
            # [Do not use Custom Partitioning Rules in case that you haven't dealt with Jax Mesh]
            extra_configs=None,
            ids_to_pop_from_dataset: list = None,
            # Ids to pop from dataset in training Loop for example ids_to_pop_from_dataset=['token_type_ids'], 
            remove_ckpt_after_load: bool = False,
            # For saving Memory and Disk Space
            configs_to_init_model_class=None,
            # config to be passed to model for initialization of model
            do_last_save: bool = True,
            model_parameters=None,
            do_shard_fns: bool = True,
            track_memory: bool = True,
            loss_remat: str = '',
            loss_chunk: int = 1024,
            is_left_padded: bool = False,
            # Is model using Left padded or right padded dataset
            warmup_steps: int = 500,
            **kwargs
    ):
        ...
```

### dataset_train

Dataset for Train model which should only contain numerical information for model such as Input_ids and attention_mask

### dataset_eval

Dataset for Evaluate the model which should only contain numerical information for model such as Input_ids and
attention_mask

### finetune

fine-tune is a boolean field which stand for should EasyDel initialize the params for pre-training the model, or you
will pass the
model parameters later in training section

### ckpt_path

path to Read Params from Checkpoint in case of passing string you are reading from file
and in case of passing none you can pass params manually to Tariner

## FineTune A LLama Model

here's a script

here in this script we will see an example for how to finetune models in EasyDel

```python
from EasyDel import TrainArguments, CausalLMTrainer, AutoEasyDelModelForCausalLM
import jax
import flax
from datasets import load_dataset

model_id = 'meta-llama/Llama-2-13b-hf'
dataset_train = load_dataset('REPO_ID_PATH_TO_DATASET')
# dataset should only contain numerical information for Model such as input_id, attention_mask , ...
model, params = AutoEasyDelModelForCausalLM.from_pretrained(
    model_id,
    dtype=jax.numpy.bfloat16,
    param_dtype=jax.numpy.bfloat16,
    precision=jax.lax.Precision('fastest'),
    device=jax.devices('cpu')[0],  # Load JAX Model and initialize or load Parameters on CPU 
    # The Rest of kwargs here will be passed to AutoModelForCausalLM huggingface such as this device_map
    device_map='auto'
)
config = model.config

# this part of code is only for making model faster and more optimized 
config.freq_max_position_embeddings = config.max_position_embeddings
config.max_position_embeddings = 4096
config.c_max_position_embeddings = config.max_position_embeddings
config.use_pjit_attention_force = False  # disable pjit attention force is recommended in case of using MP = 1 in sharding Mesh

max_length = config.max_position_embeddings

configs_to_init_model_class = {
    'config': config,
    'dtype': jax.numpy.bfloat16,
    'param_dtype': jax.numpy.bfloat16,
    'input_shape': (1, 1)
}

train_args = TrainArguments(
    model_class=type(model),
    configs_to_init_model_class=configs_to_init_model_class,
    custom_rule=config.get_partition_rules(True),
    model_name='EasyDelLLama2',
    num_train_epochs=2,
    learning_rate=2e-4,
    learning_rate_end=5e-6,
    optimizer='adamw',
    scheduler='linear',
    weight_decay=0.02,
    total_batch_size=64,
    max_steps=None,
    do_train=True,
    do_eval=False,
    backend='tpu',
    max_length=max_length,
    gradient_checkpointing='nothing_saveable',
    sharding_array=((1, -1, 1)),
    use_pjit_attention_force=False,
    gradient_accumulation_steps=1,
    remove_ckpt_after_load=True,
    ids_to_pop_from_dataset=['token_type_ids'],
    loss_remat='',
    dtype=jax.numpy.bfloat16
)

trainer = CausalLMTrainer(
    train_args,
    dataset_train,
    ckpt_path=None
)

output = trainer.train(flax.core.FrozenDict({'params': params}))

saved_model_location = f"{str(train_args.get_path())}/{output.last_save_file_name}"

print("Hey im Here in case you want to load me :", saved_model_location)

### Let Convert Model TO HF/PyTorch

from EasyDel.transform import llama_easydel_to_hf

config.rope_theta = 10000
config.attention_bias = False
model = llama_easydel_to_hf(saved_model_location, config=config)

# Here's your Huggingface Torch Llama
model = model.half()

model.push_to_hub("REPO_ID_TO_PUSH")
```

