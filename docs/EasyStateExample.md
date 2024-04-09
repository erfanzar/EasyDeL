## EasyDelState

EasyDelState is a cool feature in easydel and have a lot of options like
storing `Model Parameters`, _Optimizer State, Model Config, Model Type, Optimizer and Scheduler Configs_

Let see and examples of using EasyDelState

### Fine-tuning

Fine-tuning from a previous State or a new state

```python
from EasyDel import (
    AutoEasyDelConfig,
    EasyDelState
)
from transformers import AutoTokenizer
from jax import numpy as jnp, lax
import jax

huggingface_model_repo_id = "REPO_ID"
checkpoint_name = "CKPT_NAME"

state = EasyDelState.from_pretrained(
    pretrained_model_name_or_path=huggingface_model_repo_id,
    filename=checkpoint_name,
    optimizer="adamw",
    scheduler="none",
    tx_init=None,
    device=jax.devices('cpu')[0],  # Offload Device
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=lax.Precision("fastest"),
    sharding_axis_dims=(1, -1, 1, 1),
    sharding_axis_names=("dp", "fsdp", "tp", "sp"),
    query_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    key_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    value_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    bias_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None),
    attention_partition_spec=jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    shard_attention_computation=False,
    input_shape=(1, 1),
    backend=None,
    init_optimizer_state=False,
    free_optimizer_state=True,
    verbose=True,
    state_shard_fns=None,
)

config = AutoEasyDelConfig.from_pretrained(
    huggingface_model_repo_id
)

tokenizer = AutoTokenizer.from_pretrained(
    huggingface_model_repo_id,
    trust_remote_code=True
)

max_length = config.max_position_embeddings

configs_to_initialize_model_class = {
    'config': config,
    'dtype': jnp.bfloat16,
    'param_dtype': jnp.bfloat16,
    'input_shape': (8, 8)
}
```

`EasyDelState` also has `.load_state()` and `.save_state()` with some other usable options like `.free_opt_state()`
which
free optimizer state or `.shard_params()` which shard parameters you can read docs in order to find out more about these
options.

### Converting to Huggingface and Pytorch

Let see how you can convert a EasyDelMistral Model to Huggingface Pytorch Mistral Model from a trained State

```python

from transformers import MistralForCausalLM
from EasyDel import (
    AutoEasyDelConfig,
    EasyDelState,
    easystate_to_huggingface_model
)
import jax

huggingface_model_repo_id = "REPO_ID"

config = AutoEasyDelConfig.from_pretrained(
    huggingface_model_repo_id
)
with jax.default_device(jax.devices("cpu")[0]):
    model = easystate_to_huggingface_model(
        state=EasyDelState.load_state(
            "PATH_TO_CKPT",
            input_shape=(8, 2048)
        ),  # You can Pass EasyDelState here
        base_huggingface_module=MistralForCausalLM,
        config=config,
    )

model = model.half()  # it's a huggingface model now
```

### Other Use Cases

`EasyDelState` have a general use you can use it everywhere in easydel for example for a stand-alone model
, serve, fine-tuning and many other features, it's up to you to test how creative you are 😇.
