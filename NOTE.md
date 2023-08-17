### Note 14Aug

use `rotary_type='normal'` for llama2 or llama1 huggingface models

### Note

The llama model has a different key or params name in case you want to load the model from the hugging face
(other pre-trained llama) models you do not need to convert weight just pass the `from_pt=True` if `LlamaConfig`


#### Example

```python
from EasyDel import LlamaConfig, FlaxLlamaForCausalLM

config = LlamaConfig(from_pt=True)
model = FlaxLlamaForCausalLM.from_pretrained('<here/pretrained>', config=config)
params = model.params
partition_rules = config.get_partition_rules()
# Now partition rules are created from new key names, not Easydel type
print(partition_rules)
```

```text
 (
    ("model/embed_tokens/embedding", PS("fsdp")),
    ("self_attn/(q_proj|k_proj|v_proj)/kernel", PS("fsdp")),
    ("self_attn/o_proj/kernel", PS("fsdp")),
    ("mlp/gate_proj/kernel", PS("fsdp")),
    ("mlp/down_proj/kernel", PS("fsdp")),
    ("mlp/up_proj/kernel", PS("fsdp")),
    ("input_layernorm/kernel", PS(None)),
    ("post_attention_layernorm/kernel", PS(None)),
    ("model/norm/kernel", PS(None)),
    ("lm_head/kernel", PS("fsdp")),
    ('.*', PS(None)),
)
```


### Using Changes : 

#### In order to use OpenLlama Model set `attn_type=''` and `rotary_type='complex'`
#### In order to use OpenLlama Model set `attn_type='llama2'` and `rotary_type='llama2'` 
or `attn_type='normal'` and `rotary_type='normal'`