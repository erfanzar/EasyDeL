### Note 21Aug

You no longer need to use anything like rotary_type now complex works with all of the llama models so that options has
been removed

but you have to convert model with easydel like

```python
from src.EasyDel import llama_from_pretrained, llama_convert_hf_to_flax

params, config = llama_from_pretrained(
    '<Model-id>'
)
# or
params = llama_convert_hf_to_flax(
    pytorch_model.state_dict(), num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
    hidden_size=hidden_size
)
```

### Note

The llama model has a different key or params name in case you want to load the model from the hugging face
(other pre-trained llama) models you do not need to convert weight just pass the `from_pt=True` if `LlamaConfig`

#### Example

```python
from src.EasyDel import LlamaConfig, FlaxLlamaForCausalLM

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