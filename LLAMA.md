# Usage of Llama 2 models With EasyDel

### Note

#### All the Llama Models supported, and they will perform same as what they do in pytorch

## Example of using Llama-7B Chat or any other model With Using JAXServer

```shell
python -m examples.serving.causal-lm.llama-2-chat \
  --repo_id='meta-llama/Llama-2-7b-chat-hf' --max_length=4096 \
  --max_new_tokens=2048 --max_stream_tokens=32 --temperature=0.6 \
  --top_p=0.95 --top_k=50 \
  --dtype='fp16' --use_prefix_tokenizer

```

you can use all of the llama models not just 'meta-llama/Llama-2-7b-chat-hf'

'fp16' Or 'fp32' , 'bf16' are supported dtype

make sure to use --use_prefix_tokenizer

## Example of using Llama-7B Chat Without Using JAXServer

here's a simple function to prompt the LlamaChat Models if you don't want to use them with `JAXServe` and
want to use them manually but in case of using and serving `JAXServer` is recommended

```python
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer "
"as helpfully as possible, while being safe.  Your answers should not"
" include any harmful, unethical, racist, sexist, toxic, dangerous, or "
"illegal content. Please ensure that your responses are socially unbiased "
"and positive in nature.\nIf a question does not make any sense, or is not "
"factually coherent, explain why instead of answering something not correct. If "
"you don't know the answer to a question, please don't share false information."


def get_prompt(message: str, chat_history,
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)
```

### Loading Model

```python
import jax
from EasyDel import get_mesh, FlaxLlamaForCausalLM
from fjutils import with_sharding_constraint, match_partition_rules, make_shard_and_gather_fns
from EasyDel.transform import llama_from_pretrained
from jax.experimental import pjit
from jax.sharding import PartitionSpec as PS
from transformers import AutoTokenizer, GenerationConfig
from functools import partial

# Let Use this model since runs fast and light, but you can use all the available llama models like llama, llama2, xgen... 
model_id = 'meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True  # Optional
)
params, model_config = llama_from_pretrained(
    model_id, device=jax.devices('cpu')[0]  # Cpu Offload
)

# Creating Model
dtype = 'float16'  # available dtypes are float16,bfloat16,float32,float64,float128
model = FlaxLlamaForCausalLM(
    config=model_config,
    dtype=dtype,
    param_dtype=dtype,
    precision=jax.lax.Precision('fastest'),
    _do_init=False
)

# Sharding Model Across all of the available Devices (TPU/GPU/CPU)s

model_ps = match_partition_rules(model_config.get_partition_rules(True), params)
shard_fns, gather_fns = make_shard_and_gather_fns(model_ps)
mesh = get_mesh()


@partial(
    pjit,
    in_shardings=({'params': model_ps}, PS(), PS()),
    out_shardings=PS()
)
def forward_generate(params, batch, temperature):
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    output = model.generate(
        batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        generation_config=GenerationConfig(
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
            top_k=50,
            top_p=0.95
        )
    ).sequences[:, batch['input_ids'].shape[1]:]
    return output

```

### Done

you have loaded your llama model in easydel now you can serve it or finetune it with your own custom data😇

#### Simple Example of Generating without JAXServer

```python
prompt = get_prompt(
    message='Why jax is awsome !',
    chat_history=[],  # you can use chat history too i pass it empty cause this is first request to model
    system_prompt=DEFAULT_SYSTEM_PROMPT
)
tokenizer_prompt = tokenizer(prompt, return_tensors='jax')
with mesh:
    batch = {
        'input_ids': jnp.asarray(tokenizer_prompt.input_ids).astype('int32'),
        'attention_mask': jnp.asarray(tokenizer_prompt.attention_mask).astype('int32')
    }
    out = forward_generate({"params": params}, batch, 0.8)

print(tokenizer.decode(
    out,
    skip_special_tokens=True
))
```

