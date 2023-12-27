import EasyDel as ed
from transformers import AutoTokenizer
from jax.experimental.pjit import pjit
import jax
import jax.numpy as jnp
import fjformer
from transformers import GenerationConfig
from jax.sharding import PartitionSpec as Ps
import functools

JAXServer, JAXServerConfig, AutoEasyDelModelForCausalLM = (
    ed.JAXServer,
    ed.JAXServerConfig,
    ed.AutoEasyDelModelForCausalLM
)

with_sharding_constraint = fjformer.with_sharding_constraint


def llama2_prompt(
        message: str,
        chat_history,
        system_prompt: str
) -> str:
    do_strip = False
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return "".join(texts)


def del_prompter(
        prompt,
        history,
        system=None
):
    sys_str = f"<|system|>\n{system}</s>\n" if system is not None else ""
    histories = ""
    for user, assistance in history:
        histories += f"<|user|>\n{user}</s>\n<|assistant|>\n{assistance}</s>\n"
    return sys_str + histories + f"<|user|>\n{prompt}</s>\n<|assistant|>\n"


def main():
    pretrained_model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    model, params = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        precision=jax.lax.Precision("fastest"),
        device=jax.devices('cpu')[0]
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    params = {"params": params}
    partition_specs = fjformer.match_partition_rules(model.config.get_partition_rules(True), params)
    shard, _ = fjformer.make_shard_and_gather_fns(partition_specs, jnp.bfloat16)
    mesh = fjformer.create_mesh()
    with mesh:
        params = jax.tree_map(lambda f, p: f(p), shard, params)

    user = input(">> ")
    system = 'You are an AI be respectful and help-full.'
    prompt = llama2_prompt(
        user,
        [],
        system
    )

    @functools.partial(
        pjit,
        in_shardings=(partition_specs, Ps(), Ps()),
        out_shardings=(Ps())
    )
    def generate(parameters, input_ids, attention_mask):
        input_ids = with_sharding_constraint(input_ids, Ps(("dp", "fsdp")))
        attention_mask = with_sharding_constraint(attention_mask, Ps(("dp", "fsdp")))
        predict = model.generate(
            input_ids,
            attention_mask=attention_mask,
            params=parameters,

            generation_config=GenerationConfig(
                max_new_tokens=512,

                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,

                temperature=0.8,
                do_sample=True,
                num_beams=1,
                top_p=0.95,
                top_k=50,

            )
        ).sequences[:, input_ids.shape[1]:]
        return predict

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(prompt, padding='max_length', max_length=2048, return_tensors='jax')
    input_ids, attention_mask = tokens.input_ids, tokens.attention_mask
    with mesh:
        response = generate(params, input_ids, attention_mask)

    print(tokenizer.decode(response[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
