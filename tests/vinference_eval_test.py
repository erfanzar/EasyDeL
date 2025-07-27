import jax
import torch
import transformers
from datasets import load_dataset
from jax import numpy as jnp
from tqdm import tqdm

import easydel as ed


def calc_accuracy(actuals, preds):
    total_correct = 0
    total_examples = len(actuals)
    for actual, pred in zip(actuals, preds, strict=False):
        pred_letter = "A"
        if "A" in pred:
            pred_letter = "A"
        if "B" in pred:
            pred_letter = "B"
        if "C" in pred:
            pred_letter = "C"
        if "D" in pred:
            pred_letter = "D"
        if actual == pred_letter:
            total_correct += 1
    acc_score = total_correct / total_examples
    return acc_score


FORCE_SP = jax.device_count() > 4  # False


def main():
    if jax.device_count() > 4 and not FORCE_SP:
        sharding_axis_dims = (1, 1, 2, -1)
    else:
        sharding_axis_dims = (1, 1, 1, 1, -1)

    max_length = 4096
    if jax.default_backend() == "gpu":
        pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    else:
        pretrained_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

    partition_axis = ed.PartitionAxis()

    dtype = jnp.bfloat16

    print("LOADING MODEL ... ")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            attn_dtype=dtype,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        platform=ed.EasyDeLPlatforms.JAX,
        param_dtype=dtype,
        dtype=dtype,
        torch_dtype=torch.float16,
        partition_axis=partition_axis,
        precision=jax.lax.Precision("fastest"),
    )
    print("MODEL LOADED")
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("TOKENIZER LOADED")
    model.eval()
    print("CREATING vInference")

    inference = ed.vInference(
        model=model,
        processor_class=tokenizer,
        generation_config=ed.vInferenceConfig(
            max_new_tokens=1024,
            sampling_params=ed.SamplingParams(
                max_tokens=1024,
                temperature=0.8,
                top_p=0.95,
                top_k=10,
            ),
            eos_token_id=model.generation_config.eos_token_id,
            streaming_chunks=32,
        ),
    )

    print(model.model_task)
    print(model.model_type)
    print("Compiling")
    inference.precompile(
        ed.vInferencePreCompileConfig(
            batch_size=1,
            prefill_length=inference.model_prefill_length,
        )
    )

    print("Done Compiling")
    print("Evaluating on MMLU Lite")
    prompts = []
    pred_list = []
    actual_list = []
    data = load_dataset("CohereForAI/Global-MMLU-Lite", "en", split="test")
    for item in tqdm(data, total=len(data)):
        question = item["question"]
        option_a = item["option_a"]
        option_b = item["option_b"]
        option_c = item["option_c"]
        option_d = item["option_d"]
        actual_list.append(item["answer"])
        prompt = (
            f"Answer the following question by writing the right answer letter which can be A,B,C or D. Write only "
            f"the correct answer letter in your response. \nQuestion : {question}\nA. {option_a}. \nB. {option_b}. "
            f"\nC. {option_c}. \nD. {option_d}"
        )
        prompts.append(prompt)
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="jax",
            return_dict=True,
            max_length=inference.model_prefill_length,
            padding="max_length",
            add_generation_prompt=True,
        )

        pad_seq = inference.model_prefill_length
        for response in inference.generate(**ids):
            next_slice = slice(
                pad_seq,
                pad_seq + inference.generation_config.streaming_chunks,
            )
            pad_seq += inference.generation_config.streaming_chunks
            output = tokenizer.decode(
                response.sequences[0][next_slice],
                skip_special_tokens=True,
            )
            pred_list.append(output)
    for prompt, pred in zip(prompts, pred_list, strict=False):
        print("--------------------------------------")
        print(f"Prompt: {prompt}\nPrediction : {pred}")
    print("---------- Evaluation Score -----------------")
    acc_score = calc_accuracy(actual_list, pred_list)
    print(f"accuracy score : {acc_score}")


if __name__ == "__main__":
    main()
