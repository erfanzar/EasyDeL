import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
from jax import numpy as jnp
from lib.python.EasyDel.reinforcement_learning.trainer.dpo_trainer import (
    DPOTrainer as DPOTrainer,
    TrainArguments
)
from lib.python.EasyDel import EasyDelState, FlaxLlamaForCausalLM, LlamaConfig

from typing import Dict, Optional

import torch

torch.cuda.is_available = lambda: False

torch.set_default_device("cpu")
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir, )
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }

    return dataset.map(split_prompt_and_responses)


def main():
    with jax.default_device(jax.devices("cpu")[0]):
        model_name_or_path = "erfanzar/LLamaStory-70M"
        conf = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            use_scan_mlp=False
        )
        arguments = TrainArguments(
            num_train_epochs=4,
            model_name="DPO_TEST",
            total_batch_size=4,
            use_wandb=False
        )

        torch_dtype = torch.float32
        # model_ref = None
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        train_dataset = get_hh("train", sanity_check=True)
        eval_dataset = get_hh("test", sanity_check=True)

        # state = EasyDelState.from_pretrained(
        #     pretrained_model_name_or_path=model_name_or_path
        # )
        # ref_state = EasyDelState.from_pretrained(
        #     pretrained_model_name_or_path=model_name_or_path,
        #
        # )
        module = FlaxLlamaForCausalLM(
            config=conf,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=True
        )
        ref_module = FlaxLlamaForCausalLM(
            config=conf,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=True
        )
        state = EasyDelState.load(
            module=module,
            apply_fn=module.__call__,
            params={"params": module.params}
        )
        ref_state = EasyDelState.load(
            module=ref_module,
            apply_fn=ref_module.__call__,
            params={"params": ref_module.params}
        )

        max_length = 512
        max_target_length = 256
        max_prompt_length = 256

        dpo_trainer = DPOTrainer(
            model_state=state,
            ref_model_state=ref_state,
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            arguments=arguments,
            max_length=max_length,
            max_target_length=max_target_length,
            max_prompt_length=max_prompt_length,
            dataset_map_arguments={
                "num_proc": os.cpu_count(),
            }
        )

        dpo_trainer.train()


if __name__ == "__main__":
    main()
