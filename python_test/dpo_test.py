import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
from lib.python.EasyDel.reinforcement_learning.trainer.dpo_trainer import DPOTrainer as EasyDeLDPOTrainer, \
    TrainArguments
from lib.python.EasyDel import EasyDelState

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

torch.cuda.is_available = lambda: False

torch.set_default_device("cpu")
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


@dataclass
class ScriptArguments:
    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    max_length: int = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: int = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: int = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    sanity_check: bool = field(default=True, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
                    "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
                    "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    generate_during_eval: bool = field(default=False, metadata={"help": "Generate during evaluation"})


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
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
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

        arguments = TrainArguments(
            num_train_epochs=4,
            model_name="DPO_TEST",
            total_batch_size=1,
            use_wandb=False
        )

        torch_dtype = torch.float32
        max_length = 64
        max_target_length = 256
        max_prompt_length = 256
        # model_ref = None
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = get_hh("train", sanity_check=True)
        eval_dataset = get_hh("test", sanity_check=True)

        state = EasyDelState.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path
        )
        # ref_state = EasyDelState.from_pretrained(
        #     pretrained_model_name_or_path=model_name_or_path
        # )

        dpo_trainer = EasyDeLDPOTrainer(
            state,
            None,
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            arguments=arguments,
            max_length=max_length,
            max_target_length=max_target_length,
            max_prompt_length=max_prompt_length
        )

        dpo_trainer.train()


if __name__ == "__main__":
    main()
