from EasyDel import AutoEasyDelModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import jax
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
this in an example of how to update and resize model embedding size here we use pytorch to do that but you can use 
EasyDeL and jax to do the same thing but it will be a little bit harder than pytorch's so let just use pytorch for this
purpose and them load model in EasyDeL
"""


def add_special_tokens(model_name: str, tokenizer: AutoTokenizer, new_tokens):
    # Add new special tokens to the tokenizer
    num_new_tokens = len(new_tokens)
    tokenizer.add_tokens(new_tokens)

    # Determine the new vocabulary size
    new_vocab_size = len(tokenizer.get_vocab())

    # Update the model's embedding layer and lm_head layer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(new_vocab_size)

    # Update the position embeddings (optional)
    if hasattr(model, "position_embeddings"):
        num_positions = model.position_embeddings.weight.size(0)
        new_positions = torch.arange(num_positions + num_new_tokens, device=model.device)
        model.position_embeddings.weight = torch.nn.Parameter(
            torch.cat((model.position_embeddings.weight,
                       torch.zeros(num_new_tokens, model.position_embeddings.embedding_dim)), dim=0)
        )

    return model, tokenizer


def main():
    model_name = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_tokens = ["<|im_start|>", "<|im_end|>"]
    model, tokenizer = add_special_tokens(model_name, tokenizer, new_tokens)
    model = model.half()
    model.push_to_hub("REPO_ID")
    tokenizer.push_to_hub("REPO_ID")


if __name__ == "__main__":
    main()
