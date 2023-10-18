import torch
import struct
import numpy as np
from transformers import LlamaForCausalLM, LlamaConfig
from tqdm.auto import tqdm


def serialize(file, tensor, dtype: torch.dtype = torch.float32):
    d = tensor.detach().cpu().view(-1).to(dtype).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)


def exporter(model: LlamaForCausalLM, filepath: str, dtype: torch.dtype = torch.float32):
    pbar = tqdm([], total=15)
    byte_writer = open(filepath, 'wb')

    config: LlamaConfig = model.config

    shared_classifier = torch.equal(
        model.model.embed_tokens.weight, model.lm_head.weight)

    if not shared_classifier:
        config.vocab_size = -config.vocab_size
    num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads

    header = struct.pack(
        'iiiiiii',
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        num_key_value_heads,
        config.vocab_size,
        config.max_position_embeddings
    )
    byte_writer.write(header)
    pbar.update(1)

    serialize(byte_writer, model.model.embed_tokens.weight, dtype=dtype)
    pbar.update(1)

    for layer in model.model.layers:
        serialize(byte_writer, layer.input_layernorm.weight, dtype=dtype)
    pbar.update(1)
    for layer in model.model.layers:
        serialize(byte_writer, layer.self_attn.q_proj.weight, dtype=dtype)
    pbar.update(1)
    for layer in model.model.layers:
        serialize(byte_writer, layer.self_attn.k_proj.weight, dtype=dtype)
    pbar.update(1)
    for layer in model.model.layers:
        serialize(byte_writer, layer.self_attn.v_proj.weight, dtype=dtype)
    pbar.update(1)
    for layer in model.model.layers:
        serialize(byte_writer, layer.self_attn.o_proj.weight, dtype=dtype)
    pbar.update(1)
    # ffn weights
    for layer in model.model.layers:
        serialize(byte_writer, layer.post_attention_layernorm.weight, dtype=dtype)
    pbar.update(1)
    for layer in model.model.layers:
        serialize(byte_writer, layer.mlp.gate_proj.weight, dtype=dtype)
    pbar.update(1)
    for layer in model.model.layers:
        serialize(byte_writer, layer.mlp.down_proj.weight, dtype=dtype)
    pbar.update(1)
    for layer in model.model.layers:
        serialize(byte_writer, layer.mlp.up_proj.weight, dtype=dtype)
    pbar.update(1)

    serialize(byte_writer, model.model.norm.weight)
    pbar.update(1)
    serialize(
        byte_writer, model.model.layers[0].self_attn.rotary_emb.cos_cached[:config.max_position_embeddings],
        dtype=dtype)
    pbar.update(1)
    serialize(
        byte_writer, model.model.layers[0].self_attn.rotary_emb.sin_cached[:config.max_position_embeddings],
        dtype=dtype)
    pbar.update(1)
    if not shared_classifier:
        serialize(byte_writer, model.lm_head.weight, dtype=dtype)
    pbar.update(1)
    byte_writer.close()
