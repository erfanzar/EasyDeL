# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load Llama 3.2 3B and generate at 300+ tok/s via ``spx.run``.

Same ``Llama3.forward`` handles training AND generation:

* ``spx.run(model, inputs=ids, mode='forward')`` → logits (prefill).
* ``spx.run(model, inputs=(tok, kc, vc, pos), mode='forward')``
  → ``(logits, kc, vc)`` (decode step with KV cache).

The decode loop calls ``spx.run`` per token. SpecTrax caches the
jit + placed state so steady-state dispatch is ~3ms/tok (322 tok/s
on 4x v5p, Llama 3.2 3B, bs=1).

Requires ``transformers``, ``safetensors``, ``HF_TOKEN`` env var.

Usage::

    HF_TOKEN=hf_xxx python -m examples.02_implementation_guide.06_llama_generation
"""

from __future__ import annotations

import glob
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

if os.path.isdir("/dev/shm/mdl"):
    sys.path.insert(0, "/dev/shm/mdl")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config

MODEL_NAME = "meta-llama/Llama-3.2-3B"
CACHE_DIR = "/dev/shm/mdl/hf_cache"
MAX_NEW = 100


def load_hf_weights(model, model_name, cache_dir):
    """Load HF safetensors into a spectrax Llama3 model."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.bin", "*.bin.index.json", "original/*"],
    )
    w = {}
    for sf in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(sf, framework="jax") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k)

    _, state = spx.export(model)
    p = dict(state.raw().get("parameters", {}))
    emb = w["model.embed_tokens.weight"]
    p["embed.embed.weight"] = emb
    p["head.proj.weight"] = w.get("lm_head.weight", emb).T
    p["head.norm.weight"] = w["model.norm.weight"]
    for i in range(cfg.num_hidden_layers):
        s, b = f"model.layers.{i}", f"blocks.{i}"
        p[f"{b}.norm1.weight"] = w[f"{s}.input_layernorm.weight"]
        p[f"{b}.norm2.weight"] = w[f"{s}.post_attention_layernorm.weight"]
        for proj in ("q", "k", "v", "o"):
            p[f"{b}.{proj}.weight"] = w[f"{s}.self_attn.{proj}_proj.weight"].T
        for proj in ("gate", "up", "down"):
            p[f"{b}.{proj}.weight"] = w[f"{s}.mlp.{proj}_proj.weight"].T
    return type(state)({"parameters": p})


def main():
    """Load Llama 3.2 3B, generate with spx.run per token."""
    from transformers import AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    cfg = Llama3Config(
        vocab=128256,
        d_model=3072,
        n_heads=24,
        n_kv_heads=8,
        ffn=8192,
        n_layers=28,
        rope_theta=500_000.0,
        dtype=jnp.bfloat16,
    )

    mesh = spx.create_mesh(axis_dims=(1, 1, 1, 1, -1, 1))

    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = Llama3(cfg, rngs=spx.Rngs(0))
        print("Loading weights...")
        state = load_hf_weights(model, MODEL_NAME, CACHE_DIR)
        gdef, _ = spx.export(model)
        model = spx.bind(gdef, state)
        print(f"  L={cfg.n_layers} d={cfg.d_model} TP={mesh.shape['tp']}")

        n_layers = cfg.n_layers
        n_kv = cfg.n_kv_heads
        head_dim = cfg.head_dim
        max_seq = 512
        eos = tokenizer.eos_token_id or -1

        tok = jnp.ones((1, 1), dtype=jnp.int32)
        kc = jnp.zeros((n_layers, 1, max_seq, n_kv, head_dim), dtype=jnp.bfloat16)
        vc = jnp.zeros((n_layers, 1, max_seq, n_kv, head_dim), dtype=jnp.bfloat16)

        print("\nWarming up jit cache...")
        for i in range(5):
            out = spx.run(model, inputs=(tok, kc, vc, i), mesh=mesh, mode="forward")
            jax.block_until_ready(out[0])
        print("  warm.")

        prompts = [
            "The meaning of life is",
            "In a galaxy far far away,",
            "def fibonacci(n):",
        ]

        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            prompt_ids = jnp.array(ids)[None, :]
            prompt_len = len(ids)

            logits = spx.run(model, inputs=prompt_ids, mesh=mesh, mode="forward")
            jax.block_until_ready(logits)
            first_tok = jnp.argmax(logits[:, -1, :], axis=-1)

            kc = jnp.zeros((n_layers, 1, max_seq, n_kv, head_dim), dtype=jnp.bfloat16)
            vc = jnp.zeros((n_layers, 1, max_seq, n_kv, head_dim), dtype=jnp.bfloat16)

            generated = [int(first_tok[0])]
            tok = first_tok[:, None]

            t0 = time.perf_counter()
            for i in range(MAX_NEW - 1):
                pos = prompt_len + i
                logits, kc, vc = spx.run(
                    model,
                    inputs=(tok, kc, vc, pos),
                    mesh=mesh,
                    mode="forward",
                )
                nt = jnp.argmax(logits[:, 0, :], axis=-1)
                jax.block_until_ready(nt)
                generated.append(int(nt[0]))
                if int(nt[0]) == eos:
                    break
                tok = nt[:, None]
            elapsed = time.perf_counter() - t0

            n = len(generated)
            tps = n / elapsed if elapsed > 0 else 0
            text = tokenizer.decode(ids + generated, skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"  {n} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
            print(f"  Output: {text[:400]}")


if __name__ == "__main__":
    main()
