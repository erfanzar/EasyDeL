# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Llama 3.2 3B generation via ``@sxjit`` — true MPMD, jaxpr-split.

The ``@sxjit`` decorator traces the forward function, finds
``sxstage_iter`` markers, splits the jaxpr into per-rank
sub-programs, and compiles each rank independently. At runtime each
rank fires its **own** XLA executable — true MPMD, no shared HLO.

The user writes one forward function with explicit stage markers;
the decorator handles tracing, splitting, placement, and dispatch.

Requires ``transformers``, ``safetensors``, ``HF_TOKEN`` env var.

Usage::

    HF_TOKEN=hf_xxx python -m examples.07_mpmd.11_mpmd_jit_generation
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
from spectrax.runtime.mpmd import sxjit
from spectrax.runtime.mpmd.markers import sxstage_iter
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
    """Load Llama 3.2 3B, generate with @sxjit per token."""
    from transformers import AutoTokenizer

    n_devices = len(jax.devices())
    pp = min(n_devices, 4)
    mid = 28 // pp

    print(f"Loading {MODEL_NAME} (PP={pp}, {n_devices} devices)...")
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

    mesh = spx.create_mesh(
        axis_dims=(pp, 1, 1, 1, -1, 1),
        mpmd_axis="pp",
    )

    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = Llama3(cfg, rngs=spx.Rngs(0))
        print("Loading weights...")
        state = load_hf_weights(model, MODEL_NAME, CACHE_DIR)
        gdef, _ = spx.export(model)
        model = spx.bind(gdef, state)
        print(f"  L={cfg.n_layers} d={cfg.d_model} PP={pp}")

        if pp == 2:

            @sxjit(mesh=mesh.mpmd_mesh)
            def prefill(model, ids):
                """Two-stage @sxjit prefill: embed -> blocks[:mid] -> iter -> blocks[mid:] -> head.

                Args:
                    model: :class:`Llama3` instance.
                    ids: Token ids ``(b, seq)``.

                Returns:
                    Logits tensor ``(b, seq, vocab)``.
                """
                x = model.embed(ids)
                for blk in model.blocks[:mid]:
                    x = blk(x)
                x = sxstage_iter(x)
                for blk in model.blocks[mid:]:
                    x = blk(x)
                return model.head(x)

        elif pp == 4:

            @sxjit(mesh=mesh.mpmd_mesh)
            def prefill(model, ids):
                """Four-stage @sxjit prefill with three ``sxstage_iter`` markers.

                Args:
                    model: :class:`Llama3` instance.
                    ids: Token ids ``(b, seq)``.

                Returns:
                    Logits tensor ``(b, seq, vocab)``.
                """
                x = model.embed(ids)
                for blk in model.blocks[:mid]:
                    x = blk(x)
                x = sxstage_iter(x)
                for blk in model.blocks[mid : 2 * mid]:
                    x = blk(x)
                x = sxstage_iter(x)
                for blk in model.blocks[2 * mid : 3 * mid]:
                    x = blk(x)
                x = sxstage_iter(x)
                for blk in model.blocks[3 * mid :]:
                    x = blk(x)
                return model.head(x)

        else:
            raise ValueError(f"This example supports PP=2 or PP=4, got PP={pp}")

        print("\nCompiling per-rank jits...")
        dummy = jnp.ones((1, 1), dtype=jnp.int32)
        t0 = time.perf_counter()
        out = prefill(model, dummy)
        jax.block_until_ready(out)
        print(f"  compiled in {time.perf_counter() - t0:.1f}s")

        for _ in range(3):
            jax.block_until_ready(prefill(model, dummy))
        print("  warm.")

        eos = tokenizer.eos_token_id or -1
        prompts = [
            "The meaning of life is",
            "In a galaxy far far away,",
            "def fibonacci(n):",
        ]

        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            prompt_ids = jnp.array(ids)[None, :]

            logits = prefill(model, prompt_ids)
            jax.block_until_ready(logits)
            first_tok = jnp.argmax(logits[:, -1, :], axis=-1)

            generated = [int(first_tok[0])]
            tok = first_tok[:, None]

            t0 = time.perf_counter()
            for _i in range(MAX_NEW - 1):
                logits = prefill(model, tok)
                nt = jnp.argmax(logits[:, -1, :], axis=-1)
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
            print(f"  {n} tokens in {elapsed:.2f}s = {tps:.1f} tok/s (PP={pp}, true MPMD)")
            print(f"  Output: {text[:400]}")


if __name__ == "__main__":
    main()
