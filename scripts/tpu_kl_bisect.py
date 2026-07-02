# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bisect the teacher-vs-student forward divergence of the distillation step on TPU.

Replicates the exact two call paths of ``distillation_step`` (_fn.py) with ONE
tiny hybrid GDR+attention model used as both teacher and student — identical
weights — on real TPU kernels in bf16:

* student path: ``filter_kwargs_for_callable`` + ``sanitize_model_call_kwargs``
  then the module forward as the primal of ``jax.vjp`` (grad compilation path).
* teacher path: same kwargs pipeline, non-array kwargs split out as statics,
  forward under ``jax.checkpoint(nothing_saveable)`` of the stop-gradient
  graphstate (no-grad compilation path).

Reports per-layer hidden-state max-rel differences and the final-logit KL so
the first diverging layer/mechanism is visible. KL should be ~0; anything
material localizes the distillation self-KL bug.

Run on a TPU host with free chips:
  python scripts/tpu_kl_bisect.py [--seq 1024] [--layers 8] [--packed]
"""

from __future__ import annotations

import argparse
import functools
import sys


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--packed", action="store_true", help="packed segments instead of padded tail")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = p.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5TextConfig
    from easydel.trainers.training_utils import filter_kwargs_for_callable, sanitize_model_call_kwargs

    dtype = jnp.dtype(args.dtype)
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * max(args.layers // 4, 1)
    config = Qwen3_5TextConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=max(args.seq * 2, 2048),
        layer_types=layer_types,
        mtp_num_hidden_layers=0,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        scan_layers=False,
    )
    model = Qwen3_5ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=dtype, param_dtype=dtype)
    model.train()

    rng = np.random.default_rng(0)
    B, L = 2, args.seq
    ids = jnp.asarray(rng.integers(1, 2047, (B, L)), dtype=jnp.int32)
    if args.packed:
        attention_mask = jnp.ones((B, L), dtype=jnp.int32)
        segment_ids = jnp.asarray(np.arange(L) // max(L // 4, 1), dtype=jnp.int32)[None, :].repeat(B, 0)
        batch = {"input_ids": ids, "attention_mask": attention_mask, "segment_ids": segment_ids}
    else:
        # padded tail like the production (non-packed) distill batches
        attention_mask = jnp.asarray(np.arange(L)[None, :].repeat(B, 0) < (L - L // 8), dtype=jnp.int32)
        batch = {"input_ids": ids, "attention_mask": attention_mask}
    batch["output_hidden_states"] = True

    gdef, gstate, gother = model.split_module()

    # ---- student path: filtered kwargs, forward as primal of jax.vjp ----
    s_kwargs = filter_kwargs_for_callable(model.__call__, dict(batch))
    s_kwargs = sanitize_model_call_kwargs(s_kwargs)

    def student_fwd(tree):
        module = model.merge_module(gdef, tree, gother)
        out = module(**s_kwargs)
        return out.logits, tuple(out.hidden_states)

    (s_logits, s_hidden), _ = jax.vjp(student_fwd, gstate)

    # ---- teacher path: static split + checkpoint(nothing_saveable) + stop_gradient state ----
    t_kwargs = filter_kwargs_for_callable(model.__call__, dict(batch))
    t_kwargs = sanitize_model_call_kwargs(t_kwargs)
    t_static = {k: t_kwargs.pop(k) for k in list(t_kwargs) if not hasattr(t_kwargs[k], "shape")}

    @functools.partial(jax.checkpoint, prevent_cse=True, policy=jax.checkpoint_policies.nothing_saveable)
    def teacher_fwd(kw, tree):
        module = model.merge_module(gdef, tree, gother)
        out = module(**kw, **t_static)
        return jax.lax.stop_gradient(out.logits), jax.lax.stop_gradient(tuple(out.hidden_states))

    t_logits, t_hidden = teacher_fwd(t_kwargs, jax.lax.stop_gradient(gstate))

    def maxrel(a, b):
        a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
        return float(jnp.max(jnp.abs(a32 - b32)) / jnp.maximum(jnp.max(jnp.abs(b32)), 1e-6))

    valid = attention_mask.astype(bool)
    for i, (sh, th) in enumerate(zip(s_hidden, t_hidden, strict=True)):
        print(f"layer {i:>2} hidden maxrel: {maxrel(sh, th):.3e}")

    s_lp = jax.nn.log_softmax(s_logits.astype(jnp.float32), -1)
    t_lp = jax.nn.log_softmax(t_logits.astype(jnp.float32), -1)
    kl_tok = jnp.sum(jnp.exp(t_lp) * (t_lp - s_lp), -1)
    kl = float(jnp.sum(jnp.where(valid, kl_tok, 0.0)) / jnp.maximum(jnp.sum(valid), 1))
    print(f"logits maxrel: {maxrel(s_logits, t_logits):.3e}")
    print(f"mean KL(teacher||student) over valid tokens: {kl:.6f} nats")
    return 0 if kl < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
