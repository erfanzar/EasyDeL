# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sanity-check a (converted) checkpoint actually behaves like a trained LM.

Catches silent weight-scrambling (wrong layout/orientation/interleave) that
load-time shape/name checks cannot see: a healthy pretrained LM has next-token
entropy well below ``ln(vocab)`` on real text and near-zero entropy with
near-100% top-1 accuracy on a trivially repeated token pattern; a scrambled
one is near-uniform (the qwen3.6-27b tp-interleave incident scored
entropy≈10.4/12.4 with 0% repeat accuracy).

Usage:
    python scripts/verify_checkpoint.py gs://bucket/model --tokenizer Qwen/Qwen3.6-27B
    python scripts/verify_checkpoint.py /path/ckpt --tokenizer ... --tp 2 --seq 256

Exit code 0 = healthy, 1 = failed the battery (do NOT train/serve from it).
"""

import argparse
import math
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name_or_path", help="EasyDeL-native checkpoint path (local or gs://) or HF id")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer id")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size for the probe mesh")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--max-real-entropy", type=float, default=6.0, help="fail if real-text entropy exceeds this")
    ap.add_argument(
        "--min-repeat-acc", type=float, default=0.5, help="fail if repeat-pattern top-1 accuracy is below this"
    )
    args = ap.parse_args()

    os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

    import easydel as ed
    import jax
    import numpy as np
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    real = np.asarray(
        tok(
            "The capital of France is Paris. Water boils at one hundred degrees Celsius. "
            "The quick brown fox jumps over the lazy dog. A language model predicts the "
            "next token given the previous ones.",
            return_tensors="np",
        )["input_ids"],
        dtype=np.int32,
    )[:, : args.seq]
    rep = np.tile(np.array([101, 202, 303, 404, 505, 606, 707, 808], np.int32), max(1, args.seq // 8))[None, :]

    cfg = {
        "model": {"name_or_path": args.name_or_path, "tokenizer": args.tokenizer, "task": "auto-bind"},
        "loader": {"dtype": "bf16", "param_dtype": "bf16", "precision": "fastest"},
        "sharding": {
            "axis_dims": (1, 1, -1, 1, args.tp, 1),
            "axis_names": ("pp", "dp", "fsdp", "ep", "tp", "sp"),
            "auto_shard_model": True,
        },
        "base_config": {
            "values": {
                "attn_mechanism": ed.AttentionMechanisms.VANILLA,
                "attn_dtype": "bf16",
                "mtp_loss_coef": 0.0,
                "mtp_num_hidden_layers": 0,
            }
        },
    }
    st = ed.eLargeModel(cfg).build_state()
    mesh = st.model.mesh
    vocab = int(getattr(st.model.config, "vocab_size", 0)) or 1
    print(f"ln(vocab)={math.log(vocab):.3f}")

    @ed.ejit
    def fwd(gs, ids, am):
        return st.merge(gs)(input_ids=ids, attention_mask=am).logits

    def stats(lg, ids):
        lg = np.asarray(jax.device_get(lg)).astype(np.float32)
        lp = lg - jax.scipy.special.logsumexp(lg, -1, keepdims=True)
        p = np.exp(lp)
        return float((-(p * lp).sum(-1)).mean()), float((lg[:, :-1].argmax(-1) == ids[:, 1:]).mean())

    ok = True
    for nm, x, _check in (("real", real, None), ("repeat", rep, "acc")):
        with mesh:
            lg = fwd(st.graphstate, jnp.asarray(x), jnp.ones_like(jnp.asarray(x)))
        ent, acc = stats(lg, x)
        print(f"[VERIFY] {nm:6}: entropy={ent:6.3f}  top1_acc={acc:5.1%}")
        if nm == "real" and ent > args.max_real_entropy:
            ok = False
        if nm == "repeat" and acc < args.min_repeat_acc:
            ok = False

    print("[VERIFY] HEALTHY" if ok else "[VERIFY] FAILED — weights look scrambled/untrained; do not use")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
