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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantization-aware training in one call.

    model = ed.apply_quantization_rules(model, "int8")

That is the whole integration. The rules are matched against module paths
once and stamped onto the graph, so they travel with the ``GraphDef`` into
``EasyDeLState`` and through every trainer with no further wiring. No
model file changes; the same call works on any family in the zoo.

``int8`` is the preset to reach for on TPU: measured 1.15-1.27x faster
than bfloat16 on v5, converging identically. The fp8 presets are roughly
break-even there because the MXU has no fp8 advantage, and are worth
picking only for GPU targets or fp8 numerics work.

Run this file to train a small model both ways and compare::

    python -m easydel.examples.quantization_aware_training

Master weights stay full precision -- only the matmul operands are
narrowed -- so this does not reduce parameter memory. What it buys is
throughput, and a model whose weights have learned to tolerate the
discretization it will be served with.
"""

import time

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax
import spectrax as spx

STEPS = 40


def build_model() -> ed.LlamaForCausalLM:
    """Build a small Llama to keep the example runnable anywhere.

    Returns:
        A freshly initialized causal language model.
    """
    config = ed.LlamaConfig(
        vocab_size=4096,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=512,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    return ed.LlamaForCausalLM(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )


def train(quantization: str | None, tokens: jax.Array, labels: jax.Array) -> tuple[float, list[float]]:
    """Train one configuration and report its step time and loss curve.

    Args:
        quantization: Preset name, or ``None`` to train unquantized.
        tokens: Input token ids.
        labels: Next-token targets.

    Returns:
        ``(median_seconds_per_step, losses)``.
    """
    model = build_model()

    # ---- the only quantization-specific line -------------------------
    if quantization is not None:
        model = ed.apply_quantization_rules(model, quantization)
    # ------------------------------------------------------------------

    # Build the state *after* stamping. EasyDeLState.model rebuilds the
    # module from the stored GraphDef on every access, so rules applied to
    # a state's model would land on a throwaway copy and silently vanish.
    state = model.to_state()
    params = state.graphstate

    def loss_fn(trainable):
        """Next-token cross entropy for one parameter tree."""
        logits = state.merge(trainable)(input_ids=tokens).logits.astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        """One ordinary optimizer step; nothing here knows about quantization."""
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    params, opt_state, loss = train_step(params, opt_state)  # warm up the compile
    jax.block_until_ready(loss)

    losses, times = [], []
    for _ in range(STEPS):
        start = time.perf_counter()
        params, opt_state, loss = train_step(params, opt_state)
        jax.block_until_ready(loss)
        times.append(time.perf_counter() - start)
        losses.append(float(loss))
    return float(np.median(times)), losses


def main() -> None:
    """Train the same model unquantized and with int8, and compare."""
    tokens = jax.random.randint(jax.random.key(7), (8, 512), 0, 4096)
    labels = jax.random.randint(jax.random.key(8), (8, 512), 0, 4096)

    print(f"device: {jax.devices()[0].device_kind} x{len(jax.devices())}")
    print(f"{STEPS} steps of adam(3e-4)\n")
    print(f"{'configuration':24s} {'ms/step':>9s} {'speedup':>8s}   {'first':>7s} {'last':>9s}")
    print("-" * 62)

    baseline_ms, baseline_losses = train(None, tokens, labels)
    print(
        f"{'bfloat16':24s} {baseline_ms * 1e3:8.2f}m {1.0:7.2f}x   "
        f"{baseline_losses[0]:7.3f} {baseline_losses[-1]:9.4f}"
    )

    for preset in ("int8", "int4"):
        step_ms, losses = train(preset, tokens, labels)
        print(
            f"{preset:24s} {step_ms * 1e3:8.2f}m {baseline_ms / step_ms:7.2f}x   "
            f"{losses[0]:7.3f} {losses[-1]:9.4f}"
        )

    print("\nBoth quantized runs should track the bfloat16 loss curve.")
    print("Serving the result: model.quantize(...) applies the matching post-training format.")


if __name__ == "__main__":
    main()
