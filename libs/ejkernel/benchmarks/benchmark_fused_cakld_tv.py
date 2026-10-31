# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Benchmark ``fused_cakld_tv`` against dense and composed reference implementations.

Three implementations of the same per-token dynamic-CAKLD (+ TV) distillation loss:

* ``dense``: textbook formulation. Materializes both softmaxes, the gate, both KL terms and the TV term
  as separate fp32 ``[rows, V]`` temporaries.
* ``composed``: gamma pass + two gamma-weighted ``fused_kl_divergence`` calls (+ a TV pass), i.e. the
  pipeline ``fused_cakld_tv`` replaces (3-4 reads of the two logit tensors).
* ``fused``: one ``fused_cakld_tv`` call.

Compile-including (first call) and steady-state (median after warmup) times are reported separately,
for the forward pass and for forward + backward w.r.t. the student logits. Run on the target accelerator
(a CPU run only validates the harness)::

    flock /tmp/easydel_tpu.lock env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu \\
        JAX_PLATFORM_NAME=tpu uv run python libs/ejkernel/benchmarks/benchmark_fused_cakld_tv.py \\
        --shape 2048,131072 --shape 8192,131072
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
from ejkernel.modules.operations import fused_cakld_tv, fused_kl_divergence


def dense_loss(student, teacher, mask, temperature, tv_weight):
    """Textbook dense dynamic-CAKLD (+ TV) mean loss with materialized softmaxes."""
    s = student.astype(jnp.float32) / temperature
    t = jax.lax.stop_gradient(teacher.astype(jnp.float32)) / temperature
    p = jax.nn.softmax(t, axis=-1)
    q = jax.nn.softmax(s, axis=-1)
    log_p = jax.nn.log_softmax(t, axis=-1)
    log_q = jax.nn.log_softmax(s, axis=-1)
    reverse = jnp.sum(q * (log_q - log_p), axis=-1)
    forward = jnp.sum(p * (log_p - log_q), axis=-1)
    gamma = jnp.max(p, axis=-1)
    row = gamma * reverse + (1.0 - gamma) * forward
    if tv_weight:
        tv = 0.5 * jnp.sum(jnp.abs(p - q), axis=-1)
        row = (1.0 - tv_weight) * row + tv_weight * 2.0 * tv
    return jnp.sum(row * mask) * temperature**2 / jnp.maximum(mask.sum(), 1e-8)


def composed_loss(student, teacher, mask, temperature, tv_weight):
    """gamma pass + two weighted fused-KL calls (+ TV pass): the multi-read pipeline being replaced."""
    t32 = jax.lax.stop_gradient(teacher.astype(jnp.float32)) / temperature
    gamma = jnp.exp(jnp.max(t32, axis=-1) - jax.nn.logsumexp(t32, axis=-1))
    kwargs = {"reduction": "mean", "temperature": temperature, "platform": "xla"}
    rev = fused_kl_divergence(student, teacher, mask * gamma, direction="reverse", **kwargs)
    fwd = fused_kl_divergence(student, teacher, mask * (1.0 - gamma), direction="forward", **kwargs)
    total = rev.loss * rev.weight_sum + fwd.loss * fwd.weight_sum
    if tv_weight:
        log_p = jax.nn.log_softmax(t32, axis=-1)
        log_q = jax.nn.log_softmax(student.astype(jnp.float32) / temperature, axis=-1)
        tv = 1.0 - jnp.sum(jnp.exp(jnp.minimum(log_p, log_q)), axis=-1)
        total = (1.0 - tv_weight) * total + tv_weight * temperature**2 * 2.0 * jnp.sum(mask * tv)
    return total / jnp.maximum(mask.sum(), 1e-8)


def fused_loss(student, teacher, mask, temperature, tv_weight):
    """Single fused_cakld_tv call."""
    return fused_cakld_tv(
        student, teacher, mask, reduction="mean", temperature=temperature, tv_weight=tv_weight, platform="xla"
    ).loss


IMPLS = {"dense": dense_loss, "composed": composed_loss, "fused": fused_loss}


def time_fn(fn, student, iters):
    """Return ``(first_call_ms, median_steady_ms)`` for ``fn(student)``."""
    start = time.perf_counter()
    jax.block_until_ready(fn(student))
    first_ms = (time.perf_counter() - start) * 1e3
    for _ in range(2):
        jax.block_until_ready(fn(student))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(student))
        samples.append((time.perf_counter() - start) * 1e3)
    return first_ms, statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", action="append", default=[], help="ROWS,VOCAB (repeatable)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tv-weights", default="0.0,0.25", help="comma-separated TV mixing weights")
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()

    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shape] or [(2048, 131072), (8192, 131072)]
    dtype = jnp.dtype(args.dtype)
    print(f"devices: {jax.devices()}", flush=True)

    for rows, vocab in shapes:
        gb = rows * vocab * dtype.itemsize / 1e9
        print(f"\n== rows={rows} V={vocab} {args.dtype} (~{gb:.2f} GB per logit tensor) ==", flush=True)
        key_s, key_t = jax.random.split(jax.random.PRNGKey(0))
        student = jax.random.normal(key_s, (rows, vocab), dtype)
        teacher = jax.random.normal(key_t, (rows, vocab), dtype) * 2.0
        mask = jnp.ones((rows,), jnp.float32)
        reference = {}
        for rho in (float(x) for x in args.tv_weights.split(",")):
            for name, impl in IMPLS.items():
                loss_fn = lambda s, _i=impl, _r=rho, _t=teacher, _m=mask: _i(s, _t, _m, args.temperature, _r)  # noqa: E731
                try:
                    fwd_first, fwd_ms = time_fn(jax.jit(loss_fn), student, args.iters)
                    fb_first, fb_ms = time_fn(jax.jit(jax.value_and_grad(loss_fn)), student, args.iters)
                    value = float(jax.jit(loss_fn)(student))
                except Exception as err:  # e.g. RESOURCE_EXHAUSTED for the dense reference at large shapes
                    print(f"rho={rho:4.2f} {name:8s} | FAILED: {type(err).__name__}: {str(err).splitlines()[0][:100]}")
                    continue
                reference.setdefault(rho, value)
                print(
                    f"rho={rho:4.2f} {name:8s} | fwd {fwd_ms:8.2f} ms (first {fwd_first:7.0f}) | "
                    f"fwd+bwd {fb_ms:8.2f} ms (first {fb_first:7.0f}) | loss {value:.6f} "
                    f"(vs first impl {value - reference[rho]:+.2e})",
                    flush=True,
                )


if __name__ == "__main__":
    main()
