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

"""Shared fixtures for the eSurge serving, spec-decode and MTP tests.

``make_tiny_qwen35`` builds the tiny Qwen3.5 + MTP model that every
spec-decode / MTP test used to define its own byte-identical copy of.
``make_core_for_runner`` and ``served_request`` drive a runner through the
engine core (surgecore ``EngineCore``) the way the engine loop does:
``schedule() -> runner.execute_model() -> commit()``. Import them
with the standard sibling-helper fallback::

    try:
        from ._common import make_tiny_qwen35
    except ImportError:  # standalone `python test_x.py`
        from _common import make_tiny_qwen35
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import spectrax as spx
from easydel.modules.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

VOCAB = 256
HIDDEN = 128


def _spec_decode_axis_dims() -> tuple[int, int, int, int, int, int]:
    """Mesh dims that keep the GDN kernels on their sharded path.

    ejkernel runs ragged GDN (and its speculative-window scan) as a TPU Pallas
    kernel only inside a shard_map region, which it enters when a head axis is
    shardable. With the default ``tp=1`` mesh no axis qualifies, so on a
    multi-device TPU the kernels are asked to run unsharded and Mosaic rejects
    them -- while on CPU the same config silently falls back to XLA and passes.
    Pinning ``tp=2`` (a divisor of this model's 4 attention heads, 2 KV heads,
    16 linear key heads, and 32 linear value heads) makes these tests exercise
    the real sharded path on every backend.
    """
    devices = jax.device_count()
    tp = 2 if devices % 2 == 0 else 1
    return (1, 1, -1, 1, tp, 1)


def make_tiny_qwen35(mtp_layers: int = 1, *, precision: jax.lax.PrecisionLike = None) -> Qwen3_5ForCausalLM:
    """Build the tiny Qwen3.5 + MTP model used across the spec-decode tests.

    Args:
        mtp_layers: Number of MTP head layers (``mtp_num_hidden_layers``).
        precision: Matmul precision; exact fp32 reference tests request HIGHEST
            locally instead of changing the process-wide TPU precision.

    Returns:
        A float32 ``Qwen3_5ForCausalLM`` with a 4-layer (3 linear + 1 full)
        backbone and an MTP head.
    """
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=mtp_layers,
        mtp_loss_coef=0.0,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        sharding_axis_dims=_spec_decode_axis_dims(),
    )
    return Qwen3_5ForCausalLM(
        config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32, precision=precision
    )


class ServedRequest:
    """A test's view of one request served through the engine core.

    Holds the :class:`AdmissionRecord` it was admitted with (its attributes read through) and what the commits
    reported for it: ``output_token_ids`` (stop truncation applied), ``finish_reason``, ``stop_reason`` and
    ``num_cached_tokens`` (prefix-hit tokens of its admission, -1 before its first output).
    """

    def __init__(self, record) -> None:
        self.record = record
        self.output_token_ids: list[int] = []
        self.finish_reason = None
        self.stop_reason = None
        self.num_cached_tokens = -1

    def __getattr__(self, name: str):
        if name == "record":
            raise AttributeError(name)
        return getattr(self.record, name)

    @property
    def num_output_tokens(self) -> int:
        """Tokens emitted so far."""
        return len(self.output_token_ids)

    def is_finished(self) -> bool:
        """Whether the engine core reported a finish reason."""
        return self.finish_reason is not None


def served_request(request_id: str, prompt_token_ids, sampling_params, eos_token_id=None, **kwargs) -> ServedRequest:
    """A :class:`ServedRequest` over a fresh ``AdmissionRecord`` (the engine core's admission currency)."""
    from easydel.inference.esurge.request import AdmissionRecord

    return ServedRequest(
        AdmissionRecord(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            eos_token_id=eos_token_id,
            **kwargs,
        )
    )


class CoreDrive:
    """An ``EngineCoreDriver`` plus the :class:`ServedRequest` handles of the requests a test admitted.

    ``schedule()`` and ``commit()`` are the engine core's; ``commit`` also applies the returned outputs to the
    handles. Every other attribute reads through to the driver.
    """

    def __init__(self, core) -> None:
        self.core = core
        self.requests: dict[str, ServedRequest] = {}

    def __getattr__(self, name: str):
        if name in ("core", "requests"):
            raise AttributeError(name)
        return getattr(self.core, name)

    def __setattr__(self, name: str, value) -> None:
        if name in ("core", "requests"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.core, name, value)

    def add_requests(self, requests: list[ServedRequest]) -> None:
        """Admit requests in one call (``n>1`` siblings together)."""
        self.core.add_requests([r.record for r in requests])
        for r in requests:
            self.requests[r.request_id] = r

    def add_request(self, request: ServedRequest) -> None:
        """Admit one request."""
        self.add_requests([request])

    def schedule(self):
        """The engine core's next step (a ``SchedulerOutput``)."""
        return self.core.schedule()

    def commit(self, scheduler_output, model_output):
        """Commit a step's runner output; updates the handles and returns the engine outputs."""
        result = self.core.commit(scheduler_output, model_output)
        for engine_outputs in result.values():
            for out in engine_outputs.outputs:
                handle = self.requests.get(out.request_id)
                if handle is None:
                    continue
                handle.output_token_ids.extend(int(t) for t in out.new_token_ids)
                handle.num_cached_tokens = int(out.num_cached_tokens)
                if out.finish_reason is not None:
                    handle.finish_reason = out.finish_reason
                    handle.stop_reason = out.stop_reason
        return result

    def step(self, runner) -> bool:
        """One ``schedule -> runner.execute_model -> commit``; ``False`` for an idle step the loop would skip."""
        from easydel.inference.esurge.benchmarking import zero_schedule_needs_update

        scheduler_output = self.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0 and not zero_schedule_needs_update(scheduler_output):
            return False
        self.commit(scheduler_output, runner.execute_model(scheduler_output))
        return True

    def run(self, runner, requests: list[ServedRequest], *, max_steps: int = 64) -> list[ServedRequest]:
        """Admit ``requests`` and step until every one finishes.

        Raises:
            AssertionError: When they do not finish within ``max_steps`` steps.
        """
        self.add_requests(list(requests))
        for _ in range(max_steps):
            if all(r.is_finished() for r in requests):
                return requests
            self.step(runner)
        if all(r.is_finished() for r in requests):
            return requests
        raise AssertionError(f"generation did not finish within {max_steps} steps")


def make_core_for_runner(
    runner,
    *,
    max_num_batched_tokens: int | None = None,
    enable_prefix_caching: bool = True,
    async_scheduling: bool = True,
    long_prefill_token_threshold: int | None = None,
    num_draft_tokens: int = 0,
    **kwargs,
) -> CoreDrive:
    """The engine core a test drives a runner with, built as the engine builds it (``EngineCoreDriver.from_runner``).

    Defaults are the engine's: prefix caching on (the core turns it off for models that cannot use it), async
    scheduling on, no speculative drafts.
    """
    from easydel.inference.esurge.engine.core_driver import EngineCoreDriver

    return CoreDrive(
        EngineCoreDriver.from_runner(
            runner,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=enable_prefix_caching,
            async_scheduling=async_scheduling,
            long_prefill_token_threshold=long_prefill_token_threshold,
            num_draft_tokens=int(num_draft_tokens),
            **kwargs,
        )
    )
