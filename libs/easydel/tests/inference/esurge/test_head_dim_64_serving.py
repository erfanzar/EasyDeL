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

"""eSurge serving coverage for the head_dim-64 ragged-page kernel.

ejkernel dispatches ``ragged_page_attention_v3`` on the query head dim, and
``head_dim == 64`` routes to ``_pallas_impl_fwd_h64``, whose page layout differs
from every other head dim: K and V are concatenated into a 128-wide last axis
instead of being interleaved on the head axis.

Nothing else in the eSurge suite serves a head_dim-64 model, so EasyDeL shipped
the wrong page shape for this kernel -- twice the KV bytes, in a layout the
kernel rejects outright -- without a single test noticing. These tests pin the
allocated pool to ejkernel's own shape helper and drive a real generate through
the engine so the path has end-to-end coverage.

Token-level parity against a dense reference is deliberately not asserted here:
with randomly initialised weights the logit landscape is nearly flat and greedy
argmax flips on bf16-level noise, which makes such a check a coin toss rather
than a correctness signal.
"""

from __future__ import annotations

import gc

import easydel as ed
import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from easydel.caching.ragged_page.cache import get_page_size_bytes
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.modules.llama.modeling_llama import LlamaForCausalLM
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd_h64 import (
    get_kv_cache_shape as ejkernel_shape_hd64,
)

HEAD_DIM = 64
NUM_KV_HEADS = 8
VOCAB = 256
MAX_NEW_TOKENS = 4
PAGE_SIZE = 32


def _tiny_head_dim_64_model() -> LlamaForCausalLM:
    """Build a small model whose head dim reaches the h64 ragged-page kernel."""
    config = ed.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=NUM_KV_HEADS * HEAD_DIM,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=NUM_KV_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=256,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    model = LlamaForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    assert head_dim == HEAD_DIM, f"fixture must hit the h64 kernel, got head_dim={head_dim}"
    return model


def test_page_size_bytes_is_minimal_for_head_dim_64():
    """The h64 layout must not pay the head-axis K/V doubling twice."""
    reported = get_page_size_bytes(PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, jnp.bfloat16)
    minimum = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM * 2 * jnp.finfo(jnp.bfloat16).bits // 8

    assert reported == minimum, f"page reported as {reported} bytes, minimum for this geometry is {minimum}"


@pytest.mark.slow
def test_esurge_serves_a_head_dim_64_model():
    """A head_dim-64 model must allocate the kernel's layout and generate.

    Before the layout fix this raised ``Invalid num_kv_heads=..., kv_packing=2``
    from the kernel's shape validation the moment a request was scheduled.
    """
    model = _tiny_head_dim_64_model()
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.02,
        page_size=PAGE_SIZE,
        max_cache_tokens=2048,
        max_model_len=128,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=2,
        max_num_seq_buckets=[2],
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
    )
    try:
        runner.compile(max_num_batched_tokens=64)

        pages = jax.tree_util.tree_leaves(runner.executor_manager.kv_pages)[0]
        expected = ejkernel_shape_hd64(int(pages.shape[0]), PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, pages.dtype)
        assert tuple(pages.shape) == tuple(expected), (
            f"engine allocated {tuple(pages.shape)}, h64 kernel reads {tuple(expected)}"
        )

        scheduler = Scheduler.from_runner(
            runner,
            max_num_batched_tokens=64,
            enable_prefix_caching=False,
            async_scheduling=False,
        )
        request = EngineRequest(
            request_id="head-dim-64",
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
        )
        scheduler.add_request(request)
        for _ in range(64):
            scheduler_output = scheduler.schedule()
            output = runner.execute_model(scheduler_output)
            scheduler.update_from_output(scheduler_output, output)
            if request.is_finished():
                break
        else:
            raise AssertionError("generation did not finish within the step budget")

        produced = list(request.output_token_ids)
        assert len(produced) == MAX_NEW_TOKENS, f"expected {MAX_NEW_TOKENS} tokens, got {produced}"
        assert all(0 <= t < VOCAB for t in produced), f"token ids out of range: {produced}"
    finally:
        runner.shutdown()
        runner.executor_manager.kv_pages = None
        gc.collect()
